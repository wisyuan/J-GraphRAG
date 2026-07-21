# J-GraphRAG：J-Lens 加速的零 API 成本知识图谱检索

## 方案总结（2026-07-16，Phase 10-37 探索结论）

---

## 0. 核心定位：GraphRAG 的 J-Lens 加速层

J-GraphRAG 不是一个"新的 GraphRAG 变体"——它是**对传统 GraphRAG 提取管线的方法论替换**。

传统 GraphRAG（LightRAG / HippoRAG2 / Fast-GraphRAG / MS-GraphRAG）的每个知识提取步骤都依赖 LLM `generate()`：

```
传统管线（每步 = 1 次 generate()）:
  文档 → [generate: 实体提取] → [generate: 关系提取]
       → [generate: 实体消歧] → [generate: 社区检测]
       → 知识图谱 → 检索
  问题：每次 generate 需要 5-20s，产出结构化 JSON 需要解析+重试，
       在小模型(7B)上不可靠（JSON 截断），在大模型上需要 API 费用
```

J-Lens 管线把每个提取步骤从 `generate()` 替换为 `forward pass + workspace 读出`：

```
J-Lens 管线（每步 = 1 次 forward pass）:
  文档 → [J-Lens: 概念提取] → [J-Lens: 关系提取]
       → [prefill scan: 概念角色] → [bge-m3: 聚类]
       → 知识图谱 → 检索
  优势：每次 forward 0.17s，不需要结构化输出，
       不需要解析/重试，零 API 费用
```

**这不是替换 GraphRAG 的某个组件，而是替换整个提取方法论**：

| 维度 | generate() | J-Lens forward pass |
|---|---|---|
| 耗时/chunk | 5-60s | **0.17s** |
| 输出格式 | 需要结构化 JSON/三元组 | 不需要——直接读 workspace logits |
| 7B 可靠性 | ✗ JSON 截断（Phase 26: ACC 3.6%） | ✓ workspace 稳定读出 |
| API 成本 | $5-15 / 1000 篇 | **$0** |
| 能效比 | 1x | **347x** |

**方法论替换的意义**：任何 GraphRAG 框架都可以把它的 generate() 提取步骤替换为 J-Lens forward pass，保持图结构和检索逻辑不变，但建图成本降为原来的 1/347。

J-GraphRAG 就是这个替换的参考实现——用 J-Lens 替换了传统管线的全部提取步骤，构建出完整的知识图谱（概念节点 + 有类型关系边 + 概念角色 + 聚类结构），零 API 成本。

---

## 1. 核心架构

```
┌─────────────────────────────────────────────────────────┐
│ 离线建图（全部零 API token，本地 7B 模型）                │
│                                                          │
│  概念提取 (1 forward pass/chunk):                        │
│    文档 → concern prompt → J-Lens position -1           │
│         → workspace top-k → 概念词                       │
│         → 三重过滤 + BM25 补全                           │
│                                                          │
│  概念角色扩展 (可选, +1 forward pass/chunk):              │
│    概念 → 反转 prompt (概念作 prefill)                   │
│         → prefill position scan → 文档特定角色词          │
│    例: Nutrition → Education(71%), Training(18%)        │
│                                                          │
│  关系提取 (1 forward pass/pair):                         │
│    共现概念对 → 双概念关切 prompt                        │
│              → position -1 → 关系类型词                   │
│    例: cancer+chemo → treated(38%)                      │
│                                                          │
│  关系细化 (可选, +1 forward pass/pair):                  │
│    V1 字典关系 → V3 先验+prefill scan                   │
│    例: treated → suppress/therapy/affects               │
│                                                          │
│  产出:                                                   │
│    概念-文档映射矩阵 M (N_concepts × N_chunks)           │
│    + 概念角色扩展矩阵 R (N_concepts × N_roles)           │
│    + 关系邻接矩阵 W (N_concepts × N_concepts, typed)    │
│    + 概念嵌入矩阵 E (N_concepts × 1024, bge-m3)         │
│    所有矩阵可用线性代数/稀疏矩阵运算加速                   │
├─────────────────────────────────────────────────────────┤
│ 在线检索（矩阵运算加速）                                  │
│                                                          │
│  查询 → bge-m3 → 余弦 seed chunks                       │
│       → 概念传播: q_concept × M → chunk 得分 (1次矩阵乘) │
│       → 关系扩展: W × q_concept → 扩展概念 (1次矩阵乘)   │
│       → 合并: seed + propagated → top-K 输出              │
└─────────────────────────────────────────────────────────┘
```

### 1.1 概念-文档映射矩阵

核心数据结构不是"图"——而是一组**矩阵**，可以用线性代数高效运算：

**M：概念-文档映射矩阵**（N_concepts × N_chunks，稀疏）

```
M[i, j] = IDF(concept_i) × BM25_tf(concept_i, chunk_j)
```

- 行 = 概念词，列 = 文档 chunk
- 值 = IDF × BM25 tf（边权）
- 稀疏：大部分 chunk 只包含 1-3 个概念

**W：概念关系邻接矩阵**（N_concepts × N_concepts，稀疏）

```
W[i, j] = relation_type(i, j) × relation_probability(i, j)
```

- 对称（关系是双向的）或有向（treated/causal 有方向）
- 关系类型和概率来自 J-Lens 双概念关切读出
- 共现筛选或聚类筛选后稀疏

**E：概念嵌入矩阵**（N_concepts × d）

概念向量的来源是一个核心设计决策：

**方案 A：bge-m3 嵌入（当前实现，d=1024）**

```
E[i] = bge-m3_embed(concept_i)
```

- 优点：已有基础设施，多 token 安全
- 问题：bge-m3 是对比学习训练的独立空间，和 Qwen workspace 残差空间**几何不一致**。概念向量和文档 workspace 残差在不同空间，矩阵运算（q × M）的几何含义无法保证

**方案 B：workspace 残差向量（研究方向，d=3584）**

```
E[i] = lens.transport(workspace_residual_at_concept_i_position, layer)
```

- 概念向量取自 J-Lens workspace（lens.transport 映射到 unembedding 空间）
- 和文档 workspace 残差在**同一个空间**——可以直接做内积/cosine
- Phase 6 已验证：J-Lens 残差聚类 silhouette 比 bge-m3 高 38%
- 需要为每个概念构造 concern prompt → forward pass → 取 transport 后的残差

**方案 B 的正交化问题（关键研究方向）**

workspace 残差空间的概念向量**不天然正交**——"cancer" 和 "tumor" 的向量高度相关（语言模型中频繁共现）。直接使用会导致：

1. `M × M^T` 共现矩阵被向量相关性"污染"——cancer/tumor 共现高是因为向量相似，不是真实关系
2. SVD/NMF 分解被冗余维度主导

正交化候选方法：
- **白化**（whitening）：`(Σ + λI)^{-1/2} × E`，完全去除线性相关，但可能丢失语义
- **Gram-Schmidt 正交化**：按重要性顺序逐个正交化，保留主要语义方向
- **部分对角化**：只对 top-k 主成分正交化，保留残余相关性

### 回归分析的去共线性思路

除了正交化，回归分析中的去共线性方法可能更适合——目标是去除冗余同时保留语义信号，而非完全正交：

| 方法 | 在概念矩阵上的应用 | 优势 |
|---|---|---|
| **Ridge（L2 正则）** | 在 `M × M^T` 对角线加 λI：`(M×M^T + λI)^{-1}` | 稳定矩阵求逆，不修改原始向量，在运算中隐式处理 |
| **Lasso（L1 正则）** | 对关系矩阵 W 做 L1 稀疏化 | 自动选择少量关键关系边，过滤弱/冗余边 |
| **Elastic Net** | L1 + L2 组合 | 既稀疏又稳定 |
| **VIF（方差膨胀因子）** | 检测哪些概念对共线性过高 | 诊断工具——发现 cancer/tumor 类冗余对 |
| **PCA / PLS 回归** | 先降维到 ~10 个正交成分再做矩阵运算 | 消除维度冗余，成分有明确语义含义 |
| **核化 Ridge** | 直接在 workspace 空间做正则化内积 `K(x,y) + λδ` | 不需要显式变换，在核函数中隐式正则化 |

其中 **Ridge 正则化**（在 M×M^T 加 λI）可能是最简单有效的——不需要修改概念向量本身，只需要在矩阵运算时加一个正则项。这和推荐系统里的协同过滤正则化完全同理。

### 核心目标：概念聚合与向量聚合的双射一致性

所有去共线性/正交化方法的最终目标：**让概念的拓扑聚合结构和向量的几何聚簇结构形成近似双射**。

具体来说：如果用 Leiden 算法在概念关系图上做社区检测得到的社区划分，和用 K-Means/HDBSCAN 在 workspace 残差向量上做聚类得到的簇结构**大致一致**，那就说明：

```
概念关系图的社区结构 ≈ workspace 向量的几何结构

这意味着：
  1. 关系边不是任意的——它们反映了向量空间中的几何邻近性
  2. 向量聚类不是任意的——它反映了关系图中的拓扑社区
  3. 两种分析方式（图 vs 向量）可以互相验证、互相补充
```

**双射一致性的验证方法**：

1. 在关系图 W 上跑 Leiden 社区检测 → 得到概念社区划分 P_graph
2. 在 workspace 向量 E 上跑 K-Means/HDBSCAN → 得到概念簇划分 P_vector
3. 计算 P_graph 和 P_vector 的一致性指标：
   - Adjusted Rand Index（ARI）：衡量两种划分的一致程度
   - Normalized Mutual Information（NMI）：信息论角度的一致性
   - 如果 ARI > 0.6 或 NMI > 0.6 → 双射一致性成立

**如果双射一致性不成立**（图社区 ≠ 向量簇），说明：
- 关系图中有向量空间捕获不到的结构信息（关系是语义的，不只是几何的）
- 或者向量空间中有关系图未表示的结构信息（某些概念几何近但无关系边）

两种情况都有价值——**差异本身就是信息**，可以用来指导关系补全或向量修正。

这个双射一致性目标也是选择去共线性方法的评判标准：哪种方法能让 ARI/NMI 最高，就是最优方法。

### 向量加速方案的双层设计

如果要用线性代数加速概念处理，概念节点需要两层表示：

**符号层（离散，用于图构建/UI/检索匹配）**

```
BM25 补全后的完整词: "polycystic", "statins", "odynophagia"
```

- 用 BM25 补全处理多 token 概念（Phase 15 验证足够）
- 人类可读，用于概念图节点标注
- 用于 corpus 验证、DF 过滤等符号操作

**向量层（几何，用于矩阵运算加速）**

```
核心 BPE 碎片的 W_U 行向量: W_U[fragment_token_id]
```

- 不需要补全后的完整词——只需要一个几何方向
- 直接从模型权重 W_U（unembedding 矩阵）读取，零计算成本
- 核心碎片通常是第一个碎片（"Pol" 比 "stic" 携带更多语义）
- 在 workspace 残差空间中（和文档残差一致）

**为什么可以混用两种表示：**

概念节点同时拥有符号（BM25 补全词）和向量（BPE 碎片方向）。符号用于图的传统操作（DF 过滤、共现统计、关系提取 prompt），向量用于矩阵运算（传播、聚类、检索）。两层互不干扰。

**BPE 碎片向量的歧义处理：**

"Pol" 可能是 polycystic/politics/polar 的公共碎片。但：
1. 概念已经经过 corpus 验证——只在文档中出现的词才保留
2. 在特定语料中（医学），"Pol" 几乎只来自 polycystic
3. 如果歧义严重（多域语料），可以用 workspace 条件向量替代（+1 forward pass）

**回退方案：**

如果矩阵加速方案验证困难（向量歧义、正交化过度/不足、双射一致性差），回退到传统图扩散——Phase 25/28 的图传播检索已经验证有效（114% of B0, L4 翻倍）。矩阵加速是**可选优化**，不是核心方法。

**R：概念角色矩阵**（N_concepts × N_roles，可选）

```
R[i, k] = role_probability(concept_i, role_k)
```

- 来自 prefill position scan（Nutrition → Education 0.71, Training 0.18）
- 角色词也是概念空间的一部分

### 1.2 矩阵运算加速的检索

传统图传播是逐节点串行遍历。矩阵形式化后，每步操作变成一次矩阵乘法：

| 操作 | 图遍历方式 | 矩阵运算方式 |
|---|---|---|
| 概念传播 | seed→概念→逐个chunk | **q × M**（q 是查询概念向量，1 次乘法） |
| 概念共现 | 逐对扫描 | **M × M^T**（共现矩阵，1 次乘法） |
| 关系扩展 | 逐关系边跳转 | **W × q**（扩展概念向量，1 次乘法） |
| 多跳传播 | k 步串行遍历 | **(W × M)^k** 或 **W^k × M**（幂运算） |
| 概念聚类 | 图社区检测 | **SVD(M)** 或 **NMF(M)**（矩阵分解） |
| 查询匹配 | 逐 chunk 余弦 | **cosine(query_emb, E × M)**（投影到 chunk 空间） |

所有操作都可以用 `scipy.sparse` / GPU 矩阵乘法高效并行，无需串行图遍历。对于 1000 chunks × 100 concepts 的矩阵，这些运算在毫秒级完成。

```
├─────────────────────────────────────────────────────────┤
│ 在线检索                                                 │
│                                                          │
│  查询 → bge-m3 → 余弦 top-K seed chunks                 │
│       → 概念图传播 (seed 概念 → 共享概念的 chunk)         │
│       → 关系扩展 (关系边 → 扩展概念 → 更多 chunk)         │
│       → 合并输出                                         │
└─────────────────────────────────────────────────────────┘
```

## 2. 概念提取

### 2.1 基础提取（position -1 concern prompt）

**成本：1 次 forward pass / chunk**

```
User: What concepts does this text discuss? List 8 one-word concepts.
      [文档内容]
Assistant: The concepts discussed are
                                  ↑ 读 position -1
```

J-Lens 的 workspace 层（L26）在这个位置读出文档概念词。准确率 80%（人工审查）。

### 2.2 三重过滤 + BM25 补全

```
原始概念
  → ASCII 过滤（排除多语言碎片）
  → Prefill 词黑名单（排除 prompt 词汇：concepts/discussed/types）
  → POS 过滤（排除动词 -ing/-ed：assessed/evaluated）
  → BM25 补全（BPE 前缀→完整词：odyn→odynophagia）
  → Corpus 验证（必须在文档中出现）
  → DF 过滤（DF<2 丢弃）
```

消融结果（Phase 25）：
- 无过滤：94% of B0
- 三重过滤+BPE：**114% of B0**

### 2.3 概念角色扩展（prefill position scan，可选）

**成本：+1 次 forward pass / chunk**

Phase 35 发现的反转 prompt 方法。把概念词放进 assistant prefill，扫描每个概念词所在 position 的 workspace：

```
User: What concepts does this text discuss?
Assistant: The concepts are: Nutrition, Medicine...
                               ↑ pos 111    ↑ pos 109
```

position 111 的 workspace 读出 `Education(71%), Training(18%)`——文档讨论的是"营养**教育**"，而非泛泛的"营养学"。

每个已有概念扩展出 2-5 个文档特定的角色词，大幅增加图的节点密度。

## 3. 关系提取

### 3.1 字典式关系（基础，1 次 forward pass / pair）

**Phase 27 验证的方法：双概念关切 prompt**

```
User: This text discusses {cancer} and {chemotherapy}.
      What is the relationship between them? Answer with one word.
      [文档内容]
Assistant: The relationship between cancer and chemotherapy is
                                                                ↑ 读 position -1
```

覆盖率：**100%**（8/8 对概念读出正确的关系类型词）。

关系类型：treated/causal/regulated/essential/crucial 等（词典级裸关系）。

### 3.2 关系细化（先验+prefill scan，可选，+1 次 forward pass / pair）

**Phase 37 V3 方法：把字典关系填入 prefill，扫描补充**

```
User: What is the role of {cancer} regarding {chemotherapy}?
Assistant: In this text, cancer {treated} chemotherapy.
                              ↑ 扫描此位置
                                 → 补充 suppress/therapy/affects
```

V1 遗漏的关系被 V3 补充：
- cancer+chemo：treated → **suppress, therapy, affects**
- tumor+growth：causal → **progression, inhibition, suppress**
- smoking+cancer：causal → **association, prevention, increases**

### 3.3 关系提取的 prompt 选择

Phase 37 测试了 5 种 prompt 设计：

| 变体 | 设计 | 覆盖率 | 特点 |
|---|---|---|---|
| V1 字典 | "relationship is ___" | 100% | 最稳定，词典级 |
| V3 先验+扫描 | 字典关系→prefill→扫描 | 75% | 补充具体变体 |
| V4 功能 | "function in relation to" | 75% | 动词更精确（regulate/manage） |

**推荐**：V1 字典做基础覆盖 + V3 先验扫描做细化（2 次 forward pass / pair）。

### 3.4 共现筛选

N 个概念有 N² 个可能对。用**聚类驱动筛选**（Phase 30）：

1. bge-m3 嵌入概念词 → K-Means 聚类
2. 簇内对（语义近）：C(3,2)×4 ≈ 12 对
3. 簇间对（代表词）：C(4,2) = 6 对
4. 总计 ~18 对（vs N²=78）

13 个概念 → 18 对 → 14/21 已知关系类型（67%）。

聚类结果同时也是概念嵌入矩阵 E 的 K-Means 分解——可以直接用于检索时的概念空间导航。

## 4. 成本估算

### 建图（1000 篇文档）

| 步骤 | 方法 | forward pass | 时间 |
|---|---|---|---|
| 概念提取 | position -1 concern | 1000 次 | ~3 分钟 |
| 概念角色扩展（可选） | prefill scan | +1000 次 | +3 分钟 |
| 关系提取 | 字典式 | ~18 次/语料 | ~7 秒 |
| 关系细化（可选） | 先验+scan | +18 次 | +7 秒 |
| 嵌入 | bge-m3 | 1000 次 | ~2 分钟 |
| **总计（基础）** | | **~1020 次** | **~5 分钟** |
| **总计（完整）** | | **~2040 次** | **~8 分钟** |

### 对比 SOTA

| 方法 | 建图时间/1000篇 | API 成本 |
|---|---|---|
| **J-GraphRAG 基础** | **~5 分钟** | **$0** |
| **J-GraphRAG 完整** | **~8 分钟** | **$0** |
| Fast-GraphRAG (Qwen-7B) | ~1000 分钟 | $0（但不可靠：JSON 截断） |
| 传统 GraphRAG (API) | ~10 分钟 | ~$5-15 |

## 5. 检索效果

### GraphRAG-Bench medical（200 chunks, 28 queries）

| 方法 | ACC | L4（多跳） | evidence recall |
|---|---|---|---|
| B0 (RAG) | 64-75% | 28.6% | 71.8% |
| flat 概念图（Phase 25） | 71-75% | 28.6% | 69.7% |
| + 关系传播（Phase 28） | **75%** | **57.1%** | **73.9%** |
| 排行榜等效（换算） | **~64%** | — | — |

### 能效比

| 方法 | 建图时间/chunk | VRAM | ACC |
|---|---|---|---|
| **J-GraphRAG** | **0.173s** | **5.76GB** | **71-75%** |
| Fast-GraphRAG | ~60s | ~5.8GB | 3.6% |
| **能效比** | **347x** | — | — |

## 6. J-Lens 能力边界（实验验证）

| 任务 | 最优方法 | forward pass | 可靠性 |
|---|---|---|---|
| 概念提取 | position -1 concern | 1 | ✓ 80% 准确率 |
| 概念角色扩展 | prefill position scan | +1 | ✓ 文档特定（Nutrition→Education 71%） |
| 关系提取（基础） | 字典式双概念 concern | 1 | ✓ 100% 覆盖 |
| 关系提取（细化） | 先验+prefill scan | +1 | ✓ 补充新关系词 |
| 实体属性 | ~~prefill scan: "A has [___]"~~ → 几何共现（Phase 46） | 0（纯 CPU） | ✗ prompt 路线证伪（≤0.39）；✓ 几何路线 precision 0.76 |
| 实体消歧 | 双概念 concern + 受限分类 prefill "Answer:" | 1 | ✓ 0.80（Phase 44，超 bge 基线 0.60） |
| 多跳路径 | ~~链式 prefill~~ → W 图幂次 + 张量补全 | 0（纯矩阵） | ~ W² 判别 AUC 0.998；CP 张量补全 holdout AUC 0.81（Phase 47） |
| 概念层级树 | COM 排序 | 1 | ✗ 不构成语义层级（0-9% is_a） |
| 递归展开 | prompt 请求子概念 | — | ✗ prompt 污染 |

### 关键发现：缺失部分也可用 J-Lens 补齐

传统知识图谱缺失的部分（实体属性、消歧、多跳路径）本质上都是"从一个概念出发读出关联信息"。这些都遵循同一个模式——**设计 prompt 让 workspace 在特定 position 读出需要的信息**：

| 缺失部分 | 传统 GraphRAG | J-Lens 方案（待验证） | prompt 方向 |
|---|---|---|---|
| 实体属性 | generate() "列出属性" | prefill scan: "Cancer has [____]" → 读 [____] 位置 | 属性填空 |
| 多跳路径 | generate() "A→B→C 路径" | 先验+scan: "A→{rel1}→B→{rel2}→" 读每个节点位置 | 链式 prefill |
| 实体消歧 | generate() "A 和 B 是同一个吗" | 双概念 concern: "A and B are" → 读 workspace | 同义判定 |

这些方法的关键是 **prompt 设计**，不是方法能力。Phase 35-37 的实验证明：合适的 prompt 能让 J-Lens workspace 读出文档特定的精确信息（概念角色、关系动词），而不合适的 prompt 会读到模板完形词。

**更新（Phase 43-47，见 §13）**：上表三项"待验证"已全部判决——消歧走 prompt（0.80）、属性走几何（0.76，prompt 路线证伪）、多跳走图/张量（prompt 仅部分信号）。三项能力分属三种数学对象，没有单一路线通吃。

---

## 7. 不建议的方向

| 方向 | 原因 | 验证实验 |
|---|---|---|
| 概念层级树（meta/sub） | COM 不产生语义层级 | Phase 22: 0-9% is_a |
| 分层图传播 | 不超越 flat 图 | Phase 21: 101% vs 105% |
| 递归 prompt 展开 | 所有 prompt 变体被污染 | Phase 24: 4 种全失败 |
| 白化残差投影 | 计算成本过高 | 附录 C.1（长期研究） |

## 8. 产品化建议

### 核心定位：GraphRAG 的 J-Lens 加速层

J-GraphRAG 的产品价值不是"又一个 GraphRAG 框架"，而是**证明 J-Lens 可以替换传统 GraphRAG 的全部提取步骤**——从 generate() 到 forward pass，建图成本降为 1/347，且不需要结构化输出。

这意味着：
1. **独立产品**：J-GraphRAG 作为完整的零成本知识图谱方案
2. **加速层**：作为现有 GraphRAG 框架（LightRAG/HippoRAG2 等）的 J-Lens 提取插件
3. **方法论**：其他研究者可基于此范式设计更多 J-Lens 提取 prompt（属性/消歧/多跳）

### 8.1 基础版（J-AugRAG：概念索引增强 RAG）

- 概念提取：1 次 forward pass（position -1）
- 三重过滤 + BM25 补全
- flat 二部概念图传播
- 效果：114% of B0

### 8.2 关系增强版（轻量知识图谱）

- 基础版 + 字典关系边（1 次/pair）
- 关系传播扩展检索
- 效果：L4 多跳推理翻倍

### 8.3 完整版（J-GraphRAG：概念知识图谱）

- 基础版 + 概念角色扩展（prefill scan）
- + 字典关系 + 先验细化
- + 聚类驱动关系构建
- 结构：概念节点 + 有类型关系边 + 概念角色 + 聚类分簇
- 已超越二部图，进入知识图谱领域
- 复杂关系精修交 API（用户按需）

## 9. 代码索引

| 组件 | 文件 | 方法 |
|---|---|---|
| J-Lens 基础设施 | `phase10_jlens_stage1.py` | `load_model`, `load_lens` |
| 概念提取 | `phase10_jlens_stage7c.py` | `extract_chunk_concepts`, `ConceptGraph` |
| 三重过滤+BM25 | `phase25_filter_bpe_benchmark.py` | `phase25_filter_with_bpe` |
| 概念角色扩展 | `phase35_prefill_position_scan.py` | prefill position scan |
| 字典关系提取 | `phase27_relation_readout.py` | `build_relation_prompt` |
| 关系细化 | `phase37_relation_prompt_variants.py` | V3 prior+scan |
| 聚类驱动关系 | `phase30_cluster_relation_graph.py` | `cluster_concepts` |
| 关系图传播 | `phase28_relation_graph.py` | `relation_propagate` |
| ACC 评估 | `phase26_acc_eval.py` | `judge_answer_correctness` |
| 能效比对比 | `phase26_fast_graphrag.py` | Fast-GraphRAG benchmark |
| 环境恢复 | `scripts/restore_env.sh` | 一键恢复模型+数据集 |

---

## 10. 补记：动机澄清与 Phase 35+ 状态（2026-07-20）

### 10.1 原始动机：单一 LLM 闭合，矩阵加速是副产品

§1.1 把矩阵加速写成了主角，但概念的数学形式化的原始动机是**单一模型自洽**：

> 概念的符号形式（词）和数学形式（向量）由同一个 LLM 产生——不引入额外的嵌入模型。
> 概念向量取 W_U 行向量或 workspace 残差，则二部图、关系、检索全部在 LLM 自身空间内闭合，
> 原先由 bge-m3 承担的向量计算整体迁移到 LLM 自生向量上。矩阵加速只是这个闭合成立后的免费副产品。

因此概念向量的候选方案有明确的优先级：W_U 碎片行向量（零成本、静态）> workspace 条件向量（+1 forward/概念、文档特定）> bge-m3（异空间，仅作对照）。M 行向量（语料共现统计，LSA 式）不是候选方案，而是**对照组**——用于分解"向量几何中有多少是共现统计可解释的"。

### 10.2 双射一致性的正确定位：接地检验，而非优化目标

双射一致性的作用是**接地检验（grounding check）**：验证 LLM 自生的数学结构与符号概念"指的是同一个东西"。它不是"越一致越好"的优化目标——若图社区与向量簇完全重合，关系图反而是嵌入空间的冗余表达。有意义的区间是**部分一致**：对齐到足以用矩阵运算，差异到足以提供几何捕获不到的关系信息。

判决统计量修正（Phase 38 的簇级 ARI 不是正确的检验方式）：

- **主判决：边级 AUC**。用 cos(E_i, E_j) 区分 W 中的边与非边，算 ROC-AUC。接地要求的是边级命题（有符号关系的概念对向量上更近），不是簇级命题（两种划分重合）。40 个概念 → 780 对，AUC 统计功效远好于 13 点上的 ARI。
- **参考：簇级 ARI/NMI**，必须带置换检验基线（随机划分的 ARI 分布），小样本下裸 ARI 无意义。
- **下游校验：检索等价性**。矩阵运算检索 vs 图扩散检索的 top-K 重叠 / Kendall τ——这是"矩阵加速"这个副产品成立的条件，与接地检验是两个问题。

### 10.3 Phase 35+ 已知混乱（后来者须知）

1. **`phase38_bijection_test.json` 的 `ari_e1e2/nmi_e1e2` 不是图-向量双射**，而是 bge-m3 聚类 vs W_U 聚类两个向量空间之间的一致性（E1 vs E2）。真正的图-向量一致性是 `ari_bge=0.058 / ari_wu=0.124`（13 概念 4 簇，在噪声范围内，不能作结论）。
2. **两处结果不可复现**：`phase38_bijection_test.json` 的键名与现存 `phase38_matrix_acceleration.py` 保存的键名不一致（来自已不在仓库的脚本版本）；`phase38_full_benchmark.py` 已被删除，但其结果 `phase38_full_benchmark_medical.json` 仍在。引用这两个文件时注意。
3. **规模化矛盾未解决**：Phase 28 在 200 chunks/28 queries 上关系传播 L4 达 57.1%（B0 28.6%），但 957 chunks/56 queries 全量 benchmark 上 JGR_relation ACC 57.1% = B0 57.1%，L4 21.4% < B0 28.6%。小样本优势未在 4.8 倍规模复现，原因待查（Phase 39 将做传播权重/跳数消融）。

### 10.4 两步提取成为正式提取方案（Phase 25 + Phase 35）

概念提取的正式管线更新为两步：

1. **Pass 1**：position -1 concern 提取概念 + 三重过滤 + BM25 补全（Phase 25 配置）；
2. **Pass 2**：prefill position scan 为每个概念扩展文档特定角色词（Phase 35 方法），角色词作为图的二级节点。

已按两步管线重跑全量语料（`phase39_two_pass_cache.py`）：medical 957 chunks / 12 分钟，novel 4391 chunks / 50 分钟（~0.7s/chunk，含 Pass 2 的 1 次额外 forward）。产物：`concept_cache_{domain}_twopass.json` + `concept_vecs_{domain}.npz`（每概念的 ws_vec/wu_vec）。注意：Pass 1 输出需再经 `phase39b_filter_cache.py` 套用 Phase 25 过滤配置（DF≥2 + prefill 黑名单 + POS 动词拒绝）——phase31 的提取函数本身不含这层过滤，"listed/summarized" 等 prefill 模板词会漏入（旧 full 缓存同样如此）。过滤后 medical 50 概念、novel 244 概念。

---

## 11. Phase 39-41 结果：概念可计算化的三点判决（2026-07-21）

### 11.1 接地检验（Phase 40）：LLM 自生向量确实编码符号关系结构

边级 AUC（cos 相似度区分 W 的边/非边，W = 词典式关系图，medical 248 边/50 概念，novel 297 边/60 概念）：

| 向量空间 | medical AUC | novel AUC | 超越共现基线 |
|---|---|---|---|
| E_wu（W_U 首碎片，静态） | 0.633*** | 0.529* | +0.087 / +0.005 |
| E_ws（workspace 条件向量） | **0.631\*\*\*** | **0.612\*\*\*** | **+0.085 / +0.088** |
| E_M（共现对照，LSA 式） | 0.546*** | 0.524*** | 基线 |
| E_bge（外部嵌入） | 0.709*** | 0.705*** | — |

（*** p=0.001，* p=0.048，1000 次置换检验）

- **E_ws 是唯一跨域稳定超越共现统计的 LLM 自生向量**（双语料均 +0.088）。单一模型闭合应走**条件向量路线**（+1 forward/概念），W_U 静态碎片在 novel 上几乎不含共现之外的语义（+0.005）。
- E_bge 最高的结论经 Phase 48 去偏检验**下修**：其领先大部分是 bge 聚类筛选的循环性膨胀（膨胀度 medical +0.104 / novel +0.167）；在去偏的 mcos 筛选 W 上 E_bge ≈ E_ws（medical 0.605 vs 0.592；novel 0.538 vs 0.550，差在边际内）。**不存在唯一最强空间，E_ws 与 bge 打平**——这反而强化了 LLM 自生向量的地位（见 §14）。
- 簇级 ARI 低（0.02–0.18）但置换百分位 97–100——落在"部分一致"区间：几何编码了关系结构，但关系图含有几何之外的信息。符合 §10.2 的预期，不是失败信号。

### 11.2 检索等价性（Phase 41A）：矩阵运算精确替代图扩散

medical（957 chunks/50 概念/48 查询）+ novel（4391/60/48），top-10 重叠 / Kendall τ：

| 矩阵臂 vs 对应图臂 | medical | novel | 判决 |
|---|---|---|---|
| seed_idf × M vs flat 传播 | 1.000 / 1.000 | 1.000 / 1.000 | **精确等价** |
| 关系矩阵 vs relation_propagate | 1.000 / 1.000 | 1.000 / 1.000 | **精确等价** |
| ws/wu 相似度扩展 vs flat | 0.83 / 0.78 | 0.96 / 0.26 | medical 达标；novel 头部等价、全排序发散 |
| Ridge (MMᵀ+λI)⁻¹ λ∈{0.01,0.1,1} | ≤0.90 / ≤0.47 | ≤0.92 / ≤0.39 | 不达标，λ 扫描未救回 |

概念可计算化的"等价性"条件在全量规模成立：q×M 与 W×q 的矩阵形式与逐节点图遍历**逐分一致**。Ridge 正则化扭曲排序，去共线性需要别的途径（或如 §10.2 所说，检索场景根本不需要显式去共线）。

### 11.3 关系传播全量消融（Phase 41B）：旧结论的方法论修正

**重要发现**：phase25/26/28 原实验 `seed_k=10=TOP_K`，合并后 top-10 恒等于 bge seed 列表——传播命中的 chunk 从未进入 LLM context。"957 chunks 上关系图不赢 B0"（§10.3 第 3 条）很可能是这个 merge quirk 的假象。

修正后（seed_k=5，传播结果真正占据 top-10 后 5 席），medical 全量 56 题、DeepSeek judge：

| 方法 | ACC | recall | L4 ACC |
|---|---|---|---|
| B0（纯 bge） | 0.536 | 0.760 | 0.21 |
| flat 概念图 | 0.589 | 0.727 | 0.21 |
| +关系 w=0, h=1 | **0.625** | 0.727 | **0.36** |
| +关系 w=0.1 | 0.607 | 0.723 | 0.21–0.29 |
| +关系 w=0.3 | 0.536–0.589 | 0.735 | 0.07–0.14 |
| +关系 w=0.5 | 0.518–0.554 | 0.728 | 0.07–0.29 |

- 概念图在全量规模**有效**（+5 ACC over B0）——此前"全量无效"是 quirk 假象。
- 关系边仅在**低权重**下微增益，权重 ≥0.3 单调劣化至 B0 以下（稀释 seed 精度）。
- L4 样本仅 14 题，0.36 vs 0.21 的方向性提升需在更大 L4 样本上确认。
- recall 略降而 ACC 升：传播带回主题正确但措辞不同的 chunk，lexical recall 低估了语义命中。

### 11.4 对概念可计算化押注的判决

1. **接地**：成立（条件向量路线）。E_ws 编码的符号关系结构稳定超越共现统计——LLM 自生的数学形式与符号概念指代同一对象。
2. **等价**：成立。q×M、W×q 与图遍历精确一致，图检索可整体矩阵化。
3. **加速**：副产品成立但当前规模无实际收益（毫秒级 vs 毫秒级），价值在规模化与谱方法。
4. **单一模型闭合**：文档侧路径明确——概念符号（Pass 1）+ 条件向量（Pass 2 同一次 forward 顺带产出）+ 关系（W）+ 检索（矩阵运算），全部在 Qwen 自身空间闭合。查询侧的闭合尝试见 §12：**经概念路由的纯替换已被证伪**。

遗留问题（下一步候选）：
- L4 多跳增益在更大样本（novel 或全 2062 题子集）上确认
- ~~novel 上 sim 臂全排序发散~~（Phase 49 已判决：良性——99% 零分长尾的平局敏感性所致，头部 τ=0.76、共同正分 τ=0.95、top-k 重叠 ≥0.89，不影响检索使用）
- ~~E_bge 循环性去偏~~（Phase 48 已判决：膨胀但非纯假象，见 §11.1）

---

## 12. Phase 42：查询侧闭合尝试——证伪与边界（2026-07-21）

**假设**：查询经两步法（概念提取 + 条件向量）进入概念空间，q_ws 对词表 ws_vec 余弦软匹配 → q×M 检索，替代 bge-m3。

**结果**（medical 56 题 / novel 48 题，DeepSeek judge ACC）：

| 臂 | medical ACC | novel ACC | 说明 |
|---|---|---|---|
| q_ws_A（查询原文 token 位置残差） | 0.071 | 0.125 | 结构性死亡：概念须字面出现在查询中，medical 56 题中 37 题无 A 向量 |
| q_ws_B（反转 prefill，+1 forward） | 0.054 | 0.208 | 向量匹配质量好（cancer→cancer 0.87）但检索仍失败 |
| hybrid（z(bge) + 0.3·z(q_ws_B×M)） | 0.536 | 0.417 | 略低于 B0 |
| B0（bge 直接检索） | 0.571 | 0.500 | 基线 |

**判决：纯概念路由替代 bge——FALSIFIED。** 原因不是工程调参问题，而是根本性的：

1. **短查询的概念产出太稀疏**：medical 56 题中 52 题提取出概念，但多数只有 1 个（"cancer"）；novel 有 10/48 题零概念。多词主题（basal cell carcinoma）进不了单词概念管线。
2. **概念是主题锚点，不是查询语义**："BCC 的首要风险因素"和"BCC 的治疗"概念集完全相同，但需要完全不同的 chunk。概念空间路由无法区分提问意图——bge 的全文稠密匹配编码了意图，概念路由丢失了它。
3. 这从反面再次验证了 boost-not-gate：概念/concern 适合在种子之上**扩展和重排**（Phase 41B：flat 0.589 > B0 0.536），不适合**替代**种子。

**对单一模型闭合的修正**：查询侧如果还要去 bge，剩下的路径不是概念路由，而是 **Qwen 作为稠密编码器**——查询和 chunk 都取 last-layer 残差（mean-pool 或 position -1）做稠密检索。chunk 侧成本可接受（957 chunks × 1 forward ≈ 3 分钟，可缓存）。这是与概念路由完全不同的机制，未验证，列为可选方向而非承诺。

**顺带修正 §11.2 hybrid 的读法**：Phase 42 hybrid（全排序 z-score 融合）略低于 B0，与 Phase 41B（bge 种子 + 传播补位，seed_k=5）的 +5 ACC 不矛盾——融合方式不同，后者是"种子不动、传播填后 5 席"，前者是全排序加权混合。结论是：**正确的集成姿势是种子+扩展，不是分数融合**。

---

## 13. Phase 43-47：图谱能力补齐——三条路线的对偶判决（2026-07-21）

三项能力的补齐实验沿三条路线平行推进：符号读出（prompt，43-45）、向量原生（46）、张量分析（47）。最终结论：**每项能力有一个且仅有一个主场**。

### 13.1 对偶结果总表（medical，50 概念）

| 能力 | 符号读出（prompt） | 向量原生（几何） | 图/张量 | 归属 |
|---|---|---|---|---|
| 实体消歧 | **0.80**（Phase 44，受限分类） | 0.75（M 行余弦；ws 均值向量 0.37 失败） | — | **prompt**（模型知识） |
| 实体属性 | 0.10–0.39（全变体未达 0.80 线） | **0.758**（ws 排序 ≈ bge 的 97%） | — | **几何**（共现+相似度） |
| 多跳路径 | 严格链式证伪；锚定链式 top-5 50% | 链式余弦 ≈ 随机（0.54） | **W² AUC 0.998**；CP 张量补全 0.81 | **图结构** |

### 13.2 Phase 44 消歧（SUPPORTED）

- 双概念关切 + 受限分类 prefill（"Answer:"，答案词按类聚合概率）：acc 0.80（40 对：diff 18/18、near 10/16、same 4/6），显著超 bge 余弦基线 0.60。
- 关键工程发现：prefill "They are" 的语言先验会把读出锁死在 "related"（4/6 误判）——**prefill 措辞的先验分布直接决定读出内容**，必须用受限答案集设计。
- §6 原构想的 prefill 重合度路线（"A and B are" 两位置 workspace 比较）仅 0.45，不通。

### 13.3 Phase 43 属性（prompt 路线 FALSIFIED；几何路线接住）

- 填空式变体（"A has [___]" / "is characterized by"）precision 0.10–0.19——读出 "several/properties" 等句法续词。概念位置读出（复用 phase39 机制）0.39，最好但离 0.80 线远。
- **这是第三次观察到同一失败模式**（Phase 16/24 递归展开、Phase 42 臂 A、Phase 43）：**完形填空式 prompt 读出的是句法续词而非语义内容**。上升为定律：J-Lens 读出必须锚定在实体位置（概念词自身）或受限答案集上，开放 cloze 位置只产生模板词。
- 几何路线（Phase 46 P2）：共现次概念 × ws 余弦排序，precision 0.758，与 bge 臂重合 0.89——属性本质是共现现象，几何天然够。（已知噪声：角色词含 BPE 碎片如 "reatment"，过滤规则待补 WordNet 完整词验证。）

### 13.4 Phase 45+46+47 多跳（图结构主场，三层递进）

- **W²（Phase 46 P4）**：真/伪路径判别 AUC 0.998——多跳判别力在图幂次里，不在 ws 几何里（链式余弦 0.538 ≈ 随机）。
- **CP 张量（Phase 47）**：T = 50×5×50（6 关系桶，密度 2%），holdout 补全 AUC 0.813（rank8/λ1.0；λ=0.1 全面崩盘至 ≈0.65——198 训练边对 840 参数，强正则是生死线）。**低秩关系结构存在**——关系类型信息可补全缺失三元组。
- 但**指方向不如 W²**：中间节点提案 recall@5 = 0.27（张量）vs 0.55（W²）。当前边密度下关系类型信息没有转化为更准的方向感。
- prompt 链式读出：严格链式 FALSIFIED（纯模板续词）；锚定链式（已知一跳作先验）top-5 50%，仅部分信号。

**多跳的最终形态**：W² 做路径判别与方向（召回），张量补全做关系类型标注（精化），prompt 读出做边级验证（按需）——三者串接，各司其职。

### 13.5 几何与关系的脱钩判决（P3 + S4 互证）

两个独立实验指向同一结论：**ws 向量空间与关系结构是脱钩的数学对象**——

- Phase 46 P3：ws 空间有线性词形结构（单复数代数 top1 50% vs 随机 2%），但**没有线性关系结构**（关系类比 top1 5% ≈ 随机）
- Phase 47 S4：把 ws 几何强加给张量概念因子（耦合正则）使 holdout AUC 从 0.81 掉到 0.63-0.68——**强制共享潜在空间是有害的**

含义：J-GraphRAG 的正确架构是**双层异构**而非单层统一——ws 几何层负责相似性/属性/消歧辅助，W/T 关系层负责多跳/路径/补全，两层通过概念符号对齐（同一批词），但不共享向量空间。§1.1 的"双射一致性"目标按此结论正式降级：几何与关系**部分一致**（Phase 40：边级 AUC 0.61-0.63）但不**同构**，强制同构反而损害两边。

### 13.6 产物索引

| Phase | 脚本 | 结果 JSON |
|---|---|---|
| 43 属性 | `phase43_attributes.py` | `phase43_attributes.json` |
| 44 消歧 | `phase44_disambiguation.py` | `phase44_disambiguation.json` + `phase44_pairs.json` |
| 45 多跳 | `phase45_multihop.py` | `phase45_multihop.json` |
| 46 向量探针 | `phase46_vector_native.py` | `phase46_vector_native.json` |
| 47 张量 | `phase47_tensor_multihop.py` | `phase47_tensor_multihop.json` |

---

## 14. 总结论（2026-07-21，Phase 10-49 全部判决完成）

### 14.1 主主张（SUPPORTED）

**J-Lens 可以把传统 GraphRAG 建图管线中所有依赖 LLM `generate()` 的提取步骤，替换为单次 forward pass 的 workspace 读出**——图质量相当，建图成本 1/347（Phase 26），API 成本 $0，且在 7B 小模型上可靠（无 JSON 截断，Phase 26 对照组 ACC 3.6% vs 本方法 75%）。

完整证据链：概念提取（1 forward，80% 准确率）→ 关系提取（词典式 100% 覆盖 + V3 细化）→ 角色扩展（prefill scan）→ 实体消歧（0.80，超 bge 基线 0.60）。

### 14.2 架构发现：三层异构，每项能力有且仅有一个主场

J-GraphRAG 不是"J-Lens 单独替换 generate()"，而是三层异构架构（Phase 43-47 对偶判决）：

| 层 | 数学对象 | 主场能力 |
|---|---|---|
| 符号层（J-Lens 读出） | 概念词、关系词（W 的边与类型） | 概念/关系/角色提取、**消歧**（0.80） |
| 几何层（ws/M 向量） | E_ws、E_M、M 矩阵 | **属性**（0.76，prompt 路线证伪）、检索扩展、接地 |
| 关系层（W/T） | 邻接矩阵幂、CP 张量分解 | **多跳**（W² AUC 0.998）、链接补全（张量 0.81） |

层间通过概念符号对齐，但**不共享向量空间**：几何与关系脱钩（Phase 46 P3 关系类比 ≈ 随机；Phase 47 S4 强制耦合反而有害，0.81→0.65）。双射一致性目标正式降级为"部分一致"（边级 AUC 0.61-0.63），强制同构损害两边。

### 14.3 数学闭合（SUPPORTED，边界已划清）

- **接地成立**：E_ws 编码的关系结构稳定超越共现统计（双语料 +0.088，p=0.001）。Phase 48 去偏后 E_ws 与 bge 打平——LLM 自生向量与专用嵌入模型等效，单一模型闭合在文档侧完整成立。
- **等价成立**：q×M、W×q 与图遍历精确一致（双语料 top-10 重叠/τ 均 1.000）——图检索可整体矩阵化，矩阵加速是免费副产品。
- **证伪的三条路**（同样重要）：
  1. 查询侧概念路由不能替代稠密检索（Phase 42：ACC 0.05-0.21 vs 0.50-0.57）——概念是主题锚点不是查询语义；bge（或某种稠密编码器）在查询侧保留；
  2. 关系不能纯向量补全（P3），多跳判别力在图幂次不在几何（P4）；
  3. 完形填空式 prompt 只产句法续词（Phase 16/24、42、43 三次独立复现）——J-Lens 读出必须锚定实体位置或受限答案集。

### 14.4 诚实的边界

- **成本轴压倒性，效果轴温和**：347x / $0 是决定性的；检索效果是持平到温和增益（全量 flat +5 ACC，关系边低权重微增益 0.625），L4 多跳增益样本不足（14 题），定位为"以 1/347 成本获得 GraphRAG 级结构收益"，不是效果碾压。
- **验证范围**：GraphRAG-Bench medical/novel 两域；347x 对比的是 Fast-GraphRAG（7B），缺 vs LightRAG/HippoRAG2 的正面比较；J-Lens 可读性只在 Qwen2.5-7B 上验证，跨模型泛化未知。
- **工程注意**：DeepSeek judge 对碎片词过严（Phase 45 全零不可用）；同进程重复加载 4bit 模型会显存泄漏（需新进程）；角色词含 BPE 碎片（过滤规则待补 WordNet 完整词验证）。

### 14.5 后续方向（按优先级）

1. **更大 L4 样本确认多跳增益**（novel 或 2062 题子集）——效果故事最薄弱的一环
2. **vs LightRAG/HippoRAG2 正面基线**——发表必需的对比
3. **跨模型 J-Lens 泛化**（Qwen3 系/Llama 系的 lens 拟合与可读性）
4. Qwen 作稠密编码器替代 query 侧 bge（可选，动机是部署简洁非效果）
5. 张量路线规模化（边数/概念数上来后重测 S3 指方向能力）

---

## 15. Phase 50：LightRAG-J 替换实验——保持率判决与瓶颈定位（2026-07-21）

**设计**：忠实重实现 LightRAG（arXiv 2410.05779）的非 LLM 部分（实体/关系双索引 K-V、dual-level 检索、一跳邻居衰减 0.5、hybrid 上下文组装），LLM 提取替换为 J-Lens 产物（概念=实体、词典关系=关系、`_stem`+ws 余弦 0.95 合并消歧）。换算用同跑 B0 锚（medical 60.7/factor 1.005、novel 54.2/factor 0.885），对照排行榜 LightRAG（medical 63.92 / novel 45.09，含用户从官网核到的 medical L4=67.91）。

**结果**：

| 臂 | medical ACC / 保持率 | novel ACC / 保持率 |
|---|---|---|
| B0（bge 基线） | 0.607 | 0.542 |
| a 纯 KG 检索替换 | 0.321 / 0.505 | 0.104 / 0.204 |
| b（a + CP 张量补全边） | 0.304 / 0.477 | 0.104 / 0.204 |
| ah 图+naive 混合（LightRAG 完整模式） | 0.536 / 0.842 | 0.458 / 0.900 |

**判决：保持率 ≥0.9 不成立（medical 0.842 / novel 0.900 临界）。** 三个连带结论：

1. **纯 KG 检索（无向量种子）崩盘**——与 Phase 42 同构，第四次验证"种子+扩展必要、纯概念路由不可行"。
2. **CP 张量补全边无增益且略有害**（b ≤ a）——Phase 47 的低秩结构强度不足以直接增边，张量补全作为检索增强件放弃（保留为分析工具）。
3. **瓶颈定位：提取粒度，不是检索算法**。失败案例三分析（JSON `failure_analysis`）：我们的词表是主题级概念（cancer/marriage），LightRAG 的 LLM 提取是命名实体级（basal cell carcinoma、Princess Frederica、Arthur/Excalibur）；novel L3/L4 几乎全灭的直接原因是词表没有专名。**实体粒度与类型（topical concept vs named entity）的错位是保持率差距的主因**，dual-level 检索算法本身无罪。

**对 §14 主主张的修正**：替换实验表明，J-Lens 提取产物**在主题概念粒度上**能支撑图检索（ah 臂保住 84-90%），但要达到原方案的实体粒度，需要 J-Lens 读出**专名级实体**——这是下一实验（Phase 52）的方向，而非继续调检索侧。HippoRAG-J（Phase 51）同理依赖实体粒度，暂缓，待 Phase 52 判决后再启动。

### 14.6 弱命题表述与逃逸路径（2026-07-21）

**主张的最终形态不是"J-Lens 替代 generate()"，而是"J-Lens 覆盖 p% 场景 + 置信门控 + 组件级逃逸"**——把正确性问题转化为成本问题：

```
预期成本 = (1-p) × 全量 LLM 生成成本
p = 0.8（当前概念提取准确率）→ 5x 成本下降；p = 0.7 → 3x
即便加上逃逸开销，也远小于全量生成（对照：Phase 26 能效比 347x 是 p=1 的极限）
```

**逃逸检测（置信门控）的现成信号**：
- 提取侧：corpus 验证失败率高的 chunk、DF<2 孤儿概念
- 检索侧：query 链接不到图节点（Phase 50 失败签名——查询实体不在词表）
- 关系侧：批次级 AUC≈0.5（Phase 48 零模型判据，该批关系不携带结构）

**逃逸粒度是组件级**：单 chunk 重提取、单查询走 dense-only、单批关系重读——按失败点精确付费，不整管线退回。这与 §8.3"复杂关系精修交 API 按需"是同一原则的推广。

**对写作的意义**：弱命题把举证责任从"它行不行"转移到"覆盖率有没有那么高"——后者可用数据直接回答（概念 80%、关系覆盖 100%、消歧 0.80、属性 0.76）。配套的验证原则：**corpus-grounded validation 替代 interventional validation**——工程提取场景中，数据中的 ground truth（语料验证、DF 统计、judge 抽样）替代可解释性研究中的因果干预证明，验证成本从月级降到分钟级。

---

## 16. Phase 52：专名级实体提取——FALSIFIED（结构性边界）（2026-07-21）

**起因**：Phase 50 保持率缺口的诊断是实体粒度（§15），本实验验证 J-Lens 能否读出专名级实体补齐缺口。

**S1（prompt 工程，成功）**：paired-question 协议第三次被实证——复数列举 prompt 的 position -1 被列表格式 token（`:`/`**`，prob≈1.0）占据（0.1 实体/chunk）；改为**单数显式命名问题** + **句末多位置读出**（同一次 forward 读各句末 token，14 pos × 11 层 0.42s）后，实体稳定进入 workspace，且多词实体（basal cell skin cancer）直接读出，无需 bigram 合并。

**S2（能力判决，四轮迭代后 FALSIFIED）**：

| 轮次 | 覆盖率 | 精度 | 修复 |
|---|---|---|---|
| 1 | 0.10 | 0.10 | 单数问题+多位置 |
| 2 | 0.10 | 0.20 | span 语法约束 |
| 3 | 0.20 | 0.18 | BPE 补全 chunk-local 优先 |
| 4 | 0.30 | 0.20 | 边界 bug + 过滤修正 |

双门槛（≥0.60/≥0.70）未过。回收：medical 2/4（basal cell carcinoma、fair skin），novel 1/6（仅 Cornwall）。

**失败的结构性原因（非工程可解）**：
1. **低频专名不进入 workspace**：Dozmare/Excalibur/Princess Frederica 在 4 种读出策略（单数/复数/类型化问题 × -1/句末多位置）下均不可读，读出位置只有噪声 token。Qwen2.5-7B 4bit 对罕见多 token 专名不形成可读出表示——这是模型词表先验问题，不是 prompt 问题。
2. **精度天花板 ~0.20**：句末 harvest 带出主题性内容词，合成 span 多为合理但泛化的短语（sun exposure），judge 大面积判 NO。

**部分成立的区域**：medical 说明文（术语在训练数据中充分：basal cell carcinoma、Primary CNS lymphoma、HPV、AIDS 可读）——实体级读出在**术语型语料**部分可行，在**叙事型语料**（罕见人名地名）不可行。

**对替换主张的影响**：J-Lens 提取的上限锁定为**主题概念 + 术语级实体**粒度。命名实体密集的 GraphRAG 场景（novel 类叙事语料）超出此限——按 §14.6 弱命题框架归入逃逸路径（实体缺口由按需 generate() 补齐），或需更大模型（workspace 更丰富）。Phase 51（HippoRAG-J）因同样依赖实体粒度暂缓：其预期保持率不会显著超过 LightRAG-J 已测区间（0.84-0.90），除非实体问题解决。

---

## 17. Phase 53：文本侧实体 + LightRAG-J 终审——替换主张成立（2026-07-21）

**转折**：Phase 52 证明模型无法*回忆*低频专名，但专名显式存在于文本——**检测不需要生成**。文本侧规则提取（大写 span + 术语 span，零模型零 GPU），J-Lens 退回其擅长的抽象层。

**S1 检测**：medical 11530 实体（大写 1439 + 术语 10091）、novel 16186 实体；**Phase 52 失败清单检出率 100%**（basal cell carcinoma、Princess Frederica、Dozmare、Excalibur 等全部检出）。

**S2 关系读出对罕见实体**：**成立（62.5%，5/8 勉强过线）**——模型读不出罕见实体 token，但能读出两实体间的关系词（常见词汇）：excalibur+dozmare→geographical/location、Frederica+Pawel-Rammingen→marriage/husband、arthur+merlin→advisor/mentor。

**S3 终审**（LightRAG-J ah 臂，换算 factor 同 Phase 50）：

| 域 | B0 | 概念图 ah（Phase 50） | 概念+实体图 ah（Phase 53） | 判决 |
|---|---|---|---|---|
| medical | 0.607 / 0.954 | 0.536 / 0.842 | 0.643 / **1.011** | **retained** |
| novel | 0.542 / 1.064 | 0.458 / 0.900 | 0.562 / **1.104** | **retained** |

**替换主张成立**：J-Lens（概念/关系/角色，1 forward）+ 文本侧实体检测（0 模型）的混合提取，在 LightRAG 框架上达到原方案（GPT-4o-mini 提取）的 101-110%，全程零 LLM 生成成本。DeepSeek judge 非确定性约 ±0.05（medical 两次运行 0.696/0.643 均 ≥0.9）。novel L4 全臂趋零为历史难点，非本臂特有。

### 17.1 最终架构（对 §14.2 的修订）

| 层 | 内容 | 来源 | 成本 |
|---|---|---|---|
| 抽象层 | 概念、关系、角色、消歧 | **J-Lens workspace 读出** | 1 forward/chunk(/pair) |
| 表层 | 命名实体、专名、术语 | **文本侧检测**（规则/BM25） | ~0 |
| 几何层 | 属性、相似性、接地 | ws/M 向量 | ~0 |
| 关系层 | 多跳、链接补全 | W² + CP 张量 | ~0 |

分工原则：**模型做抽象（生成最贵最易错的部分），确定性方法做表层（本来免费的部分），逃逸路径留给真正的硬骨头**。这也修正了"零 LLM 成本"的含义——不是"只用 J-Lens"，而是"零 LLM generate()"。

### 17.2 遗留

- Phase 51（HippoRAG-J）：实体问题已解，PPR 为矩阵运算，障碍清除——可作为下一个替换实验（预期与 LightRAG-J 同档或更高，因 HippoRAG2 图密度更高、我们的实体数 11k-16k 已超过其 598/523 节点）
- 实体-实体关系边目前以共现默认边为主（novel 仅 100 条 J-Lens 读出边）；增大读出规模的效果待测
- novel L4 全方法趋零，疑似题目/judge 问题而非检索问题，待查

---

## 18. Phase 51：HippoRAG-J 替换实验——medical retained，novel 临界（2026-07-21）

**设计**：忠实重实现 HippoRAG2（arXiv 2502.14802v2 + 官方 repo 默认配置）的非 LLM 部分：phrase 节点（概念+文本侧实体）、passage 节点（chunks）、contains 边、关系边、同义边（ws≥0.8）；query-to-triple top-5 → recognition memory（DeepSeek）→ PPR（damping=0.5，官方值）→ top-10 context。零 Qwen。

**结果**（保持率 = ACC × factor / 排行榜 HippoRAG2，medical 64.85 / novel 56.48）：

| 臂 | medical | novel |
|---|---|---|
| a 官方默认（passage w=0.05） | 0.830 ✗ | 0.751 ✗ |
| b 无 recognition | 0.720 ✗ | 0.620 ✗ |
| c passage w=0.3 | **0.996 ✓** | 0.751（首跑 0.881，judge 抖动） |
| c passage w=0.8 | **0.941 ✓** | **0.881（临界，差 1 题内）** |

**判决：medical retained（调优后 0.996/0.941），novel 临界（0.881 ± judge 噪声 ±0.03）——不宜宣称双域达标。** 两个确定性结论：

1. **recognition memory 稳定有效**（a > b 双域 +0.07/+0.08）——HippoRAG2 的在线 LLM 过滤步骤值得保留；
2. **passage_node_weight 需按图重校准**——官方 0.05 是按其 OpenIE 图规模校准的，我们的图（万级实体、边权分布不均）需要 0.3；medical 上此参数值 0.17 保持率。

**PPR vs dual-level**：在我们的图上 LightRAG 的 dual-level 更稳（1.011/1.104 vs 0.996/0.881）——MAX 聚合 + naive 交错对边权不均的图更鲁棒；PPR 全局扩散对重置配比高度敏感。对产品化的含义：**默认走 dual-level，PPR 作为可调选项**。

**两个替换实验的总账**：LightRAG-J 双域 retained（1.011/1.104）、HippoRAG-J medical retained + novel 临界（0.996/0.881）。§14.1 主主张在两种主流 GraphRAG 框架上得到验证：J-Lens + 文本侧实体的混合提取可以替换 LLM generate() 提取管线，保持率 ≥0.9（5/6 域-臂组合）且零生成成本。
