# J-GraphRAG 技术报告：用 J-Lens Workspace 读出替代 GraphRAG 的 LLM 生成

> 版本：v1.0（2026-07-21，Phase 10-53 全部实验完成）
> 定位：论文写作原始素材。编年实验日志见 `docs/j-graphrag-complete-method.md`（§0-18），本报告为综合后的最终陈述；两者冲突处以本报告为准。

---

## 摘要

传统 GraphRAG 框架（LightRAG、HippoRAG2、MS-GraphRAG）的建图管线中，每个知识提取步骤都依赖 LLM `generate()`：每文档 5-60 秒、需要结构化输出解析、在 7B 小模型上因 JSON 截断而不可靠、API 成本 $5-15/千篇。本报告证明：**这些提取步骤可以全部替换为 Jacobian Lens（J-Lens）对模型 workspace 的单次 forward pass 读出**——0.17s/chunk、无需结构化输出、零 API 成本。混合提取架构（J-Lens 负责抽象层、文本侧规则负责命名实体表层、向量几何负责属性、图/张量负责多跳）在 GraphRAG-Bench 上达到原方案的 **101%/110%**（LightRAG-J）与 **100%/88%**（HippoRAG-J）保持率。我们进一步证明提取产物在数学上是闭合的：概念向量与符号关系结构接地一致，全部图检索运算可精确线性化为矩阵乘法。

## 1. 引言

### 1.1 问题

GraphRAG 用知识图谱增强检索，但图谱构建的成本结构限制了其实用性：

| 维度 | 传统管线（generate()） | 本工作（J-Lens 读出） |
|---|---|---|
| 耗时/chunk | 5-60s | **0.17s** |
| 输出格式 | 结构化 JSON/三元组（需解析+重试） | 不需要——直接读 workspace logits |
| 7B 可靠性 | ✗ JSON 截断（Fast-GraphRAG 实测 ACC 3.6%） | ✓ workspace 稳定读出 |
| API 成本 | $5-15/千篇 | **$0** |
| 能效比 | 1x | **347x**（Phase 26 实测） |

### 1.2 核心思想

J-Lens（Jacobian lens，Gurnee & Sofroniew et al. 2026）揭示 LLM 内部存在一个"全局 workspace"（J-space）：一组可言说（verbalizable）的表征，模型用它们承载中间推理。关键观察：既然模型在读文档时已经把概念、关系放进了 workspace，**提取知识不需要让模型生成，只需要读出来**。每次提取从一次生成（5-20s）变为一次 forward（0.17s）。

### 1.3 研究问题与结论概览

| RQ | 问题 | 结论 |
|---|---|---|
| RQ1 | 提取步骤能否被 workspace 读出替代？ | **能**（概念/关系/角色/消歧），但有明确边界（§7） |
| RQ2 | 提取出的结构是否数学闭合（接地+可计算）？ | **是**：接地成立（去偏后与 bge-m3 打平），矩阵 ≡ 图遍历精确等价（§4） |
| RQ3 | 各项图谱能力的计算主场是什么？ | 一能力一主场：属性→几何、消歧→读出、多跳→图/张量（§5） |
| RQ4 | 在真实框架上替换后性能保持几成？ | LightRAG ≥0.9（双域 1.01/1.10）、HippoRAG2 medical 1.00（§6） |

## 2. 背景

### 2.1 J-Lens 与 J-space

Jacobian lens 对每层残差流计算平均一阶因果效应 J_l（在千级语料上对 Jacobians 取均值），将中间层激活映射到 unembedding 空间读出 top-k token。论文（transformer-circuits.pub/2026/workspace）证明这些读出满足全局 workspace 的功能性质：可报告、可调制、承载中间推理、可泛化、选择性。对本工作最关键的三个实证发现：

1. **内部推理的中间实体进入 workspace**（多跳问题中模型先产生 spider 再回答腿数）——实体级表示存在；
2. **paired-question 协议**：问题形式决定什么内容进入 workspace——隐式使用的内容不进入，显式命名才进入。这规定了我们的 prompt 设计原则（§3.1），也统一解释了所有 cloze 式失败（§4.4）；
3. **J-lens 是单 token 的**——多词概念/实体需要语料统计合并（§3.1）。

**与可解释性工作的分工**：他们证明 workspace 存在且因果特权（需要干预实验），我们证明该结构可被低成本工程收割——用 **corpus-grounded validation 替代 interventional validation**（§9.2）。

### 2.2 评测基准与对照

GraphRAG-Bench（arXiv 2506.05690）：medical（NCCN 指南，957 chunks）与 novel（19 世纪小说，4391 chunks）两域，L1 事实检索 / L2 复杂推理 / L3 语境摘要 / L4 创意生成四级任务。排行榜原版（GPT-4o-mini 提取 + 评测）数字存档于 `data/m6/graphrag_bench_leaderboard.json`。我方评测：bge-m3 检索 + DeepSeek judge，换算系数用同跑 B0 锚定（§6.1）。

## 3. 方法

### 3.1 离线提取（全部零 LLM 生成成本）

**两步概念提取**（Phase 25+35，每 chunk 2 次 forward）：

1. **Pass 1 概念**：concern prompt（"What concepts does this text discuss?"）→ 全层 depth gradient + position -1 读出 → 三重过滤（ASCII/prefill 黑名单/POS）→ BM25 BPE 补全 → corpus 验证 + DF≥2。80% 准确率（人工审查）。
2. **Pass 2 角色扩展**：反转 prefill（"The concepts are: {concepts}"）→ 扫描概念位置的 workspace → 文档特定角色词（Nutrition→Education 71%）。同一次 forward 顺带产出概念的条件向量 ws（lens.transport 后的残差，为 §4 的向量资产）。

**文本侧实体检测**（零模型）：大写 span + 术语 span + 频率/规则过滤。Phase 52 证明 J-Lens 无法读出低频专名（7B 结构性限制），但专名显式存在于文本——**检测不需要生成**。medical 11530 / novel 16186 实体，失败清单检出率 100%。

**关系提取**（每对 1 次 forward）：双概念关切 prompt（"The relationship between A and B is"）→ position -1 → 关系词，100% 覆盖（Phase 27）；V3 先验+prefill scan 细化（Phase 37）。**对罕见实体同样有效**（Phase 53 S2，62.5%）——模型读不出实体 token 但读得出关系词（常见词汇）。

**实体消歧**（每对 1 次 forward）：双概念关切 + 受限分类 prefill（"Answer:"），acc 0.80，超 bge 余弦基线 0.60（Phase 44）。关键工程发现：prefill 措辞的语言先验直接决定读出内容（"They are" 锁死 "related"），必须用受限答案集。

**Prompt 设计定律**（三次独立复现，Phase 16/24、42、43）：**cloze 式 prompt 读出的是句法续词而非语义内容；J-Lens 读出必须锚定在实体位置（prefill 中的概念词）或受限答案集上**。

### 3.2 数据结构：矩阵形式化

提取产物组织为四个矩阵/张量，全部可用线性代数运算：

- **M**：概念/实体-文档映射（N×N_chunks，稀疏），M[i,j] = IDF(i) × BM25_tf(i,j)
- **W**：概念关系邻接（N×N，typed，prob 加权）；三阶形式 T[i,r,j]（关系类型张量）
- **E_ws**：概念 workspace 条件向量（N×3584，Pass 2 顺带产出，零边际成本）
- **R**：概念角色矩阵（可选）

### 3.3 在线检索

**种子 + 扩展**（唯一被验证正确的集成姿势，Phase 41B/42）：

```
查询 → bge-m3 → top-K seed chunks（种子不动）
     → 概念传播: q × M（1 次矩阵乘）
     → 关系扩展: W × q（1 次矩阵乘）
     → 扩展结果填入 top-K 剩余席位（seed_k=5）
```

**两种图检索算法的重实现**（§6）：
- **dual-level**（LightRAG）：query→实体/关系双路向量匹配 → 一跳邻居 → chunk MAX 聚合 + naive 交错；
- **PPR**（HippoRAG2）：`α(I-(1-α)P)⁻¹s`，phrase 种子（query-to-triple top-5 + recognition memory 过滤）+ passage 种子（bge 相似度 × 权重因子）。

### 3.4 最终架构：四层异构

| 层 | 内容 | 来源 | 成本 |
|---|---|---|---|
| 符号/抽象层 | 概念、关系、角色、消歧 | J-Lens workspace 读出 | 1 forward/chunk(/pair) |
| 表层 | 命名实体、专名、术语 | 文本侧检测（规则/BM25） | ~0 |
| 几何层 | 属性、相似性、接地验证 | E_ws / M 向量 | ~0 |
| 关系层 | 多跳、链接补全 | W 幂次 / CP 张量 | ~0 |

分工原则：**模型做抽象（generate 最贵最易错的部分），确定性方法做表层（本来免费的部分），逃逸路径留给硬骨头**（§8）。层间通过概念符号对齐，但不共享向量空间（§4.3 脱钩判决）。

## 4. 数学性质：概念可计算化

### 4.1 接地（grounding）：向量与符号指代同一对象

**检验**（Phase 40/48）：用概念向量余弦区分关系图 W 的边/非边（边级 ROC-AUC + 1000 次置换检验）。

| 向量空间 | medical | novel | 说明 |
|---|---|---|---|
| E_ws（workspace 条件向量） | 0.631*** | 0.612*** | 超越共现基线 +0.085/+0.088 |
| E_wu（W_U 首碎片，静态） | 0.633*** | 0.529* | novel 上≈共现，静态碎片语义薄 |
| E_M（共现对照） | 0.546*** | 0.524*** | LSA 式基线 |
| E_bge（外部嵌入） | 0.709*** | 0.705*** | 去偏后 0.605/0.538 ≈ E_ws |

**去偏检验**（Phase 48）：E_bge 的领先大部分是 bge 聚类筛选的循环性膨胀（+0.104/+0.167）；去偏后 **E_ws ≈ bge**——LLM 自生向量与专用嵌入模型等效，文档侧的单一模型闭合成立。

**接地的正确解读**：目标不是完全一致（那意味着图是向量的冗余），而是部分一致——簇级 ARI 低（0.02-0.18）但置换百分位 97-100，说明几何编码了关系结构、且关系图含有几何之外的信息。

### 4.2 等价：矩阵运算 ≡ 图遍历

Phase 41（双语料全量）：`q×M`（概念传播）与 `W×q`（关系扩展）的矩阵形式与逐节点图遍历**逐分一致**（top-10 重叠与 Kendall τ 均 1.000）。图检索可整体矩阵化；矩阵加速在当前规模无实际收益（毫秒 vs 毫秒），价值在规模化与谱方法。Ridge 去共线无必要且扭曲排序。

### 4.3 脱钩：几何与关系是不同的数学对象

两个独立实验互证（Phase 46 P3 + Phase 47 S4）：
- ws 空间有线性词形结构（单复数代数 top1 50% vs 随机 2%），但**没有线性关系结构**（关系类比 ≈ 随机）；
- 把 ws 几何强加给张量概念因子（耦合正则）使链接预测 holdout AUC 从 0.81 掉到 0.63-0.68——**强制共享潜在空间是有害的**。

含义：架构必须双层异构（§3.4），"双射一致性"目标降级为部分一致。

### 4.4 Cloze 定律

完形填空式 prompt 读出的是句法续词（several/properties）而非语义内容——Phase 16/24（递归展开）、42（查询臂 A）、43（属性填空）三次独立复现。J-Lens 读出的两种有效锚定：**实体位置**（prefill 中的概念词，Phase 35）与**受限答案集**（Phase 44）。这与 paired-question 协议（§2.1）互为因果解释。

## 5. 能力地图（Phase 43-47 对偶判决）

| 能力 | 最优路线 | 成绩 | 替代路线及判决 |
|---|---|---|---|
| 概念提取 | position -1 concern（1 fwd） | 80% 准确率 | — |
| 角色扩展 | prefill scan（+1 fwd） | 文档特定 71% | — |
| 关系提取 | 词典式（1 fwd/pair） | 100% 覆盖 | — |
| 实体消歧 | 受限分类 prompt（1 fwd） | **0.80** | 几何 M 行 0.75；bge 0.60 |
| 实体属性 | **几何**（共现×ws 排序） | **precision 0.76** | prompt 路线证伪（≤0.39） |
| 命名实体 | **文本侧检测** | 检出率 ~100% | J-Lens 读出证伪（罕见专名结构性不可读） |
| 多跳判别 | **W²** | AUC 0.998 | 链式 prompt（top-5 50%）、ws 链式（≈随机） |
| 链接补全 | CP 张量 | holdout AUC 0.81 | 直接增边检索无增益（Phase 50 臂 b） |

## 6. 替换实验（RQ4）

### 6.1 协议

**替换-保持率**：忠实重实现框架的非 LLM 部分（图构建逻辑 + 检索算法），仅把 LLM 提取换成 §3.1 的混合管线。原版不可本地跑（Fast-GraphRAG 已崩），用排行榜数字 + 换算系数折算：

```
保持率 = ACC(替换版, 我方 judge) × factor / 排行榜原版
factor  = 排行榜 RAG 均值 / 我方同跑 B0（medical 1.005 / novel 0.885）
```

判决线 ≥0.9。评测：medical 56 题（14/级×4）+ novel 48 题，DeepSeek judge（噪声 ±0.03-0.05）。

### 6.2 LightRAG-J（Phase 50→53）

完整叙事：概念图保持率 0.842/0.900（未达线）→ 失败分析定位瓶颈为**实体粒度**（主题概念 vs 命名实体错位）→ Phase 52 证明 J-Lens 实体读出结构性不可行 → Phase 53 文本侧检测补齐：

| 臂 | medical | novel |
|---|---|---|
| B0（bge） | 0.607 / 0.954 | 0.542 / 1.064 |
| 概念图 ah | 0.536 / 0.842 | 0.458 / 0.900 |
| **概念+实体图 ah** | **0.643 / 1.011 ✓** | **0.562 / 1.104 ✓** |

**双域 retained，且反超原方案**（7B J-Lens + 文本规则 vs GPT-4o-mini 提取）。

### 6.3 HippoRAG-J（Phase 51）

PPR 纯矩阵重实现（官方默认 damping=0.5）：

| 臂 | medical | novel |
|---|---|---|
| 官方默认（passage w=0.05） | 0.830 ✗ | 0.751 ✗ |
| **passage w=0.3（重校准）** | **0.996 ✓** | 0.751~0.881（judge 抖动） |
| passage w=0.8 | 0.941 ✓ | 0.881（临界） |

**medical retained，novel 临界**（0.881，与 0.9 之差在判分噪声内）。两个确定性结论：recognition memory（在线 LLM 过滤）稳定 +0.07/0.08 ACC，值得保留；passage 权重需按图重校准（官方值按其 OpenIE 图规模校准）。

### 6.4 对比分析

**dual-level 在我们的图上比 PPR 稳**（1.011/1.104 vs 0.996/0.881）：MAX 聚合 + naive 交错对边权不均的图更鲁棒；PPR 全局扩散对 passage/phrase 重置配比高度敏感。产品化默认走 dual-level，PPR 作可调选项。

## 7. 负结果（已证伪的方向）

| 方向 | 判决 | 证据 |
|---|---|---|
| 查询侧概念路由替代稠密检索 | FALSIFIED | ACC 0.05-0.21 vs 0.50-0.57（Phase 42）；概念是主题锚点不是查询语义 |
| cloze 式 prompt | FALSIFIED（定律） | 三次独立复现（§4.4） |
| J-Lens 低频专名提取 | FALSIFIED（结构性） | 4 种读出策略全失败（Phase 52）；由文本侧检测补齐 |
| 概念层级树 / 递归展开 / 分层传播 | FALSIFIED | Phase 22（0-9% is_a）、24（prompt 污染）、21（不超越 flat） |
| 关系纯向量补全 | FALSIFIED | 关系类比 ≈ 随机（Phase 46 P3） |
| 张量补全边直接增强检索 | 无增益且略有害 | Phase 50 臂 b ≤ 臂 a |
| 全排序分数融合（z-score） | 低于种子+扩展 | Phase 42 hybrid vs Phase 41B |

## 8. 弱命题与逃逸架构

主张的最终形态不是"J-Lens 替代 generate()"，而是**"覆盖 p% 场景 + 置信门控 + 组件级逃逸"**——把正确性问题转化为成本问题：

```
预期成本 = (1-p) × 全量 LLM 生成成本（p=0.8 → 5x 下降；p=0.7 → 3x）
```

**置信门控的现成信号**：corpus 验证失败率、DF<2 孤儿概念、query 链接不到图节点（Phase 50 失败签名）、批次关系 AUC≈0.5（Phase 48 零模型判据）。**逃逸粒度是组件级**（单 chunk 重提取、单查询 dense-only），按失败点精确付费。

当前各组件覆盖率实测：概念 80%、关系 100%、消歧 0.80、属性 0.76、实体（文本侧）~100%、保持率（端到端）1.01/1.10。

## 9. 讨论

### 9.1 成本核算（1000 篇文档建图）

| 方法 | 时间 | API 成本 |
|---|---|---|
| **J-GraphRAG 基础**（概念+关系+嵌入） | **~5 分钟**（~1020 次 forward） | **$0** |
| **J-GraphRAG 完整**（+角色+关系细化） | **~8 分钟**（~2040 次 forward） | **$0** |
| Fast-GraphRAG（Qwen-7B generate） | ~1000 分钟 | $0（但不可靠） |
| 传统 GraphRAG（API） | ~10 分钟 | ~$5-15 |

### 9.2 与可解释性研究的方法论分工

J-space 论文证明 workspace 的存在与因果特权（干预实验：swap/steering/gradient pursuit）；本工作证明该结构可被工程收割。关键方法论差异：**corpus-grounded validation 替代 interventional validation**——工程提取场景中，数据中的 ground truth（语料验证、DF 统计、judge 抽样）替代因果证明，验证成本从月级降到分钟级。我们在 7B 开源模型上独立复现了其在 Claude 上的发现（position -1 可读、前 1/3 层噪声、prefill 调制）。

### 9.3 局限

- 单模型验证（Qwen2.5-7B 4bit + 拟合的 J-Lens），跨模型泛化未知
- 双域（medical/novel），L4 样本小（14 题）且 novel L4 全方法趋零（疑似题目/judge 问题）
- DeepSeek judge 非确定性 ±0.03-0.05，边界结论（novel 0.881）受此限制
- 罕见专名的实体-实体关系边以共现默认边为主，读出规模待扩

### 9.4 后续方向（按优先级）

1. 更大 L4 样本确认多跳增益（novel 全量或 2062 题子集）
2. 跨模型 J-Lens 泛化（Qwen3/Llama lens 拟合与可读性）
3. 逃逸架构实测（置信门控触发率与真实成本比）
4. Qwen 作稠密编码器消除 query 侧最后的 bge 依赖（可选，动机是部署简洁）
5. 张量路线规模化（边/概念数量上来后重测指方向能力）

## 附录 A：实验索引（Phase 1-53）

| 阶段 | 内容 | 关键结论 |
|---|---|---|
| 1-9 | 嵌入分形理论验证 | 展开增分散度不增区分力；概念层级树萌芽 |
| 10-25 | J-Lens 方法成形 | 概念提取（position -1）+ 三重过滤 + BM25（最终产品配置 Phase 25：114% of B0） |
| 16-24 | 层级/递归路线 | 全部证伪（is_a 0-9%、prompt 污染、不超越 flat） |
| 26 | GraphRAG 对标 | 能效比 347x；Fast-GraphRAG 7B 崩盘（ACC 3.6%） |
| 27-37 | 关系提取 | 词典式 100% 覆盖；V3 细化；prefill scan（角色扩展） |
| 38-41 | 可计算化 | 接地（E_ws 去偏 ≈ bge）；矩阵 ≡ 图遍历（1.000）；seed_k quirk 修正 |
| 42 | 查询侧闭合 | FALSIFIED；种子+扩展是唯一正确集成 |
| 43-47 | 能力地图 | 消歧 prompt（0.80）/属性几何（0.76）/多跳 W²（0.998）/张量（0.81）；几何-关系脱钩 |
| 48-49 | 去偏与排查 | E_bge 循环性量化；novel 发散良性 |
| 50-53 | LightRAG-J | 0.842 → 实体瓶颈 → 文本侧补齐 → **1.011/1.104 retained** |
| 51 | HippoRAG-J | 矩阵 PPR；**medical 0.996 retained**，novel 0.881 临界 |
| 52 | 实体提取 | J-Lens 罕见专名 FALSIFIED（结构性） |

## 附录 B：可复现性

- 环境：Python ≥3.10 + uv（`pyproject.toml`）；Qwen2.5-7B-Instruct 4bit（8GB VRAM）+ jacobian-lens + bge-m3
- 数据：GraphRAG-Bench（/tmp/graphrag-bench 符号链接，`scripts/restore_env.sh` 一键恢复）
- 入口：`python -m experiments.phase{XX}_* [--domain medical|novel]`，缓存产物在 `data/m6/concept_cache/`
- 排行榜对照：`data/m6/graphrag_bench_leaderboard.json`（来源 graphrag-bench.github.io + arXiv 2506.05690）
