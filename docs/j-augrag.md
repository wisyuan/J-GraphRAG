# J-AugRAG：基于 J-Lens 的零 API 成本概念索引增强检索

## 发布文档 v1（2026-07-15）

---

## 摘要

J-AugRAG 是一种在消费级 GPU（8GB VRAM，单卡）上运行的检索增强方案。通过读取开源大语言模型（Qwen2.5-7B-Instruct）的 J-Lens workspace 激活来提取文档概念，构建二部概念图（chunk↔concept）+ 共现关系边，以图传播增强检索。整个建图流程不消耗任何大模型 API token。

### 关键数字

| 指标 | 数值 |
|---|---|
| 建图速度 | **0.173s/chunk**（J-Lens 1 次 forward pass） |
| vs SOTA（Fast-GraphRAG） | **347 倍快**（同 Qwen2.5-7B，SOTA 为 ~60s/chunk） |
| VRAM 占用 | **5.76GB**（4-bit 量化 + J-Lens） |
| 检索效果（evidence recall） | **114% of RAG** |
| 检索效果（ACC，bge-large） | **75.0%**（超过 B0 RAG 的 71.4%） |
| 排行榜等效 ACC（medical） | **~64%**（接近 HippoRAG2 的 64.9%） |
| L4 多跳推理 | **57.1%**（flat 图的 2 倍，关系图传播） |
| API 成本 | **$0** |

### 为什么叫 J-AugRAG

这不是一个完整的知识图谱系统——它是 **RAG 的概念索引增强**。J-Lens 读取模型内部的概念表示，构建轻量级概念图作为向量检索的补充层。"Aug"代表 Augmented（增强），不是替代。

---

## 1. 架构

```
┌───────────────────────────────────────────────────┐
│ 离线建图（零 API 成本）                             │
│                                                     │
│  文档 → concern prompt → J-Lens forward pass        │
│         → workspace 层 logits → 概念词               │
│         → 三重过滤 + BM25 补全                       │
│                                                     │
│  概念词 → bge-m3 余弦聚类 → 簇内/簇间关系提取        │
│         → 共现筛选 → 双概念关切读关系                 │
│                                                     │
│  产出：二部概念图（chunk↔concept）                   │
│       + 共现关系边（concept↔concept）                │
│       + IDF + BM25 tf 边加权                         │
├───────────────────────────────────────────────────┤
│ 在线检索                                            │
│                                                     │
│  查询 → bge-m3 → 余弦 top-K seed chunks             │
│       → 概念图传播（seed 概念 → 共享概念的 chunk）    │
│       → 关系扩展（seed 概念 → 关系边 → 扩展概念）     │
│       → 合并输出                                     │
└───────────────────────────────────────────────────┘
```

## 2. 核心组件

### 2.1 J-Lens 概念提取

对每篇文档做一次 forward pass，读取 workspace 层（L26）的概念表示。关键设计是**关切耦合**——用 chat template 的 assistant prefill 强制模型在读出位置形成概念：

```
User: What concepts does this text discuss? List 5 one-word concepts.
      [文档内容]
Assistant: The concepts discussed are
                                  ↑ J-Lens 读出位置
```

J-Lens 的 Jacobian transport 把 workspace 残差映射到词表空间，top-k logits 对应概念词。

**为什么不用 generate()？** 实验证明（Phase 26）：同一个 Qwen2.5-7B，generate() 做 entity extraction 每文档 60 秒（vs J-Lens 0.173 秒），且 JSON 截断导致建图失败（ACC 3.6%）。J-Lens 读 workspace logits 不需要结构化输出——1 次 forward pass 即得概念词。

### 2.2 三重过滤 + BM25 补全

J-Lens 读出包含三类噪声。最终过滤器配置（Phase 25，经 A/B 测试验证最优）：

```
原始概念
  → ASCII 过滤（排除多语言碎片：usuarios/novità/männer）
  → Prefill 词黑名单（排除 prompt 自身词汇：concepts/discussed/types）
  → POS 过滤（排除动词 -ing/-ed：assessed/evaluated/reported）
  → BM25 补全（BPE 前缀 → 语料频率最高的完整词：odyn→odynophagia）
  → Corpus 验证（补全后的词必须出现在文档中）
  → DF 过滤（DF<2 丢弃）
```

消融实验结果：

| 过滤器版本 | vs B0 | 概念数 |
|---|---|---|
| 无过滤 | 94% | 192 |
| Stage 7c（DF+corpus+BPE） | 109% | 31 |
| 三重过滤无 BPE（Phase 20） | 106% | 15 |
| **三重过滤 + BPE（Phase 25）** | **114%** | **13** |

### 2.3 概念图传播

二部图：
- chunk 节点 ↔ concept 节点
- 边权 = IDF(concept) × BM25_tf(concept, chunk)

传播算法：
```
1. 查询 → bge-m3 → 余弦 top-10 seed chunks
2. 收集 seed chunks 的所有概念
3. 沿概念边传播到共享这些概念的其他 chunk
4. 候选 chunk 得分 = Σ IDF(concept) × BM25_tf(concept, chunk)
```

### 2.4 共现关系图（可选增强）

对在 ≥2 个 chunk 中共现的概念对，用**双概念关切耦合**读取关系类型：

```
"This text discusses {cancer} and {chemotherapy}.
 The relationship between them is ___"
 → J-Lens workspace → treated(38%)
```

关系传播：seed 概念 → 沿关系边扩展概念集 → 传播到更多 chunk。在多跳推理（L4）场景下，关系扩展使 recall 从 28.6% 提升到 57.1%（翻倍）。

**关系是词典级裸关系**（cancer 和 chemotherapy 的通用关系），非语境绑定。复杂关系的精修可由用户调用 LLM API 做可选修正。

## 3. 实验

### 3.1 环境

- **模型**：Qwen2.5-7B-Instruct，4-bit NF4 量化，5.56GB VRAM
- **Lens**：neuronpedia/jacobian-lens，wikitext-103，27 层
- **嵌入**：bge-m3（1024 维 dense）
- **GPU**：NVIDIA RTX 5070 Laptop（8GB VRAM）
- **LLM judge**：DeepSeek API

### 3.2 能效比对比（核心实验）

同一个 Qwen2.5-7B-Instruct 4-bit + bge-m3，公平对比：

| 方法 | 概念提取方式 | 建图时间/chunk | VRAM | ACC |
|---|---|---|---|---|
| **J-AugRAG** | J-Lens workspace 读出 | **0.173s** | **5.76GB** | **71.4%** |
| Fast-GraphRAG | LLM generate() + JSON | ~60s | ~5.8GB | 3.6% |
| B0（纯 RAG） | 无 | 0s | — | 71.4% |

Fast-GraphRAG 在 7B 模型上几乎不可用：JSON 截断 → instructor 解析失败 → 重试 3 次 → 仍失败。J-Lens 读 workspace 不需要结构化输出。

### 3.3 排行榜定位

GraphRAG-Bench medical（ACC，GPT-4o-mini + bge-large-en-v1.5）：

| 方法 | ACC | 成本 |
|---|---|---|
| G-reasoner (#1) | 73.3% | 高（GNN 预训练 + API） |
| HippoRAG2 (#3) | 64.9% | 高（API） |
| **J-AugRAG（等效估计）** | **~64%** | **零 API** |
| LightRAG (#5) | 62.6% | 高（API） |
| RAG w/o rerank (#7) | 61.0% | 低 |
| MS-GraphRAG local (#14) | 45.2% | 极高（API） |

换算方法：用 bge-large-en-v1.5 在本地跑 B0，和 leaderboard 的 RAG w/o rerank 对比建立换算系数（0.85x）。

### 3.4 关系图效果

| 方法 | ACC | L4（多跳推理） |
|---|---|---|
| B0 | 64.3% | 28.6% |
| flat 概念图 | 71.4% | 28.6% |
| **+ 关系传播** | **75.0%** | **57.1%** |

### 3.5 成本分析

| 规模 | 传统 GraphRAG | J-AugRAG |
|---|---|---|
| 1000 篇 | ~$5-15 | $0（~3 分钟） |
| 10000 篇 | ~$50-150 | $0（~30 分钟） |
| 硬件 | 云 GPU 或 API | 单卡 8GB 消费级 GPU |

## 4. 限制

1. **概念是单 token 词**：J-Lens 读出的是词表中的单 token。多 token 概念（polycystic）只有首 token（Pol），需要 BM25 补全。
2. **关系是词典级**：双概念关切读出的是通用关系（cancer --treated--> chemotherapy），非语境绑定。
3. **28 题评估方差大**：GraphRAG-Bench 论文用 M=5 重复取平均 + 全量 2062 题。我们的 28 题单次运行 B0 ACC 波动 64-75%。完整验证待后续。
4. **ACC 评估依赖 DeepSeek**：LLM judge 有随机性。严格对比需要多次运行取平均。

## 5. 产品定位

```
┌─────────────────────────────────────────┐
│ J-AugRAG 发布版                         │
│                                         │
│ 检索层：flat 概念图 + Phase 25 过滤器   │
│   → 114% of B0, 零 API 建图             │
│                                         │
│ 关系层（可选）：共现筛选关系图           │
│   → L4 多跳推理增强                     │
│   → 裸关系，复杂精修交 API              │
├─────────────────────────────────────────┤
│ 未发布（探索中）                        │
│                                         │
│ 聚类驱动关系图 → 知识图谱雏形           │
│   → 完整 GraphRAG-Bench 验证后考虑发布  │
└─────────────────────────────────────────┘
```

## 6. 代码索引

| 组件 | 文件 |
|---|---|
| J-Lens 基础设施 | `experiments/phase10_jlens_stage1.py` |
| 概念提取 + 概念图 | `experiments/phase10_jlens_stage7c.py` |
| 三重过滤 + BM25 补全 | `experiments/phase25_filter_bpe_benchmark.py` |
| 共现关系图 | `experiments/phase28_relation_graph.py` |
| BM25 BPE 补全 | `experiments/concept_quality.py` |
| ACC 评估 | `experiments/phase26_acc_eval.py` |
| 能效比对比 | `experiments/phase26_fast_graphrag.py` |
| 排行榜换算 | `experiments/phase26b_bge_large_comparison.py` |
