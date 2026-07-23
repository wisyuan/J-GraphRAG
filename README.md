# J-GraphRAG

**用 Jacobian Lens（J-Lens）workspace 读出替代 GraphRAG 中全部 LLM `generate()` 提取步骤——零 LLM 生成成本的知识图谱构建与检索。**

```python
from jgraphrag import build_index

index = build_index(chunks)              # chunks: {id: text} 或 [text]
results = index.retrieve("your query")   # 排序后的 chunk id 列表（list[str]）
index.save("./my_index")                 # 持久化（npz + JSON）
```

建图管线：J-Lens workspace 读出（概念/关系/角色，每 chunk 2 次 forward）+ 文本侧规则实体检测（零模型）→ 图索引；检索管线：LightRAG 式 dual-level（实体/关系双路种子 + 一跳扩展）与稠密检索交错合并。建图零 API 成本（对比 GPT-4o-mini 提取管线约 1/347 成本），GraphRAG-Bench 上保持率 1.011/1.104（medical/novel，反超 LLM 提取原版）。方法依据见 `docs/j-graphrag-tech-report.md`。

## 安装

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .
bash scripts/restore_env.sh   # 链接默认模型权重到 /tmp（Qwen2.5-7B 4bit + jlens）
python -m nltk.downloader wordnet   # BPE 补全的 WordNet 候选（可选但建议）
```

## 可插拔后端

模型与存储全部走协议接口，不绑定具体模型/数据库：

```python
from jgraphrag import build_index

index = build_index(
    chunks,
    lens=MyLensProvider(),      # providers.base.LensProvider 协议：建图读出后端
    embed=MyEmbedProvider(),    # providers.base.EmbedProvider 协议：稠密嵌入
    stores=(MyVectorStore(), MyGraphStore()),  # stores.base 协议：向量/图存储
)
```

| 协议 | 默认实现 | 说明 |
|---|---|---|
| `LensProvider` | `QwenJlensLensProvider`（Qwen2.5-7B 4bit + jlens） | **唯一经端到端验证的后端**；换基座模型需拟合对应 jlens，退化为 LLM API 提取亦可实现同协议（均未验证） |
| `EmbedProvider` | `CachedBgeM3Provider`（bge-m3 + 磁盘缓存） | 查询种子与实体/关系文本嵌入 |
| `VectorStore` / `GraphStore` | `NumpyVectorStore` / `JsonGraphStore`（本地 npz/JSON） | 接口形态兼容 Qdrant/Neo4j 等外部库，外部适配器留作后续 extras |

## 环境与硬件注意

- **GPU 8GB 铁律**：同一进程只能加载一次 4bit 模型；bge-m3 与 Qwen 不同时驻留——`build_index` 内部已做编排（全部 lens 读出完成 → 释放 GPU → 再加载嵌入模型），嵌入默认走 CPU（`LINCLE_BGE_M3_DEVICE`，历史变量名）。
- 嵌入磁盘缓存按设备分命名空间（`data/.embedcache/`）——切设备会触发全量重嵌入。
- 默认模型路径可用环境变量覆盖：`JGRAPHRAG_QWEN_MODEL_PATH` / `JGRAPHRAG_JLENS_LENS_PATH` / `JGRAPHRAG_QWEN_MODEL_ID`。

## 开发

```bash
.venv/bin/python -m pytest tests/ -q   # 单元测试（CPU，mock provider，无模型）
```

## 分支说明

- `main` / `dev`：产品化代码（本仓库内容）
- `experiment`：Phase 1-53 全部实验脚本、评测链、实验数据与早期文档（研究记录，不再产品化）

已知边界（实验证伪，勿踩）：J-Lens 读不出低频专名（由文本侧检测补齐）；查询侧概念路由不能替代稠密检索；ws 几何与关系结构脱钩（三层异构是必然）。
