# AGENTS.md — J-GraphRAG（dev/main 产品分支）

> 面向 AI 编码代理与新加入者。本分支是**产品化代码**；Phase 1-53 实验脚本、评测链、实验数据、早期研究文档都在 `experiment` 分支，本分支不保留。
> 项目文档与注释以中文为主（代码内注释英文）。

## 1. 项目定位

**J-GraphRAG：用 Jacobian Lens（J-Lens）workspace 读出替代 GraphRAG 中全部 LLM `generate()` 提取步骤**——零 LLM 生成成本的知识图谱构建与检索。本包是 Phase 53 终审方案（LightRAG-J，GraphRAG-Bench 保持率 1.011/1.104）的产品化沉淀。

**方法权威文档**：`docs/j-graphrag-complete-method.md`（实验判决全记录，在 experiment 分支持续更新）与 `docs/j-graphrag-tech-report.md`。改算法行为前先读它们。

## 2. 包结构

```
jgraphrag/
├── __init__.py            # 公共 API re-export（build_index/GraphIndex/协议/默认实现）
├── config.py              # 环境变量（仅默认后端的连接参数）
├── pipeline.py            # build_index 高层编排 + 显存编排 + save/load
├── index.py               # GraphIndex：merge_entities + 实体入图 + 嵌入 + 检索入口
├── retrieve.py            # dual-level 检索（lightrag_retrieve/merge_topk/interleave_rankings）
├── providers/
│   ├── base.py            # LensProvider / EmbedProvider 协议 + ChunkExtraction
│   ├── qwen_jlens.py      # 默认 LensProvider（Qwen2.5-7B 4bit + jlens，唯一验证后端）
│   └── bge_m3.py          # 默认 EmbedProvider（BgeM3Provider + CachedBgeM3Provider 磁盘缓存）
├── stores/
│   ├── base.py            # VectorStore / GraphStore 协议
│   └── local.py           # 默认实现（NumpyVectorStore / JsonGraphStore）
└── extract/
    ├── concepts.py        # Pass 1 概念提取（concern prompt + depth gradient + fallback）
    ├── roles.py           # Pass 2 角色扩展 + ws/wu 向量（反转 prefill 1 forward）
    ├── filter.py          # 过滤规则与全部常量（PREFILL_WORDS/STOP_WORDS/POS 表等）
    ├── entities.py        # 文本侧规则实体检测（纯 CPU 零模型）
    └── relations.py       # 关系 prompt/读出 + 关系对筛选（嵌入注入）
tests/                     # pytest（CPU + mock，无模型无网络）
```

## 3. 架构纪律（必须遵守）

- **一切可插拔**：检索/索引代码只面向 `providers/base.py` 与 `stores/base.py` 的协议编程，不得 import 具体后端（qwen/bge 只在 providers/ 内部）。外部向量库/图库适配器走同协议，本期不接外部依赖。
- **import 无副作用**：所有模块 import 时不碰磁盘/模型/GPU；torch/transformers/jlens/FlagEmbedding 一律函数内懒导入。tests/ 有子进程用例把守。
- **已验证参数不可改**：MAX 聚合（SUM 已证退化）、一跳衰减 0.5、种子 top-20、merge cos≥0.95、DF≥2、MIN_COOCC=2、ee/ec 边上限 20000、prompt 原文。这些数字背后是 Phase 判决，改动需要先回 experiment 分支重验。
- **"糙但已验证"的行为原样保留**：如 `_stem("types")=="typ"`、`classify_concept_pos` 的 RB 死分支、ee/ec 边不同的归一化基准——不要"顺手修"。

## 4. 环境与硬件

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e .
bash scripts/restore_env.sh   # /tmp 模型符号链接（重启后需重跑）
python -m nltk.downloader wordnet   # BPE 补全候选（可选但建议）
.venv/bin/python -m pytest tests/ -q
```

- **GPU 8GB 铁律**：同进程不能加载两次 4bit 模型；bge-m3 与 Qwen 不同时驻留（嵌入默认 CPU，变量是历史名 `LINCLE_BGE_M3_DEVICE`）。`build_index` 内部已编排：lens 读出全完成 → 释放 GPU → 再加载嵌入。
- 嵌入缓存按设备分命名空间（`data/.embedcache/`），切设备触发全量重嵌入。
- 默认模型路径：`JGRAPHRAG_QWEN_MODEL_PATH`（/tmp/qwen25-7b-it-weights）、`JGRAPHRAG_JLENS_LENS_PATH`（/tmp/jlens-qwen25-7b-it）。

## 5. 验证状态与边界

- **已验证**：仅默认后端组合（Qwen2.5-7B + jlens 建图、bge-m3 嵌入、本地 store）在 GraphRAG-Bench 双域 retained。其他 LensProvider/EmbedProvider/外部 store 实现协议即可用，但未验证。
- **已证伪（不要再试）**：查询侧概念路由替代稠密检索、cloze 式 prompt 读出、J-Lens 读低频专名、概念层级树、关系纯向量补全、Ridge 去共线。详见 complete-method。
- 跨模型泛化（Qwen3/Llama 的 jlens 拟合）是 roadmap 未验证项。

## 6. 与 experiment 分支的关系

- 实验分支 = 研究记录与新方法验证场：Phase 1-53 脚本、`data/m6` 结果、评测链（DeepSeek judge）、早期文档。
- 新算法/新参数必须先在 experiment 分支用可证伪的 Phase 流程验证（先定义判决指标、先 mock 后真模型、3-5 例冒烟再全量），通过后按本文件 §3 的纪律收编进产品分支。
- 进行中的嵌入解耦实验（去 bge，Qwen 作稠密编码器）若成功：新增一个 EmbedProvider 实现 + 替换检索种子/naive 臂后端即可，建图代码不动（bge 四处用途备忘见技术报告与计划记录）。
