# AGENTS.md — J-GraphRAG 仓库指南

> 面向 AI 编码代理与新加入者。本文档是继续开发的第一入口，包含：项目定位、当前状态、环境实操要点（含踩过的坑）、方法论约定、下一步计划。
> 项目文档与注释以中文为主（代码内注释英文）。

## 1. 项目定位

**J-GraphRAG：用 Jacobian Lens（J-Lens）workspace 读出替代 GraphRAG 中全部 LLM `generate()` 提取步骤**——零 LLM 生成成本的知识图谱构建与检索。2026-07-21 从 lincledb（Lincle 语义检索层项目）的 experiment 分支拆分独立。

**一句话主张**（已验证）：J-Lens（抽象层：概念/关系/角色/消歧，1 forward pass）+ 文本侧规则（表层：命名实体，零模型）的混合提取，可替换 GPT-4o-mini 级 LLM 提取管线，GraphRAG-Bench 上保持率 ≥0.9，建图成本 1/347，API $0。

**权威文档**：`docs/j-graphrag-complete-method.md`（§0-18 记录 Phase 10-53 全部判决）。改方法结论前先读它。

## 2. 当前状态（2026-07-21，Phase 53 完成）

**已验证**：
- 提取替换：概念 80% 准确、关系 100% 覆盖、消歧 0.80（超 bge 0.60）
- 接地：E_ws（workspace 条件向量）编码关系结构稳定超越共现 +0.088，去偏后与 bge 打平（Phase 40/48）
- 等价：q×M、W×q 与图遍历**精确一致**（Phase 41）——图检索可整体矩阵化
- 能力地图：属性→几何（0.76）、消歧→prompt（0.80）、多跳→W²（AUC 0.998）+ 张量补全（0.81）
- 替换终审：LightRAG-J **1.011/1.104**（medical/novel retained）、HippoRAG-J **0.996**（medical）/0.881（novel 临界）

**已证伪（不要再试）**：查询侧概念路由替代 bge（42）、cloze 式 prompt（16/24/42/43 四次，只产模板续词——读出必须锚定实体位置或受限答案集）、J-Lens 读低频专名（52，7B 结构性限制）、概念层级树/递归展开（16-24）、关系纯向量补全（46 P3）、ws-关系强制耦合（47 S4）、Ridge 去共线（41A）。

**关键架构事实**：ws 几何与关系结构**脱钩**——三层异构（符号/几何/关系）是必然，不是选择。集成姿势是**种子+扩展**（bge seed → 图传播补位），不是分数融合，不是纯概念路由。

## 3. 仓库结构

```
experiments/       # Phase 1-53 脚本 + 共享模块（python -m experiments.phaseXX）
jgraphrag/         # 核心包：config.py（环境变量）、embed.py（bge-m3）、llm.py（DeepSeek）
data/m6/           # 结果 JSON（入库）+ 缓存（gitignored：concept_cache/、*.npz、.embedcache）
docs/              # complete-method（权威）+ 历史版本 + embedding-fractal 研究渊源
scripts/           # restore_env.sh（模型环境一键恢复）
```

共享模块枢纽：`phase10_jlens_stage1.py`（模型/lens 加载）、`phase10_jlens_stage7c.py`（概念提取）、`phase25_filter_bpe_benchmark.py`（过滤管线+图索引）、`phase26_acc_eval.py`（DeepSeek judge）、`phase4_dig_graphragbench.py`（数据加载）、`embed_cache.py`（bge 磁盘缓存）。

## 4. 环境实操（含踩过的坑，重要）

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e .
cp .env.example .env  # 填 DeepSeek key
bash scripts/restore_env.sh   # /tmp 符号链接 + 数据集
set -a; . .env; set +a; export HF_HUB_DISABLE_XET=1
```

- **/tmp 链接会随重启失效**：`/tmp/qwen25-7b-it-weights`、`/tmp/jlens-qwen25-7b-it`、`/tmp/graphrag-bench`、`/tmp/beir-datasets`——脚本报 FileNotFoundError 时先重跑 `scripts/restore_env.sh`（或手动 ln -s）。
- **GPU 8GB 的铁律**：① 同一进程**不能加载两次 4bit 模型**（显存不释放，第二次报 "modules dispatched on CPU or disk"）——多 domain 实验要分进程跑；② **bge-m3 与 Qwen 不同时驻留**（争显存崩溃）——bge 用 CPU（`LINCLE_BGE_M3_DEVICE=cpu`，注意是 LINCLE_ 前缀）或等 Qwen 进程结束。
- **嵌入设备环境变量是 `LINCLE_BGE_M3_DEVICE`**（历史遗留名，jgraphrag/config.py 读它）。CPU 与 GPU 的嵌入缓存是**不同命名空间**——切设备会触发全量重嵌入（20k 三元组 ~45 分钟）。
- **DeepSeek judge 非确定性 ±0.05 ACC**（56 题里约 ±3 题）：边界结论要跑重复或报噪声区间。
- **模型**：Qwen2.5-7B-Instruct 4bit（~5.7GB VRAM）+ jlens（PyPI，anthropics/jacobian-lens）+ bge-m3。`.model_cache/` 当前是指向 `../lincledb/.model_cache` 的符号链接（17G）——删 lincledb 前先把缓存实体移过来。
- **数据**：GraphRAG-Bench 在 `/tmp/graphrag-bench`（medical 957 chunks / 56 题=14×4级、novel 4391 chunks / 48 题；novel 题库实有 2062 题）。排行榜存档 `data/m6/graphrag_bench_leaderboard.json`；换算系数 medical 1.005 / novel 0.885（同跑 B0 锚）。

## 5. 方法论约定（继承自 Lincle，必须遵守）

- **先定义判决指标，再动手**：每个 Phase 是可证伪的押注（bet），SUPPORTED/FALSIFIED 由预设门槛判决（如保持率 ≥0.9、AUC>0.7）。指标设计本身会被反噬（Phase 52 第一轮教训：靶子表面形式≠原文措辞）——判决前人工抽查指标合理性。
- **先 Mock 后真模型**：逻辑用 monkeypatch/合成数据自测（`--selftest` 惯例），真模型只跑判决。
- **冒烟协议**：新 prompt/新方法先 3-5 例冒烟，失败模式归类后再全量。所有脚本 `python -c "import ..."` 必须无副作用（模型只在 main 里加载）。
- **诚实记录**：失败实验同样写进文档（证伪是资产）；judge 抖动、选择偏差、循环性都要标注。
- **写回文档**：每个 Phase 完成后结果写回 `docs/j-graphrag-complete-method.md` 对应章节 + 更新本文件的"当前状态"。

## 6. 下一步计划（按优先级）

1. **更大 L4 样本确认多跳增益**（novel 或 2062 题子集）——效果故事最薄弱环节
2. **novel L4 全方法趋零排查**——疑似题目/judge 问题而非检索问题
3. **更强基线**：补 vs LightRAG/HippoRAG2 的 retrieval 级指标对比（Table 3 recall/relevance 已存档）
4. **跨模型泛化**：Qwen3/Llama 的 J-Lens 拟合 + workspace 可读性验证
5. 可选：novel ee 边换 J-Lens 读出边重测 Phase 51 novel（预期 0.881→≥0.9）；角色词 BPE 碎片过滤（WordNet 完整词验证）；实体-实体关系边规模化读出
6. 产品化候选：jgraphrag 包沉淀建图/检索 API（目前是实验脚本集合）

## 7. 与 lincledb 的关系

- lincledb `experiment` 分支 = 本项目的拆分前存档（不再更新，顶部有存档说明）；`dev` = Lincle 产品主线（语义检索层），与本仓库无关。
- M2-M5 时代旧脚本（run_m*_ab.py 等）在本仓库仅作研究记录，其数据仍在 lincledb，不能直接复跑。
