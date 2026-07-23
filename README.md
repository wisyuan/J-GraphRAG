# J-GraphRAG

**用 Jacobian Lens（J-Lens）workspace 读出替代 GraphRAG 中全部 LLM `generate()` 提取步骤——零 LLM 生成成本的知识图谱构建与检索。**

源自 Lincle 项目（[lincledb](../lincledb)，语义检索层）的 M6 研究线，2026-07-21 拆分为独立项目。

## 核心结论（Phase 10-53，详见 `docs/j-graphrag-tech-report.md`）

- **提取管线替换**：概念/关系/角色/消歧全部由单次 forward pass 的 workspace 读出完成（0.17s/chunk vs generate() 5-60s），建图成本 1/347，API 成本 $0，7B 本地模型可靠
- **混合提取架构**：J-Lens 做抽象（概念/关系/角色），文本侧规则做表层（命名实体，零模型），几何做属性，W²/张量做多跳
- **替换验证**（GraphRAG-Bench，vs 排行榜原版）：LightRAG-J 保持率 **1.011/1.104**（medical/novel），HippoRAG-J（矩阵 PPR）medical **0.996**
- **数学闭合**：概念向量（workspace 残差）与符号关系结构接地一致（边级 AUC 0.63，去偏后与 bge-m3 打平）；矩阵检索 ≡ 图遍历（精确等价）

## 仓库结构

```
experiments/     # Phase 1-53 全部实验脚本 + 共享模块（python -m experiments.phaseXX）
jgraphrag/       # 核心包：config（环境变量）、embed（bge-m3）、llm（DeepSeek）
data/m6/         # 实验数据：concept/entity 缓存、关系图、结果 JSON
docs/            # 方法文档（complete-method 为权威）+ 研究渊源（embedding-fractal）
scripts/         # restore_env.sh（模型环境一键恢复）
```

## 环境搭建

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .  # 或 uv sync
cp .env.example .env  # 填入 DeepSeek API key 等
bash scripts/restore_env.sh  # 链接模型权重到 /tmp + 准备数据集
```

模型依赖：Qwen2.5-7B-Instruct（4bit，8GB VRAM）+ Jacobian lens（anthropics/jacobian-lens）+ bge-m3。`.model_cache/` 为模型权重缓存（当前为指向 lincledb 的符号链接）。

## 复跑示例

```bash
set -a; . .env; set +a; export HF_HUB_DISABLE_XET=1
python -m experiments.phase39_two_pass_cache --domain medical   # 两步概念提取
python -m experiments.phase40_grounding_test --domain medical   # 接地检验
python -m experiments.phase50_lightrag_j --domain medical --arm ah  # LightRAG-J
```

## 关键边界（已证伪的方向）

- 查询侧概念路由不能替代稠密检索（bge 保留，Phase 42）
- J-Lens 读不出低频专名（7B 结构性限制，由文本侧检测补齐，Phase 52/53）
- 概念层级树/递归展开/分层传播（Phase 16-24）
- 关系不能纯向量补全；ws 几何与关系结构脱钩（Phase 46/47）
