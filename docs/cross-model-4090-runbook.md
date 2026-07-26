# 跨模型实验 Runbook（RTX 4090 / 24GB 执行）

> 版本：v1.0（2026-07-24，基于 experiment 分支 a5880a1）
> 执行机器：RTX 4090 台式机（24GB VRAM）。本文档是自包含交接：环境 → 数据 → 代码改动点 → Phase 57/58 设计（含预设判决）→ 已知坑。
> 方法论沿用 AGENTS.md §5：先定义判决指标再动手、先 selftest/冒烟后全量、结果写回 complete-method。

## 0. 目标与范围

- **Phase 57（跨模型泛化）**：验证 J-Lens 提取管线在非 Qwen2.5-7B 模型上成立，直接回应论文 limitations 第一条（单模型验证）。预训练 lens 全部来自 [neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens)，**无需自拟合**。
- **Phase 58（参数量 × lens 精度消融）**：gemma-3 阶梯画规模曲线；lens 精度对下游指标的影响。
- **不在本期范围**：多模态（gemma-3 视觉侧，先单独冒烟）、qwen3-14b/32b（超出复现所需）。

## 1. 机器与仓库准备

```bash
# 仓库：experiment 分支。三种方式选一：
#   a) 主机 push 后 clone（需先确认推送）；b) rsync 整目录；c) git bundle
git clone -b experiment <repo> j-graphrag && cd j-graphrag   # 或 rsync/bundle 后 git checkout experiment

# 环境
uv venv --python 3.12 .venv && uv sync && uv pip install beir
cp .env.example .env   # 填 DeepSeek key（或从主机 scp .env）
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=...    # gemma/llama 是 gated，必须先在 HF 网页接受协议
```

数据（GraphRAG-Bench，~1GB）：

```bash
# restore_env.sh 的 /tmp 模型链接部分对本机无意义（那些链接指向主机缓存），
# 只需其数据集下载逻辑；直接手动下载更干净：
python - << 'EOF'
import os
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import hf_hub_download
for f in ['Datasets/Corpus/novel.json', 'Datasets/Corpus/medical.json',
          'Datasets/Questions/medical_questions.json', 'Datasets/Questions/novel_questions.json']:
    hf_hub_download('GraphRAG-Bench/GraphRAG-Bench', f, repo_type='dataset', local_dir='/tmp/graphrag-bench')
EOF
cd /tmp/graphrag-bench && ln -sf Datasets/Corpus/medical.json medical.json && ln -sf Datasets/Corpus/novel.json novel.json && ln -sf Datasets/Questions/medical_questions.json medical_questions.json && ln -sf Datasets/Questions/novel_questions.json novel_questions.json
```

bge-m3 嵌入：24GB 下可与 LLM 同驻留 GPU。但注意 `LINCLE_BGE_M3_DEVICE` 切换会改变嵌入缓存命名空间（`data/.embedcache` 是从主机符号链接/复制的话，设备不同会触发全量重嵌入，20k 三元组 ~45 分钟）——要么连同缓存目录一起复制且保持设备一致，要么接受一次重嵌入。

## 2. 代码改动点（唯一需要碰的地方）

`experiments/phase10_jlens_stage1.py` 的 `CANDIDATES` 列表（第 48 行起）。每个候选一个 dict，`detect_model()` 选第一个本地权重+lens 都在位的。**多模型实验必须分进程跑**（同一进程加载两次模型显存不释放），建议用环境变量或参数指定而非依赖检测顺序——如果嫌改代码麻烦，最简单的做法：每次只保留当前要跑的候选放列表最前。

新增候选示例：

```python
{
    "name": "gemma-3-4b-it",
    "model_id": "google/gemma-3-4b-it",
    "local_model_dir": "/tmp/gemma3-4b-it-weights",
    "local_lens_path": "/tmp/jlens-gemma3-4b-it/gemma-3-4b-it_jacobian_lens.pt",
    "needs_4bit": True,   # 24GB 下 4b 可 fp16，但保持 4bit 与主机结果可比
},
```

- **lens 文件名模式**：`{候选name}/jlens/Salesforce-wikitext/{model_id最后一段}_jacobian_lens.pt`（`load_lens()` 会自动从 neuronpedia 下载，~433MB/个；也可手动 `hf_hub_download` 到 local_lens_path）。
- **lens 质量元数据**：同目录 `config.yaml`（拟合参数 + `final_identity_distance`）和 `{name}_convergence.csv`——Phase 58 的 lens 精度数据源。
- `LENS_CONFIG`（第 117 行）是 qwen 残留，非 qwen 模型如报错可临时硬编码对应路径。
- **gemma-3 冒烟的第一个目的**：验证 `jlens.HFLensModel` 的残差流 hook 对 gemma 架构兼容（jlens 官方支持列表含 gemma，但我们这套 4bit + 自建读出管线只在 qwen 上跑过）。

## 3. Phase 57：跨模型泛化最小复现

**押注**：J-Lens 提取管线的关键结论（可读性、概念提取、接地）在非 Qwen 架构上复现。

**模型组**（覆盖面递进）：
1. `Qwen/Qwen3-4B`——同族对照（lens 已有；候选已在 CANDIDATES）
2. `google/gemma-3-4b-it`——跨架构、gated
3. `meta-llama/Llama-3.1-8B-Instruct`——跨架构、gated（可选，前两个成功再做）

**S0 可读性冒烟**（每模型 ~10 分钟）：5 个 medical chunks，concern prompt（"What concepts does this text discuss?"）position -1 读出 top-10。
- **判决**：≥3/5 chunks 的 top-10 含文本真实概念词（人工核对）→ lens 对该模型可用，进 S1；否则记录失败模式（hook 不兼容 / lens 未对齐 / prompt 语言先验不适配），该模型止于 S0。

**S1 概念提取准确率**（每模型 ~1-2h）：完整两步提取管线（`phase10_jlens_stage7c` 配置）跑 medical 100 chunks 随机子集，人工抽查 50 个概念。
- **判决**：准确率 ≥70%（qwen2.5-7b 基准 80%，允许跨模型损耗 10 个点）。

**S2 接地 AUC**（每模型 ~1h）：E_ws 边/非边 ROC-AUC + 1000 次置换检验（Phase 40 协议，`phase40_*` 脚本改模型入口）。
- **判决**：AUC >0.55 且置换 p<0.05，且超越共现基线（qwen2.5-7b 基准 0.631/+0.085）。

**S3 单域替换**（可选，半天）：medical LightRAG-J 全管线。
- **判决**：retention ≥0.9（**必须用 Phase 54 的 rubric L4 协议**，`experiments/phase54_l4_rubric_judge.py` 的 judge；生成侧去 concisely、max_tokens ≥800）。

**写回**：每模型一个 `data/m6/phase57_{模型名}.json`；结论写回 complete-method §20。

## 4. Phase 58：参数量 × lens 精度消融

**押注**：提取质量随参数量提升（概念质量 ↔ 模型规模的验证链条）；lens 拟合精度是独立影响因子。

**主链条：Qwen 规模链**（0.8b → 4b → 7b → 27b）。qwen3.5-0.8b / qwen3.5-4b / qwen3.6-27b 三者**同为 qwen3_5 hybrid 架构**（linear attention + 每 4 层 full attention，已查 config 确认），是一条架构受控的规模链；qwen2.5-7b-it（dense）是现有生产基线点，作为链条中点参照。27b 是 neuronpedia featured 模型（lens 质量据其声明较好）。

| 档位 | 模型 | 架构 | 显存估算（4090） | 精度方案 |
|---|---|---|---|---|
| 0.8b | Qwen/Qwen3.5-0.8B | qwen3_5 hybrid | ~2.5GB | bf16 |
| 4b | Qwen/Qwen3.5-4B | qwen3_5 hybrid | ~9.5GB | bf16（勿 4bit，见坑清单） |
| 7b | Qwen/Qwen2.5-7B-Instruct | dense（基线） | ~5.9GB | NF4 4bit（现有结果直接复用） |
| 27b | Qwen/Qwen3.6-27B | qwen3_5 hybrid | **~19-21GB（紧张）** | NF4 4bit |

27b 显存明细：权重 NF4 ~12GB + 嵌入/lm_head bf16 ~5.1GB（vocab 248320 且不 tie，是大头）+ KV/激活/lens ~2.5GB。KV 压力小（64 层仅 16 层 full attention）。跳过 vision tower（见坑清单加载注意）。

**指标**（每档跑 Phase 57 的 S0+S1+S2）：可读性通过率、概念准确率、接地 AUC → 对参数量画曲线；各 lens 的 `final_identity_distance`（config.yaml）作协变量记录——概念质量差异要能分离"模型规模"与"lens 质量"两个因子。
- **判决**：三指标随参数量单调不降，且 4b 档概念准确率 ≥70% → 规模效应成立；若 4b 即达 7b 水平（饱和），则论文部署故事更强（小模型够用）；27b 显著超 7b → 高端增益成立，"规模天花板"主张上修。
- **混淆声明**：7b 点是 qwen2.5 dense 架构，与其余三点（qwen3_5 hybrid）不同——若 7b 点偏离链条趋势，用 gemma-3 对照链（下）判定是规模效应还是架构效应。

**对照链：gemma-3 阶梯**（同架构全系有 lens）：270m / 1b / 4b / 12b（27b 4bit ~16GB 可选）。注意 CANDIDATES 注释：base 模型不能做概念抽象（Stage 2 已证），**小尺寸也优先选 -it 变体**。

**lens 精度刻度盘**（两条数据源，先做①）：
1. **现成 lens 的质量分层**：各 lens `config.yaml` 的 `final_identity_distance`（gemma-3-4b-it = 0.960，越小越好）+ `convergence.csv`——把"拟合收敛度"当自变量，与下游指标做相关。
2. **自拟合低精度 lens**（仅 ≤4b，bf16 上 4090 可行）：neuronpedia 的 `fit_lens.py`（config.yaml 里有完整命令行），`n_prompts` 取 100/300/1000 三档（拟合语料 Salesforce/wikitext-103）造精度梯度，重跑 S1/S2。
- **判决**：lens 精度档与概念准确率/接地 AUC 正相关 → "lens 质量影响下游"成立；无关 → 管线对 lens 噪声鲁棒（也是好结论）。

## 5. 已知坑（4090 特化）

- **gated 模型**：gemma/llama 必须先网页接受协议 + `HF_TOKEN`，否则 403。
- **jlens 架构兼容**：S0 就是验证它；失败先查 hook 是否挂到残差流（对比 qwen 的读出层分布）。
- **transformers 版本**：pyproject 锁 `>=5.5`，gemma-3 需要新版本支持，uv sync 后 `python -c "import transformers; print(transformers.__version__)"` 确认。
- **bitsandbytes 4bit 在 4090** 正常工作；但 4b 以下模型建议 fp16（`needs_4bit: False`），量化噪声对小模型读出影响未知。
- **qwen3_5 hybrid 架构（0.8b/4b/27b 全链）**：linear attention 层可能需要 `mamba-ssm`/`fla` 专用 kernel（config 有 `mamba_ssm_dtype`），装不上时查 transformers 5.x 是否有原生回退实现；`AutoModelForCausalLM` 对 `Qwen3_5ForConditionalGeneration`（多模态壳）可能报错或带上 vision tower——必要时改用 text_config 单独实例化或跳过 vision 权重（省 ~1GB）。jlens hook 兼容性由 qwen3.5-4b 的 S0 冒烟首次验证（下载最小、失败成本最低）。
- **27b 额外风险**：嵌入不 tie（vocab 248320，bf16 嵌入+lm_head ~5.1GB）是显存大头；若 OOM，优先压缩 activation（短 chunk）而非量化嵌入（unembed 精度是 jlens 读出的命根）。4090 无硬件 FP4（NVFP4 是 Blackwell 特性），NF4 存储+bf16 计算即等价路径，对 jlens 透明。
- **顺带的战略发现**：qwen3.5/3.6 全系原生多模态（带 vision tower）——未来多模态实验不必只走 gemma-3，qwen3.5-4b 是更轻的候选（lens 在文本侧拟合，视觉 token 可读性同样待验证）。
- **DeepSeek judge 噪声** ±0.03–0.05：边界结论跑重复。
- **生成侧必须改**：L4 相关实验生成 prompt 去掉 "concisely"、max_tokens ≥800（Phase 54 S2 证据：L4 残余低分是生成忠实度问题）。
- **不要并行两个模型进程**：24GB 也经不起两个 4bit 模型 + lens 同时驻留。
