"""Phase 10 Stage 1 — J-Lens 基础设施验证（Qwen3-4B + 预拟合 lens）。

目标：在小模型（4B 参数，单卡 8GB）上跑通 Anthropic J-Lens 工具链，验证
「家用电脑 + 一块普通显卡 + 小模型」能否复现 J-Space 全局工作空间的 token
读出能力——这是用零 LLM-API token 实现 GraphRAG 式概念提取的前提。

本脚本只做 Stage 1（基础设施验证），不涉及概念提取质量评估（那是 Stage 2）：
  1. 加载 Qwen/Qwen3-4B（4-bit 量化，fits 8GB VRAM）
  2. 加载 neuronpedia 预拟合 lens（wikitext-103 上拟合）
  3. 用 jlens.from_hf 包装模型
  4. 跑 paper 的 multi-hop 例子，在多个 source layer 读出 lens logits
  5. 对比 lens logits vs model final logits 的 top-k token —— 验证 J_l transport
     确实把中间层残差搬到了词表空间

成功标准（Stage 1 门控）：
  - 脚本跑通无 OOM
  - lens 在某个 layer 读出的 top-1 token 和 model final top-1 token 语义相关
    （不要求完全一致——lens 读出的是中间层已形成的概念，可能比 final 更早收敛）
  - 这证明工具链可用，可以进 Stage 2（概念提取质量）

为什么用 Qwen3-4B 而非 gemma-3-4b-it：
  - gemma 全系列 gated（需 HF token + license accept），当前环境无 token
  - Qwen3 系列完全开放
  - neuronpedia 对 Qwen/Qwen3-4B 拟合的 lens identity_distance=0.39（比 gemma
    的 0.96 更低——J 更接近恒等，residual stream 读出更干净）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Candidate models, in preference order (higher param = better J-Space, but
# needs 4-bit for 8GB VRAM). Auto-detection picks the first whose local weights
# + lens both exist on disk; falls back to HF repo id.
CANDIDATES = [
    # Instruction-tuned first: Stage 2 proved base models can't do concept
    # abstraction on code (they read out code tokens, not concepts). The IT
    # variant understands "summarize the shared concept" and is the only path
    # to real concept extraction. 4-bit fits 8GB VRAM (~5.9GB total).
    {
        "name": "qwen2.5-7b-it",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_model_dir": "/tmp/qwen25-7b-it-weights",
        "local_lens_path": "/tmp/jlens-qwen25-7b-it/Qwen2.5-7B-Instruct_jacobian_lens.pt",
        "needs_4bit": True,
    },
    {
        "name": "qwen3-4b",
        "model_id": "Qwen/Qwen3-4B",
        "local_model_dir": "/tmp/qwen3-4b-weights",
        "local_lens_path": "/tmp/jlens-qwen/Qwen3-4B_jacobian_lens.pt",
        "needs_4bit": True,
    },
    {
        "name": "qwen3-1.7b",
        "model_id": "Qwen/Qwen3-1.7B",
        "local_model_dir": "/tmp/qwen3-1.7b-weights",
        "local_lens_path": "/tmp/jlens-qwen-1.7b/Qwen3-1.7B_jacobian_lens.pt",
        "needs_4bit": False,
    },
]
LENS_REPO = "neuronpedia/jacobian-lens"


def _model_dir_complete(d: str) -> bool:
    """A model dir is complete if config.json + index exist AND the total size
    of all safetensors shards matches the index's metadata.total_size (within
    1% to tolerate rounding). This catches partial downloads."""
    p = Path(d)
    if not (p / "config.json").exists():
        return False
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        return False
    import json
    meta = json.loads(idx.read_text())
    expected_total = meta.get("metadata", {}).get("total_size")
    if not expected_total:
        return False
    actual_total = sum(s.stat().st_size for s in p.glob("*.safetensors"))
    # require ≥99% of expected total (the index total_size is sum of raw bytes;
    # safetensors files include a small header overhead, so be generous)
    return actual_total >= expected_total * 0.99


def detect_model() -> dict:
    """Pick the best candidate whose local weights + lens are both available."""
    for c in CANDIDATES:
        if (_model_dir_complete(c["local_model_dir"])
                and Path(c["local_lens_path"]).exists()):
            return c
    # fallback: first candidate (will trigger HF download)
    return CANDIDATES[0]


# Module-level defaults (used by Stage 2 import). Resolve lazily so importing
# this module doesn't require the weights to exist yet.
def _resolved():
    c = detect_model()
    return c["model_id"], c["local_model_dir"], c["local_lens_path"], c["needs_4bit"]


MODEL_ID, LOCAL_MODEL_DIR, LOCAL_LENS_PATH, NEEDS_4BIT = _resolved()
LENS_CONFIG = f"qwen3-{4 if '4b' in MODEL_ID else '1.7b'}/jlens/Salesforce-wikitext/config.yaml"


def load_model(model_src: str, use_4bit: bool):
    """Load model in 4-bit NF4 (for >4B models) or fp16/bf16 (for ≤2B).

    4-bit fits 8GB VRAM: ~2.5GB weights + ~1GB activations + ~0.5GB J matrices
    + ~1GB overhead. lm_head stays fp16 (not quantized) so unembed is clean.

    fp16 path (1.7B): ~4GB weights, fits with ~4GB headroom for activations.

    J-Lens compatibility: jlens.HFLensModel.forward() hooks each layer's output
    residual stream. 4-bit only changes weight *storage* (dequantized on-the-fly
    during matmul); the residual stream output is still fp16/bf32. The J_l
    transport and lm_head unembed are unaffected.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {model_src} ({'4-bit NF4' if use_4bit else 'bf16'})...", flush=True)
    # If model_src is a local directory, force local_files_only to avoid
    # network validation (transformers 5.x re-checks even for cached files)
    import os
    is_local = os.path.isdir(model_src)
    tok_kwargs = dict(local_files_only=True) if is_local else {}
    tokenizer = AutoTokenizer.from_pretrained(model_src, **tok_kwargs)
    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if is_local:
        kwargs["local_files_only"] = True
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
    model.eval()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  loaded. VRAM: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB",
          flush=True)
    return model, tokenizer


# Back-compat alias (Stage 2 imports this name)
def load_model_4bit(model_id: str):
    return load_model(model_id, use_4bit=True)


def load_lens(lens_path: str | None = None):
    """Load the pre-fitted Jacobian lens for Qwen3-4B.

    Resolution order: explicit lens_path → LOCAL_LENS_PATH → download from HF.
    """
    import jlens
    candidates = [lens_path, LOCAL_LENS_PATH]
    for c in candidates:
        if c and Path(c).exists():
            print(f"  loading lens from {c}", flush=True)
            return jlens.JacobianLens.load(c)
    print(f"  downloading lens from {LENS_REPO}", flush=True)
    # Try each candidate's lens file
    for c in CANDIDATES:
        try:
            return jlens.JacobianLens.from_pretrained(
                LENS_REPO,
                filename=f"{c['name']}/jlens/Salesforce-wikitext/"
                         f"{c['model_id'].split('/')[-1]}_jacobian_lens.pt",
            )
        except Exception:
            continue
    raise RuntimeError("Could not download any lens from neuronpedia")


def topk_tokens(logits_row, tokenizer, k=10):
    """Convert one row of vocab logits → list of (token_str, prob) tuples."""
    import torch
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(k)
    return [
        (tokenizer.decode([int(idx)]).strip(), float(p))
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist())
    ]


def run_demo(model, tokenizer, lens, lens_model):
    """Run the paper's multi-hop example + a concept-rich prompt.

    For each prompt, read out the last position at a spread of layers (early,
    mid, late) and compare lens logits vs model final logits. The point: J-Lens
    should reveal *intermediate* concepts that have already formed before the
    final layer.
    """
    import torch

    prompts = [
        # Paper's multi-hop: capital of Japan → currency of boot-shaped country
        # (two-hop: Japan→Tokyo already given; boot→Italy→Euro is the hop)
        (
            "Fact: The capital of Japan is Tokyo.\n"
            "Fact: The currency used in the country shaped like a boot is"
        ),
        # Concept-rich: should activate "password"/"hash"/"security" concepts
        "The authentication module stores user passwords using bcrypt hashing"
        " with a salt to prevent rainbow table attacks. Common vulnerabilities",
    ]

    # Pick a spread of layers: early, mid, late (lens has all source_layers)
    n_layers = lens_model.n_layers
    sample_layers = sorted({
        lens.source_layers[0],                          # earliest
        lens.source_layers[len(lens.source_layers) // 4],  # early-mid
        lens.source_layers[len(lens.source_layers) // 2],  # mid
        lens.source_layers[3 * len(lens.source_layers) // 4],  # mid-late
        lens.source_layers[-1],                          # latest (≠ final)
    })
    print(f"  model has {n_layers} layers; lens fitted at "
          f"{len(lens.source_layers)} source layers; sampling: {sample_layers}",
          flush=True)

    results = {}
    for pi, prompt in enumerate(prompts):
        print(f"\n{'='*70}")
        print(f"Prompt {pi}: {prompt[:80]}...")
        print(f"{'='*70}")

        # positions=[-1] = read out at the last token (where the model is
        # "thinking about" the next token — this is where concepts are richest)
        lens_logits, model_logits, input_ids = lens.apply(
            lens_model, prompt,
            layers=sample_layers,
            positions=[-1],
            max_seq_len=128,
        )

        # Model's own final-layer prediction at last position
        model_topk = topk_tokens(model_logits[0], tokenizer, k=5)
        print(f"\n  MODEL FINAL (layer {n_layers-1}) top-5:")
        for tok, p in model_topk:
            print(f"    {p*100:5.1f}%  {tok!r}")

        layer_readouts = {}
        for layer in sample_layers:
            topk = topk_tokens(lens_logits[layer][0], tokenizer, k=5)
            print(f"\n  LENS @ layer {layer:2d} top-5:")
            for tok, p in topk:
                print(f"    {p*100:5.1f}%  {tok!r}")
            layer_readouts[str(layer)] = [{"token": t, "prob": p} for t, p in topk]

        results[f"prompt_{pi}"] = {
            "prompt": prompt,
            "model_final": [{"token": t, "prob": p} for t, p in model_topk],
            "lens_by_layer": layer_readouts,
        }

    return results


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 1: J-Lens infra demo")
    ap.add_argument("--lens-path", default=None,
                    help="local path to lens .pt (downloads if absent)")
    ap.add_argument("--model-id", default=None,
                    help="model id/path (default: auto-detect local weights)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="force fp16/bf16 (for ≤2B models that fit without quant)")
    ap.add_argument("--out", default=str(EXP / "phase10_stage1_demo.json"))
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Resolve model source + precision
    cand = detect_model()
    if args.model_id:
        model_src = args.model_id
        use_4bit = not args.no_4bit
    else:
        model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
        use_4bit = cand["needs_4bit"] and not args.no_4bit

    print("Phase 10 Stage 1: J-Lens infrastructure verification")
    print(f"{'='*70}")
    print(f"  model: {model_src} ({'4-bit NF4' if use_4bit else 'bf16'})")
    print(f"  candidate: {cand['name']}")

    print(f"\n[1/3] Loading lens...")
    lens = load_lens(args.lens_path or cand["local_lens_path"])
    print(f"  {lens}")

    print(f"\n[2/3] Loading model...")
    model, tokenizer = load_model(model_src, use_4bit=use_4bit)

    print(f"\n[3/3] Wrapping with jlens.from_hf + running demo...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(f"  {lens_model}")

    results = run_demo(model, tokenizer, lens, lens_model)

    out_path = Path(args.out)
    results["_meta"] = {"model_src": model_src, "use_4bit": use_4bit, "candidate": cand["name"]}
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"\nStage 1 gate: check that lens logits produce coherent tokens at")
    print(f"intermediate layers (not garbage) — that confirms the toolchain works")
    print(f"and the residual stream is readable via J-Lens on a small model.")


if __name__ == "__main__":
    main()
