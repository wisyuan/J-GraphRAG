"""Phase 17: Multihop 深度梯度概念层级（A/B 对比实验）。

核心命题：J-Lens 在 workspace 层的读出随深度变化，形成天然的概念层级。
Stage 1 demo 已证实 multihop prompt 的深度梯度（L20=currency → L26=yen）。

本实验验证：普通文档是否也展现这种深度梯度？

## 两阶段 A/B 对比设计（用户指定）

### 阶段 1：纯读取（baseline）
  文档原文 → forward pass → 扫描全部 27 层 → 提取 per-layer top-k 内容词
  不加任何 concern prompt。这是文档驱动的纯净信号。

### 阶段 2：关切耦合读取
  文档 + concern prompt（"What is the main topic"）→ forward pass → 扫描全部层
  和阶段 1 对比，差异部分 = concern 驱动的概念形成。

### 关键验证问题
  1. 纯读取在 workspace 层能否看到内容词？（还是全是噪声？）
  2. 关切读取的概念词 vs 纯读取的概念词——有多少重叠？
  3. 深度梯度是否展现"抽象→具体"？
  4. 不同文档簇的深度梯度是否可区分？

如果纯读取就能在 workspace 层看到概念邻域 → 深度梯度是文档驱动的、无污染的，
可以作为概念树的层级来源。如果纯读取全是噪声、只有 concern 才能读出概念 →
概念形成依赖 prompt，需要其他方案。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase17_multihop_depth_gradient
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import (
    extract_residuals, build_concern_prompt,
)
from experiments.phase10_jlens_stage2 import _is_content_token
from experiments.phase16a_cross_domain_pos import STOP_WORDS

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── Core: depth gradient extraction ───────────────────────────────────

def _decode_layer_topk(
    logits_row: torch.Tensor,
    tokenizer,
    n_words: int = 8,
    topk_scan: int = 40,
) -> list[dict]:
    """Decode top-k content words from one layer's logits row.

    Returns list of {token, prob} for content words (filtered).
    """
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(topk_scan)

    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_WORDS
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n_words:
            break
    return results


def extract_depth_gradient(
    lens, lens_model, tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    n_words: int = 5,
    max_seq_len: int = 512,
) -> dict[int, list[dict]]:
    """Extract concept words at ALL layers in one forward pass.

    This is the core multihop primitive: a single `lens.apply(layers=all,
    positions=[-1])` gives the concept readout at every depth. The depth
    gradient (how concepts change across layers) is the concept hierarchy.

    Args:
        prompt: the full prompt string (can be plain text OR concern prompt)
        layers: which layers to read. None = all source_layers.
        n_words: max content words per layer

    Returns:
        {layer_int: [{token, prob}, ...]} for each requested layer.
    """
    if layers is None:
        layers = lens.source_layers

    lens_logits, model_logits, _ = lens.apply(
        lens_model, prompt,
        layers=layers,
        positions=[-1],
        max_seq_len=max_seq_len,
    )

    gradient = {}
    for layer in layers:
        gradient[layer] = _decode_layer_topk(
            lens_logits[layer][0], tokenizer, n_words=n_words)

    return gradient


def get_model_final_topk(
    model_logits: torch.Tensor,
    tokenizer,
    n_words: int = 5,
) -> list[dict]:
    """Decode the model's actual final-layer prediction."""
    probs = torch.softmax(model_logits[0].float(), dim=-1)
    topk = probs.topk(10)
    results = []
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n_words:
            break
    return results


# ── Two prompts: plain readout vs concern-coupled ─────────────────────

def build_plain_prompt(docs: list[str], tokenizer) -> str:
    """Plain document text — no concern prompt, no question.

    The model just processes the document. We read at the last token
    position to see what concept the workspace has formed.

    For a fair comparison with the concern prompt, we still use the chat
    template format but with a minimal user message (just the document).
    No assistant prefill — we read the residual at the document's last
    token, not at a forced prefill position.

    Rationale: this isolates "what does the model spontaneously compute
    from the document" vs "what does the concern question force it to
    compute."
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = f"Read the following text:\n\n{doc_block}"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,  # let model generate, but we read at last pos
            )
        except Exception:
            pass
    return user_msg


def build_concern_prompt_multi(docs: list[str], tokenizer) -> str:
    """Concern-coupled prompt (same as Phase 16a extract_cluster_concepts).

    Multi-doc concern prompt with assistant prefill, forcing concept
    formation at the readout position.
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List 8 one-word concepts.\n\n{doc_block}")
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


# ── Analysis ──────────────────────────────────────────────────────────

def analyze_workspace_onset(gradient: dict[int, list[dict]]) -> dict:
    """Detect the workspace onset: first layer with ≥2 content words.

    Based on the paper's §4.1: layers below workspace onset produce noise/
    fragments; workspace layers produce meaningful content words. The
    transition point is where content words first appear.

    Returns:
        {onset_layer, n_content_per_layer: {layer: n_words}, ...}
    """
    n_per_layer = {layer: len(words) for layer, words in gradient.items()}
    layers_sorted = sorted(gradient.keys())

    onset = None
    for layer in layers_sorted:
        if n_per_layer[layer] >= 2:
            onset = layer
            break

    # Also find the "stable workspace" point: first layer after which
    # content words persist (don't drop back to < 2 for 3+ consecutive layers)
    stable_onset = onset
    if onset is not None:
        consecutive_low = 0
        for layer in layers_sorted:
            if layer < onset:
                continue
            if n_per_layer[layer] >= 2:
                consecutive_low = 0
            else:
                consecutive_low += 1
                if consecutive_low >= 3 and stable_onset == onset:
                    # Found instability; the real stable onset is later
                    pass  # keep looking

    return {
        "onset_layer": onset,
        "n_content_per_layer": n_per_layer,
        "layers_sorted": layers_sorted,
    }


def compare_ab_gradients(
    plain_grad: dict[int, list[dict]],
    concern_grad: dict[int, list[dict]],
) -> dict:
    """Compare plain-readout vs concern-coupled gradients.

    For each layer:
    - overlap: how many content words appear in BOTH gradients
    - plain_only: words only in plain readout (document-driven)
    - concern_only: words only in concern readout (prompt-driven)

    This reveals which layers are document-driven vs prompt-driven.
    """
    layers = sorted(set(plain_grad.keys()) | set(concern_grad.keys()))
    per_layer = {}

    for layer in layers:
        plain_words = {w["token"].lower() for w in plain_grad.get(layer, [])}
        concern_words = {w["token"].lower() for w in concern_grad.get(layer, [])}
        overlap = plain_words & concern_words
        per_layer[layer] = {
            "overlap": sorted(overlap),
            "plain_only": sorted(plain_words - concern_words),
            "concern_only": sorted(concern_words - plain_words),
            "n_overlap": len(overlap),
            "n_plain_only": len(plain_words - concern_words),
            "n_concern_only": len(concern_words - plain_words),
        }

    # Summary: which layers are most document-driven (high plain_only)?
    doc_driven_layers = sorted(
        [(l, per_layer[l]["n_plain_only"]) for l in layers],
        key=lambda x: x[1], reverse=True)[:5]

    return {
        "per_layer": per_layer,
        "most_document_driven_layers": doc_driven_layers,
    }


# ── Main experiment ───────────────────────────────────────────────────

def run_phase17(lens, lens_model, tokenizer, doc_texts: list[str],
                max_docs: int = 300, n_clusters: int = 10):
    print("Phase 17: Multihop depth gradient — A/B comparison")
    print(f"  (A) plain readout vs (B) concern-coupled readout)")
    print(f"{'='*70}")

    n = len(doc_texts)
    layer = lens.source_layers[-1]
    all_layers = lens.source_layers  # scan all 27 layers

    # 1. L0 clustering via J-Lens residuals
    print(f"\n[1/3] Extracting J-Lens residuals ({n} docs)...")
    prompts_resid = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer,
                                    prompts_resid, layer)
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    # Take top-N largest clusters
    top_clusters = sorted(l0_valid.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(l0_valid)} clusters, analyzing top {len(top_clusters)}")

    # 2. For each cluster: extract depth gradient (plain + concern)
    print(f"\n[2/3] Extracting depth gradients (A=plain, B=concern)...")
    cluster_results = []

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]
        print(f"\n  Cluster {cid} ({len(members)} docs)")
        print(f"    sample: {docs[0][:100]}...")

        # Build both prompts
        plain_prompt = build_plain_prompt(docs, tokenizer)
        concern_prompt = build_concern_prompt_multi(docs, tokenizer)

        # Extract depth gradient (one forward pass each = 2 total per cluster)
        print(f"    [A] plain readout...", flush=True)
        plain_grad = extract_depth_gradient(
            lens, lens_model, tokenizer, plain_prompt,
            layers=all_layers, n_words=5, max_seq_len=512)

        print(f"    [B] concern readout...", flush=True)
        concern_grad = extract_depth_gradient(
            lens, lens_model, tokenizer, concern_prompt,
            layers=all_layers, n_words=5, max_seq_len=512)

        # Analyze
        plain_onset = analyze_workspace_onset(plain_grad)
        concern_onset = analyze_workspace_onset(concern_grad)
        ab_comparison = compare_ab_gradients(plain_grad, concern_grad)

        # Print layer-by-layer comparison
        print(f"\n    {'L':>3}  {'A (plain)':40} {'B (concern)':40} {'overlap':>7}")
        print(f"    {'-'*95}")
        for layer in all_layers:
            a_words = [w["token"] for w in plain_grad.get(layer, [])][:5]
            b_words = [w["token"] for w in concern_grad.get(layer, [])][:5]
            a_str = ", ".join(a_words) if a_words else "(noise)"
            b_str = ", ".join(b_words) if b_words else "(noise)"
            n_ov = ab_comparison["per_layer"][layer]["n_overlap"]
            marker = " ←" if a_words or b_words else ""
            print(f"    L{layer:>2}  {a_str:40} {b_str:40} {n_ov:>3}{marker}")

        cluster_results.append({
            "cluster_id": cid,
            "n_docs": len(members),
            "sample_doc": docs[0][:200],
            "plain_gradient": {str(l): plain_grad[l] for l in all_layers},
            "concern_gradient": {str(l): concern_grad[l] for l in all_layers},
            "plain_onset": plain_onset["onset_layer"],
            "concern_onset": concern_onset["onset_layer"],
            "ab_comparison": {
                "per_layer": {str(l): ab_comparison["per_layer"][l]
                              for l in all_layers},
                "most_document_driven_layers": ab_comparison[
                    "most_document_driven_layers"],
            },
        })

    # 3. Summary statistics
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    n_with_plain_concepts = sum(1 for c in cluster_results
                                if c["plain_onset"] is not None)
    n_with_concern_concepts = sum(1 for c in cluster_results
                                   if c["concern_onset"] is not None)
    print(f"  Clusters with plain concepts (workspace onset found): "
          f"{n_with_plain_concepts}/{len(cluster_results)}")
    print(f"  Clusters with concern concepts: "
          f"{n_with_concern_concepts}/{len(cluster_results)}")

    # Average overlap per layer (across all clusters)
    if cluster_results:
        print(f"\n  Average A/B overlap by layer:")
        for layer in all_layers:
            overlaps = [c["ab_comparison"]["per_layer"][str(layer)]["n_overlap"]
                        for c in cluster_results]
            avg_ov = np.mean(overlaps) if overlaps else 0
            print(f"    L{layer:>2}: {avg_ov:.1f} words overlap on average")

    # Save
    cand = detect_model()
    out = {
        "method": "multihop_depth_gradient_ab_comparison",
        "model": cand["name"],
        "n_docs": n,
        "n_clusters_analyzed": len(cluster_results),
        "n_total_layers": len(all_layers),
        "layers": all_layers,
        "clusters": cluster_results,
    }
    out_path = EXP / "phase17_depth_gradient.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running Phase 17 depth gradient experiment...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase17(lens, lens_model, tokenizer, doc_texts, n_clusters=8)


if __name__ == "__main__":
    main()
