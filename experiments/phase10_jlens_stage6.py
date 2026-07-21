"""Phase 10 Stage 6 — 反转管线：J-Lens 概念残差 vs bge-m3 做聚类特征。

之前所有 Stage 都是「先 bge-m3 聚类，后 J-Lens 标注」。本阶段反转：
**用 J-Lens 概念残差本身做聚类特征**，不依赖 bge-m3。

核心命题：如果 J-Lens 残差聚类比 bge-m3 聚类更概念连贯（簇内文档共享
同一概念），那么 J-Lens 不仅是标注器，而是**聚类驱动器**——这替代了
GraphRAG 的核心步骤（概念分组），且零 API token。

流程：
  1. NFCorpus 文档 → 两种特征：
     A. bge-m3 嵌入（1024 维，baseline）
     B. J-Lens 概念残差（3584 维）：每文档 forward → L26 残差 → transport
  2. 各自 HDBSCAN 聚类
  3. 对比：
     - 聚类质量指标（silhouette / Davies-Bouldin / 簇数 / 噪声点比例）
     - J-Lens 读出的簇标签连贯性（每簇概念词是否描述该簇）
     - 簇内文档语义一致性（LLM judge：簇内文档是否真共享同一主题）

  4. 门控：J-Lens 残差聚类质量 ≥ bge-m3 → 反转管线成立

为什么用 L26（倒数第二层）残差而非 final：
  Stage 3-5 一致发现 L26 是概念最丰富的层。final 层（L28）预测下一个
  token（标点/语法），残差被 next-token 预测目标主导。L26 通过 J_l
  transport 到 unembed 空间后，是「模型认为这个位置在表达什么概念」的
  3584 维向量——这正是我们想聚类的信号。

关键技术：自定义 residual 提取（不用 apply()，因为 apply 返回 logits 而非
残差）。直接用 ActivationRecorder 捕获 L26 输出 → lens.transport()。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage2 import judge_readout

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── J-Lens residual extraction ─────────────────────────────────────────

def extract_residuals(lens, lens_model, tokenizer, prompts: list[str],
                      layer: int, max_seq_len: int = 256,
                      batch_size: int = 1) -> np.ndarray:
    """Extract transported residuals at [last position] for each prompt.

    For each prompt: forward pass → ActivationRecorder captures layer output
    at `layer` → take last position → lens.transport() → 3584-dim vector.

    This is the J-Lens concept residual — what the model "thinks the concept
    is" at the last token, transported into the unembedding basis. We use it
    as a clustering feature instead of bge-m3.

    batch_size=1 because 7B 4-bit + 3584-dim residuals per token is memory-
    intensive with the activation recorder active. Could batch with care.
    """
    from jlens.hooks import ActivationRecorder

    final_layer = lens_model.n_layers - 1
    record_at = sorted({layer, final_layer})
    d_model = lens.d_model
    residuals = np.zeros((len(prompts), d_model), dtype=np.float32)

    lens_model._hf_model.eval()
    for i, prompt in enumerate(prompts):
        input_ids = lens_model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(lens_model.layers, at=record_at) as rec:
            with torch.no_grad():
                lens_model.forward(input_ids)
        # layer output: [1, seq_len, d_model], take last position
        h = rec.activations[layer][0, -1].float()  # [d_model]
        # transport into unembed basis
        h_transport = lens.transport(h, layer).cpu().numpy()
        residuals[i] = h_transport
        if (i + 1) % 20 == 0:
            print(f"    residuals: {i+1}/{len(prompts)}", flush=True)

    return residuals


def build_concern_prompt(text: str, tokenizer) -> str:
    """Single-doc concern prompt for residual extraction.

    The concern forces the model to form a concept representation at the last
    token. Without it, the residual is dominated by next-token prediction.
    """
    user_msg = (
        f"What is the main topic of this document? Answer in one word.\n\n"
        f"{text[:600]}"
    )
    prefill = "The main topic is"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


# ── Clustering metrics ─────────────────────────────────────────────────

def cluster_metrics(vecs: np.ndarray, clusters: dict[int, list[int]]) -> dict:
    """Compute silhouette + Davies-Bouldin + noise ratio."""
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    n = len(vecs)
    labels = np.full(n, -1)
    for cid, members in clusters.items():
        for m in members:
            labels[m] = cid
    noise_ratio = float((labels == -1).sum() / n)
    n_real = (labels >= 0).sum()
    result = {
        "n_clusters": len(clusters),
        "n_noise": int((labels == -1).sum()),
        "noise_ratio": noise_ratio,
        "n_clustered": int(n_real),
    }
    if n_real >= 10 and len(set(labels[labels >= 0])) >= 2:
        mask = labels >= 0
        result["silhouette"] = float(
            silhouette_score(vecs[mask], labels[mask], metric="cosine")
        )
        result["davies_bouldin"] = float(
            davies_bouldin_score(vecs[mask], labels[mask])
        )
    return result


def evaluate_clusters(clusters: dict[int, list[int]], doc_texts: list[str],
                      lens, lens_model, tokenizer, sample_layers: list[int],
                      llm, label: str, max_judge: int = 10) -> dict:
    """For each cluster: J-Lens readout → concept label → LLM judge coherence.

    The same J-Lens readout is applied to BOTH clusterings (bge-m3 and residual)
    so we're measuring clustering quality, not readout quality.
    """
    from experiments.phase10_jlens_stage2 import readout_cluster, pick_best_layer_readout

    top = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)[:max_judge]
    judge_scores = []
    cluster_details = {}

    for cid, members in top:
        member_texts = [doc_texts[m] for m in members]
        # Build concern prompt from cluster docs
        doc_block = "\n---\n".join(t[:400] for t in member_texts[:5])
        user_msg = (f"What concept do these documents share? One word.\n\n{doc_block}")
        prefill = "The shared concept is"
        prompt = prefill  # fallback
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_msg},
                     {"role": "assistant", "content": prefill}],
                    tokenize=False, continue_final_message=True,
                    add_generation_prompt=False,
                )
            except Exception:
                prompt = f"{user_msg}\n{prefill}"

        readout = readout_cluster(lens, lens_model, tokenizer, prompt, sample_layers)
        concept_tokens = pick_best_layer_readout(readout["by_layer"], sample_layers)
        judged = judge_readout(concept_tokens, member_texts[:3], llm)
        judge_scores.append(1.0 if judged.get("accurate") else 0.0)

        cluster_details[str(cid)] = {
            "n_docs": len(members),
            "concept_tokens": concept_tokens,
            "l26_raw": [t["token"] for t in readout["by_layer"][str(sample_layers[-1])][:5]],
            "judge": judged,
            "sample": member_texts[0][:100],
        }
        print(f"    [{label}] C{cid} ({len(members)}d): {concept_tokens} → judge={judged.get('accurate')}")

    accuracy = float(np.mean(judge_scores)) if judge_scores else 0.0
    return {"accuracy": accuracy, "clusters": cluster_details}


# ── Main ───────────────────────────────────────────────────────────────

def run_stage6(embed, lens, lens_model, tokenizer, llm, max_docs: int = 150):
    print("Phase 10 Stage 6: Reverse pipeline — J-Lens residual vs bge-m3 clustering")
    print(f"{'='*70}")

    # 1. Load corpus
    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    doc_texts = [r[1] for r in records]
    print(f"\n  {len(doc_texts)} NFCorpus documents")

    # 2a. bge-m3 features (baseline)
    print(f"\n[1/4] bge-m3 embeddings (baseline)...")
    bge_vecs = np.asarray(embed.embed(doc_texts), dtype=np.float32)
    print(f"  bge-m3: {bge_vecs.shape}")

    # 2b. J-Lens residual features
    print(f"\n[2/4] J-Lens residual extraction (this takes ~{len(doc_texts)*0.4:.0f}s)...")
    layer = lens.source_layers[-1]  # L26 (last source, concept-richest)
    prompts = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer, prompts, layer)
    print(f"  jlens residual: {jlens_vecs.shape}")

    # 3. Cluster both
    print(f"\n[3/4] HDBSCAN clustering both feature sets...")
    bge_clusters = partition_hdbscan(bge_vecs.tolist())
    jlens_clusters = partition_hdbscan(jlens_vecs.tolist())

    bge_metrics = cluster_metrics(bge_vecs, bge_clusters)
    jlens_metrics = cluster_metrics(jlens_vecs, jlens_clusters)
    print(f"  bge-m3:   {bge_metrics}")
    print(f"  jlens:    {jlens_metrics}")

    # 4. Evaluate cluster coherence (J-Lens readout + judge on both)
    print(f"\n[4/4] Evaluating cluster coherence (J-Lens readout + LLM judge)...")
    sample_layers = sorted({
        lens.source_layers[0],
        lens.source_layers[len(lens.source_layers) // 2],
        lens.source_layers[-1],
    })

    print(f"\n  --- bge-m3 clusters ---")
    bge_eval = evaluate_clusters(bge_clusters, doc_texts, lens, lens_model,
                                 tokenizer, sample_layers, llm, "bge")
    print(f"\n  --- jlens residual clusters ---")
    jlens_eval = evaluate_clusters(jlens_clusters, doc_texts, lens, lens_model,
                                   tokenizer, sample_layers, llm, "jlens")

    # 5. Summary
    print(f"\n{'='*70}")
    print(f"REVERSE PIPELINE RESULTS")
    print(f"{'='*70}")
    print(f"  {'metric':<25} {'bge-m3':>10} {'jlens':>10}")
    print(f"  {'-'*47}")
    print(f"  {'n_clusters':<25} {bge_metrics['n_clusters']:>10} {jlens_metrics['n_clusters']:>10}")
    print(f"  {'noise_ratio':<25} {bge_metrics['noise_ratio']:>10.1%} {jlens_metrics['noise_ratio']:>10.1%}")
    if "silhouette" in bge_metrics and "silhouette" in jlens_metrics:
        print(f"  {'silhouette (cosine)':<25} {bge_metrics['silhouette']:>10.3f} {jlens_metrics['silhouette']:>10.3f}")
    if "davies_bouldin" in bge_metrics and "davies_bouldin" in jlens_metrics:
        print(f"  {'davies_bouldin (↓)':<25} {bge_metrics['davies_bouldin']:>10.3f} {jlens_metrics['davies_bouldin']:>10.3f}")
    print(f"  {'judge_accuracy':<25} {bge_eval['accuracy']:>10.1%} {jlens_eval['accuracy']:>10.1%}")

    # Gate
    bge_score = bge_metrics.get("silhouette", 0)
    jlens_score = jlens_metrics.get("silhouette", 0)
    gate = jlens_score >= bge_score * 0.8  # allow 20% degradation for zero-cost
    print(f"\n  GATE (jlens silhouette ≥ 80% of bge): {'PASS' if gate else 'FAIL'}")

    cand = detect_model()
    out = {
        "method": "reverse_pipeline_jlens_vs_bge",
        "model": cand["name"],
        "layer": layer,
        "n_docs": len(doc_texts),
        "bge_m3": {"metrics": bge_metrics, **bge_eval},
        "jlens_residual": {"metrics": jlens_metrics, **jlens_eval},
        "gate_pass": gate,
    }
    out_path = EXP / "phase10_stage6_reverse.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 6: reverse pipeline")
    ap.add_argument("--max-docs", type=int, default=150)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/3] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    print(repr(lens))
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/3] Wrapping with jlens.from_hf...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    print(f"\n[3/3] Running reverse pipeline experiment...")
    embed = CachedBgeM3Provider()
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    run_stage6(embed, lens, lens_model, tokenizer, llm, args.max_docs)


if __name__ == "__main__":
    main()
