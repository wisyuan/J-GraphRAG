"""Phase 9b — per-position MLM head readout (bypassing pooling mismatch).

Phase 9 showed pooled-vector (mean/CLS) readout fails: MLM head reads code
punctuation (`_`, `$`, `:`) not concepts, because pooled vectors are out-of-
distribution for the MLM head (designed for single token positions).

This module tests the native MLM-head use case: run the FULL model (encoder +
lm_head) on every token position of every document in a cluster, aggregate the
250k-dim logits across the cluster, and read out the most consistently-predicted
tokens. No pooling — every position is read in its native form.

Two aggregation strategies (data decides):
  - logit_sum: Σ logits across all positions in cluster → top tokens by total logit
  - top1_vote: each position votes its argmax token → top tokens by vote count

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.xlmr_per_position --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols
from experiments.partition import partition_hdbscan
from experiments.xlmr_readout import XLMRReadout
from experiments.phase9_xlmr_vec2vec import judge_readout, _clean_token
from jgraphrag.llm import DeepSeekProvider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"

# XLM-R special tokens to exclude from readout (they dominate trivially)
_SPECIAL_TOKENS = {"<s>", "</s>", "<pad>", "<unk>", "<mask>", "<eos>", "<cls>", "<sep>"}


def per_position_readout(texts: list[str], xlmr: XLMRReadout,
                         max_length: int = 256, batch_size: int = 4,
                         strategy: str = "logit_sum",
                         top_k: int = 10) -> list[tuple[str, float]]:
    """Run full model on every token position, aggregate logits across all texts.

    strategy:
      "logit_sum"  — sum logits across positions, top-k by total logit
      "top1_vote"  — each position votes argmax token, count votes
      "top5_vote"  — each position votes top-5 tokens (weighted by rank), count

    Returns [(token_str, score), ...] top_k tokens (special tokens excluded).
    """
    import torch
    xlmr._ensure_model()
    tok = xlmr._tokenizer
    model = xlmr._model

    if strategy == "logit_sum":
        # Accumulate full 250k-dim logit vector — too big to hold for many texts.
        # Instead accumulate in fp32 on CPU, batch-processed.
        logit_acc = torch.zeros(tok.vocab_size, dtype=torch.float32)
        total_positions = 0
    elif strategy in ("top1_vote", "top5_vote"):
        vote_counter: Counter = Counter()

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        encoded = tok(batch, return_tensors="pt", max_length=max_length,
                      truncation=True, padding=True)
        input_ids = encoded["input_ids"].to(xlmr._device)
        attention_mask = encoded["attention_mask"].to(xlmr._device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, seq, 250k) fp16

        if strategy == "logit_sum":
            # Sum logits over real (non-pad) positions, across batch
            mask = attention_mask.float()  # (B, seq)
            # logits: (B, seq, V) — sum over B and seq where mask=1
            logits_f = logits.float().cpu()
            for b in range(logits_f.shape[0]):
                seq_len = int(mask[b].sum())
                logit_acc += logits_f[b, :seq_len].sum(0)
                total_positions += seq_len
        elif strategy == "top1_vote":
            preds = logits.argmax(-1).cpu().numpy()  # (B, seq)
            mask_np = attention_mask.cpu().numpy()
            for b in range(preds.shape[0]):
                for p in range(int(mask_np[b].sum())):
                    vote_counter[int(preds[b, p])] += 1
        elif strategy == "top5_vote":
            top5 = logits.topk(5, dim=-1).indices.cpu().numpy()  # (B, seq, 5)
            mask_np = attention_mask.cpu().numpy()
            for b in range(top5.shape[0]):
                for p in range(int(mask_np[b].sum())):
                    for rank, tid in enumerate(top5[b, p]):
                        vote_counter[int(tid)] += (5 - rank)  # rank-weighted

    # Decode + filter special tokens
    if strategy == "logit_sum":
        top_ids = logit_acc.topk(top_k * 5).indices.tolist()  # over-fetch then filter
        results = []
        for tid in top_ids:
            t = tok.convert_ids_to_tokens(tid)
            if t in _SPECIAL_TOKENS or t.strip() == "":
                continue
            results.append((t, float(logit_acc[tid] / max(total_positions, 1))))
            if len(results) >= top_k:
                break
        return results
    else:
        results = []
        for tid, count in vote_counter.most_common(top_k * 5):
            t = tok.convert_ids_to_tokens(tid)
            if t in _SPECIAL_TOKENS or t.strip() == "":
                continue
            results.append((t, float(count)))
            if len(results) >= top_k:
                break
        return results


def run_phase9b(repo_path: str, max_files: int = 50, n_clusters: int = 10,
                max_docs_per_cluster: int = 15):
    print("Phase 9b: per-position MLM head readout (bypassing pooling)")
    print(f"{'='*60}")

    # ── 加载语料 ──
    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
    print(f"  {len(fine_texts)} FineRecords")

    # ── bge-m3 簇（复用 Phase 9 的簇定义，保证可比）──
    embed = CachedBgeM3Provider()
    bge_vecs = np.asarray(embed.embed(fine_texts), dtype=np.float64)
    cluster_members = partition_hdbscan(bge_vecs.tolist())
    top_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(cluster_members)} clusters, top-{n_clusters} selected")

    xlmr = XLMRReadout()
    llm = DeepSeekProvider()

    # ── 三种策略对比 ──
    strategies = ["logit_sum", "top1_vote", "top5_vote"]
    all_readouts = {}
    all_judgments = {}

    for strat in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strat}")
        all_readouts[strat] = {}
        all_judgments[strat] = {}

        for cid, members in top_clusters:
            # 限制每簇文档数（per-position 计算量大）
            cluster_texts = [fine_texts[m] for m in members[:max_docs_per_cluster]]
            sample_docs = [fine_texts[m][:200] for m in members[:3]]

            tokens = per_position_readout(cluster_texts, xlmr, strategy=strat, top_k=8)
            clean_tokens = [_clean_token(t) for t, _ in tokens]
            all_readouts[strat][cid] = {
                "n_members": len(members),
                "n_docs_used": len(cluster_texts),
                "tokens": clean_tokens,
                "raw": [(t, round(s, 4)) for t, s in tokens],
                "sample_docs": sample_docs,
            }
            print(f"  Cluster {cid} ({len(members)} mem): {clean_tokens}")

            # LLM judge
            j = judge_readout(clean_tokens, sample_docs, llm)
            all_judgments[strat][cid] = j

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("=== ACCURACY SUMMARY ===")
    acc = {}
    for strat in strategies:
        a = sum(1 for c in all_judgments[strat].values() if c["accurate"]) / len(all_judgments[strat])
        acc[strat] = a
        print(f"  {strat:12s}: {a:.2f}")

    # 对比 Phase 9 的 pooled 结果（bge-m3 = 0, XLM-R pooled = 0）
    print(f"\n  (Phase 9 reference: XLM-R pooled=0.00, bge-m3=0.00)")
    best_strat = max(acc, key=acc.get)
    print(f"\n  Best per-position strategy: {best_strat} = {acc[best_strat]:.2f}")
    print(f"  Gate (per-position > pooled + 0.15): {'PASSED' if acc[best_strat] > 0.15 else 'FAILED'}")

    # ── 人工审查引导 ──
    print(f"\n{'='*60}")
    print("=== MANUAL REVIEW ===")
    for cid, _ in top_clusters[:5]:
        print(f"\n  Cluster {cid}:")
        for strat in strategies:
            print(f"    {strat:12s}: {all_readouts[strat][cid]['tokens']}")
        print(f"    sample: {all_readouts[strat][cid]['sample_docs'][0][:80]}...")

    # ── 保存 ──
    result = {
        "n_fine_records": len(fine_texts),
        "n_clusters": len(top_clusters),
        "max_docs_per_cluster": max_docs_per_cluster,
        "strategies": {
            s: {"readout": all_readouts[s], "judge": all_judgments[s], "accuracy": acc[s]}
            for s in strategies
        },
    }
    out_path = EXP / "phase9b_per_position.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Phase 9b: per-position MLM readout")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--max-docs-per-cluster", type=int, default=15)
    args = ap.parse_args()
    run_phase9b(args.repo, args.max_files, args.n_clusters, args.max_docs_per_cluster)


if __name__ == "__main__":
    main()
