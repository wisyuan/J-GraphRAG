"""Phase 9 — XLM-RoBERTa MLM head readout + vec2vec translation.

验证用户提出的方案：用 base XLM-RoBERTa-large（有 MLM head，未经 contrastive
tuning）做概念读出，并通过 vec2vec 桥接到 bge-m3 空间。

两步门控：
  测试 1: XLM-R MLM head 直接读出 vs bge-m3 负对照（复用 §12.4 reverse_lookup）
          门控：XLM-R 读出的 LLM judge 准确率 > bge-m3 + 显著
  测试 2: vec2vec Procrustes 翻译读出（仅门控通过后跑）

可读性评估：LLM judge（DeepSeek）——给 LLM 看簇文档 + 读出的 token，问是否准确。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase9_xlmr_vec2vec --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols
from experiments.partition import partition_hdbscan
from experiments.xlmr_readout import XLMRReadout
from experiments.phase6_semantic_readout import build_tokenizer_embed_space, reverse_lookup
from jgraphrag.llm import DeepSeekProvider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── LLM judge ───────────────────────────────────────────────────────────

def judge_readout(concept_tokens: list[str], sample_docs: list[str], llm) -> dict:
    """Ask DeepSeek: do these concept tokens accurately describe this cluster?

    Returns {"accurate": bool, "confidence": float, "reason": str}.
    Uses the same DeepSeekProvider + JSON-parse pattern as keyword_cache.
    """
    tokens_str = ", ".join(concept_tokens[:5])
    docs_str = "\n".join(f"  - {d[:200]}" for d in sample_docs[:3])
    prompt = (
        "You are evaluating whether a set of concept keywords accurately describes "
        "a cluster of documents.\n\n"
        f"Concept keywords: [{tokens_str}]\n\n"
        f"Document samples from the cluster:\n{docs_str}\n\n"
        "Do the keywords accurately capture what the documents are about? "
        'Respond with ONLY a JSON object: {"accurate": true/false, "confidence": 0.0-1.0, "reason": "..."}'
    )
    msg = llm.complete(prompt, 150)
    if msg.is_error:
        return {"accurate": False, "confidence": 0.0, "reason": f"llm_error: {msg.error_message}"}
    try:
        result = json.loads(msg.content.strip())
        if isinstance(result, dict):
            return {
                "accurate": bool(result.get("accurate", False)),
                "confidence": float(result.get("confidence", 0)),
                "reason": str(result.get("reason", ""))[:200],
            }
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: look for true/false in text
    text = msg.content.lower()
    accurate = "true" in text and "false" not in text.split("true")[0][-20:]
    return {"accurate": accurate, "confidence": 0.5, "reason": "fallback parse"}


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_phase9(repo_path: str, max_files: int = 50, n_clusters: int = 10,
               do_vec2vec: bool = True):
    print("Phase 9: XLM-R MLM head readout + vec2vec")
    print(f"{'='*60}")

    # ── 加载语料（同 Phase 6，保证可比）──
    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
    print(f"  {len(fine_texts)} FineRecords from {len(files)} files")

    # ── bge-m3 嵌入（负对照读出用）──
    print(f"  embedding bge-m3...", flush=True)
    embed = CachedBgeM3Provider()
    bge_vecs = np.asarray(embed.embed(fine_texts), dtype=np.float64)
    print(f"    bge-m3: {bge_vecs.shape}")

    # ── HDBSCAN 簇（在 bge-m3 空间，用于 vec2vec 翻译 bge 质心）──
    print(f"  clustering (bge-m3 space)...", flush=True)
    cluster_members = partition_hdbscan(bge_vecs.tolist())  # {label: [indices]}, noise excluded
    top_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"    {len(cluster_members)} clusters, top-{n_clusters} selected")

    # ── bge-m3 负对照读出（复用 phase6 reverse_lookup）──
    print(f"\n{'='*60}")
    print("Negative control: bge-m3 reverse lookup (§12.4, known to fail)")
    tok_space = build_tokenizer_embed_space(embed, top_n_tokens=3000)
    bge_readout = {}
    for cid, members in top_clusters:
        centroid = bge_vecs[members].mean(axis=0)
        tokens = reverse_lookup(centroid, tok_space, top_k=5)
        sample_docs = [fine_texts[m][:200] for m in members[:3]]
        bge_readout[cid] = {
            "n_members": len(members),
            "tokens": [t for t, _ in tokens],
            "sample_docs": sample_docs,
        }
        print(f"    Cluster {cid} ({len(members)} mem): {[t for t,_ in tokens]}")

    # ── XLM-R hidden states + 簇质心 ──
    print(f"\n{'='*60}")
    print("TEST 1: XLM-RoBERTa-large MLM head direct readout")
    xlmr = XLMRReadout()
    print(f"  computing XLM-R hidden states (mean pool)...", flush=True)
    xlmr_vecs_mean = xlmr.get_hidden_states(fine_texts, pooling="mean")
    print(f"    XLM-R mean: {xlmr_vecs_mean.shape}")
    print(f"  computing XLM-R hidden states (CLS pool)...", flush=True)
    xlmr_vecs_cls = xlmr.get_hidden_states(fine_texts, pooling="cls")
    print(f"    XLM-R CLS: {xlmr_vecs_cls.shape}")

    # XLM-R 簇质心（用 mean pool 版本——和 bge-m3 pooling 对齐）
    xlmr_readout = {"mean": {}, "cls": {}}
    for pool_name, xlmr_pool in [("mean", xlmr_vecs_mean), ("cls", xlmr_vecs_cls)]:
        print(f"\n  XLM-R readout ({pool_name} pool):")
        for cid, members in top_clusters:
            centroid = xlmr_pool[members].mean(axis=0)
            tokens = xlmr.readout_centroid(centroid, top_k=5)
            # Clean token strings (strip XLM-R <special> prefixes)
            clean_tokens = [_clean_token(t) for t, _ in tokens]
            sample_docs = [fine_texts[m][:200] for m in members[:3]]
            xlmr_readout[pool_name][cid] = {
                "n_members": len(members),
                "tokens": clean_tokens,
                "raw_tokens": [(t, round(s, 2)) for t, s in tokens],
                "sample_docs": sample_docs,
            }
            print(f"    Cluster {cid} ({pool_name}): {clean_tokens}")

    # ── LLM judge 对比 ──
    print(f"\n{'='*60}")
    print("LLM judge: XLM-R vs bge-m3 readout accuracy")
    llm = DeepSeekProvider()
    judge_results = {"xlmr_mean": {}, "xlmr_cls": {}, "bge": {}}
    for cid, members in top_clusters:
        sample_docs = [fine_texts[m][:200] for m in members[:3]]
        for method, tokens in [
            ("xlmr_mean", xlmr_readout["mean"][cid]["tokens"]),
            ("xlmr_cls", xlmr_readout["cls"][cid]["tokens"]),
            ("bge", bge_readout[cid]["tokens"]),
        ]:
            j = judge_readout(tokens, sample_docs, llm)
            judge_results[method][cid] = j
            print(f"    Cluster {cid} {method:10s}: accurate={j['accurate']} conf={j['confidence']:.2f}")

    # 汇总
    acc = {m: sum(1 for c in judge_results[m].values() if c["accurate"]) / len(judge_results[m])
           for m in judge_results}
    print(f"\n  Accuracy: XLM-R(mean)={acc['xlmr_mean']:.2f}  XLM-R(CLS)={acc['xlmr_cls']:.2f}  bge-m3={acc['bge']:.2f}")

    # 门控判定
    best_xlmr = "xlmr_mean" if acc["xlmr_mean"] >= acc["xlmr_cls"] else "xlmr_cls"
    best_xlmr_acc = acc[best_xlmr]
    gate_passed = best_xlmr_acc > acc["bge"] + 0.15  # 需要 15% 以上提升才算显著
    print(f"\n  GATE: best XLM-R ({best_xlmr}) = {best_xlmr_acc:.2f} vs bge-m3 = {acc['bge']:.2f}")
    print(f"  GATE {'PASSED' if gate_passed else 'FAILED'} (need +15% margin)")

    result = {
        "n_fine_records": len(fine_texts),
        "n_clusters": len(top_clusters),
        "readout": {
            "xlmr_mean": xlmr_readout["mean"],
            "xlmr_cls": xlmr_readout["cls"],
            "bge_control": bge_readout,
        },
        "judge": judge_results,
        "accuracy": acc,
        "gate_passed": gate_passed,
        "best_xlmr_pool": best_xlmr,
    }

    # ── 测试 2: vec2vec（门控通过后）──
    if gate_passed and do_vec2vec:
        print(f"\n{'='*60}")
        print("TEST 2: vec2vec Procrustes translation (bge-m3 → XLM-R)")
        # 用 best pooling 的 XLM-R 向量学映射
        xlmr_best = xlmr_vecs_mean if best_xlmr == "xlmr_mean" else xlmr_vecs_cls
        print(f"  learning Procrustes map (bge-m3 → XLM-R {best_xlmr})...", flush=True)
        M = xlmr.learn_vec2vec(bge_vecs.astype(np.float32), xlmr_best)
        residual = xlmr.translation_residual(bge_vecs.astype(np.float32), xlmr_best, M)
        print(f"    translation residual: {residual:.4f} (lower=better; ~0.3 typical cross-model)")

        # 翻译 bge-m3 簇质心 → XLM-R → MLM head 读出
        vec2vec_readout = {}
        judge_v2v = {}
        for cid, members in top_clusters:
            bge_centroid = bge_vecs[members].mean(axis=0).astype(np.float32)
            translated = xlmr.translate(bge_centroid, M)
            tokens = xlmr.readout_centroid(translated, top_k=5)
            clean_tokens = [_clean_token(t) for t, _ in tokens]
            sample_docs = [fine_texts[m][:200] for m in members[:3]]
            vec2vec_readout[cid] = {
                "n_members": len(members),
                "tokens": clean_tokens,
                "sample_docs": sample_docs,
            }
            j = judge_readout(clean_tokens, sample_docs, llm)
            judge_v2v[cid] = j
            print(f"    Cluster {cid}: {clean_tokens} accurate={j['accurate']}")

        acc_v2v = sum(1 for c in judge_v2v.values() if c["accurate"]) / len(judge_v2v)
        print(f"\n  vec2vec accuracy: {acc_v2v:.2f} (vs XLM-R direct {best_xlmr_acc:.2f}, bge {acc['bge']:.2f})")
        result["vec2vec"] = {
            "residual": residual,
            "readout": vec2vec_readout,
            "judge": judge_v2v,
            "accuracy": acc_v2v,
        }
    elif not gate_passed:
        print(f"\n  (skipping vec2vec — gate failed)")
        result["vec2vec"] = None

    # ── 保存 ──
    out_path = EXP / "phase9_xlmr_vec2vec.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")

    # ── 人工审查引导 ──
    print(f"\n{'='*60}")
    print("=== MANUAL REVIEW GUIDE ===")
    for cid, _ in top_clusters[:5]:
        print(f"\n  Cluster {cid}:")
        print(f"    XLM-R(mean): {xlmr_readout['mean'][cid]['tokens']}")
        print(f"    XLM-R(CLS):  {xlmr_readout['cls'][cid]['tokens']}")
        print(f"    bge-m3:      {bge_readout[cid]['tokens']}")
        print(f"    sample:      {xlmr_readout['mean'][cid]['sample_docs'][0][:80]}...")

    return result


def _clean_token(tok: str) -> str:
    """Strip XLM-R token prefixes/suffixes for readability."""
    # XLM-R uses SentencePiece; tokens may have ▁ prefix (space marker) or trailing ▁
    t = tok.replace("▁", " ").strip()
    return t


def main():
    ap = argparse.ArgumentParser(description="Phase 9: XLM-R readout + vec2vec")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--no-vec2vec", action="store_true", help="skip vec2vec test")
    args = ap.parse_args()

    run_phase9(args.repo, args.max_files, args.n_clusters, do_vec2vec=not args.no_vec2vec)


if __name__ == "__main__":
    main()
