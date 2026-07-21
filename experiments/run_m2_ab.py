"""M2.8 runner — end-to-end real A/B on Pride & Prejudice + pi (bet#1b + #1a).

This is the experiment that decides bet#1: does concern-fused retrieval
(route C: score_ctx + score_sem, boost-not-gate) separate cosine-similarity
from concern-fit better than pure semantic ANN?

It uses REAL bge-m3 embeddings (1024-dim) + REAL DeepSeek concern inference.
The ConcernFusion math is reimplemented in Python (it's ~50 lines of cosine +
weighted fusion — faster than a PyO3 bridge for a one-shot experiment).

Output: HSS/LHS separation verdict + nDCG comparison, printed + saved to
experiments/m2/results/.

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.run_m2_ab --domain novel --top-k 10
    python -m experiments.run_m2_ab --domain code  --top-k 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from jgraphrag.llm import DeepSeekProvider

REPO = Path(__file__).resolve().parents[1]  # lincledb/
EXP = REPO / "experiments" / "m2"

# --- ConcernFusion (route C), reimplemented in Python (mirrors Rust orchestrator.rs) ---
CONCERN_WEIGHTS = {"universal": 0.6, "domain_specific": 0.8, "personalized": 1.0}
CTX_SIGNIFICANCE = 0.3
ALPHA = 1.0  # context boost weight
BETA = 0.6   # semantic baseline weight


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_candidate(query_vec, payload_ctx_vecs, payload_sem_vec) -> tuple[float, float, bool]:
    """Return (score_ctx, score_sem, concern_matched)."""
    # score_ctx = weighted max over context_vecs
    score_ctx = 0.0
    for ctx_vec, source in payload_ctx_vecs:
        w = CONCERN_WEIGHTS.get(source, 0.6)
        score_ctx = max(score_ctx, w * cosine(query_vec, ctx_vec))
    score_sem = cosine(query_vec, payload_sem_vec)
    concern_matched = score_ctx >= CTX_SIGNIFICANCE
    return score_ctx, score_sem, concern_matched


def fuse(score_ctx, score_sem, concern_matched) -> float:
    if concern_matched:
        return ALPHA * score_ctx + BETA * score_sem
    return BETA * score_sem


def rank_concern(candidates) -> list:
    """Rank by concern-fused score (desc). candidates = [(id, score_ctx, score_sem, matched)]"""
    return sorted(candidates, key=lambda c: fuse(c[1], c[2], c[3]), reverse=True)


def rank_baseline(candidates) -> list:
    """Rank by pure semantic (desc)."""
    return sorted(candidates, key=lambda c: c[2], reverse=True)


# --- Metrics ---
def ndcg_at_k(ranked_ids, ground_truth, k):
    k = min(k, len(ranked_ids))
    topk = ranked_ids[:k]
    dcg = sum((1.0 if rid in ground_truth else 0.0) / math.log2(i + 2) for i, rid in enumerate(topk))
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked_ids, ground_truth, k):
    topk = ranked_ids[:k]
    hits = sum(1 for rid in topk if rid in ground_truth)
    return hits / k if k > 0 else 0.0


# --- HSS/LHS Separation ---
def classify(cosine_val, is_fit, threshold=0.5):
    high = cosine_val >= threshold
    if high and is_fit: return "TP"
    if high and not is_fit: return "HSS"
    if not high and is_fit: return "LHS"
    return "TN"


def generate_doc2query(llm, name: str, content: str) -> list[str]:
    """Use DeepSeek to generate record-specific concern questions (real doc2query).

    Reads the record's content and asks: 'what questions would this content
    answer?' These questions are then embedded as context vectors — and because
    they're content-specific (not template fill-in), their embeddings actually
    differentiate records under bge-m3.
    """
    prompt = (
        f"You are a doc2query generator. Read the following text and generate "
        f"3-5 specific questions that a reader might ask which this text would answer. "
        f"The questions should reflect the UNIQUE content of this text, not generic questions.\n\n"
        f"Record name: {name}\n"
        f"Content (first 1500 chars):\n{content}\n\n"
        f'Respond with ONLY a JSON array of question strings: ["q1", "q2", ...]'
    )
    msg = llm.complete(prompt, 256)
    if msg.is_error:
        return []
    # Parse JSON array.
    import json as _json
    try:
        qs = _json.loads(msg.content)
        if isinstance(qs, list):
            return [str(q)[:200] for q in qs if q][:5]
    except _json.JSONDecodeError:
        pass
    # Fallback: extract quoted strings.
    import re as _re
    matches = _re.findall(r'"([^"]{10,200})"', msg.content)
    return matches[:5]


def run_experiment(embed, llm, corpus, trials, top_k, domain_label):
    """Run the full A/B for one domain. corpus = [(id, name, text)]. trials = [{query, ground_truth_files}]."""
    print(f"\n{'='*60}")
    print(f"  M2.8 A/B — {domain_label} ({len(corpus)} records, {len(trials)} trials)")
    print(f"{'='*60}")

    # 1. Embed all corpus records → RecordPayloads (semantic_vec + context_vecs).
    print(f"  embedding {len(corpus)} records with bge-m3...")
    names = [c[1] for c in corpus]
    texts_for_sem = [c[2] for c in corpus]
    sem_vecs = embed.embed(texts_for_sem)

    # Context vectors: REAL doc2query via DeepSeek (not template fill-in).
    # Template questions ("What problem does chapter_N solve?") produce nearly-
    # identical embeddings under bge-m3 → can't differentiate records → fusion
    # degenerates to pure semantic. Instead, LLM reads each record's content and
    # generates record-specific questions a user might ask that this record answers.
    print(f"  generating context queries via DeepSeek (doc2query)...")
    payloads = []
    for i, (rid, name, text) in enumerate(corpus):
        ctx_vecs = []
        if llm is not None:
            questions = generate_doc2query(llm, name, text[:1500])
            if questions:
                ctx_vecs_raw = embed.embed(questions)
                ctx_vecs = [(v, "universal") for v in ctx_vecs_raw]
        # Fallback: if LLM unavailable or failed, at least embed the name+content snippet
        # (more differentiated than template, less good than real doc2query).
        if not ctx_vecs:
            ctx_vecs_raw = embed.embed([f"{name}: {text[:200]}"])
            ctx_vecs = [(v, "universal") for v in ctx_vecs_raw]
        payloads.append((rid, ctx_vecs, sem_vecs[i]))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(corpus)} records processed")

    # 2. For each trial, embed query, score candidates, rank both conditions.
    cf_ndcgs, bl_ndcgs = [], []
    cf_precs, bl_precs = [], []
    hss_cf_ranks, hss_bl_ranks = [], []
    lhs_cf_ranks, lhs_bl_ranks = [], []

    for ti, trial in enumerate(trials):
        query = trial["query"]
        gt = set(trial["ground_truth_files"])
        q_vec = embed.embed([query])[0]

        candidates = []
        for (rid, ctx_vecs, sem_vec) in payloads:
            sc, ss, matched = score_candidate(q_vec, ctx_vecs, sem_vec)
            candidates.append((rid, sc, ss, matched))

        cf_ranked = rank_concern(candidates)
        bl_ranked = rank_baseline(candidates)
        cf_ids = [c[0] for c in cf_ranked]
        bl_ids = [c[0] for c in bl_ranked]

        cf_ndcgs.append(ndcg_at_k(cf_ids, gt, top_k))
        bl_ndcgs.append(ndcg_at_k(bl_ids, gt, top_k))
        cf_precs.append(precision_at_k(cf_ids, gt, top_k))
        bl_precs.append(precision_at_k(bl_ids, gt, top_k))

        # HSS/LHS classification (by semantic cosine to query).
        for rank_idx, (rid, sc, ss, matched) in enumerate(cf_ranked):
            rank_cf = rank_idx + 1
            is_fit = rid in gt
            cell = classify(ss, is_fit)
            if cell == "HSS": hss_cf_ranks.append(rank_cf)
            elif cell == "LHS": lhs_cf_ranks.append(rank_cf)
        for rank_idx, (rid, sc, ss, matched) in enumerate(bl_ranked):
            rank_bl = rank_idx + 1
            is_fit = rid in gt
            cell = classify(ss, is_fit)
            if cell == "HSS": hss_bl_ranks.append(rank_bl)
            elif cell == "LHS": lhs_bl_ranks.append(rank_bl)

        if (ti + 1) % 20 == 0:
            print(f"    {ti+1}/{len(trials)} trials done")

    # 3. Aggregate.
    n = len(trials)
    cf_ndcg = sum(cf_ndcgs) / n
    bl_ndcg = sum(bl_ndcgs) / n
    cf_prec = sum(cf_precs) / n
    bl_prec = sum(bl_precs) / n
    ndcg_lift = cf_ndcg - bl_ndcg

    mean = lambda v: sum(v) / len(v) if v else 0.0
    hss_cf_mean = mean(hss_cf_ranks)
    hss_bl_mean = mean(hss_bl_ranks)
    lhs_cf_mean = mean(lhs_cf_ranks)
    lhs_bl_mean = mean(lhs_bl_ranks)

    hss_pushed = hss_cf_mean > hss_bl_mean
    lhs_pulled = lhs_cf_mean < lhs_bl_mean
    enough = len(hss_cf_ranks) >= 5 and len(lhs_cf_ranks) >= 5
    if not enough:
        sep_verdict = "Inconclusive (too few HSS/LHS samples)"
    elif hss_pushed and lhs_pulled:
        sep_verdict = "SUPPORTED (HSS pushed down + LHS pulled up)"
    else:
        sep_verdict = "FALSIFIED (no separation)"

    ndcg_verdict = ("Supported" if ndcg_lift >= 0.05 else
                    "Falsified" if ndcg_lift <= 0 else "Inconclusive")

    result = {
        "domain": domain_label,
        "n_trials": n, "n_records": len(corpus), "top_k": top_k,
        "concern_fused": {"ndcg": cf_ndcg, "precision": cf_prec},
        "baseline": {"ndcg": bl_ndcg, "precision": bl_prec},
        "ndcg_lift": ndcg_lift,
        "ndcg_verdict": ndcg_verdict,
        "separation": {
            "hss": {"cf_mean_rank": hss_cf_mean, "bl_mean_rank": hss_bl_mean, "count": len(hss_cf_ranks)},
            "lhs": {"cf_mean_rank": lhs_cf_mean, "bl_mean_rank": lhs_bl_mean, "count": len(lhs_cf_ranks)},
            "hss_pushed_down": hss_pushed, "lhs_pulled_up": lhs_pulled,
            "verdict": sep_verdict,
        },
    }

    print(f"\n  --- RESULTS ({domain_label}) ---")
    print(f"  ConcernFused:  nDCG@{top_k}={cf_ndcg:.4f}  P@{top_k}={cf_prec:.4f}")
    print(f"  Baseline:      nDCG@{top_k}={bl_ndcg:.4f}  P@{top_k}={bl_prec:.4f}")
    print(f"  nDCG lift: {ndcg_lift:+.4f}  → {ndcg_verdict}")
    print(f"  HSS: concern rank {hss_cf_mean:.1f} vs baseline {hss_bl_mean:.1f} (n={len(hss_cf_ranks)}) {'↓ pushed' if hss_pushed else '✗'}")
    print(f"  LHS: concern rank {lhs_cf_mean:.1f} vs baseline {lhs_bl_mean:.1f} (n={len(lhs_cf_ranks)}) {'↑ pulled' if lhs_pulled else '✗'}")
    print(f"  ★ Separation verdict: {sep_verdict}")
    return result


def load_novel_corpus() -> list:
    """Load novel chapters as (id, name, text) tuples."""
    from experiments.load_novel import chunk_novel
    from jgraphrag.config import NOVEL_PATH
    text = Path(NOVEL_PATH).read_text(encoding="utf-8", errors="ignore")
    chapters = chunk_novel(text)
    return [(f"chapter_{ch['id']}", ch['title'], ch['text'][:2000]) for ch in chapters]


def load_code_corpus() -> list:
    """Load pi source files as (id, name, text) tuples (sample, capped)."""
    from jgraphrag.config import PI_REPO_PATH
    repo = Path(PI_REPO_PATH)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:200]
    corpus = []
    for p in files:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")[:2000]
        except OSError:
            continue
        corpus.append((rel, rel.split("/")[-1], text))
    return corpus


def main():
    ap = argparse.ArgumentParser(description="M2.8 real A/B runner")
    ap.add_argument("--domain", choices=("novel", "code", "both"), default="novel")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-trials", type=int, default=50, help="cap trials for speed")
    args = ap.parse_args()

    embed = BgeM3Provider()
    llm = DeepSeekProvider()
    results = {}

    if args.domain in ("novel", "both"):
        trials = json.loads((EXP / "novel_trials_corrected.json").read_text())
        trials = trials[:args.max_trials]
        corpus = load_novel_corpus()
        results["novel"] = run_experiment(embed, llm, corpus, trials, args.top_k, "Novel (Pride & Prejudice, bet#1b)")

    if args.domain in ("code", "both"):
        trials = json.loads((EXP / "code_trials.json").read_text())
        trials = trials[:args.max_trials]
        corpus = load_code_corpus()
        results["code"] = run_experiment(embed, llm, corpus, trials, args.top_k, "Code (pi, bet#1a)")

    # Save results.
    out_dir = EXP / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"m2_ab_{args.domain}_k{args.top_k}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")

    # Final bet#1 verdict.
    print(f"\n{'='*60}")
    print("  BET #1 VERDICT")
    print(f"{'='*60}")
    for dom, r in results.items():
        print(f"  {dom}: {r['separation']['verdict']}")


if __name__ == "__main__":
    main()
