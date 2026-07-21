"""M3.5f: Bet#1+Bet#2 coupled experiment — cluster value AFTER concern fusion.

Previous M3.5a-e tested cluster against pure cosine — but cosine is an oracle
that already ranks correctly. The real question: does cluster provide value
ON TOP OF concern fusion (which introduces HSS/LHS ambiguity)?

Conditions:
  A (flat + concern):    concern fusion ranking → flat top-K
  B (cluster + concern): concern fusion ranking → cluster GraphRAG re-organization

Both conditions use concern fusion (doc2query context vectors) as the base
scoring, THEN cluster organizes the results. This tests whether cluster's
grouping/graph provides incremental value when cosine already has ambiguity.

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.run_m3_coupled_ab --top-k 10 --max-trials 50
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
from experiments.run_m3_cluster_ab import (
    build_cluster_graph,
    cluster_hdbscan,
    cosine,
    graphrag_score,
    simulate_flat_agent,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "m3"


def concern_fusion_score(
    query_vec: list[float],
    sem_vec: list[float],
    ctx_vecs: list[list[float]],
    alpha: float = 1.0,
    beta: float = 0.6,
    significance: float = 0.3,
) -> float:
    """Route-C concern fusion: score_ctx (max over context vecs) + score_sem.
    boost not gate — if score_ctx < significance, fall back to pure semantic.
    """
    score_ctx = max((cosine(query_vec, cv) for cv in ctx_vecs), default=0.0)
    score_sem = cosine(query_vec, sem_vec)
    if score_ctx >= significance:
        return alpha * score_ctx + beta * score_sem
    return beta * score_sem


def generate_doc2query(llm, name: str, content: str) -> list[str]:
    """DeepSeek doc2query — generate content-specific concern questions."""
    prompt = (
        f"You are a doc2query generator. Read the following text and generate "
        f"3-5 specific questions that a reader might ask which this text would answer.\n\n"
        f"Record name: {name}\nContent (first 1200 chars):\n{content[:1200]}\n\n"
        f'Respond with ONLY a JSON array: ["q1", "q2", ...]'
    )
    msg = llm.complete(prompt, 256)
    if msg.is_error:
        return []
    try:
        qs = json.loads(msg.content)
        if isinstance(qs, list):
            return [str(q)[:200] for q in qs if q][:5]
    except json.JSONDecodeError:
        import re
        return re.findall(r'"([^"]{10,200})"', msg.content)[:5]
    return []


def run_coupled_experiment(embed, llm, corpus, trials, top_k):
    n_records = len(corpus)
    print(f"\n{'='*60}")
    print(f"  M3.5f: Bet#1+Bet#2 COUPLED — cluster value AFTER concern fusion")
    print(f"  {n_records} records, {len(trials)} trials, top_k={top_k}")
    print(f"{'='*60}")

    # 1. Embed corpus (semantic + doc2query context vectors).
    print(f"  embedding {n_records} records...")
    sem_vecs = embed.embed([c[2] for c in corpus])

    print("  generating doc2query context vectors (concern fusion input, concurrent)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def doc2query_one(idx_and_record):
        idx, (rid, name, text) = idx_and_record
        questions = generate_doc2query(llm, name, text[:1200])
        return idx, questions

    # Concurrent doc2query (DeepSeek supports up to 500 concurrent).
    questions_map: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(doc2query_one, (i, rec)): i for i, rec in enumerate(corpus)}
        done_count = 0
        for future in as_completed(futures):
            idx, questions = future.result()
            questions_map[idx] = questions
            done_count += 1
            if done_count % 50 == 0:
                print(f"    doc2query: {done_count}/{n_records} done")

    # Embed questions (batch, not concurrent — bge-m3 is local).
    ctx_vecs_all: list[list[list[float]]] = []
    for i in range(n_records):
        questions = questions_map.get(i, [])
        if questions:
            ctx_vecs_all.append(embed.embed(questions))
        else:
            ctx_vecs_all.append([sem_vecs[i]])  # fallback: use sem as ctx
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_records} doc2query done")

    # 2. Cluster by semantic vectors.
    print("  clustering (HDBSCAN)...")
    clusters = cluster_hdbscan(sem_vecs, min_cluster_size=3, min_samples=2)
    n_noise = len(clusters.pop(-1, []))
    print(f"  {len(clusters)} clusters ({n_noise} noise)")
    cluster_graph = build_cluster_graph(clusters, sem_vecs, edge_threshold=0.6)
    print(f"  cluster graph: {sum(len(v) for v in cluster_graph.values()) // 2} edges")

    # 3. For each trial: concern fusion rank both conditions.
    flat_gt_ranks = []
    cluster_gt_ranks = []
    flat_found = 0
    cluster_found = 0
    promotions = 0
    demotions = 0
    corpus_ids = [c[0] for c in corpus]

    for ti, trial in enumerate(trials):
        query = trial["query"]
        gt = set(trial["ground_truth_files"])
        q_vec = embed.embed([query])[0]

        # --- Condition A: flat + concern fusion ---
        flat_scores = {}
        for i in range(n_records):
            fused = concern_fusion_score(q_vec, sem_vecs[i], ctx_vecs_all[i])
            flat_scores[corpus_ids[i]] = fused
        flat_ranked = sorted(flat_scores.items(), key=lambda x: x[1], reverse=True)
        flat_ranked_ids = [rid for rid, _ in flat_ranked[:top_k]]

        # Flat recall.
        flat_result = simulate_flat_agent(flat_ranked_ids, gt)
        if flat_result["found"]:
            flat_found += 1
        flat_rank_map = {rid: idx for idx, rid in enumerate(flat_ranked_ids)}
        flat_gt_rank = float(sum(flat_rank_map.get(g, top_k) for g in gt)) / max(len(gt), 1)
        flat_gt_ranks.append(flat_gt_rank)

        # --- Condition B: cluster GraphRAG on TOP of concern fusion ---
        # Use concern fusion scores as the SEED for GraphRAG propagation,
        # instead of pure cosine centroid scores.
        # Step 1: compute cluster scores from fused scores (avg of members).
        cluster_fused_scores: dict[int, float] = {}
        for ci, members in clusters.items():
            if members:
                scores = [flat_scores.get(corpus_ids[m], 0.0) for m in members if m < n_records]
                cluster_fused_scores[ci] = sum(scores) / len(scores) if scores else 0.0

        # Step 2: GraphRAG propagation on fused cluster scores.
        propagated = dict(cluster_fused_scores)
        for ci, base in cluster_fused_scores.items():
            for neighbor in cluster_graph.get(ci, []):
                boost = base * 0.5
                propagated[neighbor] = max(propagated.get(neighbor, 0.0), boost)

        # Step 3: score records = graph_fused_score × individual_fused_score.
        cluster_scores: dict[str, float] = {}
        for ci, members in clusters.items():
            graph_s = propagated.get(ci, 0.0)
            for m in members:
                if m < n_records:
                    individual = flat_scores.get(corpus_ids[m], 0.0)
                    cluster_scores[corpus_ids[m]] = (graph_s * individual) ** 0.5

        cluster_ranked = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        cluster_ranked_ids = [rid for rid, _ in cluster_ranked[:top_k]]

        # Cluster recall.
        cluster_result = simulate_flat_agent(cluster_ranked_ids, gt)
        if cluster_result["found"]:
            cluster_found += 1
        cluster_rank_map = {rid: idx for idx, rid in enumerate(cluster_ranked_ids)}
        cluster_gt_rank = float(sum(cluster_rank_map.get(g, top_k) for g in gt)) / max(len(gt), 1)
        cluster_gt_ranks.append(cluster_gt_rank)

        if cluster_gt_rank < flat_gt_rank:
            promotions += 1
        elif cluster_gt_rank > flat_gt_rank:
            demotions += 1

        if (ti + 1) % 20 == 0:
            print(f"    {ti+1}/{len(trials)} trials done")

    # 4. Aggregate.
    n = len(trials)
    flat_mean_rank = sum(flat_gt_ranks) / n
    cluster_mean_rank = sum(cluster_gt_ranks) / n
    rank_delta = flat_mean_rank - cluster_mean_rank  # positive = cluster better
    rank_pct = rank_delta / flat_mean_rank * 100 if flat_mean_rank > 0 else 0

    if rank_pct > 5:
        verdict = "SUPPORTED"
    elif rank_pct < -5:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "experiment": "Bet#1+Bet#2 coupled (cluster value after concern fusion)",
        "n_trials": n,
        "n_records": n_records,
        "n_clusters": len(clusters),
        "top_k": top_k,
        "flat_concern": {
            "gt_mean_rank": round(flat_mean_rank, 2),
            "found_rate": round(flat_found / n, 3),
        },
        "cluster_concern_graphrag": {
            "gt_mean_rank": round(cluster_mean_rank, 2),
            "found_rate": round(cluster_found / n, 3),
        },
        "rank_delta": round(rank_delta, 2),
        "rank_improvement_pct": round(rank_pct, 1),
        "promotions": promotions,
        "demotions": demotions,
        "verdict": verdict,
    }

    print(f"\n  --- COUPLED RESULTS (concern fusion + cluster GraphRAG) ---")
    print(f"  Flat+concern:     GT rank={flat_mean_rank:.2f}  found={flat_found}/{n}")
    print(f"  Cluster+concern:  GT rank={cluster_mean_rank:.2f}  found={cluster_found}/{n}")
    print(f"  Rank delta: {rank_delta:+.2f} ({rank_pct:+.1f}%)")
    print(f"  Promotions: {promotions}/{n}  Demotions: {demotions}/{n}")
    print(f"  ★ Verdict: {verdict}")
    return result


def load_code_corpus(max_files: int = 0):
    from jgraphrag.config import PI_REPO_PATH
    repo = Path(PI_REPO_PATH)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))
    if max_files > 0:
        files = files[:max_files]
    return [
        (str(p.relative_to(repo)).replace("\\", "/"),
         p.name,
         p.read_text(encoding="utf-8", errors="ignore")[:1500])
        for p in files
    ]


def main():
    ap = argparse.ArgumentParser(description="M3.5f coupled Bet#1+Bet#2")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-trials", type=int, default=50)
    ap.add_argument("--max-files", type=int, default=100)
    args = ap.parse_args()

    embed = BgeM3Provider()
    llm = DeepSeekProvider()

    trials = json.loads((REPO / "experiments" / "m2" / "code_trials.json").read_text())
    trials = trials[:args.max_trials]
    corpus = load_code_corpus(max_files=args.max_files)

    result = run_coupled_experiment(embed, llm, corpus, trials, args.top_k)

    out_path = EXP / f"m3_coupled_k{args.top_k}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")
    print(f"\n  ★ COUPLED VERDICT: {result['verdict']}")


if __name__ == "__main__":
    main()
