"""M3.5 A/B runner — cluster vs flat (Bet#2).

Tests whether SemanticCluster organization produces incremental value over
M2's flat ANN + ConcernFusion baseline. The key metric is Agent efficiency:
how many irrelevant expansions before finding the correct locator.

Condition A (flat baseline): flat top-K results, Agent digs linearly.
Condition B (cluster): results grouped by cluster, Agent digs cluster-first.

Simulated Agent strategy:
  - Flat: dig results one by one (sorted by score) until hitting ground-truth.
    Steps = position of first relevant result in the flat list.
  - Cluster: dig the top cluster first, check its representatives; if none
    relevant, move to next cluster. Steps = clusters checked + position
    within the target cluster.

Primary metric: steps_to_first_relevant (lower = better).
Secondary: irrelevant_expansions (how many non-relevant results expanded).

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.run_m3_cluster_ab --top-k 10 --max-trials 50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from jgraphrag.llm import DeepSeekProvider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "m3"
EXP.mkdir(parents=True, exist_ok=True)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_pairwise_distance(vectors: list[list[float]]) -> float:
    """Mean pairwise cosine distance — used as split signal proxy."""
    n = len(vectors)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - cosine(vectors[i], vectors[j])
            count += 1
    return total / count if count else 0.0


def cluster_hdbscan(
    sem_vecs: list[list[float]], min_cluster_size: int = 3, min_samples: int = 2
) -> dict[int, list[int]]:
    """HDBSCAN density clustering — discovers natural density regions without
    a fixed threshold. Better than greedy proximity for code embeddings where
    files are universally similar (greedy collapses 92/100 into one cluster).

    Falls back to proximity clustering if hdbscan/sklearn unavailable.
    """
    try:
        from sklearn.cluster import HDBSCAN as SkHDBSCAN
    except ImportError:
        # Fallback: proximity with tighter threshold.
        return cluster_by_proximity(sem_vecs, threshold=0.82)

    import numpy as np
    data = np.array(sem_vecs)
    clusterer = SkHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
    )
    labels = clusterer.fit_predict(data)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)
    # Remove noise cluster (-1) from the main cluster dict, but keep as "ungrouped".
    return dict(clusters)


def cluster_by_proximity(
    sem_vecs: list[list[float]], threshold: float = 0.82
) -> dict[int, list[int]]:
    """Greedy nearest-center clustering (fallback when HDBSCAN unavailable).
    Note: threshold tightened from 0.65 to 0.82 — code files are universally
    similar in bge-m3 space, so a loose threshold collapses everything."""
    clusters: dict[int, list[int]] = {}
    centroids: list[list[float]] = []
    for i, vec in enumerate(sem_vecs):
        best_cluster = -1
        best_sim = threshold
        for ci, centroid in enumerate(centroids):
            sim = cosine(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = ci
        if best_cluster >= 0:
            clusters[best_cluster].append(i)
            members = clusters[best_cluster]
            n = len(members)
            centroids[best_cluster] = [
                sum(sem_vecs[m][d] for m in members) / n
                for d in range(len(vec))
            ]
        else:
            ci = len(centroids)
            clusters[ci] = [i]
            centroids.append(vec[:])
    return clusters


def simulate_flat_agent(
    ranked_ids: list[str], ground_truth: set[str]
) -> dict:
    """Simulate Agent digging flat results linearly until hitting ground-truth."""
    steps = 0
    irrelevant = 0
    found = False
    for rid in ranked_ids:
        steps += 1
        if rid in ground_truth:
            found = True
            break
        irrelevant += 1
    return {
        "steps_to_first_relevant": steps if found else steps + 1,
        "irrelevant_expansions": irrelevant,
        "found": found,
    }


def graphrag_score(
    query_vec: list[float],
    sem_vecs: list[list[float]],
    clusters: dict[int, list[int]],
    cluster_graph: dict[int, list[int]],
    corpus_ids: list[str],
    decay: float = 0.5,
) -> dict[str, float]:
    """Cluster-level GraphRAG scoring (user insight 2026-07-06).

    Instead of pure cosine (point-to-point), use GRAPH PROPAGATION on the
    cluster graph to spread relevance:

    1. Score each cluster by centroid-query cosine (seed relevance).
    2. Propagate: a cluster's neighbors inherit a fraction (decay) of its score.
       This is 1-hop graph propagation — the GraphRAG advantage.
    3. Within each cluster, score members by cosine to query (local precision).
    4. Final record score = cluster_graph_score × member_cosine.

    Key difference from M3.5a-d: the FINAL ranking uses graph-reachable
    relevance (not just cosine). A record in a graph-neighbor cluster of
    the query's best cluster gets a boost that flat cosine CANNOT provide.

    Operates on the cluster graph (77 nodes) not the full record graph (379) →
    '精简到向量簇上开展' (computationally efficient GraphRAG).
    """
    # Step 1: seed cluster scores.
    cluster_scores: dict[int, float] = {}
    for ci, members in clusters.items():
        if not members:
            continue
        dim = len(sem_vecs[0])
        centroid = [sum(sem_vecs[m][d] for m in members) / len(members) for d in range(dim)]
        cluster_scores[ci] = cosine(query_vec, centroid)

    # Step 2: 1-hop graph propagation.
    propagated: dict[int, float] = dict(cluster_scores)
    for ci, base_score in cluster_scores.items():
        for neighbor in cluster_graph.get(ci, []):
            boost = base_score * decay
            propagated[neighbor] = max(propagated.get(neighbor, 0.0), boost)

    # Step 3: score records = propagated cluster score × individual cosine.
    scores: dict[str, float] = {}
    for ci, members in clusters.items():
        graph_score = propagated.get(ci, 0.0)
        for m in members:
            if m < len(corpus_ids) and m < len(sem_vecs):
                individual_cos = cosine(query_vec, sem_vecs[m])
                # GraphRAG score: geometric mean of graph reachability and local precision.
                scores[corpus_ids[m]] = (graph_score * individual_cos) ** 0.5
    return scores


def rank_by_graphrag(
    query_vec: list[float],
    sem_vecs: list[list[float]],
    clusters: dict[int, list[int]],
    cluster_graph: dict[int, list[int]],
    corpus_ids: list[str],
    top_k: int,
) -> list[str]:
    """Rank records by cluster-level GraphRAG scoring. Returns top-K ids."""
    scores = graphrag_score(query_vec, sem_vecs, clusters, cluster_graph, corpus_ids)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [rid for rid, _ in ranked[:top_k]]


def build_cluster_graph(
    clusters: dict[int, list[int]], sem_vecs: list[list[float]], edge_threshold: float = 0.6
) -> dict[int, list[int]]:
    """Build ClusterEdge graph: connect clusters whose centroids are close.
    This gives the Agent direction for inter-cluster traversal — the knowledge
    graph between concepts (v2 §4.5, user insight 2026-07-06)."""
    # Compute centroids.
    centroids: dict[int, list[float]] = {}
    for ci, members in clusters.items():
        if not members:
            continue
        dim = len(sem_vecs[0])
        centroids[ci] = [sum(sem_vecs[m][d] for m in members) / len(members) for d in range(dim)]

    # Connect clusters with centroid cosine > threshold.
    graph: dict[int, list[int]] = defaultdict(list)
    cis = list(centroids.keys())
    for i, ca in enumerate(cis):
        for cb in cis[i + 1:]:
            sim = cosine(centroids[ca], centroids[cb])
            if sim > edge_threshold:
                graph[ca].append(cb)
                graph[cb].append(ca)
    return dict(graph)


def simulate_cluster_agent(
    ranked_ids: list[str],
    ground_truth: set[str],
    clusters: dict[int, list[int]],
    sem_vecs: list[list[float]],
    query_vec: list[float],
    cluster_graph: dict[int, list[int]],
    top_k: int,
) -> dict:
    """Simulate Agent with cluster-guided search (knowledge-graph-enhanced).

    Strategy:
    1. Score each cluster by centroid-query cosine proximity.
    2. Boost clusters that are graph-neighbors of the top-scoring cluster
       (the ClusterEdge graph provides DIRECTION — related concepts get promoted).
    3. Check clusters in score order, only the top-K clusters (not all).
    4. Within each cluster, check representative (best-ranked member) first.

    Key insight (user 2026-07-06): the cluster graph's purpose is to give
    DIRECTION for inter-cluster jumps — not to BFS-traverse everything.
    A cluster co-occurring with the query's best cluster gets a proximity boost.
    """
    id_to_rank = {rid: idx for idx, rid in enumerate(ranked_ids)}

    # Compute cluster centroids + score by query proximity.
    cluster_scores: dict[int, float] = {}
    for ci, members in clusters.items():
        if not members:
            continue
        dim = len(sem_vecs[0])
        centroid = [sum(sem_vecs[m][d] for m in members) / len(members) for d in range(dim)]
        cluster_scores[ci] = cosine(query_vec, centroid)

    # Graph boost: clusters adjacent to the top cluster get a proximity lift.
    if cluster_scores:
        top_cluster = max(cluster_scores, key=cluster_scores.get)
        for neighbor in cluster_graph.get(top_cluster, []):
            if neighbor in cluster_scores:
                # Boost: blend query-proximity with graph proximity.
                cluster_scores[neighbor] *= 1.15  # 15% boost for graph neighbors

    # Sort clusters by score (descending), only check top-K clusters.
    max_clusters_to_check = min(len(cluster_scores), top_k)
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)[:max_clusters_to_check]

    steps = 0
    irrelevant = 0
    found = False

    for ci, _ in sorted_clusters:
        members = clusters.get(ci, [])

        # Check the best-ranked member of this cluster.
        steps += 1
        best_member = min(members, key=lambda i: id_to_rank.get(i, top_k + 1)) if members else -1
        if best_member >= 0 and best_member < len(ranked_ids):
            if ranked_ids[best_member] in ground_truth:
                found = True
                break

        # Check up to 2 more members.
        remaining = sorted(members, key=lambda i: id_to_rank.get(i, top_k + 1))[1:3]
        for idx in remaining:
            steps += 1
            if idx < len(ranked_ids) and ranked_ids[idx] in ground_truth:
                found = True
                break
            irrelevant += 1
        if found:
            break
        irrelevant += 1  # cluster representative was irrelevant

    return {
        "steps_to_first_relevant": steps if found else steps + 1,
        "irrelevant_expansions": irrelevant,
        "found": found,
    }


def run_experiment(embed, llm, corpus, trials, top_k, domain_label):
    print(f"\n{'='*60}")
    print(f"  M3.5 A/B — {domain_label}")
    print(f"  {len(corpus)} records, {len(trials)} trials, top_k={top_k}")
    print(f"{'='*60}")

    # 1. Embed corpus.
    print(f"  embedding {len(corpus)} records...")
    sem_vecs = embed.embed([c[2] for c in corpus])

    # 2. Cluster the corpus using HDBSCAN (density-based, no fixed threshold).
    print("  clustering (HDBSCAN)...")
    clusters = cluster_hdbscan(sem_vecs, min_cluster_size=3, min_samples=2)
    n_noise = len(clusters.pop(-1, []))
    print(f"  discovered {len(clusters)} clusters ({n_noise} noise points)")
    for ci, members in sorted(clusters.items()):
        print(f"    cluster {ci}: {len(members)} members")

    # 2b. Build ClusterEdge graph (knowledge graph between concepts).
    print("  building cluster graph (ClusterEdge)...")
    cluster_graph = build_cluster_graph(clusters, sem_vecs, edge_threshold=0.6)
    print(f"  cluster graph: {sum(len(v) for v in cluster_graph.values()) // 2} edges")

    # 3. For each trial, rank + simulate both agents + compute confidence ranking.
    flat_steps = []
    flat_irrelevant = []
    cluster_steps = []
    cluster_irrelevant = []
    flat_found = 0
    cluster_found = 0
    # Confidence ranking: does cluster push ground-truth higher in the ranking?
    flat_gt_ranks: list[float] = []   # avg rank of GT in flat cosine order
    cluster_gt_ranks: list[float] = []  # avg rank of GT in cluster-rescored order
    cluster_promotions = 0  # trials where cluster moved GT higher than flat
    cluster_demotions = 0   # trials where cluster moved GT lower

    for ti, trial in enumerate(trials):
        query = trial["query"]
        gt = set(trial["ground_truth_files"])
        q_vec = embed.embed([query])[0]

        # Score + rank candidates by semantic cosine.
        scored = [
            (corpus[i][0], cosine(q_vec, sem_vecs[i]), i)
            for i in range(len(corpus))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked_ids = [s[0] for s in scored[:top_k * 3]]  # wider pool for cluster

        # Flat agent.
        flat_result = simulate_flat_agent(ranked_ids[:top_k], gt)
        flat_steps.append(flat_result["steps_to_first_relevant"])
        flat_irrelevant.append(flat_result["irrelevant_expansions"])
        if flat_result["found"]:
            flat_found += 1

        # Cluster agent (with graph traversal).
        cluster_result = simulate_cluster_agent(
            ranked_ids, gt, clusters, sem_vecs, q_vec, cluster_graph, top_k
        )
        cluster_steps.append(cluster_result["steps_to_first_relevant"])
        cluster_irrelevant.append(cluster_result["irrelevant_expansions"])
        if cluster_result["found"]:
            cluster_found += 1

        # --- Confidence ranking: GraphRAG vs flat cosine (user insight) ---
        # Flat: rank by pure cosine.
        flat_rank: dict[str, int] = {rid: idx for idx, rid in enumerate(ranked_ids[:top_k])}
        flat_gt_rank = float(sum(flat_rank.get(g, top_k) for g in gt)) / max(len(gt), 1)
        flat_gt_ranks.append(flat_gt_rank)

        # Cluster: rank by cluster-level GraphRAG (graph propagation, NOT cosine re-weighting).
        # This is the key change from M3.5a-d: GraphRAG uses graph reachability
        # to propagate relevance, not just weighted cosine.
        corpus_ids = [c[0] for c in corpus]
        graphrag_ranked = rank_by_graphrag(
            q_vec, sem_vecs, clusters, cluster_graph, corpus_ids, top_k
        )
        cluster_rank: dict[str, int] = {rid: idx for idx, rid in enumerate(graphrag_ranked[:top_k])}
        cluster_gt_rank = float(sum(cluster_rank.get(g, top_k) for g in gt)) / max(len(gt), 1)
        cluster_gt_ranks.append(cluster_gt_rank)

        if cluster_gt_rank < flat_gt_rank:
            cluster_promotions += 1
        elif cluster_gt_rank > flat_gt_rank:
            cluster_demotions += 1

        if (ti + 1) % 20 == 0:
            print(f"    {ti+1}/{len(trials)} trials done")

    n = len(trials)
    flat_mean_steps = sum(flat_steps) / n
    cluster_mean_steps = sum(cluster_steps) / n
    flat_mean_irr = sum(flat_irrelevant) / n
    cluster_mean_irr = sum(cluster_irrelevant) / n
    flat_mean_gt_rank = sum(flat_gt_ranks) / n
    cluster_mean_gt_rank = sum(cluster_gt_ranks) / n

    step_reduction = flat_mean_steps - cluster_mean_steps
    irr_reduction = flat_mean_irr - cluster_mean_irr
    rank_improvement = flat_mean_gt_rank - cluster_mean_gt_rank  # positive = cluster ranks GT higher

    # Verdict: cluster SUPPORTED if either (a) recall efficiency improves OR
    # (b) confidence ranking improves (GT pushed higher).
    step_improvement = step_reduction / flat_mean_steps if flat_mean_steps > 0 else 0
    irr_improvement = irr_reduction / flat_mean_irr if flat_mean_irr > 0 else 0
    rank_improvement_pct = rank_improvement / flat_mean_gt_rank * 100 if flat_mean_gt_rank > 0 else 0

    if step_improvement > 0.1 or irr_improvement > 0.1 or rank_improvement_pct > 5:
        verdict = "SUPPORTED"
    elif step_improvement < -0.05 and irr_improvement < -0.05 and rank_improvement_pct < -5:
        verdict = "FALSIFIED"
    else:
        verdict = "INCONCLUSIVE"

    result = {
        "domain": domain_label,
        "n_trials": n,
        "n_records": len(corpus),
        "n_clusters": len(clusters),
        "top_k": top_k,
        "flat": {
            "mean_steps": round(flat_mean_steps, 2),
            "mean_irrelevant": round(flat_mean_irr, 2),
            "found_rate": round(flat_found / n, 3),
            "gt_mean_rank": round(flat_mean_gt_rank, 2),
        },
        "cluster": {
            "mean_steps": round(cluster_mean_steps, 2),
            "mean_irrelevant": round(cluster_mean_irr, 2),
            "found_rate": round(cluster_found / n, 3),
            "gt_mean_rank": round(cluster_mean_gt_rank, 2),
        },
        "improvement": {
            "step_reduction": round(step_reduction, 2),
            "irr_reduction": round(irr_reduction, 2),
            "step_improvement_pct": round(step_improvement * 100, 1),
            "irr_improvement_pct": round(irr_improvement * 100, 1),
            "rank_improvement": round(rank_improvement, 2),
            "rank_improvement_pct": round(rank_improvement_pct, 1),
            "gt_promotions": cluster_promotions,
            "gt_demotions": cluster_demotions,
        },
        "verdict": verdict,
    }

    print(f"\n  --- RESULTS ({domain_label}) ---")
    print(f"  Recall efficiency:")
    print(f"    Flat:    steps={flat_mean_steps:.1f}  irrelevant={flat_mean_irr:.1f}  found={flat_found}/{n}")
    print(f"    Cluster: steps={cluster_mean_steps:.1f}  irrelevant={cluster_mean_irr:.1f}  found={cluster_found}/{n}")
    print(f"  Confidence ranking (★ new dimension):")
    print(f"    Flat GT rank:    {flat_mean_gt_rank:.2f}  (lower=better)")
    print(f"    Cluster GT rank: {cluster_mean_gt_rank:.2f}")
    print(f"    Rank improvement: {rank_improvement:+.2f} ({rank_improvement_pct:+.1f}%)")
    print(f"    GT promoted (cluster ranks higher): {cluster_promotions}/{n}")
    print(f"    GT demoted  (cluster ranks lower):  {cluster_demotions}/{n}")
    print(f"  Step reduction: {step_reduction:+.1f} ({step_improvement*100:+.1f}%)")
    print(f"  ★ Verdict: {verdict}")

    return result


def load_code_corpus(max_files: int = 0) -> list:
    """Load pi source files. max_files=0 means all files (full 379)."""
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
    ap = argparse.ArgumentParser(description="M3.5 cluster vs flat A/B (Bet#2)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-trials", type=int, default=50)
    ap.add_argument("--max-files", type=int, default=0, help="0=all pi files")
    ap.add_argument("--trials-file", default="", help="path to trials JSON")
    args = ap.parse_args()

    embed = BgeM3Provider()

    # Load code trials (reuse M2's).
    trials_path = args.trials_file or str(REPO / "experiments" / "m2" / "code_trials.json")
    trials = json.loads(Path(trials_path).read_text())[:args.max_trials]
    corpus = load_code_corpus(max_files=args.max_files)

    result = run_experiment(embed, None, corpus, trials, args.top_k, "Code (pi, Bet#2)")

    out_path = EXP / f"m3_ab_cluster_k{args.top_k}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n  Results saved to {out_path}")

    print(f"\n{'='*60}")
    print(f"  BET #2 VERDICT: {result['verdict']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
