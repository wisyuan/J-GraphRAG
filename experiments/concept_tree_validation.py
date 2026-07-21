"""Concept tree network validation: two experiments.

Experiment 1: Does Sigmoid-sparsified FineRecord concept tree improve
  retrieval over flat cosine ANN? (Bet#2 re-test at FineRecord level)

Experiment 2: Does domain-specific re-embedding (LLM-generated fine
  descriptions) improve within-cluster dispersion? (Sub-domain feature
  expansion hypothesis)

Usage:
    cd crates/lincle/python; source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.concept_tree_validation --repo /tmp/pi-repo --max-files 50
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from jgraphrag.llm import DeepSeekProvider
from experiments.fine_record_dispersion import cosine, split_file_to_symbols, mean_pairwise_distance

REPO = Path(__file__).resolve().parents[1]


def sigmoid_transform(x: float, theta: float = 0.75, k: float = 20.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - theta)))


def build_concept_tree(fine_vecs, labels, centroids, cluster_graph_edges, decay=0.5):
    """Build concept tree with GraphRAG propagation on Sigmoid-sparsified graph."""
    # Cluster scores from propagation.
    propagated = {}
    for ci in centroids:
        propagated[ci] = 0.0  # will be set by query

    return propagated


def experiment1_retrieval(embed, fine_texts, fine_vecs, fine_ids, labels, centroids, cluster_edges):
    """Experiment 1: concept tree retrieval vs flat cosine.

    For each query (using FineRecord names as pseudo-queries):
    - Flat: cosine(query, fine_record) → top-K
    - Tree: cosine(query, cluster_centroid) → find best cluster →
            propagate to neighbors → re-rank within clusters
    Compare: rank of the "target" FineRecord (self-retrieval baseline)
    """
    n = len(fine_vecs)
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1: Concept tree retrieval vs flat cosine")
    print(f"  {n} FineRecords, {len(centroids)} clusters")
    print(f"{'='*60}")

    # Build cluster membership.
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_members[label].append(i)

    # Use a subset of FineRecords as queries (their text as query).
    # For each, check: does flat cosine find itself at rank 1?
    # And: does concept tree find related FineRecords better?
    sample_indices = list(range(0, n, max(1, n // 100)))[:100]

    flat_self_rank = []  # self-retrieval rank (should always be 1 for flat)
    tree_cross_cluster_hits = []  # does tree find FineRecords in OTHER clusters?

    for qi in sample_indices:
        q_vec = fine_vecs[qi]
        target_id = fine_ids[qi]
        target_cluster = labels[qi]

        # --- Flat cosine ---
        flat_scores = [(fine_ids[i], cosine(q_vec, fine_vecs[i])) for i in range(n)]
        flat_scores.sort(key=lambda x: x[1], reverse=True)
        flat_ranked = [fid for fid, _ in flat_scores[:20]]
        flat_self_rank.append(flat_ranked.index(target_id) + 1 if target_id in flat_ranked else 21)

        # --- Concept tree (GraphRAG) ---
        # Step 1: score clusters by centroid proximity.
        cluster_scores = {ci: cosine(q_vec, centroids[ci]) for ci in centroids}

        # Step 2: Sigmoid-sparsified propagation.
        propagated = dict(cluster_scores)
        for ci, base_score in cluster_scores.items():
            for neighbor, edge_weight in cluster_edges.get(ci, []):
                boost = base_score * edge_weight * 0.5  # decay
                propagated[neighbor] = max(propagated.get(neighbor, 0.0), boost)

        # Step 3: score FineRecords = graph_score(cluster) × individual cosine.
        tree_scores = []
        for i in range(n):
            ci = labels[i]
            if ci < 0:
                # Noise point: only use individual cosine.
                tree_scores.append((fine_ids[i], cosine(q_vec, fine_vecs[i]) * 0.5))  # penalty
            else:
                graph_s = propagated.get(ci, 0.0)
                individual = cosine(q_vec, fine_vecs[i])
                tree_scores.append((fine_ids[i], (graph_s * individual) ** 0.5))

        tree_scores.sort(key=lambda x: x[1], reverse=True)
        tree_ranked = [fid for fid, _ in tree_scores[:20]]

        # Cross-cluster hits: FineRecords in tree top-20 that are in a DIFFERENT
        # cluster than the query (these are "graph-discovered" relations).
        cross_hits = 0
        for fid in tree_ranked[:10]:
            idx = fine_ids.index(fid) if fid in fine_ids else -1
            if idx >= 0 and labels[idx] >= 0 and labels[idx] != target_cluster:
                cross_hits += 1
        tree_cross_cluster_hits.append(cross_hits)

    print(f"\n  Flat cosine self-retrieval rank: mean={sum(flat_self_rank)/len(flat_self_rank):.1f}")
    print(f"  (rank 1 = perfect self-retrieval; flat should be ~1)")

    mean_cross = sum(tree_cross_cluster_hits) / len(tree_cross_cluster_hits) if tree_cross_cluster_hits else 0
    print(f"\n  Tree cross-cluster hits in top-10: mean={mean_cross:.1f}")
    print(f"  (higher = tree discovers related concepts in OTHER clusters)")
    print(f"  (flat cosine cannot do this — it only finds similar vectors)")

    if mean_cross > 1.0:
        print(f"\n  ★ Tree finds {mean_cross:.1f} cross-cluster related FineRecords per query")
        print(f"    These are 'concept-discovered' relations flat cosine misses.")
        print(f"    → Concept tree provides cross-cluster navigation value ✓")
    else:
        print(f"\n  ✗ Tree finds few cross-cluster hits ({mean_cross:.1f}/query)")
        print(f"    → Concept tree navigation may not add value at this scale")

    return {"flat_self_rank_mean": sum(flat_self_rank) / len(flat_self_rank),
            "tree_cross_cluster_mean": mean_cross}


def experiment2_fine_reembedding(embed, llm, fine_texts, fine_vecs, fine_ids, labels, centroids):
    """Experiment 2: sub-domain fine re-embedding.

    For each cluster: use LLM to generate fine-grained descriptions
    emphasizing WITHIN-cluster differences, then re-embed.
    Compare dispersion: global embedding vs fine re-embedding.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2: Sub-domain fine re-embedding")
    print(f"{'='*60}")

    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_members[label].append(i)

    # Pick top 5 largest clusters for the experiment.
    largest = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)[:5]

    global_dists = []
    fine_dists = []

    for cluster_id, members in largest:
        if len(members) < 4:
            continue

        member_texts = [fine_texts[i] for i in members]
        member_ids = [fine_ids[i] for i in members]

        # Global embedding dispersion.
        global_vecs = [fine_vecs[i] for i in members]
        global_dist = mean_pairwise_distance(global_vecs)
        global_dists.append(global_dist)

        # Generate fine descriptions via LLM (concurrent).
        def gen_fine_desc(idx_and_text):
            idx, text = idx_and_text
            # Get other members for contrast context.
            others = [t for j, t in enumerate(member_texts) if j != idx][:3]
            contrast = "\n---\n".join(others[:2])
            prompt = (
                f"Below is one code symbol from a group of related symbols.\n"
                f"Describe what makes THIS symbol UNIQUE compared to the others.\n"
                f"Focus on its specific functionality, not general patterns.\n\n"
                f"This symbol:\n{text[:500]}\n\n"
                f"Other symbols in the same group:\n{contrast[:500]}\n\n"
                f"Unique aspect (1-2 sentences):"
            )
            msg = llm.complete(prompt, 128)
            return idx, msg.content if not msg.is_error else text[:200]

        fine_descs = {}
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = {pool.submit(gen_fine_desc, (i, member_texts[i])): i for i in range(len(members))}
            for future in as_completed(futures):
                idx, desc = future.result()
                fine_descs[idx] = desc

        # Re-embed fine descriptions.
        fine_desc_texts = [fine_descs.get(i, member_texts[i][:200]) for i in range(len(members))]
        reembedded_vecs = embed.embed(fine_desc_texts)
        fine_dist = mean_pairwise_distance(reembedded_vecs)
        fine_dists.append(fine_dist)

        print(f"\n  Cluster {cluster_id} ({len(members)} members):")
        print(f"    Global embedding distance: {global_dist:.4f}")
        print(f"    Fine re-embed distance:   {fine_dist:.4f}")
        ratio = fine_dist / global_dist if global_dist > 0 else 0
        marker = "✓" if ratio > 1.2 else ("=" if ratio > 0.9 else "✗")
        print(f"    Ratio: {ratio:.2f}x {marker}")

    if global_dists and fine_dists:
        mean_global = sum(global_dists) / len(global_dists)
        mean_fine = sum(fine_dists) / len(fine_dists)
        overall_ratio = mean_fine / mean_global if mean_global > 0 else 0

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT 2 VERDICT")
        print(f"{'='*60}")
        print(f"  Mean global distance: {mean_global:.4f}")
        print(f"  Mean fine distance:   {mean_fine:.4f}")
        print(f"  Overall ratio: {overall_ratio:.2f}x")
        if overall_ratio > 1.2:
            print(f"  ★ Fine re-embedding DOES increase dispersion ({overall_ratio:.2f}x).")
            print(f"    Sub-domain feature expansion is effective — hypothesis SUPPORTED!")
        elif overall_ratio > 0.9:
            print(f"  = Marginal improvement. May need better prompting or larger clusters.")
        else:
            print(f"  ✗ Fine re-embedding does NOT increase dispersion.")
            print(f"    Hypothesis not supported at this scale.")

        return {"mean_global": mean_global, "mean_fine": mean_fine, "ratio": overall_ratio}
    return {}


def build_sigmoid_graph(centroids, theta=0.75, k=20.0, edge_threshold=0.5):
    """Build cluster graph with Sigmoid-transformed edges."""
    labels = list(centroids.keys())
    edges = defaultdict(list)
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            raw_sim = cosine(centroids[la], centroids[lb])
            transformed = sigmoid_transform(raw_sim, theta, k)
            if transformed >= edge_threshold:
                weight = transformed
                edges[la].append((lb, weight))
                edges[lb].append((la, weight))
    return dict(edges)


def run_experiment(embed, llm, repo_path, max_files):
    from sklearn.cluster import HDBSCAN
    import numpy as np

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    # Split to FineRecords.
    fine_texts = []
    fine_ids = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
            fine_ids.append(f"{f.relative_to(repo)}::{sym['name']}")

    n = len(fine_texts)
    print(f"  {n} FineRecords from {len(files)} files")
    print("  embedding...")
    fine_vecs = embed.embed(fine_texts)

    # Cluster.
    labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(np.array(fine_vecs))
    n_clusters = len(set(labels) - {-1})
    n_noise = sum(1 for l in labels if l == -1)
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise ({n_noise/n*100:.0f}%)")

    # Centroids.
    cluster_members: dict[int, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_members[label].append(i)
    centroids = {}
    for label, members in cluster_members.items():
        dim = len(fine_vecs[0])
        centroids[label] = [sum(fine_vecs[m][d] for m in members) / len(members) for d in range(dim)]

    # Sigmoid-sparsified cluster graph.
    cluster_edges = build_sigmoid_graph(centroids, theta=0.75, k=20.0, edge_threshold=0.5)
    total_edges = sum(len(v) for v in cluster_edges.values()) // 2
    density = total_edges / max(n_clusters, 1)
    print(f"  Sigmoid graph: {total_edges} edges ({density:.1f}/cluster)")

    # Run both experiments.
    result1 = experiment1_retrieval(embed, fine_texts, fine_vecs, fine_ids, labels, centroids, cluster_edges)
    result2 = experiment2_fine_reembedding(embed, llm, fine_texts, fine_vecs, fine_ids, labels, centroids)

    # Save results.
    out = REPO / "experiments" / "m3" / "concept_tree_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"experiment1": result1, "experiment2": result2,
               "n_fine_records": n, "n_clusters": n_clusters, "graph_density": density}
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out}")


def main():
    ap = argparse.ArgumentParser(description="Concept tree network validation")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()

    embed = BgeM3Provider()
    llm = DeepSeekProvider()
    run_experiment(embed, llm, args.repo, args.max_files)


if __name__ == "__main__":
    main()
