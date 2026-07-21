"""FineRecord graph density: nonlinear centroid transform experiment.

M3 + dispersion experiment showed: FineRecord discovers 205 clusters but
the cluster graph is too dense (70.8 edges/cluster). This tests whether
nonlinear transforms on centroid similarities can sparsify the graph.

Approach: for each pair of cluster centroids, compute cosine similarity,
then apply a nonlinear transform, then threshold. Compare edge density
across transforms.

Usage:
    cd crates/lincle/python; source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.nonlinear_graph_density --repo /tmp/pi-repo --max-files 50
"""
from __future__ import annotations

import argparse
import math
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from experiments.fine_record_dispersion import cosine, split_file_to_symbols


def compute_centroids(labels, vecs):
    """Compute cluster centroids from labels + vectors."""
    cluster_members: dict[int, list] = {}
    for i, label in enumerate(labels):
        if label >= 0:
            cluster_members.setdefault(label, []).append(vecs[i])
    centroids = {}
    for label, members in cluster_members.items():
        dim = len(members[0])
        centroids[label] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
    return centroids


def compute_all_similarities(centroids: dict) -> list[float]:
    """Compute all pairwise centroid cosine similarities."""
    sims = []
    labels = list(centroids.keys())
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            sims.append(cosine(centroids[la], centroids[lb]))
    return sims


def count_edges(sims: list[float], threshold: float) -> int:
    return sum(1 for s in sims if s >= threshold)


def power_transform(x: float, p: float) -> float:
    """x^p — amplifies differences for p>1."""
    return x ** p if x > 0 else 0.0


def sigmoid_transform(x: float, theta: float, k: float) -> float:
    """Sigmoid centered at theta with steepness k."""
    return 1.0 / (1.0 + math.exp(-k * (x - theta)))


def mean_normalize(sims: list[float]) -> list[float]:
    """Subtract mean, divide by std → only outliers are positive."""
    if not sims:
        return sims
    mean = sum(sims) / len(sims)
    var = sum((s - mean) ** 2 for s in sims) / len(sims)
    std = math.sqrt(var) if var > 0 else 1.0
    return [(s - mean) / std for s in sims]


def topk_edges(centroids: dict, k: int = 3) -> int:
    """Each cluster connects only to its top-K nearest neighbors."""
    labels = list(centroids.keys())
    edges = set()
    for la in labels:
        sims = [(lb, cosine(centroids[la], centroids[lb])) for lb in labels if lb != la]
        sims.sort(key=lambda x: x[1], reverse=True)
        for lb, _ in sims[:k]:
            edge = (min(la, lb), max(la, lb))
            edges.add(edge)
    return len(edges)


def run_experiment(embed: BgeM3Provider, repo_path: str, max_files: int = 50):
    from sklearn.cluster import HDBSCAN
    import numpy as np

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    # Split to FineRecords.
    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")

    print(f"  {len(fine_texts)} FineRecords from {len(files)} files")
    print("  embedding...")
    fine_vecs = embed.embed(fine_texts)

    # Cluster.
    labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(np.array(fine_vecs))
    n_clusters = len(set(labels) - {-1})
    n_noise = sum(1 for l in labels if l == -1)
    print(f"  HDBSCAN: {n_clusters} clusters, {n_noise} noise ({n_noise/len(labels)*100:.0f}%)")

    # Compute centroids + all pairwise similarities.
    centroids = compute_centroids(labels, fine_vecs)
    sims = compute_all_similarities(centroids)
    n_clusters_actual = len(centroids)

    print(f"\n{'='*70}")
    print(f"  GRAPH DENSITY: NONLINEAR TRANSFORM COMPARISON")
    print(f"  {n_clusters_actual} clusters, {len(sims)} possible edges")
    print(f"{'='*70}")

    # --- Method 0: Raw cosine (baseline, M3 approach) ---
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        edges = count_edges(sims, thresh)
        density = edges / max(n_clusters_actual, 1)
        print(f"  Raw cosine ≥{thresh}:  {edges:>6d} edges ({density:.1f}/cluster)")

    # --- Method 1: Power transform (x^p) ---
    print(f"\n  Power transform x^p, threshold=0.5:")
    for p in [2, 3, 4, 5, 8]:
        transformed = [power_transform(s, p) for s in sims]
        edges = count_edges(transformed, 0.5)
        density = edges / max(n_clusters_actual, 1)
        print(f"    p={p}:  {edges:>6d} edges ({density:.1f}/cluster)")

    # --- Method 2: Sigmoid ---
    print(f"\n  Sigmoid transform, threshold=0.5:")
    for theta in [0.6, 0.7, 0.75, 0.8]:
        for k in [10, 20, 50]:
            transformed = [sigmoid_transform(s, theta, k) for s in sims]
            edges = count_edges(transformed, 0.5)
            density = edges / max(n_clusters_actual, 1)
            print(f"    θ={theta}, k={k:>2d}:  {edges:>6d} edges ({density:.1f}/cluster)")

    # --- Method 3: Mean normalize ---
    normed = mean_normalize(sims)
    for thresh in [0.0, 0.5, 1.0, 1.5, 2.0]:
        edges = count_edges(normed, thresh)
        density = edges / max(n_clusters_actual, 1)
        print(f"  Mean-normalized ≥{thresh}:  {edges:>6d} edges ({density:.1f}/cluster)")

    # --- Method 4: Top-K nearest ---
    print(f"\n  Top-K nearest neighbors:")
    for k in [1, 2, 3, 5, 10]:
        edges = topk_edges(centroids, k)
        density = edges / max(n_clusters_actual, 1)
        print(f"    k={k:>2d}:  {edges:>6d} edges ({density:.1f}/cluster)")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY: which method gives 'good' density (3-10 edges/cluster)?")
    print(f"{'='*70}")
    print(f"  Target: 3-10 edges/cluster (sparse enough for GraphRAG)")
    print(f"  M3 baseline: 33.3 edges/cluster (too dense)")
    methods = [
        ("Raw ≥0.8", count_edges(sims, 0.8) / max(n_clusters_actual, 1)),
        ("Power x^4 ≥0.5", count_edges([power_transform(s, 4) for s in sims], 0.5) / max(n_clusters_actual, 1)),
        ("Power x^8 ≥0.5", count_edges([power_transform(s, 8) for s in sims], 0.5) / max(n_clusters_actual, 1)),
        ("Sigmoid θ=0.75 k=20", count_edges([sigmoid_transform(s, 0.75, 20) for s in sims], 0.5) / max(n_clusters_actual, 1)),
        ("Mean-norm ≥1.0", count_edges(normed, 1.0) / max(n_clusters_actual, 1)),
        ("Top-K k=3", topk_edges(centroids, 3) / max(n_clusters_actual, 1)),
        ("Top-K k=5", topk_edges(centroids, 5) / max(n_clusters_actual, 1)),
    ]
    for name, density in methods:
        marker = "✓" if 3 <= density <= 10 else ("!" if density < 3 else "✗")
        print(f"  {marker} {name:30s}: {density:.1f} edges/cluster")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()
    embed = BgeM3Provider()
    run_experiment(embed, args.repo, args.max_files)


if __name__ == "__main__":
    main()
