"""FineRecord embedding dispersion pre-experiment.

Tests the core hypothesis: are FineRecord (symbol-level) embeddings
naturally more dispersed than Record (file-level) embeddings?

If yes → M3's "graph too dense" problem disappears at FineRecord level
        → vector-cluster GraphRAG may work there.
If no  → the hypothesis is falsified, need different approach.

Metrics:
  1. Mean pairwise cosine distance (higher = more dispersed)
  2. Distribution of pairwise distances (histogram)
  3. HDBSCAN cluster count (more clusters = more structure discovered)
  4. Cluster graph edge density (edges/nodes ratio — M3 had 2566/77=33)

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.fine_record_dispersion --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider

REPO = Path(__file__).resolve().parents[1]


def split_file_to_symbols(content: str) -> list[dict]:
    """Split a source file into symbol-level FineRecords using regex
    (simplified tree-sitter substitute for this pre-experiment).

    Returns [{name, kind, body}, ...] for each function/class/method.
    """
    symbols = []
    # Match function/class/interface/type definitions.
    pattern = re.compile(
        r"(?:export\s+)?(?:async\s+)?"
        r"(function|class|interface|type|const)\s+"
        r"([A-Za-z_$][A-Za-z0-9_$]*)"
        r"[\s<(]",
        re.MULTILINE,
    )
    for m in pattern.finditer(content):
        kind = m.group(1)
        name = m.group(2)
        # Extract body: from this match to the next match (or EOF).
        start = m.start()
        end = len(content)
        # Find next symbol definition.
        next_match = pattern.search(content, m.end())
        if next_match:
            end = next_match.start()
        body = content[start:end][:500]  # cap at 500 chars for embedding
        if len(body.strip()) > 20:
            symbols.append({"name": name, "kind": kind, "body": body})
    return symbols


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def mean_pairwise_distance(vecs: list[list[float]], sample_size: int = 500) -> float:
    """Mean pairwise cosine distance (sampled for speed if large).."""
    import random
    if len(vecs) > sample_size:
        vecs = random.sample(vecs, sample_size)
    n = len(vecs)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - cosine(vecs[i], vecs[j])
            count += 1
    return total / count if count else 0.0


def distance_histogram(vecs: list[list[float]], bins: int = 10) -> list[int]:
    """Histogram of pairwise cosine distances (0.0-2.0 range)."""
    import random
    if len(vecs) > 300:
        vecs = random.sample(vecs, 300)
    hist = [0] * bins
    n = len(vecs)
    for i in range(n):
        for j in range(i + 1, n):
            dist = 1.0 - cosine(vecs[i], vecs[j])
            # Clamp to [0, 2] and bin.
            idx = min(int(dist / 2.0 * bins), bins - 1)
            hist[idx] += 1
    return hist


def run_experiment(embed: BgeM3Provider, repo_path: str, max_files: int = 50):
    repo = Path(repo_path)
    files = sorted(
        p for p in repo.rglob("*.ts")
        if "node_modules" not in str(p)
    )[:max_files]

    print(f"  Loading {len(files)} files from {repo_path}...")

    # --- Record level (file-level) ---
    record_texts = []
    record_ids = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")[:2000]
        record_texts.append(text)
        record_ids.append(str(f.relative_to(repo)))

    # --- FineRecord level (symbol-level) ---
    fine_texts = []
    fine_ids = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        symbols = split_file_to_symbols(content)
        for sym in symbols:
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
            fine_ids.append(f"{f.relative_to(repo)}::{sym['name']}")

    print(f"  Records (file-level): {len(record_texts)}")
    print(f"  FineRecords (symbol-level): {len(fine_texts)}")

    # Embed both.
    print("  Embedding records...")
    record_vecs = embed.embed(record_texts)
    print("  Embedding fine records...")
    fine_vecs = embed.embed(fine_texts)

    # --- Compare dispersion ---
    print("\n" + "=" * 60)
    print("  DISPERSION COMPARISON")
    print("=" * 60)

    rec_mean_dist = mean_pairwise_distance([v for v in record_vecs])
    fine_mean_dist = mean_pairwise_distance([v for v in fine_vecs])

    print(f"\n  Mean pairwise cosine distance:")
    print(f"    Record (file):     {rec_mean_dist:.4f}")
    print(f"    FineRecord (sym):  {fine_mean_dist:.4f}")
    print(f"    Ratio:             {fine_mean_dist / rec_mean_dist:.2f}x" if rec_mean_dist > 0 else "")

    rec_hist = distance_histogram([v for v in record_vecs])
    fine_hist = distance_histogram([v for v in fine_vecs])

    print(f"\n  Distance histogram (bins of 0.2, from 0.0=similar to 2.0=opposite):")
    print(f"    {'Bin':>8s}  {'Record':>8s}  {'FineRec':>8s}  {'Bar (Fine)':>20s}")
    for i in range(len(rec_hist)):
        lo = i * 2.0 / len(rec_hist)
        hi = (i + 1) * 2.0 / len(rec_hist)
        bar = "█" * min(int(fine_hist[i] / max(max(fine_hist), 1) * 20), 20)
        print(f"    {lo:.1f}-{hi:.1f}  {rec_hist[i]:>8d}  {fine_hist[i]:>8d}  {bar}")

    # --- HDBSCAN cluster comparison ---
    print(f"\n  HDBSCAN clustering (min_cluster_size=3, min_samples=2, cosine):")
    try:
        from sklearn.cluster import HDBSCAN
        import numpy as np

        rec_labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(np.array(record_vecs))
        fine_labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(np.array(fine_vecs))

        rec_clusters = len(set(rec_labels) - {-1})
        fine_clusters = len(set(fine_labels) - {-1})
        rec_noise = sum(1 for l in rec_labels if l == -1)
        fine_noise = sum(1 for l in fine_labels if l == -1)

        print(f"    Record:     {rec_clusters} clusters, {rec_noise}/{len(rec_labels)} noise ({rec_noise/len(rec_labels)*100:.0f}%)")
        print(f"    FineRecord: {fine_clusters} clusters, {fine_noise}/{len(fine_labels)} noise ({fine_noise/len(fine_labels)*100:.0f}%)")

        if fine_clusters > 0:
            print(f"\n  ★ FineRecord discovered {fine_clusters} clusters (concepts)")
            print(f"    vs Record's {rec_clusters} clusters")
            if fine_clusters > rec_clusters:
                print(f"    → MORE structure at FineRecord level — hypothesis SUPPORTED!")
            else:
                print(f"    → Same or fewer clusters — hypothesis WEAKENED")
        else:
            print(f"\n  FineRecord clustering found no clusters — embedding may still be too dense")

    except ImportError:
        print("    (sklearn not available, skipping HDBSCAN)")

    # --- Cluster graph density estimate ---
    print(f"\n  Cluster graph density estimate:")
    print(f"    (M3 Record level had 2566 edges / 77 clusters = 33.3 edges/cluster)")
    if fine_clusters > 0:
        # Estimate: count centroid pairs with cosine > 0.6.
        cluster_centroids = {}
        for i, label in enumerate(fine_labels):
            if label >= 0:
                if label not in cluster_centroids:
                    cluster_centroids[label] = []
                cluster_centroids[label].append(fine_vecs[i])
        # Compute centroids.
        centroids = {}
        for label, members in cluster_centroids.items():
            dim = len(members[0])
            centroids[label] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
        # Count edges.
        edges = 0
        labels = list(centroids.keys())
        for i, la in enumerate(labels):
            for lb in labels[i + 1:]:
                if cosine(centroids[la], centroids[lb]) > 0.6:
                    edges += 1
        density = edges / max(fine_clusters, 1)
        print(f"    FineRecord: {edges} edges / {fine_clusters} clusters = {density:.1f} edges/cluster")
        if density < 33.3:
            print(f"    → DENSER than M3 ({density:.1f} < 33.3) — graph sparser! Hypothesis SUPPORTED!")
        else:
            print(f"    → Same density as M3 ({density:.1f} >= 33.3) — still too dense")
    else:
        print(f"    (no clusters to estimate)")

    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    if fine_mean_dist > rec_mean_dist * 1.2:
        print(f"  FineRecord embeddings ARE more dispersed ({fine_mean_dist/rec_mean_dist:.2f}x).")
        print(f"  M3's 'graph too dense' problem may disappear at FineRecord level.")
        print(f"  Vector-cluster GraphRAG worth pursuing at this granularity.")
    else:
        print(f"  FineRecord embeddings NOT significantly more dispersed.")
        print(f"  Vector-cluster approach may still fail — need different GraphRAG method.")


def main():
    ap = argparse.ArgumentParser(description="FineRecord embedding dispersion pre-experiment")
    ap.add_argument("--repo", required=True, help="path to codebase")
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()

    embed = BgeM3Provider()
    run_experiment(embed, args.repo, args.max_files)


if __name__ == "__main__":
    main()
