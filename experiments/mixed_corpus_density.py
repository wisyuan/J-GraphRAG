"""Mixed-corpus Record-level density experiment.

Tests the hypothesis (user insight): M3's graph-too-dense problem was
caused by corpus homogeneity (all pi TypeScript). If we mix code + novel
+ enterprise docs, embeddings naturally disperse → graph sparse →
GraphRAG feasible WITHOUT nonlinear transforms.

Corpora mixed:
  - pi code (TypeScript agent toolkit) — M3's homogeneous corpus
  - Pride & Prejudice chapters (19th century English novel)
  - Enterprise scenario docs/scripts/CSV descriptions (business Chinese+English)

Metrics: mean pairwise cosine distance, HDBSCAN cluster count, edge density.

Usage:
    cd crates/lincle/python; source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.mixed_corpus_density
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from experiments.fine_record_dispersion import cosine, mean_pairwise_distance, distance_histogram, split_file_to_symbols

REPO = Path(__file__).resolve().parents[1]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def load_pi_code(repo_path: str, max_files: int = 50) -> list[tuple[str, str]]:
    """Load pi .ts files → [(id, text)]."""
    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    out = []
    for f in files:
        rel = str(f.relative_to(repo)).replace("\\", "/")
        out.append((f"pi:{rel}", f.read_text(encoding="utf-8", errors="ignore")[:2000]))
    return out


def load_novel_chapters(novel_path: str) -> list[tuple[str, str]]:
    """Load novel chapters → [(id, text)]."""
    from experiments.load_novel import chunk_novel
    text = Path(novel_path).read_text(encoding="utf-8", errors="ignore")
    chapters = chunk_novel(text)
    out = []
    for ch in chapters[:30]:  # cap at 30 chapters
        out.append((f"novel:chapter_{ch['id']}", ch["text"][:2000]))
    return out


def load_enterprise(scenario_path: str) -> list[tuple[str, str]]:
    """Load enterprise scenario files → [(id, text)]."""
    import csv as csvmod
    import sqlite3
    scenario = Path(scenario_path)
    out = []

    # Docs + scripts.
    for f in sorted(scenario.rglob("*")):
        if f.is_file() and f.suffix in (".md", ".py"):
            rel = f"ent:{f.relative_to(scenario)}"
            out.append((str(rel), f.read_text(errors="ignore")[:2000]))

    # CSVs.
    for f in sorted((scenario / "data").glob("*.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csvmod.reader(fh))
            if rows:
                header = ",".join(rows[0])
                sample = "\n".join(",".join(r) for r in rows[1:4])
                out.append((f"ent:csv:{f.name}", f"CSV: {f.name}\nColumns: {header}\n{sample}"))

    # DB tables.
    db_path = scenario / "enterprise.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        for (table,) in cursor.fetchall():
            cursor.execute(f"PRAGMA table_info({table})")
            cols = ", ".join(f"{c[1]}({c[2]})" for c in cursor.fetchall())
            cursor.execute(f"SELECT * FROM {table} LIMIT 3")
            sample = "\n".join(", ".join(str(v) for v in row) for row in cursor.fetchall())
            out.append((f"ent:db:{table}", f"Table {table}: {cols}\n{sample}"))
        conn.close()

    return out


def count_edges(centroids: dict, threshold: float) -> int:
    labels = list(centroids.keys())
    edges = 0
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            if cosine(centroids[la], centroids[lb]) >= threshold:
                edges += 1
    return edges


def run_experiment(embed: BgeM3Provider):
    print("=== Mixed-corpus Record-level density experiment ===\n")

    # Load each corpus.
    pi_path = os.environ.get("LINCLE_PI_REPO", "/tmp/pi-repo")
    novel_path = os.environ.get("LINCLE_NOVEL_PATH", "/tmp/pride_prejudice.txt")
    scenario_path = str(REPO / "experiments" / "m5" / "enterprise_scenario")

    corpora = {}

    # Pure pi code.
    pi = load_pi_code(pi_path, max_files=50)
    corpora["pi_code_only"] = pi

    # Pure novel.
    novel = []
    if Path(novel_path).exists():
        novel = load_novel_chapters(novel_path)
    corpora["novel_only"] = novel

    # Pure enterprise.
    ent = load_enterprise(scenario_path)
    corpora["enterprise_only"] = ent

    # Mixed: pi + novel + enterprise.
    mixed = pi + novel + ent
    corpora["mixed_all"] = mixed

    # Mixed: pi + enterprise (no novel).
    corpora["mixed_code_ent"] = pi + ent

    from sklearn.cluster import HDBSCAN
    import numpy as np

    for name, records in corpora.items():
        if len(records) < 5:
            print(f"  {name}: only {len(records)} records, skipping")
            continue

        ids = [r[0] for r in records]
        texts = [r[1] for r in records]
        vecs = embed.embed(texts)

        # Dispersion.
        mean_dist = mean_pairwise_distance(vecs)

        # Clustering.
        labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(np.array(vecs))
        n_clusters = len(set(labels) - {-1})
        n_noise = sum(1 for l in labels if l == -1)
        noise_pct = n_noise / len(labels) * 100 if labels.any() else 100

        # Cluster centroids.
        cluster_members: dict[int, list] = {}
        for i, label in enumerate(labels):
            if label >= 0:
                cluster_members.setdefault(label, []).append(vecs[i])
        centroids = {}
        for label, members in cluster_members.items():
            dim = len(members[0])
            centroids[label] = [sum(m[d] for m in members) / len(members) for d in range(dim)]

        # Edge density at various thresholds.
        edges_06 = count_edges(centroids, 0.6) if centroids else 0
        edges_07 = count_edges(centroids, 0.7) if centroids else 0
        edges_08 = count_edges(centroids, 0.8) if centroids else 0
        density_06 = edges_06 / max(n_clusters, 1)
        density_07 = edges_07 / max(n_clusters, 1)

        print(f"\n  [{name}] {len(records)} records:")
        print(f"    Mean pairwise distance: {mean_dist:.4f}")
        print(f"    HDBSCAN: {n_clusters} clusters, {n_noise} noise ({noise_pct:.0f}%)")
        if n_clusters > 0:
            print(f"    Edge density (raw cosine):")
            print(f"      ≥0.6: {edges_06:>5d} edges ({density_06:.1f}/cluster)")
            print(f"      ≥0.7: {edges_07:>5d} edges ({density_07:.1f}/cluster)")
            print(f"      ≥0.8: {edges_08:>5d} edges ({edges_08 / max(n_clusters,1):.1f}/cluster)")
            marker = "✓" if 3 <= density_06 <= 10 else ("!" if density_06 < 3 else "✗")
            print(f"    {marker} density at ≥0.6: {density_06:.1f} (target 3-10)")
        else:
            print(f"    No clusters found (all noise)")

    # Cross-domain similarity check.
    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN SIMILARITY CHECK")
    print(f"{'='*60}")
    if pi and novel and ent:
        pi_vecs = embed.embed([r[1] for r in pi[:10]])
        novel_vecs = embed.embed([r[1] for r in novel[:10]])
        ent_vecs = embed.embed([r[1] for r in ent[:5]])

        domains = {"pi_code": pi_vecs, "novel": novel_vecs, "enterprise": ent_vecs}
        print(f"\n  Mean cosine similarity between domains:")
        domain_names = list(domains.keys())
        for i, da in enumerate(domain_names):
            for db in domain_names[i + 1:]:
                sims = [cosine(a, b) for a in domains[da] for b in domains[db]]
                mean_sim = sum(sims) / len(sims) if sims else 0
                print(f"    {da:12s} ↔ {db:12s}: {mean_sim:.4f}")

        print(f"\n  Mean cosine similarity within domains:")
        for name, vecs in domains.items():
            if len(vecs) >= 2:
                sims = [cosine(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i+1, len(vecs))]
                mean_sim = sum(sims) / len(sims) if sims else 0
                print(f"    {name:12s} internal: {mean_sim:.4f}")


def main():
    embed = BgeM3Provider()
    run_experiment(embed)


if __name__ == "__main__":
    main()
