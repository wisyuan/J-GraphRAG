"""Phase 3 — 概念树构建与图质量（4 领域跨域验证）。

回答 Q3：多尺度概念树的图质量是否达标？Sigmoid 参数能否统计化？

三个子任务：
  1. 构建多层级概念树（残差+HDBSCAN 递归）
  2. 评估图质量（密度/相干/分离）
  3. ★ Sigmoid 参数统计化（bootstrap CI，遵循架构 §5.2）

4 个领域跨域验证：
  - pi（TypeScript 代码）
  - enterprise（业务文档/SQL/CSV）
  - nfcorpus（医学营养，BEIR）
  - scifact（科学论文，BEIR）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase3_tree_quality --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.corpus_loader import load_all_corpora
from experiments.concept_tree_builder import build_concept_tree, build_sigmoid_graph
from experiments.graph_stats import bootstrap_theta, graph_quality, cluster_coherence_separation
from experiments.partition import partition_hdbscan

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

D_MAX = 2  # 概念树深度（depth 0=顶层，1=子层；Phase 2 显示 depth-2 子簇已很少）


def evaluate_corpus(name: str, records: list[tuple[str, str]], embed) -> dict:
    """对一个语料：构建树 + bootstrap θ + 图质量。"""
    n = len(records)
    print(f"\n{'='*60}")
    print(f"Domain: {name} ({n} fine records)")

    if n < 10:
        print(f"  SKIP: too few records ({n})")
        return {"n_records": n, "skipped": "too few records"}

    # 嵌入（缓存）
    texts = [t for _, t in records]
    ids = [rid for rid, _ in records]
    print(f"  embedding...", end="", flush=True)
    vecs_list = embed.embed(texts)
    print(f" done ({embed.size} cached)")
    vecs = np.asarray(vecs_list, dtype=np.float64)

    # 构建多层级概念树
    print(f"  building concept tree (d_max={D_MAX})...", end="", flush=True)
    tree = build_concept_tree(vecs, d_max=D_MAX)
    depth_dist = tree.depth_distribution()
    print(f" {len(tree.all_nodes)} nodes, depth dist: {depth_dist}")

    # 顶层聚类标签（用于 coherence/separation 和 bootstrap null）
    clusters = partition_hdbscan(vecs_list)
    labels = np.full(n, -1, dtype=int)
    for cid, members in clusters.items():
        for m in members:
            labels[m] = cid

    # Bootstrap 统计化 θ
    top_nodes = tree.nodes_at_depth(0)
    if len(top_nodes) < 2:
        print(f"  SKIP graph analysis: only {len(top_nodes)} top-level clusters (need ≥2)")
        return {
            "n_records": n, "tree": {
                "n_nodes": len(tree.all_nodes),
                "depth_distribution": tree.depth_distribution(),
                "n_compressed": sum(1 for nd in tree.all_nodes if nd.is_compressed),
            },
            "skipped": f"too few clusters ({len(top_nodes)})",
        }

    centroids = [node.centroid for node in top_nodes]
    print(f"  bootstrap theta ({len(centroids)} top-level centroids)...", end="", flush=True)
    theta_result = bootstrap_theta(centroids, vecs, labels, alpha=0.05, n_bootstrap=1000)
    theta = theta_result["theta"]
    print(f" θ={theta:.4f} (vs fixed 0.75, d_real_mean={theta_result.get('d_real_mean', 0):.4f})")

    # 用统计化 θ 构建 Sigmoid 图
    graph = build_sigmoid_graph(top_nodes, theta=theta)
    quality = graph_quality(graph, len(top_nodes))
    print(f"  graph quality (θ={theta:.3f}): {quality['edges_per_cluster']:.1f} edges/cluster, "
          f"{quality['connected_components']} components, coverage={quality['coverage']:.2f}")

    # 对比固定 θ=0.75
    graph_fixed = build_sigmoid_graph(top_nodes, theta=0.75)
    quality_fixed = graph_quality(graph_fixed, len(top_nodes))
    print(f"  graph quality (θ=0.750): {quality_fixed['edges_per_cluster']:.1f} edges/cluster")

    # 簇内相干 vs 簇间分离
    coh_sep = cluster_coherence_separation(vecs, labels)
    print(f"  coherence: within={coh_sep['within_cluster_dispersion']:.4f}, "
          f"between={coh_sep['between_cluster_dispersion']:.4f}, "
          f"silhouette={coh_sep['silhouette_proxy']:.4f}")

    # 子层图质量（如果有子簇）
    sub_quality = None
    sub_nodes = tree.nodes_at_depth(1)
    if len(sub_nodes) >= 2:
        sub_centroids = [node.centroid for node in sub_nodes]
        sub_theta = bootstrap_theta(sub_centroids, vecs, labels, alpha=0.05, n_bootstrap=500)
        sub_graph = build_sigmoid_graph(sub_nodes, theta=sub_theta["theta"])
        sub_quality = graph_quality(sub_graph, len(sub_nodes))
        print(f"  sub-level ({len(sub_nodes)} nodes): {sub_quality['edges_per_cluster']:.1f} edges/cluster")

    return {
        "n_records": n,
        "tree": {
            "n_nodes": len(tree.all_nodes),
            "depth_distribution": depth_dist,
            "n_compressed": sum(1 for nd in tree.all_nodes if nd.is_compressed),
        },
        "bootstrap_theta": theta_result,
        "graph_quality_statistical_theta": quality,
        "graph_quality_fixed_075": quality_fixed,
        "coherence_separation": coh_sep,
        "sub_level_quality": sub_quality,
    }


def run_experiment(embed, repo_path: str, max_pi_files: int = 50, max_beir_docs: int = 1000):
    print("Loading 4-domain corpora...")
    corpora = load_all_corpora(repo_path, max_pi_files, max_beir_docs)

    results = {"corpora": {}}
    for name, records in corpora.items():
        results["corpora"][name] = evaluate_corpus(name, records, embed)

    # 跨域 θ 稳定性分析
    print(f"\n{'='*60}")
    print("=== Cross-domain theta stability ===")
    thetas = {}
    for name, res in results["corpora"].items():
        if "bootstrap_theta" in res and "theta" in res["bootstrap_theta"]:
            thetas[name] = res["bootstrap_theta"]["theta"]
            print(f"  {name:15s}: θ = {thetas[name]:.4f}")
    if thetas:
        theta_values = list(thetas.values())
        print(f"  mean θ = {np.mean(theta_values):.4f}, std = {np.std(theta_values):.4f}")
        print(f"  range = [{min(theta_values):.4f}, {max(theta_values):.4f}]")
        results["cross_domain_theta"] = {
            "per_domain": thetas,
            "mean": float(np.mean(theta_values)),
            "std": float(np.std(theta_values)),
            "stable": float(np.std(theta_values)) < 0.1,  # std<0.1 = 可全局统一
        }

    # 保存
    out_path = EXP / "phase3_tree_quality.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Phase 3: Tree quality + statistical theta")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-pi-files", type=int, default=50)
    ap.add_argument("--max-beir-docs", type=int, default=1000)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    print(f"Phase 3 Tree Quality: 4 domains, d_max={D_MAX}")
    run_experiment(embed, args.repo, args.max_pi_files, args.max_beir_docs)


if __name__ == "__main__":
    main()
