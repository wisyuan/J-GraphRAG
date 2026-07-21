"""Phase 1 — 展开算子 × 范畴构建横向对比（Sub-vector Bake-off）。

回答 Q1：哪种展开算子（零 LLM）最能展开被压缩的精细结构？哪种 C_k 构造配合最好？

两阶段评估：
  阶段 1（快筛）：top-5 最大簇上测每个 (C_k, 算子) 的展开倍率 ratio。
                  ratio >= 1.5 的组合进入阶段 2。
  阶段 2（正裁）：阶段 1 胜出组合在所有 ≥4 成员簇上做 permutation test。
                  p < 0.01 且 ratio >= 1.5 → SUPPORTED。

完全压缩簇（global_dist=0）专项测试：哪些算子能展开它们。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase1_bakeoff --repo /tmp/pi-repo --max-files 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider
from jgraphrag.llm import DeepSeekProvider
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols, mean_pairwise_distance
from experiments.partition import partition_hdbscan, partition_component_activation
from experiments.operators import MATH_OPERATORS, op_llm_reembed

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

SCREEN_THRESHOLD = 1.5   # 阶段 1 筛选门槛（比 Phase 0 的 1.2 严）
ALPHA = 0.01             # permutation test 显著性水平
N_PERMUTATIONS = 100     # permutation test 重排次数（100 足以检测 p<0.01：1/100=0.01）
MIN_CLUSTER_MEMBERS = 4  # 阶段 2 只测 ≥4 成员簇（和 Phase 0 对齐）
TOP_K_SCREEN = 5         # 阶段 1 只测 top-5 最大簇
MAX_STAGE2_CLUSTERS = 30 # 阶段 2 最多测 top-30 最大簇（控计算量；全量可后续 GPU 批量化）
COMPRESSED_THRESHOLD = 1e-6  # global_dist < 此值 = 完全压缩簇（ratio 会溢出，单独报告）


# ── numpy 版 dispersion（permutation test 用，快 100x）────────────────────

def mean_pairwise_distance_np(vecs: np.ndarray) -> float:
    """numpy 向量化的 mean_pairwise_distance，数学等价于 fine_record_dispersion 版。

    mean over pairs of (1 - cosine(v_i, v_j))。
    输入 (n, d)，返回标量。
    """
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    # 归一化后矩阵乘 = 两两 cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # 防除零
    normalized = vecs / norms
    cos_matrix = normalized @ normalized.T  # (n, n)
    # 取上三角（不含对角）的所有 pair
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - cos_matrix[iu]
    return float(distances.mean())


def vecs_to_np(vec_list: list[list[float]]) -> np.ndarray:
    """list[list[float]] → np.ndarray (float64)。"""
    return np.asarray(vec_list, dtype=np.float64)


# ── 单簇评估 ────────────────────────────────────────────────────────────

def evaluate_cluster_math(
    member_vecs: np.ndarray,
    op_fn,
    op_params: dict,
) -> dict:
    """对单个簇跑一个纯数学算子，返回 global_dist / fine_dist / ratio。

    Args:
        member_vecs: (n, 1024) 簇成员向量
        op_fn: 算子函数 (np.ndarray, **params) -> np.ndarray
        op_params: 算子参数
    Returns:
        {"global_dist": float, "fine_dist": float, "ratio": float, "n_members": int}
    """
    global_dist = mean_pairwise_distance_np(member_vecs)
    try:
        sub_vecs = op_fn(member_vecs, **op_params)
        fine_dist = mean_pairwise_distance_np(np.asarray(sub_vecs, dtype=np.float64))
    except Exception as e:
        # 算子可能在退化簇上失败（如 Σ≈0 的马氏变换）——记录但不崩。
        return {"global_dist": global_dist, "fine_dist": None, "ratio": None,
                "n_members": member_vecs.shape[0], "error": str(e)}

    # 完全压缩簇（global_dist ≈ 0）：ratio 会溢出，标记为 compressed 单独报告
    if global_dist < COMPRESSED_THRESHOLD:
        return {"global_dist": global_dist, "fine_dist": fine_dist, "ratio": None,
                "n_members": member_vecs.shape[0], "compressed": True}

    ratio = fine_dist / global_dist
    return {"global_dist": global_dist, "fine_dist": fine_dist, "ratio": ratio,
            "n_members": member_vecs.shape[0], "compressed": False}


def evaluate_cluster_llm(
    member_texts: list[str],
    member_vecs: np.ndarray,
    embed,
    llm,
) -> dict:
    """对单个簇跑 LLM 重嵌入（ceiling），返回 global_dist / fine_dist / ratio。"""
    global_dist = mean_pairwise_distance_np(member_vecs)
    reembedded = op_llm_reembed(member_texts, embed, llm)
    fine_dist = mean_pairwise_distance_np(vecs_to_np(reembedded))
    if global_dist < COMPRESSED_THRESHOLD:
        ratio = None
        compressed = True
    else:
        ratio = fine_dist / global_dist
        compressed = False
    return {"global_dist": global_dist, "fine_dist": fine_dist, "ratio": ratio,
            "compressed": compressed,
            "n_members": len(member_texts)}


# ── 阶段 1：快筛 ────────────────────────────────────────────────────────

def stage1_quick_screen(
    fine_vecs: np.ndarray,
    fine_texts: list[str],
    partitions: dict[str, dict[int, list[int]]],
    embed,
    llm,
) -> dict:
    """阶段 1：top-5 最大簇上测每个 (C_k, 算子) 的 ratio。

    Returns:
        nested dict: {partition_name: {operator_name: {ratios, mean_ratio, passed, details}}}
    """
    results = {}

    for part_name, clusters in partitions.items():
        print(f"\n  [{part_name}] {len(clusters)} clusters")
        results[part_name] = {}

        # 选 top-K 最大簇
        sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
        screen_clusters = [(cid, members) for cid, members in sorted_clusters
                           if len(members) >= MIN_CLUSTER_MEMBERS][:TOP_K_SCREEN]
        print(f"    screening top-{len(screen_clusters)} clusters (≥{MIN_CLUSTER_MEMBERS} members)")

        for op_name, (op_fn, op_params) in MATH_OPERATORS.items():
            per_cluster = []
            for cid, members in screen_clusters:
                member_vecs = fine_vecs[members]
                r = evaluate_cluster_math(member_vecs, op_fn, op_params)
                r["cluster_id"] = cid
                per_cluster.append(r)

            valid_ratios = [r["ratio"] for r in per_cluster if r["ratio"] is not None]
            mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
            passed = mean_ratio is not None and mean_ratio >= SCREEN_THRESHOLD

            results[part_name][op_name] = {
                "ratios_per_cluster": [r.get("ratio") for r in per_cluster],
                "mean_ratio": mean_ratio,
                "passed": passed,
                "n_clusters_tested": len(per_cluster),
                "details": per_cluster,
            }
            marker = "✓ PASS" if passed else "✗"
            print(f"    {op_name:20s} mean_ratio={mean_ratio:.4f} {marker}" if mean_ratio
                  else f"    {op_name:20s} FAILED")

        # LLM ceiling（只在 HDBSCAN 上跑，且需要 LLM provider）
        if part_name == "hdbscan" and llm is not None:
            per_cluster_llm = []
            for cid, members in screen_clusters:
                member_texts = [fine_texts[i] for i in members]
                member_vecs = fine_vecs[members]
                print(f"      LLM re-embed cluster {cid} ({len(members)} members)...", end="", flush=True)
                r = evaluate_cluster_llm(member_texts, member_vecs, embed, llm)
                r["cluster_id"] = cid
                per_cluster_llm.append(r)
                print(f" ratio={r['ratio']:.2f}x")

            valid_ratios = [r["ratio"] for r in per_cluster_llm if r["ratio"] is not None]
            mean_ratio = sum(valid_ratios) / len(valid_ratios) if valid_ratios else None
            results[part_name]["llm_reembed"] = {
                "ratios_per_cluster": [r.get("ratio") for r in per_cluster_llm],
                "mean_ratio": mean_ratio,
                "ceiling": True,
                "details": per_cluster_llm,
            }
            print(f"    {'llm_reembed':20s} mean_ratio={mean_ratio:.4f} (CEILING)" if mean_ratio
                  else f"    {'llm_reembed':20s} FAILED")

    return results


# ── 阶段 2：permutation test ─────────────────────────────────────────────

def stage2_permutation_test(
    fine_vecs: np.ndarray,
    partitions: dict[str, dict[int, list[int]]],
    candidates: list[tuple[str, str]],  # [(partition_name, op_name), ...]
    n_permutations: int = N_PERMUTATIONS,
    alpha: float = ALPHA,
) -> dict:
    """阶段 2：对阶段 1 胜出组合在所有 ≥4 成员簇上做 permutation test。

    流程（对每个候选组合）：
      1. 算真实 ratio_obs = mean over clusters of (fine_dist_k / global_dist_k)
      2. 对 n_permutations 次：打乱范畴成员标签 → 算 ratio_perm
      3. p-value = fraction(ratio_perm >= ratio_obs)
      4. p < alpha 且 ratio_obs >= SCREEN_THRESHOLD → SUPPORTED
    """
    results = {}
    rng = np.random.default_rng(42)

    for part_name, op_name in candidates:
        op_fn, op_params = MATH_OPERATORS[op_name]
        clusters = partitions[part_name]

        # 所有 ≥4 成员簇（限制为 top-N 最大，控计算量）
        eval_clusters = sorted(
            [(cid, members) for cid, members in clusters.items()
             if len(members) >= MIN_CLUSTER_MEMBERS],
            key=lambda x: len(x[1]), reverse=True
        )[:MAX_STAGE2_CLUSTERS]
        if not eval_clusters:
            results[f"{part_name}+{op_name}"] = {"error": "no clusters ≥4 members"}
            continue

        print(f"\n  [{part_name}+{op_name}] {len(eval_clusters)} clusters, "
              f"{n_permutations} permutations...")

        # 真实 ratio
        real_ratios = []
        for cid, members in eval_clusters:
            member_vecs = fine_vecs[members]
            r = evaluate_cluster_math(member_vecs, op_fn, op_params)
            if r["ratio"] is not None:
                real_ratios.append(r["ratio"])
        ratio_obs = sum(real_ratios) / len(real_ratios) if real_ratios else 0.0

        # 收集所有参与簇成员的下标池（用于 permutation 的 null model）
        all_member_indices = sorted(set(idx for _, members in eval_clusters for idx in members))
        member_counts = [len(members) for _, members in eval_clusters]

        # Permutation：保持簇大小不变，随机重分配成员
        perm_ratios = []
        print(f"    permutations:", end=" ", flush=True)
        for perm_i in range(n_permutations):
            if perm_i % 50 == 0 and perm_i > 0:
                print(f"{perm_i}", end=" ", flush=True)
            shuffled = rng.permutation(all_member_indices)
            perm_ratios_one = []
            offset = 0
            for count in member_counts:
                perm_members_idx = shuffled[offset:offset + count]
                offset += count
                perm_member_vecs = fine_vecs[perm_members_idx]
                r = evaluate_cluster_math(perm_member_vecs, op_fn, op_params)
                if r["ratio"] is not None:
                    perm_ratios_one.append(r["ratio"])
            if perm_ratios_one:
                perm_ratios.append(sum(perm_ratios_one) / len(perm_ratios_one))

        # p-value: fraction of permutations >= observed
        if perm_ratios and ratio_obs > 0:
            p_value = sum(1 for pr in perm_ratios if pr >= ratio_obs) / len(perm_ratios)
        else:
            p_value = 1.0

        supported = p_value < alpha and ratio_obs >= SCREEN_THRESHOLD

        results[f"{part_name}+{op_name}"] = {
            "ratio_obs": ratio_obs,
            "p_value": p_value,
            "n_clusters": len(eval_clusters),
            "n_permutations": len(perm_ratios),
            "supported": supported,
            "perm_ratio_mean": sum(perm_ratios) / len(perm_ratios) if perm_ratios else None,
            "perm_ratio_std": float(np.std(perm_ratios)) if perm_ratios else None,
        }
        verdict = "✓ SUPPORTED" if supported else "✗ not supported"
        print(f"    ratio_obs={ratio_obs:.4f}, p={p_value:.4f}, {verdict}")

    return results


# ── 完全压缩簇专项测试 ──────────────────────────────────────────────────

def test_compressed_clusters(
    fine_vecs: np.ndarray,
    fine_texts: list[str],
    partitions: dict[str, dict[int, list[int]]],
    embed,
    llm,
) -> dict:
    """对 global_dist ≈ 0 的簇（Cluster 82 类），单独报告哪些算子能展开它们。"""
    results = {"n_zero_clusters": 0, "by_partition": {}, "compressed_detail": []}

    for part_name, clusters in partitions.items():
        zero_clusters = []
        for cid, members in clusters.items():
            if len(members) < MIN_CLUSTER_MEMBERS:
                continue
            member_vecs = fine_vecs[members]
            global_dist = mean_pairwise_distance_np(member_vecs)
            if global_dist < 1e-6:  # 完全压缩
                zero_clusters.append((cid, members))

        if not zero_clusters:
            continue

        results["n_zero_clusters"] += len(zero_clusters)
        print(f"\n  [{part_name}] {len(zero_clusters)} fully-compressed clusters (global_dist≈0)")

        for cid, members in zero_clusters:
            member_vecs = fine_vecs[members]
            member_texts = [fine_texts[i] for i in members]
            cluster_result = {"partition": part_name, "cluster_id": cid,
                              "n_members": len(members), "global_dist": 0.0}

            # 测每个数学算子
            for op_name, (op_fn, op_params) in MATH_OPERATORS.items():
                r = evaluate_cluster_math(member_vecs, op_fn, op_params)
                cluster_result[f"{op_name}_fine_dist"] = r.get("fine_dist")
                cluster_result[f"{op_name}_error"] = r.get("error")

            # LLM 重嵌入（只 HDBSCAN，且只测第一个这类簇，控成本）
            if part_name == "hdbscan" and llm is not None and not results["compressed_detail"]:
                r = evaluate_cluster_llm(member_texts, member_vecs, embed, llm)
                cluster_result["llm_fine_dist"] = r["fine_dist"]
                cluster_result["llm_ratio"] = r["ratio"]

            results["compressed_detail"].append(cluster_result)

    return results


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_experiment(embed, llm, repo_path: str, max_files: int):
    from sklearn.cluster import HDBSCAN  # noqa: F401 — 提前验证依赖

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    # 切分 FineRecord（复用 Phase 0 的格式）
    fine_texts = []
    fine_ids = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
            fine_ids.append(f"{f.relative_to(repo)}::{sym['name']}")

    n = len(fine_texts)
    print(f"  {n} FineRecords from {len(files)} files")
    print("  embedding (cached)...", end="", flush=True)
    fine_vecs_list = embed.embed(fine_texts)
    print(f" done ({embed.size} cached)")
    fine_vecs = vecs_to_np(fine_vecs_list)

    # 两种范畴构造
    print("  partitioning (HDBSCAN)...", end="", flush=True)
    hdbscan_clusters = partition_hdbscan(fine_vecs_list)
    print(f" {len(hdbscan_clusters)} clusters")

    print("  partitioning (component-activation)...", end="", flush=True)
    ca_clusters = partition_component_activation(fine_vecs_list, n_dims=50, top_fraction=0.2)
    print(f" {len(ca_clusters)} categories")

    partitions = {"hdbscan": hdbscan_clusters, "component_activation": ca_clusters}

    # 阶段 1：快筛
    print("\n=== Stage 1: Quick Screen (top-5 clusters) ===")
    stage1 = stage1_quick_screen(fine_vecs, fine_texts, partitions, embed, llm)

    # 收集阶段 2 候选（阶段 1 passed 的数学算子）
    candidates = []
    for part_name, ops in stage1.items():
        for op_name, res in ops.items():
            if isinstance(res, dict) and res.get("passed") and not res.get("ceiling"):
                candidates.append((part_name, op_name))

    # 阶段 2：permutation test
    print("\n=== Stage 2: Permutation Test ===")
    if candidates:
        stage2 = stage2_permutation_test(fine_vecs, partitions, candidates)
    else:
        print("  No candidates passed stage 1 (ratio >= 1.5). Skipping.")
        stage2 = {}

    # 完全压缩簇专项
    print("\n=== Compressed Clusters ===")
    compressed = test_compressed_clusters(fine_vecs, fine_texts, partitions, embed, llm)

    # 汇总
    result = {
        "n_fine_records": n,
        "n_clusters_hdbscan": len(hdbscan_clusters),
        "n_clusters_component_activation": len(ca_clusters),
        "stage1_screen": stage1,
        "stage2_permutation": stage2,
        "compressed_clusters": compressed,
        "config": {
            "screen_threshold": SCREEN_THRESHOLD,
            "alpha": ALPHA,
            "n_permutations": N_PERMUTATIONS,
            "min_cluster_members": MIN_CLUSTER_MEMBERS,
            "top_k_screen": TOP_K_SCREEN,
        },
    }

    out_path = EXP / "phase1_bakeoff.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")

    # 打印总结
    print("\n=== SUMMARY ===")
    print(f"  FineRecords: {n} | HDBSCAN clusters: {len(hdbscan_clusters)} | "
          f"CA categories: {len(ca_clusters)}")
    print(f"  Stage 1 passed: {len(candidates)} combinations")
    supported = [k for k, v in stage2.items() if v.get("supported")]
    print(f"  Stage 2 SUPPORTED: {supported if supported else 'NONE'}")
    llm_ceil = stage1.get("hdbscan", {}).get("llm_reembed", {}).get("mean_ratio")
    if llm_ceil:
        print(f"  LLM ceiling: {llm_ceil:.2f}x")

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 1: Sub-vector bake-off")
    ap.add_argument("--repo", required=True, help="path to pi repo")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--no-llm", action="store_true", help="skip LLM ceiling (math operators only)")
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    llm = DeepSeekProvider() if not args.no_llm else None

    print(f"Phase 1 Bake-off: {args.repo} (max {args.max_files} files)")
    run_experiment(embed, llm, args.repo, args.max_files)


if __name__ == "__main__":
    main()
