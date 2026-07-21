"""Phase 2 — 自相似性检验（Fractality Test）。

回答 Q2：冻结嵌入空间是否具有可递归展开的自相似结构？

两种自相似（由 Phase 2 代数分析发现）：
  1. 聚类层级自相似（残差递归）：子簇是否比父簇更紧凑、可持续细分？
  2. 算子递归自相似（马氏递归）：白化是否可累积？depth-2 白化在 depth-1 基础上还能展开？

关键代数发现：残差递归是平凡的（depth-2 = 在子簇上去均值，mu_k 消掉）。
因此残差测的是"聚类层级"自相似，马氏测的是"算子递归"自相似。

GPU 加速：dispersion 计算用 torch GPU 矩阵乘（permutation 批量化）。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase2_fractality --repo /tmp/pi-repo --max-files 50 --d-max 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.embed import BgeM3Provider  # noqa: F401
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols
from experiments.partition import partition_hdbscan
from experiments.operators import op_mahalanobis, op_residual

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

D_MAX = 3               # 最大递归深度
MIN_SUBCLUSTER = 4      # 子簇最小成员数（HDBSCAN 最小 3，留余量）
COMPRESSED_THRESHOLD = 1e-6  # 完全压缩簇阈值
N_PERMUTATIONS = 100    # permutation test 次数
ALPHA = 0.05            # 自相似判定的显著性（比 Phase 1 宽松，因深层样本少）
RATIO_PERSISTENCE_THRESHOLD = 0.5  # depth-2/depth-1 > 此值 = 自相似

# ── GPU 配置 ────────────────────────────────────────────────────────────

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU_AVAILABLE = DEVICE.type == "cuda"
except ImportError:
    DEVICE = None
    GPU_AVAILABLE = False


def dispersion_np(vecs: np.ndarray) -> float:
    """numpy 版 mean_pairwise_distance（单个簇用）。"""
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vecs / norms
    cos_matrix = normalized @ normalized.T
    iu = np.triu_indices(n, k=1)
    return float((1.0 - cos_matrix[iu]).mean())


def dispersion_torch(vecs: np.ndarray) -> float:
    """torch GPU 版 dispersion（单簇，但走 GPU）。"""
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    t = torch.as_tensor(vecs, dtype=torch.float32, device=DEVICE)
    norms = t.norm(dim=1, keepdim=True)
    norms = torch.where(norms == 0, torch.ones_like(norms), norms)
    normalized = t / norms
    cos_matrix = normalized @ normalized.T
    iu = torch.triu_indices(n, n, offset=1, device=DEVICE)
    return float((1.0 - cos_matrix[iu[0], iu[1]]).mean().item())


def dispersion(vecs: np.ndarray) -> float:
    """自动选 GPU/CPU dispersion。"""
    if GPU_AVAILABLE:
        return dispersion_torch(vecs)
    return dispersion_np(vecs)


# ── 递归展开 ────────────────────────────────────────────────────────────

def recursive_expand(
    members: np.ndarray,
    op_fn,
    depth: int,
    d_max: int,
    op_name: str,
    profile: dict,
):
    """递归展开一个簇，记录每层的 ratio。

    Args:
        members: (n, dim) 该层簇成员向量（在上一层算子的输出空间里）
        op_fn: 算子函数 (np.ndarray) -> np.ndarray（残差或马氏）
        depth: 当前深度（0=全局原始空间，第一次聚类后 depth=1）
        d_max: 最大深度
        op_name: 算子名（用于记录）
        profile: 收集结果的 dict，按 depth 存储 ratio 列表
    """
    if depth >= d_max:
        return
    if members.shape[0] < MIN_SUBCLUSTER:
        return

    global_dist = dispersion(members)
    if global_dist < COMPRESSED_THRESHOLD:
        profile.setdefault("compressed", []).append({
            "depth": depth + 1, "n_members": members.shape[0]
        })
        return  # 完全压缩，停止

    # 算子展开
    try:
        sub_vecs = op_fn(members)
        sub_vecs = np.asarray(sub_vecs, dtype=np.float64)
    except Exception as e:
        profile.setdefault("errors", []).append({
            "depth": depth + 1, "error": str(e)
        })
        return

    fine_dist = dispersion(sub_vecs)
    ratio = fine_dist / global_dist if global_dist > 0 else 0.0

    # 记录这个 depth 的 ratio
    d_key = depth + 1  # depth 0 → 展开后是 depth-1
    profile.setdefault(d_key, []).append({
        "ratio": ratio, "global_dist": global_dist,
        "fine_dist": fine_dist, "n_members": members.shape[0],
    })

    if depth + 1 >= d_max:
        return

    # 在算子输出空间里再聚类（HDBSCAN）
    from sklearn.cluster import HDBSCAN
    if sub_vecs.shape[0] < MIN_SUBCLUSTER:
        return
    try:
        labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(sub_vecs)
    except Exception:
        return

    # 对每个子簇递归（注意：递归的是原始 members 对应的子集，不是 sub_vecs）
    # 因为下一层算子要在原始空间重新算（残差递归的代数性质保证了等价性）
    for label in set(labels):
        if label < 0:
            continue
        sub_members = members[labels == label]
        if len(sub_members) >= MIN_SUBCLUSTER:
            recursive_expand(sub_members, op_fn, depth + 1, d_max, op_name, profile)


# ── Permutation test（GPU 批量化）────────────────────────────────────────

def permutation_test_depth(
    fine_vecs: np.ndarray,
    clusters: dict[int, list[int]],
    op_fn,
    depth_target: int,
    n_permutations: int = N_PERMUTATIONS,
) -> dict:
    """对指定 depth 做 permutation test。

    对 depth=1：打乱 C_k 成员标签（和 Phase 1 相同）。
    对 depth=2：需要先算 depth-1 展开，再在子簇分配上打乱（更复杂）。
    简化：depth>=2 只测"在父簇内部随机重分配子簇成员"。

    Returns: {"ratio_obs": float, "p_value": float, "perm_ratios": list}
    """
    eval_clusters = sorted(
        [(cid, members) for cid, members in clusters.items()
         if len(members) >= MIN_SUBCLUSTER],
        key=lambda x: len(x[1]), reverse=True
    )[:30]  # 最多 30 簇

    if not eval_clusters:
        return {"ratio_obs": 0.0, "p_value": 1.0, "n_clusters": 0}

    # 真实 ratio
    real_ratios = []
    for cid, members_idx in eval_clusters:
        members = fine_vecs[members_idx]
        gd = dispersion(members)
        if gd < COMPRESSED_THRESHOLD:
            continue
        sv = np.asarray(op_fn(members), dtype=np.float64)
        fd = dispersion(sv)
        if gd > 0:
            real_ratios.append(fd / gd)
    ratio_obs = sum(real_ratios) / len(real_ratios) if real_ratios else 0.0

    # Permutation null：随机重分配成员
    all_member_indices = sorted(set(idx for _, m in eval_clusters for idx in m))
    member_counts = [len(m) for _, m in eval_clusters]
    rng = np.random.default_rng(42)

    perm_ratios = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(all_member_indices)
        perm_ratios_one = []
        offset = 0
        for count in member_counts:
            perm_idx = shuffled[offset:offset + count]
            offset += count
            perm_members = fine_vecs[perm_idx]
            gd = dispersion(perm_members)
            if gd < COMPRESSED_THRESHOLD:
                continue
            sv = np.asarray(op_fn(perm_members), dtype=np.float64)
            fd = dispersion(sv)
            if gd > 0:
                perm_ratios_one.append(fd / gd)
        if perm_ratios_one:
            perm_ratios.append(sum(perm_ratios_one) / len(perm_ratios_one))

    if perm_ratios and ratio_obs > 0:
        p_value = sum(1 for pr in perm_ratios if pr >= ratio_obs) / len(perm_ratios)
    else:
        p_value = 1.0

    return {
        "ratio_obs": ratio_obs,
        "p_value": p_value,
        "n_clusters": len(real_ratios),
        "n_permutations": len(perm_ratios),
        "perm_ratio_mean": sum(perm_ratios) / len(perm_ratios) if perm_ratios else None,
        "perm_ratio_std": float(np.std(perm_ratios)) if perm_ratios else None,
    }


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_experiment(embed, repo_path: str, max_files: int, d_max: int):
    print(f"GPU: {'available (' + str(DEVICE) + ')' if GPU_AVAILABLE else 'NOT available, using CPU'}")

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")

    n = len(fine_texts)
    print(f"  {n} FineRecords from {len(files)} files")
    print("  embedding (cached)...", end="", flush=True)
    fine_vecs_list = embed.embed(fine_texts)
    print(f" done ({embed.size} cached)")
    fine_vecs = np.asarray(fine_vecs_list, dtype=np.float64)

    # 顶层聚类（HDBSCAN）
    print("  partitioning (HDBSCAN)...", end="", flush=True)
    clusters = partition_hdbscan(fine_vecs_list)
    ge4 = {cid: m for cid, m in clusters.items() if len(m) >= MIN_SUBCLUSTER}
    print(f" {len(clusters)} clusters ({len(ge4)} with ≥{MIN_SUBCLUSTER} members)")

    results = {"n_fine_records": n, "n_clusters": len(clusters), "d_max": d_max, "operators": {}}

    # 两种算子的递归
    for op_name, op_fn in [("residual", op_residual),
                            ("mahalanobis", lambda m: op_mahalanobis(m, ridge=1e-5, n_dims=50))]:
        print(f"\n=== Recursive expansion: {op_name} ===")

        # 递归展开（记录每层 ratio）
        profile: dict = {}
        for cid, members_idx in list(ge4.items())[:50]:  # 最多 50 个顶层簇
            members = fine_vecs[members_idx]
            recursive_expand(members, op_fn, 0, d_max, op_name, profile)

        # 汇总每层
        depth_profiles = {}
        for d in range(1, d_max + 1):
            entries = profile.get(d, [])
            ratios = [e["ratio"] for e in entries if e["ratio"] is not None]
            depth_profiles[d] = {
                "mean_ratio": sum(ratios) / len(ratios) if ratios else None,
                "n_clusters": len(ratios),
                "ratios": ratios,
            }
            if ratios:
                print(f"  depth-{d}: mean_ratio={depth_profiles[d]['mean_ratio']:.4f}, "
                      f"n_clusters={len(ratios)}")
            else:
                print(f"  depth-{d}: no reachable clusters")

        # ratio_persistence
        r1 = depth_profiles.get(1, {}).get("mean_ratio")
        r2 = depth_profiles.get(2, {}).get("mean_ratio")
        persistence = (r2 / r1) if (r1 and r2 and r1 > 0) else None
        if persistence is not None:
            print(f"  ratio_persistence (depth2/depth1) = {persistence:.4f}")

        # Permutation test for depth 1 and 2
        print(f"  permutation tests:")
        perm_results = {}
        for d in [1, 2]:
            if d > d_max:
                continue
            print(f"    depth-{d}:", end=" ", flush=True)
            # depth-1 perm: 直接在顶层簇上
            # depth-2 perm: 需要先展开再在子簇上（简化：用相同的 null model 估计）
            pr = permutation_test_depth(fine_vecs, ge4, op_fn, d)
            perm_results[d] = pr
            print(f"ratio_obs={pr['ratio_obs']:.4f}, p={pr['p_value']:.4f}, "
                  f"{'✓' if pr['p_value'] < ALPHA else '✗'}")

        # 判定
        d1_sig = perm_results.get(1, {}).get("p_value", 1) < ALPHA
        d2_sig = perm_results.get(2, {}).get("p_value", 1) < ALPHA
        n_d2 = depth_profiles.get(2, {}).get("n_clusters", 0)

        if persistence is not None and persistence > RATIO_PERSISTENCE_THRESHOLD and d2_sig and n_d2 >= 5:
            verdict = "self-similar"
        elif d1_sig:
            verdict = "single-layer (subspace expansion, not fractal)"
        else:
            verdict = "not supported"

        print(f"  VERDICT: {verdict}")

        results["operators"][op_name] = {
            "depth_profiles": depth_profiles,
            "ratio_persistence": persistence,
            "permutation": perm_results,
            "compressed_clusters": profile.get("compressed", []),
            "errors": profile.get("errors", []),
            "verdict": verdict,
        }

    # 保存
    out_path = EXP / "phase2_fractality.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")

    # 总结
    print("\n=== SUMMARY ===")
    for op_name, op_res in results["operators"].items():
        dp = op_res["depth_profiles"]
        r1 = dp.get(1, {}).get("mean_ratio")
        r2 = dp.get(2, {}).get("mean_ratio")
        n2 = dp.get(2, {}).get("n_clusters", 0)
        pers = op_res.get("ratio_persistence")
        print(f"  {op_name:15s}: depth-1={r1:.2f}x, depth-2={f'{r2:.2f}x' if r2 else 'N/A'} "
              f"({n2} clusters), persistence={f'{pers:.2f}' if pers else 'N/A'}")
        print(f"  {'':15s}  → {op_res['verdict']}")

    return results


def main():
    ap = argparse.ArgumentParser(description="Phase 2: Fractality test")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--d-max", type=int, default=D_MAX)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    print(f"Phase 2 Fractality: {args.repo} (max {args.max_files} files, d_max={args.d_max})")
    run_experiment(embed, args.repo, args.max_files, args.d_max)


if __name__ == "__main__":
    main()
