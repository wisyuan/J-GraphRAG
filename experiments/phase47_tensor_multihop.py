"""Phase 47: 张量分析能否为多跳检索指方向、缩解空间（纯 CPU，禁载模型）。

前置结论（本实验出发点）：
  - Phase 46 P3：ws 概念向量空间无线性关系结构（关系类比 ≈ 随机）。
  - Phase 46 P4：W² 矩阵幂判别多跳路径 AUC 0.998，但 W 是压扁的 2D
    邻接，丢失了关系类型信息。
本实验把关系图升级为三阶张量 T[concept, relation_type, concept]，
用 CP 分解（手写 numpy CP-ALS，venv 无 tensorly）检验低秩结构：
  S1 建张量：relation 词 → _stem 归一 + 人工规则桶（6 桶），T[i,r,j]=prob。
  S2 CP 分解 + holdout（核心判决）：20% 边按桶分层 mask，rank∈{3,5,8,12}、
     λ∈{0.1,1}、每配置 ≥5 随机起点取 train 重构误差最小，测试边 vs 等量
     随机负三元组的评分 ROC-AUC。5 次重复（不同 mask 种子）报均值±std。
     AUC>0.7 低秩结构存在；<0.6 当前规模下不成熟（如实记录）。
  S3 路径提案（S2 成立才跑）：真二跳路径 A→B→C（B 度≥2），mask 掉 A→B、
     B→C 两边后 refit，T̂[A,:,·]×T̂[·,:,C] 评分候选中间节点，真 B 的
     recall@5/@10；基线 = 同 mask 下 W² 行 (W²)[A,m]×(W²)[m,C]。
  S4 耦合臂（S2 成立才跑）：ws_vec PCA 降维到 rank 初始化概念因子 /
     耦合正则 ||A-α·ws_pca||²，比较 holdout AUC 变化——检验几何（ws）与
     关系（T）是否共享潜在空间（Phase 40 双射一致性在 P3 阴性后的强化检验）。

复用（只 import 不改）：
  - phase46_vector_native.load_inputs（cache/npz/relations 三件套加载）
  - phase46_vector_native._stem（= phase39_two_pass_cache._stem 词形归并）

运行：
    cd crates/lincle/python
    source .venv/bin/activate
    python -m experiments.phase47_tensor_multihop --probe all
    python -c "import experiments.phase47_tensor_multihop"   # 零副作用
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase47_tensor_multihop.json"

# ── S1 关系词 → 类型桶（人工规则，键为 _stem 归一后的词形） ──────────────
# 248 条边的 relation 词是 LLM 抽取的自由片段，噪声大；按语义归 6 桶。
BUCKET_RULES: dict[str, str] = {
    # treat：治疗/处置/操作类
    "therapeutic": "treat", "treat": "treat", "treated": "treat",
    "surgical": "treat", "procedural": "treat", "performed": "treat",
    "guided": "treat", "protect": "treat", "assessed": "treat",
    "diagnostic": "treat",
    # causal：因果/调控/机制类
    "causal": "causal", "regulated": "causal", "regulatory": "causal",
    "influenced": "causal", "mediated": "causal", "determined": "causal",
    "genetic": "causal", "inherit": "causal", "metabolism": "causal",
    "agini": "causal",
    # structural：解剖/组成/结构类
    "anatom": "structural", "integral": "structural",
    "contained": "structural", "inclusive": "structural",
    "inclu": "structural",  # _stem("inclus") → "inclu"
    "singular": "structural", "plural": "structural",
    "aternity": "structural", "inté": "structural", "impse": "structural",
    "errated": "structural", "complex": "structural",
    "intricate": "structural", "iplinary": "structural",
    "reflected": "structural",
    # importance：重要性/显著性修饰类
    "critical": "importance", "crucial": "importance", "vital": "importance",
    "significant": "importance", "essential": "importance",
    "minimal": "importance", "rarely": "importance", "evident": "importance",
    "closely": "importance",
    # assoc：泛关联/共现类（语义最弱的一桶）
    "indirect": "assoc", "indirectly": "assoc", "often": "assoc",
    "alink": "assoc", "unrelated": "assoc", "inverse": "assoc",
    "sequential": "assoc",
}
BUCKET_ORDER = ["treat", "causal", "structural", "importance", "assoc", "other"]


# ── 数据加载 + 张量构建 ─────────────────────────────────────────────────


def _stem(word: str) -> str:
    """Re-export of phase46._stem (lazy import, unmodified)."""
    from experiments.phase46_vector_native import _stem as stem_impl

    return stem_impl(word)


def load_inputs(domain: str):
    """Re-export of phase46.load_inputs (lazy import, unmodified)."""
    from experiments.phase46_vector_native import load_inputs as impl

    return impl(domain)


def relation_bucket(word: str) -> str:
    """relation 词 → _stem 归一 → 人工规则桶；未命中归 'other'。"""
    return BUCKET_RULES.get(_stem(word.strip().lower()), "other")


def build_tensor(domain: str):
    """S1：T[i,r,j]=prob（i≠j，同 (i,r,j) 多边取 max prob，口径同 P4 的 W）。

    Returns:
        T: (n_concepts, n_buckets, n_concepts) dense tensor
        concepts: 概念轴（sorted，全部 50 个均在 relations 中出现）
        buckets: 关系类型轴
        edges: [(i, r, j, prob)] 去重后条目（下标基于 concepts/buckets 轴）
        W: 压扁 2D 邻接（max prob over buckets，P4 口径，S3 基线用）
        bucket_map: {relation_word: bucket} 映射表
    """
    _cache, _vecs, relations = load_inputs(domain)
    raw = relations["edges"]
    used = sorted({e["concept_a"] for e in raw} | {e["concept_b"] for e in raw})
    cidx = {c: i for i, c in enumerate(used)}

    cell_pre: dict[tuple[str, str, str], float] = {}
    bucket_map: dict[str, str] = {}
    for e in raw:
        a, b, rel = e["concept_a"], e["concept_b"], str(e["relation"])
        if a == b:
            continue
        bucket = relation_bucket(rel)
        bucket_map[rel] = bucket
        key = (a, bucket, b)
        cell_pre[key] = max(cell_pre.get(key, 0.0), float(e["prob"]))

    # 只保留实际出现的桶（避免全零关系切片进入 CP）
    present = {bucket for (_a, bucket, _b) in cell_pre}
    buckets = [b for b in BUCKET_ORDER if b in present]
    bidx = {b: r for r, b in enumerate(buckets)}

    n, nr = len(used), len(buckets)
    T = np.zeros((n, nr, n), dtype=np.float64)
    W = np.zeros((n, n), dtype=np.float64)
    edges = []
    for (a, bucket, b), p in sorted(cell_pre.items()):
        i, r, j = cidx[a], bidx[bucket], cidx[b]
        T[i, r, j] = p
        W[i, j] = max(W[i, j], p)
        edges.append((i, r, j, p))
    return T, used, buckets, edges, W, bucket_map


def probe_s1(domain: str, verbose: bool = True) -> dict:
    """S1：张量规模、密度、每桶边数、桶映射表。"""
    T, concepts, buckets, edges, _W, bucket_map = build_tensor(domain)
    per_bucket = {b: 0 for b in buckets}
    for (_i, r, _j, _p) in edges:
        per_bucket[buckets[r]] += 1
    density = float(np.count_nonzero(T) / T.size)
    out = {
        "n_concepts": len(concepts),
        "n_buckets": len(buckets),
        "buckets": buckets,
        "tensor_shape": list(T.shape),
        "n_edges_raw": 248,
        "n_cells_nonzero": int(np.count_nonzero(T)),
        "density": density,
        "edges_per_bucket": per_bucket,
        "bucket_map": dict(sorted(bucket_map.items())),
        "bucket_rule": "relation 词 → phase39._stem 归一 → 人工规则桶"
        "（treat/causal/structural/importance/assoc），未命中归 other",
    }
    if verbose:
        print(f"  S1 [{domain}] tensor={out['tensor_shape']} "
              f"nonzero={out['n_cells_nonzero']} density={density:.4f}")
        print(f"  S1 edges/bucket: {per_bucket}")
    return out


# ── CP 分解（手写 numpy CP-ALS，dense 小张量） ───────────────────────────


def _khatri_rao(mats: list[np.ndarray]) -> np.ndarray:
    """Column-wise Kronecker product, rows ordered with last factor fastest."""
    out = mats[0]
    for m in mats[1:]:
        out = np.einsum("ir,jr->ijr", out, m).reshape(-1, out.shape[1])
    return out


def cp_reconstruct(factors: list[np.ndarray]) -> np.ndarray:
    """T̂ = Σ_r a_r ∘ b_r ∘ c_r."""
    a, b, c = factors
    return np.einsum("ir,jr,kr->ijk", a, b, c)


def cp_score(factors: list[np.ndarray], i: int, r: int, j: int) -> float:
    """Single-entry reconstruction score T̂[i,r,j]."""
    a, b, c = factors
    return float(np.sum(a[i] * b[r] * c[j]))


def cp_als(
    X: np.ndarray,
    rank: int,
    lam: float = 1.0,
    n_iter: int = 150,
    tol: float = 1e-9,
    seed: int = 0,
    init: list[np.ndarray] | None = None,
    couple: dict[int, tuple[float, np.ndarray]] | None = None,
    mask: np.ndarray | None = None,
) -> tuple[list[np.ndarray], float]:
    """CP-ALS with L2 regularization, optional ws-coupling and obs-mask.

    mask: True = observed entry (fit target); masked-out entries are zeroed
    in the unfoldings and excluded from the reconstruction error.
    couple[mode] = (mu, Z): adds mu*||factor - Z||² to that mode's ALS step.
    Factors of modes 0/1 are column-normalized each sweep (mode 2 carries
    scale) to keep the tiny problem numerically tame.
    Returns (factors, masked reconstruction SSE).
    """
    rng = np.random.default_rng(seed)
    dims = X.shape
    if init is None:
        factors = [rng.standard_normal((d, rank)) * 0.1 for d in dims]
    else:
        factors = [f.copy() for f in init]
    Xm = X * mask if mask is not None else X

    prev = None
    err = float("inf")
    for _it in range(n_iter):
        for mode in range(3):
            others = [m for m in range(3) if m != mode]
            x_unf = np.moveaxis(Xm, mode, 0).reshape(dims[mode], -1)
            kr = _khatri_rao([factors[m] for m in others])
            gram = np.ones((rank, rank))
            for m in others:
                gram *= factors[m].T @ factors[m]
            reg = lam * np.eye(rank)
            rhs = x_unf @ kr
            if couple is not None and mode in couple:
                mu, z = couple[mode]
                reg = reg + mu * np.eye(rank)
                rhs = rhs + mu * z
            factors[mode] = np.linalg.solve(gram + reg, rhs.T).T
            if mode != 2:
                norms = np.linalg.norm(factors[mode], axis=0)
                norms[norms == 0] = 1.0
                factors[mode] = factors[mode] / norms
        diff = cp_reconstruct(factors) - X
        if mask is not None:
            diff = diff * mask
        err = float(np.sum(diff**2))
        if prev is not None and abs(prev - err) <= tol * max(prev, 1e-12):
            break
        prev = err
    return factors, err


def _auc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney ROC-AUC (no sklearn dependency): P(pos > neg) + ½ ties."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def fit_cp_best(
    T: np.ndarray, rank: int, lam: float, n_restarts: int, rng: np.random.Generator,
    mask: np.ndarray | None = None,
    init: list[np.ndarray] | None = None,
    couple: dict[int, tuple[float, np.ndarray]] | None = None,
) -> tuple[list[np.ndarray], float]:
    """Multi-start CP-ALS; returns the restart with min masked recon error.

    When init is given (S4 pca_init arm), restart 0 uses it and the rest are
    random — same selection rule (min error) applies to all.
    """
    best_f, best_err = None, float("inf")
    for rs in range(n_restarts):
        f, err = cp_als(
            T, rank, lam=lam, seed=int(rng.integers(2**31)), mask=mask,
            init=init if rs == 0 else None, couple=couple)
        if err < best_err:
            best_f, best_err = f, err
    return best_f, best_err


# ── S2 CP + holdout（核心判决） ──────────────────────────────────────────


def stratified_holdout(
    edges: list[tuple[int, int, int, float]], frac: float,
    rng: np.random.Generator,
) -> list[int]:
    """按桶分层随机 mask frac 比例的边，返回被 mask 的 edge 下标。"""
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for idx, (_i, r, _j, _p) in enumerate(edges):
        by_bucket[r].append(idx)
    picked: list[int] = []
    for idxs in by_bucket.values():
        k = max(1, round(len(idxs) * frac))
        picked.extend(int(t) for t in rng.permutation(idxs)[:k])
    return sorted(picked)


def sample_negatives(
    T: np.ndarray, n_neg: int, rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """等量随机负三元组：全张量中未观测（任何 mask 状态下都不是边）。"""
    n, nr, _ = T.shape
    negs: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    attempts = 0
    while len(negs) < n_neg and attempts < 100000:
        attempts += 1
        trip = (int(rng.integers(n)), int(rng.integers(nr)),
                int(rng.integers(n)))
        if trip[0] == trip[2] or T[trip] > 0 or trip in seen:
            continue
        seen.add(trip)
        negs.append(trip)
    return negs


def run_holdout_auc(
    domain: str, seed: int, ranks: tuple[int, ...], lams: tuple[float, ...],
    n_restarts: int = 5, n_repeats: int = 5, holdout_frac: float = 0.2,
    arm: str = "random", mus: tuple[float, ...] = (),
    verbose: bool = True,
) -> dict:
    """S2/S4 共用的 holdout AUC 评估。

    arm="random"：S2 主实验，网格 (rank × lam)。
    arm="pca_init"/"coupled"：S4，固定调用方传入的 ranks（单 rank），
      lams（单 λ）；pca_init 用 ws PCA top-rank 初始化概念因子，
      coupled 额外加 ||A-Z||²、||C-Z||² 正则（μ∈mus）。
    """
    T, concepts, buckets, edges, _W, _bm = build_tensor(domain)
    n, nr, _ = T.shape

    z_pca = None
    if arm in ("pca_init", "coupled"):
        _cache, vecs, _rel = load_inputs(domain)
        vidx = {c: k for k, c in enumerate(vecs["concepts"])}
        ws = np.stack([vecs["ws_vec"][vidx[c]] for c in concepts])
        ws_c = ws - ws.mean(axis=0, keepdims=True)
        u, s, _vt = np.linalg.svd(ws_c, full_matrices=False)
        z_full = u * s  # (n, 3584) PCA scores
        z_pca = z_full

    results: dict[str, dict] = {}
    for rank in ranks:
        z = None
        if z_pca is not None:
            z = z_pca[:, :rank].copy()
            z /= np.where(np.linalg.norm(z, axis=0, keepdims=True) > 0,
                          np.linalg.norm(z, axis=0, keepdims=True), 1.0)
        for lam in lams:
            mu_grid = mus if arm == "coupled" else (0.0,)
            for mu in mu_grid:
                aucs: list[float] = []
                for rep in range(n_repeats):
                    rng = np.random.default_rng(seed * 1000 + rep)
                    test_idx = stratified_holdout(edges, holdout_frac, rng)
                    mask = np.ones_like(T, dtype=bool)
                    for t in test_idx:
                        i, r, j, _p = edges[t]
                        mask[i, r, j] = False
                    init = None
                    couple = None
                    if arm == "pca_init":
                        rng_init = np.random.default_rng(seed * 77 + rep)
                        init = [
                            z + 0.01 * rng_init.standard_normal(z.shape),
                            rng_init.standard_normal((nr, rank)) * 0.1,
                            z + 0.01 * rng_init.standard_normal(z.shape),
                        ]
                    elif arm == "coupled":
                        couple = {0: (mu, z), 2: (mu, z)}
                    factors, _err = fit_cp_best(
                        T, rank, lam, n_restarts, rng, mask=mask,
                        init=init, couple=couple)
                    pos = [cp_score(factors, i, r, j)
                           for t in test_idx for (i, r, j, _p) in [edges[t]]]
                    neg_trips = sample_negatives(T, len(pos), rng)
                    neg = [cp_score(factors, i, r, j)
                           for (i, r, j) in neg_trips]
                    auc = _auc(pos, neg)
                    if auc is not None:
                        aucs.append(auc)
                key = f"rank{rank}_lam{lam}"
                if arm == "coupled":
                    key += f"_mu{mu}"
                results[key] = {
                    "mean": float(np.mean(aucs)),
                    "std": float(np.std(aucs)),
                    "aucs": [round(a, 4) for a in aucs],
                    "n_repeats": len(aucs),
                }
                if verbose:
                    print(f"    {key:<22} AUC={results[key]['mean']:.3f}"
                          f"±{results[key]['std']:.3f}  {results[key]['aucs']}")
    return results


def _best_config(results: dict) -> tuple[str, dict]:
    """mean AUC 最高的配置；并列取 key 字典序最小（确定性）。"""
    return max(sorted(results.items()), key=lambda kv: kv[1]["mean"])


def probe_s2(domain: str, seed: int = 0, verbose: bool = True) -> dict:
    """S2 核心判决：CP rank×λ 网格的 holdout AUC（5 mask 种子均值±std）。"""
    ranks = (3, 5, 8, 12)
    lams = (0.1, 1.0)
    # 复杂度账：CP 参数数 = rank*(n+n_buckets+n)，观测数 = 0.8*nonzero
    T, _c, buckets, edges, _W, _bm = build_tensor(domain)
    n_obs_train = round(0.8 * len(edges))
    param_counts = {f"rank{r}": r * (T.shape[0] + T.shape[1] + T.shape[2])
                    for r in ranks}
    if verbose:
        print(f"  S2 [{domain}] train obs≈{n_obs_train} edges; "
              f"CP params/config: {param_counts}")
    results = run_holdout_auc(domain, seed, ranks, lams, verbose=verbose)
    best_key, best = _best_config(results)
    mean_auc = best["mean"]
    if mean_auc > 0.7:
        verdict = "tensor_route_supported"
    elif mean_auc < 0.6:
        verdict = "immature_at_current_scale"
    else:
        verdict = "marginal"
    if verbose:
        print(f"  S2 verdict: {verdict} (best={best_key} AUC={mean_auc:.3f})")
    return {
        "protocol": "20% 边按桶分层 holdout × 5 mask 种子；每配置 5 随机起点"
        "取 train 重构误差最小；测试边 vs 等量随机负三元组评分 ROC-AUC",
        "ranks": list(ranks), "lams": list(lams),
        "n_obs_train_approx": n_obs_train,
        "cp_param_counts": param_counts,
        "results": results,
        "best_config": best_key,
        "best_auc_mean": best["mean"],
        "best_auc_std": best["std"],
        "verdict": verdict,
    }


# ── S3 路径提案：张量 vs 压扁 W² ─────────────────────────────────────────


def find_two_hop_paths(
    edges: list[tuple[int, int, int, float]], n: int, max_paths: int,
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    """真实二跳路径 A→B→C（A≠C，B 总度≥2），去重，超量随机抽样。"""
    adj: dict[int, list[int]] = defaultdict(list)
    degree: dict[int, int] = defaultdict(int)
    for (i, _r, j, _p) in edges:
        adj[i].append(j)
        degree[i] += 1
        degree[j] += 1
    paths: set[tuple[int, int, int]] = set()
    for (i, _r, j, _p) in edges:
        if degree[j] < 2:
            continue
        for k in adj[j]:
            if k != i:
                paths.add((i, j, k))
    paths = sorted(paths)
    if len(paths) > max_paths:
        paths = [paths[t]
                 for t in sorted(rng.permutation(len(paths))[:max_paths])]
    return paths


def probe_s3(domain: str, seed: int, rank: int, lam: float,
             max_paths: int = 60, n_restarts: int = 3,
             verbose: bool = True) -> dict:
    """S3：mask A→B、B→C 两边后 refit，评分候选中间节点 B 的 recall@k。

    张量评分 s_t(m) = (Σ_r1 T̂[A,r1,m]) · (Σ_r2 T̂[m,r2,C])；
    基线 s_w(m) = (W²)[A,m] · (W²)[m,C]（同 mask 下的压扁图）。
    候选集 = 全部概念 \\ {A, C}（真 B 在候选集内）。
    """
    T, concepts, buckets, edges, W, _bm = build_tensor(domain)
    rng = np.random.default_rng(seed + 5000)
    paths = find_two_hop_paths(edges, T.shape[0], max_paths, rng)
    if len(paths) < 20:
        note = f"仅 {len(paths)} 条真二跳路径（<20），结果仅供参考"
    else:
        note = None
    if verbose:
        print(f"  S3 [{domain}] {len(paths)} real 2-hop paths "
              f"(rank={rank} lam={lam}) {note or ''}")

    hits_t5 = hits_t10 = hits_w5 = hits_w10 = 0
    details = []
    for (a, b, c) in paths:
        # mask：抹去 A→B 与 B→C 的全部关系切片，refit
        tm = T.copy()
        tm[a, :, b] = 0.0
        tm[b, :, c] = 0.0
        factors, _err = fit_cp_best(tm, rank, lam, n_restarts, rng)
        t_hat = cp_reconstruct(factors)
        cand = [m for m in range(T.shape[0]) if m not in (a, c)]
        s_t = {m: float(t_hat[a, :, m].sum() * t_hat[m, :, c].sum())
               for m in cand}
        wm = W.copy()
        wm[a, b] = 0.0
        wm[b, c] = 0.0
        w2 = wm @ wm
        s_w = {m: float(w2[a, m] * w2[m, c]) for m in cand}
        rank_t = sorted(cand, key=lambda m: -s_t[m])
        rank_w = sorted(cand, key=lambda m: -s_w[m])
        h_t5, h_t10 = b in rank_t[:5], b in rank_t[:10]
        h_w5, h_w10 = b in rank_w[:5], b in rank_w[:10]
        hits_t5 += h_t5
        hits_t10 += h_t10
        hits_w5 += h_w5
        hits_w10 += h_w10
        details.append({
            "path": [concepts[a], concepts[b], concepts[c]],
            "tensor_top5": [concepts[m] for m in rank_t[:5]],
            "w2_top5": [concepts[m] for m in rank_w[:5]],
            "hit_t5": bool(h_t5), "hit_t10": bool(h_t10),
            "hit_w5": bool(h_w5), "hit_w10": bool(h_w10),
        })
    m = max(len(paths), 1)
    out = {
        "n_paths": len(paths), "note": note, "rank": rank, "lam": lam,
        "mask_protocol": "逐路径抹去 A→B、B→C 全关系切片后 refit CP "
        f"（{n_restarts} 起点取最优）；W² 基线在同 mask 图上计算",
        "tensor": {"recall@5": hits_t5 / m, "recall@10": hits_t10 / m},
        "w2_baseline": {"recall@5": hits_w5 / m, "recall@10": hits_w10 / m},
        "chance": {"recall@5": 5 / (T.shape[0] - 2),
                   "recall@10": 10 / (T.shape[0] - 2)},
        "verdict": ("tensor_better" if hits_t10 > hits_w10 else
                    "w2_better" if hits_w10 > hits_t10 else "tie"),
        "details": details,
    }
    if verbose:
        print(f"  S3 recall@5/@10: tensor={out['tensor']['recall@5']:.3f}/"
              f"{out['tensor']['recall@10']:.3f}  w2="
              f"{out['w2_baseline']['recall@5']:.3f}/"
              f"{out['w2_baseline']['recall@10']:.3f}  chance="
              f"{out['chance']['recall@10']:.3f}  → {out['verdict']}")
    return out


# ── S4 耦合臂：ws 几何与关系张量是否共享潜在空间 ─────────────────────────


def probe_s4(domain: str, seed: int, rank: int, lam: float,
             n_repeats: int = 5, verbose: bool = True) -> dict:
    """S4：random-init vs ws-PCA-init vs 耦合正则（μ∈{0.3,1,3}）holdout AUC。

    固定 S2 最优 (rank, λ)，同样的 5 个 mask 种子；回答：几何（ws）与
    关系（T）是否存在共享潜在空间。
    """
    if verbose:
        print(f"  S4 [{domain}] rank={rank} lam={lam}")
        print("    arm=random（= S2 该配置重跑，同种子）")
    res_random = run_holdout_auc(domain, seed, (rank,), (lam,), arm="random",
                                 verbose=verbose)
    if verbose:
        print("    arm=pca_init（ws PCA top-rank 初始化概念因子）")
    res_pca = run_holdout_auc(domain, seed, (rank,), (lam,), arm="pca_init",
                              verbose=verbose)
    if verbose:
        print("    arm=coupled（||A-α·ws_pca||² 耦合正则, μ 网格）")
    res_coupled = run_holdout_auc(domain, seed, (rank,), (lam,),
                                  arm="coupled", mus=(0.3, 1.0, 3.0),
                                  verbose=verbose)
    key = f"rank{rank}_lam{lam}"
    base_auc = res_random[key]["mean"]
    pca_auc = res_pca[key]["mean"]
    coupled_best_key, coupled_best = _best_config(res_coupled)
    verdict = ("shared_latent_space"
               if max(pca_auc, coupled_best["mean"]) > base_auc + 0.02
               else "no_evidence_of_shared_space")
    out = {
        "rank": rank, "lam": lam,
        "random": res_random,
        "pca_init": res_pca,
        "coupled": res_coupled,
        "delta": {
            "pca_init_minus_random": pca_auc - base_auc,
            "coupled_best_minus_random": coupled_best["mean"] - base_auc,
            "coupled_best_key": coupled_best_key,
        },
        "gain_threshold": 0.02,
        "verdict": verdict,
    }
    if verbose:
        print(f"  S4 verdict: {verdict} (random={base_auc:.3f} "
              f"pca_init={pca_auc:.3f} coupled_best={coupled_best['mean']:.3f}"
              f" @{coupled_best_key})")
    return out


# ── 输出合并 + CLI ──────────────────────────────────────────────────────


def merge_out(domain: str, probe: str, result: dict,
              out_path: Path = OUT_PATH) -> None:
    out = {}
    if out_path.exists():
        out = json.loads(out_path.read_text())
    out.setdefault("method", "phase47_tensor_multihop")
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.setdefault("probes", {}).setdefault(domain, {})[probe] = result
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  saved {probe}/{domain} → {out_path}")


def _parse_config(key: str) -> tuple[int, float]:
    """'rank8_lam1.0' → (8, 1.0)."""
    r_part, l_part = key.split("_")
    return int(r_part[4:]), float(l_part[3:])


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 47 张量多跳探针（纯 CPU）")
    ap.add_argument("--domain", default="medical", choices=["medical"])
    ap.add_argument("--probe", default="all",
                    choices=["s1", "s2", "s3", "s4", "all"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    probes = ["s1", "s2", "s3", "s4"] if args.probe == "all" else [args.probe]
    print(f"Phase 47 tensor multihop: domain={args.domain} probes={probes} "
          f"seed={args.seed}（纯 CPU，无模型加载）")
    t0 = time.time()
    s2_result = None

    def _ensure_s2() -> dict:
        nonlocal s2_result
        if s2_result is None:
            s2_result = probe_s2(args.domain, args.seed)
        return s2_result

    for p in probes:
        print(f"\n── {p.upper()} ──")
        if p == "s1":
            merge_out(args.domain, p, probe_s1(args.domain))
        elif p == "s2":
            s2_result = probe_s2(args.domain, args.seed)
            merge_out(args.domain, p, s2_result)
        elif p == "s3":
            s2r = _ensure_s2()
            if s2r["verdict"] == "immature_at_current_scale":
                skipped = {
                    "skipped": True,
                    "reason": "S2 判决 immature_at_current_scale"
                    f"（best AUC={s2r['best_auc_mean']:.3f}<0.6），"
                    "低秩结构未确立，路径提案无意义，按任务书跳过",
                    "s2_best_config": s2r["best_config"],
                    "s2_best_auc_mean": s2r["best_auc_mean"],
                }
                print(f"  S3 skipped: {skipped['reason']}")
                merge_out(args.domain, p, skipped)
            else:
                rank, lam = _parse_config(s2r["best_config"])
                merge_out(args.domain, p,
                          probe_s3(args.domain, args.seed, rank, lam))
        elif p == "s4":
            s2r = _ensure_s2()
            if s2r["verdict"] == "immature_at_current_scale":
                skipped = {
                    "skipped": True,
                    "reason": "S2 判决 immature_at_current_scale"
                    f"（best AUC={s2r['best_auc_mean']:.3f}<0.6），"
                    "按任务书跳过耦合臂",
                    "s2_best_config": s2r["best_config"],
                    "s2_best_auc_mean": s2r["best_auc_mean"],
                }
                print(f"  S4 skipped: {skipped['reason']}")
                merge_out(args.domain, p, skipped)
            else:
                rank, lam = _parse_config(s2r["best_config"])
                merge_out(args.domain, p,
                          probe_s4(args.domain, args.seed, rank, lam))
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
