"""Phase 49: novel sim 臂发散排查（纯分析，无 GPU）。

问题：Phase 41A novel 上 matrix_sim_ws/wu vs graph_flat 的 top-10 重叠
0.96 但 Kendall τ 仅 0.26（medical 同臂 τ 0.78 达标）。top-k 几乎一致
而全候选排序相关崩塌——排查原因，判定良性还是恶性。

逐假设检验（同一查询集 = phase41 的 48 个 GraphRAG-Bench 问题，
sim_alpha=0.5，两臂得分按 phase41 口径重算）：
  a. 长尾平局：τ 对平局敏感而 top-10 不敏感。统计两端得分在全体
     chunk 上的零分比例、正值区的平局比例；复算 top-20/top-50 重叠率、
     top-50 并集上的 τ、以及仅双端正分候选上的 τ。
  b. 概念摊薄：novel 244 概念 vs medical 50——sim 扩展 α·S@q 把概率
     质量摊到更多概念，chunk 得分的有效动态范围被压缩。统计 q_seed/q2
     的熵、扩展质量比、chunk 正分的动态范围（max/median、min/max、
     唯一值数）。
  c. S 矩阵致密化：大词表随机余弦噪声更高，α·S@q 把噪声注入。
     比较两域 S（行归一化前的概念余弦）的非零率、平均非对角余弦、
     行参与率（participation ratio）。

判决：top-k（10/20/50）重叠高且头部（top-50 并集）τ 与 medical 相当 →
良性（仅长尾排序噪声，不影响 top-k 检索使用）；头部之外排序也错 → 恶性。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.phase49_novel_sim_diagnosis
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase41_retrieval_equivalence import (
    EXP, SEED_K, TOP_K,
    RetrievalBase, load_corpus_texts, load_phase41_inputs, load_questions,
    merge_ranking, propagate_scored, scores_to_dict, topk_overlap,
)
from experiments.embed_cache import CachedBgeM3Provider

SIM_ALPHA = 0.5  # phase41 default
TOPK_LEVELS = (10, 20, 50)
HEAD_K = 50  # "head" = union of both sides' top-50


# ── Per-query score recomputation (phase41 semantics) ──────────────────


def arm_scores(base: RetrievalBase, qv: np.ndarray, seed_ids: list[str],
               which: str = "ws_vec") -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    """Recompute matrix_sim (q2 @ M) and graph_flat scores, phase41-style.

    Returns (m_sim, g_flat, q_seed, q2); chunk scores exclude seed chunks.
    """
    q_seed = base.seed_indicator_q(seed_ids)
    s = base.concept_sim(which)
    q2 = q_seed + SIM_ALPHA * (s @ q_seed)
    exclude = set(seed_ids)
    m_sim = scores_to_dict(q2 @ base.m, base.chunk_ids, exclude)
    g_flat = propagate_scored(base.graph, seed_ids)
    return m_sim, g_flat, q_seed, q2


# ── Hypothesis probes ──────────────────────────────────────────────────


def _tau(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(kendalltau(x, y).statistic)


def tail_tie_stats(
    m_sim: dict[str, float], g_flat: dict[str, float], chunk_ids: list[str],
    exclude: set[str],
) -> dict:
    """Hypothesis a: zero mass + ties on the full chunk axis, τ variants."""
    ids = [cid for cid in chunk_ids if cid not in exclude]
    mx = np.array([m_sim.get(cid, 0.0) for cid in ids])
    gx = np.array([g_flat.get(cid, 0.0) for cid in ids])

    def side(v: np.ndarray) -> dict:
        pos = v[v > 0]
        _, counts = np.unique(pos, return_counts=True)
        return {
            "zero_frac": round(float((v == 0).mean()), 4),
            "n_positive": int(pos.size),
            "pos_tie_frac": (round(float((counts > 1).mean()), 4)
                             if counts.size else None),
            "max_tie_run": int(counts.max()) if counts.size else 0,
            "n_unique_pos": int(counts.size),
        }

    # τ variants
    tau_full = _tau(mx, gx)
    top_m = set(np.array(ids)[np.argsort(-mx)[:HEAD_K]])
    top_g = set(np.array(ids)[np.argsort(-gx)[:HEAD_K]])
    head = sorted(top_m | top_g)
    tau_head = _tau(np.array([m_sim.get(c, 0.0) for c in head]),
                    np.array([g_flat.get(c, 0.0) for c in head]))
    both = sorted(set(m_sim) & set(g_flat))
    tau_both_pos = _tau(np.array([m_sim[c] for c in both]),
                        np.array([g_flat[c] for c in both]))
    return {
        "matrix": side(mx),
        "graph": side(gx),
        "tau_full_union": tau_full,
        "tau_head50_union": tau_head,
        "tau_both_positive": tau_both_pos,
        "n_both_positive": len(both),
    }


def thinning_stats(q_seed: np.ndarray, q2: np.ndarray,
                   m_sim: dict[str, float]) -> dict:
    """Hypothesis b: expansion dilutes score mass; chunk-score dynamic range."""

    def ent(v: np.ndarray) -> float:
        p = v[v > 0] / v.sum() if v.sum() > 0 else v
        p = p[p > 0]
        return float(-(p * np.log(p)).sum()) if p.size else 0.0

    pos = np.array(sorted(m_sim.values()), dtype=np.float64)
    dyn = None
    if pos.size:
        dyn = {
            "max": float(pos[-1]),
            "median": float(np.median(pos)),
            "max_over_median": round(float(pos[-1] / np.median(pos)), 3)
            if np.median(pos) > 0 else None,
            "min_over_max": round(float(pos[0] / pos[-1]), 6),
        }
    return {
        "n_seed_concepts": int((q_seed > 0).sum()),
        "n_q2_concepts": int((q2 > 0).sum()),
        "expansion_mass_ratio": round(float((q2.sum() - q_seed.sum())
                                            / q_seed.sum()), 4)
        if q_seed.sum() > 0 else None,
        "entropy_q_seed": round(ent(q_seed), 4),
        "entropy_q2": round(ent(q2), 4),
        "chunk_pos_dynamic_range": dyn,
    }


def s_matrix_stats(base: RetrievalBase, which: str = "ws_vec") -> dict:
    """Hypothesis c: raw concept-cosine density/noise + row participation."""
    mat = base.vecs[which]
    npz_index = {c: i for i, c in enumerate(base.vecs["concepts"])}
    rows = [npz_index[c] for c in base.concepts if c in npz_index]
    sub = mat[rows]
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    sub = sub / np.where(norms > 0, norms, 1.0)
    sim = sub @ sub.T
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    off = sim[np.triu_indices(n, k=1)]
    pos = off[off > 0]
    # participation ratio of |row|² distribution: (Σs²)²/Σs⁴ per row, averaged
    s2 = sim ** 2
    pr = (s2.sum(axis=1) ** 2) / np.where((s2 ** 2).sum(axis=1) > 0,
                                          (s2 ** 2).sum(axis=1), 1.0)
    return {
        "n_concepts": n,
        "offdiag_pos_frac": round(float((off > 0).mean()), 4),
        "offdiag_mean": round(float(off.mean()), 5),
        "offdiag_pos_mean": round(float(pos.mean()), 5) if pos.size else None,
        "offdiag_pos_p50": round(float(np.median(pos)), 5) if pos.size else None,
        "offdiag_pos_p99": round(float(np.quantile(pos, 0.99)), 5)
        if pos.size else None,
        "row_participation_ratio_mean": round(float(pr.mean()), 2),
        "row_participation_ratio_frac": round(float(pr.mean() / n), 4),
    }


# ── Driver ──────────────────────────────────────────────────────────────


def diagnose_domain(domain: str, max_queries: int = 50,
                    verbose: bool = True) -> dict:
    cache, vecs, relations = load_phase41_inputs(domain)
    corpus = load_corpus_texts(domain, cache)
    questions = load_questions(domain, max_queries)
    embed_fn = CachedBgeM3Provider().embed
    base = RetrievalBase(cache, vecs, relations, corpus, embed_fn)
    if verbose:
        print(f"  [{domain}] {len(base.chunk_ids)} chunks, "
              f"{len(base.concepts)} concepts, {len(questions)} queries")

    query_emb = np.asarray(embed_fn([q["question"] for q in questions]),
                           dtype=np.float64)

    per_query = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi]
        seed_ids, b0_ids = base.seed_and_b0(qv)
        m_sim, g_flat, q_seed, q2 = arm_scores(base, qv, seed_ids)
        m_rank = merge_ranking(seed_ids, m_sim, b0_ids, top_k=max(TOPK_LEVELS))
        g_rank = merge_ranking(seed_ids, g_flat, b0_ids, top_k=max(TOPK_LEVELS))
        rec = {
            "qid": q.get("id", str(qi)),
            "topk_overlap": {str(k): round(topk_overlap(m_rank, g_rank, k), 4)
                             for k in TOPK_LEVELS},
            **tail_tie_stats(m_sim, g_flat, base.chunk_ids, set(seed_ids)),
            "thinning": thinning_stats(q_seed, q2, m_sim),
        }
        per_query.append(rec)
        if verbose and (qi + 1) % 12 == 0:
            print(f"    {qi + 1}/{len(questions)}", flush=True)

    def agg(key_path, filt=None):
        vals = []
        for r in per_query:
            v = r
            for k in key_path:
                if v is None:
                    break
                v = v[k]
            if v is not None and (filt is None or filt(r)):
                vals.append(v)
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "topk_overlap": {str(k): agg(("topk_overlap", str(k)))
                         for k in TOPK_LEVELS},
        "tau_full_union": agg(("tau_full_union",)),
        "tau_head50_union": agg(("tau_head50_union",)),
        "tau_both_positive": agg(("tau_both_positive",)),
        "zero_frac": {"matrix": agg(("matrix", "zero_frac")),
                      "graph": agg(("graph", "zero_frac"))},
        "n_positive": {"matrix": agg(("matrix", "n_positive")),
                       "graph": agg(("graph", "n_positive"))},
        "pos_tie_frac": {"matrix": agg(("matrix", "pos_tie_frac")),
                         "graph": agg(("graph", "pos_tie_frac")),
                         "graph_only_tied_queries": sum(
                             1 for r in per_query
                             if (r["graph"]["pos_tie_frac"] or 0) > 0.5)},
        "max_tie_run": {"matrix": agg(("matrix", "max_tie_run")),
                        "graph": agg(("graph", "max_tie_run"))},
        "n_both_positive": agg(("n_both_positive",)),
        "thinning": {
            "n_seed_concepts": agg(("thinning", "n_seed_concepts")),
            "n_q2_concepts": agg(("thinning", "n_q2_concepts")),
            "expansion_mass_ratio": agg(("thinning", "expansion_mass_ratio")),
            "entropy_q_seed": agg(("thinning", "entropy_q_seed")),
            "entropy_q2": agg(("thinning", "entropy_q2")),
            "chunk_max_over_median": agg(
                ("thinning", "chunk_pos_dynamic_range", "max_over_median")),
            "chunk_min_over_max": agg(
                ("thinning", "chunk_pos_dynamic_range", "min_over_max")),
        },
        "s_matrix": s_matrix_stats(base),
    }
    return {
        "n_chunks": len(base.chunk_ids),
        "n_concepts": len(base.concepts),
        "n_queries": len(per_query),
        "summary": summary,
        "per_query": per_query,
    }


def verdict(novel: dict, medical: dict) -> dict:
    """Benign vs malignant from head/top-k agreement vs tail τ collapse."""
    ns, ms = novel["summary"], medical["summary"]
    head_ok = (ns["tau_head50_union"] is not None
               and ns["tau_head50_union"] >= 0.5)
    topk_ok = all(ns["topk_overlap"][str(k)] >= 0.7 for k in TOPK_LEVELS)
    driver = []
    if (ns["zero_frac"]["graph"] or 0) > 0.8:
        driver.append(f"graph 零分率 {ns['zero_frac']['graph']:.2f} "
                      f"(medical {ms['zero_frac']['graph']:.2f})")
    if (ns["pos_tie_frac"]["graph"] or 0) > 0.5:
        driver.append(f"graph 正值平局率 {ns['pos_tie_frac']['graph']:.2f} "
                      f"(medical {ms['pos_tie_frac']['graph']:.2f})")
    if (ns["pos_tie_frac"]["matrix"] or 0) > 0.5:
        driver.append(f"matrix 正值平局率 {ns['pos_tie_frac']['matrix']:.2f}")
    if head_ok and topk_ok:
        verdict_str = (
            "良性：top-10/20/50 重叠均 ≥0.7 且头部（top-50 并集）τ ≥0.5，"
            "τ 崩塌由长尾零分/平局主导，不影响 top-k 检索使用")
    else:
        verdict_str = (
            "恶性：头部排序也不一致（top-k 重叠或 head τ 不达标），"
            "sim 臂与图扩散在 novel 上不是同一检索器")
    return {
        "verdict": verdict_str,
        "head_tau_novel": ns["tau_head50_union"],
        "head_tau_medical": ms["tau_head50_union"],
        "topk_overlap_novel": ns["topk_overlap"],
        "tail_drivers": driver,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 49: novel sim-arm diagnosis")
    ap.add_argument("--max-queries", type=int, default=50)
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = {}
    for domain in ("novel", "medical"):
        print(f"\n{'=' * 60}\nPhase 49 [{domain}]\n{'=' * 60}")
        results[domain] = diagnose_domain(domain, args.max_queries)
        s = results[domain]["summary"]
        print(f"  top-k overlap: {s['topk_overlap']}")
        print(f"  τ full={s['tau_full_union']}  head50={s['tau_head50_union']}  "
              f"both-pos={s['tau_both_positive']}")
        print(f"  zero-frac M/G: {s['zero_frac']['matrix']}/"
              f"{s['zero_frac']['graph']}  tie-frac M/G: "
              f"{s['pos_tie_frac']['matrix']}/{s['pos_tie_frac']['graph']}")

    v = verdict(results["novel"], results["medical"])
    print(f"\nVERDICT: {v['verdict']}")
    if v["tail_drivers"]:
        print(f"  tail drivers: {'; '.join(v['tail_drivers'])}")

    out = {
        "method": "phase49_novel_sim_diagnosis",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {"sim_alpha": SIM_ALPHA, "topk_levels": list(TOPK_LEVELS),
                   "head_k": HEAD_K, "seed_k": SEED_K, "top_k": TOP_K,
                   "max_queries": args.max_queries},
        "question": ("phase41 novel matrix_sim_ws vs graph_flat: top-10 "
                     "overlap 0.96 but Kendall τ 0.26 (medical τ 0.78)"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "verdict": v,
        "domains": results,
    }
    out_path = EXP / "phase49_novel_sim_diagnosis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
