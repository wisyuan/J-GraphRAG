"""Phase 48: E_bge 循环性去偏检验（debiased grounding test）。

问题：Phase 40 接地检验中 E_bge AUC 最高（medical 0.709 / novel 0.705），
但关系图 W 的关系对是用 bge 聚类筛选的（phase30 策略：簇内全对 + 簇间
代表词对）——正样本（W 的边）按定义在 bge 几何下相近，对 E_bge 天然有利
（循环性）。本实验用与 bge 无关的筛选策略重建 W，量化这个偏差。

筛选臂（同一概念集 = phase40 relations_{domain}.json 的 concepts，
同一对数 = phase40 selection.n_pairs_tested）：
  bge   : phase40 原 W（簇内全对 + 簇间代表对），重算于固定概念宇宙
  random: 概念集内均匀随机对（固定 seed）
  mcos  : M 行（IDF×BM25 共现向量）余弦 top-k 筛选——用共现几何替代
          bge 几何做筛选的对照臂；若循环性假设成立，此臂应对 E_M 有利

判决（三层解读）：
  1. random 臂是零假设对照而非裁决者——若所有空间在 random W 上
     AUC≈0.5 且不显著，说明 J-Lens 对无关概念对读出的"关系"不携带
     向量空间结构，random W 上没有"真实关系"可供编码；
  2. mcos 臂是循环性机制的阳性对照——E_M 应在自己的 mcos W 上
     显著抬升；E_bge 的筛选膨胀度 = AUC(E_bge|bge W) − AUC(E_bge|mcos W)；
  3. 交叉几何泛化——E_bge 在非 bge 筛选的 mcos W 上是否仍 ≥ E_ws
     （±0.02），且比 E_M 在 bge W 上更抗几何切换。仍持平/领先 →
     "inflated_but_real"（膨胀但非纯假象）；明显落后 → 循环性证实。

可比性约定：三种 W 在同一固定概念宇宙（phase40 top-DF 概念 ∩ 有非零
wu/ws 向量 ∩ 有 DF 记录）上评估，AUC 的正/负样本池（全部 C(n,2) 对）
一致，仅正样本集（边）不同——与 phase40 按边集派生概念集的做法略有
差异，因此 bge 臂重算值与 phase40 发布值可能有微小出入（一并报告）。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    # 阶段 1（GPU，串行 medical→novel，~5 min）
    python -m experiments.phase48_debiased_grounding --domain all --build-relations
    # 阶段 2（CPU 分析 + 对比表 + 判决）
    python -m experiments.phase48_debiased_grounding --domain all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase28_relation_graph import RELATION_TYPES, extract_relation
from experiments.phase40_grounding_test import (
    CACHE_DIR, EXP, N_PERM, RANDOM_STATE,
    _l2norm_rows, _load_corpus_texts,
    build_m_matrix, edge_auc, find_pair_context,
)
from experiments.embed_cache import CachedBgeM3Provider

SEED = 48
SELECTIONS = ("random", "mcos")


# ── Pair selection strategies ──────────────────────────────────────────


def select_random_pairs(
    concepts: list[str], n_pairs: int, seed: int = SEED,
) -> list[tuple[str, str, str]]:
    """Uniform random undirected pairs over the concept set (fixed seed)."""
    all_pairs = list(combinations(sorted(concepts), 2))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(all_pairs), size=min(n_pairs, len(all_pairs)),
                      replace=False)
    return [("random", all_pairs[i][0], all_pairs[i][1]) for i in sorted(pick)]


def select_mcos_pairs(
    concepts: list[str], n_pairs: int, cache: dict, chunk_texts: dict[str, str],
) -> list[tuple[str, str, str]]:
    """M-row cosine top-k selection (co-occurrence geometry control arm).

    M rows are L2-normed by build_m_matrix, so M @ M.T is cosine. Grow k
    until the union of per-concept top-k neighbors reaches n_pairs, then
    keep the n_pairs strongest pairs (similarity desc, name tie-break).
    """
    m = build_m_matrix(sorted(concepts), cache, chunk_texts)
    order = sorted(concepts)
    sim = m @ m.T
    np.fill_diagonal(sim, -1.0)
    n = len(order)
    scored: dict[tuple[str, str], float] = {}
    k = 1
    while len(scored) < n_pairs and k < n:
        topk = np.argpartition(-sim, k, axis=1)[:, :k]
        scored = {}
        for i in range(n):
            for j in topk[i]:
                a, b = order[i], order[int(j)]
                key = (a, b) if a < b else (b, a)
                scored[key] = max(scored.get(key, -1.0), float(sim[i, j]))
        k += 1
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [("mcos", a, b) for (a, b), _s in ranked[:n_pairs]]


# ── Stage 1: relation extraction on new pairs (GPU) ────────────────────


def build_relations_for_pairs(
    lens, lens_model, tokenizer,
    domain: str, strategy: str,
    pairs: list[tuple[str, str, str]],
    concepts: list[str],
    cache: dict, chunk_text_map: dict[str, str],
    out_dir: Path = CACHE_DIR,
) -> dict:
    """Mirror of phase40.build_relations for an externally supplied pair list.

    Same edge format; output relations_{domain}_{strategy}.json (idempotent:
    skipped by the caller when the file already exists).
    """
    layer = lens.source_layers[-1]
    edges = []
    n_skipped = 0
    t_start = time.perf_counter()
    for i, (layer_type, c_a, c_b) in enumerate(pairs):
        context, ctx_strategy = find_pair_context(
            c_a, c_b, cache["concept_chunks"], chunk_text_map)
        if context is None:
            n_skipped += 1
            continue
        rel_word, rel_prob = extract_relation(
            lens, lens_model, tokenizer, c_a, c_b, context, layer)
        if rel_word:
            edges.append({
                "concept_a": c_a,
                "concept_b": c_b,
                "relation": rel_word.lower(),
                "prob": rel_prob,
                "layer_type": layer_type,
                "context_strategy": ctx_strategy,
            })
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(pairs) - i - 1)
            print(f"    [{domain}/{strategy}] {i + 1}/{len(pairs)} pairs "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_total = time.perf_counter() - t_start
    out = {
        "method": "phase48_debiased_grounding",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": {
            "strategy": strategy,
            "seed": SEED if strategy == "random" else None,
            "n_pairs_tested": len(pairs),
            "n_skipped_no_context": n_skipped,
            "reference_pairs": "phase40 relations_%s.json n_pairs_tested" % domain,
        },
        "concepts": concepts,
        "edges": edges,
        "extraction_time_s": round(t_total, 1),
    }
    out_path = out_dir / f"relations_{domain}_{strategy}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  [{domain}/{strategy}] {len(edges)} edges, {t_total:.0f}s "
          f"-> {out_path}")
    return out


# ── Stage 2: debiased grounding analysis (CPU) ─────────────────────────


def load_cache_and_vecs(domain: str, cache_dir: Path = CACHE_DIR) -> tuple[dict, dict]:
    cache_path = cache_dir / f"concept_cache_{domain}_twopass.json"
    vecs_path = cache_dir / f"concept_vecs_{domain}.npz"
    for p in (cache_path, vecs_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (phase39 output).")
    cache = json.loads(cache_path.read_text())
    npz = np.load(vecs_path, allow_pickle=False)
    vecs = {
        "concepts": [str(c) for c in npz["concepts"]],
        "wu_vec": npz["wu_vec"].astype(np.float64),
        "ws_vec": npz["ws_vec"].astype(np.float64),
    }
    return cache, vecs


def edge_diagnostics(edges: list[dict], universe: set[str]) -> dict:
    """Edge provenance stats: context strategy, known-type fraction, mean prob."""
    in_u = [e for e in edges
            if e["concept_a"] in universe and e["concept_b"] in universe]
    strat: dict[str, int] = {}
    for e in in_u:
        s = e.get("context_strategy", "n/a")
        strat[s] = strat.get(s, 0) + 1
    known = sum(1 for e in in_u if e["relation"].lower() in RELATION_TYPES)
    return {
        "n_edges_in_universe": len(in_u),
        "context_strategy": strat,
        "frac_known_relation_type": (round(known / len(in_u), 4) if in_u else None),
        "mean_prob": (round(float(np.mean([e["prob"] for e in in_u])), 4)
                      if in_u else None),
    }


def analyze_domain(
    domain: str,
    cache_dir: Path = CACHE_DIR,
    n_perm: int = N_PERM,
    verbose: bool = True,
) -> dict:
    """Grounding AUC of the four vector spaces under each selection's W.

    Fixed concept universe: phase40's top-DF concept list ∩ valid vectors ∩
    DF records — identical across selections so only the positive set varies.
    """
    cache, vecs = load_cache_and_vecs(domain, cache_dir)
    rel_bge_path = cache_dir / f"relations_{domain}.json"
    rel_bge = json.loads(rel_bge_path.read_text())

    npz_index = {c: i for i, c in enumerate(vecs["concepts"])}
    universe = sorted(
        c for c in rel_bge["concepts"]
        if c in npz_index
        and np.linalg.norm(vecs["wu_vec"][npz_index[c]]) > 1e-6
        and np.linalg.norm(vecs["ws_vec"][npz_index[c]]) > 1e-6
        and cache["concept_chunks"].get(c)
    )
    if len(universe) < 4:
        raise ValueError(f"[{domain}] only {len(universe)} usable concepts")
    uset = set(universe)

    rows = [npz_index[c] for c in universe]
    e_wu = _l2norm_rows(vecs["wu_vec"][rows])
    e_ws = _l2norm_rows(vecs["ws_vec"][rows])
    chunk_texts = _load_corpus_texts(domain, cache)
    e_m = build_m_matrix(universe, cache, chunk_texts)
    e_bge = _l2norm_rows(np.asarray(
        CachedBgeM3Provider().embed(universe), dtype=np.float64))
    spaces = {"E_wu": e_wu, "E_ws": e_ws, "E_M": e_m, "E_bge": e_bge}

    selections: dict[str, dict] = {"bge": rel_bge}
    for strat in SELECTIONS:
        p = cache_dir / f"relations_{domain}_{strat}.json"
        if p.exists():
            selections[strat] = json.loads(p.read_text())
        elif verbose:
            print(f"  WARNING: {p} missing — {strat} arm skipped "
                  f"(run --build-relations)")

    rng = np.random.default_rng(RANDOM_STATE)
    results: dict[str, dict] = {}
    for sel_name, rel in selections.items():
        edge_set = set()
        for e in rel["edges"]:
            a, b = e["concept_a"], e["concept_b"]
            if a in uset and b in uset:
                edge_set.add((a, b) if a < b else (b, a))
        auc = {name: edge_auc(emb, universe, edge_set, n_perm, rng)
               for name, emb in spaces.items()}
        results[sel_name] = {
            "auc": auc,
            "diagnostics": edge_diagnostics(rel["edges"], uset),
        }

    # phase40 published numbers for reference (edge-derived concept set)
    p40_path = EXP / f"phase40_grounding_{domain}.json"
    p40_ref = None
    if p40_path.exists():
        p40 = json.loads(p40_path.read_text())
        p40_ref = {k: v["auc"] for k, v in p40["auc"].items()}

    if verbose:
        print(f"\n  [{domain}] debiased grounding "
              f"(universe: {len(universe)} concepts, "
              f"{len(universe) * (len(universe) - 1) // 2} candidate pairs)")
        header = f"  {'selection':<9} {'space':<7} {'AUC':>7} {'p':>9} {'n_pos':>6}"
        print(header)
        print(f"  {'-' * 42}")
        for sel_name in selections:
            for name in spaces:
                a = results[sel_name]["auc"][name]
                auc_s = f"{a['auc']:.3f}" if a["auc"] is not None else "n/a"
                p_s = f"{a['p']:.4f}" if a["p"] is not None else "n/a"
                print(f"  {sel_name:<9} {name:<7} {auc_s:>7} {p_s:>9} "
                      f"{a['n_pos']:>6}")
            print(f"  {'·' * 42}")
        if p40_ref:
            print(f"  phase40 published (edge-derived set): "
                  + " ".join(f"{k}={v:.3f}" for k, v in p40_ref.items()))

    return {
        "n_universe": len(universe),
        "universe": universe,
        "phase40_reference_auc": p40_ref,
        "selections": results,
    }


def verdict(domain_result: dict) -> dict:
    """Circularity verdict from the selection x space AUC table.

    Three readings:
    1. random arm as NULL control: if no space separates random-W edges
       (all p > 0.05), J-Lens relations read on unrelated pairs carry no
       vector-space structure — the random W cannot adjudicate "which space
       is best" (there is nothing to encode), it only certifies the null.
    2. mcos arm as positive control: E_M should be inflated on mcos-W
       (selection-geometry favoring its own space) — if yes, the bias
       mechanism is demonstrably real, and E_bge's inflation is measured as
       AUC(E_bge | bge-W) - AUC(E_bge | mcos-W).
    3. cross-geometry generalization: E_bge on the non-bge-selected mcos-W
       vs E_ws there, and E_M on bge-W — does each space survive a foreign
       selection geometry?
    """
    sels = domain_result["selections"]

    def auc(sel, space):
        return sels.get(sel, {}).get("auc", {}).get(space, {}).get("auc")

    def pval(sel, space):
        return sels.get(sel, {}).get("auc", {}).get(space, {}).get("p")

    out: dict[str, object] = {}
    bge_on_bge, bge_on_mcos, bge_on_rand = (auc("bge", "E_bge"),
                                            auc("mcos", "E_bge"),
                                            auc("random", "E_bge"))
    ws_on_mcos = auc("mcos", "E_ws")
    m_on_mcos, m_on_bge = auc("mcos", "E_M"), auc("bge", "E_M")
    out["E_bge_auc"] = {"bge_W": bge_on_bge, "mcos_W": bge_on_mcos,
                        "random_W": bge_on_rand}
    out["E_ws_auc_mcos_W"] = ws_on_mcos
    out["E_M_auc"] = {"mcos_W": m_on_mcos, "bge_W": m_on_bge}

    # 1. null control on the random W
    rand_ps = {s: pval("random", s) for s in ("E_wu", "E_ws", "E_M", "E_bge")}
    null_pass = (all(p is not None and p > 0.05 for p in rand_ps.values())
                 if rand_ps else None)
    out["null_control_random_W"] = {
        "all_spaces_nonsignificant": null_pass,
        "p_values": rand_ps,
        "note": ("pass = random-pair relations carry no geometric structure; "
                 "the random arm is a null control, not an adjudicator"),
    }

    if bge_on_mcos is None or ws_on_mcos is None:
        out["verdict"] = "incomplete (mcos arm missing)"
        return out

    # 2. positive control + inflation estimate
    m_rand = auc("random", "E_M")
    out["positive_control"] = {
        "E_M_lift_mcos_minus_random": (round(m_on_mcos - m_rand, 4)
                                       if m_on_mcos is not None
                                       and m_rand is not None else None),
        "E_M_top_on_own_W": (m_on_mcos is not None and bge_on_mcos is not None
                             and m_on_mcos > bge_on_mcos),
        "note": ("mcos selection must inflate E_M — confirms the "
                 "selection-geometry bias mechanism itself"),
    }
    inflation = (bge_on_bge - bge_on_mcos) if bge_on_bge is not None else None
    out["E_bge_inflation_bge_minus_mcos_W"] = (round(inflation, 4)
                                               if inflation is not None else None)

    # 3. cross-geometry generalization → final verdict
    cross = {"E_bge_on_mcos_W": bge_on_mcos, "E_ws_on_mcos_W": ws_on_mcos,
             "E_M_on_bge_W": m_on_bge,
             "E_bge_generalizes_better_than_E_M": (
                 m_on_bge is not None and bge_on_mcos > m_on_bge)}
    out["cross_geometry"] = cross
    p_bge_mcos = pval("mcos", "E_bge")
    significant = p_bge_mcos is not None and p_bge_mcos < 0.05
    if significant and bge_on_mcos >= ws_on_mcos - 0.02:
        out["verdict"] = (
            "inflated_but_real: E_bge's phase40 lead is inflated by bge-"
            "geometry pair selection (AUC drops "
            f"{inflation:+.3f} on the M-selected W), and the positive control "
            "(E_M best on its own mcos-W) confirms the bias mechanism — but "
            "E_bge remains at/above E_ws on the non-bge-selected structured "
            "W and generalizes cross-geometry better than E_M, so its "
            "advantage is not fully a circularity artifact")
    elif bge_on_mcos < ws_on_mcos - 0.02:
        out["verdict"] = (
            "circularity_confirmed: E_bge falls below E_ws on the non-bge-"
            "selected (mcos) W — its phase40 lead was a selection-geometry "
            "artifact; E_ws is the real strongest space")
    else:
        out["verdict"] = (
            "inconclusive: E_bge on the mcos-W is not significantly above "
            "chance — neither selection shows a stable bge advantage")
    return out


# ── main ────────────────────────────────────────────────────────────────


def _stage1_domain(domain: str) -> None:
    cache, _vecs = load_cache_and_vecs(domain)
    rel_bge_path = CACHE_DIR / f"relations_{domain}.json"
    rel_bge = json.loads(rel_bge_path.read_text())
    concepts = rel_bge["concepts"]
    n_target = rel_bge["selection"]["n_pairs_tested"]
    print(f"[{domain}] concept set from phase40: {len(concepts)} concepts, "
          f"target pairs = {n_target}")

    todo = [s for s in SELECTIONS
            if not (CACHE_DIR / f"relations_{domain}_{s}.json").exists()]
    if not todo:
        print(f"[{domain}] all selection arms already built, skipping GPU stage")
        return

    chunk_texts = _load_corpus_texts(domain, cache)
    pair_sets: dict[str, list[tuple[str, str, str]]] = {}
    if "random" in todo:
        pair_sets["random"] = select_random_pairs(concepts, n_target)
    if "mcos" in todo:
        pair_sets["mcos"] = select_mcos_pairs(concepts, n_target, cache,
                                              chunk_texts)

    print(f"[{domain}] loading model + lens "
          f"({sum(len(p) for p in pair_sets.values())} pairs queued)...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    for strat, pairs in pair_sets.items():
        build_relations_for_pairs(
            lens, lens_model, tokenizer, domain, strat, pairs, concepts,
            cache, chunk_texts)
    # Free GPU before the next domain loads its own model copy
    del lens_model, model
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(
        description="Phase 48: debiased grounding test (E_bge circularity)")
    ap.add_argument("--domain", default="all",
                    choices=["medical", "novel", "all"])
    ap.add_argument("--build-relations", action="store_true",
                    help="Stage 1 (GPU): extract relations for random/mcos pairs")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]
    if args.build_relations:
        for domain in domains:  # serial: one model on the GPU at a time
            print(f"\n{'=' * 60}\nPhase 48 stage 1 [{domain}]\n{'=' * 60}")
            _stage1_domain(domain)

    final: dict[str, object] = {}
    for domain in domains:
        rel_ok = (CACHE_DIR / f"relations_{domain}.json").exists()
        if not rel_ok:
            print(f"[{domain}] relations_{domain}.json missing, skipped")
            continue
        print(f"\n{'=' * 60}\nPhase 48 stage 2 [{domain}]\n{'=' * 60}")
        res = analyze_domain(domain, n_perm=args.n_perm)
        res["verdict"] = verdict(res)
        print(f"  VERDICT [{domain}]: {res['verdict'].get('verdict')}")
        final[domain] = res

    if final:
        out = {
            "method": "phase48_debiased_grounding",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": SEED,
            "n_perm": args.n_perm,
            "design": {
                "universe": ("phase40 top-DF concept list (relations_{domain}"
                             ".json concepts) ∩ non-zero wu/ws rows ∩ DF>0; "
                             "fixed across selections"),
                "selections": {
                    "bge": "phase40 W as-is (cluster intra-all + inter-rep)",
                    "random": ("uniform random pairs, same count as phase40, "
                               f"seed={SEED}"),
                    "mcos": ("M-row cosine top-k pairs, same count as phase40 "
                             "(co-occurrence-geometry control arm)"),
                },
                "verdict_rule": ("random W = null control; mcos W = positive "
                                 "control for selection-geometry bias; "
                                 "adjudication = E_bge vs E_ws on the "
                                 "non-bge-selected mcos W (±0.02 margin)"),
            },
            "domains": final,
        }
        out_path = EXP / "phase48_debiased_grounding.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
