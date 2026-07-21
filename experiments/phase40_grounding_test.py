"""Phase 40: 概念接地检验（concept grounding test）。

问题：LLM 自生的概念数学形式（wu_vec / ws_vec 向量）与符号关系结构
（J-Lens 读出的关系图 W）是否"指代同一个东西"？

判据：如果向量空间的余弦相似度能区分 W 的边与非边（ROC-AUC 显著 >0.5），
且向量聚类与 W 的图社区划分一致（ARI/NMI 高于置换基线），则接地成立。
E_M（IDF×BM25 共现统计矩阵行向量）是对照组——它天然与共现结构耦合，
E_wu/E_ws 相对 E_M 的 AUC 差 = 模型语义超越共现统计的净信息量。
E_bge（bge-m3 概念字符串编码）是异空间 baseline。

四路概念向量（同一概念集 = W 中出现的概念 ∩ 有 wu/ws 向量 ∩ 有共现记录）：
  E_wu : W_U 首碎片行向量（phase39 npz，已归一）
  E_ws : prefill workspace 残差 lens.transport 均值（phase39 npz，重新 L2 归一）
  E_M  : M[i,j] = IDF(c_i) × BM25_tf(c_i, chunk_j)，行 L2 归一（对照组）
  E_bge: CachedBgeM3Provider 概念字符串编码（baseline）

阶段 1（--build-relations，需 GPU）：簇内全对 + 簇间代表词对筛选
  （复用 phase30 策略），每对用 phase28 extract_relation 读关系词，
  产出 relations_{domain}.json（幂等：存在且无 --build-relations 时直接加载）。

阶段 2（默认，纯 CPU）：边级 AUC + 置换检验（主判决）；
  Leiden 图社区 vs KMeans 向量聚类的 ARI/NMI + 置换百分位（参考）。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase40_grounding_test --domain medical --build-relations
    python -m experiments.phase40_grounding_test --domain medical   # 仅分析
    python -m experiments.phase40_grounding_test --selftest         # 合成数据自检
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase27_relation_readout import build_relation_prompt  # noqa: F401  (re-exported usage doc)
from experiments.phase28_relation_graph import extract_relation
from experiments.phase30_cluster_relation_graph import cluster_concepts
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"

BM25_K1 = 1.5
BM25_B = 0.75
N_PERM = 1000
RANDOM_STATE = 42


# ── Stage 1: relation graph W construction (GPU) ─────────────────────


def select_top_concepts(cache: dict, top_n: int) -> list[str]:
    """Top-N unique concepts by document frequency (len of concept_chunks)."""
    df = {c: len(cids) for c, cids in cache["concept_chunks"].items()}
    ranked = sorted(df, key=lambda c: (-df[c], c))
    return ranked[:top_n] if top_n > 0 else ranked


def select_relation_pairs(
    concepts: list[str], embed_fn, max_pairs: int = 0,
) -> tuple[list[tuple[str, str, str]], dict]:
    """Cluster-driven pair selection (phase 30 strategy).

    Intra-cluster: all pairs within each bge/KMeans cluster.
    Inter-cluster: pairs of cluster representatives (bridges).
    Returns (pairs, clusters); pairs are (layer_type, concept_a, concept_b),
    deduplicated as undirected pairs, capped at max_pairs (0 = no cap).
    """
    clusters = cluster_concepts(concepts, embed_fn)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(layer_type: str, a: str, b: str) -> None:
        key = (a, b) if a < b else (b, a)
        if a == b or key in seen:
            return
        seen.add(key)
        pairs.append((layer_type, a, b))

    for cluster in clusters.values():
        members = cluster["concepts"]
        for a, b in combinations(members, 2):
            _add("intra", a, b)
    reps = [clusters[cid]["representative"] for cid in sorted(clusters)]
    for a, b in combinations(reps, 2):
        _add("inter", a, b)

    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs, clusters


def find_pair_context(
    c_a: str,
    c_b: str,
    concept_chunks: dict[str, list[str]],
    chunk_text_map: dict[str, str],
) -> tuple[str | None, str]:
    """Find context for a concept pair.

    Prefer a chunk containing both concepts; otherwise concatenate the
    shortest chunk containing a with the shortest containing b.
    Returns (text, strategy) or (None, "missing").
    """
    chunks_a = [cid for cid in concept_chunks.get(c_a, []) if cid in chunk_text_map]
    chunks_b = [cid for cid in concept_chunks.get(c_b, []) if cid in chunk_text_map]
    common = set(chunks_a) & set(chunks_b)
    if common:
        cid = min(common, key=lambda c: len(chunk_text_map[c]))
        return chunk_text_map[cid], "shared"
    if chunks_a and chunks_b:
        ta = chunk_text_map[min(chunks_a, key=lambda c: len(chunk_text_map[c]))]
        tb = chunk_text_map[min(chunks_b, key=lambda c: len(chunk_text_map[c]))]
        return f"{ta}\n{tb}", "concat"
    return None, "missing"


def build_relations(
    lens,
    lens_model,
    tokenizer,
    domain: str,
    cache: dict,
    chunk_text_map: dict[str, str],
    embed_fn,
    top_concepts: int = 60,
    max_pairs: int = 0,
    out_dir: Path = CACHE_DIR,
) -> dict:
    """Stage 1: read a relation word + prob for each selected concept pair."""
    concepts = select_top_concepts(cache, top_concepts)
    print(f"  [{domain}] top-{len(concepts)} concepts by DF")

    pairs, clusters = select_relation_pairs(concepts, embed_fn, max_pairs)
    n_intra = sum(1 for t, _, _ in pairs if t == "intra")
    print(f"  [{domain}] pairs: {n_intra} intra + {len(pairs) - n_intra} inter "
          f"= {len(pairs)} total ({len(clusters)} clusters)")

    layer = lens.source_layers[-1]
    edges = []
    n_skipped = 0
    t_start = time.perf_counter()
    for i, (layer_type, c_a, c_b) in enumerate(pairs):
        context, strategy = find_pair_context(
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
                "context_strategy": strategy,
            })
        if (i + 1) % 25 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(pairs) - i - 1)
            print(f"    [{domain}] {i + 1}/{len(pairs)} pairs "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_total = time.perf_counter() - t_start
    out = {
        "method": "phase40_relation_graph",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection": {
            "top_concepts": top_concepts,
            "max_pairs": max_pairs,
            "strategy": "cluster intra-all + inter-representative (phase30)",
            "n_clusters": len(clusters),
            "n_pairs_tested": len(pairs),
            "n_skipped_no_context": n_skipped,
        },
        "concepts": concepts,
        "edges": edges,
        "extraction_time_s": round(t_total, 1),
    }
    out_path = out_dir / f"relations_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  [{domain}] {len(edges)} edges, {t_total:.0f}s "
          f"({t_total / max(1, len(pairs)):.2f}s/pair) -> {out_path}")
    return out


# ── Stage 2: grounding analysis (CPU) ────────────────────────────────


def _l2norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms > 0, norms, 1.0)


def build_m_matrix(
    concepts: list[str],
    cache: dict,
    chunk_texts: dict[str, str],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> np.ndarray:
    """M[i,j] = IDF(concept_i) x BM25_tf(concept_i, chunk_j), rows L2-normed.

    tf = literal (case-insensitive) occurrence count of the concept string in
    the chunk text; falls back to occurrences in the cached per-chunk concept
    list when the string is not found verbatim (BPE-completed concepts).
    """
    concept_chunks = cache["concept_chunks"]
    chunk_ids = sorted({cid for c in concepts for cid in concept_chunks.get(c, [])})
    n_docs = len(chunk_ids)
    chunk_index = {cid: j for j, cid in enumerate(chunk_ids)}

    doc_len = np.array(
        [len(chunk_texts.get(cid, "").split()) for cid in chunk_ids], dtype=np.float64)
    avgdl = float(doc_len.mean()) if n_docs and doc_len.mean() > 0 else 1.0

    m = np.zeros((len(concepts), n_docs), dtype=np.float64)
    for i, concept in enumerate(concepts):
        cids = [cid for cid in concept_chunks.get(concept, []) if cid in chunk_index]
        df = len(cids)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        pattern = re.compile(re.escape(concept), re.IGNORECASE)
        for cid in cids:
            text = chunk_texts.get(cid, "")
            tf = len(pattern.findall(text)) if text else 0
            if tf == 0:
                tf = sum(1 for c in cache["chunks"].get(cid, {}).get("concepts", [])
                         if c.lower() == concept)
            dl = doc_len[chunk_index[cid]]
            tf_norm = tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl))
            m[i, chunk_index[cid]] = idf * tf_norm
    return _l2norm_rows(m)


def edge_auc(
    emb: np.ndarray,
    concepts: list[str],
    edge_set: set[tuple[str, str]],
    n_perm: int = N_PERM,
    rng: np.random.Generator | None = None,
) -> dict:
    """ROC-AUC of cosine similarity distinguishing W edges from non-edges.

    Positive = pair present in W (undirected), negative = absent.
    p-value from n_perm label-shuffle permutations (one-sided, +1 corrected).
    """
    rng = rng or np.random.default_rng(RANDOM_STATE)
    n = len(concepts)
    idx_i, idx_j = np.triu_indices(n, k=1)
    keys = [(concepts[i], concepts[j]) if concepts[i] < concepts[j]
            else (concepts[j], concepts[i])
            for i, j in zip(idx_i.tolist(), idx_j.tolist())]
    y = np.array([1.0 if k in edge_set else 0.0 for k in keys])
    sim = emb @ emb.T
    scores = sim[idx_i, idx_j]
    if y.sum() == 0 or y.sum() == len(y):
        return {"auc": None, "p": None, "n_pos": int(y.sum()), "n_pairs": len(y)}
    auc = float(roc_auc_score(y, scores))
    count = 0
    for _ in range(n_perm):
        if roc_auc_score(rng.permutation(y), scores) >= auc:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"auc": auc, "p": p, "n_pos": int(y.sum()), "n_pairs": len(y)}


def graph_leiden_partition(
    edges: list[dict], concepts: list[str],
) -> list[int]:
    """Leiden communities of W (weight = edge prob), labels aligned to concepts."""
    import igraph as ig

    index = {c: i for i, c in enumerate(concepts)}
    g = ig.Graph(n=len(concepts))
    g.vs["name"] = concepts
    edge_list, weights = [], []
    for e in edges:
        a, b = e["concept_a"], e["concept_b"]
        if a in index and b in index:
            edge_list.append((index[a], index[b]))
            weights.append(max(float(e["prob"]), 1e-6))
    g.add_edges(edge_list)
    part = g.community_leiden(objective_function="modularity", weights=weights)
    return list(part.membership)


def cluster_agreement(
    emb: np.ndarray,
    graph_labels: list[int],
    n_perm: int = N_PERM,
    random_state: int = RANDOM_STATE,
) -> dict:
    """KMeans(k = #graph communities) vs Leiden partition: ARI/NMI.

    Permutation baseline: n_perm random partitions preserving the KMeans
    cluster-size distribution; report the percentile of the observed ARI
    within that null distribution.
    """
    k = len(set(graph_labels))
    n = len(graph_labels)
    if k < 2 or k >= n:
        return {"ari": None, "nmi": None, "ari_percentile": None, "k": k}
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    vec_labels = km.fit_predict(emb)
    ari = float(adjusted_rand_score(graph_labels, vec_labels))
    nmi = float(normalized_mutual_info_score(graph_labels, vec_labels))

    rng = np.random.default_rng(random_state + 1)
    sizes = np.bincount(vec_labels, minlength=k)
    null_aris = np.empty(n_perm)
    for t in range(n_perm):
        perm = np.repeat(np.arange(k), sizes)
        rng.shuffle(perm)
        null_aris[t] = adjusted_rand_score(graph_labels, perm)
    percentile = float((null_aris < ari).mean() * 100.0)
    return {"ari": ari, "nmi": nmi, "ari_percentile": percentile, "k": k}


def load_grounding_inputs(
    domain: str, cache_dir: Path,
) -> tuple[dict, dict, dict]:
    """Load twopass cache, concept vectors, and relation edges for a domain."""
    cache_path = cache_dir / f"concept_cache_{domain}_twopass.json"
    vecs_path = cache_dir / f"concept_vecs_{domain}.npz"
    rel_path = cache_dir / f"relations_{domain}.json"
    for p in (cache_path, vecs_path, rel_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. "
                f"Run phase39_two_pass_cache.py first, and stage 1 with "
                f"--build-relations for relations_{domain}.json.")
    cache = json.loads(cache_path.read_text())
    npz = np.load(vecs_path, allow_pickle=False)
    vecs = {
        "concepts": [str(c) for c in npz["concepts"]],
        "wu_vec": npz["wu_vec"].astype(np.float64),
        "ws_vec": npz["ws_vec"].astype(np.float64),
        "count": npz["count"],
    }
    relations = json.loads(rel_path.read_text())
    return cache, vecs, relations


def run_analysis(
    domain: str,
    cache_dir: Path = CACHE_DIR,
    out_dir: Path = EXP,
    chunk_texts: dict[str, str] | None = None,
    embed_fn=None,
    n_perm: int = N_PERM,
    verbose: bool = True,
) -> dict:
    """Stage 2: grounding analysis for one domain. Pure CPU.

    chunk_texts: {cid: full chunk text} for the M matrix. When None, loads
    the corpus via load_graphrag_bench(domain) (same chunking as phase39,
    so chunk ids match the cache).
    embed_fn: concept-string encoder for E_bge; defaults to CachedBgeM3Provider.
    """
    cache, vecs, relations = load_grounding_inputs(domain, cache_dir)
    edges = relations["edges"]

    edge_concepts = {e["concept_a"] for e in edges} | {e["concept_b"] for e in edges}
    npz_index = {c: i for i, c in enumerate(vecs["concepts"])}
    # Concept set: in W, has both vectors (non-zero rows), and has DF records.
    concepts = sorted(
        c for c in edge_concepts
        if c in npz_index
        and np.linalg.norm(vecs["wu_vec"][npz_index[c]]) > 1e-6
        and np.linalg.norm(vecs["ws_vec"][npz_index[c]]) > 1e-6
        and cache["concept_chunks"].get(c)
    )
    if len(concepts) < 4:
        raise ValueError(f"[{domain}] only {len(concepts)} usable concepts; need >= 4")
    if verbose:
        print(f"  [{domain}] concept set: {len(concepts)} "
              f"(edges: {len(edges)}, edge concepts: {len(edge_concepts)})")

    # Four vector spaces on the same concept set
    rows = [npz_index[c] for c in concepts]
    e_wu = _l2norm_rows(vecs["wu_vec"][rows])
    e_ws = _l2norm_rows(vecs["ws_vec"][rows])  # re-normalize: saved as mean of normed

    if chunk_texts is None:
        corpus, _ = load_graphrag_bench(domain, max_queries=1)
        chunk_texts = {cid: t for cid, t in corpus.items()
                       if cid in cache["chunks"]}
    e_m = build_m_matrix(concepts, cache, chunk_texts)

    if embed_fn is None:
        embed_fn = CachedBgeM3Provider().embed
    e_bge = _l2norm_rows(np.asarray(embed_fn(concepts), dtype=np.float64))

    edge_set = set()
    for e in edges:
        a, b = e["concept_a"], e["concept_b"]
        if a in concepts and b in concepts:
            edge_set.add((a, b) if a < b else (b, a))

    spaces = {"E_wu": e_wu, "E_ws": e_ws, "E_M": e_m, "E_bge": e_bge}

    # 1. Edge-level AUC (primary verdict)
    rng = np.random.default_rng(RANDOM_STATE)
    auc_results = {name: edge_auc(emb, concepts, edge_set, n_perm, rng)
                   for name, emb in spaces.items()}

    # 2. Cluster-level agreement (reference)
    graph_labels = graph_leiden_partition(
        [e for e in edges
         if e["concept_a"] in concepts and e["concept_b"] in concepts],
        concepts)
    clu_results = {name: cluster_agreement(emb, graph_labels, n_perm)
                   for name, emb in spaces.items()}

    if verbose:
        print(f"\n  [{domain}] Grounding table "
              f"(k={clu_results['E_wu']['k']} graph communities, "
              f"{auc_results['E_wu']['n_pos']} edges / "
              f"{auc_results['E_wu']['n_pairs']} pairs)")
        print(f"  {'space':<7} {'AUC':>7} {'p':>8} {'ARI':>7} {'NMI':>7} "
              f"{'ARI pct':>8}")
        print(f"  {'-' * 46}")
        for name in spaces:
            a, c = auc_results[name], clu_results[name]
            auc_s = f"{a['auc']:.3f}" if a["auc"] is not None else "n/a"
            p_s = f"{a['p']:.4f}" if a["p"] is not None else "n/a"
            ari_s = f"{c['ari']:.3f}" if c["ari"] is not None else "n/a"
            nmi_s = f"{c['nmi']:.3f}" if c["nmi"] is not None else "n/a"
            pct_s = (f"{c['ari_percentile']:.1f}"
                     if c["ari_percentile"] is not None else "n/a")
            print(f"  {name:<7} {auc_s:>7} {p_s:>8} {ari_s:>7} {nmi_s:>7} "
                  f"{pct_s:>8}")
        wu_auc = auc_results["E_wu"]["auc"]
        m_auc = auc_results["E_M"]["auc"]
        if wu_auc is not None and m_auc is not None:
            print(f"\n  Net model semantics over co-occurrence: "
                  f"AUC(E_wu) - AUC(E_M) = {wu_auc - m_auc:+.3f}")

    out = {
        "method": "phase40_grounding_test",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_concepts": len(concepts),
        "n_edges": len(edge_set),
        "n_perm": n_perm,
        "concepts": concepts,
        "auc": auc_results,
        "cluster": clu_results,
        "interpretation": {
            "verdict": ("AUC > 0.5 with p < 0.05 = vector space encodes the "
                        "symbolic relation structure (grounding holds)"),
            "net_info": ("AUC(E_wu/E_ws) - AUC(E_M) = model semantics beyond "
                         "co-occurrence statistics"),
        },
    }
    out_path = out_dir / f"phase40_grounding_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  saved to {out_path}")
    return out


# ── Synthetic self-test (no model, no real data files) ───────────────


def run_synthetic_selftest(tmp_root: Path | None = None, n_perm: int = 200) -> dict:
    """Verify the stage-2 analysis on synthetic data with planted structure.

    20 concepts in 2 natural clusters of 10. Vector spaces (wu/ws/bge) are
    cluster centers + small noise; chunk texts make each cluster's concepts
    co-occur (so E_M also reflects the clusters). W = all intra-cluster pairs.
    Expected: AUC ~ 1.0, ARI high, permutation p significant for all spaces.
    Writes everything under a temp dir — never touches experiments/m6/.
    """
    import tempfile

    rng = np.random.default_rng(7)
    dim = 64
    n_per_cluster = 10
    concepts = [f"concept{i:02d}" for i in range(2 * n_per_cluster)]
    group = {c: i // n_per_cluster for i, c in enumerate(concepts)}

    # Orthogonal cluster centers + small noise -> intra-cluster cosine >> inter
    centers = np.zeros((2, dim))
    centers[0, : dim // 2] = 1.0
    centers[1, dim // 2:] = 1.0
    centers = _l2norm_rows(centers)

    def _make_space(noise: float) -> np.ndarray:
        v = centers[[group[c] for c in concepts]] + noise * rng.normal(
            size=(len(concepts), dim))
        return _l2norm_rows(v)

    e_wu = _make_space(0.10)
    e_ws = _make_space(0.12)
    e_bge = _make_space(0.15)

    # Chunks: 4 per cluster; each chunk text contains the cluster's concepts
    chunk_texts: dict[str, str] = {}
    concept_chunks: dict[str, list[str]] = {c: [] for c in concepts}
    chunks_entry: dict[str, dict] = {}
    for g in range(2):
        members = concepts[g * n_per_cluster:(g + 1) * n_per_cluster]
        for j in range(4):
            cid = f"synthetic::chunk_{g * 4 + j}"
            present = sorted(rng.choice(members, size=7, replace=False))
            text = " ".join(
                w for c in present for w in [c] * int(rng.integers(1, 4)))
            chunk_texts[cid] = text
            chunks_entry[cid] = {"concepts": present, "roles": {}}
            for c in present:
                concept_chunks[c].append(cid)

    cache = {
        "domain": "synthetic",
        "chunks": chunks_entry,
        "concept_chunks": concept_chunks,
        "n_unique_concepts": len(concepts),
    }
    edges = []
    for a, b in combinations(concepts, 2):
        if group[a] == group[b]:
            edges.append({"concept_a": a, "concept_b": b,
                          "relation": "related", "prob": float(rng.uniform(0.3, 0.9))})
    relations = {"domain": "synthetic", "concepts": concepts, "edges": edges}

    tmp = Path(tempfile.mkdtemp(prefix="phase40_selftest_",
                                dir=tmp_root or tempfile.gettempdir()))
    (tmp / "concept_cache_synthetic_twopass.json").write_text(json.dumps(cache))
    np.savez(tmp / "concept_vecs_synthetic.npz",
             concepts=np.array(concepts),
             wu_vec=e_wu.astype(np.float32),
             ws_vec=e_ws.astype(np.float32),
             count=np.array([4] * len(concepts), dtype=np.int64))
    (tmp / "relations_synthetic.json").write_text(json.dumps(relations))

    bge_by_concept = {c: e_bge[i] for i, c in enumerate(concepts)}
    out = run_analysis(
        "synthetic", cache_dir=tmp, out_dir=tmp,
        chunk_texts=chunk_texts,
        embed_fn=lambda cs: [bge_by_concept[c] for c in cs],
        n_perm=n_perm, verbose=True)

    # Assertions: planted structure must be recovered by every space
    failures = []
    for name in ("E_wu", "E_ws", "E_M", "E_bge"):
        auc = out["auc"][name]["auc"]
        p = out["auc"][name]["p"]
        ari = out["cluster"][name]["ari"]
        if auc is None or auc < 0.9:
            failures.append(f"{name} AUC={auc} (< 0.9)")
        if p is None or p >= 0.05:
            failures.append(f"{name} p={p} (>= 0.05)")
        if ari is None or ari < 0.5:
            failures.append(f"{name} ARI={ari} (< 0.5)")
    if failures:
        raise AssertionError("synthetic self-test FAILED: " + "; ".join(failures))
    print(f"\n  SELF-TEST PASSED (artifacts in {tmp})")
    return out


# ── main ──────────────────────────────────────────────────────────────


def _load_corpus_texts(domain: str, cache: dict) -> dict[str, str]:
    """Full chunk texts keyed by the cache's chunk ids (same chunking as phase39)."""
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    return {cid: t for cid, t in corpus.items() if cid in cache["chunks"]}


def _run_domain(domain: str, args) -> None:
    cache_path = CACHE_DIR / f"concept_cache_{domain}_twopass.json"
    rel_path = CACHE_DIR / f"relations_{domain}.json"

    if args.build_relations:
        cache = json.loads(cache_path.read_text())
        print(f"[{domain}] Loading model + lens for relation building...")
        cand = detect_model()
        lens = load_lens(cand["local_lens_path"])
        model_src = (cand["local_model_dir"]
                     if _model_dir_complete(cand["local_model_dir"])
                     else cand["model_id"])
        model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
        import jlens
        lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
        embed = CachedBgeM3Provider()
        chunk_texts = _load_corpus_texts(domain, cache)
        build_relations(
            lens, lens_model, tokenizer, domain, cache, chunk_texts,
            embed.embed, top_concepts=args.top_concepts,
            max_pairs=args.max_pairs)
    elif not rel_path.exists():
        raise SystemExit(
            f"{rel_path} not found. Run with --build-relations first (GPU).")

    run_analysis(domain, n_perm=args.n_perm)


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(description="Phase 40: concept grounding test")
    ap.add_argument("--domain", default="medical",
                    choices=["medical", "novel", "all"])
    ap.add_argument("--build-relations", action="store_true",
                    help="Stage 1 (GPU): build relations_{domain}.json via J-Lens")
    ap.add_argument("--top-concepts", type=int, default=60,
                    help="Top-N concepts by DF entering the relation graph")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="Cap on relation pairs (0 = no cap, debug use)")
    ap.add_argument("--n-perm", type=int, default=N_PERM,
                    help="Permutation count for p-values / null distributions")
    ap.add_argument("--selftest", action="store_true",
                    help="Run synthetic-data self-test of stage 2 and exit")
    args = ap.parse_args()

    if args.selftest:
        run_synthetic_selftest()
        return

    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]
    for domain in domains:
        print(f"\n{'=' * 60}\nPhase 40: grounding test [{domain}]\n{'=' * 60}")
        _run_domain(domain, args)


if __name__ == "__main__":
    main()
