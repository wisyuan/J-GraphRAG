"""Phase 41: J-GraphRAG 检索侧两个实验（全量规模，纯 CPU/网络，无需 Qwen GPU）。

A. 检索等价性：矩阵运算检索 vs 图扩散检索
   对同一查询比较两种检索器的 chunk 排序：
     R_graph        : Phase 25/28 图传播（flat 概念图 + 关系传播，复用现有函数）
     R_matrix       : 矩阵运算版 q_concept × M（M = IDF×BM25 概念×chunk 矩阵，
                      从 twopass 缓存构建，公式参考 phase38.build_matrices）
     R_matrix_ridge : 传播步加 Ridge 正则 (M M^T + λI)^{-1}，λ ∈ {0.01, 0.1, 1.0}
   度量：top-10 chunk 重叠率 + chunk 得分 Kendall τ（不需要 LLM judge）。
   判决：top-10 重叠 ≥0.7 且 τ ≥0.6 → "矩阵可替代图扩散"。

   q_concept 初始化两臂 + 扩展两臂：
     - seed_idf：seed chunk 概念的指示向量（与图传播的输入严格一致——
       这一臂是"矩阵乘法 ≡ 图循环"的直接检验，预期 τ≈1.0）
     - bge：query 与概念字符串 bge 嵌入的 cosine（查询语义直接入概念空间）
     - 扩展臂 ws/wu：q2 = q + α·(S @ q)，S = 概念向量（ws_vec/wu_vec，phase39
       npz）的 cosine 相似度矩阵——纯向量空间的关系扩展，替代图上的关系边扩展
     - 关系扩展臂：与 phase28 relation_propagate 等价的指示语义矩阵版
       （q2[c] = 1[seed] + w^hop·1[hop-neighbor]），预期与 R_graph_relation 严格一致

B. 关系传播消融（medical 全量 957 chunks）
   网格：weight ∈ {0, 0.1, 0.3, 0.5}（0 = 不用关系）× hops ∈ {1, 2}，
   用 relation_propagate 的参数化版本（本文件 relation_propagate_param，
   weight=0.7/hops=1 时与 phase28.relation_propagate 逐分一致）。
   每格：evidence recall（默认确定性词面覆盖，--llm-recall 可切 LLM judge）
   + ACC（DeepSeek generate_answer + judge_answer_correctness）。
   对照：B0（纯 bge）与 flat（无关系概念图）同批跑出。
   回答："关系传播在全量规模是否有益、最优权重是多少"。

注意（phase25/26/28 的合并 quirk）：原实验 seed_k=10=TOP_K，merged[:10] 恒等于
bge seed 列表，传播命中的 chunk 实际从未进入 context。本脚本默认 SEED_K=5，
让传播结果填充 top-10 的后 5 席（--seed-k 可调）。跑 B 部分网格时这一点必须
成立，否则所有格子都退化成 B0。

前置数据（phase39/40 产出，按格式契约加载）：
  concept_cache_{domain}_twopass.json : chunks {cid:{concepts,roles,...}},
                                        concept_chunks {concept_lower:[cid]}
  concept_vecs_{domain}.npz           : concepts, wu_vec, ws_vec [N,3584], count
  relations_{domain}.json             : edges [{concept_a, concept_b, relation,
                                        prob, layer_type, context_strategy}]
  （relations 缺失时 A 的关系臂跳过并告警；B 无法运行）

运行（GPU 被占用时也能跑；bge 嵌入走磁盘缓存，缓存冷时请设
LINCLE_BGE_M3_DEVICE=cpu 避免与 GPU 任务争用）：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase41_retrieval_equivalence --selftest
    python -m experiments.phase41_retrieval_equivalence --part a --domain medical --max-queries 50
    python -m experiments.phase41_retrieval_equivalence --part b --domain medical --max-queries 0 --quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage7c import ConceptGraph, cosine_topk_ids
from experiments.phase25_filter_bpe_benchmark import rebuild_graph_index
from experiments.phase28_relation_graph import relation_propagate
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall, split_evidence,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"

TOP_K = 10
SEED_K = 5  # < TOP_K so propagated chunks actually enter the context (see docstring)
PROPAGATE_K = 20
REL_WEIGHT_PHASE28 = 0.7  # phase28.relation_propagate's hardcoded discount
# The full-scale (957-chunk) benchmark phase38_full_benchmark used 56 questions
# = 14 per level x 4 levels via load_graphrag_bench(domain, 56). --max-queries 0
# selects exactly that full set.
FULL_BENCH_QUERIES = 56

DEFAULT_WEIGHTS = [0.0, 0.1, 0.3, 0.5]
DEFAULT_HOPS = [1, 2]
DEFAULT_RIDGE_LAMBDAS = [0.01, 0.1, 1.0]


# ── Data loading (contract-programmed against phase39/40 outputs) ──────


def load_phase41_inputs(
    domain: str, cache_dir: Path = CACHE_DIR,
) -> tuple[dict, dict, dict | None]:
    """Load twopass cache, concept vectors, and (optional) relation edges."""
    cache_path = cache_dir / f"concept_cache_{domain}_twopass.json"
    vecs_path = cache_dir / f"concept_vecs_{domain}.npz"
    rel_path = cache_dir / f"relations_{domain}.json"
    for p in (cache_path, vecs_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run phase39_two_pass_cache.py first.")
    cache = json.loads(cache_path.read_text())
    npz = np.load(vecs_path, allow_pickle=False)
    vecs = {
        "concepts": [str(c) for c in npz["concepts"]],
        "wu_vec": npz["wu_vec"].astype(np.float64),
        "ws_vec": npz["ws_vec"].astype(np.float64),
        "count": npz["count"],
    }
    relations = json.loads(rel_path.read_text()) if rel_path.exists() else None
    return cache, vecs, relations


def load_corpus_texts(domain: str, cache: dict) -> dict[str, str]:
    """Full chunk texts keyed by the cache's chunk ids (same chunking as phase39)."""
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    return {cid: t for cid, t in corpus.items() if cid in cache["chunks"]}


def load_questions(domain: str, max_queries: int) -> list[dict]:
    """Questions from GraphRAG-Bench. max_queries=0 → the 56-question full set."""
    n = FULL_BENCH_QUERIES if max_queries == 0 else max_queries
    _, questions = load_graphrag_bench(domain, n)
    return questions


# ── Graph + matrix construction ────────────────────────────────────────


def build_graph_from_cache(
    cache: dict, chunk_text_map: dict[str, str],
) -> ConceptGraph:
    """Rebuild a ConceptGraph from the twopass cache (already phase25-filtered).

    IDF uses the same smoothed formula as phase25_filter_with_bpe's meta
    (log((N+1)/(df+1))+1); BM25 tf comes from ConceptGraph.compute_tf over the
    full chunk texts, so graph propagate scoring is identical to a live build.
    """
    concept_chunks = {c: list(cids) for c, cids in cache["concept_chunks"].items()}
    n_total = len(cache["chunks"])
    idf = {c: math.log((n_total + 1) / (len(cids) + 1)) + 1
           for c, cids in concept_chunks.items()}
    graph = ConceptGraph()
    rebuild_graph_index(graph, concept_chunks, idf)
    graph.compute_tf(chunk_text_map)
    return graph


def build_m_matrix(
    graph: ConceptGraph, chunk_ids: list[str],
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """M[i,j] = IDF(concept_i) x BM25_tf(concept_i, chunk_j) — dense float64.

    Same formula as phase38.build_matrices (IDF x BM25 concept x chunk), but
    built directly from the graph's own idf/_bm25_tf_norm so that
    indicator(seed_concepts) @ M reproduces ConceptGraph.propagate scores
    exactly (the equivalence arm). Rows follow sorted(graph.concept_chunks).
    """
    concepts = sorted(graph.concept_chunks.keys())
    chunk_index = {cid: j for j, cid in enumerate(chunk_ids)}
    m = np.zeros((len(concepts), len(chunk_ids)), dtype=np.float64)
    for i, concept in enumerate(concepts):
        idf = graph.idf.get(concept, 1.0)
        for cid in graph.concept_chunks.get(concept, []):
            j = chunk_index.get(cid)
            if j is None:
                continue
            if (cid, concept) in graph.tf:
                tf_weight = graph._bm25_tf_norm(cid, graph.tf[(cid, concept)])
            else:
                tf_weight = 1.0  # same fallback as ConceptGraph.propagate
            m[i, j] = idf * tf_weight
    return m, concepts, chunk_index


def build_concept_sim(
    vecs: dict, concepts: list[str], which: str,
) -> np.ndarray:
    """Row-normalized non-negative cosine similarity S over the concept set.

    which: "ws_vec" | "wu_vec" (phase39 npz). Concepts missing from the npz
    get zero rows/cols. Diagonal zeroed — S is an expansion operator, not an
    identity channel.
    """
    mat = vecs[which]
    npz_index = {c: i for i, c in enumerate(vecs["concepts"])}
    n = len(concepts)
    s = np.zeros((n, n), dtype=np.float64)
    rows = [npz_index.get(c) for c in concepts]
    valid = [i for i, r in enumerate(rows) if r is not None]
    if valid:
        sub = mat[[rows[i] for i in valid]]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        sub = sub / np.where(norms > 0, norms, 1.0)
        sim = np.maximum(sub @ sub.T, 0.0)
        for a, i in enumerate(valid):
            for b, j in enumerate(valid):
                s[i, j] = sim[a, b]
    np.fill_diagonal(s, 0.0)
    row_sum = s.sum(axis=1, keepdims=True)
    return s / np.where(row_sum > 0, row_sum, 1.0)


def build_relation_adj(relations: dict | None) -> dict[str, list[tuple[str, str]]]:
    """Undirected concept → [(other, relation)] adjacency, phase40 edge format."""
    adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not relations:
        return {}
    for e in relations["edges"]:
        a, b, rel = e["concept_a"], e["concept_b"], e["relation"]
        adj[a].append((b, rel))
        adj[b].append((a, rel))
    return dict(adj)


# ── Scored retrievers ──────────────────────────────────────────────────


def propagate_scored(
    graph: ConceptGraph,
    seed_chunk_ids: list[str],
    use_idf: bool = True,
    use_bm25: bool = True,
) -> dict[str, float]:
    """ConceptGraph.propagate, but returning the internal {cid: score} dict.

    Mirrors propagate()'s scoring loop line-for-line (idf x bm25 over seed
    concepts, seeds excluded); ordering of dict items sorted desc matches
    propagate()'s returned list.
    """
    if not hasattr(graph, "idf"):
        graph.compute_idf()
    seed_concepts = set()
    for cid in seed_chunk_ids:
        seed_concepts.update(graph.chunk_concepts.get(cid, []))
    chunk_score: dict[str, float] = defaultdict(float)
    for concept in seed_concepts:
        idf_weight = graph.idf.get(concept, 1.0) if use_idf else 1.0
        for cid in graph.concept_chunks.get(concept, []):
            if cid not in seed_chunk_ids:
                if use_bm25 and hasattr(graph, "tf") and (cid, concept) in graph.tf:
                    tf_weight = graph._bm25_tf_norm(cid, graph.tf[(cid, concept)])
                else:
                    tf_weight = 1.0
                chunk_score[cid] += idf_weight * tf_weight
    return dict(chunk_score)


def relation_propagate_param(
    graph: ConceptGraph,
    relation_adj: dict[str, list[tuple[str, str]]],
    seed_chunk_ids: list[str],
    weight: float = REL_WEIGHT_PHASE28,
    hops: int = 1,
) -> tuple[dict[str, float], int]:
    """Parameterized phase28.relation_propagate: relation-expansion weight x hops.

    weight: IDF discount for relation-expanded concepts (0 = no relation
        expansion → reduces to flat propagation; hop-h concepts get weight**h).
    hops: relation expansion rounds (1 = phase28).
    With weight=0.7, hops=1 the scores equal phase28.relation_propagate's
    (set semantics: each expanded concept counted once, tf fallback 1).
    Returns ({cid: score}, n_expanded_concepts).
    """
    if not hasattr(graph, "idf"):
        graph.compute_idf()
    seed_concepts = set()
    for cid in seed_chunk_ids:
        seed_concepts.update(graph.chunk_concepts.get(cid, []))

    concept_weight = {c: 1.0 for c in seed_concepts}
    frontier = set(seed_concepts)
    for hop in range(1, hops + 1):
        nxt = set()
        if weight > 0:
            for concept in frontier:
                for related, _rel in relation_adj.get(concept, []):
                    if related not in concept_weight:
                        concept_weight[related] = weight ** hop
                        nxt.add(related)
        frontier = nxt
    n_expanded = len(concept_weight) - len(seed_concepts)

    chunk_score: dict[str, float] = defaultdict(float)
    for concept, w in concept_weight.items():
        if w == 0:
            continue
        idf_weight = graph.idf.get(concept, 1.0) * w
        for cid in graph.concept_chunks.get(concept, []):
            if cid not in seed_chunk_ids:
                tf = graph.tf.get((cid, concept), 1)
                tf_weight = graph._bm25_tf_norm(cid, tf)
                chunk_score[cid] += idf_weight * tf_weight
    return dict(chunk_score), n_expanded


def concept_weight_vector(
    graph: ConceptGraph,
    relation_adj: dict[str, list[tuple[str, str]]],
    seed_chunk_ids: list[str],
    concept_index: dict[str, int],
    weight: float,
    hops: int,
) -> np.ndarray:
    """Indicator-semantics q_concept for the relation matrix arm.

    q[i] = 1 for seed concepts, weight**hop for hop-neighbors — the exact
    matrix analogue of relation_propagate_param, so q @ M equals its scores.
    """
    seed_concepts = set()
    for cid in seed_chunk_ids:
        seed_concepts.update(graph.chunk_concepts.get(cid, []))
    concept_weight = {c: 1.0 for c in seed_concepts}
    frontier = set(seed_concepts)
    for hop in range(1, hops + 1):
        nxt = set()
        if weight > 0:
            for concept in frontier:
                for related, _rel in relation_adj.get(concept, []):
                    if related not in concept_weight:
                        concept_weight[related] = weight ** hop
                        nxt.add(related)
        frontier = nxt
    q = np.zeros(len(concept_index), dtype=np.float64)
    for concept, w in concept_weight.items():
        i = concept_index.get(concept)
        if i is not None:
            q[i] = w
    return q


def merge_ranking(
    seed_ids: list[str],
    scores: dict[str, float],
    b0_ids: list[str],
    top_k: int = TOP_K,
    seed_keep: int = SEED_K,
) -> list[str]:
    """Seeds first (seed_keep), then score-descending propagated, bge backfill."""
    merged = list(seed_ids[:seed_keep])
    for cid, _s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        if cid not in merged:
            merged.append(cid)
        if len(merged) >= top_k:
            break
    if len(merged) < top_k:
        for cid in b0_ids:
            if cid not in merged:
                merged.append(cid)
            if len(merged) >= top_k:
                break
    return merged[:top_k]


def scores_to_dict(score_vec: np.ndarray, chunk_ids: list[str],
                   exclude: set[str] | None = None) -> dict[str, float]:
    """Dense score vector → {cid: score} for positive entries (seeds excluded)."""
    exclude = exclude or set()
    return {cid: float(score_vec[j]) for j, cid in enumerate(chunk_ids)
            if cid not in exclude and score_vec[j] > 0}


# ── Metrics ────────────────────────────────────────────────────────────


def topk_overlap(ranking_a: list[str], ranking_b: list[str], k: int = TOP_K) -> float:
    """|A ∩ B| / k over the top-k merged rankings.

    k is capped at the shorter ranking's length so tiny corpora (self-tests)
    still max out at 1.0; real runs always have len == k after bge backfill.
    """
    k_eff = min(k, len(ranking_a), len(ranking_b))
    if k_eff == 0:
        return 0.0
    a, b = set(ranking_a[:k_eff]), set(ranking_b[:k_eff])
    return len(a & b) / k_eff


def kendall_tau_scores(s_a: dict[str, float], s_b: dict[str, float]) -> float | None:
    """Kendall τ over the union of positively-scored chunks (0-filled).

    None when fewer than 2 candidates or one side is constant (τ undefined).
    """
    keys = sorted(set(s_a) | set(s_b))
    if len(keys) < 2:
        return None
    x = np.array([s_a.get(k, 0.0) for k in keys])
    y = np.array([s_b.get(k, 0.0) for k in keys])
    if np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(kendalltau(x, y).statistic)


_LEX_STOP = {
    "that", "this", "with", "from", "have", "been", "were", "they", "them",
    "their", "there", "which", "what", "when", "where", "than", "then",
    "also", "more", "most", "some", "such", "only", "into", "about",
    "these", "those", "other", "between", "through", "during", "after",
    "before", "because", "while", "being", "does", "done", "each",
}


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z]{4,}", text.lower())
            if t not in _LEX_STOP}


def lexical_evidence_recall(evidence_stmts: list[str], context: str) -> float:
    """Deterministic evidence-recall proxy: mean content-token coverage.

    For each evidence statement, fraction of its content tokens (len≥4,
    stopword-free) that appear in the retrieved context; averaged over
    statements. Free and reproducible — the cheap default; --llm-recall
    switches to phase4's llm_judge_evidence_recall.
    """
    if not evidence_stmts:
        return 0.0
    ctx = _content_tokens(context)
    scores = []
    for stmt in evidence_stmts:
        toks = _content_tokens(stmt)
        scores.append(len(toks & ctx) / len(toks) if toks else 0.0)
    return float(np.mean(scores))


def normalize_evidence(evidence) -> list[str]:
    """Evidence field → list of atomic statements (list or ';'-joined str)."""
    if isinstance(evidence, str):
        return split_evidence(evidence)
    stmts = []
    for e in evidence or []:
        stmts.extend(split_evidence(e) if isinstance(e, str) else [])
    return stmts


# ── Shared per-query retrieval context ─────────────────────────────────


class RetrievalBase:
    """Corpus + graph + matrices shared by all retriever arms of one run."""

    def __init__(self, cache, vecs, relations, chunk_text_map, embed_fn):
        self.cache = cache
        self.vecs = vecs
        self.relation_adj = build_relation_adj(relations)
        self.has_relations = bool(self.relation_adj)
        self.chunk_ids = sorted(cache["chunks"].keys())
        self.chunk_text_map = chunk_text_map
        self.graph = build_graph_from_cache(cache, chunk_text_map)
        self.m, self.concepts, self.chunk_index = build_m_matrix(
            self.graph, self.chunk_ids)
        self.concept_index = {c: i for i, c in enumerate(self.concepts)}
        self.embed_fn = embed_fn
        self.chunk_emb = np.asarray(
            embed_fn([chunk_text_map[cid] for cid in self.chunk_ids]),
            dtype=np.float64)
        self._concept_bge = None
        self._sim_cache: dict[str, np.ndarray] = {}
        self._mmt = None

    def concept_bge_emb(self) -> np.ndarray:
        if self._concept_bge is None:
            emb = np.asarray(self.embed_fn(self.concepts), dtype=np.float64)
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            self._concept_bge = emb / np.where(norms > 0, norms, 1.0)
        return self._concept_bge

    def concept_sim(self, which: str) -> np.ndarray:
        if which not in self._sim_cache:
            self._sim_cache[which] = build_concept_sim(
                self.vecs, self.concepts, which)
        return self._sim_cache[which]

    def mmt(self) -> np.ndarray:
        if self._mmt is None:
            self._mmt = self.m @ self.m.T
        return self._mmt

    def seed_and_b0(self, query_vec: np.ndarray) -> tuple[list[str], list[str]]:
        b0 = [cid for cid, _ in cosine_topk_ids(
            query_vec, self.chunk_emb, self.chunk_ids, TOP_K)]
        return list(b0[:SEED_K]), b0

    def seed_indicator_q(self, seed_ids: list[str]) -> np.ndarray:
        q = np.zeros(len(self.concepts), dtype=np.float64)
        for cid in seed_ids:
            for concept in self.graph.chunk_concepts.get(cid, []):
                i = self.concept_index.get(concept)
                if i is not None:
                    q[i] = 1.0
        return q


# ── Part A: retrieval equivalence (matrix vs graph diffusion) ──────────


def matrix_arms(
    base: RetrievalBase, query_vec: np.ndarray, seed_ids: list[str],
    sim_alpha: float, ridge_lambdas: list[float],
) -> dict[str, dict[str, float]]:
    """All matrix retriever arms → {arm_name: {cid: score}} (seeds excluded)."""
    exclude = set(seed_ids)
    arms: dict[str, dict[str, float]] = {}

    q_seed = base.seed_indicator_q(seed_ids)
    arms["matrix_seed_idf"] = scores_to_dict(q_seed @ base.m, base.chunk_ids, exclude)

    if base.has_relations:
        q_rel = concept_weight_vector(
            base.graph, base.relation_adj, seed_ids, base.concept_index,
            weight=REL_WEIGHT_PHASE28, hops=1)
        arms["matrix_relation"] = scores_to_dict(
            q_rel @ base.m, base.chunk_ids, exclude)

    for which, name in (("ws_vec", "ws"), ("wu_vec", "wu")):
        s = base.concept_sim(which)
        q2 = q_seed + sim_alpha * (s @ q_seed)
        arms[f"matrix_sim_{name}"] = scores_to_dict(
            q2 @ base.m, base.chunk_ids, exclude)

    # bge query→concept arm: query semantics enter concept space directly
    q_bge = np.maximum(base.concept_bge_emb() @ (
        query_vec / (np.linalg.norm(query_vec) + 1e-8)), 0.0)
    arms["matrix_bge"] = scores_to_dict(q_bge @ base.m, base.chunk_ids, exclude)

    # Ridge-regularized propagation: (q @ (M M^T + λI)^{-1}) @ M
    for lam in ridge_lambdas:
        a = base.mmt() + lam * np.eye(len(base.concepts))
        q_reg = np.linalg.solve(a.T, q_seed).T  # q_seed @ A^{-1}
        arms[f"matrix_ridge_{lam:g}"] = scores_to_dict(
            q_reg @ base.m, base.chunk_ids, exclude)

    return arms


def graph_arms(
    base: RetrievalBase, seed_ids: list[str],
) -> dict[str, dict[str, float]]:
    """Graph-diffusion retrievers → {arm_name: {cid: score}}."""
    arms = {"graph_flat": propagate_scored(base.graph, seed_ids)}
    if base.has_relations:
        # phase28 optimal config: relation_propagate, discount 0.7, 1 hop
        rel_pairs, _n = relation_propagate(
            base.graph, base.relation_adj, seed_ids, max_propagate=10**9)
        arms["graph_relation"] = dict(rel_pairs)
    return arms


def run_equivalence(
    domain: str = "medical",
    max_queries: int = 50,
    sim_alpha: float = 0.5,
    ridge_lambdas: list[float] | None = None,
    cache_dir: Path = CACHE_DIR,
    out_dir: Path = EXP,
    corpus: dict[str, str] | None = None,
    questions: list[dict] | None = None,
    embed_fn=None,
    verbose: bool = True,
) -> dict:
    """Part A: top-10 overlap + Kendall τ between matrix and graph retrievers."""
    ridge_lambdas = ridge_lambdas or list(DEFAULT_RIDGE_LAMBDAS)
    cache, vecs, relations = load_phase41_inputs(domain, cache_dir)
    if relations is None and verbose:
        print(f"  WARNING: relations_{domain}.json missing — relation arms skipped")
    if corpus is None:
        corpus = load_corpus_texts(domain, cache)
    if questions is None:
        questions = load_questions(domain, max_queries)
    if embed_fn is None:
        from experiments.embed_cache import CachedBgeM3Provider
        embed_fn = CachedBgeM3Provider().embed

    base = RetrievalBase(cache, vecs, relations, corpus, embed_fn)
    if verbose:
        print(f"  [{domain}] {len(base.chunk_ids)} chunks, "
              f"{len(base.concepts)} concepts, {len(questions)} queries, "
              f"relations: {'yes' if base.has_relations else 'NO'}")

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    per_query = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi]
        seed_ids, b0_ids = base.seed_and_b0(qv)
        g_arms = graph_arms(base, seed_ids)
        m_arms = matrix_arms(base, qv, seed_ids, sim_alpha, ridge_lambdas)
        g_rank = {name: merge_ranking(seed_ids, sc, b0_ids)
                  for name, sc in g_arms.items()}
        m_rank = {name: merge_ranking(seed_ids, sc, b0_ids)
                  for name, sc in m_arms.items()}
        rec = {"qid": q.get("id", str(qi)), "level": q.get("level"), "pairs": {}}
        for g_name, g_scores in g_arms.items():
            for m_name, m_scores in m_arms.items():
                rec["pairs"][f"{m_name}__vs__{g_name}"] = {
                    "top10_overlap": topk_overlap(m_rank[m_name], g_rank[g_name]),
                    "kendall_tau": kendall_tau_scores(m_scores, g_scores),
                }
        per_query.append(rec)
        if verbose and (qi + 1) % 10 == 0:
            print(f"    {qi + 1}/{len(questions)}", flush=True)

    # Aggregate per arm-pair
    pair_names = sorted({p for rec in per_query for p in rec["pairs"]})
    agg = {}
    for pair in pair_names:
        ovs = [r["pairs"][pair]["top10_overlap"] for r in per_query
               if pair in r["pairs"]]
        taus = [r["pairs"][pair]["kendall_tau"] for r in per_query
                if pair in r["pairs"] and r["pairs"][pair]["kendall_tau"] is not None]
        ov = float(np.mean(ovs)) if ovs else 0.0
        tau = float(np.mean(taus)) if taus else None
        agg[pair] = {
            "mean_top10_overlap": ov,
            "mean_kendall_tau": tau,
            "n_queries": len(ovs),
            "n_tau": len(taus),
            "verdict": ("matrix_can_replace_graph"
                        if ov >= 0.7 and tau is not None and tau >= 0.6
                        else "not_equivalent"),
        }

    if verbose:
        print(f"\n  {'arm pair':<44} {'overlap':>8} {'tau':>8}  verdict")
        print(f"  {'-' * 78}")
        for pair in pair_names:
            a = agg[pair]
            tau_s = f"{a['mean_kendall_tau']:.3f}" if a["mean_kendall_tau"] is not None else "n/a"
            print(f"  {pair:<44} {a['mean_top10_overlap']:>8.3f} {tau_s:>8}  "
                  f"{a['verdict']}")

    out = {
        "method": "phase41_retrieval_equivalence",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "top_k": TOP_K, "seed_k": SEED_K, "sim_alpha": sim_alpha,
            "ridge_lambdas": ridge_lambdas,
            "relation_weight": REL_WEIGHT_PHASE28, "relation_hops": 1,
            "max_queries": max_queries,
        },
        "n_chunks": len(base.chunk_ids),
        "n_concepts": len(base.concepts),
        "n_queries": len(per_query),
        "has_relations": base.has_relations,
        "verdict_rule": "top10_overlap >= 0.7 and kendall_tau >= 0.6",
        "summary": agg,
        "per_query": per_query,
    }
    out_path = out_dir / f"phase41_equivalence_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  saved to {out_path}")
    return out


# ── Part B: relation-propagation ablation (full 957-chunk scale) ───────


def ablation_retrievers(
    base: RetrievalBase, seed_ids: list[str],
    weights: list[float], hops_list: list[int],
) -> dict[str, dict[str, float]]:
    """B0/flat controls + the weight×hops grid → {method: {cid: score}}."""
    scores = {"B0": {}, "flat": propagate_scored(base.graph, seed_ids)}
    for w in weights:
        for h in hops_list:
            sc, _n = relation_propagate_param(
                base.graph, base.relation_adj, seed_ids, weight=w, hops=h)
            scores[f"rel_w{w:g}_h{h}"] = sc
    return scores


def run_relation_ablation(
    domain: str = "medical",
    max_queries: int = 0,
    quick: bool = False,
    weights: list[float] | None = None,
    hops_list: list[int] | None = None,
    llm_recall: bool = False,
    cache_dir: Path = CACHE_DIR,
    out_dir: Path = EXP,
    corpus: dict[str, str] | None = None,
    questions: list[dict] | None = None,
    embed_fn=None,
    llm_factory=None,
    verbose: bool = True,
) -> dict:
    """Part B: relation weight × hops grid on the full corpus.

    evidence recall: lexical_evidence_recall by default (deterministic, free);
    --llm-recall switches to phase4's LLM judge. ACC: phase26's
    generate_answer + judge_answer_correctness (DeepSeek, llm_factory
    injectable for tests).
    """
    weights = weights if weights is not None else list(DEFAULT_WEIGHTS)
    hops_list = hops_list if hops_list is not None else list(DEFAULT_HOPS)
    cache, vecs, relations = load_phase41_inputs(domain, cache_dir)
    if relations is None:
        raise FileNotFoundError(
            f"relations_{domain}.json not found in {cache_dir}. "
            f"Run phase40_grounding_test.py --build-relations first (GPU).")
    if corpus is None:
        corpus = load_corpus_texts(domain, cache)
    if questions is None:
        questions = load_questions(domain, max_queries)
    if quick:
        questions = questions[:10]
    if embed_fn is None:
        from experiments.embed_cache import CachedBgeM3Provider
        embed_fn = CachedBgeM3Provider().embed
    if llm_factory is None:
        from jgraphrag.llm import DeepSeekProvider
        llm_factory = DeepSeekProvider

    base = RetrievalBase(cache, vecs, relations, corpus, embed_fn)
    method_names = ["B0", "flat"] + [f"rel_w{w:g}_h{h}"
                                     for w in weights for h in hops_list]
    if verbose:
        n_q = len(questions)
        print(f"  [{domain}] {len(base.chunk_ids)} chunks, "
              f"{len(base.concepts)} concepts, {n_q} queries, "
              f"{len(method_names)} methods")
        print(f"  API estimate: ~{2 * len(method_names) * n_q} DeepSeek calls "
              f"(answer + judge per method per query)"
              + (f" + {len(method_names) * n_q} recall judge" if llm_recall
                 else " (recall: lexical, free)"))

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    # Phase 1: retrieval for every method × query (local, no LLM)
    contexts: list[dict] = []
    for qi, q in enumerate(questions):
        seed_ids, b0_ids = base.seed_and_b0(query_emb[qi])
        method_scores = ablation_retrievers(base, seed_ids, weights, hops_list)
        method_scores["B0"] = {}  # B0 = pure cosine ranking
        entry = {"qid": q.get("id", str(qi)), "level": q.get("level"),
                 "question": q["question"], "answer": q.get("answer", ""),
                 "evidence": normalize_evidence(q.get("evidence")), "ctx": {}}
        for name, sc in method_scores.items():
            if name == "B0":
                ranked = b0_ids[:TOP_K]
            else:
                ranked = merge_ranking(seed_ids, sc, b0_ids)
            entry["ctx"][name] = " ".join(
                base.chunk_text_map[cid] for cid in ranked)
        contexts.append(entry)
        if verbose and (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)

    # Phase 2: evaluation (LLM for ACC; lexical or LLM for recall)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi_entry):
        qi, entry = qi_entry
        llm = llm_factory()
        rec = {"qid": entry["qid"], "level": entry["level"], "methods": {}}
        for name, ctx in entry["ctx"].items():
            if llm_recall:
                recall = llm_judge_evidence_recall(
                    entry["question"], ctx, entry["evidence"], llm)
            else:
                recall = lexical_evidence_recall(entry["evidence"], ctx)
            acc = None
            if entry["answer"]:
                ans = generate_answer(entry["question"], ctx, llm)
                acc = bool(judge_answer_correctness(
                    entry["question"], ans, entry["answer"], llm))
            rec["methods"][name] = {"recall": recall, "acc": acc}
        return qi, rec

    per_query: list[dict | None] = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_eval, (i, e)) for i, e in enumerate(contexts)]
        done = 0
        for future in as_completed(futures):
            qi, rec = future.result()
            per_query[qi] = rec
            done += 1
            if verbose and done % 10 == 0:
                print(f"    eval {done}/{len(contexts)}", flush=True)
    per_query = [r for r in per_query if r is not None]

    # Aggregate: overall + by level
    levels = ["L1", "L2", "L3", "L4"]
    results = {}
    for name in method_names:
        accs = [1.0 if r["methods"][name]["acc"] else 0.0
                for r in per_query if r["methods"][name]["acc"] is not None]
        recalls = [r["methods"][name]["recall"] for r in per_query]
        by_level = {}
        for lv in levels:
            rs = [r for r in per_query if r["level"] == lv]
            lv_accs = [1.0 if r["methods"][name]["acc"] else 0.0
                       for r in rs if r["methods"][name]["acc"] is not None]
            by_level[lv] = {
                "acc": float(np.mean(lv_accs)) if lv_accs else None,
                "recall": (float(np.mean([r["methods"][name]["recall"]
                                          for r in rs])) if rs else None),
                "n": len(rs),
            }
        results[name] = {
            "acc": float(np.mean(accs)) if accs else None,
            "recall": float(np.mean(recalls)) if recalls else None,
            "by_level": by_level,
            "n": len(per_query),
        }

    if verbose:
        print(f"\n  {'method':<14} {'ACC':>7} {'recall':>7} "
              f"{'L1':>14} {'L2':>14} {'L3':>14} {'L4':>14}")
        print(f"  {'-' * 88}")
        for name in method_names:
            r = results[name]
            cells = []
            for lv in levels:
                b = r["by_level"][lv]
                cells.append(f"{b['acc']:.2f}/{b['recall']:.2f}"
                             if b["acc"] is not None else "n/a")
            acc_s = f"{r['acc']:.3f}" if r["acc"] is not None else "n/a"
            print(f"  {name:<14} {acc_s:>7} {r['recall']:>7.3f} "
                  f"{cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}")
        print("  (level cells: ACC/recall)")

    out = {
        "method": "phase41_relation_ablation",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "weights": weights, "hops": hops_list, "top_k": TOP_K,
            "seed_k": SEED_K, "quick": quick, "llm_recall": llm_recall,
            "max_queries": max_queries,
        },
        "n_chunks": len(base.chunk_ids),
        "n_concepts": len(base.concepts),
        "n_queries": len(per_query),
        "results": results,
        "per_query": per_query,
    }
    out_path = out_dir / f"phase41_relation_ablation_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if verbose:
        print(f"  saved to {out_path}")
    return out


# ── Synthetic self-tests (no model, no real data files, no API) ────────


def _synthetic_fixture(tmp: Path) -> dict:
    """Deterministic synthetic domain: 4 concepts, 6 chunks, 1 relation edge.

    Written in the exact phase39/40 cache formats to a temp dir, so the real
    loaders/builders run against it. Deterministic fake embed_fn: token-level
    seeded random vectors, averaged (hash-based, no model, no RNG state).
    """
    rng = np.random.default_rng(11)
    dim = 32

    chunk_concepts = {
        "Syn::chunk_0": ["alpha", "beta"],
        "Syn::chunk_1": ["alpha", "gamma"],
        "Syn::chunk_2": ["beta", "gamma"],
        "Syn::chunk_3": ["delta"],
        "Syn::chunk_4": ["alpha", "beta", "gamma", "delta"],
        "Syn::chunk_5": [],
        "Syn::chunk_6": ["alpha"],
        "Syn::chunk_7": ["beta"],
        "Syn::chunk_8": ["gamma", "delta"],
        "Syn::chunk_9": ["alpha", "delta"],
        "Syn::chunk_10": ["beta", "delta"],
        "Syn::chunk_11": [],
    }
    chunk_texts = {
        "Syn::chunk_0": "alpha alpha beta tail",
        "Syn::chunk_1": "alpha gamma tail",
        "Syn::chunk_2": "beta gamma gamma tail",
        "Syn::chunk_3": "delta delta delta tail",
        "Syn::chunk_4": "alpha beta gamma delta tail",
        "Syn::chunk_5": "unrelated filler words only",
        "Syn::chunk_6": "alpha alpha alpha tail",
        "Syn::chunk_7": "beta tail",
        "Syn::chunk_8": "gamma delta tail",
        "Syn::chunk_9": "alpha delta delta tail",
        "Syn::chunk_10": "beta beta delta tail",
        "Syn::chunk_11": "more unrelated filler here",
    }
    concept_chunks: dict[str, list[str]] = defaultdict(list)
    for cid, cs in chunk_concepts.items():
        for c in cs:
            concept_chunks[c].append(cid)

    cache = {
        "domain": "synthetic",
        "n_chunks": len(chunk_concepts),
        "chunks": {cid: {"concepts": cs, "roles": {}}
                   for cid, cs in chunk_concepts.items()},
        "concept_chunks": dict(concept_chunks),
        "n_unique_concepts": len(concept_chunks),
    }
    concepts = sorted(concept_chunks)
    (tmp / "concept_cache_synthetic_twopass.json").write_text(json.dumps(cache))
    np.savez(tmp / "concept_vecs_synthetic.npz",
             concepts=np.array(concepts),
             wu_vec=rng.normal(size=(len(concepts), dim)).astype(np.float32),
             ws_vec=rng.normal(size=(len(concepts), dim)).astype(np.float32),
             count=np.array([len(concept_chunks[c]) for c in concepts]))
    relations = {
        "domain": "synthetic",
        "concepts": concepts,
        "edges": [{"concept_a": "alpha", "concept_b": "delta",
                   "relation": "causes", "prob": 0.9,
                   "layer_type": "intra", "context_strategy": "shared"}],
    }
    (tmp / "relations_synthetic.json").write_text(json.dumps(relations))

    def fake_embed(texts):
        # Stable per-token vectors (sha256-seeded; str hash() is salted per
        # process and would make the self-test nondeterministic).
        import hashlib

        vecs = []
        for text in texts:
            acc = np.zeros(dim)
            for tok in re.findall(r"[a-zA-Z]{2,}", text.lower()):
                seed = int.from_bytes(
                    hashlib.sha256(tok.encode()).digest()[:4], "little")
                acc += np.random.default_rng(seed).normal(size=dim)
            if np.linalg.norm(acc) == 0:
                acc[0] = 1.0
            vecs.append(acc.tolist())
        return vecs

    questions = [{
        "id": "Syn-q0", "question": "alpha beta", "level": "L1",
        "answer": "alpha and beta", "evidence": ["alpha beta tail"],
    }]
    return {"cache": cache, "chunk_texts": chunk_texts,
            "embed_fn": fake_embed, "questions": questions,
            "concepts": concepts}


def run_selftest_a(tmp_root: Path | None = None) -> None:
    """Part A self-test: matrix propagation ≡ graph propagation on planted data.

    Case 1 (equivalent): q = indicator(seed concepts) — by construction
    q @ M = propagate's internal scores, so top-10 overlap = 1.0 and τ = 1.0.
    Also: the relation matrix arm (indicator semantics, w=0.7, 1 hop) must
    equal phase28.relation_propagate score-for-score.
    Case 2 (negative control): reversed chunk scores — overlap and τ must drop.
    """
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="phase41_selftest_a_",
                                dir=tmp_root or tempfile.gettempdir()))
    fx = _synthetic_fixture(tmp)

    cache, vecs, relations = load_phase41_inputs("synthetic", tmp)
    base = RetrievalBase(cache, vecs, relations, fx["chunk_texts"], fx["embed_fn"])
    qv = np.asarray(fx["embed_fn"](["alpha beta"]), dtype=np.float64)[0]
    seed_ids, b0_ids = base.seed_and_b0(qv)
    # Keep only 2 seeds so >=2 non-seed candidates remain (Kendall τ needs
    # at least 2 items); SEED_K=5 would leave a single candidate on 6 chunks.
    seed_ids = seed_ids[:2]
    seed_keep = 2

    failures = []

    # propagate_scored must reproduce ConceptGraph.propagate's ordering
    g_flat = propagate_scored(base.graph, seed_ids)
    g_list = base.graph.propagate(seed_ids, max_propagate=10**9,
                                  use_idf=True, use_bm25=True)
    if g_list != [cid for cid, _ in sorted(g_flat.items(),
                                           key=lambda x: x[1], reverse=True)]:
        failures.append("propagate_scored ordering != ConceptGraph.propagate")

    # Case 1: matrix_seed_idf ≡ graph_flat (exact score equality)
    m_arms = matrix_arms(base, qv, seed_ids, sim_alpha=0.5,
                         ridge_lambdas=[0.1])
    m_flat = m_arms["matrix_seed_idf"]
    keys = set(g_flat) | set(m_flat)
    if not np.allclose([g_flat.get(k, 0.0) for k in keys],
                       [m_flat.get(k, 0.0) for k in keys], atol=1e-9):
        failures.append("matrix_seed_idf scores != graph_flat scores")
    g_rank = merge_ranking(seed_ids, g_flat, b0_ids, seed_keep=seed_keep)
    m_rank = merge_ranking(seed_ids, m_flat, b0_ids, seed_keep=seed_keep)
    ov = topk_overlap(m_rank, g_rank)
    tau = kendall_tau_scores(m_flat, g_flat)
    if ov != 1.0:
        failures.append(f"equivalent case: top10 overlap {ov} != 1.0")
    if tau is None or tau < 0.999:
        failures.append(f"equivalent case: kendall tau {tau} < 0.999")

    # Case 1b: relation matrix arm ≡ phase28.relation_propagate (exact)
    rel_pairs, _n = relation_propagate(base.graph, base.relation_adj,
                                       seed_ids, max_propagate=10**9)
    g_rel = dict(rel_pairs)
    m_rel = m_arms["matrix_relation"]
    keys = set(g_rel) | set(m_rel)
    if not np.allclose([g_rel.get(k, 0.0) for k in keys],
                       [m_rel.get(k, 0.0) for k in keys], atol=1e-9):
        failures.append("matrix_relation scores != phase28 relation_propagate")
    # relation_propagate_param(w=0.7, h=1) must also match phase28 exactly
    p_scores, _ = relation_propagate_param(base.graph, base.relation_adj,
                                           seed_ids, weight=0.7, hops=1)
    keys = set(p_scores) | set(g_rel)
    if not np.allclose([p_scores.get(k, 0.0) for k in keys],
                       [g_rel.get(k, 0.0) for k in keys], atol=1e-9):
        failures.append("relation_propagate_param(0.7,1) != phase28 relation_propagate")

    # Case 2: reversed scores (deterministic negative control) — the merged
    # non-seed slots pull the *lowest*-scored chunks, so with a small top_k
    # the overlap must drop below 1.0 and τ must go strongly negative.
    m_rev = {cid: -s for cid, s in g_flat.items()}
    ov2 = topk_overlap(merge_ranking(seed_ids, m_rev, b0_ids,
                                     top_k=4, seed_keep=seed_keep),
                       merge_ranking(seed_ids, g_flat, b0_ids,
                                     top_k=4, seed_keep=seed_keep))
    tau2 = kendall_tau_scores(m_rev, g_flat)
    if ov2 >= 1.0 or tau2 is None or tau2 >= 0.0:
        failures.append(f"unrelated case did NOT drop: overlap={ov2}, tau={tau2}")

    # ws/wu/ridge arms must run and produce scores
    for arm in ("matrix_sim_ws", "matrix_sim_wu", "matrix_bge",
                    "matrix_ridge_0.1"):
        if arm not in m_arms:
            failures.append(f"arm {arm} missing")

    if failures:
        raise AssertionError("SELF-TEST A FAILED: " + "; ".join(failures))
    print(f"  case1 (equivalent): overlap={ov:.3f}, tau={tau:.3f}")
    print(f"  case2 (unrelated):  overlap={ov2:.3f}, "
          f"tau={tau2 if tau2 is None else round(tau2, 3)}")
    print(f"  SELF-TEST A PASSED (artifacts in {tmp})")


class _MockMessage:
    def __init__(self, content):
        self.content = content
        self.is_error = False


class _MockLLM:
    """DeepSeek surface: complete() → message with .content / .is_error."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, max_tokens=200):
        self.calls += 1
        if "YES or NO" in prompt:
            return _MockMessage("YES")
        return _MockMessage("mock answer")


def run_selftest_b(tmp_root: Path | None = None) -> None:
    """Part B self-test: grid loop + aggregation with mock LLM (zero API calls)."""
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="phase41_selftest_b_",
                                dir=tmp_root or tempfile.gettempdir()))
    fx = _synthetic_fixture(tmp)
    questions = [
        {"id": f"Syn-q{i}", "question": "alpha beta", "level": f"L{i % 2 + 1}",
         "answer": "alpha and beta", "evidence": ["alpha beta tail"]}
        for i in range(3)
    ]
    out = run_relation_ablation(
        domain="synthetic", questions=questions, corpus=fx["chunk_texts"],
        embed_fn=fx["embed_fn"], llm_factory=_MockLLM,
        weights=[0.0, 0.5], hops_list=[1, 2],
        cache_dir=tmp, out_dir=tmp, verbose=False)

    failures = []
    expected_methods = ["B0", "flat", "rel_w0_h1", "rel_w0_h2",
                        "rel_w0.5_h1", "rel_w0.5_h2"]
    if sorted(out["results"].keys()) != sorted(expected_methods):
        failures.append(f"methods mismatch: {sorted(out['results'])}")
    for name in expected_methods:
        r = out["results"].get(name, {})
        if r.get("n") != 3:
            failures.append(f"{name}: n={r.get('n')} != 3")
        if r.get("acc") is None or not (0.0 <= r["acc"] <= 1.0):
            failures.append(f"{name}: bad acc {r.get('acc')}")
        if r.get("recall") is None or not (0.0 <= r["recall"] <= 1.0):
            failures.append(f"{name}: bad recall {r.get('recall')}")
        if "L4" not in r.get("by_level", {}):
            failures.append(f"{name}: by_level incomplete")
    # weight=0 must reduce exactly to flat (deterministic recall identical)
    for h in (1, 2):
        if out["results"][f"rel_w0_h{h}"]["recall"] != out["results"]["flat"]["recall"]:
            failures.append(f"rel_w0_h{h} recall != flat recall (should reduce to flat)")
    if not (tmp / "phase41_relation_ablation_synthetic.json").exists():
        failures.append("output JSON not written")

    if failures:
        raise AssertionError("SELF-TEST B FAILED: " + "; ".join(failures))
    print(f"  grid methods evaluated: {len(out['results'])} x "
          f"{out['n_queries']} queries (mock LLM, 0 real API calls)")
    print(f"  SELF-TEST B PASSED (artifacts in {tmp})")


# ── main ───────────────────────────────────────────────────────────────


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(
        description="Phase 41: matrix-vs-graph retrieval equivalence + "
                    "relation-propagation ablation (full scale)")
    ap.add_argument("--part", choices=["a", "b", "both"], default="both")
    ap.add_argument("--domain", default="medical", choices=["medical", "novel"])
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = auto (A: 50, B: 56-question full set)")
    ap.add_argument("--quick", action="store_true",
                    help="Part B: 10 queries per cell instead of the full set")
    ap.add_argument("--llm-recall", action="store_true",
                    help="Part B: LLM judge for evidence recall (default: "
                         "deterministic lexical recall, free)")
    ap.add_argument("--weights", type=float, nargs="+", default=None,
                    help="Part B relation weights (default: 0 0.1 0.3 0.5)")
    ap.add_argument("--hops", type=int, nargs="+", default=None,
                    help="Part B relation hops (default: 1 2)")
    ap.add_argument("--sim-alpha", type=float, default=0.5,
                    help="Part A: weight of ws/wu vector-similarity expansion")
    ap.add_argument("--ridge-lambdas", type=float, nargs="+", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="Run synthetic-data self-tests (A + B) and exit")
    args = ap.parse_args()

    if args.selftest:
        print("Phase 41 self-tests (synthetic data, no model, no API)")
        run_selftest_a()
        run_selftest_b()
        return

    if args.part in ("a", "both"):
        mq = args.max_queries if args.max_queries > 0 else 50
        print(f"\n{'=' * 60}\nPhase 41-A: retrieval equivalence [{args.domain}]\n"
              f"{'=' * 60}")
        run_equivalence(domain=args.domain, max_queries=mq,
                        sim_alpha=args.sim_alpha,
                        ridge_lambdas=args.ridge_lambdas)
    if args.part in ("b", "both"):
        print(f"\n{'=' * 60}\nPhase 41-B: relation ablation [{args.domain}]\n"
              f"{'=' * 60}")
        run_relation_ablation(domain=args.domain,
                              max_queries=args.max_queries,
                              quick=args.quick,
                              weights=args.weights, hops_list=args.hops,
                              llm_recall=args.llm_recall)


if __name__ == "__main__":
    main()
