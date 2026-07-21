"""Phase 10 Stage 7a — J-Lens 概念检索 vs 纯向量检索 Benchmark（NFCorpus/scifact）。

验证产品核心命题：J-Lens 概念聚类增强的检索能否超越纯 bge-m3 向量检索？
如果 J-Lens 方案在 BEIR benchmark 上达到 B0 的 60-70%，结合其零 API 成本，
就构成对传统 GraphRAG 的明确产品差异化。

四种检索方案对比：
  B0:    纯 bge-m3 余弦（baseline，普通 RAG）
  B0+:   B0 + LLM 精细重嵌入（已验证的增强，Phase 4）
  JL-1:  J-Lens 概念簇传播（B0 召回 → query 找最近概念簇 → 扩展邻居簇）
  JL-2:  J-Lens 概念簇 rerank（B0 召回 top-100 → 用概念簇 membership 重排）

J-Lens 检索的设计原则：
  - 不替代 B0 召回（B0 保证不丢文档），在 B0 候选集内用概念信号重排/扩展
  - 概念聚类用 Stage 6 验证的 J-Lens 残差（L26 transport）
  - query→概念匹配：query 也提取 J-Lens 残差，找最近概念簇

指标：nDCG@{1,3,10}, Recall@{10,100}, MRR@10 + paired permutation test

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage7a
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.phase4_retrieval import (
    load_beir_benchmark, evaluate_baseline, per_query_ndcg,
    paired_permutation_test, K_VALUES, TOP_K,
)
from experiments.baselines import FlatSearch
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── J-Lens concept search baselines ────────────────────────────────────

class JLensConceptExpand:
    """J-Lens 概念簇传播检索。

    流程：
      1. [fit] 用 J-Lens 残差聚类文档 → 概念簇；算簇间 Sigmoid 图
      2. [search] query → J-Lens 残差 → 最近簇 → B0 召回 + 邻居簇扩展

    B0 召回保证不丢文档（recall_k=100），概念扩展在候选集内加权邻居簇文档。
    score = bge_cosine * (1 + expand_weight * is_neighbor_cluster)
    """

    name = "JL-1"

    def __init__(self, recall_k: int = 100, expand_weight: float = 0.3):
        self.recall_k = recall_k
        self.expand_weight = expand_weight

    def _setup_concepts(self, jlens_vecs: np.ndarray, bge_vecs: np.ndarray,
                        corpus_ids: list[str]):
        """Cluster docs by J-Lens residual, build cluster graph."""
        # Normalize for cosine
        norms = np.linalg.norm(jlens_vecs, axis=1, keepdims=True)
        jlens_normed = jlens_vecs / norms.clip(min=1e-8)

        # HDBSCAN cluster on J-Lens residual
        clusters = partition_hdbscan(jlens_vecs.tolist())
        # compute centroids
        self.cluster_centroids = {}
        self.doc_to_cluster = {}
        for cid, members in clusters.items():
            centroid = jlens_normed[members].mean(axis=0)
            cn = np.linalg.norm(centroid)
            self.cluster_centroids[cid] = centroid / cn if cn > 0 else centroid
            for m in members:
                self.doc_to_cluster[m] = cid

        # cluster graph: cosine between centroids, keep top-k neighbors
        cids = list(self.cluster_centroids.keys())
        self.cluster_neighbors = {}
        for i, ci in enumerate(cids):
            sims = []
            for j, cj in enumerate(cids):
                if i == j:
                    continue
                sim = float(np.dot(self.cluster_centroids[ci], self.cluster_centroids[cj]))
                sims.append((cj, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            self.cluster_neighbors[ci] = [c for c, _ in sims[:3]]  # 1-3 neighbors

        self.clusters = clusters
        self.corpus_ids = corpus_ids

    def fit(self, corpus_emb, corpus_ids):
        """corpus_emb is bge-m3; J-Lens residuals set separately via set_jlens."""
        self.bge_norm = _normalize_np(np.asarray(corpus_emb, dtype=np.float32))
        self.corpus_ids_bge = corpus_ids

    def set_jlens(self, jlens_vecs: np.ndarray, corpus_ids: list[str]):
        """Set J-Lens residuals and build concept clusters."""
        self._setup_concepts(jlens_vecs, self.bge_norm, corpus_ids)

    def set_query_jlens(self, query_jlens: np.ndarray):
        """Set J-Lens residuals for queries (for cluster matching)."""
        norms = np.linalg.norm(query_jlens, axis=1, keepdims=True)
        self.query_jlens_normed = query_jlens / norms.clip(min=1e-8)

    def search(self, query_emb, query_ids, top_k=100):
        results = {}
        q_bge = _normalize_np(np.asarray(query_emb, dtype=np.float32))

        for i, qid in enumerate(query_ids):
            # B0 recall
            scores_bge = q_bge[i] @ self.bge_norm.T
            top_idx = np.argsort(scores_bge)[::-1][:self.recall_k]

            # find query's nearest cluster via J-Lens residual
            q_jl = self.query_jlens_normed[i]
            cluster_sims = {cid: float(np.dot(q_jl, c))
                            for cid, c in self.cluster_centroids.items()}
            if cluster_sims:
                best_cluster = max(cluster_sims, key=cluster_sims.get)
                neighbor_clusters = set([best_cluster] + self.cluster_neighbors.get(best_cluster, []))
            else:
                neighbor_clusters = set()

            # rerank: boost docs in neighbor clusters
            reranked = []
            for idx in top_idx:
                base_score = float(scores_bge[idx])
                doc_cluster = self.doc_to_cluster.get(idx)
                boost = self.expand_weight if doc_cluster in neighbor_clusters else 0.0
                reranked.append((idx, base_score * (1 + boost)))
            reranked.sort(key=lambda x: x[1], reverse=True)

            results[qid] = {self.corpus_ids_bge[idx]: score
                            for idx, score in reranked[:top_k]}
        return results


class JLensRerank:
    """J-Lens 概念簇 rerank（不扩展，只重排 B0 候选）。

    更保守：在 B0 top-100 内，用「query 和 doc 是否同概念簇」加权重排。
    不引入新文档，只改变排序。
    """

    name = "JL-2"

    def __init__(self, recall_k: int = 100, concept_weight: float = 0.15):
        self.recall_k = recall_k
        self.concept_weight = concept_weight

    def fit(self, corpus_emb, corpus_ids):
        self.bge_norm = _normalize_np(np.asarray(corpus_emb, dtype=np.float32))
        self.corpus_ids_bge = corpus_ids

    def set_jlens(self, jlens_vecs: np.ndarray, corpus_ids: list[str]):
        norms = np.linalg.norm(jlens_vecs, axis=1, keepdims=True)
        jlens_normed = jlens_vecs / norms.clip(min=1e-8)
        clusters = partition_hdbscan(jlens_vecs.tolist())
        self.doc_to_cluster = {}
        self.cluster_centroids = {}
        for cid, members in clusters.items():
            centroid = jlens_normed[members].mean(axis=0)
            cn = np.linalg.norm(centroid)
            self.cluster_centroids[cid] = centroid / cn if cn > 0 else centroid
            for m in members:
                self.doc_to_cluster[m] = cid

    def set_query_jlens(self, query_jlens: np.ndarray):
        norms = np.linalg.norm(query_jlens, axis=1, keepdims=True)
        self.query_jlens_normed = query_jlens / norms.clip(min=1e-8)

    def search(self, query_emb, query_ids, top_k=100):
        results = {}
        q_bge = _normalize_np(np.asarray(query_emb, dtype=np.float32))
        for i, qid in enumerate(query_ids):
            scores_bge = q_bge[i] @ self.bge_norm.T
            top_idx = np.argsort(scores_bge)[::-1][:self.recall_k]

            # query cluster
            q_jl = self.query_jlens_normed[i]
            cluster_sims = {cid: float(np.dot(q_jl, c))
                            for cid, c in self.cluster_centroids.items()}
            q_cluster = max(cluster_sims, key=cluster_sims.get) if cluster_sims else None

            reranked = []
            for idx in top_idx:
                base = float(scores_bge[idx])
                same = (self.doc_to_cluster.get(idx) == q_cluster)
                reranked.append((idx, base + (self.concept_weight if same else 0)))
            reranked.sort(key=lambda x: x[1], reverse=True)
            results[qid] = {self.corpus_ids_bge[idx]: score
                            for idx, score in reranked[:top_k]}
        return results


def _normalize_np(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms.clip(min=1e-8)


# ── Main experiment ────────────────────────────────────────────────────

def run_benchmark(name: str, corpus, queries, qrels, embed, lens, lens_model,
                  tokenizer, max_docs: int = 500):
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    # Prepare corpus + queries as text lists
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid].get("text", "") or corpus[cid].get("title", "")
                    for cid in corpus_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    # filter to queries with qrels
    query_ids = [qid for qid in query_ids if qrels.get(qid)]
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  {len(corpus_ids)} docs, {len(query_ids)} queries")

    # bge-m3 embeddings
    print(f"  embedding corpus + queries (bge-m3)...", end="", flush=True)
    corpus_emb = np.asarray(embed.embed(corpus_texts), dtype=np.float32)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float32)
    print(f" done")

    # J-Lens residuals (subsample corpus if too large)
    n_for_jlens = min(len(corpus_ids), max_docs)
    print(f"  extracting J-Lens residuals for {n_for_jlens} docs + {len(query_ids)} queries...")
    layer = lens.source_layers[-1]
    # corpus residuals
    doc_prompts = [build_topic_prompt(t, tokenizer) for t in corpus_texts[:n_for_jlens]]
    corpus_jlens = extract_residuals(lens, lens_model, tokenizer, doc_prompts, layer,
                                     max_seq_len=256)
    # query residuals
    query_prompts = [build_topic_prompt(q, tokenizer) for q in query_texts]
    query_jlens = extract_residuals(lens, lens_model, tokenizer, query_prompts, layer,
                                    max_seq_len=128)

    # Pad corpus_jlens if subsampled
    if n_for_jlens < len(corpus_ids):
        padded = np.zeros((len(corpus_ids), corpus_jlens.shape[1]), dtype=np.float32)
        padded[:n_for_jlens] = corpus_jlens
        corpus_jlens = padded

    # Evaluate baselines
    baselines = {
        "B0": FlatSearch(),
        "JL-1": JLensConceptExpand(recall_k=100, expand_weight=0.3),
        "JL-2": JLensRerank(recall_k=100, concept_weight=0.15),
    }

    metrics = {}
    all_results = {}
    for label, bl in baselines.items():
        print(f"\n  --- {label} ---")
        if hasattr(bl, "set_jlens"):
            bl.fit(corpus_emb, corpus_ids)
            bl.set_jlens(corpus_jlens, corpus_ids)
            bl.set_query_jlens(query_jlens)
        else:
            bl.fit(corpus_emb, corpus_ids)
        results = bl.search(query_emb, query_ids, top_k=TOP_K)
        m = evaluate_baseline_simple(bl, corpus_emb, corpus_ids, query_emb, query_ids, qrels)
        metrics[label] = m
        all_results[label] = results
        print(f"    nDCG@10={m['ndcg_10']:.4f} Recall@10={m['recall_10']:.4f} "
              f"Recall@100={m['recall_100']:.4f} MRR={m['mrr']:.4f}")

    # Permutation tests (JL vs B0)
    perms = {}
    b0_pq = per_query_ndcg(all_results["B0"], qrels, k=10)
    for label in ("JL-1", "JL-2"):
        jl_pq = per_query_ndcg(all_results[label], qrels, k=10)
        perms[f"{label}_vs_B0"] = paired_permutation_test(jl_pq, b0_pq)

    return {
        "n_docs": len(corpus_ids),
        "n_queries": len(query_ids),
        "results": metrics,
        "permutation": perms,
    }


def evaluate_baseline_simple(baseline, corpus_emb, corpus_ids, query_emb,
                             query_ids, qrels) -> dict:
    """Evaluate using BEIR metrics."""
    from beir.retrieval.evaluation import EvaluateRetrieval
    results = baseline.search(query_emb, query_ids, top_k=TOP_K)
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, K_VALUES, "mrr")
    return {
        "ndcg_10": ndcg.get("NDCG@10", 0),
        "ndcg_3": ndcg.get("NDCG@3", 0),
        "ndcg_1": ndcg.get("NDCG@1", 0),
        "recall_10": recall.get("Recall@10", 0),
        "recall_100": recall.get("Recall@100", 0),
        "mrr": mrr.get("MRR@10", 0) if isinstance(mrr, dict) else mrr,
    }


def build_topic_prompt(text: str, tokenizer) -> str:
    """Concern prompt for J-Lens residual extraction (doc or query)."""
    user_msg = f"What is the main topic? One word.\n\n{text[:500]}"
    prefill = "The main topic is"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 7a: retrieval benchmark")
    ap.add_argument("--max-docs", type=int, default=500,
                    help="max docs for J-Lens residual extraction (caps runtime)")
    ap.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact"])
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/3] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/3] Wrapping with jlens.from_hf...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    print(f"\n[3/3] Running retrieval benchmarks...")
    embed = CachedBgeM3Provider()

    all_results = {"benchmarks": {}}
    for name in args.datasets:
        corpus, queries, qrels = load_beir_benchmark(name)
        all_results["benchmarks"][name] = run_benchmark(
            name, corpus, queries, qrels, embed, lens, lens_model, tokenizer,
            args.max_docs
        )

    # Summary
    print(f"\n{'='*70}")
    print("=== SUMMARY (nDCG@10 / Recall@100) ===")
    print(f"{'Benchmark':<12} {'B0':>14} {'JL-1':>14} {'JL-2':>14}")
    for name, res in all_results["benchmarks"].items():
        r = res["results"]
        b0 = f"{r['B0']['ndcg_10']:.3f}/{r['B0']['recall_100']:.3f}"
        j1 = f"{r['JL-1']['ndcg_10']:.3f}/{r['JL-1']['recall_100']:.3f}"
        j2 = f"{r['JL-2']['ndcg_10']:.3f}/{r['JL-2']['recall_100']:.3f}"
        print(f"{name:<12} {b0:>14} {j1:>14} {j2:>14}")

    # JL as % of B0
    print(f"\n=== J-Lens as % of B0 (nDCG@10) ===")
    for name, res in all_results["benchmarks"].items():
        r = res["results"]
        b0 = r["B0"]["ndcg_10"]
        for label in ("JL-1", "JL-2"):
            pct = r[label]["ndcg_10"] / b0 * 100 if b0 > 0 else 0
            p = res["permutation"].get(f"{label}_vs_B0", {})
            sig = "✓" if p.get("significant") else ""
            delta = p.get("delta", 0)
            print(f"  {name} {label}: {pct:.0f}% of B0 (Δ{delta:+.4f}{sig})")

    out_path = EXP / "phase10_stage7a_benchmark.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
