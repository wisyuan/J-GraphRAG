"""Phase 38: 验证矩阵加速检索 vs 传统图扩散。

核心命题：
1. BPE 碎片的 W_U 行向量能否有效表示概念方向？
2. 概念关系图社区 ≈ 向量聚类？（双射一致性 ARI/NMI）
3. 矩阵乘法检索 ≈ 图传播检索？（检索效果对比）

实验设计：
  A. 构建概念-文档映射矩阵 M（从 Phase 25 的概念图）
  B. 提取概念向量 E（两种方案对比）：
     E1: bge-m3 嵌入（baseline）
     E2: BPE 碎片 W_U 行向量（新方案）
  C. 概念关系图 W 上的 Leiden 社区 vs E 上的 K-Means 聚类
     → 计算 ARI/NMI
  D. 三种检索对比：
     R1: 传统图传播（Phase 25 baseline）
     R2: 矩阵乘法 q × M（用 E2 概念向量）
     R3: 矩阵乘法 + Ridge 正则化 (M×M^T + λI)^{-1}

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase38_matrix_acceleration
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage7c import (
    ConceptGraph, cosine_topk_ids, extract_chunk_concepts,
)
from experiments.phase25_filter_bpe_benchmark import (
    phase25_filter_with_bpe, rebuild_graph_index,
)
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
TOP_K = 10


def get_bpe_fragment_vectors(concepts: list[str], model, tokenizer) -> dict[str, np.ndarray]:
    """Get W_U row vectors for core BPE fragments of each concept.

    For multi-token concepts, use the first BPE fragment (core semantic).
    """
    # Get the unembedding matrix (lm_head weight)
    lm_head = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    # lm_head shape: [vocab_size, hidden_size]

    vectors = {}
    for concept in concepts:
        # Tokenize concept — get BPE fragments
        token_ids = tokenizer.encode(concept, add_special_tokens=False)
        if not token_ids:
            continue

        # Use first fragment (core semantic direction)
        # But also try averaging first 2 fragments for stability
        if len(token_ids) == 1:
            vec = lm_head[token_ids[0]]
        else:
            # Average first 2 fragments (weighted by inverse frequency if available)
            n = min(2, len(token_ids))
            vec = np.mean(lm_head[token_ids[:n]], axis=0)

        vectors[concept.lower()] = vec / (np.linalg.norm(vec) + 1e-8)

    return vectors


def build_matrices(graph: ConceptGraph, chunk_ids: list[str]) -> dict:
    """Build concept-document mapping matrix M and relation adjacency W."""
    concepts = sorted(graph.concept_chunks.keys())
    n_concepts = len(concepts)
    n_chunks = len(chunk_ids)

    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    chunk_to_idx = {c: i for i, c in enumerate(chunk_ids)}

    # M: concept × chunk matrix (IDF × BM25 tf)
    import scipy.sparse as sp
    M_data = []
    M_row = []
    M_col = []

    for concept, chunks in graph.concept_chunks.items():
        ci = concept_to_idx[concept]
        idf = graph.idf.get(concept, 1.0)
        for cid in chunks:
            if cid in chunk_to_idx:
                cj = chunk_to_idx[cid]
                tf = graph.tf.get((cid, concept), 1)
                tf_weight = graph._bm25_tf_norm(cid, tf)
                M_data.append(idf * tf_weight)
                M_row.append(ci)
                M_col.append(cj)

    M = sp.csr_matrix((M_data, (M_row, M_col)), shape=(n_concepts, n_chunks))

    # W: concept × concept co-occurrence matrix
    W_data = []
    W_row = []
    W_col = []

    for c_a, neighbors in graph.concept_cooccur.items():
        if c_a not in concept_to_idx:
            continue
        for c_b, count in neighbors.items():
            if c_b not in concept_to_idx:
                continue
            W_data.append(float(count))
            W_row.append(concept_to_idx[c_a])
            W_col.append(concept_to_idx[c_b])

    W = sp.csr_matrix((W_data, (W_row, W_col)), shape=(n_concepts, n_concepts))

    return {
        "M": M,
        "W": W,
        "concepts": concepts,
        "concept_to_idx": concept_to_idx,
        "chunk_to_idx": chunk_to_idx,
        "n_concepts": n_concepts,
        "n_chunks": n_chunks,
    }


def matrix_retrieve(query_vec, chunk_emb, chunk_ids, matrices, concept_vectors,
                     graph: ConceptGraph, seed_ids: list[str],
                     lambda_ridge: float = 0.01) -> list[str]:
    """Matrix-based retrieval: q_concept × M → chunk scores.

    Steps:
    1. Find query concepts from seed chunks
    2. Build query concept vector q
    3. q × M → chunk scores (concept propagation via matrix mult)
    4. Optionally: W × q → expanded concepts → propagate again
    """
    import scipy.sparse as sp

    M = matrices["M"]
    W = matrices["W"]
    concepts = matrices["concepts"]
    concept_to_idx = matrices["concept_to_idx"]
    chunk_to_idx = matrices["chunk_to_idx"]

    # 1. Get seed concepts
    seed_concepts = set()
    for cid in seed_ids:
        seed_concepts.update(graph.chunk_concepts.get(cid, []))

    # 2. Build query concept vector
    q = np.zeros(len(concepts))
    for concept in seed_concepts:
        if concept in concept_to_idx:
            idx = concept_to_idx[concept]
            idf = graph.idf.get(concept, 1.0)
            q[idx] = idf

    if q.sum() == 0:
        return seed_ids[:TOP_K]

    # 3. Concept propagation: q × M → chunk scores
    chunk_scores = q @ M.toarray()  # [n_chunks]

    # 4. Relation expansion: W × q → expanded concepts
    if W.nnz > 0:
        q_expanded = W @ q  # [n_concepts]
        # Discount expanded concepts
        q_expanded *= 0.7
        # Add to chunk scores
        chunk_scores += (q_expanded @ M.toarray()) * 0.5

    # 5. Merge with seed priority
    result_scores = {}
    for i, cid in enumerate(chunk_ids):
        if cid not in seed_ids:
            result_scores[cid] = chunk_scores[i]

    # Sort by score, take top-K after seeds
    sorted_chunks = sorted(result_scores.items(), key=lambda x: x[1], reverse=True)
    merged = list(seed_ids[:10])
    for cid, _ in sorted_chunks:
        if cid not in merged:
            merged.append(cid)
        if len(merged) >= TOP_K:
            break

    # Backfill from cosine if needed
    if len(merged) < TOP_K:
        b0_hits = cosine_topk_ids(query_vec, chunk_emb, chunk_ids, TOP_K)
        for cid, _ in b0_hits:
            if cid not in merged:
                merged.append(cid)
            if len(merged) >= TOP_K:
                break

    return merged[:TOP_K]


def run_experiment(lens, lens_model, tokenizer, model, embed,
                   domain: str = "medical", max_queries: int = 28,
                   max_chunks: int = 200):
    print(f"Phase 38: Matrix acceleration validation")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # 1. Load corpus + embed
    print(f"\n[1/5] Loading corpus + embedding...")
    corpus, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts), dtype=np.float32)
    query_emb = np.asarray(embed.embed([q["question"] for q in questions]), dtype=np.float32)

    # 2. Concept extraction + graph
    print(f"\n[2/5] Concept extraction + graph building...")
    raw_concepts = {}
    for i, cid in enumerate(chunk_ids):
        raw_concepts[cid] = extract_chunk_concepts(lens, lens_model, tokenizer,
                                                    chunk_text_map[cid], n_words=5)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)}", flush=True)

    graph = ConceptGraph()
    for cid in chunk_ids:
        graph.add_chunk(cid, raw_concepts[cid])
    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), chunk_texts, len(chunk_ids))
    rebuild_graph_index(graph, filtered, meta["idf"])
    graph.compute_tf(chunk_text_map)
    concepts = sorted(graph.concept_chunks.keys())
    print(f"  Concepts: {len(concepts)}")

    # 3. Build matrices
    print(f"\n[3/5] Building matrices...")
    matrices = build_matrices(graph, chunk_ids)
    print(f"  M: {matrices['M'].shape}, nnz={matrices['M'].nnz}")
    print(f"  W: {matrices['W'].shape}, nnz={matrices['W'].nnz}")

    # 4. Get concept vectors (two methods)
    print(f"\n[4/5] Extracting concept vectors...")
    # E1: bge-m3
    e1_vecs = embed.embed(concepts)
    E1 = np.asarray(e1_vecs, dtype=np.float32)

    # E2: BPE fragment W_U vectors
    E2_dict = get_bpe_fragment_vectors(concepts, model, tokenizer)
    E2 = np.asarray([E2_dict.get(c.lower(), np.zeros(3584)) for c in concepts], dtype=np.float32)

    print(f"  E1 (bge-m3): {E1.shape}")
    print(f"  E2 (W_U fragment): {E2.shape}")

    # 5. Bijection consistency: graph communities vs vector clusters
    print(f"\n[5/5] Bijection consistency analysis...")

    # Graph communities: use co-occurrence W for clustering
    # Simple approach: use W's connected components or spectral clustering
    W_dense = matrices["W"].toarray()
    n_clusters = min(max(2, len(concepts) // 3), 8)

    # Spectral clustering on W
    try:
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                                 random_state=42, n_init=10)
        graph_labels = sc.fit_predict(W_dense + np.eye(len(concepts)))
    except Exception:
        # Fallback: use W row sums as cluster assignment
        graph_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(W_dense)

    # Vector clusters
    e1_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(E1)
    e2_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(E2)

    # Compute ARI and NMI
    ari_e1 = adjusted_rand_score(graph_labels, e1_labels)
    nmi_e1 = normalized_mutual_info_score(graph_labels, e1_labels)
    ari_e2 = adjusted_rand_score(graph_labels, e2_labels)
    nmi_e2 = normalized_mutual_info_score(graph_labels, e2_labels)

    print(f"\n  Bijection consistency (graph communities vs vector clusters):")
    print(f"  {'method':<20} {'ARI':>8} {'NMI':>8}")
    print(f"  {'-'*38}")
    print(f"  {'bge-m3 (E1)':<20} {ari_e1:>8.3f} {nmi_e1:>8.3f}")
    print(f"  {'W_U fragment (E2)':<20} {ari_e2:>8.3f} {nmi_e2:>8.3f}")
    print(f"  (ARI > 0.6 = strong bijection, > 0.3 = moderate)")

    # Also: E1 vs E2 consistency (are the two vector spaces aligned?)
    ari_e1e2 = adjusted_rand_score(e1_labels, e2_labels)
    nmi_e1e2 = normalized_mutual_info_score(e1_labels, e2_labels)
    print(f"  {'E1 vs E2':<20} {ari_e1e2:>8.3f} {nmi_e1e2:>8.3f}")

    # 6. Retrieval comparison
    print(f"\n{'='*70}")
    print(f"Retrieval comparison (3 methods × {len(questions)} queries)")
    print(f"{'='*70}")

    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_query(q_idx, q_data):
        q = q_data
        level = q["level"]
        question = q["question"]
        gold_answer = q.get("answer", "")
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]

        q_vec = query_emb[q_idx]
        b0_hits = cosine_topk_ids(q_vec, chunk_emb, chunk_ids, TOP_K)
        seed_ids = [cid for cid, _ in cosine_topk_ids(q_vec, chunk_emb, chunk_ids, 10)]

        # R1: Traditional graph propagation (Phase 25)
        flat_prop = graph.propagate(seed_ids, max_propagate=20,
                                     use_idf=True, use_bm25=True)
        merged_r1 = list(seed_ids[:10])
        for pid in flat_prop:
            if pid not in merged_r1: merged_r1.append(pid)
            if len(merged_r1) >= TOP_K: break

        # R2: Matrix retrieval
        merged_r2 = matrix_retrieve(q_vec, chunk_emb, chunk_ids, matrices,
                                     E2_dict, graph, seed_ids)

        # B0 context
        b0_ctx = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)
        r1_ctx = " ".join(chunk_text_map[cid] for cid in merged_r1[:TOP_K])
        r2_ctx = " ".join(chunk_text_map[cid] for cid in merged_r2[:TOP_K])

        llm_local = DeepSeekProvider()
        b0_ans = generate_answer(question, b0_ctx, llm_local)
        r1_ans = generate_answer(question, r1_ctx, llm_local)
        r2_ans = generate_answer(question, r2_ctx, llm_local)

        b0_acc = judge_answer_correctness(question, b0_ans, gold_answer, llm_local)
        r1_acc = judge_answer_correctness(question, r1_ans, gold_answer, llm_local)
        r2_acc = judge_answer_correctness(question, r2_ans, gold_answer, llm_local)

        b0_recall = llm_judge_evidence_recall(question, b0_ctx, evidence, llm_local)
        r1_recall = llm_judge_evidence_recall(question, r1_ctx, evidence, llm_local)
        r2_recall = llm_judge_evidence_recall(question, r2_ctx, evidence, llm_local)

        return {"level": level,
                "B0": {"acc": b0_acc, "recall": b0_recall},
                "graph": {"acc": r1_acc, "recall": r1_recall},
                "matrix": {"acc": r2_acc, "recall": r2_recall}}

    methods = ["B0", "graph", "matrix"]
    results_list = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_eval_query, i, q): i
                   for i, q in enumerate(questions) if q.get("answer")}
        done = 0
        for future in as_completed(futures):
            results_list.append(future.result())
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(questions)}", flush=True)

    # Summary
    acc = {m: [] for m in methods}
    recall = {m: [] for m in methods}
    for r in results_list:
        for m in methods:
            acc[m].append(1.0 if r[m]["acc"] else 0.0)
            recall[m].append(r[m]["recall"])

    print(f"\n  {'method':<15} {'ACC':>8} {'recall':>8}")
    print(f"  {'-'*33}")
    for m in methods:
        a = np.mean(acc[m]) if acc[m] else 0
        r = np.mean(recall[m]) if recall[m] else 0
        print(f"  {m:<15} {a:>7.1%} {r:>7.1%}")

    # Save
    out = {
        "method": "matrix_acceleration_validation",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "n_concepts": len(concepts),
        "bijection": {
            "ari_e1_bge": round(ari_e1, 3),
            "nmi_e1_bge": round(nmi_e1, 3),
            "ari_e2_wu": round(ari_e2, 3),
            "nmi_e2_wu": round(nmi_e2, 3),
            "ari_e1_vs_e2": round(ari_e1e2, 3),
            "nmi_e1_vs_e2": round(nmi_e1e2, 3),
        },
        "results": {m: {
            "acc": float(np.mean(acc[m])) if acc[m] else 0,
            "recall": float(np.mean(recall[m])) if recall[m] else 0,
        } for m in methods},
    }
    out_path = EXP / f"phase38_matrix_acceleration_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running matrix acceleration experiment...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    embed = CachedBgeM3Provider()
    run_experiment(lens, lens_model, tokenizer, model, embed,
                   domain="medical", max_queries=28, max_chunks=200)


if __name__ == "__main__":
    main()
