"""Phase 30: 聚类驱动的关系图构建 + 检索 benchmark。

用户方案：用 word2vec 向量对概念词聚类，分簇 + 簇间两层提取关系。

流程：
1. 概念提取（Phase 25）→ 三重过滤 + BM25 → 概念集
2. bge-m3 嵌入概念词 → K-Means 聚类 → 概念簇
3. 簇内关系：同簇概念对（语义近）→ J-Lens 读关系
4. 簇间关系：每簇取离中心最近的词做代表 → 代表词对 → J-Lens 读关系
5. 关系图传播检索（Phase 28 方法）

和 Phase 28 的区别：
  Phase 28: 共现筛选（≥2 chunk 共现）→ 3 对 → 漏掉不共现但有关系的概念
  Phase 30: 聚类结构 → 簇内 + 簇间 → 覆盖更完整

成本估算：
  13 概念 → 4 簇 → 簇内 ~12 对 + 簇间 ~6 对 = ~18 次 forward pass
  vs Phase 28 的 3 对（共现筛太严）vs N²=78 对（暴力穷举）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase30_cluster_relation_graph
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
from experiments.phase27_relation_readout import build_relation_prompt, decode_topk
from experiments.phase28_relation_graph import (
    RELATION_TYPES, relation_propagate,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


def cluster_concepts(concepts: list[str], embed_fn, n_clusters: int | None = None) -> dict:
    """Cluster concept words using bge-m3 embeddings + K-Means.

    Returns {cluster_id: {concepts, representative, centroid}}.
    Representative = concept closest to cluster centroid.
    """
    if len(concepts) <= 2:
        return {0: {"concepts": concepts, "representative": concepts[0],
                     "centroid": None}}

    # Embed concept words
    vecs = np.asarray(embed_fn(concepts), dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / (norms + 1e-8)

    # Determine cluster count
    if n_clusters is None:
        n_clusters = max(2, min(len(concepts) // 3, 6))
    n_clusters = min(n_clusters, len(concepts) - 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs_norm)
    centroids = kmeans.cluster_centers_

    clusters = {}
    for cid in range(n_clusters):
        members = [concepts[i] for i in range(len(concepts)) if labels[i] == cid]
        if not members:
            continue

        # Find representative (closest to centroid)
        member_indices = [i for i in range(len(concepts)) if labels[i] == cid]
        member_vecs = vecs_norm[member_indices]
        distances = np.linalg.norm(member_vecs - centroids[cid], axis=1)
        rep_idx = member_indices[np.argmin(distances)]
        representative = concepts[rep_idx]

        clusters[cid] = {
            "concepts": members,
            "representative": representative,
            "centroid": centroids[cid].tolist(),
            "n": len(members),
        }

    return clusters


def build_cluster_relation_graph(
    lens, lens_model, tokenizer,
    concepts: list[str],
    clusters: dict,
    graph: ConceptGraph,
    chunk_text_map: dict[str, str],
    layer: int,
) -> dict:
    """Build relation graph using cluster structure.

    Two layers:
    1. Intra-cluster: all pairs within each cluster (semantic neighbors)
    2. Inter-cluster: representative pairs across clusters (semantic bridges)
    """
    relations = []
    pairs_tested = []

    def _find_doc_for_pair(c_a, c_b):
        """Find a document mentioning at least one of the concepts."""
        chunks_a = set(graph.concept_chunks.get(c_a, []))
        chunks_b = set(graph.concept_chunks.get(c_b, []))
        common = chunks_a & chunks_b
        if common:
            return chunk_text_map.get(list(common)[0], "")
        if chunks_a:
            return chunk_text_map.get(list(chunks_a)[0], "")
        if chunks_b:
            return chunk_text_map.get(list(chunks_b)[0], "")
        # Fallback: any chunk
        return list(chunk_text_map.values())[0] if chunk_text_map else ""

    def _read_relation(c_a, c_b, doc_text):
        prompt = build_relation_prompt(doc_text, c_a, c_b, tokenizer)
        lens_logits, _, _ = lens.apply(
            lens_model, prompt, layers=[layer],
            positions=[-1], max_seq_len=512)
        words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)
        for w in words:
            if w["token"].lower() in RELATION_TYPES:
                return w["token"], w["prob"], True
        if words:
            return words[0]["token"], words[0]["prob"], False
        return None, 0.0, False

    # === Layer 1: Intra-cluster relations ===
    intra_pairs = 0
    for cid, cluster in clusters.items():
        members = cluster["concepts"]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs_tested.append(("intra", cid, members[i], members[j]))
                intra_pairs += 1

    # === Layer 2: Inter-cluster relations ===
    cluster_ids = sorted(clusters.keys())
    reps = [clusters[cid]["representative"] for cid in cluster_ids]
    inter_pairs = 0
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            pairs_tested.append(("inter", f"{cluster_ids[i]}-{cluster_ids[j]}",
                                 reps[i], reps[j]))
            inter_pairs += 1

    print(f"  Pairs to test: {intra_pairs} intra + {inter_pairs} inter = {len(pairs_tested)} total")

    # Execute forward passes
    t_start = time.perf_counter()
    seen_pairs = set()

    for layer_type, group_id, c_a, c_b in pairs_tested:
        pair_key = tuple(sorted([c_a.lower(), c_b.lower()]))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        doc_text = _find_doc_for_pair(c_a, c_b)
        if not doc_text:
            continue

        rel_word, rel_prob, is_known = _read_relation(c_a, c_b, doc_text)

        if rel_word:
            relations.append({
                "concept_a": c_a,
                "concept_b": c_b,
                "relation": rel_word.lower(),
                "prob": rel_prob,
                "layer_type": layer_type,
                "group": str(group_id),
                "is_known_type": is_known,
            })

    t_rel = time.perf_counter() - t_start
    print(f"  Relations: {len(relations)} ({sum(1 for r in relations if r['is_known_type'])} known type)")
    print(f"  Time: {t_rel:.1f}s ({t_rel/max(1,len(seen_pairs)):.2f}s/pair)")

    # Build adjacency
    relation_adj = defaultdict(list)
    for r in relations:
        relation_adj[r["concept_a"]].append((r["concept_b"], r["relation"]))
        relation_adj[r["concept_b"]].append((r["concept_a"], r["relation"]))

    return {
        "relations": relations,
        "adjacency": dict(relation_adj),
        "n_relations": len(relations),
        "n_known_type": sum(1 for r in relations if r["is_known_type"]),
        "n_intra": sum(1 for r in relations if r["layer_type"] == "intra"),
        "n_inter": sum(1 for r in relations if r["layer_type"] == "inter"),
        "extraction_time_s": round(t_rel, 1),
        "n_pairs_tested": len(seen_pairs),
        "n_unique_pairs": len(seen_pairs),
    }


def run_benchmark(lens, lens_model, tokenizer, embed,
                  domain: str = "medical",
                  max_queries: int = 28, max_chunks: int = 200):
    print(f"Phase 30: Cluster-driven relation graph benchmark")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # 1. Load corpus + embed
    print(f"\n[1/5] Loading corpus + embedding...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts_list), dtype=np.float32)
    query_emb = np.asarray(embed.embed([q["question"] for q in questions]), dtype=np.float32)
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Concept extraction + graph (Phase 25)
    print(f"\n[2/5] Concept extraction + graph building...")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    raw_concepts = {}
    for i, cid in enumerate(chunk_ids):
        raw_concepts[cid] = extract_chunk_concepts(
            lens, lens_model, tokenizer, chunk_text_map[cid], n_words=5)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)}", flush=True)

    graph = ConceptGraph()
    for cid in chunk_ids:
        graph.add_chunk(cid, raw_concepts[cid])
    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), chunk_texts_list, len(chunk_ids))
    rebuild_graph_index(graph, filtered, meta["idf"])
    graph.compute_tf(chunk_text_map)

    t_build = time.perf_counter() - t_start
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    concepts = list(filtered.keys())
    print(f"  Build: {t_build:.1f}s, concepts={len(concepts)}, VRAM={vram_peak:.2f}GB")
    print(f"  Concepts: {concepts}")

    # 3. Cluster concepts
    print(f"\n[3/5] Clustering concepts (bge-m3 + K-Means)...")
    clusters = cluster_concepts(concepts, embed.embed)
    for cid, cluster in clusters.items():
        print(f"  Cluster {cid}: {cluster['concepts']} "
              f"(rep: {cluster['representative']})")

    # 4. Relation extraction (cluster-driven)
    print(f"\n[4/5] Relation extraction (cluster-driven)...")
    layer = lens.source_layers[-1]
    rel_data = build_cluster_relation_graph(
        lens, lens_model, tokenizer, concepts, clusters,
        graph, chunk_text_map, layer)

    known_rels = [r for r in rel_data["relations"] if r["is_known_type"]]
    known_rels.sort(key=lambda x: x["prob"], reverse=True)
    print(f"\n  Top known-type relations:")
    for r in known_rels[:15]:
        print(f"    [{r['layer_type']:5}] {r['concept_a']:15} --{r['relation']:12}--> "
              f"{r['concept_b']:15} (p={r['prob']:.2f})")

    t_total = t_build + rel_data["extraction_time_s"]
    avg_total = t_total / len(chunk_ids)

    # 5. Retrieval evaluation
    print(f"\n[5/5] Retrieval evaluation (3 methods × {len(questions)} queries)...")
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    relation_adj = rel_data["adjacency"]

    def _eval_query(q_idx, q_data):
        q = q_data
        level = q["level"]
        question = q["question"]
        gold_answer = q.get("answer", "")
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]

        q_vec = query_emb[q_idx]

        # B0
        b0_hits = cosine_topk_ids(q_vec, chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)

        # flat
        seed_ids = [cid for cid, _ in cosine_topk_ids(q_vec, chunk_emb, chunk_ids, 10)]
        flat_prop = graph.propagate(seed_ids, max_propagate=20,
                                     use_idf=True, use_bm25=True)
        merged_flat = list(seed_ids[:10])
        for pid in flat_prop:
            if pid not in merged_flat: merged_flat.append(pid)
            if len(merged_flat) >= TOP_K: break
        flat_context = " ".join(chunk_text_map[cid] for cid in merged_flat[:TOP_K])

        # relation
        rel_prop, n_exp = relation_propagate(graph, relation_adj, seed_ids, max_propagate=20)
        merged_rel = list(seed_ids[:10])
        for pid, _ in rel_prop:
            if pid not in merged_rel: merged_rel.append(pid)
            if len(merged_rel) >= TOP_K: break
        if len(merged_rel) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_rel: merged_rel.append(cid)
                if len(merged_rel) >= TOP_K: break
        rel_context = " ".join(chunk_text_map[cid] for cid in merged_rel[:TOP_K])

        llm_local = DeepSeekProvider()
        b0_answer = generate_answer(question, b0_context, llm_local)
        flat_answer = generate_answer(question, flat_context, llm_local)
        rel_answer = generate_answer(question, rel_context, llm_local)

        b0_acc = judge_answer_correctness(question, b0_answer, gold_answer, llm_local)
        flat_acc = judge_answer_correctness(question, flat_answer, gold_answer, llm_local)
        rel_acc = judge_answer_correctness(question, rel_answer, gold_answer, llm_local)

        b0_recall = llm_judge_evidence_recall(question, b0_context, evidence, llm_local)
        flat_recall = llm_judge_evidence_recall(question, flat_context, evidence, llm_local)
        rel_recall = llm_judge_evidence_recall(question, rel_context, evidence, llm_local)

        return {"level": level, "n_expanded": n_exp,
                "B0": {"acc": b0_acc, "recall": b0_recall},
                "flat": {"acc": flat_acc, "recall": flat_recall},
                "relation": {"acc": rel_acc, "recall": rel_recall}}

    methods = ["B0", "flat", "relation"]
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
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    acc = {m: [] for m in methods}
    recall = {m: [] for m in methods}
    by_level_acc = {m: defaultdict(list) for m in methods}
    for r in results_list:
        for m in methods:
            acc[m].append(1.0 if r[m]["acc"] else 0.0)
            recall[m].append(r[m]["recall"])
            by_level_acc[m][r["level"]].append(1.0 if r[m]["acc"] else 0.0)

    print(f"\n  {'method':<15} {'ACC':>8} {'recall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*63}")
    b0_mean = np.mean(acc["B0"]) if acc["B0"] else 1
    for m in methods:
        a = np.mean(acc[m]) if acc[m] else 0
        r = np.mean(recall[m]) if recall[m] else 0
        lvls = []
        for lv in ["L1","L2","L3","L4"]:
            s = by_level_acc[m].get(lv, [])
            lvls.append(f"{np.mean(s):>7.1%}" if s else f"{'N/A':>7}")
        rel_pct = f" ({a/b0_mean:.0%})" if b0_mean > 0 and m != "B0" else ""
        print(f"  {m:<15} {a:>7.1%}{rel_pct} {r:>7.1%} {lvls[0]} {lvls[1]} {lvls[2]} {lvls[3]}")

    print(f"\n  Efficiency:")
    print(f"    Concept build:  {t_build:.1f}s ({t_build/len(chunk_ids):.3f}s/chunk)")
    print(f"    Relation build: {rel_data['extraction_time_s']}s ({rel_data['n_unique_pairs']} pairs)")
    print(f"    Total build:    {t_total:.1f}s ({avg_total:.3f}s/chunk)")
    print(f"    VRAM:           {vram_peak:.2f}GB")
    print(f"    Relations:      {rel_data['n_relations']} "
          f"({rel_data['n_known_type']} known, "
          f"{rel_data['n_intra']} intra, {rel_data['n_inter']} inter)")

    # Save
    out = {
        "method": "cluster_relation_graph_benchmark",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "clusters": {str(k): {"concepts": v["concepts"],
                              "representative": v["representative"],
                              "n": v["n"]} for k, v in clusters.items()},
        "efficiency": {
            "concept_build_s": round(t_build, 1),
            "relation_build_s": rel_data["extraction_time_s"],
            "total_build_s": round(t_total, 1),
            "per_chunk_s": round(avg_total, 4),
            "vram_peak_gb": round(vram_peak, 2),
            "n_concepts": len(concepts),
            "n_relations": rel_data["n_relations"],
            "n_known_type": rel_data["n_known_type"],
            "n_intra": rel_data["n_intra"],
            "n_inter": rel_data["n_inter"],
            "n_pairs_tested": rel_data["n_unique_pairs"],
        },
        "results": {m: {
            "acc": float(np.mean(acc[m])) if acc[m] else 0,
            "recall": float(np.mean(recall[m])) if recall[m] else 0,
            "acc_by_level": {lv: (float(np.mean(by_level_acc[m][lv]))
                                  if by_level_acc[m].get(lv) else None)
                             for lv in ["L1","L2","L3","L4"]},
            "n": len(acc[m]),
        } for m in methods},
        "top_relations": [{"a": r["concept_a"], "b": r["concept_b"],
                           "relation": r["relation"], "prob": r["prob"],
                           "type": r["layer_type"]}
                          for r in known_rels[:20]],
    }
    out_path = EXP / f"phase30_cluster_relation_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running cluster relation graph benchmark...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    embed = CachedBgeM3Provider()
    run_benchmark(lens, lens_model, tokenizer, embed,
                  domain="medical", max_queries=28, max_chunks=200)


if __name__ == "__main__":
    main()
