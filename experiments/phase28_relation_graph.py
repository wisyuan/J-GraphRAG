"""Phase 28: 关系图构建 + 检索 benchmark。

Phase 27 PoC 证明双概念关切耦合可以读出关系类型（treated/causal/regulated）。
本实验正式构建有类型关系图并测试检索效果。

## 流程

1. 概念提取（Phase 25 方法）→ 三重过滤 + BM25 补全 → 概念集
2. 共现筛选：concept_cooccur 中共现 ≥ min_cooccur 的概念对才做 forward pass
3. 双概念关切读关系：每对共现概念 → 1 次 forward pass → 关系类型词
4. 关系图构建：concept↔concept 有类型边（treatment/causal/regulated...）
5. 检索对比：
   - B0: 纯余弦 RAG
   - flat: 二部概念图传播（Phase 25）
   - relation: 二部图 + 关系图传播（沿关系边扩展概念集 → 再传播到 chunk）

## 关系图传播机制

```
seed chunk → 概念集 C1
C1 的概念 → 沿关系边找到关联概念 C2（如 cancer --treated_by--> chemotherapy）
C1 ∪ C2 → 扩展概念集 → 传播到更多 chunk
```

和 flat 图传播的区别：flat 只找共享 C1 概念的 chunk；relation 额外找
共享 C2（关系扩展概念）的 chunk。这应该提升多跳推理（L4）的召回。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase28_relation_graph
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
from experiments.phase27_relation_readout import (
    build_relation_prompt, decode_topk, STOP_REL,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10

# Meaningful relation words (from Phase 27 PoC analysis)
RELATION_TYPES = {
    "treatment", "treat", "treats", "treated", "treating", "therapy",
    "therapeutic", "therapies", "reatment",
    "cause", "causes", "caused", "causing", "causal", "caus",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "risk", "risks",
    "component", "contains", "source", "sources",
    "builds", "build", "strengthens", "strengthen",
    "spreads", "spread", "progression", "progress",
    "growth", "grows",
    "inhibits", "inhibit", "inhibition",
    "promotes", "promote",
    "manages", "manage",
    "removal", "removes", "remove",
    "associated", "association",
    "induces", "induce",
    "reduces", "reduce",
    "increases", "increase",
    "affects", "affect",
    "improves", "improve",
    "protects", "protect", "protection",
    "targets", "target",
    "kills", "kill",
    "supports", "support",
    "requires", "require",
    "produces", "produce",
    "regulates", "regulate", "regulated", "regulatory",
    "stimulates", "stimulate",
    "suppresses", "suppress",
    "essential", "crucial", "vital", "integral", "critical",
    "central",
}


def extract_relation(
    lens, lens_model, tokenizer,
    concept_a: str, concept_b: str,
    doc_text: str,
    layer: int,
) -> tuple[str | None, float]:
    """Read relation type between two concepts via dual-concept concern.

    Returns (relation_word, probability) or (None, 0) if no meaningful relation.
    """
    prompt = build_relation_prompt(doc_text, concept_a, concept_b, tokenizer)
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)

    for w in words:
        if w["token"].lower() in RELATION_TYPES:
            return w["token"], w["prob"]
    # Return top word even if not in known set (for analysis)
    if words:
        return words[0]["token"], words[0]["prob"]
    return None, 0.0


def build_relation_graph(
    lens, lens_model, tokenizer,
    graph: ConceptGraph,
    chunk_text_map: dict[str, str],
    layer: int,
    min_cooccur: int = 2,
) -> dict:
    """Build typed concept-relation edges using co-occurrence filtering.

    Only concept pairs that co-occur in >= min_cooccur chunks get a forward pass.
    """
    # Get co-occurrence from the graph
    cooccur = graph.concept_cooccur  # {concept_a: {concept_b: count}}

    # Find pairs to test
    pairs_to_test = []
    for c_a, neighbors in cooccur.items():
        for c_b, count in neighbors.items():
            if count >= min_cooccur and c_a < c_b:  # avoid duplicates
                pairs_to_test.append((c_a, c_b, count))

    print(f"  Co-occurrence filter: {len(pairs_to_test)} pairs (≥{min_cooccur} co-occur) "
          f"out of {len(graph.concept_chunks)*(len(graph.concept_chunks)-1)//2} possible")

    # For each pair, find a representative document (one that contains both)
    relations = []
    t_start = time.perf_counter()

    for i, (c_a, c_b, cooccur_count) in enumerate(pairs_to_test):
        # Find a chunk containing both concepts
        chunks_a = set(graph.concept_chunks.get(c_a, []))
        chunks_b = set(graph.concept_chunks.get(c_b, []))
        common_chunks = chunks_a & chunks_b

        doc_text = ""
        if common_chunks:
            # Use the first common chunk
            cid = list(common_chunks)[0]
            doc_text = chunk_text_map.get(cid, "")
        elif chunks_a:
            doc_text = chunk_text_map.get(list(chunks_a)[0], "")

        if not doc_text:
            continue

        rel_word, rel_prob = extract_relation(
            lens, lens_model, tokenizer, c_a, c_b, doc_text, layer)

        if rel_word:
            relations.append({
                "concept_a": c_a,
                "concept_b": c_b,
                "relation": rel_word.lower(),
                "prob": rel_prob,
                "cooccur": cooccur_count,
                "is_known_type": rel_word.lower() in RELATION_TYPES,
            })

        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    {i+1}/{len(pairs_to_test)} pairs processed ({elapsed:.0f}s)",
                  flush=True)

    t_rel = time.perf_counter() - t_start
    print(f"  Relation extraction: {t_rel:.1f}s for {len(relations)} relations "
          f"({t_rel/max(1,len(pairs_to_test)):.2f}s/pair)")

    # Build adjacency: concept → [(related_concept, relation_type)]
    relation_adj = defaultdict(list)
    for r in relations:
        relation_adj[r["concept_a"]].append((r["concept_b"], r["relation"]))
        relation_adj[r["concept_b"]].append((r["concept_a"], r["relation"]))

    return {
        "relations": relations,
        "adjacency": dict(relation_adj),
        "n_relations": len(relations),
        "n_known_type": sum(1 for r in relations if r["is_known_type"]),
        "extraction_time_s": round(t_rel, 1),
        "n_pairs_tested": len(pairs_to_test),
    }


def relation_propagate(
    graph: ConceptGraph,
    relation_adj: dict[str, list[tuple[str, str]]],
    seed_chunk_ids: list[str],
    max_propagate: int = 30,
) -> list[tuple[str, float]]:
    """Graph propagation with relation expansion.

    1. Seed chunks → concept set C1
    2. C1 concepts → relation graph → expanded concept set C2
    3. C1 ∪ C2 → propagate to chunks via membership edges
    """
    if not hasattr(graph, 'idf'):
        graph.compute_idf()

    # Step 1: collect seed concepts
    seed_concepts = set()
    for cid in seed_chunk_ids:
        seed_concepts.update(graph.chunk_concepts.get(cid, []))

    # Step 2: expand via relation edges
    expanded_concepts = set(seed_concepts)
    for concept in seed_concepts:
        for related, rel_type in relation_adj.get(concept, []):
            expanded_concepts.add(related)

    n_expanded = len(expanded_concepts) - len(seed_concepts)

    # Step 3: propagate to chunks using expanded concept set
    chunk_score = defaultdict(float)
    for concept in expanded_concepts:
        idf_weight = graph.idf.get(concept, 1.0)
        # Give relation-expanded concepts slightly lower weight
        # (they're inferred, not directly from seed)
        if concept not in seed_concepts:
            idf_weight *= 0.7  # discount inferred concepts

        for cid in graph.concept_chunks.get(concept, []):
            if cid not in seed_chunk_ids:
                tf = graph.tf.get((cid, concept), 1)
                tf_weight = graph._bm25_tf_norm(cid, tf)
                chunk_score[cid] += idf_weight * tf_weight

    result = sorted(chunk_score.items(), key=lambda x: x[1], reverse=True)
    return result[:max_propagate], n_expanded


def run_benchmark(lens, lens_model, tokenizer, embed,
                  domain: str = "medical",
                  max_queries: int = 28, max_chunks: int = 200):
    print(f"Phase 28: Relation graph retrieval benchmark")
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
    print(f"  Build: {t_build:.1f}s, concepts={meta['n_after']}, VRAM={vram_peak:.2f}GB")

    # 3. Relation extraction (co-occurrence filtered)
    print(f"\n[3/5] Relation extraction (co-occurrence filtered)...")
    layer = lens.source_layers[-1]
    rel_data = build_relation_graph(
        lens, lens_model, tokenizer, graph, chunk_text_map, layer,
        min_cooccur=2)

    print(f"  Relations: {rel_data['n_relations']} total, "
          f"{rel_data['n_known_type']} known type")
    print(f"  Time: {rel_data['extraction_time_s']}s "
          f"({rel_data['extraction_time_s']/max(1,rel_data['n_pairs_tested']):.2f}s/pair)")

    # Print top relations
    known_rels = [r for r in rel_data["relations"] if r["is_known_type"]]
    known_rels.sort(key=lambda x: x["prob"], reverse=True)
    print(f"\n  Top known-type relations:")
    for r in known_rels[:15]:
        print(f"    {r['concept_a']:15} --{r['relation']:12}--> {r['concept_b']:15} "
              f"(p={r['prob']:.2f}, co={r['cooccur']})")

    t_total = t_build + rel_data["extraction_time_s"]
    avg_total = t_total / len(chunk_ids)

    # 4. Retrieval evaluation
    print(f"\n[4/5] Retrieval evaluation (3 methods × {len(questions)} queries)...")
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

        # flat concept propagation (Phase 25)
        seed_ids = [cid for cid, _ in cosine_topk_ids(q_vec, chunk_emb, chunk_ids, 10)]
        flat_propagated = graph.propagate(seed_ids, max_propagate=20,
                                          use_idf=True, use_bm25=True)
        merged_flat = list(seed_ids[:10])
        for pid in flat_propagated:
            if pid not in merged_flat: merged_flat.append(pid)
            if len(merged_flat) >= TOP_K: break
        flat_context = " ".join(chunk_text_map[cid] for cid in merged_flat[:TOP_K])

        # relation propagation
        rel_propagated, n_expanded = relation_propagate(
            graph, relation_adj, seed_ids, max_propagate=20)
        merged_rel = list(seed_ids[:10])
        for pid, _ in rel_propagated:
            if pid not in merged_rel: merged_rel.append(pid)
            if len(merged_rel) >= TOP_K: break
        if len(merged_rel) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_rel: merged_rel.append(cid)
                if len(merged_rel) >= TOP_K: break
        rel_context = " ".join(chunk_text_map[cid] for cid in merged_rel[:TOP_K])

        # Evaluate
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

        return {
            "level": level,
            "n_expanded": n_expanded,
            "B0": {"acc": b0_acc, "recall": b0_recall},
            "flat": {"acc": flat_acc, "recall": flat_recall},
            "relation": {"acc": rel_acc, "recall": rel_recall},
        }

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

    # 5. Summary
    print(f"\n[5/5] Summary")
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
        rel = f" ({a/b0_mean:.0%})" if b0_mean > 0 and m != "B0" else ""
        print(f"  {m:<15} {a:>7.1%}{rel} {r:>7.1%} {lvls[0]} {lvls[1]} {lvls[2]} {lvls[3]}")

    print(f"\n  Efficiency:")
    print(f"    Concept build:  {t_build:.1f}s ({t_build/len(chunk_ids):.3f}s/chunk)")
    print(f"    Relation build: {rel_data['extraction_time_s']}s "
          f"({rel_data['n_pairs_tested']} pairs)")
    print(f"    Total build:    {t_total:.1f}s ({avg_total:.3f}s/chunk)")
    print(f"    VRAM peak:      {vram_peak:.2f}GB")
    print(f"    Relations:      {rel_data['n_relations']} "
          f"({rel_data['n_known_type']} known type)")

    # Save
    out = {
        "method": "relation_graph_benchmark",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "efficiency": {
            "concept_build_s": round(t_build, 1),
            "relation_build_s": rel_data["extraction_time_s"],
            "total_build_s": round(t_total, 1),
            "per_chunk_s": round(avg_total, 4),
            "vram_peak_gb": round(vram_peak, 2),
            "n_concepts": meta["n_after"],
            "n_relations": rel_data["n_relations"],
            "n_known_type_relations": rel_data["n_known_type"],
            "n_pairs_tested": rel_data["n_pairs_tested"],
        },
        "results": {m: {
            "acc": float(np.mean(acc[m])) if acc[m] else 0,
            "recall": float(np.mean(recall[m])) if recall[m] else 0,
            "acc_by_level": {lv: (float(np.mean(by_level_acc[m][lv]))
                                  if by_level_acc[m].get(lv) else None)
                             for lv in ["L1","L2","L3","L4"]},
            "n": len(acc[m]),
        } for m in methods},
        "top_relations": [
            {"a": r["concept_a"], "b": r["concept_b"],
             "relation": r["relation"], "prob": r["prob"],
             "cooccur": r["cooccur"]}
            for r in known_rels[:20]
        ],
    }
    out_path = EXP / f"phase28_relation_graph_{domain}.json"
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

    print(f"\n[2/2] Running relation graph benchmark...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    embed = CachedBgeM3Provider()
    run_benchmark(lens, lens_model, tokenizer, embed,
                  domain="medical", max_queries=28, max_chunks=200)


if __name__ == "__main__":
    main()
