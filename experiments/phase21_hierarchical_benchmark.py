"""Phase 21: 分层概念图传播检索 benchmark。

Phase 20 验证了 concern+全层COM+三重过滤能产生 meta/sub 概念层级（100% 覆盖率）。
本实验验证：分层概念图传播是否比 flat 概念图（Stage 7c）提升检索性能？

## 三种检索方法对比

1. **B0**（baseline）：纯 bge-m3 余弦 top-K（标准 RAG）
2. **flat_concept**（Stage 7c 复现）：flat 概念图传播（B0 seed → 概念传播 → merge）
3. **hierarchical**（Phase 21 新方法）：两阶段分层传播
   - Phase 1: B0 seed → **meta 概念**传播（宽召回，meta 概念 DF 高，连接更多 chunk）
   - Phase 2: 候选 chunk → **sub 概念**重排（精排序，sub 概念更具体）
   - 合并：meta 传播召回 + sub 重排优先

## 核心假设

meta 概念（COM 低，宽泛）作为召回扩展器——它们连接语义相关但词汇不同的 chunk。
sub 概念（COM 高，具体）作为精排序器——它们区分真正相关的 chunk。

例：查询"膳食纤维对肠道的影响"
- flat: seed 命中含 "fiber" 的 chunk → 传播到其他含 "fiber" 的 chunk
- hierarchical: meta 传播通过 "food/diet/nutrition" 找到含 "nutrition" 的相关
  chunk（flat 会错过），再用 sub "fibre/soluble" 精排

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase21_hierarchical_benchmark
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals, build_concern_prompt
from experiments.phase10_jlens_stage7c import (
    ConceptGraph, cosine_topk_ids, extract_chunk_concepts,
    STOP_CONCEPTS,
)
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
    classify_by_com_gap, PREFILL_WORDS,
)
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english, build_corpus_word_set,
)
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


# ── Hierarchical concept extraction ───────────────────────────────────

def extract_chunk_concepts_hierarchical(
    lens, lens_model, tokenizer,
    chunk_text: str,
    all_layers: list[int],
) -> tuple[list[str], list[str]]:
    """Extract meta + sub concepts from a single chunk using Phase 20 algorithm.

    Returns (meta_concepts, sub_concepts).

    Instead of reading only L26 (like Stage 7c), we do a full depth-gradient
    scan and classify concepts by COM. This gives us typed concepts per chunk.
    """
    # Build concern prompt for this chunk
    user_msg = (
        f"What concepts does this text discuss? List 8 one-word concepts.\n\n"
        f"{chunk_text[:800]}"
    )
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            prompt = f"{user_msg}\n{prefill}"
    else:
        prompt = f"{user_msg}\n{prefill}"

    # Full depth gradient (one forward pass, all layers)
    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    # Phase 20 classification
    corpus_words = build_corpus_word_set([chunk_text])
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=3,
        require_corpus=True, require_noun=True)
    profiles = classify_by_com_gap(profiles)

    # Relax if too strict
    if len(profiles) < 2:
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=2,
            require_corpus=False, require_noun=True)
        profiles = classify_by_com_gap(profiles)

    meta = [p.word for p in profiles if p.role == "meta"]
    sub = [p.word for p in profiles if p.role == "sub"]

    # Fallback: if nothing classified, use top-5 content words as meta
    if not meta and not sub:
        for layer_data in gradient.values():
            for w in layer_data:
                tok = w["token"].lower()
                if (is_ascii_english(tok) and len(tok) >= 4
                        and tok not in STOP_WORDS_EXTENDED
                        and tok not in PREFILL_WORDS):
                    meta.append(tok)
                    break
            if len(meta) >= 5:
                break
        meta = meta[:5]

    return meta[:5], sub[:5]


# ── Hierarchical propagation ──────────────────────────────────────────

def hierarchical_propagate(
    graph_meta: ConceptGraph,
    graph_sub: ConceptGraph,
    seed_chunk_ids: list[str],
    max_propagate: int = 30,
    meta_boost: float = 1.0,
    sub_boost: float = 1.5,
) -> list[tuple[str, float]]:
    """Two-stage hierarchical propagation.

    Stage 1 (meta): seed chunks → meta concepts → candidate chunks (broad recall)
    Stage 2 (sub): re-rank candidates by sub-concept overlap (precision)

    Returns [(chunk_id, score), ...] sorted by combined score.
    """
    # Stage 1: meta propagation (broad)
    meta_scores = defaultdict(float)
    seed_meta_concepts = set()
    for cid in seed_chunk_ids:
        seed_meta_concepts.update(graph_meta.chunk_concepts.get(cid, []))

    for concept in seed_meta_concepts:
        idf_weight = graph_meta.idf.get(concept, 1.0)
        for cid in graph_meta.concept_chunks.get(concept, []):
            if cid not in seed_chunk_ids:
                tf = graph_meta.tf.get((cid, concept), 1)
                tf_weight = graph_meta._bm25_tf_norm(cid, tf)
                meta_scores[cid] += idf_weight * tf_weight * meta_boost

    # Stage 2: sub re-ranking on meta candidates
    seed_sub_concepts = set()
    for cid in seed_chunk_ids:
        seed_sub_concepts.update(graph_sub.chunk_concepts.get(cid, []))

    sub_scores = defaultdict(float)
    for concept in seed_sub_concepts:
        idf_weight = graph_sub.idf.get(concept, 1.0)
        for cid in graph_sub.concept_chunks.get(concept, []):
            if cid not in seed_chunk_ids:
                tf = graph_sub.tf.get((cid, concept), 1)
                tf_weight = graph_sub._bm25_tf_norm(cid, tf)
                sub_scores[cid] += idf_weight * tf_weight * sub_boost

    # Combine: candidates from both stages
    all_candidates = set(meta_scores.keys()) | set(sub_scores.keys())
    combined = []
    for cid in all_candidates:
        score = meta_scores.get(cid, 0) + sub_scores.get(cid, 0)
        combined.append((cid, score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:max_propagate]


# ── Main benchmark ────────────────────────────────────────────────────

def run_benchmark(lens, lens_model, tokenizer, embed,
                  domain: str = "medical",
                  max_queries: int = 30,
                  seed_k: int = 10,
                  max_chunks: int = 200):
    print(f"Phase 21: Hierarchical concept graph retrieval benchmark")
    print(f"  domain={domain}, max_queries={max_queries}, max_chunks={max_chunks}")
    print(f"{'='*70}")

    all_layers = lens.source_layers

    # 1. Load corpus + questions
    print(f"\n[1/4] Loading {domain} corpus + questions...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    # Limit chunks
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    corpus_chunks = chunk_text_map
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Embed
    print(f"\n[2/4] Embedding (bge-m3)...")
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts_list), dtype=np.float32)
    query_texts = [q["question"] for q in questions]
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float32)

    # 3. Build three concept representations
    print(f"\n[3/4] Building concept graphs...")
    # flat graph (Stage 7c style)
    graph_flat = ConceptGraph()
    # hierarchical: separate meta/sub graphs
    graph_meta = ConceptGraph()
    graph_sub = ConceptGraph()

    for i, cid in enumerate(chunk_ids):
        text = chunk_text_map[cid]

        # Flat extraction (Stage 7c method, single layer L26)
        flat_concepts = extract_chunk_concepts(lens, lens_model, tokenizer, text, n_words=5)
        graph_flat.add_chunk(cid, flat_concepts)

        # Hierarchical extraction (Phase 20 depth gradient)
        meta_concepts, sub_concepts = extract_chunk_concepts_hierarchical(
            lens, lens_model, tokenizer, text, all_layers)
        graph_meta.add_chunk(cid, meta_concepts)
        graph_sub.add_chunk(cid, sub_concepts)

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(chunk_ids)} chunks processed", flush=True)

    # Compute TF + IDF for all graphs
    for graph, name in [(graph_flat, "flat"), (graph_meta, "meta"), (graph_sub, "sub")]:
        graph.compute_tf(chunk_text_map)
        graph.compute_idf()
        stats = graph.graph_stats()
        print(f"  {name}: {stats['n_concepts']} concepts, "
              f"{stats['avg_concepts_per_chunk']:.1f} per chunk")

    # 4. Retrieval evaluation
    print(f"\n[4/4] Retrieval evaluation (3 methods × {len(questions)} queries)...")

    eval_tasks = []
    for i, q in enumerate(questions):
        level = q["level"]
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not evidence:
            continue

        # Method 1: B0 (pure cosine)
        b0_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(corpus_chunks[cid] for cid, _ in b0_hits)
        eval_tasks.append(("B0", level, q["question"], b0_context, evidence))

        # Method 2: flat concept propagation (Stage 7c style)
        seed_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, seed_k)
        seed_ids = [cid for cid, _ in seed_hits]
        propagated_flat = graph_flat.propagate(seed_ids, max_propagate=20,
                                                use_idf=True, use_bm25=True)
        merged_flat = seed_ids[:seed_k]
        for pid in propagated_flat:
            if pid not in merged_flat:
                merged_flat.append(pid)
            if len(merged_flat) >= TOP_K:
                break
        flat_context = " ".join(corpus_chunks[cid] for cid in merged_flat[:TOP_K])
        eval_tasks.append(("flat_concept", level, q["question"], flat_context, evidence))

        # Method 3: hierarchical propagation (Phase 21)
        hier_propagated = hierarchical_propagate(
            graph_meta, graph_sub, seed_ids,
            max_propagate=20, meta_boost=1.0, sub_boost=1.5)
        merged_hier = seed_ids[:seed_k]
        for pid, _ in hier_propagated:
            if pid not in merged_hier:
                merged_hier.append(pid)
            if len(merged_hier) >= TOP_K:
                break
        # Backfill from B0 if needed
        if len(merged_hier) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_hier:
                    merged_hier.append(cid)
                if len(merged_hier) >= TOP_K:
                    break
        hier_context = " ".join(corpus_chunks[cid] for cid in merged_hier[:TOP_K])
        eval_tasks.append(("hierarchical", level, q["question"], hier_context, evidence))

    # Concurrent LLM judge
    print(f"  Concurrent LLM judge ({len(eval_tasks)} calls, 8 workers)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    def _judge_one(task):
        method, level, question, context, evidence = task
        llm_local = DeepSeekProvider()
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)
        return method, level, recall

    methods = ["B0", "flat_concept", "hierarchical"]
    results = {m: [] for m in methods}
    by_level = {m: defaultdict(list) for m in methods}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_judge_one, t): t for t in eval_tasks}
        done = 0
        for future in as_completed(futures):
            method, level, recall = future.result()
            results[method].append(recall)
            by_level[method][level].append(recall)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(eval_tasks)} judged", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS ({domain})")
    print(f"{'='*70}")
    print(f"  {'method':<18} {'overall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*58}")
    for m in methods:
        overall = np.mean(results[m]) if results[m] else 0
        levels = []
        for lv in ["L1", "L2", "L3", "L4"]:
            scores = by_level[m].get(lv, [])
            levels.append(f"{np.mean(scores):>7.1%}" if scores else f"{'N/A':>7}")
        print(f"  {m:<18} {overall:>7.1%}  {levels[0]} {levels[1]} {levels[2]} {levels[3]}")

    # Relative to B0
    b0_mean = np.mean(results["B0"]) if results["B0"] else 1
    print(f"\n  Relative to B0:")
    for m in ["flat_concept", "hierarchical"]:
        m_mean = np.mean(results[m]) if results[m] else 0
        ratio = m_mean / b0_mean if b0_mean > 0 else 0
        print(f"    {m:<18} {ratio:>6.1%} of B0")

    # Save
    out = {
        "method": "hierarchical_concept_graph_benchmark",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(questions),
        "results": {
            m: {
                "overall": float(np.mean(results[m])) if results[m] else 0,
                "by_level": {lv: (float(np.mean(by_level[m][lv]))
                                  if by_level[m].get(lv) else None)
                             for lv in ["L1", "L2", "L3", "L4"]},
                "n": len(results[m]),
            }
            for m in methods
        },
        "graph_stats": {
            "flat": graph_flat.graph_stats(),
            "meta": graph_meta.graph_stats(),
            "sub": graph_sub.graph_stats(),
        },
    }
    out_path = EXP / f"phase21_hierarchical_benchmark_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medical", choices=["medical", "novel"])
    ap.add_argument("--max-queries", type=int, default=30)
    ap.add_argument("--max-chunks", type=int, default=200)
    args = ap.parse_args()

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running benchmark...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    embed = CachedBgeM3Provider()
    run_benchmark(lens, lens_model, tokenizer, embed,
                  domain=args.domain,
                  max_queries=args.max_queries,
                  max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
