"""Phase 23: 混合 benchmark——Phase 20 过滤器替换 Stage 7c 的 optimize_concepts。

Phase 21 证明分层概念图传播不超越 flat 图。Phase 22 证明 COM-based meta/sub
不是语义层级（0-9% is_a/part_of）。但 Phase 20 的三重过滤器本身可能提升
概念质量——它比 Stage 7c 的 optimize_concepts 更严格（POS 过滤排除动词形式，
prefill 黑名单排除 prompt 词）。

核心对比（medical 域）：
  方法 1: B0 (RAG baseline)
  方法 2: flat_stage7c (Stage 7c 原版：optimize_concepts 过滤 + 概念图传播)
  方法 3: flat_phase20 (Phase 20 过滤器：corpus+POS+prefill + 概念图传播)

如果 flat_phase20 > flat_stage7c → Phase 20 的过滤器提升了概念质量
如果 flat_phase20 ≈ flat_stage7c → 两种过滤器效果相当，产品用哪个都行

这是一个"概念质量过滤器 A/B 测试"——不涉及分层，纯粹测三重过滤器 vs optimize_concepts。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase23_hybrid_filtered_benchmark
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
from experiments.phase10_jlens_stage7c import (
    ConceptGraph, cosine_topk_ids, extract_chunk_concepts,
)
from experiments.phase20_concern_full_com import PREFILL_WORDS
from experiments.phase18_centroid_hierarchy import STOP_WORDS_EXTENDED, is_ascii_english
from experiments.phase16a_cross_domain_pos import classify_concept_pos
from experiments.concept_quality import optimize_concepts
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


# ── Phase 20 triple-filter concept cleaning ───────────────────────────

def phase20_filter_concepts(
    concept_chunks: dict[str, list[str]],
    chunk_texts: list[str],
    n_total: int,
) -> tuple[dict[str, list[str]], dict]:
    """Apply Phase 20's triple filter to concept_chunks.

    Replaces optimize_concepts. Three filters:
    1. Corpus verification: concept must appear as a real word in docs
    2. POS filter: reject verb -ing/-ed forms
    3. Prefill word blacklist + extended stopwords + ASCII

    Returns (filtered concept_chunks, metadata).
    """
    # Build corpus word set
    corpus_words = set()
    for text in chunk_texts:
        for m in re.finditer(r'[a-zA-Z]{4,}', text):
            corpus_words.add(m.group().lower())

    kept = {}
    removed = {"not_in_corpus": [], "verb_form": [], "prefill_stopword": [],
               "non_ascii": [], "too_rare": []}

    for concept, chunks in concept_chunks.items():
        df = len(chunks)
        concept_lower = concept.lower()

        # Filter: min DF = 2
        if df < 2:
            removed["too_rare"].append((concept, df))
            continue

        # Filter 1: ASCII
        if not is_ascii_english(concept_lower):
            removed["non_ascii"].append((concept, df))
            continue

        # Filter 2: prefill words + extended stopwords
        if concept_lower in PREFILL_WORDS or concept_lower in STOP_WORDS_EXTENDED:
            removed["prefill_stopword"].append((concept, df))
            continue

        # Filter 3: POS — reject verbs
        pos = classify_concept_pos(concept_lower)
        if pos in ("VBG", "VBD"):
            removed["verb_form"].append((concept, df))
            continue

        # Filter 4: corpus verification
        if concept_lower not in corpus_words:
            removed["not_in_corpus"].append((concept, df))
            continue

        kept[concept] = chunks

    # Compute IDF
    idf = {}
    for concept, chunks in kept.items():
        df = len(chunks)
        idf[concept] = math.log((n_total + 1) / (df + 1)) + 1

    meta = {
        "n_before": len(concept_chunks),
        "n_after": len(kept),
        "removed": {k: len(v) for k, v in removed.items()},
        "removed_detail": removed,
        "idf": idf,
    }
    return kept, meta


# ── Build + propagate a flat concept graph ────────────────────────────

def build_and_propagate(
    graph: ConceptGraph,
    seed_ids: list[str],
    propagate_k: int = 20,
) -> list[str]:
    """Run flat concept graph propagation (Stage 7c method)."""
    propagated = graph.propagate(seed_ids, max_propagate=propagate_k,
                                  use_idf=True, use_bm25=True)
    merged = list(seed_ids[:10])  # keep top-10 seeds
    for pid in propagated:
        if pid not in merged:
            merged.append(pid)
        if len(merged) >= TOP_K:
            break
    # Backfill from nothing — caller handles B0 backfill
    return merged[:TOP_K]


def rebuild_graph_index(graph: ConceptGraph, concept_chunks: dict[str, list[str]],
                         idf: dict[str, float]):
    """Rebuild graph's internal index after filtering."""
    graph.concept_chunks = concept_chunks
    graph.idf = idf
    graph.chunk_concepts = defaultdict(list)
    for concept, cids in concept_chunks.items():
        for cid in cids:
            if concept not in graph.chunk_concepts[cid]:
                graph.chunk_concepts[cid].append(concept)


# ── Main benchmark ────────────────────────────────────────────────────

def run_benchmark(lens, lens_model, tokenizer, embed,
                  domain: str = "medical",
                  max_queries: int = 30,
                  max_chunks: int = 200):
    print(f"Phase 23: Hybrid filtered benchmark (Phase 20 filter vs Stage 7c)")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # 1. Load corpus
    print(f"\n[1/4] Loading {domain} corpus...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Embed
    print(f"\n[2/4] Embedding...")
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts_list), dtype=np.float32)
    query_emb = np.asarray(embed.embed([q["question"] for q in questions]), dtype=np.float32)

    # 3. Extract concepts (once) + build TWO graphs with different filters
    print(f"\n[3/4] Extracting concepts + building graphs...")
    # Extract raw concepts per chunk (same as Stage 7c)
    raw_concepts = {}
    for i, cid in enumerate(chunk_ids):
        text = chunk_text_map[cid]
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer, text, n_words=5)
        raw_concepts[cid] = concepts
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)} chunks", flush=True)

    # Build graph with Stage 7c filter (optimize_concepts)
    graph_7c = ConceptGraph()
    for cid in chunk_ids:
        graph_7c.add_chunk(cid, raw_concepts[cid])
    build_texts = [chunk_text_map[cid] for cid in chunk_ids]
    optimized_7c, meta_7c = optimize_concepts(
        dict(graph_7c.concept_chunks), len(chunk_ids),
        chunk_texts=build_texts)
    rebuild_graph_index(graph_7c, optimized_7c, meta_7c["idf"])
    graph_7c.compute_tf(chunk_text_map)
    print(f"  Stage 7c filter: {meta_7c['n_before']} → {meta_7c['n_after']} concepts "
          f"(removed {len(meta_7c.get('removed_artifacts',[]))} artifacts)")

    # Build graph with Phase 20 filter
    graph_p20 = ConceptGraph()
    for cid in chunk_ids:
        graph_p20.add_chunk(cid, raw_concepts[cid])
    filtered_p20, meta_p20 = phase20_filter_concepts(
        dict(graph_p20.concept_chunks), build_texts, len(chunk_ids))
    rebuild_graph_index(graph_p20, filtered_p20, meta_p20["idf"])
    graph_p20.compute_tf(chunk_text_map)
    print(f"  Phase 20 filter: {meta_p20['n_before']} → {meta_p20['n_after']} concepts "
          f"(removed: {meta_p20['removed']})")

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

        # B0
        b0_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)
        eval_tasks.append(("B0", level, q["question"], b0_context, evidence))

        # Stage 7c flat
        seed_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, 10)
        seed_ids = [cid for cid, _ in seed_hits]
        merged_7c = build_and_propagate(graph_7c, seed_ids)
        if len(merged_7c) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_7c:
                    merged_7c.append(cid)
                if len(merged_7c) >= TOP_K:
                    break
        ctx_7c = " ".join(chunk_text_map[cid] for cid in merged_7c[:TOP_K])
        eval_tasks.append(("flat_stage7c", level, q["question"], ctx_7c, evidence))

        # Phase 20 flat
        merged_p20 = build_and_propagate(graph_p20, seed_ids)
        if len(merged_p20) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_p20:
                    merged_p20.append(cid)
                if len(merged_p20) >= TOP_K:
                    break
        ctx_p20 = " ".join(chunk_text_map[cid] for cid in merged_p20[:TOP_K])
        eval_tasks.append(("flat_phase20", level, q["question"], ctx_p20, evidence))

    # LLM judge
    print(f"  Concurrent LLM judge ({len(eval_tasks)} calls)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    def _judge(task):
        method, level, question, context, evidence = task
        llm_local = DeepSeekProvider()
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)
        return method, level, recall

    methods = ["B0", "flat_stage7c", "flat_phase20"]
    results = {m: [] for m in methods}
    by_level = {m: defaultdict(list) for m in methods}

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_judge, t): t for t in eval_tasks}
        done = 0
        for future in as_completed(futures):
            method, level, recall = future.result()
            results[method].append(recall)
            by_level[method][level].append(recall)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(eval_tasks)}", flush=True)

    # Summary
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS ({domain})")
    print(f"{'='*70}")
    print(f"  {'method':<18} {'overall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*58}")
    b0_mean = np.mean(results["B0"]) if results["B0"] else 1
    for m in methods:
        overall = np.mean(results[m]) if results[m] else 0
        lvls = []
        for lv in ["L1","L2","L3","L4"]:
            s = by_level[m].get(lv, [])
            lvls.append(f"{np.mean(s):>7.1%}" if s else f"{'N/A':>7}")
        rel = f" ({overall/b0_mean:.0%})" if b0_mean > 0 and m != "B0" else ""
        print(f"  {m:<18} {overall:>7.1%}{rel} {lvls[0]} {lvls[1]} {lvls[2]} {lvls[3]}")

    # Save
    out = {
        "method": "hybrid_filtered_benchmark",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(questions),
        "results": {
            m: {
                "overall": float(np.mean(results[m])) if results[m] else 0,
                "by_level": {lv: (float(np.mean(by_level[m][lv]))
                                  if by_level[m].get(lv) else None)
                             for lv in ["L1","L2","L3","L4"]},
                "n": len(results[m]),
            } for m in methods
        },
        "filter_stats": {
            "stage7c": {"n_before": meta_7c["n_before"], "n_after": meta_7c["n_after"]},
            "phase20": {"n_before": meta_p20["n_before"], "n_after": meta_p20["n_after"],
                        "removed": meta_p20["removed"]},
        },
    }
    out_path = EXP / f"phase23_hybrid_benchmark_{domain}.json"
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
                  domain=args.domain, max_queries=args.max_queries,
                  max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
