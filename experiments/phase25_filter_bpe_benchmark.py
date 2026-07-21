"""Phase 25: 三重过滤 + BM25 补全——修正 Phase 23 的遗漏。

Phase 23 证明 Phase 20 三重过滤器 > Stage 7c optimize_concepts（106% vs 105%）。
但 Phase 23 的过滤器**丢掉了 BPE 补全**——14 个可补全的前缀（odyn→odynophagia,
preg→pregnancy）被当作 "not_in_corpus" 过滤掉。

本实验修正：在 corpus 验证**之前**插入 BM25 补全。

流程对比：
  Phase 20: ASCII → POS → prefill → corpus验证 → DF过滤
  Phase 25: ASCII → POS → prefill → BM25补全 → corpus验证 → DF过滤

补全后，BPE 前缀变成完整词（odyn→odynophagia），通过 corpus 验证。

三种方法对比：
  1. B0 (RAG baseline)
  2. flat_stage7c (optimize_concepts: DF + corpus + BPE补全)
  3. flat_phase25 (三重过滤 + BPE补全)

预期：flat_phase25 > flat_stage7c > B0
  因为 Phase 25 = Phase 20 的严格过滤 + Stage 7c 的 BPE 补全

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase25_filter_bpe_benchmark
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
from experiments.concept_quality import (
    optimize_concepts, complete_prefix, build_corpus_term_freq,
    _get_wordnet_nouns,
)
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


def phase25_filter_with_bpe(
    concept_chunks: dict[str, list[str]],
    chunk_texts: list[str],
    n_total: int,
) -> tuple[dict[str, list[str]], dict]:
    """Phase 25: triple filter + BM25 BPE completion.

    Key fix vs Phase 23: BPE completion happens BEFORE corpus verification,
    so prefixes like 'odyn'→'odynophagia' pass instead of being dropped.

    Pipeline: ASCII → POS → prefill → BM25补全 → corpus验证 → DF过滤
    """
    # Build corpus resources
    corpus_words = set()
    for text in chunk_texts:
        for m in re.finditer(r'[a-zA-Z]{4,}', text):
            corpus_words.add(m.group().lower())
    corpus_freq = build_corpus_term_freq(chunk_texts)
    wn_nouns = _get_wordnet_nouns()

    kept = {}  # completed_concept → merged chunk list
    bpe_completions = {}
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

        # ★ BM25 completion (NEW vs Phase 23)
        # If concept is short (≤5 chars), try to complete it to a full word
        completed = concept_lower
        if len(concept_lower) <= 5:
            result = complete_prefix(concept_lower, corpus_freq, wn_nouns)
            if result and result != concept_lower:
                completed = result.lower()
                bpe_completions[concept] = completed

        # Filter 4: corpus verification (on completed word)
        if completed not in corpus_words:
            removed["not_in_corpus"].append((concept, df))
            continue

        # Merge: if completed word already exists, merge chunk lists
        if completed in kept:
            existing = set(kept[completed])
            existing.update(chunks)
            kept[completed] = sorted(existing)
        else:
            kept[completed] = chunks

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
        "bpe_completions": bpe_completions,
        "idf": idf,
    }
    return kept, meta


def rebuild_graph_index(graph: ConceptGraph, concept_chunks: dict[str, list[str]],
                         idf: dict[str, float]):
    graph.concept_chunks = concept_chunks
    graph.idf = idf
    graph.chunk_concepts = defaultdict(list)
    for concept, cids in concept_chunks.items():
        for cid in cids:
            if concept not in graph.chunk_concepts[cid]:
                graph.chunk_concepts[cid].append(concept)


def build_and_propagate(graph: ConceptGraph, seed_ids: list[str],
                        propagate_k: int = 20) -> list[str]:
    propagated = graph.propagate(seed_ids, max_propagate=propagate_k,
                                  use_idf=True, use_bm25=True)
    merged = list(seed_ids[:10])
    for pid in propagated:
        if pid not in merged:
            merged.append(pid)
        if len(merged) >= TOP_K:
            break
    return merged[:TOP_K]


def run_benchmark(lens, lens_model, tokenizer, embed,
                  domain: str = "medical",
                  max_queries: int = 30, max_chunks: int = 200):
    print(f"Phase 25: Triple filter + BM25 completion benchmark")
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

    # 3. Extract concepts + build graphs
    print(f"\n[3/4] Extracting concepts + building graphs...")
    raw_concepts = {}
    for i, cid in enumerate(chunk_ids):
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer,
                                           chunk_text_map[cid], n_words=5)
        raw_concepts[cid] = concepts
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)}", flush=True)

    build_texts = [chunk_text_map[cid] for cid in chunk_ids]

    # Graph 1: Stage 7c (optimize_concepts: DF + corpus + BPE)
    graph_7c = ConceptGraph()
    for cid in chunk_ids:
        graph_7c.add_chunk(cid, raw_concepts[cid])
    opt_7c, meta_7c = optimize_concepts(
        dict(graph_7c.concept_chunks), len(chunk_ids), chunk_texts=build_texts)
    rebuild_graph_index(graph_7c, opt_7c, meta_7c["idf"])
    graph_7c.compute_tf(chunk_text_map)
    print(f"  Stage 7c: {meta_7c['n_before']}→{meta_7c['n_after']} "
          f"(BPE: {len(meta_7c.get('bpe_completions',{}))} completions)")

    # Graph 2: Phase 25 (triple filter + BM25 completion)
    graph_p25 = ConceptGraph()
    for cid in chunk_ids:
        graph_p25.add_chunk(cid, raw_concepts[cid])
    filtered_p25, meta_p25 = phase25_filter_with_bpe(
        dict(graph_p25.concept_chunks), build_texts, len(chunk_ids))
    rebuild_graph_index(graph_p25, filtered_p25, meta_p25["idf"])
    graph_p25.compute_tf(chunk_text_map)
    print(f"  Phase 25: {meta_p25['n_before']}→{meta_p25['n_after']} "
          f"(BPE: {len(meta_p25['bpe_completions'])} completions)")
    if meta_p25["bpe_completions"]:
        print(f"    completions: {dict(list(meta_p25['bpe_completions'].items())[:10])}")

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
        b0_ctx = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)
        eval_tasks.append(("B0", level, q["question"], b0_ctx, evidence))

        # Stage 7c
        seed_ids = [cid for cid, _ in cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, 10)]
        merged_7c = build_and_propagate(graph_7c, seed_ids)
        if len(merged_7c) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_7c: merged_7c.append(cid)
                if len(merged_7c) >= TOP_K: break
        eval_tasks.append(("flat_stage7c", level, q["question"],
                           " ".join(chunk_text_map[cid] for cid in merged_7c[:TOP_K]), evidence))

        # Phase 25
        merged_p25 = build_and_propagate(graph_p25, seed_ids)
        if len(merged_p25) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_p25: merged_p25.append(cid)
                if len(merged_p25) >= TOP_K: break
        eval_tasks.append(("flat_phase25", level, q["question"],
                           " ".join(chunk_text_map[cid] for cid in merged_p25[:TOP_K]), evidence))

    # LLM judge
    print(f"  Concurrent LLM judge ({len(eval_tasks)} calls)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    def _judge(task):
        method, level, question, context, evidence = task
        llm_local = DeepSeekProvider()
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)
        return method, level, recall

    methods = ["B0", "flat_stage7c", "flat_phase25"]
    results = {m: [] for m in methods}
    by_level = {m: defaultdict(list) for m in methods}

    with ThreadPoolExecutor(max_workers=8) as pool:
        done = 0
        for future in as_completed({pool.submit(_judge, t): t for t in eval_tasks}):
            method, level, recall = future.result()
            results[method].append(recall)
            by_level[method][level].append(recall)
            done += 1
            if done % 20 == 0: print(f"    {done}/{len(eval_tasks)}", flush=True)

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
        "method": "triple_filter_bpe_completion_benchmark",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(questions),
        "results": {m: {
            "overall": float(np.mean(results[m])) if results[m] else 0,
            "by_level": {lv: (float(np.mean(by_level[m][lv]))
                              if by_level[m].get(lv) else None)
                         for lv in ["L1","L2","L3","L4"]},
            "n": len(results[m]),
        } for m in methods},
        "filter_stats": {
            "stage7c": {"n_before": meta_7c["n_before"], "n_after": meta_7c["n_after"],
                        "bpe_completions": meta_7c.get("bpe_completions", {})},
            "phase25": {"n_before": meta_p25["n_before"], "n_after": meta_p25["n_after"],
                        "bpe_completions": meta_p25["bpe_completions"],
                        "removed": meta_p25["removed"]},
        },
    }
    out_path = EXP / f"phase25_filter_bpe_benchmark_{domain}.json"
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
