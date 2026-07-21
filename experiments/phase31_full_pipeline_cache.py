"""Phase 31: 完整概念提取管线缓存。

从零实现 Phase 25 验证过的完整管线，对全量 GraphRAG-Bench 语料做概念提取。

完整管线（每 chunk）：
  1. concern prompt 构建（Phase 20）
  2. extract_depth_gradient → 全 27 层 workspace 扫描（Phase 17）
  3. compute_concept_profiles_filtered → 稳定性 + corpus + POS 过滤（Phase 20）
  4. BM25 BPE 补全（Phase 25 加回来的 Artifact 修复）

产出（每 chunk）：
  - raw_gradient: 27 层的 top-k token（诊断用）
  - filtered_concepts: 经完整管线过滤 + BM25 补全后的概念词
  - com_profiles: 概念的 COM 深度分布（监控用）
  - extraction_time_s: 提取耗时

后续实验直接加载缓存，不需要 Qwen forward pass。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase31_full_pipeline_cache [--domain medical] [--max-chunks 0]
"""
from __future__ import annotations

import argparse
import json
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
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
    classify_by_com_gap, PREFILL_WORDS,
)
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english, build_corpus_word_set,
)
from experiments.phase16a_cross_domain_pos import classify_concept_pos
from experiments.concept_quality import (
    complete_prefix, build_corpus_term_freq, _get_wordnet_nouns,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6" / "concept_cache"
EXP.mkdir(parents=True, exist_ok=True)


def extract_concepts_full_pipeline(
    lens, lens_model, tokenizer,
    chunk_text: str,
    all_layers: list[int],
    corpus_words: set[str],
    corpus_freq: dict[str, int],
    wn_nouns: set[str],
) -> dict:
    """Full Phase 25 pipeline for a single chunk.

    1. concern prompt → full-layer depth gradient
    2. compute_concept_profiles_filtered (stability + corpus + POS)
    3. BM25 BPE completion
    4. Return filtered concepts + COM profiles

    This replaces Stage 7c's single-layer extract_chunk_concepts.
    """
    # Step 1: concern prompt + full depth gradient (1 forward pass, all 27 layers)
    prompt = build_concern_prompt_full([chunk_text], tokenizer)
    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    # Step 2: Phase 20 triple filter — strict first, then relax
    # Strict: require_corpus=True (only concepts verified in the chunk text)
    # This filters out lens artifacts (amac/emonic/alink) that aren't real words
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=2,
        require_corpus=True, require_noun=True)

    # Relax: if too strict (< 2 concepts survived), allow non-corpus nouns
    if len(profiles) < 2:
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=2,
            require_corpus=False, require_noun=True)
        # Filter manually: only keep if in corpus (final safety net)
        profiles = [p for p in profiles if p.in_corpus]

    # Collect concept words from profiles
    concepts_raw = [p.word for p in profiles]

    # Fallback: if too few after strict filtering, use single-layer L26 readout
    # (same as Stage 7c — this is the proven fallback from Phase 25 benchmark)
    if len(concepts_raw) < 2:
        from experiments.phase10_jlens_stage7c import extract_chunk_concepts
        concepts_raw = extract_chunk_concepts(
            lens, lens_model, tokenizer, chunk_text, n_words=5)
        # Re-run corpus verification on fallback concepts
        concepts_raw = [c for c in concepts_raw if c.lower() in corpus_words]

    # Step 3: BM25 BPE completion
    concepts_completed = []
    for c in concepts_raw[:8]:
        if len(c) <= 5:
            result = complete_prefix(c, corpus_freq, wn_nouns)
            concepts_completed.append(result if result else c)
        else:
            concepts_completed.append(c)

    # Deduplicate
    seen = set()
    concepts_final = []
    for c in concepts_completed:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            concepts_final.append(c)

    # Build COM profiles for output (diagnostic)
    com_profiles = []
    for p in profiles:
        com_profiles.append({
            "word": p.word,
            "com": round(p.com, 1),
            "n_layers": p.n_layers,
            "first": p.first_layer,
            "last": p.last_layer,
            "in_corpus": p.in_corpus,
        })

    # Raw gradient summary (top-3 per layer, for diagnostics)
    gradient_summary = {}
    for layer in all_layers:
        words = gradient.get(layer, [])
        gradient_summary[str(layer)] = [
            {"token": w["token"], "prob": w["prob"]}
            for w in words[:3]
        ]

    return {
        "concepts": concepts_final[:8],
        "n_concepts": len(concepts_final),
        "com_profiles": com_profiles,
        "gradient_summary": gradient_summary,
    }


def run_full_extraction(lens, lens_model, tokenizer, domain: str,
                        max_chunks: int = 0) -> dict:
    """Extract concepts for all chunks using full Phase 25 pipeline."""
    print(f"\n  [{domain}] Loading corpus...")
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())
    if max_chunks > 0:
        chunk_items = chunk_items[:max_chunks]

    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    print(f"  [{domain}] {len(chunk_ids)} chunks to process")

    # Build corpus resources for BM25 + corpus verification
    print(f"  [{domain}] Building corpus resources...")
    corpus_words = build_corpus_word_set(chunk_texts_list)
    corpus_freq = build_corpus_term_freq(chunk_texts_list)
    wn_nouns = _get_wordnet_nouns()
    print(f"  [{domain}] Corpus words: {len(corpus_words)}, "
          f"WN nouns: {len(wn_nouns)}")

    all_layers = lens.source_layers

    cache = {
        "domain": domain,
        "model": detect_model()["name"],
        "pipeline": "phase25_full (concern + depth_gradient + triple_filter + BM25)",
        "n_chunks": len(chunk_ids),
        "chunks": {},
    }

    t_start = time.perf_counter()
    extraction_times = []

    for i, cid in enumerate(chunk_ids):
        t0 = time.perf_counter()
        result = extract_concepts_full_pipeline(
            lens, lens_model, tokenizer,
            chunk_text_map[cid], all_layers,
            corpus_words, corpus_freq, wn_nouns)
        t1 = time.perf_counter()

        cache["chunks"][cid] = {
            "concepts": result["concepts"],
            "n_concepts": result["n_concepts"],
            "com_profiles": result["com_profiles"],
            "text_excerpt": chunk_text_map[cid][:100],
            "extraction_time_s": round(t1 - t0, 4),
        }
        extraction_times.append(t1 - t0)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(chunk_ids) - i - 1)
            print(f"    [{domain}] {i+1}/{len(chunk_ids)} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_total = time.perf_counter() - t_start
    cache["extraction_time_s"] = round(t_total, 1)
    cache["avg_per_chunk_s"] = round(np.mean(extraction_times), 4)

    # Stats: concept frequency across all chunks
    all_concepts = []
    for data in cache["chunks"].values():
        all_concepts.extend(data["concepts"])
    from collections import Counter
    concept_freq = Counter(c.lower() for c in all_concepts)
    cache["concept_frequency"] = dict(concept_freq.most_common(40))
    cache["n_unique_concepts"] = len(concept_freq)

    # Build concept_chunks for graph construction
    concept_chunks = defaultdict(list)
    for cid in chunk_ids:
        for concept in cache["chunks"][cid]["concepts"]:
            concept_chunks[concept.lower()].append(cid)
    cache["concept_chunks"] = dict(concept_chunks)

    print(f"  [{domain}] Done: {t_total:.0f}s ({cache['avg_per_chunk_s']}s/chunk)")
    print(f"  [{domain}] Unique concepts: {cache['n_unique_concepts']}")
    print(f"  [{domain}] Top: {list(concept_freq.most_common(10))}")

    return cache


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["medical", "novel", "all"])
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="0 = all chunks")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase 31: Full pipeline concept extraction + cache")
    print("  Pipeline: Phase 25 (concern + depth_gradient + triple_filter + BM25)")
    print("=" * 60)

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print(f"\n[2/2] Extracting concepts...")
    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]

    for domain in domains:
        cache = run_full_extraction(
            lens, lens_model, tokenizer, domain,
            max_chunks=args.max_chunks)

        out_path = EXP / f"concept_cache_{domain}_full.json"
        out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
