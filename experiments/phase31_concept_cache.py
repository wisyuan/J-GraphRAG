"""Phase 31: 全量概念提取 + 缓存。

为后续所有方案（J-AugRAG / 关系图 / 聚类关系图）准备基础数据。
全量提取 GraphRAG-Bench medical + novel 的所有 chunk 概念，
缓存到磁盘，后续实验直接读取。

缓存格式：{
    domain: "medical",
    n_chunks: 876,
    model: "qwen2.5-7b-it",
    extraction_time_s: 152,
    chunks: {
        chunk_id: {
            "text_excerpt": "...",
            "concepts": ["cancer", "tumor", ...],
            "extraction_time_s": 0.173
        }
    }
}

后续实验：
    cache = json.load(open("concept_cache_medical.json"))
    raw_concepts = {cid: data["concepts"] for cid, data in cache["chunks"].items()}

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase31_concept_cache
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage7c import extract_chunk_concepts
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6" / "concept_cache"
EXP.mkdir(parents=True, exist_ok=True)


def extract_all_concepts(lens, lens_model, tokenizer, domain: str,
                         max_chunks: int | None = None) -> dict:
    """Extract concepts for ALL chunks in a domain corpus.

    Returns cache dict with per-chunk concepts + metadata.
    """
    print(f"\n  [{domain}] Loading corpus...")
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())
    if max_chunks:
        chunk_items = chunk_items[:max_chunks]

    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    print(f"  [{domain}] {len(chunk_ids)} chunks to process")

    cache = {
        "domain": domain,
        "model": detect_model()["name"],
        "n_chunks": len(chunk_ids),
        "chunks": {},
    }

    t_start = time.perf_counter()
    extraction_times = []

    for i, cid in enumerate(chunk_ids):
        t0 = time.perf_counter()
        concepts = extract_chunk_concepts(
            lens, lens_model, tokenizer, chunk_text_map[cid], n_words=5)
        t1 = time.perf_counter()

        cache["chunks"][cid] = {
            "concepts": concepts,
            "text_excerpt": chunk_text_map[cid][:100],
            "extraction_time_s": round(t1 - t0, 4),
        }
        extraction_times.append(t1 - t0)

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(chunk_ids) - i - 1)
            print(f"    [{domain}] {i+1}/{len(chunk_ids)} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_total = time.perf_counter() - t_start
    cache["extraction_time_s"] = round(t_total, 1)
    cache["avg_per_chunk_s"] = round(np.mean(extraction_times), 4)

    print(f"  [{domain}] Done: {t_total:.0f}s ({cache['avg_per_chunk_s']}s/chunk)")

    # Stats
    all_concepts = []
    for data in cache["chunks"].values():
        all_concepts.extend(data["concepts"])
    from collections import Counter
    concept_freq = Counter(c.lower() for c in all_concepts)
    cache["concept_frequency"] = dict(concept_freq.most_common(30))
    cache["n_unique_concepts"] = len(concept_freq)

    return cache


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 60)
    print("Phase 31: Full concept extraction + cache")
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

    print(f"\n[2/2] Extracting concepts for all domains...")

    for domain in ["medical", "novel"]:
        cache = extract_all_concepts(lens, lens_model, tokenizer, domain)

        out_path = EXP / f"concept_cache_{domain}.json"
        out_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
        print(f"\n  Saved to {out_path}")
        print(f"  Chunks: {cache['n_chunks']}")
        print(f"  Unique concepts: {cache['n_unique_concepts']}")
        print(f"  Time: {cache['extraction_time_s']}s ({cache['avg_per_chunk_s']}s/chunk)")
        print(f"  Top concepts: {list(cache['concept_frequency'].items())[:10]}")


if __name__ == "__main__":
    main()
