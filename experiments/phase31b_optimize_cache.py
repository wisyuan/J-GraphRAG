"""Phase 31b: 对概念缓存应用完整优化管线。

Phase 31 缓存了原始概念提取结果（未过滤，含 artifact）。
本脚本对缓存应用已验证的优化管线：

1. Phase 25 三重过滤 + BM25 补全（用于 J-AugRAG 检索）
2. Phase 18 深度梯度 COM 分析（用于概念监控）
3. 生成过滤后缓存 + 统计报告

输入：concept_cache_{domain}.json（Phase 31 原始缓存）
输出：concept_cache_{domain}_optimized.json（过滤 + 补全后）

后续实验直接加载优化后缓存，包含：
  - raw_concepts: 原始概念（未过滤）
  - filtered_concepts: Phase 25 三重过滤 + BM25 补全后
  - concept_chunks: {concept → [chunk_ids]}（过滤后）
  - graph_stats: 图统计
  - filter_meta: 过滤元数据（移除了哪些 artifact）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate
    python -m experiments.phase31b_optimize_cache
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase25_filter_bpe_benchmark import phase25_filter_with_bpe
from experiments.phase10_jlens_stage7c import ConceptGraph
from experiments.concept_quality import optimize_concepts

REPO = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO / "data" / "m6" / "concept_cache"


def optimize_cache(domain: str) -> dict:
    """Apply full optimization pipeline to raw concept cache.

    Pipeline:
      raw concepts → ConceptGraph → Phase 25 triple filter + BM25 → optimized cache
    """
    raw_path = CACHE_DIR / f"concept_cache_{domain}.json"
    if not raw_path.exists():
        print(f"  [{domain}] SKIP — raw cache not found at {raw_path}")
        return None

    raw = json.loads(raw_path.read_text())
    print(f"\n  [{domain}] Raw cache: {raw['n_chunks']} chunks, "
          f"{raw['n_unique_concepts']} unique concepts")

    # Reconstruct chunk texts + raw concepts
    chunk_ids = list(raw["chunks"].keys())
    chunk_text_map = {cid: data["text_excerpt"] for cid, data in raw["chunks"].items()}
    # Note: text_excerpt is only 100 chars — for BM25 we need full text
    # Load full corpus for BM25 completion
    from experiments.phase4_dig_graphragbench import load_graphrag_bench
    full_corpus, _ = load_graphrag_bench(domain, max_queries=1)
    full_text_map = {cid: full_corpus.get(cid, chunk_text_map.get(cid, ""))
                     for cid in chunk_ids}
    chunk_texts_list = [full_text_map.get(cid, "") for cid in chunk_ids]

    # Build raw concept graph
    graph = ConceptGraph()
    raw_concepts = {}
    for cid in chunk_ids:
        concepts = raw["chunks"][cid]["concepts"]
        raw_concepts[cid] = concepts
        graph.add_chunk(cid, concepts)

    n_before = len(graph.concept_chunks)
    print(f"  [{domain}] Raw graph: {n_before} concepts")

    # === Phase 25: triple filter + BM25 completion ===
    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), chunk_texts_list, len(chunk_ids))

    # Rebuild graph index
    graph.concept_chunks = filtered
    graph.idf = meta["idf"]
    graph.chunk_concepts = defaultdict(list)
    for concept, cids in filtered.items():
        for cid in cids:
            if concept not in graph.chunk_concepts[cid]:
                graph.chunk_concepts[cid].append(concept)

    n_after = len(filtered)
    print(f"  [{domain}] Phase 25 filter: {n_before}→{n_after} concepts")
    print(f"    Removed: {meta['removed']}")
    if meta.get("bpe_completions"):
        print(f"    BPE completions: {meta['bpe_completions']}")

    # Build optimized per-chunk concept mapping
    optimized_chunks = {}
    for cid in chunk_ids:
        optimized_chunks[cid] = {
            "raw_concepts": raw_concepts[cid],
            "filtered_concepts": graph.chunk_concepts.get(cid, []),
            "text_excerpt": raw["chunks"][cid]["text_excerpt"],
        }

    # Graph stats
    stats = graph.graph_stats()

    # Concept frequency after filtering
    from collections import Counter
    all_filtered = []
    for cid in chunk_ids:
        all_filtered.extend(graph.chunk_concepts.get(cid, []))
    filtered_freq = Counter(c.lower() for c in all_filtered)

    result = {
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "optimization": {
            "method": "phase25_triple_filter_bm25",
            "n_before": n_before,
            "n_after": n_after,
            "removed": meta["removed"],
            "bpe_completions": meta.get("bpe_completions", {}),
        },
        "graph_stats": stats,
        "filtered_concept_frequency": dict(filtered_freq.most_common(30)),
        "concept_chunks": {k: v for k, v in filtered.items()},  # concept → [chunk_ids]
        "chunks": optimized_chunks,
        "raw_extraction_time_s": raw["extraction_time_s"],
    }

    # Save
    out_path = CACHE_DIR / f"concept_cache_{domain}_optimized.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  [{domain}] Saved to {out_path}")
    print(f"    Filtered concepts: {n_after}")
    print(f"    Top filtered: {list(filtered_freq.most_common(10))}")

    return result


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 60)
    print("Phase 31b: Apply optimization pipeline to concept cache")
    print("=" * 60)

    for domain in ["medical", "novel"]:
        result = optimize_cache(domain)
        if result:
            print(f"\n  Summary [{domain}]:")
            print(f"    Raw → Filtered: {result['optimization']['n_before']} → "
                  f"{result['optimization']['n_after']} concepts")
            print(f"    Graph: {result['graph_stats']}")


if __name__ == "__main__":
    main()
