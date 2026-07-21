"""Phase 10 Stage 7b — 概念锚点检索：query→J-Lens 概念词→bge-m3 匹配文档。

Stage 7a 证明 J-Lens 聚类不增强检索。但 Stage 3/5 证明 J-Lens 关切耦合能
读出高质量概念词（自然语言域 80%）。本阶段测一个未被 Stage 7a 覆盖的路径：

  query → J-Lens 读出概念词（如 "statins", "PCOS"）
       → 概念词 bge-m3 嵌入 → 余弦匹配文档 chunks
       → top-k → LLM judge evidence recall

这测的是 J-Lens **概念提取**质量（核心命题），不是聚类质量。概念词是 query
的语义摘要，用 bge-m3 匹配文档保持了检索精度。

对比方案：
  B0:        query → bge-m3 → cosine（baseline）
  JL-concept: query → J-Lens 概念词 → bge-m3 → cosine（概念锚点）
  JL-hybrid: B0 top-50 ∪ concept top-50 → 合并去重排序

指标：evidence recall（复用 Phase 4 GraphRAG-Bench 的 LLM judge 方法）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage7b --domain medical
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10  # chunks for evidence recall context (matches Phase 4)


def extract_concept_words(lens, lens_model, tokenizer, query: str,
                          n_words: int = 5) -> list[str]:
    """J-Lens concept extraction for a query.

    Uses concern-coupled prompt (chat template + assistant prefill).
    Returns top-n concept words from the last source layer (L26).
    Filters BPE fragments + stopwords, keeps real content words.
    """
    user_msg = (
        f"What are the key concepts in this question? "
        f"List {n_words} one-word concepts.\n\n{query}"
    )
    prefill = "The key concepts are"
    prompt = prefill
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            prompt = f"{user_msg}\n{prefill}"

    sample_layers = [lens.source_layers[-1]]
    lens_logits, model_logits, _ = lens.apply(
        lens_model, prompt, layers=sample_layers,
        positions=[-1], max_seq_len=256,
    )

    # Get top-20 tokens from last source layer, filter to content words
    last = lens.source_layers[-1]
    probs = torch.softmax(lens_logits[last][0].float(), dim=-1)
    topk = probs.topk(20)
    STOP = {"the", "and", "for", "that", "with", "from", "this", "are", "was",
            "were", "been", "have", "has", "will", "would", "could", "should",
            "not", "but", "into", "function", "def", "return", "class", "import",
            "concept", "concepts", "key", "main", "topic", "also", "they",
            "them", "than", "then", "when", "what", "each", "more", "most",
            "some", "such", "only", "very", "just", "like", "question", "list",
            "following", "above", "based"}

    words = []
    seen = set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        if (len(tok) >= 3 and tok.isalpha() and tok.lower() not in STOP
                and tok.lower() not in seen
                and not (tok[0].isupper() and len(tok) < 3)):
            seen.add(tok.lower())
            words.append(tok)
        if len(words) >= n_words:
            break
    return words


def cosine_search(query_emb: np.ndarray, corpus_emb: np.ndarray,
                  corpus_ids: list[str], top_k: int = TOP_K) -> dict[str, float]:
    """Cosine similarity search, return {cid: score} for top_k."""
    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    c_norm = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-8)
    scores = c_norm @ q_norm
    top_idx = np.argsort(scores)[::-1][:top_k]
    return {corpus_ids[i]: float(scores[i]) for i in top_idx}


def run_stage7b(embed, lens, lens_model, tokenizer, llm, domain: str = "medical",
                max_queries: int = 100):
    print(f"Phase 10 Stage 7b: Concept-anchor retrieval ({domain})")
    print(f"{'='*70}")

    # 1. Load GraphRAG-Bench corpus + questions
    corpus_chunks, questions = load_graphrag_bench(domain, max_queries)
    chunk_ids = list(corpus_chunks.keys())
    chunk_texts = [corpus_chunks[cid] for cid in chunk_ids]
    print(f"\n  {len(chunk_ids)} chunks, {len(questions)} questions")

    level_dist = Counter(q["level"] for q in questions)
    print(f"  levels: {dict(level_dist)}")

    # 2. Embed chunks (bge-m3)
    print(f"  embedding chunks...", end="", flush=True)
    chunk_emb = np.asarray(embed.embed(chunk_texts), dtype=np.float32)
    print(f" done ({chunk_emb.shape})")

    # 3. Embed queries (bge-m3) for B0 baseline
    query_texts = [q["question"] for q in questions]
    print(f"  embedding queries...", end="", flush=True)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float32)
    print(f" done")

    # 4. J-Lens concept extraction for each query
    print(f"  extracting J-Lens concept words for {len(questions)} queries...")
    query_concepts = []
    for i, q in enumerate(questions):
        words = extract_concept_words(lens, lens_model, tokenizer, q["question"])
        query_concepts.append(words)
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(questions)}", flush=True)

    # Show some examples
    print(f"\n  Sample concept extractions:")
    for i in range(min(5, len(questions))):
        print(f"    Q: {questions[i]['question'][:70]}...")
        print(f"    concepts: {query_concepts[i]}")

    # 5. Embed concept words (bge-m3) for JL-concept path
    print(f"\n  embedding concept words...", end="", flush=True)
    # Join concept words into a single "concept query" per question
    concept_queries = [" ".join(words) if words else q for words, q in
                       zip(query_concepts, query_texts)]
    concept_emb = np.asarray(embed.embed(concept_queries), dtype=np.float32)
    print(f" done")

    # 6. Run three retrieval methods + evidence recall
    print(f"\n  Running retrieval + evidence recall for {len(questions)} queries...")

    results = {"B0": [], "JL-concept": [], "JL-hybrid": []}
    by_level = {"B0": {}, "JL-concept": {}, "JL-hybrid": {}}

    for i, q in enumerate(questions):
        level = q["level"]
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not evidence:
            continue

        # B0: query embedding → cosine
        b0_hits = cosine_search(query_emb[i], chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(corpus_chunks[cid] for cid in b0_hits)
        b0_recall = llm_judge_evidence_recall(q["question"], b0_context, evidence, llm)
        results["B0"].append(b0_recall)
        by_level["B0"].setdefault(level, []).append(b0_recall)

        # JL-concept: concept words embedding → cosine
        jl_hits = cosine_search(concept_emb[i], chunk_emb, chunk_ids, TOP_K)
        jl_context = " ".join(corpus_chunks[cid] for cid in jl_hits)
        jl_recall = llm_judge_evidence_recall(q["question"], jl_context, evidence, llm)
        results["JL-concept"].append(jl_recall)
        by_level["JL-concept"].setdefault(level, []).append(jl_recall)

        # JL-hybrid: merge B0 top-50 + concept top-50, dedup, take top-10
        b0_50 = cosine_search(query_emb[i], chunk_emb, chunk_ids, 50)
        jl_50 = cosine_search(concept_emb[i], chunk_emb, chunk_ids, 50)
        merged = {**b0_50}
        for cid, score in jl_50.items():
            if cid in merged:
                merged[cid] = max(merged[cid], score)
            else:
                merged[cid] = score * 0.95  # slight penalty for concept-only hits
        top_merged = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        hybrid_context = " ".join(corpus_chunks[cid] for cid, _ in top_merged)
        hybrid_recall = llm_judge_evidence_recall(q["question"], hybrid_context, evidence, llm)
        results["JL-hybrid"].append(hybrid_recall)
        by_level["JL-hybrid"].setdefault(level, []).append(hybrid_recall)

        if (i + 1) % 20 == 0:
            b0_m = np.mean(results["B0"])
            jl_m = np.mean(results["JL-concept"])
            hy_m = np.mean(results["JL-hybrid"])
            print(f"    {i+1}/{len(questions)}: B0={b0_m:.3f} JL={jl_m:.3f} hybrid={hy_m:.3f}",
                  flush=True)

    # 7. Summary
    print(f"\n{'='*70}")
    print(f"RESULTS ({domain}, {len(results['B0'])} questions)")
    print(f"{'='*70}")

    print(f"\n  {'Method':<15} {'Overall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*55}")
    for method in ("B0", "JL-concept", "JL-hybrid"):
        overall = np.mean(results[method])
        levels = []
        for level in ("L1", "L2", "L3", "L4"):
            vals = by_level[method].get(level, [])
            levels.append(f"{np.mean(vals):.3f}" if vals else "  -  ")
        print(f"  {method:<15} {overall:>8.3f} {levels[0]:>8} {levels[1]:>8} {levels[2]:>8} {levels[3]:>8}")

    # J-Lens as % of B0
    b0_overall = np.mean(results["B0"])
    print(f"\n  === J-Lens as % of B0 (overall evidence recall) ===")
    for method in ("JL-concept", "JL-hybrid"):
        pct = np.mean(results[method]) / b0_overall * 100 if b0_overall > 0 else 0
        delta = np.mean(results[method]) - b0_overall
        print(f"    {method}: {pct:.0f}% of B0 (Δ{delta:+.3f})")

    out = {
        "method": "concept_anchor_retrieval",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results["B0"]),
        "overall": {m: float(np.mean(results[m])) for m in results},
        "by_level": {m: {l: float(np.mean(v)) if v else 0
                         for l, v in by_level[m].items()} for m in by_level},
        "sample_concepts": [
            {"question": questions[i]["question"][:100],
             "concepts": query_concepts[i]}
            for i in range(min(10, len(questions)))
        ],
    }
    out_path = EXP / f"phase10_stage7b_concept_anchor_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 7b: concept-anchor retrieval")
    ap.add_argument("--domain", default="medical", choices=["medical", "novel"])
    ap.add_argument("--max-queries", type=int, default=100)
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

    print(f"\n[3/3] Running concept-anchor retrieval benchmark...")
    embed = CachedBgeM3Provider()
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    run_stage7b(embed, lens, lens_model, tokenizer, llm, args.domain, args.max_queries)


if __name__ == "__main__":
    main()
