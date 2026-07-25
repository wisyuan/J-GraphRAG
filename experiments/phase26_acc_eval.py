"""Phase 26: ACC 评估链 + J-GraphRAG 基线（含时间/VRAM 记录）。

核心命题：只要 J-GraphRAG 的 ACC 不远低于 SOTA（同 Qwen + 同 bge-m3），
且运算量（建图时间/VRAM）显著小于 SOTA，就证明 J-Lens 的优势。

本文件实现：
1. ACC 评估链：检索 context → DeepSeek 生成答案 → DeepSeek judge 答案正确性
2. J-GraphRAG 基线：Phase 25 配置 + ACC + evidence recall + 建图时间 + VRAM

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase26_acc_eval
"""
from __future__ import annotations

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
from experiments.corpus_loader import load_beir_fine_records
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage7c import (
    ConceptGraph, cosine_topk_ids, extract_chunk_concepts,
)
from experiments.phase25_filter_bpe_benchmark import (
    phase25_filter_with_bpe, rebuild_graph_index, build_and_propagate,
)
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


# ── ACC evaluation chain ──────────────────────────────────────────────

def generate_answer(question: str, context: str, llm) -> str:
    """Generate an answer from retrieved context using LLM."""
    prompt = (
        f"Based on the following context, answer the question concisely.\n\n"
        f"Context: {context[:6000]}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    try:
        msg = llm.complete(prompt, max_tokens=200)
        return msg.content if hasattr(msg, 'content') else str(msg)
    except Exception as e:
        return f"[ERROR: {e}]"


def judge_answer_correctness(
    question: str,
    generated_answer: str,
    gold_answer: str,
    llm,
) -> bool:
    """Judge whether the generated answer is correct (ACC).

    Uses LLM to compare generated answer against gold answer.
    Returns True if correct, False otherwise.
    """
    prompt = (
        f"Question: {question}\n\n"
        f"Gold answer: {gold_answer}\n\n"
        f"Generated answer: {generated_answer}\n\n"
        f"Is the generated answer correct? Does it convey the same "
        f"information as the gold answer? Answer with only YES or NO."
    )
    try:
        msg = llm.complete(prompt, max_tokens=500, thinking=False)  # v4 推理模型：judge 关 thinking（旧值 10 + 推理链 = 静默全 False）
        resp = msg.content if hasattr(msg, 'content') else str(msg)
        return resp.strip().upper().startswith("YES")
    except Exception:
        return False


# ── J-GraphRAG baseline with timing ───────────────────────────────────

def run_jgraphrag_baseline(lens, lens_model, tokenizer, embed,
                            domain: str = "medical",
                            max_queries: int = 28,
                            max_chunks: int = 200):
    print(f"Phase 26: J-GraphRAG baseline with ACC + timing")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats()

    # 1. Load corpus
    print(f"\n[1/5] Loading corpus...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Embed
    print(f"\n[2/5] Embedding...")
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts_list), dtype=np.float32)
    query_emb = np.asarray(embed.embed([q["question"] for q in questions]), dtype=np.float32)

    # 3. Concept extraction + graph building (TIMED)
    print(f"\n[3/5] Concept extraction + graph building (timed)...")
    t_start = time.perf_counter()

    raw_concepts = {}
    extraction_times = []  # per-chunk timing
    for i, cid in enumerate(chunk_ids):
        t0 = time.perf_counter()
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer,
                                           chunk_text_map[cid], n_words=5)
        t1 = time.perf_counter()
        extraction_times.append(t1 - t0)
        raw_concepts[cid] = concepts
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)}", flush=True)

    t_extract = time.perf_counter() - t_start
    avg_extract = np.mean(extraction_times)

    # Build + filter graph
    build_texts = [chunk_text_map[cid] for cid in chunk_ids]
    graph = ConceptGraph()
    for cid in chunk_ids:
        graph.add_chunk(cid, raw_concepts[cid])
    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), build_texts, len(chunk_ids))
    rebuild_graph_index(graph, filtered, meta["idf"])
    graph.compute_tf(chunk_text_map)

    t_build = time.perf_counter() - t_start  # total build time
    vram_peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"  Concepts: {meta['n_before']}→{meta['n_after']} "
          f"(BPE: {len(meta['bpe_completions'])})")
    print(f"  Extraction time: {t_extract:.1f}s ({avg_extract:.3f}s/chunk)")
    print(f"  Total build time: {t_build:.1f}s")
    print(f"  VRAM peak: {vram_peak:.2f}GB")

    # 4. Retrieval + evaluation
    print(f"\n[4/5] Retrieval + evaluation (ACC + evidence recall)...")
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_query(q_idx, question_data):
        q = question_data
        level = q["level"]
        question = q["question"]
        gold_answer = q.get("answer", "")
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]

        # B0 retrieval
        b0_hits = cosine_topk_ids(query_emb[q_idx], chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)

        # J-GraphRAG retrieval
        seed_ids = [cid for cid, _ in cosine_topk_ids(query_emb[q_idx], chunk_emb, chunk_ids, 10)]
        merged = build_and_propagate(graph, seed_ids)
        if len(merged) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged:
                    merged.append(cid)
                if len(merged) >= TOP_K:
                    break
        jgr_context = " ".join(chunk_text_map[cid] for cid in merged[:TOP_K])

        # LLM evaluation
        llm_local = DeepSeekProvider()

        # Generate answers
        b0_answer = generate_answer(question, b0_context, llm_local)
        jgr_answer = generate_answer(question, jgr_context, llm_local)

        # ACC
        b0_acc = judge_answer_correctness(question, b0_answer, gold_answer, llm_local)
        jgr_acc = judge_answer_correctness(question, jgr_answer, gold_answer, llm_local)

        # Evidence recall
        b0_recall = llm_judge_evidence_recall(question, b0_context, evidence, llm_local)
        jgr_recall = llm_judge_evidence_recall(question, jgr_context, evidence, llm_local)

        return {
            "level": level,
            "B0": {"acc": b0_acc, "recall": b0_recall, "answer": b0_answer[:100]},
            "JGR": {"acc": jgr_acc, "recall": jgr_recall, "answer": jgr_answer[:100]},
        }

    eval_tasks = [(i, q) for i, q in enumerate(questions) if q.get("answer")]
    results_list = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_eval_query, i, q): i for i, q in eval_tasks}
        done = 0
        for future in as_completed(futures):
            results_list.append(future.result())
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(eval_tasks)}", flush=True)

    # 5. Summary
    print(f"\n[5/5] Summary")
    print(f"{'='*70}")

    methods = ["B0", "JGR"]
    acc = {m: [] for m in methods}
    recall = {m: [] for m in methods}
    by_level_acc = {m: defaultdict(list) for m in methods}
    by_level_recall = {m: defaultdict(list) for m in methods}

    for r in results_list:
        for m in methods:
            acc[m].append(1.0 if r[m]["acc"] else 0.0)
            recall[m].append(r[m]["recall"])
            by_level_acc[m][r["level"]].append(1.0 if r[m]["acc"] else 0.0)
            by_level_recall[m][r["level"]].append(r[m]["recall"])

    print(f"\n  {'method':<15} {'ACC':>8} {'recall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*63}")
    for m in methods:
        a = np.mean(acc[m]) if acc[m] else 0
        r = np.mean(recall[m]) if recall[m] else 0
        lvls = []
        for lv in ["L1","L2","L3","L4"]:
            s = by_level_acc[m].get(lv, [])
            lvls.append(f"{np.mean(s):>7.1%}" if s else f"{'N/A':>7}")
        print(f"  {m:<15} {a:>7.1%} {r:>7.1%} {lvls[0]} {lvls[1]} {lvls[2]} {lvls[3]}")

    # Efficiency metrics
    print(f"\n  Efficiency:")
    print(f"    Build time:       {t_build:.1f}s ({avg_extract:.3f}s/chunk)")
    print(f"    VRAM peak:        {vram_peak:.2f}GB")
    print(f"    Concepts:         {meta['n_after']}")
    jgr_acc_mean = np.mean(acc["JGR"]) if acc["JGR"] else 0
    efficiency = jgr_acc_mean / max(t_build, 0.001)
    print(f"    J-GraphRAG ACC/build_time: {efficiency:.4f} (higher=better)")

    # Save
    out = {
        "method": "jgraphrag_baseline_acc_timing",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "efficiency": {
            "build_time_s": round(t_build, 1),
            "extraction_time_per_chunk_s": round(avg_extract, 4),
            "vram_peak_gb": round(vram_peak, 2),
            "n_concepts": meta["n_after"],
            "bpe_completions": meta["bpe_completions"],
        },
        "results": {
            m: {
                "acc": float(np.mean(acc[m])) if acc[m] else 0,
                "recall": float(np.mean(recall[m])) if recall[m] else 0,
                "acc_by_level": {lv: (float(np.mean(by_level_acc[m][lv]))
                                      if by_level_acc[m].get(lv) else None)
                                 for lv in ["L1","L2","L3","L4"]},
                "recall_by_level": {lv: (float(np.mean(by_level_recall[m][lv]))
                                         if by_level_recall[m].get(lv) else None)
                                    for lv in ["L1","L2","L3","L4"]},
                "n": len(acc[m]),
            } for m in methods
        },
        "per_query": results_list,
    }
    out_path = EXP / f"phase26_jgraphrag_baseline_{domain}.json"
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

    print(f"\n[2/2] Running J-GraphRAG baseline...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    embed = CachedBgeM3Provider()
    run_jgraphrag_baseline(lens, lens_model, tokenizer, embed,
                           domain="medical", max_queries=28, max_chunks=200)


if __name__ == "__main__":
    main()
