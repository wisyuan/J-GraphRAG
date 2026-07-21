"""Phase 26b: bge-large-en-v1.5 对比——排行榜换算基准。

GraphRAG-Bench leaderboard 使用 bge-large-en-v1.5 作为嵌入基线 + GPT-4o-mini 做
生成。我们在本地用 bge-large-en-v1.5 + Qwen2.5-7B 跑 B0 和 J-GraphRAG，获得：

1. bge-large-en-v1.5 的 B0 ACC（和 leaderboard 的 RAG w/o rerank 对比）
2. bge-large-en-v1.5 的 J-GraphRAG ACC
3. bge-m3 的 B0 ACC（已有：71.4%）
4. 换算系数 = leaderboard_RAG / our_bge_large_RAG
5. J-GraphRAG 在 leaderboard 标准下的等效分数

这样可以不用安装所有 SOTA 框架，通过嵌入模型基线换算估算 J-GraphRAG 的
leaderboard 等效排名。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase26b_bge_large_comparison
"""
from __future__ import annotations

import json
import os
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
    phase25_filter_with_bpe, rebuild_graph_index, build_and_propagate,
)
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10
BGE_LARGE_PATH = "/tmp/bge-large-en-v1.5"


class BgeLargeProvider:
    """bge-large-en-v1.5 embedding provider (sentence-transformers).

    Runs on CPU to avoid VRAM contention with Qwen 4-bit model.
    bge-large is only 335M params — CPU inference is fast enough
    for 200 chunks (~30s), and it keeps GPU VRAM free for Qwen.
    """

    def __init__(self, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(BGE_LARGE_PATH, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  bge-large-en-v1.5 loaded on {device} (dim={self.dim})")

    def embed(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)


def run_comparison(lens, lens_model, tokenizer,
                   domain: str = "medical",
                   max_queries: int = 28,
                   max_chunks: int = 200):
    print(f"Phase 26b: bge-large-en-v1.5 vs bge-m3 comparison")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # 1. Load corpus
    print(f"\n[1/5] Loading corpus...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    query_texts = [q["question"] for q in questions]
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Embed with bge-m3 (GPU) — embed all chunks + queries
    print(f"\n[2/5] Embedding with bge-m3...")
    embed_m3 = CachedBgeM3Provider()
    chunk_emb_m3 = np.asarray(embed_m3.embed(chunk_texts_list), dtype=np.float32)
    query_emb_m3 = np.asarray(embed_m3.embed(query_texts), dtype=np.float32)
    print(f"  bge-m3 done: chunks={chunk_emb_m3.shape}")

    # Release bge-m3 from VRAM before loading Qwen for concept extraction
    del embed_m3
    import jgraphrag.embed as _embed_mod
    _embed_mod._BGEM3 = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"  bge-m3 released. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 3. Embed with bge-large-en-v1.5 (CPU — no VRAM contention)
    print(f"\n[3/5] Embedding with bge-large-en-v1.5 (CPU)...")
    embed_large = BgeLargeProvider(device="cpu")
    chunk_emb_large = embed_large.embed(chunk_texts_list)
    query_emb_large = embed_large.embed(query_texts)
    print(f"  bge-large done: chunks={chunk_emb_large.shape}")
    del embed_large; gc.collect()

    # 4. J-GraphRAG concept extraction (Qwen already loaded, embeddings released)
    print(f"\n[4/5] J-GraphRAG concept extraction + graph building...")
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    raw_concepts = {}
    for i, cid in enumerate(chunk_ids):
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer,
                                           chunk_text_map[cid], n_words=5)
        raw_concepts[cid] = concepts
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(chunk_ids)}", flush=True)

    build_texts = chunk_texts_list
    graph = ConceptGraph()
    for cid in chunk_ids:
        graph.add_chunk(cid, raw_concepts[cid])
    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), build_texts, len(chunk_ids))
    rebuild_graph_index(graph, filtered, meta["idf"])
    graph.compute_tf(chunk_text_map)

    t_build = time.perf_counter() - t_start
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Build: {t_build:.1f}s, concepts={meta['n_after']}, VRAM={vram_peak:.2f}GB")

    # 5. Evaluation: 4 configs (2 embeds × 2 methods)
    print(f"\n[5/5] Evaluation (4 configs × {len(questions)} queries)...")
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    configs = {
        "B0_bge_m3": (chunk_emb_m3, query_emb_m3, False),
        "B0_bge_large": (chunk_emb_large, query_emb_large, False),
        "JGR_bge_m3": (chunk_emb_m3, query_emb_m3, True),
        "JGR_bge_large": (chunk_emb_large, query_emb_large, True),
    }

    def _eval_one(q_idx, q_data, chunk_emb, query_emb_matrix, use_graph):
        q = q_data
        level = q["level"]
        question = q["question"]
        gold_answer = q.get("answer", "")
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]

        q_vec = query_emb_matrix[q_idx]

        # Retrieve
        b0_hits = cosine_topk_ids(q_vec, chunk_emb, chunk_ids, TOP_K)
        if use_graph:
            seed_ids = [cid for cid, _ in cosine_topk_ids(q_vec, chunk_emb, chunk_ids, 10)]
            merged = build_and_propagate(graph, seed_ids)
            if len(merged) < TOP_K:
                for cid, _ in b0_hits:
                    if cid not in merged: merged.append(cid)
                    if len(merged) >= TOP_K: break
        else:
            merged = [cid for cid, _ in b0_hits]

        context = " ".join(chunk_text_map[cid] for cid in merged[:TOP_K])

        # Evaluate
        llm_local = DeepSeekProvider()
        answer = generate_answer(question, context, llm_local)
        acc = judge_answer_correctness(question, answer, gold_answer, llm_local)
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)

        return {"level": level, "acc": acc, "recall": recall}

    results = {}
    for config_name, (chunk_emb, query_emb_vec, use_graph) in configs.items():
        print(f"\n  Running {config_name}...", flush=True)
        config_results = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(_eval_one, i, q, chunk_emb, query_emb_vec, use_graph): i
                for i, q in enumerate(questions) if q.get("answer")
            }
            done = 0
            for future in as_completed(futures):
                config_results.append(future.result())
                done += 1
                if done % 10 == 0:
                    print(f"    {done}/{len(questions)}", flush=True)

        acc_vals = [1.0 if r["acc"] else 0.0 for r in config_results]
        recall_vals = [r["recall"] for r in config_results]
        by_level_acc = defaultdict(list)
        for r in config_results:
            by_level_acc[r["level"]].append(1.0 if r["acc"] else 0.0)

        results[config_name] = {
            "acc": float(np.mean(acc_vals)) if acc_vals else 0,
            "recall": float(np.mean(recall_vals)) if recall_vals else 0,
            "acc_by_level": {lv: (float(np.mean(by_level_acc[lv]))
                                  if by_level_acc.get(lv) else None)
                             for lv in ["L1","L2","L3","L4"]},
            "n": len(acc_vals),
        }
        print(f"    ACC={results[config_name]['acc']:.1%} "
              f"recall={results[config_name]['recall']:.1%}")

    # 5. Summary + leaderboard conversion
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS ({domain})")
    print(f"{'='*70}")
    print(f"  {'config':<20} {'ACC':>8} {'recall':>8}")
    print(f"  {'-'*38}")
    for name, r in results.items():
        print(f"  {name:<20} {r['acc']:>7.1%} {r['recall']:>7.1%}")

    # Leaderboard conversion
    # GraphRAG-Bench medical leaderboard: RAG w/o rerank = 61.0% ACC (with GPT-4o-mini + bge-large-en-v1.5)
    LEADERBOARD_RAG_ACC = 61.0
    our_b0_large_acc = results["B0_bge_large"]["acc"]

    if our_b0_large_acc > 0:
        # Conversion factor: how much our setup differs from leaderboard
        # If our B0_bge_large < leaderboard_RAG, our Qwen is weaker than GPT-4o-mini
        # Conversion = leaderboard / ours
        conversion = LEADERBOARD_RAG_ACC / (our_b0_large_acc * 100)

        print(f"\n  Leaderboard conversion:")
        print(f"    Leaderboard RAG w/o rerank: {LEADERBOARD_RAG_ACC}% (GPT-4o-mini + bge-large)")
        print(f"    Our B0 bge-large:          {our_b0_large_acc:.1%} (Qwen-7B + bge-large)")
        print(f"    Conversion factor:         {conversion:.2f}x")
        print(f"    (factor >1 means our setup is weaker than leaderboard)")

        # Estimate J-GraphRAG's leaderboard-equivalent score
        jgr_large_acc = results["JGR_bge_large"]["acc"]
        jgr_m3_acc = results["JGR_bge_m3"]["acc"]

        # Method 1: direct conversion from bge-large J-GraphRAG
        jgr_estimated_1 = jgr_large_acc * 100 * conversion

        # Method 2: relative improvement over B0 × leaderboard RAG
        if our_b0_large_acc > 0:
            jgr_relative = jgr_large_acc / our_b0_large_acc
            jgr_estimated_2 = LEADERBOARD_RAG_ACC * jgr_relative

        print(f"\n  J-GraphRAG leaderboard estimation:")
        print(f"    Our JGR bge-large ACC:     {jgr_large_acc:.1%}")
        print(f"    Est. method 1 (×factor):   {jgr_estimated_1:.1f}%")
        print(f"    Est. method 2 (relative):  {jgr_estimated_2:.1f}%")
        print(f"\n  For reference (leaderboard medical):")
        print(f"    G-reasoner:        73.3%")
        print(f"    HippoRAG2:         64.9%")
        print(f"    LightRAG:          62.6%")
        print(f"    RAG w/o rerank:    61.0%")
        print(f"    MS-GraphRAG local: 45.2%")

    # Save
    out = {
        "method": "bge_large_leaderboard_comparison",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(questions),
        "efficiency": {
            "build_time_s": round(t_build, 1),
            "build_time_per_chunk_s": round(t_build / len(chunk_ids), 4),
            "vram_peak_gb": round(vram_peak, 2),
        },
        "results": results,
        "leaderboard_conversion": {
            "leaderboard_rag_acc": LEADERBOARD_RAG_ACC,
            "our_b0_large_acc": round(our_b0_large_acc, 4),
            "conversion_factor": round(conversion, 2) if our_b0_large_acc > 0 else None,
            "jgr_estimated_acc_method1": round(jgr_estimated_1, 1) if our_b0_large_acc > 0 else None,
            "jgr_estimated_acc_method2": round(jgr_estimated_2, 1) if our_b0_large_acc > 0 else None,
        },
    }
    out_path = EXP / f"phase26b_bge_large_comparison_{domain}.json"
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

    print(f"\n[2/2] Running bge-large comparison...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_comparison(lens, lens_model, tokenizer, domain="medical",
                   max_queries=28, max_chunks=200)


if __name__ == "__main__":
    main()
