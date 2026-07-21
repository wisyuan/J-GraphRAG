"""Phase 31: J-AugRAG vs RAG 增强方案对比。

四方法对比：
  1. B0: 纯 bge-m3 余弦 top-K
  2. B0+reranker: bge-m3 召回 top-50 → bge-reranker-v2-m3 重排 → top-K
  3. J-AugRAG (flat): bge-m3 seed → 概念图传播
  4. J-AugRAG (relation): bge-m3 seed → 概念图传播 + 关系扩展

VRAM 管理：reranker 先跑（GPU），释放后再加载 Qwen 做关系提取。
概念提取从缓存加载（如果存在）。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase31_rag_comparison
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage7c import ConceptGraph, cosine_topk_ids
from experiments.phase25_filter_bpe_benchmark import phase25_filter_with_bpe, rebuild_graph_index
from experiments.phase28_relation_graph import relation_propagate, RELATION_TYPES
from experiments.phase27_relation_readout import build_relation_prompt, decode_topk
from experiments.phase4_dig_graphragbench import load_graphrag_bench, llm_judge_evidence_recall
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
TOP_K = 10
RECALL_K = 50
CACHE_DIR = EXP / "concept_cache"


def load_concept_cache(domain: str) -> dict | None:
    """Load cached concept extraction results if available."""
    path = CACHE_DIR / f"concept_cache_{domain}_full.json"
    if path.exists():
        print(f"  Loading concept cache from {path}")
        return json.loads(path.read_text())
    return None


def build_graph_from_cache(cache: dict, chunk_text_map: dict[str, str],
                           chunk_ids: list[str]) -> tuple[ConceptGraph, dict]:
    """Build concept graph from cached data (no J-Lens forward pass needed)."""
    raw_concepts = cache.get("chunks", {})
    build_texts = [chunk_text_map.get(cid, "") for cid in chunk_ids]

    graph = ConceptGraph()
    for cid in chunk_ids:
        # Cache may use different chunk_id format; match by index
        concepts = raw_concepts.get(cid, [])
        if not concepts:
            # Try positional fallback
            pass
        graph.add_chunk(cid, concepts)

    filtered, meta = phase25_filter_with_bpe(
        dict(graph.concept_chunks), build_texts, len(chunk_ids))
    rebuild_graph_index(graph, filtered, meta["idf"])
    graph.compute_tf(chunk_text_map)
    return graph, meta


def rerank_query(query: str, candidates: list[str], reranker, top_k: int) -> list[int]:
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def run_comparison(embed, domain="medical", max_queries=28, max_chunks=200):
    print(f"Phase 31: J-AugRAG vs RAG enhancement comparison")
    print(f"  domain={domain}")
    print(f"{'='*70}")

    # 1. Load corpus + embed
    print(f"\n[1/6] Loading corpus + embedding...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    chunk_emb = np.asarray(embed.embed(chunk_texts_list), dtype=np.float32)
    query_emb = np.asarray(embed.embed([q["question"] for q in questions]), dtype=np.float32)
    query_texts = [q["question"] for q in questions]
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # Release bge-m3 before loading reranker
    del embed
    import jgraphrag.embed as _em; _em._BGEM3 = None
    gc.collect(); torch.cuda.empty_cache()

    # 2. Reranker (GPU, before Qwen)
    print(f"\n[2/6] B0 + bge-reranker-v2-m3 (GPU)...")
    from FlagEmbedding import FlagReranker
    reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

    rerank_results = {}
    rerank_times = []
    for qi in range(len(questions)):
        b0_cands = cosine_topk_ids(query_emb[qi], chunk_emb, chunk_ids, RECALL_K)
        cand_texts = [chunk_text_map[cid] for cid, _ in b0_cands]
        t0 = time.perf_counter()
        ranked = rerank_query(query_texts[qi], cand_texts, reranker, TOP_K)
        t1 = time.perf_counter()
        rerank_times.append(t1 - t0)
        rerank_results[qi] = [b0_cands[i][0] for i in ranked]
    avg_rerank_time = np.mean(rerank_times)
    print(f"  Reranker: {avg_rerank_time:.3f}s/query")

    del reranker; gc.collect(); torch.cuda.empty_cache()
    print(f"  Reranker released. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # 3. Build concept graph from cache (or live extraction)
    print(f"\n[3/6] Building concept graph...")
    cache = load_concept_cache(domain)

    if cache:
        graph, meta = build_graph_from_cache(cache, chunk_text_map, chunk_ids)
        t_concept = 0  # cached, no extraction time
        print(f"  From cache: {meta['n_after']} concepts")
    else:
        # Live extraction (loads Qwen)
        print(f"  Cache not found, loading Qwen for live extraction...")
        from experiments.phase10_jlens_stage1 import detect_model, load_model, load_lens
        from experiments.phase10_jlens_stage7c import extract_chunk_concepts
        cand = detect_model()
        lens = load_lens(cand["local_lens_path"])
        model, tokenizer = load_model(cand["local_model_dir"], use_4bit=cand["needs_4bit"])
        import jlens; lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

        t_start = time.perf_counter()
        raw_concepts = {}
        for i, cid in enumerate(chunk_ids):
            raw_concepts[cid] = extract_chunk_concepts(
                lens, lens_model, tokenizer, chunk_text_map[cid], n_words=5)
            if (i+1) % 50 == 0: print(f"    {i+1}/{len(chunk_ids)}", flush=True)
        graph = ConceptGraph()
        for cid in chunk_ids: graph.add_chunk(cid, raw_concepts[cid])
        filtered, meta = phase25_filter_with_bpe(
            dict(graph.concept_chunks), chunk_texts_list, len(chunk_ids))
        rebuild_graph_index(graph, filtered, meta["idf"])
        graph.compute_tf(chunk_text_map)
        t_concept = time.perf_counter() - t_start
        print(f"  Live extraction: {t_concept:.1f}s, {meta['n_after']} concepts")

    # 4. Relation extraction (needs Qwen if not cached)
    print(f"\n[4/6] Relation extraction...")
    relation_adj = {}

    # Check if Qwen is already loaded (from live extraction above)
    qwen_loaded = 'lens_model' in dir() or 'lens_model' in locals()

    if not qwen_loaded:
        print(f"  Loading Qwen for relation extraction...")
        from experiments.phase10_jlens_stage1 import detect_model, load_model, load_lens
        cand = detect_model()
        lens = load_lens(cand["local_lens_path"])
        model, tokenizer = load_model(cand["local_model_dir"], use_4bit=cand["needs_4bit"])
        import jlens; lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    layer = lens.source_layers[-1]
    cooccur = graph.concept_cooccur
    pairs = [(a, b) for a, nbrs in cooccur.items()
             for b, c in nbrs.items() if c >= 2 and a < b]
    print(f"  Co-occurrence pairs: {len(pairs)}")

    relations = []
    t_rel_start = time.perf_counter()
    for c_a, c_b in pairs:
        chunks_a = set(graph.concept_chunks.get(c_a, []))
        chunks_b = set(graph.concept_chunks.get(c_b, []))
        common = chunks_a & chunks_b
        doc_text = chunk_text_map.get(list(common)[0], "") if common else ""
        if not doc_text: continue

        prompt = build_relation_prompt(doc_text, c_a, c_b, tokenizer)
        lens_logits, _, _ = lens.apply(
            lens_model, prompt, layers=[layer],
            positions=[-1], max_seq_len=512)
        words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)
        for w in words:
            if w["token"].lower() in RELATION_TYPES:
                relations.append({"a": c_a, "b": c_b, "rel": w["token"].lower()})
                relation_adj.setdefault(c_a, []).append((c_b, w["token"].lower()))
                relation_adj.setdefault(c_b, []).append((c_a, w["token"].lower()))
                break
    t_rel = time.perf_counter() - t_rel_start
    print(f"  Relations: {len(relations)} ({t_rel:.1f}s)")

    # Release Qwen before LLM judge phase
    if 'model' in dir() or 'model' in locals():
        del model, tokenizer, lens, lens_model
    gc.collect(); torch.cuda.empty_cache()

    # 5. Evaluation (4 methods)
    print(f"\n[5/6] Evaluation (4 methods × {len(questions)} queries)...")
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi, q):
        level = q["level"]
        question = q["question"]
        gold = q.get("answer", "")
        evidence = q.get("evidence", [])
        if isinstance(evidence, str): evidence = [evidence]
        q_vec = query_emb[qi]

        # B0
        b0_hits = cosine_topk_ids(q_vec, chunk_emb, chunk_ids, TOP_K)
        b0_ctx = " ".join(chunk_text_map[cid] for cid, _ in b0_hits)

        # B0+reranker
        rr_ids = rerank_results.get(qi, [cid for cid,_ in b0_hits])
        rr_ctx = " ".join(chunk_text_map[cid] for cid in rr_ids[:TOP_K])

        # J-AugRAG flat
        seed_ids = [cid for cid,_ in cosine_topk_ids(q_vec, chunk_emb, chunk_ids, 10)]
        flat_prop = graph.propagate(seed_ids, max_propagate=20, use_idf=True, use_bm25=True)
        merged_f = list(seed_ids[:10])
        for p in flat_prop:
            if p not in merged_f: merged_f.append(p)
            if len(merged_f) >= TOP_K: break
        flat_ctx = " ".join(chunk_text_map[cid] for cid in merged_f[:TOP_K])

        # J-AugRAG relation
        rel_prop, n_exp = relation_propagate(graph, relation_adj, seed_ids, max_propagate=20)
        merged_r = list(seed_ids[:10])
        for p,_ in rel_prop:
            if p not in merged_r: merged_r.append(p)
            if len(merged_r) >= TOP_K: break
        if len(merged_r) < TOP_K:
            for cid,_ in b0_hits:
                if cid not in merged_r: merged_r.append(cid)
                if len(merged_r) >= TOP_K: break
        rel_ctx = " ".join(chunk_text_map[cid] for cid in merged_r[:TOP_K])

        llm = DeepSeekProvider()
        ctxs = {"B0": b0_ctx, "B0_reranker": rr_ctx,
                "JAugRAG_flat": flat_ctx, "JAugRAG_relation": rel_ctx}
        res = {"level": level, "n_expanded": n_exp}
        for m, ctx in ctxs.items():
            ans = generate_answer(question, ctx, llm)
            res[m] = {"acc": judge_answer_correctness(question, ans, gold, llm),
                      "recall": llm_judge_evidence_recall(question, ctx, evidence, llm)}
        return res

    methods = ["B0", "B0_reranker", "JAugRAG_flat", "JAugRAG_relation"]
    results_list = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_eval, i, q): i
                   for i, q in enumerate(questions) if q.get("answer")}
        done = 0
        for f in as_completed(futures):
            results_list.append(f.result()); done += 1
            if done % 10 == 0: print(f"    {done}/{len(questions)}", flush=True)

    # 6. Summary
    print(f"\n[6/6] Summary\n{'='*70}")
    acc = {m: [] for m in methods}
    recall = {m: [] for m in methods}
    by_lvl = {m: defaultdict(list) for m in methods}
    for r in results_list:
        for m in methods:
            acc[m].append(1.0 if r[m]["acc"] else 0.0)
            recall[m].append(r[m]["recall"])
            by_lvl[m][r["level"]].append(1.0 if r[m]["acc"] else 0.0)

    b0_mean = np.mean(acc["B0"]) if acc["B0"] else 1
    print(f"\n  {'method':<22} {'ACC':>8} {'recall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*70}")
    for m in methods:
        a = np.mean(acc[m]) if acc[m] else 0
        r = np.mean(recall[m]) if recall[m] else 0
        lvls = [f"{np.mean(by_lvl[m].get(lv,[])):>7.1%}" if by_lvl[m].get(lv) else f"{'N/A':>7}"
                for lv in ["L1","L2","L3","L4"]]
        pct = f" ({a/b0_mean:.0%})" if b0_mean > 0 and m != "B0" else ""
        print(f"  {m:<22} {a:>7.1%}{pct} {r:>7.1%} {lvls[0]} {lvls[1]} {lvls[2]} {lvls[3]}")

    t_total_build = t_concept + t_rel
    print(f"\n  Efficiency:")
    print(f"    B0:              0s build + 0s/query")
    print(f"    B0+reranker:     0s build + {avg_rerank_time:.3f}s/query")
    print(f"    J-AugRAG:        {t_total_build:.1f}s build + ~0s/query")
    if t_concept == 0:
        print(f"    (concept extraction from cache, relation: {t_rel:.1f}s)")

    out = {
        "method": "rag_enhancement_comparison",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "efficiency": {
            "concept_build_s": round(t_concept, 1),
            "relation_build_s": round(t_rel, 1),
            "reranker_per_query_s": round(avg_rerank_time, 4),
            "n_concepts": meta["n_after"],
            "n_relations": len(relations),
        },
        "results": {m: {
            "acc": float(np.mean(acc[m])) if acc[m] else 0,
            "recall": float(np.mean(recall[m])) if recall[m] else 0,
            "acc_by_level": {lv: (float(np.mean(by_lvl[m][lv]))
                                  if by_lvl[m].get(lv) else None) for lv in ["L1","L2","L3","L4"]},
            "n": len(acc[m]),
        } for m in methods},
    }
    out_path = EXP / f"phase31_rag_comparison_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medical")
    ap.add_argument("--max-queries", type=int, default=28)
    ap.add_argument("--max-chunks", type=int, default=200)
    args = ap.parse_args()

    print("[1/2] Loading bge-m3...")
    embed = CachedBgeM3Provider()
    print(f"\n[2/2] Running comparison...")
    run_comparison(embed, domain=args.domain,
                   max_queries=args.max_queries, max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
