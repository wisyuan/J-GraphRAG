"""Phase 12: Reverse pipeline — J-Lens residual clustering vs bge-m3 for
concept graph retrieval.

Stage 6 proved J-Lens residual clustering has better silhouette (+38%) than
bge-m3. Stage 7c proved concept-graph propagation ≈ or > plain RAG when
clustered with bge-m3. This phase tests: does clustering with J-Lens
residuals (instead of bge-m3) produce a better concept graph?

Two pipelines, same docs, same queries (eliminates sampling variance):
  A. bge-m3 clustering → J-Lens concept words → concept graph → propagation
     (current Stage 7c approach)
  B. J-Lens residual clustering → J-Lens concept words → concept graph → propagation
     (reverse pipeline — concept space throughout)

Both use the same retrieval (bge-m3 cosine seed + graph propagation) and the
same evidence recall evaluation. The ONLY difference is what feature space
HDBSCAN clusters in.

If B > A → J-Lens residual clustering produces better concept communities
than bge-m3 → the reverse pipeline is validated for retrieval, not just
clustering geometry.

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase12_reverse_pipeline
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.concept_quality import optimize_concepts
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals
from experiments.phase10_jlens_stage7c import (
    ConceptGraph, extract_chunk_concepts, cosine_topk_ids,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10


def build_and_eval_graph(name: str,
                         chunk_seed_vecs: np.ndarray, query_seed_vecs: np.ndarray,
                         chunk_ids: list[str], chunk_texts: list[str],
                         chunk_emb: np.ndarray, query_emb: np.ndarray,
                         questions: list,
                         lens, lens_model, tokenizer, llm,
                         seed_k: int = 10, propagate_k: int = 20):
    """Build concept graph, evaluate retrieval.

    chunk_seed_vecs: vectors for chunk seed search (bge-m3 or J-Lens residual)
    query_seed_vecs: vectors for query seed search (same space as chunk_seed_vecs)
    chunk_emb: bge-m3 chunk embeddings (for fallback fill)
    query_emb: bge-m3 query embeddings (for fallback fill)
    """
    print(f"\n  --- Pipeline {name}: building concept graph ---")

    # The concept graph is identical for both pipelines (chunk-level J-Lens concepts).
    # The ONLY difference is the seed search vector space (bge-m3 vs J-Lens residual).

    # Build concept graph (same as Stage 7c)
    print(f"    building concept graph (J-Lens chunk concepts)...")
    graph = ConceptGraph()
    for i, (cid, text) in enumerate(zip(chunk_ids, chunk_texts)):
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer, text)
        graph.add_chunk(cid, concepts)

    # Optimize concepts (DF filter + BPE completion)
    build_texts_map = {cid: chunk_texts[i] for i, cid in enumerate(chunk_ids)}
    optimized, opt_meta = optimize_concepts(
        dict(graph.concept_chunks), len(chunk_ids),
        chunk_texts=[build_texts_map[c] for c in chunk_ids])
    graph.concept_chunks = optimized
    graph.idf = opt_meta.get("idf", {})
    graph.chunk_concepts = defaultdict(list)
    for concept, cids in optimized.items():
        for cid in cids:
            if concept not in graph.chunk_concepts[cid]:
                graph.chunk_concepts[cid].append(concept)
    # BM25 tf
    graph.compute_tf(build_texts_map)
    print(f"    graph: {graph.graph_stats()}")

    # Retrieval: seed search in cluster_vecs space + graph propagation
    print(f"    retrieval + evidence recall...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    eval_tasks = []
    for i, q in enumerate(questions):
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not evidence:
            continue

        # Seed search in the pipeline's vector space
        seed_hits = cosine_topk_ids(query_seed_vecs[i], chunk_seed_vecs,
                                    chunk_ids, seed_k)
        seed_ids = [cid for cid, _ in seed_hits]

        # Graph propagation
        propagated = graph.propagate(seed_ids, max_propagate=propagate_k,
                                      use_idf=True, use_bm25=True)
        merged = seed_ids[:seed_k]
        remaining = TOP_K - len(merged)
        for pid in propagated[:remaining]:
            if pid not in merged:
                merged.append(pid)
        # fill from bge-m3 if not enough
        b0_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, TOP_K)
        if len(merged) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged:
                    merged.append(cid)
                if len(merged) >= TOP_K:
                    break

        context = " ".join(chunk_texts[chunk_ids.index(cid)] for cid in merged)
        eval_tasks.append((name, q["level"], q["question"], context, evidence))

    def _judge(task):
        method, level, question, context, evidence = task
        llm_local = DeepSeekProvider()
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)
        return method, level, recall

    results = []
    by_level = defaultdict(list)
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_judge, t): t for t in eval_tasks}
        for future in as_completed(futures):
            method, level, recall = future.result()
            results.append(recall)
            by_level[level].append(recall)
            done += 1
            if done % 20 == 0:
                print(f"      {done}/{len(eval_tasks)} judged", flush=True)

    overall = float(np.mean(results)) if results else 0
    print(f"    {name} overall: {overall:.3f}")
    return overall, dict(by_level), results


def run_phase12(embed, lens, lens_model, tokenizer, llm,
                domain="medical", max_queries=50):
    print(f"Phase 12: Reverse pipeline — J-Lens residual vs bge-m3 clustering ({domain})")
    print(f"{'='*70}")

    # 1. Load corpus
    corpus_chunks, questions = load_graphrag_bench(domain, max_queries)
    chunk_ids = list(corpus_chunks.keys())
    chunk_texts = [corpus_chunks[cid] for cid in chunk_ids]
    query_texts = [q["question"] for q in questions]
    query_ids = [q["id"] for q in questions]
    print(f"\n  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. bge-m3 embeddings (for both pipelines' retrieval + pipeline A clustering)
    print(f"  embedding chunks + queries (bge-m3)...")
    chunk_emb = np.asarray(embed.embed(chunk_texts), dtype=np.float32)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float32)

    # 3. J-Lens residuals (for pipeline B clustering + retrieval)
    print(f"  extracting J-Lens residuals for {len(chunk_ids)} chunks...")
    layer = lens.source_layers[-1]
    doc_prompts = [_build_topic_prompt(t, tokenizer) for t in chunk_texts]
    chunk_jlens = extract_residuals(lens, lens_model, tokenizer, doc_prompts, layer,
                                    max_seq_len=256)
    print(f"  extracting J-Lens residuals for {len(questions)} queries...")
    query_prompts = [_build_topic_prompt(q, tokenizer) for q in query_texts]
    query_jlens = extract_residuals(lens, lens_model, tokenizer, query_prompts, layer,
                                    max_seq_len=128)

    # 4. Pipeline A: bge-m3 seed search + concept graph propagation
    print(f"\n  Pipeline A: bge-m3 seed search + concept graph")
    a_overall, a_levels, a_results = build_and_eval_graph(
        "A_bge", chunk_emb, query_emb,
        chunk_ids, chunk_texts,
        chunk_emb, query_emb, questions,
        lens, lens_model, tokenizer, llm)

    # 5. Pipeline B: J-Lens residual seed search + concept graph propagation
    print(f"\n  Pipeline B: J-Lens residual seed search + concept graph")
    b_overall, b_levels, b_results = build_and_eval_graph(
        "B_jlens", chunk_jlens, query_jlens,
        chunk_ids, chunk_texts,
        chunk_emb, query_emb, questions,
        lens, lens_model, tokenizer, llm)

    # 6. Compare
    print(f"\n{'='*70}")
    print(f"RESULTS ({domain}, {len(a_results)} queries)")
    print(f"{'='*70}")
    print(f"\n  Pipeline A (bge-m3 seed):    {a_overall:.3f}")
    print(f"  Pipeline B (J-Lens seed):    {b_overall:.3f}")
    delta = b_overall - a_overall
    print(f"  B vs A: Δ{delta:+.3f} ({'B wins' if delta > 0 else 'A wins'})")

    print(f"\n  Per-level:")
    print(f"  {'Pipeline':<20} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    for name, levels in [("A (bge-m3)", a_levels), ("B (J-Lens)", b_levels)]:
        vals = [f"{np.mean(levels.get(l, [0])):.3f}" for l in ("L1", "L2", "L3", "L4")]
        print(f"  {name:<20} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8}")

    out = {
        "method": "reverse_pipeline_comparison",
        "domain": domain,
        "n_chunks": len(chunk_ids),
        "n_queries": len(a_results),
        "A_bge_seed": {"overall": a_overall, "by_level": {l: float(np.mean(v)) for l, v in a_levels.items()}},
        "B_jlens_seed": {"overall": b_overall, "by_level": {l: float(np.mean(v)) for l, v in b_levels.items()}},
    }
    out_path = EXP / f"phase12_reverse_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def _build_topic_prompt(text: str, tokenizer) -> str:
    """Concern prompt for residual extraction."""
    user_msg = f"What is the main topic? One word.\n\n{text[:500]}"
    prefill = "The main topic is"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="medical", choices=["medical", "novel"])
    ap.add_argument("--max-queries", type=int, default=50)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Wrapping + running reverse pipeline...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    embed = CachedBgeM3Provider()
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    run_phase12(embed, lens, lens_model, tokenizer, llm, args.domain, args.max_queries)


if __name__ == "__main__":
    main()
