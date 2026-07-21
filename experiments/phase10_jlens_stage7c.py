"""Phase 10 Stage 7c — J-Lens 概念图构建 + 图传播检索。

核心洞察（用户提出）：J-Lens 的价值不在直接检索增强（Stage 7a/7b 已证伪），
而在**图构建**。概念词作为节点，文档-概念关系作为边 → 概念图。图一旦建好，
GraphRAG 式的传播检索（query → seed docs → concept neighbors → more docs）
就有可能超越纯余弦检索。

管线：
  1. [建图] 每个 chunk → J-Lens 读出概念词（文档级长 prompt，残差稳定）
     → 二部图：chunk ↔ concept（membership 边）
     → concept-concept 边（共现权重）
  2. [检索] query bge-m3 → B0 top-K seed chunks
     → 从 seed 找到它们的概念 → 概念的其他 chunks（1-hop 图传播）
     → 合并 seed + propagated，重排
  3. [评估] evidence recall vs B0

关键区别 vs Stage 7a（聚类 rerank）：
  - Stage 7a 用簇 ID（粗粒度，二元 boost）→ 失败
  - Stage 7c 用概念词节点（细粒度，多概念交叉）→ 文档可通过多个概念路径被发现

关键区别 vs Stage 7b（单 query 概念提取）：
  - Stage 7b 在短 query 上提取概念（残差不稳定）→ 1.4% 失败
  - Stage 7c 在长 chunk 上提取概念（残差稳定，Stage 5: 80%）→ 概念图质量高
  - query 只通过 bge-m3 进入图（不需要 J-Lens 提取 query 概念）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage7c --domain medical
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.phase4_dig_graphragbench import (
    load_graphrag_bench, llm_judge_evidence_recall,
)
from experiments.concept_quality import optimize_concepts
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10  # final context size for evidence recall


# ── Concept extraction (document-level, stable) ────────────────────────

STOP_CONCEPTS = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "function", "def", "return", "class", "import",
    "concept", "concepts", "key", "main", "topic", "also", "they", "them",
    "than", "then", "when", "what", "each", "more", "most", "some", "such",
    "only", "very", "just", "like", "question", "list", "following",
    "above", "based", "study", "studies", "result", "results", "method",
    "methods", "patient", "patients", "group", "groups", "treatment",
    "associated", "compared", "significantly", "clinical", "using",
    "data", "analysis", "research", "health", "disease", "medical",
    # filter generic medical-paper boilerplate that appears in every chunk
}


def extract_chunk_concepts(lens, lens_model, tokenizer, chunk_text: str,
                           n_words: int = 5) -> list[str]:
    """Extract concept words from a document chunk (long prompt → stable residual).

    Uses the same concern-coupled pattern as Stage 5 (which achieved 80%
    accuracy on NFCorpus clusters). The chunk text provides enough context
    for the model to form stable concept representations, avoiding the
    short-prompt instability that sank Stage 7b (1.4%).
    """
    user_msg = (
        f"What concepts does this text discuss? List {n_words} one-word concepts.\n\n"
        f"{chunk_text[:800]}"
    )
    prefill = "The concepts discussed are"
    prompt = f"{user_msg}\n{prefill}"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    last = lens.source_layers[-1]
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[last],
        positions=[-1], max_seq_len=512,
    )

    probs = torch.softmax(lens_logits[last][0].float(), dim=-1)
    topk = probs.topk(30)  # over-extract, then filter

    words = []
    seen = set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_CONCEPTS
                and low not in seen):
            # reject mixed-case BPE fragments (like Stage 5 filter)
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── Concept graph ──────────────────────────────────────────────────────

class ConceptGraph:
    """Bipartite concept-document graph + concept-concept co-occurrence.

    Nodes: chunks (documents) + concepts (words)
    Edges:
      - chunk ↔ concept (membership: concept extracted from chunk)
      - concept ↔ concept (co-occurrence: appear in same chunk, weighted)
    """

    def __init__(self):
        self.chunk_concepts: dict[str, list[str]] = {}  # chunk_id → [concept words]
        self.concept_chunks: dict[str, list[str]] = defaultdict(list)  # concept → [chunk_ids]
        self.concept_cooccur: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.tf: dict[tuple[str, str], int] = {}  # (chunk_id, concept) → term frequency in chunk text
        self.chunk_lens: dict[str, int] = {}  # chunk_id → text length (for BM25 length norm)

    def add_chunk(self, chunk_id: str, concepts: list[str]):
        self.chunk_concepts[chunk_id] = concepts
        for c in concepts:
            self.concept_chunks[c].append(chunk_id)
        # co-occurrence
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                self.concept_cooccur[c1][c2] += 1
                self.concept_cooccur[c2][c1] += 1

    def compute_tf(self, chunk_texts: dict[str, str]):
        """Compute BM25 term frequencies: how many times each concept word
        actually appears in each chunk's source text.

        This upgrades the graph from binary membership (J-Lens extracted
        concept = edge exists) to weighted edges (concept appears N times =
        stronger edge). A chunk that mentions "cancer" 5 times is more
        strongly connected to the "cancer" concept than one that mentions
        it once.
        """
        import re
        total_len = 0
        for cid, text in chunk_texts.items():
            self.chunk_lens[cid] = len(text.split())
            total_len += self.chunk_lens[cid]
            text_lower = text.lower()
            for concept in self.chunk_concepts.get(cid, []):
                # count occurrences of the concept word in the chunk text
                count = len(re.findall(r'\b' + re.escape(concept.lower()) + r'\b', text_lower))
                self.tf[(cid, concept)] = max(count, 1)  # at least 1 (J-Lens found it)
        self.avg_chunk_len = total_len / max(len(chunk_texts), 1)

    def _bm25_tf_norm(self, chunk_id: str, tf: int, k1: float = 1.2, b: float = 0.75) -> float:
        """BM25 term frequency saturation + length normalization."""
        dl = self.chunk_lens.get(chunk_id, self.avg_chunk_len)
        norm = dl / max(self.avg_chunk_len, 1)
        return tf * (k1 + 1) / (tf + k1 * (1 - b + b * norm))

    def compute_idf(self):
        """IDF for each concept: log(N / df). High-DF artifacts get ~0 weight."""
        import math
        N = len(self.chunk_concepts)
        self.idf = {}
        for concept, chunks in self.concept_chunks.items():
            df = len(chunks)
            self.idf[concept] = math.log((N + 1) / (df + 1)) + 1  # smoothed IDF

    def filter_by_df(self, max_df_ratio: float = 0.3, min_df: int = 2):
        """Remove concept nodes that appear in too many chunks (artifacts) or
        too few (noise).

        max_df_ratio: concepts appearing in >this fraction of chunks are removed.
            0.3 = remove concepts in >30% of chunks (kills alink@84%, summarized@81%)
        min_df: concepts appearing in <this many chunks are removed (isolated noise).
        """
        N = len(self.chunk_concepts)
        max_df = int(N * max_df_ratio)
        removed = []
        for concept in list(self.concept_chunks.keys()):
            df = len(self.concept_chunks[concept])
            if df > max_df or df < min_df:
                # remove this concept from all data structures
                removed.append((concept, df))
                del self.concept_chunks[concept]
                if concept in self.concept_cooccur:
                    for other in list(self.concept_cooccur[concept].keys()):
                        del self.concept_cooccur[other][concept]
                    del self.concept_cooccur[concept]
                # remove from chunk_concepts
                for cid in self.chunk_concepts:
                    if concept in self.chunk_concepts[cid]:
                        self.chunk_concepts[cid] = [c for c in self.chunk_concepts[cid]
                                                     if c != concept]
        self.compute_idf()
        return removed

    def propagate(self, seed_chunk_ids: list[str], max_propagate: int = 50,
                  use_idf: bool = True, use_bm25: bool = True) -> list[str]:
        """Graph propagation: seed chunks → their concepts → other chunks.

        Returns a list of propagated chunk_ids (excluding seeds), ordered by
        BM25-weighted concept-path score.

        Edge weight = IDF(concept) × BM25_tf(concept, chunk)
        - IDF: downweights high-DF artifacts that connect to everything
        - BM25 tf: upweights chunks where the concept appears frequently
          (a chunk mentioning "cancer" 5x is a stronger match than 1x)
        """
        if not hasattr(self, 'idf'):
            self.compute_idf()

        # collect concepts from seed chunks
        seed_concepts = set()
        for cid in seed_chunk_ids:
            seed_concepts.update(self.chunk_concepts.get(cid, []))

        # find other chunks sharing these concepts, weighted by IDF × BM25 tf
        chunk_score = defaultdict(float)
        for concept in seed_concepts:
            idf_weight = self.idf.get(concept, 1.0) if use_idf else 1.0
            for cid in self.concept_chunks.get(concept, []):
                if cid not in seed_chunk_ids:
                    if use_bm25 and hasattr(self, 'tf') and (cid, concept) in self.tf:
                        tf = self.tf[(cid, concept)]
                        tf_weight = self._bm25_tf_norm(cid, tf)
                    else:
                        tf_weight = 1.0
                    chunk_score[cid] += idf_weight * tf_weight

        return [cid for cid, _ in sorted(chunk_score.items(),
                                          key=lambda x: x[1], reverse=True)[:max_propagate]]

    def graph_stats(self) -> dict:
        return {
            "n_chunks": len(self.chunk_concepts),
            "n_concepts": len(self.concept_chunks),
            "avg_concepts_per_chunk": np.mean([len(v) for v in self.chunk_concepts.values()]) if self.chunk_concepts else 0,
            "avg_chunks_per_concept": np.mean([len(v) for v in self.concept_chunks.values()]) if self.concept_chunks else 0,
            "n_cooccur_edges": sum(len(v) for v in self.concept_cooccur.values()) // 2,
        }


# ── Retrieval methods ──────────────────────────────────────────────────

def cosine_topk(query_vec, corpus_mat, top_k):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    c = corpus_mat / (np.linalg.norm(corpus_mat, axis=1, keepdims=True) + 1e-8)
    scores = c @ q
    idx = np.argsort(scores)[::-1][:top_k]
    return [(idx, float(scores[idx])) for _ in zip(idx, idx)]  # placeholder
    # actually return indices + scores


def cosine_topk_ids(query_vec, corpus_mat, ids, top_k):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    c = corpus_mat / (np.linalg.norm(corpus_mat, axis=1, keepdims=True) + 1e-8)
    scores = c @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in top_idx]


# ── Main ───────────────────────────────────────────────────────────────

def run_stage7c(embed, lens, lens_model, tokenizer, llm, domain="medical",
                max_queries=50, concept_k=5, seed_k=10, propagate_k=20,
                max_chunks=None):
    print(f"Phase 10 Stage 7c: Concept graph + propagation retrieval ({domain})")
    print(f"{'='*70}")

    # 1. Load corpus
    corpus_chunks, questions = load_graphrag_bench(domain, max_queries)
    chunk_ids = list(corpus_chunks.keys())
    chunk_texts = [corpus_chunks[cid] for cid in chunk_ids]
    print(f"\n  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Embed chunks + queries (bge-m3)
    print(f"  embedding {len(chunk_ids)} chunks...", end="", flush=True)
    chunk_emb = np.asarray(embed.embed(chunk_texts), dtype=np.float32)
    query_texts = [q["question"] for q in questions]
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float32)
    print(f" done")

    # 3. Build concept graph (J-Lens extraction per chunk)
    # For large corpora (novel: 4391 chunks), subsample to cap J-Lens runtime.
    # The graph covers the subsample; retrieval still uses full corpus via bge-m3.
    build_ids = chunk_ids
    build_texts = chunk_texts
    if max_chunks and len(chunk_ids) > max_chunks:
        # Evenly sample across all chunks to preserve book diversity
        step = len(chunk_ids) / max_chunks
        idxs = [int(i * step) for i in range(max_chunks)]
        build_ids = [chunk_ids[i] for i in idxs]
        build_texts = [chunk_texts[i] for i in idxs]
        print(f"  subsampling to {len(build_ids)} chunks (of {len(chunk_ids)}) for graph build")
    print(f"  building concept graph (J-Lens extraction for {len(build_ids)} chunks)...")
    graph = ConceptGraph()
    for i, (cid, text) in enumerate(zip(build_ids, build_texts)):
        concepts = extract_chunk_concepts(lens, lens_model, tokenizer, text, concept_k)
        graph.add_chunk(cid, concepts)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(build_ids)} chunks processed", flush=True)

    stats = graph.graph_stats()
    print(f"  graph: {stats}")
    # show sample concepts
    print(f"\n  Sample chunk concepts:")
    for cid in build_ids[:3]:
        print(f"    {cid}: {graph.chunk_concepts.get(cid, [])}")
    # most common concepts
    common = Counter()
    for cids in graph.concept_chunks.values():
        pass
    concept_freq = sorted(graph.concept_chunks.items(),
                          key=lambda x: len(x[1]), reverse=True)
    print(f"\n  Most connected concepts (by chunk count, BEFORE filtering):")
    for concept, chunks in concept_freq[:10]:
        print(f"    {concept}: {len(chunks)} chunks")

    # 3b. Optimized concept filtering: adaptive DF + corpus-verification + BPE completion
    print(f"\n  Optimized concept filtering (adaptive DF + corpus verify + BPE completion)...")
    # Prepare chunk_texts for corpus-verification (use the build subset)
    build_texts_for_filter = [chunk_texts[chunk_ids.index(cid)]
                              for cid in build_ids if cid in chunk_ids]
    optimized_chunks, opt_meta = optimize_concepts(
        dict(graph.concept_chunks), len(build_ids),
        chunk_texts=build_texts_for_filter)
    print(f"  adaptive knee: {opt_meta['max_df_ratio']:.2f} (max_df={opt_meta['max_df']})")
    print(f"  removed {len(opt_meta['removed_artifacts'])} artifacts + {opt_meta['removed_noise']} noise")
    for concept, df in opt_meta['removed_artifacts'][:8]:
        print(f"    artifact: {concept} (was in {df} chunks)")
    if opt_meta.get('bpe_completions'):
        print(f"  BPE completions ({len(opt_meta['bpe_completions'])}):")
        for old, new in list(opt_meta['bpe_completions'].items())[:8]:
            print(f"    {old} → {new}")

    # Rebuild graph with optimized concepts
    graph.concept_chunks = optimized_chunks
    graph.idf = opt_meta['idf']
    # Rebuild chunk_concepts from optimized concept_chunks
    graph.chunk_concepts = defaultdict(list)
    for concept, cids in optimized_chunks.items():
        for cid in cids:
            if concept not in graph.chunk_concepts[cid]:
                graph.chunk_concepts[cid].append(concept)

    stats_filtered = graph.graph_stats()
    print(f"  optimized graph: {stats_filtered}")
    concept_freq_f = sorted(graph.concept_chunks.items(),
                            key=lambda x: len(x[1]), reverse=True)
    print(f"  Top concepts AFTER optimization:")
    for concept, chunks in concept_freq_f[:10]:
        print(f"    {concept}: {len(chunks)} chunks (IDF={graph.idf.get(concept,0):.2f})")

    # 3c. Compute BM25 term frequencies for edge weighting
    # Build chunk_id → text mapping for chunks in the graph
    chunk_text_map = {cid: chunk_texts[chunk_ids.index(cid)]
                      for cid in build_ids if cid in chunk_ids}
    print(f"  computing BM25 term frequencies for {len(chunk_text_map)} chunks...")
    graph.compute_tf(chunk_text_map)
    print(f"  avg chunk length: {graph.avg_chunk_len:.0f} words")

    # 4. Retrieval: B0 vs graph-propagation (BM25-weighted)
    # Note: previous run established raw graph (unfiltered) = 67.4% (94% of B0).
    # This run tests the user's hypothesis: DF filtering + IDF weighting fixes
    # the artifact problem and lets graph propagation beat B0.
    print(f"\n  Running retrieval (local, fast)...")
    # Phase 1: collect all retrieval contexts (local cosine + graph propagation, no LLM)
    eval_tasks = []  # [(method, level, question, context, evidence), ...]
    for i, q in enumerate(questions):
        level = q["level"]
        evidence = q.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not evidence:
            continue

        # B0: pure cosine top-K
        b0_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, TOP_K)
        b0_context = " ".join(corpus_chunks[cid] for cid, _ in b0_hits)
        eval_tasks.append(("B0", level, q["question"], b0_context, evidence))

        # Graph-filtered: B0 seed → propagate on DF-filtered + IDF-weighted graph
        seed_hits = cosine_topk_ids(query_emb[i], chunk_emb, chunk_ids, seed_k)
        seed_ids = [cid for cid, _ in seed_hits]
        propagated_f = graph.propagate(seed_ids, max_propagate=propagate_k,
                                        use_idf=True, use_bm25=True)
        merged_ids_f = seed_ids[:seed_k]
        remaining_f = TOP_K - len(merged_ids_f)
        for pid in propagated_f[:remaining_f]:
            if pid not in merged_ids_f:
                merged_ids_f.append(pid)
        if len(merged_ids_f) < TOP_K:
            for cid, _ in b0_hits:
                if cid not in merged_ids_f:
                    merged_ids_f.append(cid)
                if len(merged_ids_f) >= TOP_K:
                    break
        graph_f_context = " ".join(corpus_chunks[cid] for cid in merged_ids_f)
        eval_tasks.append(("graph_filtered", level, q["question"], graph_f_context, evidence))

    # Phase 2: concurrent LLM judge (DeepSeek concurrency = 2500, use 8 workers)
    print(f"  Concurrent LLM judge ({len(eval_tasks)} calls, 8 workers)...")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    def _judge_one(task):
        method, level, question, context, evidence = task
        llm_local = DeepSeekProvider()  # thread-local provider
        recall = llm_judge_evidence_recall(question, context, evidence, llm_local)
        return method, level, recall

    results = {"B0": [], "graph_filtered": []}
    by_level = {"B0": defaultdict(list), "graph_filtered": defaultdict(list)}
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_judge_one, t): t for t in eval_tasks}
        for future in as_completed(futures):
            method, level, recall = future.result()
            results[method].append(recall)
            by_level[method][level].append(recall)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(eval_tasks)} judged", flush=True)

    # 5. Summary
    print(f"\n{'='*70}")
    print(f"RESULTS ({domain}, {len(results['B0'])} questions)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<18} {'Overall':>8} {'L1':>8} {'L2':>8} {'L3':>8} {'L4':>8}")
    print(f"  {'-'*58}")
    for method in ("B0", "graph_filtered"):
        overall = np.mean(results[method])
        levels = []
        for level in ("L1", "L2", "L3", "L4"):
            vals = by_level[method].get(level, [])
            levels.append(f"{np.mean(vals):.3f}" if vals else "  -  ")
        print(f"  {method:<18} {overall:>8.3f} {levels[0]:>8} {levels[1]:>8} {levels[2]:>8} {levels[3]:>8}")

    b0_overall = np.mean(results["B0"])
    gf_overall = np.mean(results["graph_filtered"])
    pct = gf_overall / b0_overall * 100 if b0_overall > 0 else 0
    delta = gf_overall - b0_overall
    print(f"\n  graph_filtered as % of B0: {pct:.0f}% (Δ{delta:+.3f})")
    print(f"  (previous run's raw graph = 94% of B0, Δ-0.040)")

    out = {
        "method": "concept_graph_propagation",
        "domain": domain,
        "graph_stats": stats,
        "graph_stats_filtered": stats_filtered,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results["B0"]),
        "params": {"concept_k": concept_k, "seed_k": seed_k, "propagate_k": propagate_k},
        "optimization": opt_meta,
        "overall": {m: float(np.mean(results[m])) for m in results},
        "by_level": {m: {l: float(np.mean(v)) if v else 0
                         for l, v in by_level[m].items()} for m in by_level},
        "top_concepts_filtered": [(c, len(cs), float(graph.idf.get(c, 0)))
                                   for c, cs in concept_freq_f[:20]],
    }
    out_path = EXP / f"phase10_stage7c_concept_graph_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 7c: concept graph")
    ap.add_argument("--domain", default="medical", choices=["medical", "novel"])
    ap.add_argument("--max-queries", type=int, default=50)
    ap.add_argument("--concept-k", type=int, default=5)
    ap.add_argument("--seed-k", type=int, default=10)
    ap.add_argument("--propagate-k", type=int, default=20)
    ap.add_argument("--max-chunks", type=int, default=None,
                    help="cap chunks for J-Lens graph build (subsample evenly)")
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

    print(f"\n[3/3] Building concept graph + propagation retrieval...")
    embed = CachedBgeM3Provider()
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    run_stage7c(embed, lens, lens_model, tokenizer, llm, args.domain,
                args.max_queries, args.concept_k, args.seed_k, args.propagate_k,
                args.max_chunks)


if __name__ == "__main__":
    main()
