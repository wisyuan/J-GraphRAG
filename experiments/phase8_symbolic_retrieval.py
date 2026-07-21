"""Phase 8 — 符号概念集合检索（Symbolic Concept-Set Retrieval）。

回答一个 Phase 4 无法外推的新命题：
  ★ LLM 抽取的概念关键词集合 + 符号相似度匹配，
    能否在 ssearch 上接近 bge-m3 flat 余弦？

Phase 4 测的全是文档向量的线性变换（残差/马氏/级联子空间），全部撞上
"子空间余弦和全局余弦高度相关"的墙。本实验的匹配信号是关键词集合的符号
相似度——和文档向量的线性视图正交，是真正未测过的信号空间。

3 个 baseline（同一批关键词集合，单因子=匹配机制）：
  B0       (已有, phase4): bge-m3 文档向量全局余弦——基线
  S-jaccard: 加权 Jaccard (纯符号)——符号匹配的纯能力
  S-embed  : 关键词级 embedding 软匹配——embedding 软化的增量

3 个 benchmark（复用 phase4）：NFCorpus / SciFact / pi-code
指标：nDCG@10 (主), Recall@10, MRR + paired permutation test
权重源：weight_logprob (模型 logprob) — 优于 weight_llm 自报告

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.phase8_symbolic_retrieval --repo /tmp/pi-repo
    # 单域快速验证:
    python -m experiments.phase8_symbolic_retrieval --only pi-code --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.keyword_cache import CachedKeywordExtractor
from experiments.symbolic_baselines import WeightedJaccardSearch, EmbeddingSoftMatchSearch
# 复用 phase4 的所有基础设施
from experiments.phase4_retrieval import (
    REPO, EXP,
    load_beir_benchmark, load_pi_code_benchmark,
    per_query_ndcg, paired_permutation_test,
)

K_VALUES = [1, 3, 5, 10, 100, 1000]
TOP_K = max(K_VALUES)
WEIGHT_KEY = "weight_logprob"  # 用 logprob 权重（比 LLM 自报告更可靠）


def evaluate_symbolic(baseline, query_kw_sets, query_ids, corpus_kw_sets,
                      corpus_ids, qrels) -> dict:
    """评估一个符号 baseline：fit + search + BEIR 指标。

    复用 phase4 的 BEIR 指标计算，但 fit/search 接口吃关键词集不吃 embedding。
    """
    from beir.retrieval.evaluation import EvaluateRetrieval

    baseline.fit(corpus_kw_sets, corpus_ids)
    results = baseline.search(query_kw_sets, query_ids, top_k=TOP_K)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    mrr = EvaluateRetrieval.evaluate_custom(qrels, results, K_VALUES, "mrr")

    return {
        "ndcg_10": ndcg.get("NDCG@10", 0),
        "ndcg_1": ndcg.get("NDCG@1", 0),
        "ndcg_3": ndcg.get("NDCG@3", 0),
        "recall_10": recall.get("Recall@10", 0),
        "recall_100": recall.get("Recall@100", 0),
        "mrr": mrr.get("MRR@10", 0) if isinstance(mrr, dict) else mrr,
        "map_10": _map.get("MAP@10", 0),
    }


def run_symbolic_benchmark(name: str, corpus, queries, qrels,
                            kw_ext: CachedKeywordExtractor,
                            embed=None) -> dict:
    """对一个 benchmark 跑 B0 (从 phase4 读) + S-jaccard + S-embed。

    B0 不重跑——直接从 phase4_retrieval.json 读结果（避免重复 embedding）。
    S-jaccard / S-embed 需要关键词集，过 LLM 抽取。
    """
    from beir.retrieval.evaluation import EvaluateRetrieval
    from experiments.baselines import FlatSearch

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    corpus_texts = [corpus[cid].get("text", "") or corpus[cid].get("title", "")
                    for cid in corpus_ids]
    query_texts = [queries[qid] for qid in query_ids]
    print(f"  {len(corpus_ids)} docs, {len(query_ids)} queries")

    # ── 抽取关键词（docs + queries 都要，用不同 prompt）──
    print(f"  extracting doc keywords...", flush=True)
    corpus_kw = kw_ext.extract_keywords_batch(corpus_texts, desc=f"{name}-docs", mode="doc")
    print(f"  extracting query keywords...", flush=True)
    query_kw = kw_ext.extract_keywords_batch(query_texts, desc=f"{name}-queries", mode="query")

    # 统计
    n_doc_ok = sum(1 for r in corpus_kw if r)
    n_q_ok = sum(1 for r in query_kw if r)
    avg_kw = (sum(len(r) for r in corpus_kw) / max(n_doc_ok, 1))
    print(f"  keywords: {n_doc_ok}/{len(corpus_kw)} docs ok, "
          f"{n_q_ok}/{len(query_kw)} queries ok, avg {avg_kw:.1f} kw/doc")

    # ── S-jaccard ──
    print(f"  S-jaccard:", end=" ", flush=True)
    s_jac = WeightedJaccardSearch(weight_key=WEIGHT_KEY)
    m_jac = evaluate_symbolic(s_jac, query_kw, query_ids, corpus_kw, corpus_ids, qrels)
    print(f"nDCG@10={m_jac['ndcg_10']:.4f}, Recall@10={m_jac['recall_10']:.4f}, "
          f"MRR={m_jac['mrr']:.4f}")

    # ── S-embed (需要 bge-m3 嵌关键词) ──
    print(f"  S-embed:", end=" ", flush=True)
    if embed is None:
        embed = CachedBgeM3Provider()
    s_emb = EmbeddingSoftMatchSearch(embed.embed, tau=0.5, weight_key=WEIGHT_KEY)
    m_emb = evaluate_symbolic(s_emb, query_kw, query_ids, corpus_kw, corpus_ids, qrels)
    print(f"nDCG@10={m_emb['ndcg_10']:.4f}, Recall@10={m_emb['recall_10']:.4f}, "
          f"MRR={m_emb['mrr']:.4f}")

    # ── B0 (重跑 flat，需要 embedding) ──
    # 不从 phase4 json 读——因为 phase4 的 per-query results 没存（只存了均值）。
    # 为了做 permutation test 需要 per-query，所以这里重跑 B0。
    print(f"  B0 (rerun for per-query):", end=" ", flush=True)
    b0 = FlatSearch()
    corpus_emb = np.asarray(embed.embed(corpus_texts), dtype=np.float64)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float64)
    b0.fit(corpus_emb, corpus_ids)
    b0_raw = b0.search(query_emb, query_ids, top_k=TOP_K)
    ndcg_b0, _map_b0, recall_b0, _ = EvaluateRetrieval.evaluate(qrels, b0_raw, K_VALUES)
    mrr_b0 = EvaluateRetrieval.evaluate_custom(qrels, b0_raw, K_VALUES, "mrr")
    m_b0 = {
        "ndcg_10": ndcg_b0.get("NDCG@10", 0),
        "recall_10": recall_b0.get("Recall@10", 0),
        "recall_100": recall_b0.get("Recall@100", 0),
        "mrr": mrr_b0.get("MRR@10", 0) if isinstance(mrr_b0, dict) else mrr_b0,
        "map_10": _map_b0.get("MAP@10", 0),
    }
    print(f"nDCG@10={m_b0['ndcg_10']:.4f}, Recall@10={m_b0['recall_10']:.4f}, "
          f"MRR={m_b0['mrr']:.4f}")

    # ── per-query nDCG@10 + permutation test ──
    print(f"  permutation tests...", end=" ", flush=True)
    per_query = {
        "B0": per_query_ndcg(b0_raw, qrels, k=10),
        "S-jaccard": per_query_ndcg(s_jac.search(query_kw, query_ids, top_k=TOP_K), qrels, k=10),
        "S-embed": per_query_ndcg(s_emb.search(query_kw, query_ids, top_k=TOP_K), qrels, k=10),
    }
    perm = {
        "S-jaccard_vs_B0": paired_permutation_test(per_query["S-jaccard"], per_query["B0"]),
        "S-embed_vs_B0": paired_permutation_test(per_query["S-embed"], per_query["B0"]),
        "S-embed_vs_S-jaccard": paired_permutation_test(per_query["S-embed"], per_query["S-jaccard"]),
    }
    for pair, res in perm.items():
        sig = "✓" if res["significant"] else "✗"
        print(f"\n    {pair}: Δ={res['delta']:+.4f}, p={res['p_value']:.4f} {sig}", end="")
    print()

    return {
        "n_docs": len(corpus_ids),
        "n_queries": len(query_ids),
        "n_doc_kw_ok": n_doc_ok,
        "n_query_kw_ok": n_q_ok,
        "avg_kw_per_doc": round(avg_kw, 2),
        "weight_key": WEIGHT_KEY,
        "results": {"B0": m_b0, "S-jaccard": m_jac, "S-embed": m_emb},
        "permutation": perm,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 8: Symbolic concept-set retrieval")
    ap.add_argument("--repo", default="/tmp/pi-repo", help="pi repo path for pi-code")
    ap.add_argument("--max-pi-files", type=int, default=50)
    ap.add_argument("--only", default=None,
                    help="run only this benchmark (nfcorpus/scifact/pi-code)")
    args = ap.parse_args()

    kw_ext = CachedKeywordExtractor()
    embed = CachedBgeM3Provider()
    print("Phase 8 Symbolic Retrieval: S-jaccard + S-embed vs B0")
    print(f"  weight source: {WEIGHT_KEY} (logprob-derived)")
    print(f"  keyword cache: {kw_ext._cache_file}")

    results = {
        "weight_key": WEIGHT_KEY,
        "prompt_version": "v2",  # doc/query prompt split (v1 had intent/content mismatch)
        "benchmarks": {},
    }

    benchmarks = []
    if args.only:
        benchmarks = [args.only]
    else:
        benchmarks = ["nfcorpus", "scifact", "pi-code"]

    for name in benchmarks:
        if name in ("nfcorpus", "scifact"):
            corpus, queries, qrels = load_beir_benchmark(name)
        elif name == "pi-code":
            corpus, queries, qrels, _, _ = load_pi_code_benchmark(
                args.repo, args.max_pi_files, embed)
        else:
            print(f"  unknown benchmark: {name}")
            continue

        results["benchmarks"][name] = run_symbolic_benchmark(
            name, corpus, queries, qrels, kw_ext, embed
        )

    # ── 总结 ──
    print(f"\n{'='*60}")
    print("=== SUMMARY (nDCG@10) ===")
    header = (f"{'Benchmark':<12} {'B0':>8} {'S-jaccard':>11} {'S-embed':>9} "
              f"{'S-j vs B0':>11} {'S-e vs B0':>11} {'S-e vs S-j':>11}")
    print(header)
    for name, res in results["benchmarks"].items():
        r = res["results"]
        pj = res["permutation"]["S-jaccard_vs_B0"]
        pe = res["permutation"]["S-embed_vs_B0"]
        pej = res["permutation"]["S-embed_vs_S-jaccard"]
        sj = "✓" if pj["significant"] else "✗"
        se = "✓" if pe["significant"] else "✗"
        sej = "✓" if pej["significant"] else "✗"
        print(f"{name:<12} {r['B0']['ndcg_10']:>8.4f} {r['S-jaccard']['ndcg_10']:>11.4f} "
              f"{r['S-embed']['ndcg_10']:>9.4f} {pj['delta']:+.4f}{sj:>2} "
              f"{pe['delta']:+.4f}{se:>2} {pej['delta']:+.4f}{sej:>2}")

    # 保存
    out_path = EXP / "phase8_symbolic_retrieval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
