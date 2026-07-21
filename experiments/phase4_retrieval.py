"""Phase 4 — 检索价值评估（核心 3 baseline × 3 benchmark）。

回答 Q4 的核心命题：
  ★ 命题1：B-fractal vs B0 → 嵌入分形理论核心方法能否 beat 普通 RAG？
    命题2：B0+ vs B0 → 残差展开单独的检索价值？
    命题3：B-fractal vs B0+ → 理论路径 vs 简化路径？

3 个 benchmark：NFCorpus（医学）、SciFact（科学）、pi-code（TypeScript）
指标：nDCG@10（主）、Recall@10、MRR（BEIR 标准实现）
统计检验：paired permutation test

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase4_retrieval --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.baselines import FlatSearch, FractalSearch, ResidualSearch

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 3, 5, 10, 100, 1000]
TOP_K = max(K_VALUES)


# ── Benchmark 加载 ──────────────────────────────────────────────────────

def load_beir_benchmark(name: str, data_path: str = "/tmp/beir-datasets"):
    """加载 BEIR 数据集（NFCorpus/SciFact）。"""
    from beir.datasets.data_loader import GenericDataLoader

    folder = str(Path(data_path) / name)
    corpus, queries, qrels = GenericDataLoader(data_folder=folder).load(split="test")
    return corpus, queries, qrels


def load_pi_code_benchmark(repo_path: str, max_files: int = 50, embed=None):
    """加载 pi-code 检索 benchmark（从 code_trials.json 包装为 BEIR 格式）。

    FineRecord id = "{path}::{name}"，ground_truth 是文件路径。
    匹配规则：文件级 GT → 该文件的所有 FineRecord 都算相关。
    """
    from experiments.fine_record_dispersion import split_file_to_symbols

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    # 构建 corpus
    corpus = {}
    fine_texts = []
    fine_ids = []
    file_to_fine_ids: dict[str, list[str]] = {}  # 文件路径 → FineRecord id 列表

    for f in files:
        rel = str(f.relative_to(repo))
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fid = f"{rel}::{sym['name']}"
            text = f"{sym['kind']} {sym['name']}:\n{sym['body']}"
            corpus[fid] = {"text": text, "title": sym["name"]}
            fine_texts.append(text)
            fine_ids.append(fid)
            file_to_fine_ids.setdefault(rel, []).append(fid)

    # 加载 trials
    trials_path = REPO / "experiments" / "m2" / "code_trials.json"
    trials = json.loads(trials_path.read_text())

    # 构建 queries + qrels
    queries = {}
    qrels = {}
    for i, trial in enumerate(trials):
        qid = f"q{i}"
        queries[qid] = trial["query"]
        qrels[qid] = {}
        for gt_file in trial.get("ground_truth_files", []):
            # 文件级 GT → 该文件所有 FineRecord 相关
            for fid in file_to_fine_ids.get(gt_file, []):
                qrels[qid][fid] = 1  # 二值相关

    # 只保留有 GT 的 query
    queries = {qid: q for qid, q in queries.items() if qrels.get(qid)}
    qrels = {qid: rels for qid, rels in qrels.items() if rels}

    return corpus, queries, qrels, fine_texts, fine_ids


# ── 评估 ────────────────────────────────────────────────────────────────

def evaluate_baseline(baseline, corpus_emb, corpus_ids, query_emb, query_ids,
                      qrels) -> dict:
    """评估单个 baseline：fit + search + BEIR 指标。"""
    from beir.retrieval.evaluation import EvaluateRetrieval

    # fit
    baseline.fit(corpus_emb, corpus_ids)

    # search
    results = baseline.search(query_emb, query_ids, top_k=TOP_K)

    # evaluate
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


def per_query_ndcg(results: dict, qrels: dict, k: int = 10) -> dict[str, float]:
    """算每个 query 的 nDCG@k（用于 paired permutation test）。"""
    from beir.retrieval.evaluation import EvaluateRetrieval
    ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, results, [k])
    # BEIR 返回的是平均值，需要 per-query——手动算
    per_query = {}
    for qid in qrels:
        if qid not in results:
            per_query[qid] = 0.0
            continue
        # DCG
        dcg = 0.0
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        for rank, (cid, _) in enumerate(ranked, 1):
            rel = qrels[qid].get(cid, 0)
            dcg += rel / np.log2(rank + 1)
        # IDCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
        per_query[qid] = dcg / idcg if idcg > 0 else 0.0
    return per_query


def paired_permutation_test(scores_a: dict[str, float], scores_b: dict[str, float],
                             n_permutations: int = 1000) -> dict:
    """Paired permutation test：a 是否显著优于 b？"""
    common_qids = sorted(set(scores_a) & set(scores_b))
    if len(common_qids) < 5:
        return {"p_value": 1.0, "significant": False, "n_queries": len(common_qids)}

    diffs = np.array([scores_a[q] - scores_b[q] for q in common_qids])
    observed_mean = diffs.mean()

    if abs(observed_mean) < 1e-8:
        return {"p_value": 1.0, "significant": False, "delta": 0.0, "n_queries": len(common_qids)}

    rng = np.random.default_rng(42)
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_mean = (diffs * signs).mean()
        if abs(perm_mean) >= abs(observed_mean):
            count += 1

    p_value = count / n_permutations
    return {
        "delta": float(observed_mean),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "n_queries": len(common_qids),
    }


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_benchmark(name: str, corpus, queries, qrels, embed,
                  extra_fine_texts=None, extra_fine_ids=None) -> dict:
    """对一个 benchmark 跑 3 个 baseline + 统计检验。"""
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    corpus_texts = [corpus[cid].get("text", "") or corpus[cid].get("title", "") for cid in corpus_ids]
    query_texts = [queries[qid] for qid in query_ids]

    # 嵌入（缓存）
    print(f"  {len(corpus_ids)} docs, {len(query_ids)} queries")
    print(f"  embedding corpus...", end="", flush=True)
    corpus_emb = np.asarray(embed.embed(corpus_texts), dtype=np.float64)
    print(f" done")
    print(f"  embedding queries...", end="", flush=True)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float64)
    print(f" done")

    # 跑 3 个 baseline
    baselines = {
        "B0": FlatSearch(),
        "B-fractal": FractalSearch(n_dims=50, top_fraction=0.2,
                                    activation_threshold=0.5, recall_k0=100,
                                    cascade_ks=[50, 20, 10], max_depth=3),
        "B0+": ResidualSearch(),
    }

    all_results_raw = {}  # 保留 raw results for per-query
    metrics = {}
    for bname, baseline in baselines.items():
        print(f"  {bname}:", end=" ", flush=True)
        m = evaluate_baseline(baseline, corpus_emb, corpus_ids, query_emb, query_ids, qrels)
        metrics[bname] = m
        print(f"nDCG@10={m['ndcg_10']:.4f}, Recall@10={m['recall_10']:.4f}, MRR={m['mrr']:.4f}")

    # per-query nDCG@10 for permutation test
    print(f"  permutation tests...", end=" ", flush=True)
    per_query = {}
    for bname, baseline in baselines.items():
        baseline.fit(corpus_emb, corpus_ids)
        raw = baseline.search(query_emb, query_ids, top_k=TOP_K)
        per_query[bname] = per_query_ndcg(raw, qrels, k=10)

    perm = {
        "B-fractal_vs_B0": paired_permutation_test(per_query["B-fractal"], per_query["B0"]),
        "B0+_vs_B0": paired_permutation_test(per_query["B0+"], per_query["B0"]),
        "B-fractal_vs_B0+": paired_permutation_test(per_query["B-fractal"], per_query["B0+"]),
    }
    for pair, res in perm.items():
        sig = "✓" if res["significant"] else "✗"
        print(f"\n    {pair}: Δ={res['delta']:+.4f}, p={res['p_value']:.4f} {sig}", end="")
    print()

    return {
        "n_docs": len(corpus_ids),
        "n_queries": len(query_ids),
        "results": metrics,
        "permutation": perm,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 4: Retrieval evaluation")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-pi-files", type=int, default=50)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    print("Phase 4 Retrieval: 3 baselines × 3 benchmarks")

    results = {"benchmarks": {}}

    # BEIR benchmarks
    for name in ["nfcorpus", "scifact"]:
        corpus, queries, qrels = load_beir_benchmark(name)
        results["benchmarks"][name] = run_benchmark(name, corpus, queries, qrels, embed)

    # pi-code
    corpus, queries, qrels, fine_texts, fine_ids = load_pi_code_benchmark(
        args.repo, args.max_pi_files, embed
    )
    results["benchmarks"]["pi-code"] = run_benchmark(
        "pi-code", corpus, queries, qrels, embed, fine_texts, fine_ids
    )

    # 总结
    print(f"\n{'='*60}")
    print("=== SUMMARY (nDCG@10) ===")
    print(f"{'Benchmark':<15} {'B0':>8} {'B-fractal':>10} {'B0+':>8} {'B-f vs B0':>12} {'B0+ vs B0':>12}")
    for name, res in results["benchmarks"].items():
        r = res["results"]
        p1 = res["permutation"]["B-fractal_vs_B0"]
        p2 = res["permutation"]["B0+_vs_B0"]
        sig1 = "✓" if p1["significant"] else "✗"
        sig2 = "✓" if p2["significant"] else "✗"
        print(f"{name:<15} {r['B0']['ndcg_10']:>8.4f} {r['B-fractal']['ndcg_10']:>10.4f} "
              f"{r['B0+']['ndcg_10']:>8.4f} {p1['delta']:+.4f}{sig1:>3} {p2['delta']:+.4f}{sig2:>3}")

    # 保存
    out_path = EXP / "phase4_retrieval.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
