"""Phase 4-dig — dig 场景的跨文档导航价值验证。

核心命题：概念树/聚类的价值在 dig（跨文档关联）而非 ssearch（精确召回）。
bridge 问题上 D1/D2 应胜 D0；comparison 问题上应持平。

benchmark：
  - HotpotQA-distractor（bridge + comparison 分层）
  - pi-code（code_trials + code_kg D2 KG 遍历）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase4_dig --repo /tmp/pi-repo --max-hotpot 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.dig_baselines import FlatDig, ConceptTreeDig, KGDig

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
K_VALUES = [1, 3, 5, 10, 100]
TOP_K = max(K_VALUES)


# ── HotpotQA 加载 ───────────────────────────────────────────────────────

def load_musique(max_queries=500):
    """加载 MuSiQue（2-4 hop 多跳 QA），转为 BEIR 格式。

    MuSiQue 的 paragraphs 列表里每个段落有 is_supporting: bool。
    is_supporting=True → qrels。比 HotpotQA 更难（2-4 hop）。
    """
    from datasets import load_dataset

    ds = load_dataset("bdsaglam/musique", split="validation")

    corpus = {}
    queries = {}
    qrels = {}
    query_types = {}  # 按 hop 数分层

    count = 0
    for entry in ds:
        if count >= max_queries:
            break

        qid = entry["id"]
        query = entry["question"]
        paragraphs = entry.get("paragraphs", [])

        supporting_titles = [p["title"] for p in paragraphs if p.get("is_supporting")]
        if not supporting_titles:
            continue

        queries[qid] = query
        qrels[qid] = {}
        hop_count = len(supporting_titles)
        query_types[qid] = f"{hop_count}_hop"

        for p in paragraphs:
            title = p["title"]
            text = p.get("paragraph_text", "")
            corpus[title] = {"text": text, "title": title}
            if p.get("is_supporting"):
                qrels[qid][title] = 1

        if qrels[qid]:
            count += 1

    return corpus, queries, qrels, query_types


def load_hotpotqa(max_queries=500):
    """加载 HotpotQA-distractor，转为 BEIR 格式。

    supporting_facts["title"] → qrels（二值相关）。
    返回 (corpus, queries, qrels, query_types)。
    query_types: {qid: "bridge"|"comparison"} 用于分层报告。
    """
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    corpus = {}
    queries = {}
    qrels = {}
    query_types = {}

    count = 0
    for entry in ds:
        if count >= max_queries:
            break

        qid = entry["id"]
        query = entry["question"]
        qtype = entry.get("type", "bridge")  # "bridge" or "comparison"

        # supporting_facts → qrels
        supporting = entry.get("supporting_facts", {})
        titles = supporting.get("title", [])
        if not titles:
            continue

        queries[qid] = query
        qrels[qid] = {}
        query_types[qid] = qtype

        # context → corpus
        context = entry.get("context", {})
        ctx_titles = context.get("title", [])
        ctx_sents = context.get("sentences", [])

        for i, title in enumerate(ctx_titles):
            cid = title  # 文档 id = title
            text = " ".join(ctx_sents[i]) if i < len(ctx_sents) else ""
            corpus[cid] = {"text": text, "title": title}
            if title in titles:
                qrels[qid][cid] = 1

        if qrels[qid]:
            count += 1

    return corpus, queries, qrels, query_types


# ── pi-code 加载（复用 phase4_retrieval）────────────────────────────────

def load_pi_code(repo_path, max_files=50):
    from experiments.fine_record_dispersion import split_file_to_symbols

    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]

    corpus = {}
    file_to_fine_ids = {}

    for f in files:
        rel = str(f.relative_to(repo))
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fid = f"{rel}::{sym['name']}"
            text = f"{sym['kind']} {sym['name']}:\n{sym['body']}"
            corpus[fid] = {"text": text, "title": sym["name"]}
            file_to_fine_ids.setdefault(rel, []).append(fid)

    trials_path = REPO / "experiments" / "m2" / "code_trials.json"
    trials = json.loads(trials_path.read_text())

    queries = {}
    qrels = {}
    for i, trial in enumerate(trials):
        qid = f"q{i}"
        queries[qid] = trial["query"]
        qrels[qid] = {}
        for gt_file in trial.get("ground_truth_files", []):
            for fid in file_to_fine_ids.get(gt_file, []):
                qrels[qid][fid] = 1

    queries = {qid: q for qid, q in queries.items() if qrels.get(qid)}
    qrels = {qid: rels for qid, rels in qrels.items() if rels}
    query_types = {qid: "code" for qid in queries}  # 代码域无 bridge/comparison 区分

    return corpus, queries, qrels, query_types


# ── 评估 ────────────────────────────────────────────────────────────────

def evaluate_dig_baseline(baseline, corpus_emb, corpus_ids, query_emb, query_ids, qrels):
    """评估单个 baseline，返回 BEIR 指标。"""
    from beir.retrieval.evaluation import EvaluateRetrieval

    baseline.fit(corpus_emb, corpus_ids)
    results = baseline.search(query_emb, query_ids, top_k=TOP_K)

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    return {
        "ndcg_10": ndcg.get("NDCG@10", 0),
        "recall_10": recall.get("Recall@10", 0),
        "recall_100": recall.get("Recall@100", 0),
        "mrr": ndcg.get("NDCG@1", 0),  # 近似 MRR
    }


def evaluate_stratified(baseline, corpus_emb, corpus_ids, query_emb, query_ids, qrels, query_types):
    """分层评估：bridge vs comparison vs code。"""
    from beir.retrieval.evaluation import EvaluateRetrieval

    baseline.fit(corpus_emb, corpus_ids)
    all_results = baseline.search(query_emb, query_ids, top_k=TOP_K)

    strata = {}
    for qtype in set(query_types.values()):
        qids_strata = [qid for qid in query_ids if query_types.get(qid) == qtype]
        qrels_strata = {qid: qrels[qid] for qid in qids_strata if qid in qrels}
        results_strata = {qid: all_results[qid] for qid in qids_strata if qid in all_results}

        if not qrels_strata:
            continue

        ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels_strata, results_strata, K_VALUES)
        strata[qtype] = {
            "n_queries": len(qids_strata),
            "ndcg_10": ndcg.get("NDCG@10", 0),
            "recall_10": recall.get("Recall@10", 0),
        }

    # 全量
    ndcg, _, recall, _ = EvaluateRetrieval.evaluate(qrels, all_results, K_VALUES)
    strata["_all"] = {
        "n_queries": len(query_ids),
        "ndcg_10": ndcg.get("NDCG@10", 0),
        "recall_10": recall.get("Recall@10", 0),
    }
    return strata


def paired_permutation_test_qrels(qrels, results_a, results_b, k=10, n_perm=1000):
    """Paired permutation test on per-query nDCG@k."""
    common = sorted(set(qrels) & set(results_a) & set(results_b))
    if len(common) < 5:
        return {"p_value": 1.0, "significant": False}

    def per_query_ndcg(qid, results):
        if qid not in results or qid not in qrels:
            return 0.0
        dcg = 0.0
        ranked = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]
        for rank, (cid, _) in enumerate(ranked, 1):
            rel = qrels[qid].get(cid, 0)
            dcg += rel / np.log2(rank + 1)
        ideal = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 1) for i, r in enumerate(ideal, 1))
        return dcg / idcg if idcg > 0 else 0.0

    diffs = np.array([per_query_ndcg(q, results_a) - per_query_ndcg(q, results_b) for q in common])
    obs = diffs.mean()
    if abs(obs) < 1e-8:
        return {"p_value": 1.0, "delta": 0.0, "significant": False, "n": len(common)}

    rng = np.random.default_rng(42)
    count = sum(1 for _ in range(n_perm) if abs((diffs * rng.choice([-1, 1], len(diffs))).mean()) >= abs(obs))
    p = count / n_perm
    return {"delta": float(obs), "p_value": float(p), "significant": p < 0.05, "n": len(common)}


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_benchmark(name, corpus, queries, qrels, query_types, embed, code_kg_path=None):
    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")

    corpus_ids = list(corpus.keys())
    query_ids = list(queries.keys())
    corpus_texts = [corpus[c].get("text", "") for c in corpus_ids]
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  {len(corpus_ids)} docs, {len(query_ids)} queries")
    print(f"  embedding corpus...", end="", flush=True)
    corpus_emb = np.asarray(embed.embed(corpus_texts), dtype=np.float64)
    print(f" done")
    print(f"  embedding queries...", end="", flush=True)
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float64)
    print(f" done")

    # query 类型分布
    type_dist = {}
    for qid in query_ids:
        qt = query_types.get(qid, "unknown")
        type_dist[qt] = type_dist.get(qt, 0) + 1
    print(f"  query types: {type_dist}")

    # baselines
    baselines = {
        "D0-flat": FlatDig(),
        "D1-concept-tree": ConceptTreeDig(),
    }
    if code_kg_path:
        baselines["D2-kg"] = KGDig()

    strata_results = {}
    raw_results_all = {}

    for bname, baseline in baselines.items():
        print(f"  {bname}:", end=" ", flush=True)
        strata = evaluate_stratified(baseline, corpus_emb, corpus_ids, query_emb, query_ids, qrels, query_types)
        strata_results[bname] = strata
        all_res = strata.get("_all", {})
        print(f"nDCG@10={all_res.get('ndcg_10', 0):.4f}, Recall@10={all_res.get('recall_10', 0):.4f}")

        # 保留 raw results for permutation test
        baseline.fit(corpus_emb, corpus_ids)
        raw_results_all[bname] = baseline.search(query_emb, query_ids, top_k=TOP_K)

    # permutation tests
    print(f"  permutation tests:")
    perms = {}
    pairs = [("D1-concept-tree", "D0-flat")]
    if code_kg_path:
        pairs.append(("D2-kg", "D0-flat"))

    for a, b in pairs:
        if a in raw_results_all and b in raw_results_all:
            perm = paired_permutation_test_qrels(qrels, raw_results_all[a], raw_results_all[b])
            perms[f"{a}_vs_{b}"] = perm
            sig = "✓" if perm["significant"] else "✗"
            print(f"    {a} vs {b}: Δ={perm['delta']:+.4f}, p={perm['p_value']:.4f} {sig}")

    return {
        "n_docs": len(corpus_ids),
        "n_queries": len(query_ids),
        "query_type_distribution": type_dist,
        "stratified": strata_results,
        "permutation": perms,
    }


def main():
    ap = argparse.ArgumentParser(description="Phase 4-dig: dig evaluation")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-pi-files", type=int, default=50)
    ap.add_argument("--max-hotpot", type=int, default=500)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    print("Phase 4-dig: Cross-document navigation evaluation")

    results = {"benchmarks": {}}

    # MuSiQue (primary, 2-4 hop, hardest)
    print("\nLoading MuSiQue...")
    corpus, queries, qrels, qtypes = load_musique(args.max_hotpot)
    print(f"  {len(corpus)} docs, {len(queries)} queries")
    results["benchmarks"]["musique"] = run_benchmark(
        "musique", corpus, queries, qrels, qtypes, embed
    )

    # HotpotQA (continuity)
    print("\nLoading HotpotQA...")
    corpus, queries, qrels, qtypes = load_hotpotqa(args.max_hotpot)
    print(f"  {len(corpus)} docs, {len(queries)} queries")
    results["benchmarks"]["hotpotqa"] = run_benchmark(
        "hotpotqa", corpus, queries, qrels, qtypes, embed
    )

    # pi-code + D2 KG
    print("\nLoading pi-code...")
    corpus, queries, qrels, qtypes = load_pi_code(args.repo, args.max_pi_files)
    code_kg_path = str(REPO / "experiments" / "m2" / "code_kg.json")
    results["benchmarks"]["pi-code"] = run_benchmark(
        "pi-code", corpus, queries, qrels, qtypes, embed, code_kg_path=code_kg_path
    )

    # 总结
    print(f"\n{'='*60}")
    print("=== SUMMARY (nDCG@10 by query type) ===")
    for name, res in results["benchmarks"].items():
        print(f"\n  {name}:")
        strata = res.get("stratified", {})
        for bname in ["D0-flat", "D1-concept-tree", "D2-kg"]:
            if bname not in strata:
                continue
            bs = strata[bname]
            for qtype in sorted(bs.keys()):
                s = bs[qtype]
                print(f"    {bname:20s} {qtype:15s}: nDCG@10={s['ndcg_10']:.4f}, "
                      f"Recall@10={s['recall_10']:.4f} ({s['n_queries']} queries)")

    out_path = EXP / "phase4_dig.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
