"""Phase 4-dig v3 — GraphRAG-Bench + LLM judge evidence recall。

GraphRAG-Bench 的 corpus 是整本书（大库），克服 distractor 配置的方法论瓶颈。
用 LLM judge 评估 evidence recall（subagent 评判检索到的 context 是否覆盖 ground truth evidence）。

4 个难度级别：
  L1 Fact Retrieval → 预期 D0≈D1（全局余弦已足够）
  L2 Complex Reasoning → 预期 D1>D0（跨文档关联）
  L3 Contextual Summarize → 预期 D1>D0
  L4 Creative Generation → 预期 D1>D0

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase4_dig_graphragbench --domain novel --max-queries 200
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.llm import DeepSeekProvider
from experiments.embed_cache import CachedBgeM3Provider
from experiments.dig_baselines import FlatDig, ConceptTreeDig

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
BENCH_DIR = Path("/tmp/graphrag-bench")

CHUNK_SIZE = 1200    # 字符（和 LightRAG 一致）
CHUNK_OVERLAP = 100  # 重叠字符

# 难度级别映射
LEVEL_MAP = {
    "Fact Retrieval": "L1",
    "Complex Reasoning": "L2",
    "Contextual Summarize": "L3",
    "Creative Generation": "L4",
}


# ── Corpus 加载 + Chunking ──────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把长文本切成重叠的段落级 chunks。"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:  # 过滤太短的
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_graphrag_bench(domain: str = "novel", max_queries: int = 200):
    """加载 GraphRAG-Bench 数据，返回 (corpus_chunks, queries, question_levels, evidences)。"""
    # Corpus
    corpus_path = BENCH_DIR / f"{domain}.json"
    with open(corpus_path) as f:
        raw_corpus = json.load(f)

    # 切分整本书为 chunks
    corpus_chunks = {}  # {chunk_id: text}
    if isinstance(raw_corpus, list):
        for book in raw_corpus:
            book_name = book.get("corpus_name", "unknown")
            text = book.get("context", "")
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                cid = f"{book_name}::chunk_{i}"
                corpus_chunks[cid] = chunk
    elif isinstance(raw_corpus, dict):
        book_name = raw_corpus.get("corpus_name", domain)
        text = raw_corpus.get("context", "")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            cid = f"{book_name}::chunk_{i}"
            corpus_chunks[cid] = chunk

    # Questions
    q_path = BENCH_DIR / f"{domain}_questions.json"
    with open(q_path) as f:
        raw_questions = json.load(f)

    # 过滤 + 子采样（按难度级别均衡）
    questions = []
    for q in raw_questions:
        qt = q.get("question_type", "")
        level = LEVEL_MAP.get(qt, "L1")
        questions.append({
            "id": q["id"],
            "question": q["question"],
            "answer": q.get("answer", ""),
            "level": level,
            "evidence": q.get("evidence", ""),
            "source": q.get("source", ""),
        })

    # 子采样：每级别尽量均匀
    by_level = {}
    for q in questions:
        by_level.setdefault(q["level"], []).append(q)

    sampled = []
    per_level = max_queries // len(by_level) if by_level else max_queries
    for level, qs in sorted(by_level.items()):
        sampled.extend(qs[:per_level])

    return corpus_chunks, sampled


# ── LLM Judge Evidence Recall ───────────────────────────────────────────

def split_evidence(evidence_str: str) -> list[str]:
    """把分号分隔的 evidence 分成原子声明。"""
    statements = [s.strip() for s in evidence_str.split(";") if s.strip()]
    return statements


def llm_judge_evidence_recall(query: str, context: str, evidence_statements: list[str],
                               llm: DeepSeekProvider) -> float:
    """用 LLM judge 评估 context 是否覆盖了所有 evidence statements。

    对每个 evidence statement，LLM 判断它能否从 context 推导出来。
    Score = 覆盖的 statements 数 / 总 statements 数。
    """
    if not evidence_statements:
        return 0.0

    # 截断 context 到 ~8000 字符（控制 LLM 输入）
    context_truncated = context[:8000]

    # 批量判断：一次 LLM 调用判断所有 statements
    statements_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(evidence_statements))
    prompt = (
        f"You are evaluating whether a retrieved context supports specific evidence statements.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved Context:\n{context_truncated}\n\n"
        f"Evidence Statements to verify:\n{statements_text}\n\n"
        f"For each statement, answer YES if it can be inferred from the context, NO if not.\n"
        f"Format: one line per statement, e.g. '1. YES' or '2. NO'."
    )

    msg = llm.complete(prompt, max_tokens=256)
    if msg.is_error:
        return 0.0

    # 解析 YES/NO
    response = msg.content
    covered = 0
    for i, _ in enumerate(evidence_statements):
        pattern = rf"{i+1}\.\s*(YES|NO)"
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.group(1).upper() == "YES":
            covered += 1

    return covered / len(evidence_statements)


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_experiment(domain: str, max_queries: int, embed, llm):
    print(f"\nPhase 4-dig v3: GraphRAG-Bench ({domain})")
    print(f"Loading corpus + chunking...")
    corpus_chunks, questions = load_graphrag_bench(domain, max_queries)
    print(f"  {len(corpus_chunks)} chunks, {len(questions)} questions")

    # 按级别统计
    from collections import Counter
    level_dist = Counter(q["level"] for q in questions)
    print(f"  level distribution: {dict(level_dist)}")

    # 嵌入 chunks
    chunk_ids = list(corpus_chunks.keys())
    chunk_texts = [corpus_chunks[cid] for cid in chunk_ids]
    print(f"  embedding {len(chunk_ids)} chunks...", end="", flush=True)
    chunk_emb = np.asarray(embed.embed(chunk_texts), dtype=np.float64)
    print(f" done")

    # 嵌入 queries
    query_texts = [q["question"] for q in questions]
    query_ids = [q["id"] for q in questions]
    query_emb = np.asarray(embed.embed(query_texts), dtype=np.float64)

    # 跑 D0 和 D1
    baselines = {
        "D0-flat": FlatDig(),
        "D1-concept-tree": ConceptTreeDig(recall_k=50, expand_weight=1.0),
    }

    all_results = {}
    for bname, baseline in baselines.items():
        print(f"  {bname}: fit + search...", end="", flush=True)
        baseline.fit(chunk_emb, chunk_ids)
        results = baseline.search(query_emb, query_ids, top_k=10)
        all_results[bname] = results
        print(f" done")

    # LLM judge evidence recall
    print(f"  LLM judge evidence recall...", end="", flush=True)
    evidence_recall = {bname: {} for bname in baselines}
    evidence_recall_by_level = {bname: {} for bname in baselines}

    for qi, q in enumerate(questions):
        qid = q["id"]
        evidence_stmts = split_evidence(q["evidence"])
        if not evidence_stmts:
            continue

        for bname in baselines:
            # 取 top-5 chunks 作为 context
            top_chunks = list(all_results[bname].get(qid, {}).keys())[:5]
            context = "\n\n".join(corpus_chunks.get(cid, "") for cid in top_chunks)

            er = llm_judge_evidence_recall(q["question"], context, evidence_stmts, llm)
            evidence_recall[bname][qid] = er

            level = q["level"]
            if level not in evidence_recall_by_level[bname]:
                evidence_recall_by_level[bname][level] = []
            evidence_recall_by_level[bname][level].append(er)

        if (qi + 1) % 50 == 0:
            print(f" {qi+1}", end="", flush=True)
    print(" done")

    # 汇总
    print(f"\n{'='*60}")
    print(f"=== Evidence Recall by Level ({domain}) ===")
    print(f"{'Level':<8} {'n_queries':<10} {'D0-flat':>12} {'D1-concept':>12} {'Delta':>10}")

    results_by_level = {}
    all_levels = sorted(set(q["level"] for q in questions))
    for level in all_levels:
        d0_scores = evidence_recall_by_level["D0-flat"].get(level, [])
        d1_scores = evidence_recall_by_level["D1-concept-tree"].get(level, [])
        d0_mean = np.mean(d0_scores) if d0_scores else 0
        d1_mean = np.mean(d1_scores) if d1_scores else 0
        delta = d1_mean - d0_mean
        n = max(len(d0_scores), len(d1_scores))
        print(f"{level:<8} {n:<10} {d0_mean:>12.4f} {d1_mean:>12.4f} {delta:>+10.4f}")
        results_by_level[level] = {
            "n": n, "d0": d0_mean, "d1": d1_mean, "delta": delta
        }

    # 总体
    d0_all = [v for v in evidence_recall["D0-flat"].values()]
    d1_all = [v for v in evidence_recall["D1-concept-tree"].values()]
    d0_overall = np.mean(d0_all) if d0_all else 0
    d1_overall = np.mean(d1_all) if d1_all else 0
    print(f"{'ALL':<8} {len(d0_all):<10} {d0_overall:>12.4f} {d1_overall:>12.4f} {d1_overall-d0_overall:>+10.4f}")

    result = {
        "domain": domain,
        "n_chunks": len(corpus_chunks),
        "n_questions": len(questions),
        "level_distribution": dict(level_dist),
        "evidence_recall_by_level": results_by_level,
        "evidence_recall_overall": {"D0": d0_overall, "D1": d1_overall, "delta": d1_overall - d0_overall},
    }

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 4-dig v3: GraphRAG-Bench")
    ap.add_argument("--domain", choices=["novel", "medical", "both"], default="both")
    ap.add_argument("--max-queries", type=int, default=200)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    llm = DeepSeekProvider()

    results = {"benchmarks": {}}
    domains = ["novel", "medical"] if args.domain == "both" else [args.domain]

    for domain in domains:
        results["benchmarks"][domain] = run_experiment(domain, args.max_queries, embed, llm)

    out_path = EXP / "phase4_dig_graphragbench.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
