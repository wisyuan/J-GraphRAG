"""Phase 55: novel L4 全量（Creative Generation 67 题）多跳增益确认。

背景：Phase 54 修复了 L4 评测协议（rubric judge 替代严格金标准匹配），并发现
L4 残余低分是生成侧忠实度问题（"concisely" + max_tokens=200 与创意任务不适配）。
本实验在 novel 全量 67 题 Creative Generation 上，用修正协议 + 生成侧改造重测
图增益——这是效果故事最薄弱环节（原 L4 仅 12 题）。

两臂（同题配对）：
  b0 = bge top-10（无图基线）
  ah = Phase 53 概念+实体图 dual-level + 交错合并（最终 LightRAG-J 配置，
       novel ee 边 = J-Lens 读出，需一次性加载 Qwen，读完即释放显存）

判决指标（预设，先于运行定义；n=67，1 题 ≈ 0.015）：
  主判决（配对 discordant pairs）：b = ah✓b0✗ 题数，c = ah✗b0✓ 题数
    b - c >= 5  → 多跳增益 SUPPORTED
    |b - c| <= 2 → 无增益（主张收缩为 "L4 无损害"）
    c - b >= 3  → FALSIFIED（图扩展在 L4 有害）
  次要①：coverage 维度 ah vs b0（图应提升证据覆盖，即使 ACC 打平）
  次要②：生成改造收益——与 Phase 54 同 12 题子集 b0 rubric ACC=0.333 对比

用法：
  python -m experiments.phase55_l4_scale --selftest   # mock，零 GPU 零 API
  python -m experiments.phase55_l4_scale --smoke      # 3 题全链路，人工核对
  python -m experiments.phase55_l4_scale              # 全量 67 题
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase4_dig_graphragbench import BENCH_DIR
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, TOP_K,
)
from experiments.phase50_lightrag_j import LightRagIndex, lightrag_retrieve
from experiments.phase53_textside_entities import (
    augment_index, jlens_readout_top_pairs, S2_READOUT_TOP_PAIRS, CACHE_DIR,
)
from experiments.phase54_l4_rubric_judge import rubric_judge

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase55_l4_scale.json"

# 主判决阈值（见模块 docstring）
GATE_SUPPORT = 5   # b - c >= 5 → SUPPORTED
GATE_NULL = 2      # |b - c| <= 2 → 无增益
# 次要②参照：Phase 54 S2 同 12 题子集 b0 rubric ACC
PHASE54_B0_SUBSET_ACC = 0.333

# 生成侧改造（Phase 54 S2 证据：去 concisely、任务适配、max_tokens 800）
GEN_PROMPT = """Based on the following context from the source text, complete the task faithfully.

Context:
{context}

Task: {question}

Requirements:
- Stay faithful to the facts, entities, and events in the context. Do NOT invent details that contradict or are unsupported by the source.
- Fully adopt the requested format and perspective (diary entry, news article, first-person narration, etc.).

Answer:"""
GEN_MAX_TOKENS = 800


def load_l4_questions() -> list[dict]:
    """novel 全量 Creative Generation 题（67 题）。"""
    qs = json.load(open(BENCH_DIR / "novel_questions.json"))
    return [q for q in qs if q.get("question_type") == "Creative Generation"]


def generate_answer_faithful(question: str, context: str, llm) -> str:
    """生成侧改造版：任务适配 prompt + max_tokens=800（phase26 旧版是 concisely/200）。
    v4 推理模型可能静默返回空（推理链吃掉 budget），空答案重试一次。"""
    prompt = GEN_PROMPT.format(context=context[:12000], question=question)
    for _ in range(2):
        msg = llm.complete(prompt, max_tokens=GEN_MAX_TOKENS)
        text = msg.content if hasattr(msg, "content") else str(msg)
        if text.strip():
            return text
    return text


def verdict_from_pairs(rows: list[dict]) -> dict:
    """配对 discordant pairs 判决。rows: [{b0: 0/1, ah: 0/1}, ...]（None 已过滤）。"""
    b = sum(1 for r in rows if r["ah"] == 1 and r["b0"] == 0)
    c = sum(1 for r in rows if r["ah"] == 0 and r["b0"] == 1)
    diff = b - c
    if diff >= GATE_SUPPORT:
        verdict = "SUPPORTED"
    elif abs(diff) <= GATE_NULL:
        verdict = "NO_GAIN"
    elif -diff >= 3:
        verdict = "FALSIFIED"
    else:
        verdict = f"INCONCLUSIVE(diff={diff}, 介于预设区间之间)"
    return {"b_ah_only": b, "c_b0_only": c, "diff": diff, "verdict": verdict}


def build_graph_and_retrieve(questions: list[dict]) -> tuple[dict, dict]:
    """图构建（一次性 Qwen）+ 两臂检索。返回 (corpus, {qid: {b0, ah}})。"""
    from experiments.embed_cache import CachedBgeM3Provider

    cache, vecs, relations = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)
    entity_cache = json.loads(
        (CACHE_DIR / "entity_cache_textside_novel.json").read_text())

    # J-Lens 读出 novel ee 边（唯一需要 Qwen 的步骤，函数内部加载并释放）
    readout_edges = jlens_readout_top_pairs(entity_cache, S2_READOUT_TOP_PAIRS)

    embed_fn = CachedBgeM3Provider().embed
    chunk_ids = sorted(cache["chunks"].keys())
    chunk_emb = np.asarray(
        embed_fn([corpus[cid] for cid in chunk_ids]), dtype=np.float64)
    norms = np.linalg.norm(chunk_emb, axis=1, keepdims=True)
    chunk_emb = chunk_emb / np.where(norms > 0, norms, 1.0)

    index = LightRagIndex(cache, vecs, relations)
    stats = augment_index(index, entity_cache, readout_edges)
    print(f"  graph: {stats['n_base_entities']} concepts "
          f"+ {stats['n_new_entities']} entities, "
          f"+{stats['n_ee_edges']} ee +{stats['n_ec_edges']} ec "
          f"(jlens_readout)", flush=True)
    index.build_embeddings(embed_fn)

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    ranked: dict[str, dict] = {}
    for qi, q in enumerate(questions):
        qv = query_emb[qi] / (np.linalg.norm(query_emb[qi]) + 1e-12)
        q_sims = chunk_emb @ qv
        b0_ids = [chunk_ids[j] for j in np.argsort(-q_sims)[:TOP_K]]
        bge_sim = {chunk_ids[j]: float(q_sims[j]) for j in range(len(chunk_ids))}
        scores, _dbg = lightrag_retrieve(index, qv)
        graph_ranked = [cid for cid, _s in sorted(
            scores.items(), key=lambda x: (x[1], bge_sim.get(x[0], 0.0)),
            reverse=True)]
        merged: list[str] = []
        pools = [iter(graph_ranked), iter(b0_ids)]
        while len(merged) < TOP_K:
            for pool in pools:
                if len(merged) >= TOP_K:
                    break
                for cid in pool:
                    if cid not in merged:
                        merged.append(cid)
                        break
        ranked[q["id"]] = {"b0": b0_ids, "ah": merged}
        if (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)
    return corpus, ranked


def run_eval(questions: list[dict], corpus: dict, ranked: dict,
             llm, limit: int | None = None, verbose: bool = True) -> list[dict]:
    """生成 + rubric 判分（双臂，API 并行）。"""
    gold_map = {q["id"]: {"answer": q.get("answer", ""),
                          "evidence": q.get("evidence") or []} for q in questions}
    qs = questions[:limit] if limit else questions

    def _eval_one(args):
        q, arm = args
        ctx = " ".join(corpus[cid] for cid in ranked[q["id"]][arm])
        ans = generate_answer_faithful(q["question"], ctx, llm)
        res = rubric_judge(q["question"], ans, gold_map[q["id"]], llm)
        return q["id"], arm, {**res, "answer": ans}

    rows: dict[str, dict] = {q["id"]: {"qid": q["id"], "question": q["question"]}
                             for q in qs}
    tasks = [(q, arm) for q in qs for arm in ["b0", "ah"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_eval_one, t) for t in tasks]
        done = 0
        for future in as_completed(futures):
            qid, arm, rec = future.result()
            rows[qid][arm] = rec
            done += 1
            if verbose and done % 20 == 0:
                print(f"    eval {done}/{len(tasks)}", flush=True)
    return [rows[q["id"]] for q in qs]


def analyze(rows: list[dict]) -> dict:
    """主判决 + 次要指标。"""
    paired = [{"b0": r["b0"].get("acc_binary"), "ah": r["ah"].get("acc_binary")}
              for r in rows
              if r.get("b0", {}).get("acc_binary") is not None
              and r.get("ah", {}).get("acc_binary") is not None]
    v = verdict_from_pairs(paired)
    acc = lambda arm: (sum(p[arm] for p in paired) / len(paired)) if paired else None
    cov = lambda arm: float(np.mean(
        [r[arm].get("coverage", 0) for r in rows if arm in r])) if rows else None
    return {
        "n": len(rows), "n_paired": len(paired),
        "b0_acc": acc("b0"), "ah_acc": acc("ah"),
        "b0_coverage": cov("b0"), "ah_coverage": cov("ah"),
        **v,
    }


def repair_empty_generations() -> None:
    """修复首轮空答案（v4 推理链吃掉 budget）：升 budget 重生成 → 重判 → 重算判决。"""
    report = json.load(open(OUT_PATH))
    ranked = report["ranked"]
    cache, _v, _r = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)
    questions = {q["id"]: q for q in load_l4_questions()}

    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    def _gen_escalated(question: str, ctx: str) -> str:
        prompt = GEN_PROMPT.format(context=ctx[:12000], question=question)
        for kwargs in [dict(max_tokens=2000),  # 同协议升 budget
                       dict(max_tokens=GEN_MAX_TOKENS, thinking=False)]:  # 兜底关推理
            msg = llm.complete(prompt, **kwargs)
            text = msg.content if hasattr(msg, "content") else str(msg)
            if text.strip():
                return text
        return ""

    n_fixed = 0
    for row in report["per_query"]:
        q = questions[row["qid"]]
        gold = {"answer": q.get("answer", ""), "evidence": q.get("evidence") or []}
        for arm in ["b0", "ah"]:
            rec = row.get(arm) or {}
            if (rec.get("answer") or "").strip():
                continue
            ctx = " ".join(corpus[cid] for cid in ranked[row["qid"]][arm])
            ans = _gen_escalated(q["question"], ctx)
            res = rubric_judge(q["question"], ans, gold, llm)
            row[arm] = {**res, "answer": ans, "repaired": True}
            n_fixed += 1
            print(f"  repaired {row['qid']}[{arm}]: len={len(ans)} "
                  f"acc={res.get('acc_binary')}", flush=True)

    report["analysis"] = analyze(report["per_query"])
    report.setdefault("notes", []).append(
        f"repair: {n_fixed} empty generations regenerated (escalated budget), "
        "rubric_judge now rejects empty answers (guard)")
    json.dump(report, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=1))
    print(f"repaired {n_fixed}, saved -> {OUT_PATH}")


def _selftest():
    """判决逻辑 + 生成 prompt 形状，零 GPU 零 API。"""
    # verdict gates
    mk = lambda b, c: ([{"b0": 0, "ah": 1}] * b + [{"b0": 1, "ah": 0}] * c
                       + [{"b0": 1, "ah": 1}] * 10)
    assert verdict_from_pairs(mk(5, 0))["verdict"] == "SUPPORTED"
    assert verdict_from_pairs(mk(2, 0))["verdict"] == "NO_GAIN"
    assert verdict_from_pairs(mk(1, 1))["verdict"] == "NO_GAIN"
    assert verdict_from_pairs(mk(0, 3))["verdict"] == "FALSIFIED"
    assert "INCONCLUSIVE" in verdict_from_pairs(mk(4, 0))["verdict"]
    # prompt shape
    p = GEN_PROMPT.format(context="CTX", question="Q?")
    assert "concisely" not in p and "CTX" in p and "Q?" in p
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="3 题全链路，打印生成与判分供人工核对")
    ap.add_argument("--repair", action="store_true",
                    help="修复首轮空答案（升 budget 重生成+重判），不重跑检索")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if args.repair:
        repair_empty_generations()
        return

    questions = load_l4_questions()
    print(f"novel Creative Generation: {len(questions)} questions", flush=True)

    corpus, ranked = build_graph_and_retrieve(questions)

    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    if args.smoke:
        rows = run_eval(questions, corpus, ranked, llm, limit=3)
        for r in rows:
            print(f"== {r['qid']}\nQ: {r['question'][:150]}")
            for arm in ["b0", "ah"]:
                rec = r[arm]
                print(f"  [{arm}] acc_binary={rec.get('acc_binary')} "
                      f"accuracy={rec.get('accuracy')} cov={rec.get('coverage')} "
                      f"fe={rec.get('factual_errors')}")
                print(f"       {(rec.get('answer') or '')[:200]}")
        return

    rows = run_eval(questions, corpus, ranked, llm)
    report = {
        "method": "Phase 55: novel L4 全量多跳增益确认（rubric judge + 生成改造）",
        "gates": {"support_b_minus_c": GATE_SUPPORT, "null_abs": GATE_NULL},
        "analysis": analyze(rows),
        "ranked": {qid: r for qid, r in ranked.items()},
        "per_query": rows,
    }
    json.dump(report, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=1))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
