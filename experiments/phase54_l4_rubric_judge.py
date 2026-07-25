"""Phase 54 S2: L4 (Creative Generation) rubric re-judge.

Phase 54 S1 (docs/j-graphrag-complete-method.md §19) 判决：novel L4 趋零是评测协议
artifact —— 旧 judge 要求生成答案与 gold"传达相同信息"，对创意生成任务结构性产生
假阴性；官方该级别用 Accuracy / Factual Score / Coverage 三维 rubric。

本脚本不重跑检索与生成，只对已存答案重判：
  - phase50_lightrag_j.json  : arms {b0, a, b, ah}（概念图 LightRAG-J）
  - phase53_textside_entities.json s3: arm concept+entity ah（最终 LightRAG-J）
  - phase51_hipporag_j.json  : arms {a, b, c_w0.3, c_w0.5, c_w0.8}（HippoRAG-J）

预设判决指标（先于运行定义）：
  主判决 —— rubric judge 下 B0 novel L4 ACC >= 0.4（官方排行榜普通 RAG 同题 38-42%，
  人工抽查答案质量合格）→ 确认旧 judge 假阴性主导，协议修复有效。
  次输出 —— 各臂 L4 ACC 替换后重算总 ACC 与保持率（L1-L3 沿用旧判）。

用法：
  python -m experiments.phase54_l4_rubric_judge --selftest   # mock，零 API
  python -m experiments.phase54_l4_rubric_judge --smoke      # 3 例真实 judge，人工核对
  python -m experiments.phase54_l4_rubric_judge              # 全量重判
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
BENCH_DIR = Path("/tmp/graphrag-bench")

# 预设判决线（见模块 docstring）
B0_NOVEL_L4_GATE = 0.4
# rubric 二值化规则：内容准确度 >= 0.5 且无事实冲突 → ACC 1
RUBRIC_ACC_MIN = 0.5

RUBRIC_PROMPT = """You are evaluating a creative-generation answer to a question about a source text.

Question: {question}

Reference (gold) answer — ONE valid realization, NOT the only acceptable one:
{gold_answer}

Key facts the answer should be consistent with (evidence):
{evidence}

Generated answer:
{generated_answer}

Evaluate the generated answer on three dimensions:
1. factual_errors: number of statements that CONTRADICT the evidence or gold answer (creative embellishments consistent with the source are NOT errors).
2. accuracy: 0-1, how faithfully the answer's content matches the source facts (style/wording may differ freely).
3. coverage: 0-1, fraction of the evidence points the answer addresses.

Reply with ONLY a JSON object, no other text:
{{"factual_errors": <int>, "accuracy": <float>, "coverage": <float>}}"""


def load_gold(domain: str) -> dict:
    """qid -> {answer, evidence} for a domain's full question bank."""
    qs = json.load(open(BENCH_DIR / f"{domain}_questions.json"))
    out = {}
    for q in qs:
        ev = q.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        out[q["id"]] = {"answer": q.get("answer", ""), "evidence": ev}
    return out


def rubric_judge(question: str, generated: str, gold: dict, llm) -> dict:
    """Judge one L4 answer. Returns {factual_errors, accuracy, coverage, acc_binary, error?}."""
    if not generated.strip():
        # 空答案触发 judge 真空 1.0 假阳性（Phase 55 实测 14 例）——拒绝评分
        return {"acc_binary": None, "error": "empty generated answer"}
    prompt = RUBRIC_PROMPT.format(
        question=question,
        gold_answer=gold["answer"],
        evidence="\n".join(f"- {e}" for e in gold["evidence"]) or "(none provided)",
        generated_answer=generated,
    )
    # v4 是推理模型：judge 必须关 thinking，否则长答案触发推理链爆炸、
    # content 静默为空（finish_reason=length）
    msg = llm.complete(prompt, max_tokens=800, thinking=False)
    text = msg.content if hasattr(msg, "content") else str(msg)
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {"acc_binary": None, "error": f"no JSON in judge output: {text[:80]}"}
    try:
        r = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"acc_binary": None, "error": f"bad JSON: {m.group(0)[:80]}"}
    fe = int(r.get("factual_errors", 1))
    acc = float(r.get("accuracy", 0.0))
    cov = float(r.get("coverage", 0.0))
    return {
        "factual_errors": fe,
        "accuracy": acc,
        "coverage": cov,
        "acc_binary": 1 if (acc >= RUBRIC_ACC_MIN and fe == 0) else 0,
    }


def rejudge_arm(per_query: list, arm_getter, gold_map: dict, llm, only_l4: bool = True,
                limit: int | None = None, verbose: bool = False) -> dict:
    """Re-judge L4 answers of one arm; return per-question results + aggregate."""
    rows = []
    for q in per_query:
        if only_l4 and q.get("level") != "L4":
            continue
        got = arm_getter(q)
        if got is None:
            continue
        gold = gold_map.get(q["qid"])
        if gold is None:
            continue
        res = rubric_judge(q["question"], got.get("answer") or "", gold, llm)
        res.update({"qid": q["qid"], "old_acc": bool(got.get("acc")),
                    "answer_head": (got.get("answer") or "")[:150]})
        rows.append(res)
        if verbose:
            print(f"  {q['qid']}: old={res['old_acc']} new={res.get('acc_binary')} "
                  f"acc={res.get('accuracy')} cov={res.get('coverage')} fe={res.get('factual_errors')}")
        if limit and len(rows) >= limit:
            break
    judged = [r for r in rows if r.get("acc_binary") is not None]
    return {
        "n": len(rows),
        "n_judged": len(judged),
        "old_l4_acc": sum(r["old_acc"] for r in rows) / len(rows) if rows else None,
        "new_l4_acc": sum(r["acc_binary"] for r in judged) / len(judged) if judged else None,
        "mean_coverage": sum(r.get("coverage", 0) for r in judged) / len(judged) if judged else None,
        "rows": rows,
    }


def recompute_total(per_query: list, arm: str | None, new_l4: dict) -> float | None:
    """Total ACC with L1-L3 from old judge and L4 from rubric re-judge."""
    new_by_qid = {r["qid"]: r["acc_binary"] for r in new_l4["rows"]
                  if r.get("acc_binary") is not None}
    n, correct = 0, 0
    for q in per_query:
        if arm is not None:
            got = (q.get("methods") or {}).get(arm)
        else:
            got = q  # phase53 flat per_query
        if got is None:
            continue
        n += 1
        if q["level"] == "L4" and q["qid"] in new_by_qid:
            correct += new_by_qid[q["qid"]]
        elif q["level"] != "L4":
            correct += 1 if got.get("acc") else 0
    return correct / n if n else None


def _selftest():
    """Mock LLM: parsing + aggregation, zero API."""

    class MockMsg:
        def __init__(self, c):
            self.content = c

    class MockLLM:
        def complete(self, prompt, max_tokens=150, thinking=None):
            if "fish" in prompt.lower():
                return MockMsg('{"factual_errors": 0, "accuracy": 0.8, "coverage": 0.6}')
            return MockMsg("I cannot judge this.")  # malformed path

    llm = MockLLM()
    gold = {"answer": "g", "evidence": ["e1"]}
    r1 = rubric_judge("fish question", "ans", gold, llm)
    assert r1["acc_binary"] == 1 and r1["coverage"] == 0.6, r1
    r2 = rubric_judge("other", "ans", gold, llm)
    assert r2["acc_binary"] is None and "error" in r2, r2
    pq = [
        {"qid": "q1", "level": "L1", "question": "x", "methods": {"b0": {"acc": True, "answer": "a"}}},
        {"qid": "q2", "level": "L4", "question": "fish?", "methods": {"b0": {"acc": False, "answer": "a"}}},
    ]
    out = rejudge_arm(pq, lambda q: q["methods"]["b0"], {"q2": gold}, llm)
    assert out["n"] == 1 and out["new_l4_acc"] == 1.0 and out["old_l4_acc"] == 0.0, out
    tot = recompute_total(pq, "b0", out)
    assert tot == 1.0, tot  # L1 old True + L4 re-judged 1
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="3 real judge calls, print for review")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    p50 = json.load(open(EXP / "phase50_lightrag_j.json"))
    gold_novel = load_gold("novel")

    if args.smoke:
        l4 = [q for q in p50["domains"]["novel"]["per_query"] if q["level"] == "L4"][:3]
        for q in l4:
            b0 = q["methods"]["b0"]
            res = rubric_judge(q["question"], b0.get("answer") or "", gold_novel[q["qid"]], llm)
            print(f"== {q['qid']}\nQ: {q['question'][:120]}\n"
                  f"old_acc={b0.get('acc')} -> {json.dumps(res)}\n"
                  f"answer head: {(b0.get('answer') or '')[:200]}\n")
        return

    leaderboard = json.load(open(EXP / "graphrag_bench_leaderboard.json"))
    p53 = json.load(open(EXP / "phase53_textside_entities.json"))
    p51 = json.load(open(EXP / "phase51_hipporag_j.json"))
    gold_med = load_gold("medical")

    report: dict = {"gate": {"b0_novel_l4_acc_min": B0_NOVEL_L4_GATE},
                    "lightrag_concept_graph": {}, "lightrag_final": {}, "hipporag": {}}

    # ── phase50 LightRAG-J concept-graph arms (b0/a/b/ah) ──
    for dom, gold in [("medical", gold_med), ("novel", gold_novel)]:
        pq = p50["domains"][dom]["per_query"]
        for arm in ["b0", "a", "b", "ah"]:
            r = rejudge_arm(pq, lambda q, a=arm: (q.get("methods") or {}).get(a), gold, llm)
            total = recompute_total(pq, arm, r)
            lb_pct = leaderboard[dom]["lightrag"]["avg"]
            factor = p53["s3"][dom]["factor"]
            report["lightrag_concept_graph"][f"{dom}/{arm}"] = {
                **r,  # keep per-question rows for threshold-sensitivity analysis
                "total_acc_recomputed": total,
                "retention_recomputed": (total * factor / (lb_pct / 100)) if total else None,
            }

    # ── phase53 final arm (concept+entity ah), flat per_query ──
    for dom, gold in [("medical", gold_med), ("novel", gold_novel)]:
        blk = p53["s3"][dom]
        pq = blk["per_query"]
        r = rejudge_arm(pq, lambda q: q, gold, llm)
        total = recompute_total(pq, None, r)
        lb_pct = leaderboard[dom]["lightrag"]["avg"]
        report["lightrag_final"][dom] = {
            **r,  # keep per-question rows
            "total_acc_recomputed": total,
            "retention_recomputed": (total * blk["factor"] / (lb_pct / 100)) if total else None,
            "retention_old": blk["retention_vs_leaderboard_lightrag"],
        }

    # ── phase51 HippoRAG-J arms ──
    for dom, gold in [("medical", gold_med), ("novel", gold_novel)]:
        pq = p51["domains"][dom]["per_query"]
        factor = p51["domains"][dom]["summary"]["factor"]
        for arm in ["a", "b", "c_w0.3", "c_w0.5", "c_w0.8"]:
            r = rejudge_arm(pq, lambda q, a=arm: (q.get("methods") or {}).get(a), gold, llm)
            total = recompute_total(pq, arm, r)
            lb_pct = leaderboard[dom]["hipporag2"]["avg"]
            report["hipporag"][f"{dom}/{arm}"] = {
                **r,  # keep per-question rows for threshold-sensitivity analysis
                "total_acc_recomputed": total,
                "retention_recomputed": (total * factor / (lb_pct / 100)) if total else None,
            }

    # ── verdict ──
    b0_novel = report["lightrag_concept_graph"]["novel/b0"]["new_l4_acc"]
    report["verdict"] = {
        "b0_novel_l4_new_acc": b0_novel,
        "gate_passed": (b0_novel is not None and b0_novel >= B0_NOVEL_L4_GATE),
    }
    out = EXP / "phase54_l4_rejudge.json"
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=1))
    print(f"saved -> {out}")


if __name__ == "__main__":
    sys.exit(main())
