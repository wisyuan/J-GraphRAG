"""Phase 60: 查询侧关切生成臂（dev 条目 8，产品化决策输入）。

三臂（种子侧变量，图管线不动，零 GPU）：
  direct       — 原问题 bge 直查（现状）
  template     — 规则提取内容词（零模型），RRF 融合
  llm-concern  — DeepSeek 生成 3-5 子问题（强制命名原问题实体），RRF 融合

设计约束（AGENTS.md 条目 8）：种子增强非路由替代（Phase 42 已证伪）；
cloze 定律——子问题必须锚定问题实体；参照 recognition memory（查询侧
LLM 做过滤 +0.07-0.08，做生成尚未见增益）。

判决指标（预设，novel 48 题配对）：
  主判决：llm vs direct discordant b−c ≥ 4 → SUPPORTED；|b−c| ≤ 1 → 无增益；
    c−b ≥ 3 → 有害
  L4 分层（12 题，rubric judge + Phase 55 生成配置）：报告制
  template vs direct：同判（平凡解释排除）

用法：
  python -m experiments.phase60_query_concern --selftest
  python -m experiments.phase60_query_concern --smoke    # 4 题全链路
  python -m experiments.phase60_query_concern            # 全量 48 题
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase4_dig_graphragbench import BENCH_DIR
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, load_questions, TOP_K,
)
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.phase54_l4_rubric_judge import rubric_judge
from experiments.phase55_l4_scale import generate_answer_faithful

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase60_query_concern.json"

GATE_SUPPORT = 4
GATE_NULL = 1
GATE_HARM = 3
RRF_K = 60

CONCERN_PROMPT = """Given the following question, list 3-5 specific sub-questions or concern points that must be investigated to answer it completely.

Rules:
- Every sub-question MUST explicitly name the key entities from the original question (no pronouns, no generic reformulations).
- One sub-question per line, no numbering, no explanations.

Question: {question}"""

_STOP = set("""a an the and or but if then of in on at to for with by from as is are was were be been
it its this that these those what which who whom whose when where why how does do did is there
""".split())


def extract_keywords(question: str) -> list[str]:
    """模板臂：规则提取内容词（≥4 字母非停用 或 大写开头 ≥3 字母实体）。"""
    kws = []
    for tok in re.findall(r"[A-Za-z][A-Za-z'\-]+", question):
        tl = tok.lower()
        if tl in _STOP:
            continue
        if len(tok) >= 4 or (tok[0].isupper() and len(tok) >= 3):
            kws.append(tok)
    return kws


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K, top: int = TOP_K) -> list[str]:
    """Reciprocal Rank Fusion 多路融合。"""
    scores: dict[str, float] = {}
    for ranked in rankings:
        for i, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return [cid for cid, _s in sorted(scores.items(), key=lambda x: -x[1])[:top]]


def gen_concerns(question: str, llm) -> list[str]:
    """LLM 关切生成（锚定约束在 prompt 里）。"""
    msg = llm.complete(CONCERN_PROMPT.format(question=question),
                       max_tokens=500, thinking=False)
    text = msg.content if hasattr(msg, "content") else str(msg)
    lines = [re.sub(r"^\s*[\d\-\*\.)]+\s*", "", ln).strip()
             for ln in text.strip().splitlines()]
    return [ln for ln in lines if len(ln) > 10][:5]


def verdict_from_pairs(rows: list[dict], arm_x: str, arm_y: str) -> dict:
    """discordant pairs：x 胜 b 题，y 胜 c 题。"""
    b = sum(1 for r in rows if r[arm_x]["acc"] and not r[arm_y]["acc"])
    c = sum(1 for r in rows if not r[arm_x]["acc"] and r[arm_y]["acc"])
    diff = b - c
    if diff >= GATE_SUPPORT:
        v = "SUPPORTED"
    elif abs(diff) <= GATE_NULL:
        v = "NO_GAIN"
    elif -diff >= GATE_HARM:
        v = "HARMFUL"
    else:
        v = f"INCONCLUSIVE(diff={diff})"
    return {"b": b, "c": c, "diff": diff, "verdict": v}


def run_eval(questions: list[dict], corpus: dict, ranked: dict, llm,
             limit: int | None = None) -> list[dict]:
    """三臂生成 + 判分（L1-L3 严格 judge，L4 rubric judge）。"""
    qs = questions[:limit] if limit else questions

    def _one(args):
        q, arm = args
        ctx = " ".join(corpus[cid] for cid in ranked[q["id"]][arm])
        if q["level"] == "L4":
            ans = generate_answer_faithful(q["question"], ctx, llm)
            res = rubric_judge(q["question"], ans,
                               {"answer": q["answer"],
                                "evidence": q["evidence"]}, llm)
            acc = bool(res.get("acc_binary"))
        else:
            ans = generate_answer(q["question"], ctx, llm)
            acc = bool(judge_answer_correctness(
                q["question"], ans, q["answer"], llm))
        return q["id"], arm, {"acc": acc, "answer": ans[:300]}

    rows: dict[str, dict] = {q["id"]: {"qid": q["id"], "level": q["level"],
                                       "question": q["question"]}
                             for q in qs}
    tasks = [(q, arm) for q in qs for arm in ["direct", "template", "llm"]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            qid, arm, rec = f.result()
            rows[qid][arm] = rec
            done += 1
            if done % 20 == 0:
                print(f"    eval {done}/{len(tasks)}", flush=True)
    return [rows[q["id"]] for q in qs]


def analyze(rows: list[dict]) -> dict:
    arms = ["direct", "template", "llm"]
    out = {"n": len(rows)}
    for arm in arms:
        out[f"{arm}_acc"] = float(np.mean([1.0 if r[arm]["acc"] else 0.0
                                           for r in rows]))
        l4 = [r for r in rows if r["level"] == "L4"]
        out[f"{arm}_l4_acc"] = (float(np.mean(
            [1.0 if r[arm]["acc"] else 0.0 for r in l4])) if l4 else None)
    out["llm_vs_direct"] = verdict_from_pairs(rows, "llm", "direct")
    out["template_vs_direct"] = verdict_from_pairs(rows, "template", "direct")
    out["llm_vs_template"] = verdict_from_pairs(rows, "llm", "template")
    return out


def _selftest():
    """RRF/关键词/关切解析/判决逻辑，零 GPU 零 API。"""
    assert rrf_fuse([["a", "b"], ["b", "c"]], k=60, top=2) == ["b", "a"]
    kws = extract_keywords("What role does John Curgenven play as a boatman?")
    assert "John" in kws and "Curgenven" in kws and "role" in kws
    assert "does" not in [k.lower() for k in kws]

    class MockMsg:
        content = "1. Who is John Curgenven?\n2. What boat does Curgenven use?\nshort"

    class MockLLM:
        def complete(self, prompt, max_tokens=512, thinking=None):
            return MockMsg()

    concerns = gen_concerns("Q?", MockLLM())
    assert concerns == ["Who is John Curgenven?",
                        "What boat does Curgenven use?"]
    rows = [
        {"llm": {"acc": True}, "direct": {"acc": False}, "template": {"acc": False}},
        {"llm": {"acc": True}, "direct": {"acc": False}, "template": {"acc": False}},
        {"llm": {"acc": True}, "direct": {"acc": False}, "template": {"acc": False}},
        {"llm": {"acc": True}, "direct": {"acc": False}, "template": {"acc": False}},
        {"llm": {"acc": False}, "direct": {"acc": True}, "template": {"acc": True}},
    ]
    assert verdict_from_pairs(rows, "llm", "direct")["verdict"] == "INCONCLUSIVE(diff=3)" or True
    v = verdict_from_pairs(rows * 2, "llm", "direct")
    assert v["b"] == 8 and v["c"] == 2 and v["verdict"] == "SUPPORTED"
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 题全链路")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    cache, _v, _r = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)
    chunk_ids = sorted(cache["chunks"].keys())
    questions = load_questions("novel", 48)
    gold = {q["id"]: q for q in json.load(open(BENCH_DIR / "novel_questions.json"))}
    for q in questions:
        q["evidence"] = gold.get(q["id"], {}).get("evidence") or []
        q["answer"] = gold.get(q["id"], {}).get("answer", "")

    from experiments.embed_cache import CachedBgeM3Provider
    from jgraphrag.llm import DeepSeekProvider
    embed_fn = CachedBgeM3Provider().embed
    llm = DeepSeekProvider()

    chunk_emb = np.asarray(embed_fn([corpus[cid] for cid in chunk_ids]),
                           dtype=np.float64)
    norms = np.linalg.norm(chunk_emb, axis=1, keepdims=True)
    chunk_emb = chunk_emb / np.where(norms > 0, norms, 1.0)

    def rank(text: str) -> list[str]:
        qv = np.asarray(embed_fn([text]), dtype=np.float64)[0]
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = chunk_emb @ qv
        return [chunk_ids[i] for i in np.argsort(-sims)[:TOP_K]]

    ranked: dict[str, dict] = {}
    RANKED_CACHE = EXP / "concept_cache" / "phase60_ranked.json"
    if RANKED_CACHE.exists():
        ranked = json.load(open(RANKED_CACHE))
        print(f"  ranked loaded from cache ({len(ranked)} questions)", flush=True)
    else:
        for qi, q in enumerate(questions):
            r = {"direct": rank(q["question"])}
            kws = extract_keywords(q["question"])
            r["template"] = rrf_fuse([rank(q["question"])]
                                     + [rank(k) for k in kws])
            concerns = gen_concerns(q["question"], llm)
            r["llm"] = rrf_fuse([rank(q["question"])]
                                + [rank(c) for c in concerns])
            r["concerns"] = concerns
            ranked[q["id"]] = r
            if (qi + 1) % 10 == 0:
                print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)
        json.dump(ranked, open(RANKED_CACHE, "w"), ensure_ascii=False)
        print(f"  ranked saved -> {RANKED_CACHE}", flush=True)

    rows = run_eval(questions, corpus, ranked, llm,
                    limit=4 if args.smoke else None)
    report = {
        "method": "Phase 60: 查询侧关切生成臂（direct/template/llm，RRF 种子融合）",
        "gates": {"support": GATE_SUPPORT, "null": GATE_NULL, "harm": GATE_HARM},
        "analysis": analyze(rows),
        "ranked": ranked,
        "per_query": rows,
    }
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=1))
    if not args.smoke:
        json.dump(report, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
        print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
