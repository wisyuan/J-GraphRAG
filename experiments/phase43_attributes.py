"""Phase 43: 实体属性——prefill scan 能否读出概念的文档特定属性。

问题：对 medical top-15 概念（按 DF = len(concept_chunks) 排序），
J-Lens 1 次 forward 的 prefill 读出能否给出该概念在此文档中的真实
属性，精度能否达到 DeepSeek generate()（~10s/次）的水平。

变体（同一块含概念的 chunk 上下文）：
  V1: assistant prefill "{concept} has" → 读 position -1 workspace
  V2: assistant prefill "{concept} is characterized by" → 读 position -1
  V3（对照/上限）: DeepSeek generate() 同 chunk 直接列属性
  V4: 概念自身位置读出（phase39/44 验证过的机制）——user 问概念 +
      prefill "The concepts are: {concept}" → 读概念位置的 workspace
      （V1/V2 冒烟为 template cloze 失败：position -1 在 cloze prefill
      后只产出句法续词 "several/properties/characteristics"，与文档无关）

属性词过滤（phase39 精神）：decode_topk_custom（phase35 STOP）后再过
ROLE_STOP（stem-aware）、剔除概念自身/词形变体，保留 top-5。

判决：DeepSeek judge 逐属性判定"该属性是否为该概念在此文档中的真实
属性"（precision）；并统计 V1/V2 与 V3 的重合度（exact/stem 匹配）。
成功线：precision >= 80%（1 次 forward vs generate 的 ~10s）。

产出：experiments/m6/phase43_attributes.json

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase43_attributes            # 全量 top-15
    python -m experiments.phase43_attributes --smoke 3  # 冒烟 3 概念（不落盘）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase35_prefill_position_scan import STOP, decode_topk_custom
from experiments.phase37_relation_prompt_variants import wrap_chat
from experiments.phase39_two_pass_cache import ROLE_STOP, _stem

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE = EXP / "concept_cache" / "concept_cache_medical_twopass.json"
OUT_PATH = EXP / "phase43_attributes.json"

TOP_K = 15
N_ATTR = 5


# ── attribute word filtering (phase39 spirit) ─────────────────────────

def _dedupe_stem(word: str) -> str:
    """Stronger stem for dedupe keys only: _stem + gerund/past stripping
    (tests/testing/tested waste slots otherwise)."""
    w = _stem(word)
    for suf in ("ing", "ed"):
        if len(w) > 6 and w.endswith(suf):
            w = w[: -len(suf)]
    return w


def filter_attrs(words: list[dict], concept: str, n: int = N_ATTR) -> list[str]:
    """Keep content attribute words: drop concept itself/inflections and
    ROLE_STOP template words (stem-aware)."""
    out: list[str] = []
    seen: set[str] = set()
    cstem = _stem(concept)
    for w in words:
        tok = w["token"]
        low = tok.lower()
        stem = _stem(tok)
        if low in STOP or stem in ROLE_STOP or low in ROLE_STOP:
            continue
        if low == concept.lower() or stem == cstem:
            continue
        if _dedupe_stem(tok) in seen:
            continue
        seen.add(_dedupe_stem(tok))
        out.append(tok)
        if len(out) >= n:
            break
    return out


def read_attrs(lens, lens_model, tokenizer, layer: int, chunk: str,
               concept: str, prefill: str) -> list[str]:
    user = f"What are the properties of {concept} in this text?\n\n{chunk[:500]}"
    prompt = wrap_chat(tokenizer, user, prefill)
    logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                              positions=[-1], max_seq_len=1024)
    words = decode_topk_custom(logits[layer][0], tokenizer, n=10, scan=50)
    return filter_attrs(words, concept)


def read_attrs_at_concept(lens, lens_model, tokenizer, layer: int,
                          chunk: str, concept: str,
                          corpus_words: set[str]) -> list[str]:
    """V4: read the workspace AT the concept's own prefill position
    ("The concepts are: {concept}") — the mechanism validated in
    phase39/44, which yields document-specific role/attribute words."""
    user = f"What concepts does this text discuss?\n\n{chunk[:500]}"
    prefill = f"The concepts are: {concept}"
    prompt = wrap_chat(tokenizer, user, prefill)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    texts = [tokenizer.decode([t]) for t in ids]
    offsets, pos = [], 0
    for t in texts:
        offsets.append(pos)
        pos += len(t)
    anchor = "".join(texts).rfind(prefill)
    if anchor < 0:
        return []
    cpos = next((i for i in range(len(offsets) - 1, -1, -1)
                 if offsets[i] <= anchor + len("The concepts are: ")), None)
    if cpos is None:
        return []
    logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                              positions=[cpos], max_seq_len=1024)
    words = decode_topk_custom(logits[layer][0], tokenizer, n=16, scan=60)
    words = [w for w in words if w["token"].lower() in corpus_words]
    return filter_attrs(words, concept)


# ── DeepSeek V3 + judge ───────────────────────────────────────────────

def deepseek_attrs(llm, concept: str, chunk: str) -> list[str]:
    prompt = (
        f"Text:\n{chunk[:600]}\n\n"
        f"List up to 5 one-word attributes or properties of \"{concept}\" "
        f"as described in this text. Output one word per line, no numbering."
    )
    try:
        msg = llm.complete(prompt, max_tokens=60)
        resp = msg.content if hasattr(msg, "content") else str(msg)
    except Exception:
        return []
    words = []
    for line in resp.splitlines():
        w = re.sub(r"[^A-Za-z-]", "", line.strip()).strip("-").lower()
        if len(w) >= 4 and w.isalpha() and w not in STOP \
                and _stem(w) not in ROLE_STOP and _stem(w) != _stem(concept):
            words.append(w)
    return words[:N_ATTR]


def judge_attrs(llm, concept: str, attrs: list[str], chunk: str) -> dict:
    """One batched judge call per concept. Returns {attr: bool|None}."""
    if not attrs:
        return {}
    lst = "\n".join(f"- {a}" for a in attrs)
    prompt = (
        f"Text:\n{chunk[:600]}\n\n"
        f"For each word below, decide whether it is a TRUE attribute or "
        f"property of \"{concept}\" as described in this text. "
        f"Answer one line per word, exactly \"word: YES\" or \"word: NO\".\n\n"
        f"{lst}"
    )
    verdicts = {a: None for a in attrs}
    try:
        msg = llm.complete(prompt, max_tokens=120)
        resp = msg.content if hasattr(msg, "content") else str(msg)
    except Exception:
        return verdicts
    for line in resp.splitlines():
        m = re.match(r"\s*[-*\d.]*\s*([A-Za-z-]+)\s*[:\-]\s*(YES|NO)",
                     line.strip(), re.IGNORECASE)
        if m:
            w = m.group(1).lower()
            if w in verdicts:
                verdicts[w] = m.group(2).upper() == "YES"
    return verdicts


# ── main run ──────────────────────────────────────────────────────────

def run(lens, lens_model, tokenizer, llm, concepts: list[str],
        concept_chunks: dict, chunk_texts: dict, smoke: bool) -> dict:
    layer = lens.source_layers[-1]
    corpus_words = {
        m.group().lower()
        for text in chunk_texts.values()
        for m in re.finditer(r"[a-zA-Z]{4,}", text)
    }
    results = []
    t0 = time.perf_counter()

    # GPU readouts first (sequential, GPU-exclusive)
    for i, concept in enumerate(concepts):
        cids = concept_chunks.get(concept, [])
        chunk = chunk_texts[cids[0]] if cids else ""
        ctx_src = cids[0] if cids else "none"
        v1 = read_attrs(lens, lens_model, tokenizer, layer, chunk, concept,
                        f"{concept} has")
        v2 = read_attrs(lens, lens_model, tokenizer, layer, chunk, concept,
                        f"{concept} is characterized by")
        v4 = read_attrs_at_concept(lens, lens_model, tokenizer, layer,
                                   chunk, concept, corpus_words)
        results.append({"concept": concept, "chunk_id": ctx_src,
                        "chunk": chunk[:600], "v1": v1, "v2": v2, "v4": v4})
        if smoke or i < 5:
            print(f"\n  [{concept}] ({ctx_src})")
            print(f"    V1 '{concept} has':               {v1}")
            print(f"    V2 '{concept} is characterized by': {v2}")
            print(f"    V4 concept-position:              {v4}")

    gpu_s = time.perf_counter() - t0

    # DeepSeek V3 + judge (parallel, CPU)
    t1 = time.perf_counter()
    def _ds(r):
        r["v3"] = deepseek_attrs(llm, r["concept"], r["chunk"])
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_ds, results))

    def _judge(r):
        attrs = sorted(set(r["v1"]) | set(r["v2"]) | set(r["v4"]))
        r["judge"] = judge_attrs(llm, r["concept"], attrs, r["chunk"])
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_judge, results))
    ds_s = time.perf_counter() - t1

    # Metrics
    def precision(key, r):
        vs = [r["judge"].get(a) for a in r[key]]
        vs = [v for v in vs if v is not None]
        return (sum(vs) / len(vs)) if vs else None

    def overlap(r, key):
        v3s = {_stem(w) for w in r["v3"]}
        return sum(1 for a in r[key] if _stem(a) in v3s)

    n_judged = {k: [0, 0] for k in ("v1", "v2", "v4")}  # [true, total]
    for r in results:
        for key in ("v1", "v2", "v4"):
            r[f"{key}_precision"] = precision(key, r)
            r[f"{key}_overlap_v3"] = overlap(r, key)
        for key in ("v1", "v2", "v4"):
            for a in r[key]:
                v = r["judge"].get(a)
                if v is not None:
                    n_judged[key][1] += 1
                    n_judged[key][0] += int(v)
        if smoke:
            print(f"    V3 DeepSeek: {r['v3']}  judge: {r['judge']}")

    summary = {"n_concepts": len(results), "gpu_time_s": round(gpu_s, 1),
               "deepseek_time_s": round(ds_s, 1)}
    for key in ("v1", "v2", "v4"):
        t, n = n_judged[key]
        precs = [r[f"{key}_precision"] for r in results
                 if r[f"{key}_precision"] is not None]
        summary[f"{key}_precision_micro"] = round(t / n, 3) if n else None
        summary[f"{key}_precision_macro"] = (
            round(sum(precs) / len(precs), 3) if precs else None)
        summary[f"{key}_n_judged"] = n
        summary[f"{key}_overlap_v3_total"] = sum(
            r[f"{key}_overlap_v3"] for r in results)
    return {"summary": summary, "results": results}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N concepts, no JSON output")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    concept_chunks = cache["concept_chunks"]
    # top-k concepts by document frequency
    concepts = sorted(concept_chunks, key=lambda c: -len(concept_chunks[c]))
    concepts = concepts[: args.smoke or args.top_k]
    print(f"concepts ({len(concepts)}): {concepts}")

    from experiments.phase4_dig_graphragbench import load_graphrag_bench
    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    chunk_texts = dict(corpus)

    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print("\n[2/2] Running attribute readout...")
    out = run(lens, lens_model, tokenizer, llm, concepts, concept_chunks,
              chunk_texts, smoke=bool(args.smoke))

    s = out["summary"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({s['n_concepts']} concepts)")
    for key in ("v1", "v2", "v4"):
        print(f"  {key.upper()}: precision micro={s[f'{key}_precision_micro']}"
              f" macro={s[f'{key}_precision_macro']}"
              f" (n={s[f'{key}_n_judged']})"
              f"  overlap-with-V3={s[f'{key}_overlap_v3_total']}")
    print(f"  GPU: {s['gpu_time_s']}s, DeepSeek: {s['deepseek_time_s']}s")
    print(f"  success line: precision >= 0.80")

    if not args.smoke:
        out["model"] = cand["name"]
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
