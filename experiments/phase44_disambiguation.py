"""Phase 44: 实体消歧——双概念关切 prompt 能否区分同一概念/词形变体/近义/无关。

问题：给定概念对 (A, B)（来自 medical 50 概念词表），J-Lens 符号读出
能否判别三类关系：
  same — 同一概念 / 词形变体（tumor/tumors, surgery/surgeries, ...）
  near — 近义但不同概念（cancer/carcinoma, drugs/medications, ...）
  diff — 无关概念对（blood/breast, liver/radiation, ...）

测试集构造（DeepSeek 预标注确认，见 build_pairs）：
  候选对生成后由 DeepSeek 分类 SAME/NEAR/DIFF，只保留标注与意图一致
  且每类 >=8 的对。标注缓存于 phase44_pairs.json（不含 GPU 操作）。

两个 prompt 变体（每对 2 次 forward）：
  V1 双概念关切 position -1（受限分类）：
     user: "This text discusses {A} and {B}. Are \"{A}\" and \"{B}\" the
            same concept (word-form variants count as the same), similar
            but distinct concepts, or unrelated concepts? Answer with
            exactly one word: same, similar, or unrelated." + 共现 chunk
     assistant prefill: "Answer:"
     → 读 position -1 原始 top-15 token，映射 same/similar/unrelated。
     （注：prefill "They are" 的语言先验会把读出锁死在 "related" 上，
     冒烟 4/6 误判，故改用 "Answer:"。）
  V2 反转 prefill scan（phase35/39 的列表式 prefill）：
     user: "What concepts does this text discuss?" + 共现 chunk
     assistant prefill: "The concepts are: {A}, {B}"
     → 读 A、B 各自位置的 workspace 词集（decode_topk_custom top-8 +
     corpus 词过滤），用两位置读出词集的重合度分类：
     共享 >=2 → same，0 → diff，1 → near。
     （注："{A} and {B} are" 形状下 A 位置的下一 token 是 "and"，
     workspace 被句法主导，冒烟 0.167 acc，故改用列表式 prefill——
     该设置在 phase39 中已验证能读出文档角色词。）

上下文：两概念共现 chunk（concept_chunks 交集）；无共现时退化为
各自首个 chunk 拼接（concat 策略，与 relations 构建一致）。

对照基线：bge-m3 余弦（CachedBgeM3Provider, BGE_M3_DEVICE=cpu）——
嵌入阈值法 vs J-Lens 读出。

产出：experiments/m6/phase44_disambiguation.json
  每对的 V1/V2 读出、判断、label，两变体与 BGE 基线的 3 类准确率。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase44_disambiguation              # 全量
    python -m experiments.phase44_disambiguation --smoke 5    # 冒烟（不落盘）
    python -m experiments.phase44_disambiguation --rebuild-pairs  # 重标测试集
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
import re

from experiments.phase35_prefill_position_scan import decode_topk_custom
from experiments.phase37_relation_prompt_variants import wrap_chat
from experiments.phase39_two_pass_cache import _stem

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE = EXP / "concept_cache" / "concept_cache_medical_twopass.json"
PAIRS_PATH = EXP / "phase44_pairs.json"
OUT_PATH = EXP / "phase44_disambiguation.json"

MIN_PER_CLASS = 8

# V1 answer-token → class (exact lowercase match; note "unrelated" must
# NOT be caught by "related", so no substring matching here).
V1_CLASS_TOKENS = {
    "same": "same", "identical": "same", "synonym": "same",
    "synonymous": "same", "equivalent": "same",
    "similar": "near", "related": "near", "associated": "near",
    "connected": "near", "alike": "near",
    "unrelated": "diff", "different": "diff", "distinct": "diff",
    "separate": "diff", "opposite": "diff",
}

# ── candidate pair pools (DeepSeek confirms the final labels) ─────────

SAME_CANDIDATES = [
    ("tumor", "tumors"), ("surgery", "surgeries"), ("cancer", "cancers"),
    ("breast", "breasts"), ("kidney", "kidneys"), ("genetic", "genetics"),
    ("injectable", "injection"), ("genes", "genetic"),
    ("medicine", "medications"), ("surgery", "surgical"),
    ("surgical", "surgeries"), ("surgeon", "surgical"),
    # NOTE: the 50-concept medical vocab only yields ~7 DeepSeek-confirmed
    # word-form pairs; rejected morphological candidates (genetic/genetics,
    # injectable/injection, ...) come back NEAR and move to the near pool.
]

NEAR_CANDIDATES = [
    ("cancer", "carcinoma"), ("tumor", "cancer"), ("tumor", "carcinoma"),
    ("drugs", "medications"), ("therapy", "treatments"),
    ("chemotherapy", "therapy"), ("surgery", "surgeon"),
    ("biopsy", "diagnosis"), ("symptoms", "diagnosis"),
    ("radiation", "chemotherapy"), ("liver", "kidney"),
    ("sugar", "blood"), ("breast", "chest"), ("heart", "chest"),
    ("genes", "mutation"), ("injection", "drugs"),
    ("genetic", "genetics"), ("genes", "genetics"),
    ("medicine", "medications"), ("injectable", "injection"),
    ("drugs", "medicine"),
]

DIFF_CANDIDATES = [
    ("cancer", "surgery"), ("blood", "breast"), ("liver", "radiation"),
    ("heart", "prostate"), ("muscle", "genes"), ("sugar", "carcinoma"),
    ("lungs", "pregnancy"), ("immune", "bones"), ("headaches", "kidneys"),
    ("trials", "safety"), ("family", "tumor"), ("chest", "drugs"),
    ("breath", "cervical"), ("surgery", "genetics"), ("blood", "surgery"),
    ("radiation", "pregnancy"), ("cancer", "kidney"), ("bones", "therapy"),
]


# ── test-set construction (DeepSeek annotation, no GPU) ───────────────

def _deepseek_label(a: str, b: str, llm) -> str | None:
    """Classify a pair as same/near/diff via DeepSeek. None on parse failure."""
    prompt = (
        f"Two medical terms: \"{a}\" and \"{b}\".\n"
        "Classify their relationship into exactly one of:\n"
        "  SAME — the same concept, e.g. singular/plural or word-form "
        "variants of one term\n"
        "  NEAR — distinct concepts but near-synonyms or very closely "
        "related in medical meaning\n"
        "  DIFF — unrelated or only loosely associated concepts\n"
        "Answer with one word: SAME, NEAR, or DIFF."
    )
    try:
        msg = llm.complete(prompt, max_tokens=8)
        resp = (msg.content if hasattr(msg, "content") else str(msg)).strip().upper()
    except Exception:
        return None
    for lab in ("SAME", "NEAR", "DIFF"):
        if resp.startswith(lab):
            return lab.lower()
    return None


def build_pairs(concepts: list[str]) -> dict:
    """Build the labeled pair set with DeepSeek pre-annotation."""
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    vocab = set(concepts)

    pairs: list[dict] = []
    counts = {"same": 0, "near": 0, "diff": 0}
    pools = [("same", SAME_CANDIDATES), ("near", NEAR_CANDIDATES),
             ("diff", DIFF_CANDIDATES)]
    for intended, pool in pools:
        random.shuffle(pool)
        for a, b in pool:
            if a not in vocab or b not in vocab:
                continue
            lab = _deepseek_label(a, b, llm)
            print(f"  [{intended:4}] {a}/{b}: DeepSeek={lab}")
            # keep only when DeepSeek confirms the intended class
            if lab == intended:
                pairs.append({"a": a, "b": b, "label": lab})
                counts[intended] += 1

    out = {"pairs": pairs, "counts": counts}
    PAIRS_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  pairs: {counts} → saved {PAIRS_PATH}")
    short = {k: MIN_PER_CLASS - v for k, v in counts.items() if v < MIN_PER_CLASS}
    if short:
        print(f"  WARNING: classes below {MIN_PER_CLASS}: {short}")
    return out


def load_pairs(concepts: list[str], rebuild: bool) -> dict:
    if PAIRS_PATH.exists() and not rebuild:
        return json.loads(PAIRS_PATH.read_text())
    print("[pairs] Building labeled pair set via DeepSeek...")
    return build_pairs(concepts)


# ── context selection ─────────────────────────────────────────────────

def pick_context(a: str, b: str, concept_chunks: dict, chunk_texts: dict,
                 max_chars: int = 700) -> tuple[str, str]:
    """Co-occurring chunk if any; else concat of each concept's first chunk."""
    ca = set(concept_chunks.get(a, []))
    cb = set(concept_chunks.get(b, []))
    inter = sorted(ca & cb)
    if inter:
        cid = inter[0]
        return chunk_texts[cid][:max_chars], f"cooccur:{cid}"
    parts = []
    ids = []
    for c, pool in ((a, sorted(ca)), (b, sorted(cb))):
        if pool:
            ids.append(pool[0])
            parts.append(chunk_texts[pool[0]][: max_chars // 2])
    return "\n".join(parts), f"concat:{'+'.join(ids) if ids else 'none'}"


# ── V1: restricted classification at position -1 ─────────────────────

def decode_raw_topk(logits_row, tokenizer, n: int = 15, scan: int = 60):
    """Raw top-k decode without content-word filtering (answers may be
    stoplisted words like 'same'/'different')."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(scan)
    out = []
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        if tok and tok.replace("'", "").isalpha():
            out.append({"token": tok, "prob": round(float(p), 4)})
        if len(out) >= n:
            break
    return out


def v1_classify(raw_words: list[dict]) -> str | None:
    """Aggregate answer-token probabilities per class; return argmax."""
    agg: dict[str, float] = {}
    for w in raw_words:
        lab = V1_CLASS_TOKENS.get(w["token"].lower())
        if lab:
            agg[lab] = agg.get(lab, 0.0) + w["prob"]
    return max(agg, key=agg.get) if agg else None


# ── V2: reversed prefill scan, workspace overlap ─────────────────────

def find_prefill_positions(tokenizer, prompt: str, a: str, b: str
                           ) -> dict[str, int]:
    """Locate A and B first-token positions in the TRAILING prefill
    "The concepts are: {a}, {b}", via char-offset mapping.

    Token-level matching fails here two ways: concept mentions inside the
    chunk context win naive scans, and BPE splits like "t"+"umor" defeat
    prefix matching. Instead we find the char offset of the prefill
    string in the reassembled prompt (rfind — the prefill is the tail)
    and map it back to the token index.
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    texts = [tokenizer.decode([t]) for t in ids]
    offsets = []
    pos = 0
    for t in texts:
        offsets.append(pos)
        pos += len(t)
    full = "".join(texts)

    def token_at(char_idx: int) -> int | None:
        for i in range(len(offsets) - 1, -1, -1):
            if offsets[i] <= char_idx:
                return i
        return None

    anchor = full.rfind(f"{a}, {b}")
    if anchor < 0:
        return {}
    pa = token_at(anchor)
    pb = token_at(anchor + len(a) + len(", "))
    if pa is None or pb is None or pa == pb:
        return {}
    return {a: pa, b: pb}


def decode_corpus_words(logits_row, tokenizer, corpus_words: set[str],
                        n: int = 8, scan: int = 60) -> list[dict]:
    """decode_topk_custom + corpus-word filter (drops BPE garbage like
    'akter'/'nintendo' that survives the STOP filter)."""
    words = decode_topk_custom(logits_row, tokenizer, n=n * 2, scan=scan)
    out = [w for w in words if w["token"].lower() in corpus_words]
    return out[:n]


def v2_overlap(words_a: list[dict], words_b: list[dict]) -> dict:
    sa = {w["token"].lower() for w in words_a}
    sb = {w["token"].lower() for w in words_b}
    # stem-level overlap to be fair to inflections
    sta = {_stem(w) for w in sa}
    stb = {_stem(w) for w in sb}
    shared = sorted((sa & sb) | {w for w in sa if _stem(w) in stb}
                    | {w for w in sb if _stem(w) in sta})
    n = len(shared)
    # fixed rule: >=2 shared → same, 1 → near, 0 → diff
    pred = "same" if n >= 2 else ("near" if n == 1 else "diff")
    union = len(sa | sb) or 1
    return {"shared": shared, "n_shared": n,
            "jaccard": round(len(sa & sb) / union, 3), "pred": pred}


# ── BGE cosine baseline ───────────────────────────────────────────────

def bge_baseline(pairs: list[dict]) -> dict[str, float]:
    os.environ.setdefault("BGE_M3_DEVICE", "cpu")
    from experiments.embed_cache import CachedBgeM3Provider
    embed = CachedBgeM3Provider()
    words = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    vecs = np.asarray(embed.embed(words), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    lut = {w: vecs[i] for i, w in enumerate(words)}
    sims = {}
    for p in pairs:
        sims[f"{p['a']}|||{p['b']}"] = round(float(lut[p["a"]] @ lut[p["b"]]), 4)
    return sims


def bge_predict(cos: float) -> str:
    # fixed thresholds: near-duplicate word forms embed ~0.85+
    if cos >= 0.80:
        return "same"
    if cos >= 0.55:
        return "near"
    return "diff"


# ── main run ──────────────────────────────────────────────────────────

def accuracy(pairs: list[dict], key) -> float:
    n = sum(1 for p in pairs if key(p) == p["label"])
    return round(n / len(pairs), 3) if pairs else 0.0


def run(lens, lens_model, tokenizer, pairs: list[dict], concept_chunks: dict,
        chunk_texts: dict, smoke: bool) -> dict:
    layer = lens.source_layers[-1]
    corpus_words = {
        m.group().lower()
        for text in chunk_texts.values()
        for m in re.finditer(r"[a-zA-Z]{4,}", text)
    }
    results = []
    t0 = time.perf_counter()

    for i, pair in enumerate(pairs):
        a, b, label = pair["a"], pair["b"], pair["label"]
        context, ctx_src = pick_context(a, b, concept_chunks, chunk_texts)

        # --- V1: restricted classification at position -1 ---
        # Prefill "Answer:" instead of "They are" — the latter's language
        # prior locks the readout onto "related" (smoke: 4/6 pairs).
        user_v1 = (
            f"This text discusses {a} and {b}. Are \"{a}\" and \"{b}\" the "
            f"same concept (word-form variants like singular/plural count "
            f"as the same), similar but distinct concepts, or unrelated "
            f"concepts? Answer with exactly one word: same, similar, or "
            f"unrelated.\n\n{context}"
        )
        prompt_v1 = wrap_chat(tokenizer, user_v1, "Answer:")
        logits_v1, _, _ = lens.apply(
            lens_model, prompt_v1, layers=[layer], positions=[-1],
            max_seq_len=1024)
        raw = decode_raw_topk(logits_v1[layer][0], tokenizer, n=15)
        v1_pred = v1_classify(raw)

        # --- V2: list-style prefill scan (phase39 shape) ---
        user_v2 = f"What concepts does this text discuss?\n\n{context}"
        prompt_v2 = wrap_chat(tokenizer, user_v2,
                              f"The concepts are: {a}, {b}")
        pos_map = find_prefill_positions(tokenizer, prompt_v2, a, b)
        v2 = {"shared": [], "n_shared": 0, "jaccard": 0.0, "pred": None,
              "words_a": [], "words_b": [], "positions": pos_map}
        if a in pos_map and b in pos_map and pos_map[a] != pos_map[b]:
            positions = sorted({pos_map[a], pos_map[b]})
            logits_v2, _, _ = lens.apply(
                lens_model, prompt_v2, layers=[layer], positions=positions,
                max_seq_len=1024)
            pos_to_idx = {p: j for j, p in enumerate(positions)}
            wa = decode_corpus_words(
                logits_v2[layer][pos_to_idx[pos_map[a]]], tokenizer,
                corpus_words, n=8)
            wb = decode_corpus_words(
                logits_v2[layer][pos_to_idx[pos_map[b]]], tokenizer,
                corpus_words, n=8)
            v2.update(v2_overlap(wa, wb))
            v2["words_a"] = [w["token"] for w in wa]
            v2["words_b"] = [w["token"] for w in wb]
        else:
            v2["positions"] = pos_map

        results.append({
            "a": a, "b": b, "label": label, "context_src": ctx_src,
            "v1_raw": raw, "v1_pred": v1_pred,
            "v2": v2,
        })
        if smoke or i < 5:
            print(f"\n  [{label}] {a} / {b}   ({ctx_src})")
            print(f"    V1 raw: {[(w['token'], w['prob']) for w in raw[:6]]}"
                  f" → {v1_pred}")
            print(f"    V2 pos: {pos_map}  shared={v2['shared']}"
                  f" → {v2['pred']}")
            print(f"    V2 A-words: {v2['words_a'][:6]}")
            print(f"    V2 B-words: {v2['words_b'][:6]}")

    gpu_s = time.perf_counter() - t0

    # --- BGE baseline (CPU) ---
    print("\n  [baseline] bge-m3 cosine (CPU)...")
    sims = bge_baseline(pairs)
    for r in results:
        cos = sims[f"{r['a']}|||{r['b']}"]
        r["bge_cos"] = cos
        r["bge_pred"] = bge_predict(cos)

    summary = {
        "n_pairs": len(results),
        "label_counts": {lab: sum(1 for r in results if r["label"] == lab)
                         for lab in ("same", "near", "diff")},
        "v1_accuracy": accuracy(results, lambda r: r["v1_pred"]),
        "v1_unparsed": sum(1 for r in results if r["v1_pred"] is None),
        "v2_accuracy": accuracy(results, lambda r: r["v2"]["pred"]),
        "v2_unlocated": sum(1 for r in results if r["v2"]["pred"] is None),
        "bge_accuracy": accuracy(results, lambda r: r["bge_pred"]),
        "gpu_time_s": round(gpu_s, 1),
    }
    return {"summary": summary, "results": results}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0,
                    help="run only N pairs (2 per class), no JSON output")
    ap.add_argument("--rebuild-pairs", action="store_true",
                    help="re-annotate the pair set via DeepSeek")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    concept_chunks = cache["concept_chunks"]
    concepts = sorted(concept_chunks)

    pair_data = load_pairs(concepts, args.rebuild_pairs)
    pairs = pair_data["pairs"]
    if args.smoke:
        by_label = {lab: [p for p in pairs if p["label"] == lab]
                    for lab in ("same", "near", "diff")}
        per = max(1, args.smoke // 3)
        pairs = [p for lab in ("same", "near", "diff")
                 for p in by_label[lab][:per]]
        print(f"[smoke] {len(pairs)} pairs")

    # chunk texts for context
    from experiments.phase4_dig_graphragbench import load_graphrag_bench
    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    chunk_texts = dict(corpus)

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print("\n[2/2] Running disambiguation readout...")
    out = run(lens, lens_model, tokenizer, pairs, concept_chunks,
              chunk_texts, smoke=bool(args.smoke))

    s = out["summary"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({s['n_pairs']} pairs, {s['label_counts']})")
    print(f"  V1 (dual-concern -1): acc={s['v1_accuracy']}"
          f"  (unparsed: {s['v1_unparsed']})")
    print(f"  V2 (prefill overlap): acc={s['v2_accuracy']}"
          f"  (unlocated: {s['v2_unlocated']})")
    print(f"  BGE cosine baseline:  acc={s['bge_accuracy']}")
    print(f"  GPU time: {s['gpu_time_s']}s")

    if not args.smoke:
        out["model"] = cand["name"]
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
