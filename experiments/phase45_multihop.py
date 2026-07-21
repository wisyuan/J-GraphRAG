"""Phase 45: 多跳路径——链式 prefill 能否读出二跳关系。

问题：已知关系图（relations_medical.json，248 边）中的路径 A→B→C
（B 度 >=2），链式 prefill 能否利用已知一跳关系词 rel1(A,B) 读出
二跳关系 rel2(B,C)。

样本：从关系图选 A→B→C 路径（B 度 >=2，优先高 prob 边、B 多样化），
>=15 条。

变体（每条路径 3 次 forward，同一上下文）：
  V1 直接读出（对照，已有能力）：phase27 build_relation_prompt 直接读
     B→C 的关系词（user 问 B,C 关系 + 共现 chunk，prefill
     "The relationship between B and C is" → position -1）。
  V2s 严格链式：user 提及三概念 + chunk，assistant prefill
     "{A} {rel1} {B}"（rel1 为图中已知边词）→ 读 position -1
     （预测链上下一个词）。
  V2a 锚定链式：同一 user，prefill "{A} {rel1} {B}. The relationship
     between {B} and {C} is" → 读 position -1。已知一跳作为先验填入，
     读出二跳（phase37 V3 prior+scan 精神的链式化）。

判决：
  - 一致率：V2 与 V1 的关系词一致（top-1：同 stem 或 DeepSeek 判定同义；
    top-5：stem 重合）。
  - 图一致性：各变体 top-5 与图中已知 B→C 边词（rel_bc_graph）的
    stem 重合率。
  - 正确率：DeepSeek judge 判定 top-1 读出词是否准确描述文中 B→C
    关系（对碎片词如 "anatom" 偏严，如实记录）。

产出：experiments/m6/phase45_multihop.json

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase45_multihop            # 全量
    python -m experiments.phase45_multihop --smoke 4  # 冒烟（不落盘）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase27_relation_readout import (
    build_relation_prompt, decode_topk,
)
from experiments.phase37_relation_prompt_variants import wrap_chat
from experiments.phase39_two_pass_cache import _stem

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE = EXP / "concept_cache" / "concept_cache_medical_twopass.json"
RELATIONS = EXP / "concept_cache" / "relations_medical.json"
OUT_PATH = EXP / "phase45_multihop.json"

N_PATHS = 18


# ── path sampling from the relation graph ─────────────────────────────

def sample_paths(n: int = N_PATHS) -> list[dict]:
    edges = json.loads(RELATIONS.read_text())["edges"]
    # undirected adjacency: neighbor → (relation, prob)
    adj: dict[str, dict[str, tuple[str, float]]] = defaultdict(dict)
    for e in edges:
        a, b, rel, prob = e["concept_a"], e["concept_b"], e["relation"], e["prob"]
        adj[a][b] = (rel, prob)
        adj[b][a] = (rel, prob)

    paths = []
    used_b: set[str] = set()
    # B sorted by degree desc; neighbors sorted by edge prob desc
    for b in sorted(adj, key=lambda x: -len(adj[x])):
        nbrs = sorted(adj[b].items(), key=lambda kv: -kv[1][1])
        if len(nbrs) < 2 or b in used_b:
            continue
        (a, (rel1, p1)), (c, (rel2, p2)) = nbrs[0], nbrs[1]
        paths.append({"a": a, "b": b, "c": c, "rel1": rel1,
                      "rel_bc_graph": rel2,
                      "prob_ab": p1, "prob_bc": p2})
        used_b.add(b)
        if len(paths) >= n:
            break
    return paths


def pick_context(terms: list[str], concept_chunks: dict,
                 chunk_texts: dict, max_chars: int = 700) -> tuple[str, str]:
    """Chunk containing all terms if possible; else pairwise; else concat."""
    sets = [set(concept_chunks.get(t, [])) for t in terms]
    common = sorted(set.intersection(*sets)) if all(sets) else []
    if common:
        return chunk_texts[common[0]][:max_chars], f"all:{common[0]}"
    pair = sorted(sets[1] & sets[2]) if len(sets) == 3 else []
    if pair:
        return chunk_texts[pair[0]][:max_chars], f"bc:{pair[0]}"
    parts, ids = [], []
    for t, s in zip(terms, sets):
        if s:
            cid = sorted(s)[0]
            ids.append(cid)
            parts.append(chunk_texts[cid][: max_chars // len(terms)])
    return "\n".join(parts), f"concat:{'+'.join(ids) if ids else 'none'}"


# ── readouts ──────────────────────────────────────────────────────────

def read_word(lens, lens_model, tokenizer, layer: int, prompt: str,
              n: int = 5) -> list[dict]:
    logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                              positions=[-1], max_seq_len=1024)
    return decode_topk(logits[layer][0], tokenizer, n=n)


# ── DeepSeek judges ───────────────────────────────────────────────────

def judge_synonym(llm, w1: str, w2: str, b: str, c: str) -> bool | None:
    prompt = (
        f"Two words describing the relationship between the medical "
        f"concepts \"{b}\" and \"{c}\": \"{w1}\" and \"{w2}\".\n"
        f"Do they describe the same kind of relationship here? "
        f"Answer YES or NO."
    )
    try:
        msg = llm.complete(prompt, max_tokens=8)
        resp = (msg.content if hasattr(msg, "content") else str(msg)).strip().upper()
        return resp.startswith("YES")
    except Exception:
        return None


def judge_relation(llm, word: str, b: str, c: str, chunk: str) -> bool | None:
    prompt = (
        f"Text:\n{chunk[:600]}\n\n"
        f"In this text, is the relationship between \"{b}\" and \"{c}\" "
        f"accurately described by the word \"{word}\"? Answer YES or NO."
    )
    try:
        msg = llm.complete(prompt, max_tokens=8)
        resp = (msg.content if hasattr(msg, "content") else str(msg)).strip().upper()
        return resp.startswith("YES")
    except Exception:
        return None


# ── main run ──────────────────────────────────────────────────────────

def run(lens, lens_model, tokenizer, llm, paths: list[dict],
        concept_chunks: dict, chunk_texts: dict, smoke: bool) -> dict:
    layer = lens.source_layers[-1]
    results = []
    t0 = time.perf_counter()

    for i, p in enumerate(paths):
        a, b, c, rel1 = p["a"], p["b"], p["c"], p["rel1"]
        context, ctx_src = pick_context([a, b, c], concept_chunks, chunk_texts)

        # V1: direct B→C readout (phase27, existing capability)
        prompt_v1 = build_relation_prompt(context, b, c, tokenizer)
        v1_words = read_word(lens, lens_model, tokenizer, layer, prompt_v1)

        # shared user message for V2 variants
        user_v2 = (
            f"This text discusses {a}, {b}, and {c}. In this text, "
            f"{a} {rel1} {b}.\n\n{context}"
        )
        # V2-strict: chain prefill, read next word
        prompt_v2s = wrap_chat(tokenizer, user_v2, f"{a} {rel1} {b}")
        v2s_words = read_word(lens, lens_model, tokenizer, layer, prompt_v2s)

        # V2-anchored: known hop as prior, read second hop
        prompt_v2a = wrap_chat(
            tokenizer, user_v2,
            f"{a} {rel1} {b}. The relationship between {b} and {c} is")
        v2a_words = read_word(lens, lens_model, tokenizer, layer, prompt_v2a)

        results.append({
            **p, "context_src": ctx_src, "chunk": context[:600],
            "v1_words": v1_words,
            "v2s_words": v2s_words,
            "v2a_words": v2a_words,
            "v1_rel": v1_words[0]["token"] if v1_words else None,
            "v2s_rel": v2s_words[0]["token"] if v2s_words else None,
            "v2a_rel": v2a_words[0]["token"] if v2a_words else None,
        })
        if smoke or i < 5:
            print(f"\n  [{a} -{rel1}-> {b} -?-> {c}]  graph B→C: "
                  f"{p['rel_bc_graph']}  ({ctx_src})")
            print(f"    V1 direct B→C:  {[w['token'] for w in v1_words]}")
            print(f"    V2s chain:      {[w['token'] for w in v2s_words]}")
            print(f"    V2a anchored:   {[w['token'] for w in v2a_words]}")

    gpu_s = time.perf_counter() - t0

    # DeepSeek judges (parallel)
    t1 = time.perf_counter()

    def stem_in(words: list[dict], target: str | None) -> bool:
        if not target:
            return False
        return any(_stem(w["token"]) == _stem(target) for w in words)

    def _judge(r):
        b, c = r["b"], r["c"]
        grel = r["rel_bc_graph"]
        v1w = r["v1_rel"]
        # top-5 stem-overlap metrics (no API cost)
        r["v1_match_graph_top5"] = stem_in(r["v1_words"], grel)
        for key in ("v2s", "v2a"):
            r[f"{key}_match_v1_top5"] = stem_in(r[f"{key}_words"], v1w)
            r[f"{key}_match_graph_top5"] = stem_in(r[f"{key}_words"], grel)
        # top-1 synonym judges
        for key in ("v2s", "v2a"):
            w = r[f"{key}_rel"]
            if not w:
                r[f"{key}_match_v1"] = None
                r[f"{key}_correct"] = None
                continue
            if v1w and _stem(w) == _stem(v1w):
                r[f"{key}_match_v1"] = True
            elif v1w:
                r[f"{key}_match_v1"] = judge_synonym(llm, w, v1w, b, c)
            else:
                r[f"{key}_match_v1"] = None
            r[f"{key}_correct"] = judge_relation(llm, w, b, c, r["chunk"])
        r["v1_correct"] = (judge_relation(llm, r["v1_rel"], b, c, r["chunk"])
                           if r["v1_rel"] else None)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_judge, results))
    ds_s = time.perf_counter() - t1

    def rate(key):
        vs = [r[key] for r in results if r[key] is not None]
        return round(sum(vs) / len(vs), 3) if vs else None

    summary = {
        "n_paths": len(results),
        "v1_match_graph_top5": rate("v1_match_graph_top5"),
        "v2s_match_v1": rate("v2s_match_v1"),
        "v2a_match_v1": rate("v2a_match_v1"),
        "v2s_match_v1_top5": rate("v2s_match_v1_top5"),
        "v2a_match_v1_top5": rate("v2a_match_v1_top5"),
        "v2s_match_graph_top5": rate("v2s_match_graph_top5"),
        "v2a_match_graph_top5": rate("v2a_match_graph_top5"),
        "v2s_correct": rate("v2s_correct"),
        "v2a_correct": rate("v2a_correct"),
        "v1_correct": rate("v1_correct"),
        "gpu_time_s": round(gpu_s, 1),
        "deepseek_time_s": round(ds_s, 1),
    }
    return {"summary": summary, "results": results}


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--n-paths", type=int, default=N_PATHS)
    args = ap.parse_args()

    paths = sample_paths(args.smoke or args.n_paths)
    print(f"paths ({len(paths)}):")
    for p in paths:
        print(f"  {p['a']} -{p['rel1']}-> {p['b']} -{p['rel_bc_graph']}-> {p['c']}")

    cache = json.loads(CACHE.read_text())
    concept_chunks = cache["concept_chunks"]

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

    print("\n[2/2] Running multi-hop readout...")
    out = run(lens, lens_model, tokenizer, llm, paths, concept_chunks,
              chunk_texts, smoke=bool(args.smoke))

    s = out["summary"]
    print(f"\n{'=' * 60}")
    print(f"SUMMARY ({s['n_paths']} paths)")
    print(f"  V1 direct: correct={s['v1_correct']}  match-graph(top5)={s['v1_match_graph_top5']}  (reference)")
    print(f"  V2s strict:   match-V1={s['v2s_match_v1']} (top5={s['v2s_match_v1_top5']})"
          f"  match-graph(top5)={s['v2s_match_graph_top5']}  correct={s['v2s_correct']}")
    print(f"  V2a anchored: match-V1={s['v2a_match_v1']} (top5={s['v2a_match_v1_top5']})"
          f"  match-graph(top5)={s['v2a_match_graph_top5']}  correct={s['v2a_correct']}")
    print(f"  GPU: {s['gpu_time_s']}s, DeepSeek: {s['deepseek_time_s']}s")

    if not args.smoke:
        out["model"] = cand["name"]
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n  saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
