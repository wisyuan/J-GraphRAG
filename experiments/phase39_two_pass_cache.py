"""Phase 39: 两步概念缓存——Phase 25 提取 + Phase 35 角色扩展。

Pass 1（概念提取，复用 Phase 31）：
  extract_concepts_full_pipeline() —— concern prompt + 全层 depth gradient
  + 三重过滤 + BM25 BPE 补全。行为与 concept_cache_{domain}_full.json 完全一致。

Pass 2（角色扩展，Phase 35 方法，每 chunk 1 次额外 forward）：
  构造反转 prompt：
    User: "What concepts does this text discuss?\n\n{chunk[:400]}"
    Assistant prefill: "The concepts are: {Pass 1 concepts (≤8)}"
  在 prefill 区域定位每个概念的位置，一次 forward 读所有概念位置的
  workspace，decode top-8 → 过滤 → top-5 角色词。

向量收集（为后续接地检验省一次 GPU pass）：
  - ws_vec：prefill 概念位置的残差向量，经 lens.transport(J_l @ h) 映射到
    final-layer 基（unembedding 前空间，d_model=3584）。jlens 的
    lens.apply() 只返回 unembed 后的 logits、不暴露残差，因此这里直接用
    jlens.hooks.ActivationRecorder 抓残差再 lens.transport()——这正是
    lens.apply 内部做的事（lens.py 的 apply: record → transport → unembed），
    但同一次 forward 同时产出 logits（角色解码）和 transported 残差
    （向量收集），只需 1 次 forward。
  - wu_vec：概念第一个 BPE 碎片 token 的 W_U（lm_head）行向量
    （参考 Phase 38 get_bpe_fragment_vectors，只取第一个碎片）。

产出（experiments/m6/concept_cache/）：
  - concept_cache_{domain}_twopass.json：chunks + roles + 频率统计
  - concept_vecs_{domain}.npz：每唯一概念的 wu_vec/ws_vec/count + concepts 索引

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase39_two_pass_cache [--domain medical] [--max-chunks 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase31_full_pipeline_cache import extract_concepts_full_pipeline
from experiments.phase35_prefill_position_scan import STOP, decode_topk_custom
from experiments.phase18_centroid_hierarchy import build_corpus_word_set
from experiments.concept_quality import build_corpus_term_freq, _get_wordnet_nouns
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6" / "concept_cache"
EXP.mkdir(parents=True, exist_ok=True)

MAX_PREFILL_CONCEPTS = 8
N_ROLE_DECODE = 8   # decode top-8 per concept position
N_ROLE_KEEP = 5     # keep top-5 after filtering

# Role-specific blocklist: generic template words the model emits at prefill
# positions regardless of document content (Phase 35-37's "template cloze"
# failure mode). Distinct from phase35's STOP (which targets prompt words).
ROLE_STOP = {
    "type", "types", "aspect", "aspects", "basic", "basics", "part", "parts",
    "kind", "kinds", "form", "forms", "thing", "things", "way", "ways",
    "overview", "summary", "example", "examples", "detail", "details",
    "role", "roles", "feature", "features", "element", "elements",
    "component", "components", "category", "categories", "variety", "varieties",
}


def _stem(word: str) -> str:
    """Crude singular form for dedupe keys only (not for output)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def find_concept_positions(tokenizer, prompt: str,
                           concepts: list[str]) -> dict[str, int]:
    """Locate each concept's token position in the prefill region.

    Reproduces Phase 35's two-stage matching: exact/normalized match first,
    then first-BPE-fragment prefix match (for multi-token concepts).
    Returns {concept: position} (position = first fragment token index).
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]) for t in ids]

    # Find prefill start (right after "concepts are")
    prefill_start = None
    for i, _t in enumerate(token_texts):
        if "concepts are" in "".join(token_texts[max(0, i - 3):i + 1]).lower():
            prefill_start = i + 1
            break
    if prefill_start is None:
        return {}

    concept_positions: dict[str, int] = {}
    # Pass A: exact or containment match
    for i in range(prefill_start, len(token_texts)):
        tok_text = token_texts[i].strip().rstrip(",").lower()
        for c in concepts:
            if c in concept_positions:
                continue
            if tok_text == c.lower() or c.lower() in tok_text:
                concept_positions[c] = i
                break
    # Pass B: first-fragment prefix match (multi-token concepts)
    for i in range(prefill_start, len(token_texts)):
        tok = token_texts[i].strip().lower()
        for c in concepts:
            if c in concept_positions:
                continue
            cl = c.lower()
            if tok == cl or (len(cl) > 3 and cl.startswith(tok) and len(tok) >= 3):
                concept_positions[c] = i
                break
    return concept_positions


def pass2_role_expansion(lens, lens_model, tokenizer, chunk_text: str,
                         concepts: list[str], corpus_words: set[str],
                         layer: int
                         ) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Phase 35 reversed-prefill role expansion in ONE forward pass.

    Returns (roles, ws_vecs):
      roles:   {concept: [role words]} — top-5 filtered role words
      ws_vecs: {concept: np.ndarray[d_model]} — L2-normalized transported
               residual (J_l @ h) at the concept's prefill position

    jlens note: lens.apply() returns only unembedded logits; to get the
    residual itself we hook the layer with jlens.hooks.ActivationRecorder and
    call lens.transport(residual, layer) — identical math to what apply()
    does internally (record → transport → unembed), but a single forward
    yields both the logits (role decoding) and the residuals (vectors).
    """
    roles: dict[str, list[str]] = {}
    ws_vecs: dict[str, np.ndarray] = {}
    if not concepts:
        return roles, ws_vecs

    prefill_concepts = concepts[:MAX_PREFILL_CONCEPTS]
    concept_str = ", ".join(prefill_concepts)
    user_msg = f"What concepts does this text discuss?\n\n{chunk_text[:400]}"
    prefill = f"The concepts are: {concept_str}"

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    concept_positions = find_concept_positions(tokenizer, prompt, prefill_concepts)
    if not concept_positions:
        return roles, ws_vecs

    positions_to_read = sorted(set(concept_positions.values()))

    # One forward pass: record residual at `layer`, then transport + unembed
    from jlens.hooks import ActivationRecorder
    input_ids = lens_model.encode(prompt, max_length=1024)
    seq_len = input_ids.shape[1]
    positions_to_read = [p for p in positions_to_read if p < seq_len]
    if not positions_to_read:
        return roles, ws_vecs

    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as rec:
        lens_model.forward(input_ids)
    # rec.activations[layer]: [1, seq_len, d_model]
    resid = rec.activations[layer][0, positions_to_read].float()  # [n_pos, d]
    transported = lens.transport(resid, layer)      # [n_pos, d_model] final-layer basis
    logits = lens_model.unembed(transported).float().cpu()  # [n_pos, vocab]

    pos_to_idx = {p: i for i, p in enumerate(positions_to_read)}
    chunk_concepts_lower = {c.lower() for c in prefill_concepts}
    chunk_concept_stems = {_stem(c) for c in prefill_concepts}

    for concept, pos in sorted(concept_positions.items(), key=lambda x: x[1]):
        idx = pos_to_idx.get(pos)
        if idx is None:
            continue

        # --- role words: decode top-8, filter, keep top-5 ---
        candidates = decode_topk_custom(logits[idx], tokenizer,
                                        n=N_ROLE_DECODE, scan=40)
        concept_roles: list[str] = []
        seen_stems: set[str] = set()
        for w in candidates:
            tok = w["token"]
            low = tok.lower()
            stem = _stem(tok)
            if low in chunk_concepts_lower or stem in chunk_concept_stems:
                continue                         # not the concept itself / sibling concept
            if low in ROLE_STOP or stem in ROLE_STOP:
                continue                         # generic template word
            if low not in corpus_words:          # must be a real corpus word
                continue
            if stem in seen_stems:               # inflection dedupe (Types vs Type)
                continue
            seen_stems.add(stem)
            concept_roles.append(tok)
            if len(concept_roles) >= N_ROLE_KEEP:
                break
        roles[concept] = concept_roles

        # --- workspace vector: L2-normalized transported residual ---
        vec = transported[idx].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            ws_vecs[concept] = vec / norm

    return roles, ws_vecs


def get_wu_first_fragment_vec(concept: str, lm_head_wu: np.ndarray,
                              tokenizer) -> np.ndarray | None:
    """W_U row vector of the concept's first BPE fragment (L2-normalized).

    Follows Phase 38's get_bpe_fragment_vectors but uses only the first
    fragment (core semantic direction).
    """
    token_ids = tokenizer.encode(concept, add_special_tokens=False)
    if not token_ids:
        return None
    vec = lm_head_wu[token_ids[0]].astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def run_twopass_extraction(lens, lens_model, tokenizer, model, domain: str,
                           max_chunks: int = 0, verbose_samples: int = 3) -> None:
    """Two-pass extraction over the corpus; writes JSON cache + NPZ vectors."""
    print(f"\n  [{domain}] Loading corpus...")
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())
    if max_chunks > 0:
        chunk_items = chunk_items[:max_chunks]

    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    chunk_texts_list = [chunk_text_map[cid] for cid in chunk_ids]
    print(f"  [{domain}] {len(chunk_ids)} chunks to process")

    print(f"  [{domain}] Building corpus resources...")
    corpus_words = build_corpus_word_set(chunk_texts_list)
    corpus_freq = build_corpus_term_freq(chunk_texts_list)
    wn_nouns = _get_wordnet_nouns()
    print(f"  [{domain}] Corpus words: {len(corpus_words)}, "
          f"WN nouns: {len(wn_nouns)}")

    all_layers = lens.source_layers
    last_layer = all_layers[-1]

    cache = {
        "domain": domain,
        "model": detect_model()["name"],
        "pipeline": "twopass (phase25 extraction + phase35 role expansion)",
        "n_chunks": len(chunk_ids),
        "chunks": {},
    }

    # Vector accumulation per unique concept (lowercased key)
    ws_sum: dict[str, np.ndarray] = {}
    ws_count: dict[str, int] = defaultdict(int)

    t_start = time.perf_counter()
    extraction_times = []
    n_located_total = 0
    n_concepts_total = 0

    for i, cid in enumerate(chunk_ids):
        t0 = time.perf_counter()

        # Pass 1: Phase 25 full pipeline (identical to phase31 cache)
        p1 = extract_concepts_full_pipeline(
            lens, lens_model, tokenizer,
            chunk_text_map[cid], all_layers,
            corpus_words, corpus_freq, wn_nouns)
        concepts = p1["concepts"]

        # Pass 2: role expansion (1 extra forward)
        roles, ws_vecs = pass2_role_expansion(
            lens, lens_model, tokenizer, chunk_text_map[cid],
            concepts, corpus_words, last_layer)

        t1 = time.perf_counter()

        # Accumulate workspace vectors
        for concept, vec in ws_vecs.items():
            key = concept.lower()
            if key in ws_sum:
                ws_sum[key] += vec
            else:
                ws_sum[key] = vec.copy()
            ws_count[key] += 1
        n_located_total += len(ws_vecs)
        n_concepts_total += len(concepts)

        cache["chunks"][cid] = {
            "concepts": concepts,
            "roles": roles,
            "n_concepts": len(concepts),
            "text_excerpt": chunk_text_map[cid][:100],
            "extraction_time_s": round(t1 - t0, 4),
        }
        extraction_times.append(t1 - t0)

        if i < verbose_samples:
            print(f"\n    --- sample chunk {i} ({cid}) ---")
            print(f"    text: {chunk_text_map[cid][:80]}...")
            print(f"    concepts: {concepts}")
            for c, rs in roles.items():
                print(f"      {c}: {rs}")

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (i + 1) * (len(chunk_ids) - i - 1)
            print(f"    [{domain}] {i + 1}/{len(chunk_ids)} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    t_total = time.perf_counter() - t_start
    cache["extraction_time_s"] = round(t_total, 1)
    cache["avg_per_chunk_s"] = round(float(np.mean(extraction_times)), 4)

    # Stats
    all_concepts = []
    all_roles = []
    for data in cache["chunks"].values():
        all_concepts.extend(data["concepts"])
        for rs in data["roles"].values():
            all_roles.extend(rs)
    concept_freq = Counter(c.lower() for c in all_concepts)
    role_freq = Counter(r.lower() for r in all_roles)
    cache["concept_frequency"] = dict(concept_freq.most_common(40))
    cache["n_unique_concepts"] = len(concept_freq)
    cache["role_frequency"] = dict(role_freq.most_common(40))
    cache["n_unique_roles"] = len(role_freq)

    concept_chunks = defaultdict(list)
    for cid in chunk_ids:
        for concept in cache["chunks"][cid]["concepts"]:
            concept_chunks[concept.lower()].append(cid)
    cache["concept_chunks"] = dict(concept_chunks)

    print(f"\n  [{domain}] Done: {t_total:.0f}s "
          f"({cache['avg_per_chunk_s']}s/chunk)")
    print(f"  [{domain}] Unique concepts: {cache['n_unique_concepts']}, "
          f"unique roles: {cache['n_unique_roles']}")
    print(f"  [{domain}] Position located: {n_located_total}/{n_concepts_total} "
          f"concept instances")
    print(f"  [{domain}] Top concepts: {list(concept_freq.most_common(10))}")
    print(f"  [{domain}] Top roles: {list(role_freq.most_common(10))}")

    # --- Save JSON cache ---
    out_json = EXP / f"concept_cache_{domain}_twopass.json"
    out_json.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    print(f"  [{domain}] Saved cache to {out_json}")

    # --- Save NPZ vectors ---
    # Union of concepts that have a ws_vec and/or a W_U vector
    lm_head_wu = (model.get_output_embeddings().weight
                  .detach().float().cpu().numpy())  # [vocab, 3584]
    uniq_concepts = sorted(set(concept_freq) | set(ws_sum))
    concepts_arr, wu_rows, ws_rows, counts = [], [], [], []
    d_model = lm_head_wu.shape[1]
    for c in uniq_concepts:
        wu = get_wu_first_fragment_vec(c, lm_head_wu, tokenizer)
        n = ws_count.get(c, 0)
        if wu is None and n == 0:
            continue
        concepts_arr.append(c)
        wu_rows.append(wu if wu is not None else np.zeros(d_model, np.float32))
        ws_rows.append(ws_sum[c] / n if n > 0 else np.zeros(d_model, np.float32))
        counts.append(n)

    out_npz = EXP / f"concept_vecs_{domain}.npz"
    np.savez(out_npz,
             concepts=np.array(concepts_arr),
             wu_vec=np.stack(wu_rows).astype(np.float32),
             ws_vec=np.stack(ws_rows).astype(np.float32),
             count=np.array(counts, dtype=np.int64))
    print(f"  [{domain}] Saved vectors to {out_npz} "
          f"({len(concepts_arr)} concepts, d={d_model})")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all", choices=["medical", "novel", "all"])
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="0 = all chunks")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase 39: Two-pass concept cache")
    print("  Pass 1: Phase 25 extraction (via phase31 full pipeline)")
    print("  Pass 2: Phase 35 role expansion (reversed prefill scan)")
    print("=" * 60)

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print("\n[2/2] Two-pass extraction...")
    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]
    for domain in domains:
        run_twopass_extraction(lens, lens_model, tokenizer, model, domain,
                               max_chunks=args.max_chunks)


if __name__ == "__main__":
    main()
