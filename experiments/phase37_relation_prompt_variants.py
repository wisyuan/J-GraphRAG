"""Phase 37: 关系 prefill prompt 变体实验。

Phase 36 的 prefill scan 失败了（3/8 vs 字典 8/8），但可能是 prompt 设计问题。
本实验测试多种 prompt 设计，找到让 prefill position scan 发挥优势的方向。

5 种 prompt 变体：
  V1 字典式（Phase 27 基线）: "The relationship between {A} and {B} is"
  V2 关切+填空: "In this text, {A} [verb] {B}" → 读 [verb] 位置
  V3 字典先验+prefill: 先用字典关系填入，再读 workspace 验证
     "In this text, {A} {dict_relation} {B}. This means {A}"
     → 读 {dict_relation} 位置的 workspace
  V4 功能问句: "How does {A} affect {B}? {A}" → 读 {A} 后的位置
  V5 因果链: "{A} leads to {B} through" → 读末尾位置

关键设计思路：
  V3 最有潜力——用 Phase 27 的字典关系作为先验填入 prefill，
  然后读该位置的 workspace 看是否能细化/修正。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase37_relation_prompt_variants
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase27_relation_readout import decode_topk, STOP_REL
from experiments.phase28_relation_graph import RELATION_TYPES

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def wrap_chat(tokenizer, user_msg, prefill):
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


def find_positions(tokenizer, prompt, target_words):
    """Find token positions of target words in prompt."""
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    token_texts = [tokenizer.decode([token_ids[0, i].item()]) for i in range(token_ids.shape[1])]

    positions = {}
    for target in target_words:
        tl = target.lower()
        for i, t in enumerate(token_texts):
            if t.strip().lower() == tl or (len(tl) >= 4 and tl.startswith(t.strip().lower()) and len(t.strip()) >= 3):
                if target not in positions:
                    positions[target] = i
                break
    return positions


def read_at_positions(lens, lens_model, tokenizer, prompt, positions, layer):
    """Read workspace at specified positions."""
    results = {}
    try:
        lens_logits, _, _ = lens.apply(
            lens_model, prompt, layers=[layer],
            positions=positions, max_seq_len=512)
        for i, pos in enumerate(positions):
            if layer in lens_logits:
                words = decode_topk(lens_logits[layer][i], tokenizer, n=10, scan=50)
                results[pos] = words
    except Exception as e:
        print(f"      lens.apply error: {e}")
    return results


def run_variants(lens, lens_model, tokenizer):
    print("Phase 37: Relation prompt variants")
    print(f"{'='*70}")

    # Load corpus
    try:
        from experiments.phase4_dig_graphragbench import load_graphrag_bench
        corpus, _ = load_graphrag_bench("medical", max_queries=1)
        chunks = list(corpus.values())
    except Exception:
        from experiments.corpus_loader import load_beir_fine_records
        records = load_beir_fine_records("nfcorpus", max_docs=200)
        chunks = [r[1] for r in records]

    layer = lens.source_layers[-1]

    TEST_PAIRS = [
        ("cancer", "chemotherapy", "treatment"),
        ("cancer", "surgery", "treatment"),
        ("blood", "heart", "pumps/circulation"),
        ("insulin", "diabetes", "regulates"),
        ("calcium", "bones", "builds"),
        ("diagnosis", "treatment", "precedes"),
        ("smoking", "cancer", "causes"),
        ("tumor", "growth", "causes"),
    ]

    all_results = []

    for concept_a, concept_b, expected in TEST_PAIRS:
        # Find document
        doc_text = ""
        for chunk in chunks:
            cl = chunk.lower()
            if concept_a in cl and concept_b in cl:
                doc_text = chunk
                break
        if not doc_text:
            for chunk in chunks:
                if concept_a in chunk.lower():
                    doc_text = chunk
                    break
        if not doc_text:
            doc_text = chunks[0]

        print(f"\n{'='*60}")
        print(f"[{concept_a} + {concept_b}] expected: {expected}")
        print(f"{'='*60}")

        pair_data = {"concept_a": concept_a, "concept_b": concept_b, "expected": expected}

        # === V1: Dictionary baseline (Phase 27) ===
        user_v1 = (f"This text discusses {concept_a} and {concept_b}. "
                   f"What is the relationship between {concept_a} and {concept_b}? "
                   f"Answer with one word.\n\n{doc_text[:500]}")
        prefill_v1 = f"The relationship between {concept_a} and {concept_b} is"
        prompt_v1 = wrap_chat(tokenizer, user_v1, prefill_v1)
        ws_v1 = read_at_positions(lens, lens_model, tokenizer, prompt_v1, [-1], layer)
        v1_words = ws_v1.get(-1, [])
        v1_known = [w["token"] for w in v1_words if w["token"].lower() in RELATION_TYPES]
        print(f"  V1 Dictionary: {v1_known or '(none)'}  top3: {[w['token'] for w in v1_words[:3]]}")

        # Get dictionary relation for V3
        dict_rel = v1_known[0] if v1_known else "affects"

        # === V2: Blank-fill with explicit verb slot ===
        user_v2 = f"What does {concept_a} do to {concept_b} in this text?\n\n{doc_text[:400]}"
        prefill_v2 = f"In this text, {concept_a}"
        prompt_v2 = wrap_chat(tokenizer, user_v2, prefill_v2)
        ws_v2 = read_at_positions(lens, lens_model, tokenizer, prompt_v2, [-1], layer)
        v2_words = ws_v2.get(-1, [])
        v2_known = [w["token"] for w in v2_words if w["token"].lower() in RELATION_TYPES]
        print(f"  V2 Blank-fill: {v2_known or '(none)'}  top3: {[w['token'] for w in v2_words[:3]]}")

        # === V3: Dictionary prior + prefill scan ===
        # Put the dictionary relation IN the prefill, read its position
        user_v3 = f"What is the role of {concept_a} regarding {concept_b}?\n\n{doc_text[:400]}"
        prefill_v3 = f"In this text, {concept_a} {dict_rel} {concept_b}. This means {concept_a}"
        prompt_v3 = wrap_chat(tokenizer, user_v3, prefill_v3)

        # Find positions of dict_rel and the last concept_a
        positions_v3 = find_positions(tokenizer, prompt_v3, [dict_rel, concept_a])
        read_pos_v3 = sorted(set(positions_v3.values()) | {-1})

        ws_v3 = read_at_positions(lens, lens_model, tokenizer, prompt_v3, read_pos_v3, layer)
        v3_all_words = []
        for pos, words in ws_v3.items():
            v3_all_words.extend(words)
        v3_known = list(set(w["token"] for w in v3_all_words if w["token"].lower() in RELATION_TYPES))

        # Show per-position
        for pos in sorted(ws_v3.keys(), key=lambda x: (x >= 0, x)):
            label = "end" if pos == -1 else f"pos{pos}"
            top3 = [(w["token"], round(w["prob"], 2)) for w in ws_v3[pos][:3]]
            known = [w["token"] for w in ws_v3[pos] if w["token"].lower() in RELATION_TYPES]
            print(f"  V3 Prior+scan [{label}]: {top3}  known: {known}")

        # === V4: Functional question ===
        user_v4 = f"How does {concept_a} function in relation to {concept_b}?\n\n{doc_text[:400]}"
        prefill_v4 = f"The function of {concept_a} in relation to {concept_b} is to"
        prompt_v4 = wrap_chat(tokenizer, user_v4, prefill_v4)
        ws_v4 = read_at_positions(lens, lens_model, tokenizer, prompt_v4, [-1], layer)
        v4_words = ws_v4.get(-1, [])
        v4_known = [w["token"] for w in v4_words if w["token"].lower() in RELATION_TYPES]
        print(f"  V4 Functional: {v4_known or '(none)'}  top3: {[w['token'] for w in v4_words[:3]]}")

        # === V5: Causal chain ===
        user_v5 = f"Describe the connection between {concept_a} and {concept_b}.\n\n{doc_text[:400]}"
        prefill_v5 = f"The connection is that {concept_a}"
        prompt_v5 = wrap_chat(tokenizer, user_v5, prefill_v5)
        ws_v5 = read_at_positions(lens, lens_model, tokenizer, prompt_v5, [-1], layer)
        v5_words = ws_v5.get(-1, [])
        v5_known = [w["token"] for w in v5_words if w["token"].lower() in RELATION_TYPES]
        print(f"  V5 Causal:     {v5_known or '(none)'}  top3: {[w['token'] for w in v5_words[:3]]}")

        pair_data.update({
            "v1_dict": {"known": v1_known, "top5": [w["token"] for w in v1_words[:5]]},
            "v2_blank": {"known": v2_known, "top5": [w["token"] for w in v2_words[:5]]},
            "v3_prior": {"known": v3_known, "dict_rel_used": dict_rel},
            "v4_func": {"known": v4_known, "top5": [w["token"] for w in v4_words[:5]]},
            "v5_causal": {"known": v5_known, "top5": [w["token"] for w in v5_words[:5]]},
        })
        all_results.append(pair_data)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    variants = {
        "V1 Dictionary": lambda r: r["v1_dict"]["known"],
        "V2 Blank-fill": lambda r: r["v2_blank"]["known"],
        "V3 Prior+scan": lambda r: r["v3_prior"]["known"],
        "V4 Functional": lambda r: r["v4_func"]["known"],
        "V5 Causal": lambda r: r["v5_causal"]["known"],
    }

    print(f"  {'variant':<20} {'pairs with known rel':>20}")
    print(f"  {'-'*42}")
    for name, extract in variants.items():
        n = sum(1 for r in all_results if extract(r))
        print(f"  {name:<20} {n:>10}/{len(all_results)}")

    # Also check: V3 known words different from V1?
    print(f"\n  V3 (prior+scan) unique vs V1:")
    for r in all_results:
        v1 = set(r["v1_dict"]["known"])
        v3 = set(r["v3_prior"]["known"])
        novel_v3 = v3 - v1
        if novel_v3:
            print(f"    {r['concept_a']}+{r['concept_b']}: V1={list(v1)} V3 novel={list(novel_v3)}")

    # Save
    cand = detect_model()
    out = {
        "method": "relation_prompt_variants",
        "model": cand["name"],
        "n_pairs": len(all_results),
        "results": all_results,
    }
    out_path = EXP / "phase37_relation_prompt_variants.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running relation prompt variants...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_variants(lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
