"""Phase 36 PoC: 关系提取的 prefill position scan。

Phase 35 证明 prefill position scan 能读出文档特定的概念角色
（Nutrition → Education 71%）。
本实验验证：把两个概念放进 prefill，能否读出它们之间的文档特定关系。

Prompt 设计：
  User: "What is the relationship between {A} and {B}?"
  Assistant: "In this text, {A} is {A_RELATION} {B}, where {B} is"

  扫描 A_RELATION position 的 workspace → 读出 A 对 B 的关系词
  扫描 B 后续 position 的 workspace → 读出 B 对 A 的关系词

对比 Phase 27（字典关系）：
  Phase 27: position -1 单次读出 → treated/causal（泛泛）
  Phase 36: prefill position scan → 文档特定关系？

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase36_relation_prefill_scan
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
from experiments.phase27_relation_readout import decode_topk, build_relation_prompt, STOP_REL
from experiments.phase28_relation_graph import RELATION_TYPES

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def find_concept_positions(tokenizer, prompt: str, concepts: list[str]) -> dict:
    """Find token positions of concepts in the prompt."""
    token_ids = tokenizer.encode(prompt, return_tensors="pt")
    token_texts = []
    for i in range(token_ids.shape[1]):
        tok = tokenizer.decode([token_ids[0, i].item()])
        token_texts.append(tok)

    positions = {}
    full_text = "".join(token_texts).lower()

    # Find each concept in token stream
    for concept in concepts:
        cl = concept.lower()
        # Try exact token match first
        for i, t in enumerate(token_texts):
            if t.strip().lower() == cl:
                if concept not in positions:
                    positions[concept] = i
                break
        # Fallback: partial match
        if concept not in positions:
            for i, t in enumerate(token_texts):
                if cl.startswith(t.strip().lower()) and len(t.strip()) >= 3:
                    if concept not in positions:
                        positions[concept] = i
                    break

    return positions


def run_poc(lens, lens_model, tokenizer):
    print("Phase 36 PoC: Relation extraction via prefill position scan")
    print(f"{'='*70}")

    # Load corpus for document context
    try:
        from experiments.phase4_dig_graphragbench import load_graphrag_bench
        corpus, _ = load_graphrag_bench("medical", max_queries=1)
        chunks = list(corpus.values())
        source = "GraphRAG-Bench medical"
    except Exception:
        from experiments.corpus_loader import load_beir_fine_records
        records = load_beir_fine_records("nfcorpus", max_docs=200)
        chunks = [r[1] for r in records]
        source = "NFCorpus"

    print(f"  Source: {source} ({len(chunks)} chunks)")

    layer = lens.source_layers[-1]

    TEST_PAIRS = [
        ("cancer", "chemotherapy", "treatment"),
        ("cancer", "surgery", "treatment"),
        ("blood", "heart", "pumps/circulation"),
        ("insulin", "diabetes", "regulates"),
        ("calcium", "bones", "builds"),
        ("diet", "nutrition", "related"),
        ("diagnosis", "treatment", "precedes"),
        ("smoking", "cancer", "causes"),
    ]

    results = []

    for concept_a, concept_b, expected in TEST_PAIRS:
        # Find document containing both concepts
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

        print(f"\n  [{concept_a} + {concept_b}] (expected: {expected})")
        print(f"    doc: {doc_text[:80]}...")

        # === Method A: Phase 27 dictionary relation (position -1) ===
        prompt_a = build_relation_prompt(doc_text, concept_a, concept_b, tokenizer)
        lens_logits_a, _, _ = lens.apply(
            lens_model, prompt_a, layers=[layer],
            positions=[-1], max_seq_len=512)
        dict_words = decode_topk(lens_logits_a[layer][0], tokenizer, n=10, scan=50)
        dict_known = [w["token"] for w in dict_words if w["token"].lower() in RELATION_TYPES]

        print(f"    A) Dictionary (pos -1): {[w['token'] for w in dict_words[:5]]}")
        if dict_known:
            print(f"       known: {dict_known}")

        # === Method B: Prefill position scan ===
        # Design: put a "fill in the blank" structure in prefill
        # "In this text, {A} [BLANK] {B}. The {B} [BLANK2]"
        # Read workspace at BLANK positions
        user_msg = (
            f"What is the relationship between {concept_a} and {concept_b} "
            f"in this text?\n\n{doc_text[:400]}"
        )
        # Prefill with a template that creates positions between concepts
        prefill = f"In this text, {concept_a} "

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_b = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        else:
            prompt_b = f"{user_msg}\n{prefill}"

        # Find the position of the last token (right after concept_a in prefill)
        # This is where the model would generate the RELATION word
        # We read position -1 here (end of prefill) — this is the model
        # "about to describe" what A does to B
        lens_logits_b, _, _ = lens.apply(
            lens_model, prompt_b, layers=[layer],
            positions=[-1], max_seq_len=512)
        prefill_words = decode_topk(lens_logits_b[layer][0], tokenizer, n=10, scan=50)
        prefill_known = [w["token"] for w in prefill_words if w["token"].lower() in RELATION_TYPES]

        print(f"    B) Prefill '{prefill.strip()}' (pos -1):")
        print(f"       {[w['token'] for w in prefill_words[:5]]}")
        if prefill_known:
            print(f"       known: {prefill_known}")

        # === Method C: Full prefill with both concepts ===
        # "In this text, {A} [REL] {B}, where {B} [REL2]"
        # Put a marker word we can find, then scan
        marker = "_____"
        prefill_c = f"In this text, {concept_a} {marker} {concept_b}"

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_c = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill_c}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        else:
            prompt_c = f"{user_msg}\n{prefill_c}"

        # Find marker position and concept_b position
        positions_c = find_concept_positions(tokenizer, prompt_c, [marker, concept_b, concept_a])

        # Read position -1 (after concept_b) and marker position
        read_positions = []
        if marker in positions_c:
            read_positions.append(positions_c[marker])  # position between A and B
        read_positions.append(-1)  # end of prompt (after B)

        # Deduplicate
        read_positions = list(set(read_positions))

        try:
            lens_logits_c, _, _ = lens.apply(
                lens_model, prompt_c, layers=[layer],
                positions=read_positions, max_seq_len=512)
        except Exception as e:
            print(f"    C) Error: {e}")
            lens_logits_c = {}

        pos_to_idx = {p: i for i, p in enumerate(read_positions)}

        full_prefill_words = set()
        for pos in read_positions:
            idx = pos_to_idx.get(pos)
            if idx is not None and layer in lens_logits_c:
                words = decode_topk(lens_logits_c[layer][idx], tokenizer, n=10, scan=50)
                for w in words:
                    full_prefill_words.add(w["token"].lower())

                if pos == -1:
                    pos_label = "end(after B)"
                elif marker in positions_c and pos == positions_c[marker]:
                    pos_label = f"marker(between A,B at {pos})"
                else:
                    pos_label = f"pos {pos}"

                top5 = [(w["token"], w["prob"]) for w in words[:5]]
                known = [w["token"] for w in words if w["token"].lower() in RELATION_TYPES]
                print(f"    C) Prefill '{prefill_c}' at {pos_label}:")
                print(f"       top5: {top5[:3]}")
                if known:
                    print(f"       known: {known}")

        full_known = [w for w in full_prefill_words if w in RELATION_TYPES]

        # Summary
        print(f"\n    Summary:")
        print(f"      Dictionary: {dict_known if dict_known else '(none)'}")
        print(f"      Prefill A:  {prefill_known if prefill_known else '(none)'}")
        print(f"      Prefill C:  {full_known if full_known else '(none)'}")

        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "expected": expected,
            "dictionary_words": [w["token"] for w in dict_words[:5]],
            "dictionary_known": dict_known,
            "prefill_a_words": [w["token"] for w in prefill_words[:5]],
            "prefill_a_known": prefill_known,
            "prefill_c_words": sorted(full_prefill_words)[:10],
            "prefill_c_known": full_known,
        })

    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")

    methods = {
        "Dictionary (Phase 27)": lambda r: r["dictionary_known"],
        "Prefill A (after A)": lambda r: r["prefill_a_known"],
        "Prefill C (full A_B)": lambda r: r["prefill_c_known"],
    }

    for name, extract in methods.items():
        n = sum(1 for r in results if extract(r))
        print(f"  {name:30}: {n}/{len(results)} pairs have known relation words")

    # Save
    cand = detect_model()
    out = {
        "method": "relation_prefill_position_scan",
        "model": cand["name"],
        "n_pairs": len(results),
        "results": results,
    }
    out_path = EXP / "phase36_relation_prefill_scan.json"
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

    print(f"\n[2/2] Running relation prefill scan PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
