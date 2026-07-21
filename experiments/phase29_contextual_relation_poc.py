"""Phase 29 PoC: 双文档语境关系读出。

Phase 27/28 的关系读出是裸关系（词典级）——同一对概念在所有文档里
读出同一关系。本 PoC 验证：用两篇代表性文档的交叉语境锚定关系，
能否读出语境绑定的关系？

Prompt 设计：
  "Document 1 discusses {concept_A}: [doc1 excerpt]
   Document 2 discusses {concept_B}: [doc2 excerpt]
   In the context of these documents, the relationship between
   {concept_A} and {concept_B} is ___"

对比：
  A. 单文档裸关系（Phase 27 方法）：1 篇文档 + 概念对
  B. 双文档语境关系（新方法）：2 篇文档（各代表一个概念）+ 概念对

关键验证：同一对概念，用不同的文档对，读出的关系是否不同？
  cancer + diet:
    文档对1（癌症营养 + 膳食指南）→ affects/influences
    文档对2（癌症遗传 + 减肥饮食）→ ??? (should differ)

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase29_contextual_relation_poc
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
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase27_relation_readout import decode_topk, STOP_REL
from experiments.phase28_relation_graph import RELATION_TYPES

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def build_contextual_relation_prompt(
    doc_a: str, doc_b: str,
    concept_a: str, concept_b: str,
    tokenizer,
) -> str:
    """Dual-document contextual relation prompt.

    Two document excerpts anchor the relation in specific context,
    rather than generic dictionary knowledge.
    """
    user_msg = (
        f"Document 1 discusses {concept_a}:\n{doc_a[:400]}\n\n"
        f"Document 2 discusses {concept_b}:\n{doc_b[:400]}\n\n"
        f"In the context of these documents, what is the relationship "
        f"between {concept_a} and {concept_b}? Answer with one word."
    )
    prefill = f"In this context, {concept_a} and {concept_b} are"
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


def build_bare_relation_prompt(
    doc_text: str, concept_a: str, concept_b: str,
    tokenizer,
) -> str:
    """Phase 27 bare relation prompt (single doc, for comparison)."""
    user_msg = (
        f"This text discusses {concept_a} and {concept_b}. "
        f"What is the relationship between {concept_a} and {concept_b}? "
        f"Answer with one word.\n\n{doc_text[:600]}"
    )
    prefill = f"The relationship between {concept_a} and {concept_b} is"
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


def read_relation(lens, lens_model, tokenizer, prompt, layer):
    """Read relation from workspace."""
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=768)
    words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)
    return words


def run_poc(lens, lens_model, tokenizer):
    print("Phase 29 PoC: Dual-document contextual relation readout")
    print(f"{'='*70}")

    # Load medical corpus
    corpus, _ = load_graphrag_bench("medical", max_queries=5)
    chunks = list(corpus.values())
    layer = lens.source_layers[-1]

    # Test: same concept pair, different document contexts
    # Goal: show that relation changes with context

    TEST_CASES = [
        {
            "pair": ("cancer", "diet"),
            "doc_sets": [
                {
                    "label": "nutrition-oncology context",
                    "doc_a": None,  # find chunk with "cancer" + nutrition/diet
                    "doc_b": None,
                    "keywords_a": "cancer",
                    "keywords_b": "diet nutrition food",
                },
                {
                    "label": "genetic-risk context",
                    "doc_a": None,
                    "doc_b": None,
                    "keywords_a": "cancer genetic tumor",
                    "keywords_b": "diet weight loss",
                },
            ],
        },
        {
            "pair": ("cancer", "surgery"),
            "doc_sets": [
                {
                    "label": "treatment context",
                    "doc_a": None, "doc_b": None,
                    "keywords_a": "cancer treatment",
                    "keywords_b": "surgery removal",
                },
                {
                    "label": "prevention context",
                    "doc_a": None, "doc_b": None,
                    "keywords_a": "cancer prevention screening",
                    "keywords_b": "surgery risk",
                },
            ],
        },
        {
            "pair": ("tumor", "growth"),
            "doc_sets": [
                {
                    "label": "pathology context",
                    "doc_a": None, "doc_b": None,
                    "keywords_a": "tumor cell pathology",
                    "keywords_b": "growth proliferation",
                },
                {
                    "label": "treatment context",
                    "doc_a": None, "doc_b": None,
                    "keywords_a": "tumor shrink reduce",
                    "keywords_b": "growth inhibit suppress",
                },
            ],
        },
    ]

    # Find representative documents for each context
    for tc in TEST_CASES:
        for ds in tc["doc_sets"]:
            # Find doc_a: mentions keywords_a
            for chunk in chunks:
                cl = chunk.lower()
                kw_a = ds["keywords_a"].split()
                if any(k in cl for k in kw_a):
                    ds["doc_a"] = chunk
                    break
            # Find doc_b: mentions keywords_b, prefer different from doc_a
            for chunk in chunks:
                cl = chunk.lower()
                kw_b = ds["keywords_b"].split()
                if any(k in cl for k in kw_b) and chunk != ds["doc_a"]:
                    ds["doc_b"] = chunk
                    break

    results = []

    for tc in TEST_CASES:
        concept_a, concept_b = tc["pair"]
        print(f"\n{'='*60}")
        print(f"Concept pair: {concept_a} + {concept_b}")
        print(f"{'='*60}")

        pair_results = []

        for ds in tc["doc_sets"]:
            doc_a = ds["doc_a"]
            doc_b = ds["doc_b"]
            if not doc_a or not doc_b:
                print(f"\n  [{ds['label']}] SKIP (no matching docs)")
                continue

            # Method A: bare (single doc = doc_a)
            bare_prompt = build_bare_relation_prompt(
                doc_a, concept_a, concept_b, tokenizer)
            bare_words = read_relation(lens, lens_model, tokenizer,
                                        bare_prompt, layer)

            # Method B: contextual (dual doc)
            ctx_prompt = build_contextual_relation_prompt(
                doc_a, doc_b, concept_a, concept_b, tokenizer)
            ctx_words = read_relation(lens, lens_model, tokenizer,
                                       ctx_prompt, layer)

            bare_top5 = [(w["token"], w["prob"]) for w in bare_words[:5]]
            ctx_top5 = [(w["token"], w["prob"]) for w in ctx_words[:5]]

            # Check overlap
            bare_set = {w["token"].lower() for w in bare_words[:5]}
            ctx_set = {w["token"].lower() for w in ctx_words[:5]}
            overlap = bare_set & ctx_set

            # Check for known relation types
            bare_known = [w["token"] for w in bare_words[:5]
                          if w["token"].lower() in RELATION_TYPES]
            ctx_known = [w["token"] for w in ctx_words[:5]
                         if w["token"].lower() in RELATION_TYPES]

            print(f"\n  [{ds['label']}]")
            print(f"    doc_a: {doc_a[:80]}...")
            print(f"    doc_b: {doc_b[:80]}...")
            print(f"    BARE:     {bare_top5}")
            if bare_known:
                print(f"              known: {bare_known}")
            print(f"    CONTEXT:  {ctx_top5}")
            if ctx_known:
                print(f"              known: {ctx_known}")
            print(f"    overlap:  {overlap if overlap else '(none)'}")

            pair_results.append({
                "label": ds["label"],
                "doc_a_excerpt": doc_a[:150],
                "doc_b_excerpt": doc_b[:150],
                "bare_top5": [{"token": t, "prob": p} for t, p in bare_top5],
                "contextual_top5": [{"token": t, "prob": p} for t, p in ctx_top5],
                "bare_known": bare_known,
                "contextual_known": ctx_known,
                "overlap": list(overlap),
            })

        # Compare across contexts
        if len(pair_results) >= 2:
            ctx1_words = set(pair_results[0]["contextual_top5"][0]["token"].lower()
                            for _ in [0])  # top-1 of first context
            ctx2_words = set(pair_results[1]["contextual_top5"][0]["token"].lower()
                            for _ in [0])
            all_ctx1 = {w["token"].lower() for w in pair_results[0]["contextual_top5"]}
            all_ctx2 = {w["token"].lower() for w in pair_results[1]["contextual_top5"]}
            ctx_overlap = all_ctx1 & all_ctx2

            print(f"\n  Context comparison:")
            print(f"    Context 1 top-5: {all_ctx1}")
            print(f"    Context 2 top-5: {all_ctx2}")
            print(f"    Overlap: {ctx_overlap if ctx_overlap else '(none — relations differ!)'}")
            print(f"    → {'Same relation across contexts' if ctx_overlap else 'DIFFERENT relations — context matters!'}")

        results.append({
            "concept_pair": [concept_a, concept_b],
            "contexts": pair_results,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Key question: do different document contexts produce")
    print(f"  different relation readouts for the same concept pair?\n")

    n_different = 0
    n_same = 0
    for r in results:
        if len(r["contexts"]) >= 2:
            c1 = {w["token"].lower() for w in r["contexts"][0]["contextual_top5"]}
            c2 = {w["token"].lower() for w in r["contexts"][1]["contextual_top5"]}
            if c1 & c2:
                n_same += 1
                verdict = "same"
            else:
                n_different += 1
                verdict = "DIFFERENT"
            print(f"  {r['concept_pair'][0]:10} + {r['concept_pair'][1]:10}: {verdict}")
            print(f"    ctx1: {c1}")
            print(f"    ctx2: {c2}")

    print(f"\n  Result: {n_different}/{n_different+n_same} pairs produced context-dependent relations")
    if n_different > 0:
        print(f"  → Context matters! Dual-doc method captures context-specific relations.")
    else:
        print(f"  → Relations are context-independent (dictionary-level only).")

    # Save
    cand = detect_model()
    out = {
        "method": "contextual_relation_poc",
        "model": cand["name"],
        "n_pairs": len(results),
        "n_context_dependent": n_different,
        "results": results,
    }
    out_path = EXP / "phase29_contextual_relation_poc.json"
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

    print(f"\n[2/2] Running contextual relation PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
