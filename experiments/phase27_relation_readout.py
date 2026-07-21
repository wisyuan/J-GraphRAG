"""Phase 27 PoC: J-Lens 关切耦合读取概念间关系。

验证：给定两个概念词，J-Lens workspace 能否读出它们之间的关系类型？

Prompt 设计：
  "This text discusses {concept_A} and {concept_B}.
   The relationship between {concept_A} and {concept_B} is"
   ↑ 读出位置

和概念提取的关切耦合完全同一机制——只是 prompt 里放两个概念而非一个。
J-Lens 读 workspace top-k → 关系词（treatment/cause/prevent/symptom...）

测试用例（来自 medical 语料的已知概念对）：
  cancer + chemotherapy → treatment/therapy
  diet + fibre → contains/component
  calcium + bones → builds/strengthens
  tumor + metastasis → spreads/progression
  insulin + diabetes → treats/manages

对比基准：
  1. 双概念 prompt（A+B）→ 读出关系
  2. 单概念 prompt（A only）→ 读出概念（验证 workspace 没有被污染）
  3. 无概念 prompt（文档 only）→ 读出概念（baseline）

如果双概念 prompt 读出的关系词是合理的（treatment/cause/prevent），
则关切耦合可以创建关系边。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase27_relation_readout
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

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Stop words for relation readout filtering
STOP_REL = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study",
    "relationship", "between", "related", "based", "associated",
    "discussed", "discusses", "discuss", "discussing",
    "described", "describes", "describe", "describing",
    "shown", "shows", "show", "showing",
    "found", "finds", "finding", "findings",
    "reported", "reports", "report", "reporting",
    "include", "includes", "including",
    "involve", "involves", "involving",
    "cover", "covers", "covered",
    "focus", "focuses", "focused",
    "address", "addresses", "addressed",
    "explore", "explores", "explored",
    "examine", "examines", "examined",
    "consider", "considers", "considered",
    "analyze", "analyzes", "analyzed",
    "investigate", "investigates", "investigated",
    "highlight", "highlights", "highlighted",
    "demonstrate", "demonstrates", "demonstrated",
    "suggest", "suggests", "suggested",
    "indicate", "indicates", "indicated",
    "reveal", "reveals", "revealed",
    "present", "presents", "presented",
    "provide", "provides", "provided",
    "specific", "specifically", "particular",
    "various", "different", "certain", "general",
    "important", "possible", "available",
    "first", "second", "last", "new", "old",
    "however", "furthermore", "moreover", "additionally",
    "given", "unless", "except", "among", "despite",
    "until", "since", "today", "currently",
    "following", "above", "document", "documents",
    "text", "texts", "passage", "passages",
    "both", "either", "neither", "other", "another",
    "same", "similar", "different",
    "overall", "generally", "typically",
    "result", "results", "method", "methods",
    "patient", "patients", "treatment", "treatments",
    "clinical", "using", "data", "analysis",
    "research", "health", "disease", "medical",
    "group", "groups", "study", "studies",
}


def decode_topk(logits_row, tokenizer, n: int = 10, scan: int = 40):
    """Decode top-k content words from logits row."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(scan)
    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_REL
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n:
            break
    return results


def build_relation_prompt(doc_text: str, concept_a: str, concept_b: str,
                          tokenizer) -> str:
    """Dual-concept concern prompt for relation readout."""
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


def build_single_concept_prompt(doc_text: str, concept: str,
                                tokenizer) -> str:
    """Single-concept prompt (control: does workspace stay on topic?)."""
    user_msg = (
        f"What concepts does this text discuss? List 8 one-word concepts.\n\n{doc_text[:600]}"
    )
    prefill = "The concepts discussed are"
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


# ── Test cases from medical corpus ────────────────────────────────────

# Known concept pairs from Phase 25 medical results + domain knowledge
TEST_PAIRS = [
    # (concept_a, concept_b, expected_relation, doc_keywords)
    ("cancer", "chemotherapy", "treatment/therapy", "cancer chemotherapy treatment"),
    ("cancer", "tumor", "related/same", "cancer tumor growth"),
    ("cancer", "metastasis", "spreads/progression", "cancer metastasis spread"),
    ("diet", "fibre", "contains/component", "diet fibre fiber nutrition"),
    ("diet", "nutrition", "related/part_of", "diet nutrition food"),
    ("calcium", "bones", "builds/strengthens", "calcium bones bone"),
    ("insulin", "diabetes", "treats/manages", "insulin diabetes blood sugar"),
    ("cancer", "surgery", "treatment/removal", "cancer surgery removal"),
    ("tumor", "growth", "causes/involves", "tumor growth cell"),
    ("smoking", "cancer", "causes/risk", "smoking cancer risk"),
]


def run_poc(lens, lens_model, tokenizer):
    print("Phase 27 PoC: J-Lens relation readout via dual-concept concern")
    print(f"{'='*70}")

    # Load medical corpus for context
    corpus, _ = load_graphrag_bench("medical", max_queries=5)
    chunks = list(corpus.values())[:50]
    # Find chunks containing both concepts for each pair
    all_layers = lens.source_layers
    layer = all_layers[-1]  # L26

    results = []

    for concept_a, concept_b, expected, keywords in TEST_PAIRS:
        # Find a chunk that mentions both concepts
        doc_text = None
        for chunk in chunks:
            cl = chunk.lower()
            if concept_a in cl and concept_b in cl:
                doc_text = chunk
                break
        if not doc_text:
            # Fallback: find chunk mentioning at least one
            for chunk in chunks:
                if concept_a in chunk.lower() or concept_b in chunk.lower():
                    doc_text = chunk
                    break
        if not doc_text:
            doc_text = chunks[0]

        # === Test 1: Dual-concept relation prompt ===
        rel_prompt = build_relation_prompt(doc_text, concept_a, concept_b, tokenizer)
        lens_logits, _, _ = lens.apply(
            lens_model, rel_prompt, layers=[layer],
            positions=[-1], max_seq_len=512)
        rel_words = decode_topk(lens_logits[layer][0], tokenizer, n=8)

        # === Test 2: Single concept prompt (control) ===
        single_prompt = build_single_concept_prompt(doc_text, concept_a, tokenizer)
        lens_logits2, _, _ = lens.apply(
            lens_model, single_prompt, layers=[layer],
            positions=[-1], max_seq_len=512)
        single_words = decode_topk(lens_logits2[layer][0], tokenizer, n=8)

        # Print
        rel_str = ", ".join(f"{w['token']}({w['prob']:.2f})" for w in rel_words[:5])
        single_str = ", ".join(w["token"] for w in single_words[:5])

        print(f"\n  [{concept_a} + {concept_b}] (expected: {expected})")
        print(f"    doc excerpt: {doc_text[:100]}...")
        print(f"    RELATION readout: {rel_str}")
        print(f"    SINGLE concept:   {single_str}")

        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "expected": expected,
            "relation_words": rel_words,
            "single_concept_words": [w["token"] for w in single_words],
            "doc_excerpt": doc_text[:200],
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  Key question: do relation readouts contain relation-type words")
    print(f"  (treatment/cause/prevent/component) rather than just repeating")
    print(f"  the concepts or producing noise?\n")

    # Check if relation words are meaningful
    RELATION_WORDS = {
        "treatment", "treat", "treats", "treated", "treating",
        "therapy", "therapeutic", "therapies",
        "cause", "causes", "caused", "causing",
        "prevent", "prevents", "prevented", "preventing", "prevention",
        "risk", "risks", "risky",
        "component", "contains", "contained", "containing",
        "part", "parts", "source", "sources",
        "builds", "build", "built", "building", "strengthens", "strengthen",
        "spreads", "spread", "spreading", "progression", "progress",
        "growth", "grows", "growing",
        "inhibits", "inhibit", "inhibiting", "inhibition",
        "promotes", "promote", "promoted", "promoting",
        "manages", "manage", "managed", "managing",
        "removal", "removes", "remove", "removed",
        "associated", "associate", "association",
        "induces", "induce", "induced", "inducing",
        "reduces", "reduce", "reduced", "reducing",
        "increases", "increase", "increased", "increasing",
        "affects", "affect", "affected", "affecting",
        "improves", "improve", "improved", "improving",
        "worsens", "worsen", "worsened",
        "protects", "protect", "protected", "protection",
        "targets", "target", "targeted", "targeting",
        "kills", "kill", "killed", "killing",
        "supports", "support", "supported", "supporting",
        "requires", "require", "required", "requiring",
        "produces", "produce", "produced", "producing",
        "regulates", "regulate", "regulated", "regulating",
        "stimulates", "stimulate", "stimulated", "stimulating",
        "suppresses", "suppress", "suppressed", "suppressing",
    }

    n_meaningful = 0
    for r in results:
        rel_tokens = [w["token"].lower() for w in r["relation_words"]]
        meaningful = [t for t in rel_tokens if t in RELATION_WORDS]
        has_meaningful = len(meaningful) > 0
        if has_meaningful:
            n_meaningful += 1
        print(f"  {r['concept_a']:12} + {r['concept_b']:12} "
              f"→ {'✓' if has_meaningful else '✗'} "
              f"relation words: {rel_tokens[:5]}"
              f"{'  [' + ', '.join(meaningful) + ']' if meaningful else ''}")

    print(f"\n  Result: {n_meaningful}/{len(results)} pairs produced meaningful relation words")
    if n_meaningful >= len(results) * 0.4:
        print(f"  → Promising! Worth a formal experiment.")
    else:
        print(f"  → Relation readout quality insufficient for formal experiment.")

    # Save
    cand = detect_model()
    out = {
        "method": "relation_readout_poc",
        "model": cand["name"],
        "n_test_pairs": len(results),
        "n_meaningful": n_meaningful,
        "results": results,
    }
    out_path = EXP / "phase27_relation_readout_poc.json"
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

    print(f"\n[2/2] Running relation readout PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
