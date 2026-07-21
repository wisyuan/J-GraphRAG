"""Phase 24: 两遍提取 + BM25 补全——递归展开的新尝试。

## 背景

Phase 17-22 证明多层 workspace 提取 + COM 是一种改进的 plain 层级提取（不是
递归展开）。Phase 23 证明三重过滤器提升了 flat 图检索（106% of B0）。

Phase 16 尝试过递归展开（先验 prompt），失败于 "What types of X" 触发结构词。
但递归展开路线本身还没被证伪——只是那个 prompt 不行。

## 本实验

两遍提取：
  Pass 1: 文档 → concern prompt → J-Lens → 概念集 C1（plain 提取）
  补全:   C1 中的 BPE 前缀 → BM25 补全（Stat→statins）
  Pass 2: 文档 + C1 → 不同 prompt 变体 → J-Lens → 概念集 C2（递归提取）

4 个 prompt 变体 A/B/C/D 对比：
  A. 领域限定: "These documents are in the field of {C1}. What specific topics?"
  B. 已知排除: "Already identified: {C1}. What ELSE do they discuss?"
  C. Phase16: "What specific types of {concept}?" （对照组，已知触发结构词）
  D. 叙述续接: "This text is about {concept}. Specifically, it discusses ___"

## 评估

关键指标：C2 是否包含比 C1 更具体的概念？
  - C2 概念词的平均长度是否 > C1？（更具体 = 更长）
  - C2 概念的 corpus_hit_rate 是否 > C1？（更具体 = 更可能在文档中）
  - C2 是否避免了结构词（types/aspects/factors）？
  - LLM judge: C2 的概念是否比 C1 "更具体"？

如果某个 prompt 变体的 C2 产出领域概念而非结构词 → 递归展开可行。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase24_recursive_extraction
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals, build_concern_prompt
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
    classify_by_com_gap, PREFILL_WORDS,
)
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english, build_corpus_word_set,
)
from experiments.phase16a_cross_domain_pos import classify_concept_pos, STOP_WORDS
from experiments.concept_quality import (
    complete_prefix, build_corpus_term_freq, _get_wordnet_nouns,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Structural/meta words that indicate prompt contamination
STRUCTURAL_WORDS = {
    "types", "aspects", "factors", "elements", "terms", "mentions",
    "references", "topics", "categories", "kinds", "sorts", "forms",
    "versions", "methods", "approaches", "techniques", "procedures",
    "processes", "steps", "stages", "levels", "degrees", "ranges",
    "discussed", "discusses", "discuss", "discussing",
    "described", "describes", "describe", "describing",
    "examined", "examines", "examine", "examining",
    "analyzed", "analyzes", "analyze", "analyzing",
    "specific", "specifically", "particular", "particularly",
    "various", "different", "certain", "general", "overall",
    "include", "includes", "including", "involve", "involves",
    "cover", "covers", "covered", "covering",
    "focus", "focuses", "focused", "focusing",
}


# ── Pass 1: plain concept extraction ──────────────────────────────────

def pass1_extract(lens, lens_model, tokenizer, docs: list[str],
                  all_layers: list[int]) -> list[str]:
    """Pass 1: standard concern prompt extraction (Phase 20 method)."""
    prompt = build_concern_prompt_full(docs, tokenizer)
    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    corpus_words = build_corpus_word_set(docs)
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=2,
        require_corpus=False, require_noun=True)
    # Just take all concept words (no COM classification needed for pass 1)
    concepts = [p.word for p in profiles]

    # Fallback: if too few, take raw top-k from latest workspace layer
    if len(concepts) < 3:
        for layer in reversed(all_layers):
            for w in gradient.get(layer, []):
                tok = w["token"].lower()
                if (is_ascii_english(tok) and len(tok) >= 4
                        and tok not in STOP_WORDS_EXTENDED
                        and tok not in PREFILL_WORDS
                        and tok not in concepts):
                    concepts.append(tok)
            if len(concepts) >= 5:
                break

    return concepts[:8]


def bm25_complete(concepts: list[str], chunk_texts: list[str]) -> list[str]:
    """BM25 BPE completion: complete short prefixes to full words.

    Re-adds the BPE completion that Phase 20's filter dropped.
    """
    corpus_freq = build_corpus_term_freq(chunk_texts)
    wn_nouns = _get_wordnet_nouns()

    completed = []
    for c in concepts:
        if len(c) <= 5:
            result = complete_prefix(c, corpus_freq, wn_nouns)
            completed.append(result if result else c)
        else:
            completed.append(c)
    return completed


# ── Pass 2: recursive extraction with 4 prompt variants ──────────────

def build_prompt_A(docs: list[str], concepts: list[str], tokenizer) -> str:
    """A. 领域限定: concepts as field context."""
    concept_str = ", ".join(concepts[:5])
    doc_block = "\n---\n".join(d[:300] for d in docs[:6])
    user_msg = (
        f"These documents are in the field of {concept_str}. "
        f"What specific topics within this field do they cover? "
        f"List 8 one-word terms.\n\n{doc_block}"
    )
    prefill = "The specific topics covered are"
    return _wrap_chat(tokenizer, user_msg, prefill)


def build_prompt_B(docs: list[str], concepts: list[str], tokenizer) -> str:
    """B. 已知排除: 'what ELSE'."""
    concept_str = ", ".join(concepts[:5])
    doc_block = "\n---\n".join(d[:300] for d in docs[:6])
    user_msg = (
        f"We already identified these concepts: {concept_str}. "
        f"What other concepts do these documents discuss that are NOT in this list? "
        f"List 8 one-word terms.\n\n{doc_block}"
    )
    prefill = "Additional concepts discussed are"
    return _wrap_chat(tokenizer, user_msg, prefill)


def build_prompt_C(docs: list[str], concepts: list[str], tokenizer) -> str:
    """C. Phase 16 control: 'what types of X'."""
    # Pick the first concept as the prior
    concept = concepts[0] if concepts else "the topic"
    doc_block = "\n---\n".join(d[:300] for d in docs[:6])
    user_msg = (
        f"These documents are about {concept}. "
        f"What specific types or subcategories of {concept} do they discuss? "
        f"List 8 one-word terms.\n\n{doc_block}"
    )
    prefill = f"The specific types of {concept} include"
    return _wrap_chat(tokenizer, user_msg, prefill)


def build_prompt_D(docs: list[str], concepts: list[str], tokenizer) -> str:
    """D. 叙述续接: 'Specifically, it discusses ___'."""
    concept = concepts[0] if concepts else "the topic"
    doc_block = "\n---\n".join(d[:300] for d in docs[:6])
    user_msg = (
        f"Read these documents about {concept}.\n\n{doc_block}\n\n"
        f"Summarize the specific details discussed."
    )
    prefill = f"Regarding {concept}, these texts specifically discuss"
    return _wrap_chat(tokenizer, user_msg, prefill)


def _wrap_chat(tokenizer, user_msg: str, prefill: str) -> str:
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


PROMPT_VARIANTS = {
    "A_field": build_prompt_A,
    "B_exclude": build_prompt_B,
    "C_phase16": build_prompt_C,
    "D_narrative": build_prompt_D,
}


# ── Pass 2 extraction ─────────────────────────────────────────────────

def pass2_extract(lens, lens_model, tokenizer, docs: list[str],
                  concepts: list[str], variant: str,
                  all_layers: list[int]) -> list[str]:
    """Pass 2: recursive extraction using a specific prompt variant."""
    prompt_fn = PROMPT_VARIANTS[variant]
    prompt = prompt_fn(docs, concepts, tokenizer)

    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    # Extract content words from workspace layers
    # Use the same filter as Phase 20 but more lenient (pass 2 is exploratory)
    words = []
    seen = set()
    for layer in all_layers:
        for w in gradient.get(layer, []):
            tok = w["token"]
            low = tok.lower()
            if (len(tok) >= 4 and tok.isalpha() and is_ascii_english(low)
                    and low not in STOP_WORDS_EXTENDED
                    and low not in PREFILL_WORDS
                    and low not in STRUCTURAL_WORDS
                    and low not in seen
                    and low not in [c.lower() for c in concepts]):  # exclude pass 1
                if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                    seen.add(low)
                    words.append(tok)
        if len(words) >= 8:
            break

    return words[:8]


# ── Analysis ──────────────────────────────────────────────────────────

def analyze_concepts(concepts: list[str], corpus_words: set[str]) -> dict:
    """Analyze concept quality: length, corpus hit, POS, structural ratio."""
    if not concepts:
        return {"n": 0, "avg_len": 0, "corpus_hit_rate": 0,
                "structural_ratio": 0, "pos_dist": {}}

    lengths = [len(c) for c in concepts]
    corpus_hits = sum(1 for c in concepts if c.lower() in corpus_words)
    structural = sum(1 for c in concepts if c.lower() in STRUCTURAL_WORDS)
    pos_counts = defaultdict(int)
    for c in concepts:
        pos_counts[classify_concept_pos(c)] += 1

    return {
        "n": len(concepts),
        "avg_len": round(np.mean(lengths), 1),
        "corpus_hit_rate": round(corpus_hits / len(concepts), 2),
        "structural_ratio": round(structural / len(concepts), 2),
        "pos_dist": dict(pos_counts),
        "concepts": concepts,
    }


# ── Main experiment ───────────────────────────────────────────────────

def run_phase24(lens, lens_model, tokenizer, doc_texts: list[str],
                n_clusters: int = 8):
    print("Phase 24: Two-pass extraction + BM25 completion")
    print(f"  Pass 1: plain concern → concepts → BM25 complete")
    print(f"  Pass 2: 4 prompt variants (A/B/C/D) → recursive extraction")
    print(f"{'='*70}")

    n = len(doc_texts)
    layer = lens.source_layers[-1]
    all_layers = lens.source_layers

    # L0 clustering
    print(f"\n[1/3] Clustering ({n} docs)...")
    prompts_resid = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer,
                                    prompts_resid, layer)
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    top_clusters = sorted(l0_valid.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(top_clusters)} clusters")

    # Per-cluster two-pass extraction
    print(f"\n[2/3] Two-pass extraction per cluster...")
    results = []

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]
        corpus_words = build_corpus_word_set(docs)

        # Pass 1: plain extraction
        c1_raw = pass1_extract(lens, lens_model, tokenizer, docs, all_layers)
        # BM25 completion
        c1_completed = bm25_complete(c1_raw, docs)
        # Filter completed (remove duplicates, keep order)
        seen = set()
        c1 = []
        for c in c1_completed:
            if c.lower() not in seen:
                seen.add(c.lower())
                c1.append(c)

        c1_analysis = analyze_concepts(c1, corpus_words)

        print(f"\n  C{cid} ({len(members)}d):")
        print(f"    Pass 1: {c1}")
        print(f"      avg_len={c1_analysis['avg_len']} "
              f"corpus_hit={c1_analysis['corpus_hit_rate']:.0%} "
              f"structural={c1_analysis['structural_ratio']:.0%}")

        # Pass 2: 4 variants
        pass2_results = {}
        for variant, prompt_fn in PROMPT_VARIANTS.items():
            c2 = pass2_extract(lens, lens_model, tokenizer, docs, c1,
                               variant, all_layers)
            c2_analysis = analyze_concepts(c2, corpus_words)
            pass2_results[variant] = c2_analysis

            # Key metric: is C2 more specific than C1?
            more_specific = c2_analysis["avg_len"] > c1_analysis["avg_len"]
            no_structural = c2_analysis["structural_ratio"] == 0
            tag = "✓" if (more_specific and no_structural) else "✗"

            print(f"    Pass 2 [{variant}]: {c2}")
            print(f"      {tag} avg_len={c2_analysis['avg_len']} "
                  f"corpus_hit={c2_analysis['corpus_hit_rate']:.0%} "
                  f"structural={c2_analysis['structural_ratio']:.0%} "
                  f"pos={c2_analysis['pos_dist']}")

        results.append({
            "cluster_id": cid,
            "n_docs": len(members),
            "sample_doc": docs[0][:150],
            "pass1": c1_analysis,
            "pass2": pass2_results,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Does pass 2 produce more specific concepts?")
    print(f"{'='*70}")
    print(f"  {'variant':<15} {'avg_len':>8} {'corpus%':>8} "
          f"{'struct%':>8} {'more_spec':>9}")
    print(f"  {'-'*52}")

    # Pass 1 baseline
    p1_lens = [r["pass1"]["avg_len"] for r in results if r["pass1"]["n"] > 0]
    p1_corpus = [r["pass1"]["corpus_hit_rate"] for r in results if r["pass1"]["n"] > 0]
    p1_struct = [r["pass1"]["structural_ratio"] for r in results if r["pass1"]["n"] > 0]
    print(f"  {'Pass1(baseline)':<15} {np.mean(p1_lens):>8.1f} "
          f"{np.mean(p1_corpus):>7.0%} {np.mean(p1_struct):>7.0%} {'---':>9}")

    for variant in PROMPT_VARIANTS:
        v_lens = [r["pass2"][variant]["avg_len"] for r in results
                  if r["pass2"][variant]["n"] > 0]
        v_corpus = [r["pass2"][variant]["corpus_hit_rate"] for r in results
                     if r["pass2"][variant]["n"] > 0]
        v_struct = [r["pass2"][variant]["structural_ratio"] for r in results
                     if r["pass2"][variant]["n"] > 0]
        more_spec = sum(1 for r in results
                        if r["pass2"][variant]["avg_len"] > r["pass1"]["avg_len"]
                        and r["pass1"]["n"] > 0 and r["pass2"][variant]["n"] > 0)
        n_total = sum(1 for r in results
                      if r["pass1"]["n"] > 0 and r["pass2"][variant]["n"] > 0)
        print(f"  {variant:<15} {np.mean(v_lens):>8.1f} "
              f"{np.mean(v_corpus):>7.0%} {np.mean(v_struct):>7.0%} "
              f"{more_spec}/{n_total}")

    # Save
    cand = detect_model()
    out = {
        "method": "two_pass_extraction_bm25_completion",
        "model": cand["name"],
        "n_docs": n,
        "n_clusters": len(results),
        "clusters": results,
    }
    out_path = EXP / "phase24_recursive_extraction.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running two-pass extraction...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase24(lens, lens_model, tokenizer, doc_texts, n_clusters=8)


if __name__ == "__main__":
    main()
