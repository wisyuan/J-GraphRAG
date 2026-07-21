"""Phase 24: 两遍关切耦合概念提取——验证真正的子概念。

Phase 22 证明当前 COM 分类不产生语义层级（97% 只是"相关"）。用户洞察：
真正的子概念需要把 Pass 1 提取的概念耦合进关切，再做一次提取。

## 两遍提取设计

Pass 1（宽泛概念）:
  文档 → "What concepts does this text discuss?" → J-Lens → [food, diet, fiber]
  这些是宽泛的平行概念（COM 分类没有语义层级意义）

Pass 2（具体内容）:
  文档 + "These documents are about {concept}. What does this text SPECIFICALLY
  discuss in the context of {concept}?" → J-Lens → 具体术语

  关键设计：用 Pass 1 的 top 概念作为"领域"上下文，"具体"强制模型
  越过宽泛概念，触及文档中的具体实体/方法/指标。

## 提示词备选测试

用户建议: "这篇文章在 XXX 领域具体讨论了什么"
翻译为 J-Lens prompt:
  user: "This text is in the field of {concept}. What does it specifically
        discuss in this field?"
  assistant prefill: "Specifically, this text discusses"

对比 Phase 16 的失败 prompt:
  "What specific TYPES of {concept}?" → 读出 types/aspects（结构词）
  "What SPECIFICALLY discusses in {concept}" → 读出领域内容

## 验证方法

对 Pass 2 读出的"子概念"：
1. 是否比 Pass 1 更具体？（人工审查）
2. LLM judge: Pass 2 的词是否是 Pass 1 top 概念的 is_a/part_of？（语义层级验证）
3. 是否避免了 Phase 16 的结构词污染（aspects/terms/discussed）？

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase24_two_pass_concern
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from itertools import product
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
from experiments.phase10_jlens_stage7c import extract_chunk_concepts, STOP_CONCEPTS
from experiments.phase20_concern_full_com import PREFILL_WORDS
from experiments.phase18_centroid_hierarchy import STOP_WORDS_EXTENDED, is_ascii_english
from experiments.phase16a_cross_domain_pos import classify_concept_pos, STOP_WORDS as STOP_16A
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20b_cross_domain import load_graphrag_corpus

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Combined stop words (Phase 16a + 18 + 20)
ALL_STOP = STOP_16A | STOP_WORDS_EXTENDED | PREFILL_WORDS | STOP_CONCEPTS


# ── Pass 1: broad concept extraction ──────────────────────────────────

def extract_pass1_concepts(
    lens, lens_model, tokenizer,
    docs: list[str], layer: int,
    n_words: int = 5,
) -> list[str]:
    """Pass 1: extract broad concepts (same as Stage 5/7c method).

    Single concern prompt → L26 readout → content words.
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List {n_words} one-word concepts.\n\n{doc_block}")
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(30)

    words, seen = [], set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in ALL_STOP
                and low not in seen and is_ascii_english(low)):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── Pass 2: specific concept extraction (concern-coupled) ─────────────

def extract_pass2_specific(
    lens, lens_model, tokenizer,
    docs: list[str], meta_concept: str, layer: int,
    n_words: int = 5,
) -> list[str]:
    """Pass 2: extract SPECIFIC concepts by coupling meta_concept into concern.

    User's prompt design: "这篇文章在 XXX 领域具体讨论了什么"

    The meta_concept from Pass 1 is used as "领域" context. "具体" forces
    the model past broad concepts to specific content.

    Key difference from Phase 16:
      Phase 16: "What TYPES of X?" → structural words (types/aspects)
      Pass 2:   "What SPECIFICALLY in the field of X?" → domain content
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (
        f"This text is in the field of {meta_concept}. "
        f"What does it specifically discuss in this field? "
        f"List {n_words} specific one-word terms.\n\n{doc_block}"
    )
    prefill = f"Specifically, in the field of {meta_concept}, this text discusses"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(30)

    words, seen = [], set()
    # Also exclude Pass 1's meta_concept itself and its variants
    meta_lower = meta_concept.lower()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in ALL_STOP
                and low not in seen and is_ascii_english(low)
                and low != meta_lower and not low.startswith(meta_lower)):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── LLM judge for sub-concept quality ─────────────────────────────────

def judge_sub_concept(
    meta_concept: str,
    sub_concept: str,
    doc_context: str,
    llm,
) -> dict:
    """Judge if sub_concept is a genuine sub-concept of meta_concept.

    Uses the same IS_A/PART_OF/RELATED/UNRELATED schema as Phase 22.
    """
    prompt = (
        f"You are evaluating whether a specific concept is a sub-concept "
        f"of a broader concept, both extracted from the same documents.\n\n"
        f"Document excerpt: \"{doc_context[:400]}\"\n\n"
        f"Broader concept (from first pass): {meta_concept}\n"
        f"Specific concept (from second pass): {sub_concept}\n\n"
        f"Judge the relationship:\n"
        f"  IS_A: {sub_concept} is a specific type/instance of {meta_concept}\n"
        f"  PART_OF: {sub_concept} is a component/mechanism/aspect of {meta_concept}\n"
        f"  RELATED: they co-occur but no hierarchical relationship\n"
        f"  UNRELATED: no meaningful connection\n\n"
        f"Reply with ONLY the label (IS_A, PART_OF, RELATED, or UNRELATED) "
        f"followed by a brief reason.\n"
        f"Example: IS_A - calcium oxalate is a type of mineral compound"
    )

    try:
        msg = llm.complete(prompt, max_tokens=60)
        resp = msg.content if hasattr(msg, 'content') else str(msg)
        resp_lower = resp.strip().lower()

        if "is_a" in resp_lower.split("\n")[0] or resp_lower.startswith("is a"):
            judgment = "is_a"
        elif "part_of" in resp_lower.split("\n")[0] or resp_lower.startswith("part"):
            judgment = "part_of"
        elif "unrelated" in resp_lower.split("\n")[0]:
            judgment = "unrelated"
        elif "related" in resp_lower.split("\n")[0]:
            judgment = "related"
        else:
            judgment = "related"

        return {"judgment": judgment, "reasoning": resp[:150]}
    except Exception as e:
        return {"judgment": "error", "reasoning": str(e)[:80]}


# ── Main experiment ───────────────────────────────────────────────────

def run_phase24(lens, lens_model, tokenizer, doc_texts: list[str], llm,
                domain_name: str = "nfcorpus",
                n_clusters: int = 6):
    print(f"Phase 24: Two-pass concern-coupled sub-concept extraction")
    print(f"  domain={domain_name}")
    print(f"{'='*70}")

    n = len(doc_texts)
    layer = lens.source_layers[-1]

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
    print(f"  {len(l0_valid)} clusters, analyzing top {len(top_clusters)}")

    # Two-pass extraction per cluster
    print(f"\n[2/3] Two-pass extraction...")
    all_results = []
    all_judgments = []

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]
        print(f"\n  Cluster {cid} ({len(members)} docs)")
        print(f"    sample: {docs[0][:100]}...")

        # Pass 1: broad concepts
        pass1 = extract_pass1_concepts(lens, lens_model, tokenizer, docs, layer)
        print(f"    Pass 1 (broad):    {pass1}")

        if not pass1:
            continue

        # Pass 2: specific concepts for each Pass 1 meta-concept
        for mc in pass1[:3]:  # top 3 meta-concepts
            pass2 = extract_pass2_specific(
                lens, lens_model, tokenizer, docs, mc, layer)
            print(f"    Pass 2 ({mc:12}): {pass2}")

            if not pass2:
                continue

            # LLM judge: is each Pass 2 word a sub-concept of the Pass 1 concept?
            doc_context = docs[0][:500]
            for sc in pass2[:4]:  # top 4 sub-concepts
                result = judge_sub_concept(mc, sc, doc_context, llm)
                result["cluster_id"] = cid
                result["meta"] = mc
                result["sub"] = sc
                all_judgments.append(result)

                tag = "✓" if result["judgment"] in ("is_a", "part_of") else " "
                print(f"      {tag} {mc} → {sc}: {result['judgment']}")

            all_results.append({
                "cluster_id": cid,
                "n_docs": len(members),
                "meta_concept": mc,
                "pass1_broad": pass1,
                "pass2_specific": pass2,
                "sample_doc": docs[0][:200],
            })

    # Summary
    print(f"\n{'='*70}")
    print(f"SUB-CONCEPT QUALITY SUMMARY")
    print(f"{'='*70}")

    n_total = len(all_judgments)
    n_is_a = sum(1 for j in all_judgments if j["judgment"] == "is_a")
    n_part_of = sum(1 for j in all_judgments if j["judgment"] == "part_of")
    n_related = sum(1 for j in all_judgments if j["judgment"] == "related")
    n_unrelated = sum(1 for j in all_judgments if j["judgment"] == "unrelated")
    n_error = sum(1 for j in all_judgments if j["judgment"] == "error")
    n_hier = n_is_a + n_part_of

    print(f"  Total pairs judged:     {n_total}")
    print(f"  IS_A:                   {n_is_a}")
    print(f"  PART_OF:                {n_part_of}")
    print(f"  RELATED:                {n_related}")
    print(f"  UNRELATED:              {n_unrelated}")
    print(f"  ERROR:                  {n_error}")
    print(f"  Hierarchical (is_a+part_of): {n_hier}/{n_total} "
          f"({n_hier/max(1,n_total):.0%})")

    print(f"\n  vs Phase 22 (COM-based): 0-9% hierarchical")
    print(f"  vs Phase 24 (two-pass):  {n_hier/max(1,n_total):.0%} hierarchical")

    # Best sub-concepts
    hier_pairs = [j for j in all_judgments
                  if j["judgment"] in ("is_a", "part_of")]
    if hier_pairs:
        print(f"\n  Verified sub-concepts:")
        for j in hier_pairs[:10]:
            print(f"    {j['meta']} → {j['sub']} ({j['judgment']})")

    # Save
    cand = detect_model()
    out = {
        "method": "two_pass_concern_sub_concept",
        "model": cand["name"],
        "domain": domain_name,
        "n_clusters": len(top_clusters),
        "n_pairs_judged": n_total,
        "n_is_a": n_is_a,
        "n_part_of": n_part_of,
        "n_related": n_related,
        "n_unrelated": n_unrelated,
        "hierarchy_rate": round(n_hier / max(1, n_total), 3),
        "results": all_results,
        "judgments": all_judgments,
    }
    out_path = EXP / f"phase24_two_pass_{domain_name}.json"
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

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    # Run on medical domain (best from Phase 20b)
    records = load_beir_fine_records("nfcorpus", max_docs=200)
    doc_texts = [r[1] for r in records]
    run_phase24(lens, lens_model, tokenizer, doc_texts, llm,
                domain_name="nfcorpus", n_clusters=6)


if __name__ == "__main__":
    main()
