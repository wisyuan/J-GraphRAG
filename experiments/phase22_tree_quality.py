"""Phase 22: 概念树质量评估——LLM judge meta→sub 父子关系。

Phase 20 产生了 meta→sub 概念层级（100% 覆盖率）。但覆盖率高不等于质量好——
需要验证 meta→sub 关系是否在语义上成立。

## 评估方法

对每个簇的 meta→sub 对，LLM judge 回答三个问题：

1. **Is-a 关系**：sub 概念是否是 meta 概念的子类型/实例？
   例：fiber is-a dietary? ✓    bone is-a cancer? ✗（bone 是 cancer 的转移部位，不是子类型）

2. **Part-of 关系**：sub 概念是否是 meta 概念的组成部分？
   例：fiber part-of nutrition? ✓    calcium part-of diet? ✓（钙是膳食成分）

3. **相关但非层级**：sub 和 meta 相关，但不是 is-a 或 part-of？
   例：questionnaire related-to diet（问卷是评估饮食的工具，不是饮食的子类型）

评分：每个 meta→sub 对判定为 hierarchically_correct (is-a/part-of) / related / unrelated。
一个簇的"层级质量分"= hierarchically_correct 对的比例。

## 同时评估两种概念树

- **Phase 20 树**（concern + 全层 COM + 三重过滤）
- **Phase 18 树**（纯读取 + 全层 COM）—— 作为对照

## 跨域评估

- NFCorpus（医学营养）
- Medical QA（GraphRAG-Bench）
- Novel（小说）—— 预期质量较低（叙事性）
- SciFact（科学论文）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase22_tree_quality
"""
from __future__ import annotations

import json
import os
import re
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
from experiments.phase17_multihop_depth_gradient import (
    extract_depth_gradient, build_plain_prompt,
)
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
    classify_by_com_gap,
)
from experiments.phase18_centroid_hierarchy import build_corpus_word_set
from experiments.phase20b_cross_domain import load_graphrag_corpus, load_scifact_docs

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── LLM judge for hierarchical correctness ────────────────────────────

def judge_hierarchy(
    meta_concept: str,
    sub_concept: str,
    doc_context: str,
    llm,
) -> dict:
    """Ask LLM whether sub_concept is hierarchically below meta_concept.

    Returns {judgment: "is_a"/"part_of"/"related"/"unrelated",
             confidence: float, reasoning: str}
    """
    prompt = (
        f"You are evaluating a concept hierarchy extracted from documents.\n\n"
        f"Document excerpt: \"{doc_context[:400]}\"\n\n"
        f"The system produced this hierarchy:\n"
        f"  Parent (broader concept): {meta_concept}\n"
        f"  Child (specific concept): {sub_concept}\n\n"
        f"Judge the relationship between parent and child:\n"
        f"  1. IS_A: {sub_concept} is a type/instance of {meta_concept}\n"
        f"  2. PART_OF: {sub_concept} is a component/aspect of {meta_concept}\n"
        f"  3. RELATED: they are related but NOT hierarchically (neither is-a nor part-of)\n"
        f"  4. UNRELATED: no meaningful connection\n\n"
        f"Reply with ONLY the label (IS_A, PART_OF, RELATED, or UNRELATED) "
        f"followed by a one-sentence reason.\n"
        f"Example output: IS_A - a fiber is a type of dietary component"
    )

    try:
        msg = llm.complete(prompt, max_tokens=60)
        resp = msg.content if hasattr(msg, 'content') else str(msg)
        resp_lower = resp.strip().lower()

        if resp_lower.startswith("is_a") or "is_a" in resp_lower.split("\n")[0]:
            judgment = "is_a"
        elif resp_lower.startswith("part_of") or "part_of" in resp_lower.split("\n")[0]:
            judgment = "part_of"
        elif resp_lower.startswith("related") or "related" in resp_lower.split("\n")[0]:
            judgment = "related"
        elif resp_lower.startswith("unrelated") or "unrelated" in resp_lower.split("\n")[0]:
            judgment = "unrelated"
        else:
            # Fallback: search for keywords
            if "is a" in resp_lower and "type" in resp_lower:
                judgment = "is_a"
            elif "part" in resp_lower or "component" in resp_lower:
                judgment = "part_of"
            elif "not" in resp_lower and ("hierarch" in resp_lower or "type" in resp_lower):
                judgment = "related"
            else:
                judgment = "related"  # default to related (conservative)

        return {"judgment": judgment, "reasoning": resp[:200]}
    except Exception as e:
        return {"judgment": "error", "reasoning": str(e)[:100]}


# ── Concept tree extraction (reuse Phase 20) ──────────────────────────

def extract_tree_for_cluster(
    lens, lens_model, tokenizer,
    docs: list[str],
    all_layers: list[int],
    method: str = "phase20",
) -> dict:
    """Extract meta/sub concepts for a cluster.

    method="phase20": concern prompt + full-layer COM + triple filter
    method="phase18": plain readout + full-layer COM (no concern)
    """
    if method == "phase20":
        prompt = build_concern_prompt_full(docs, tokenizer)
    else:
        prompt = build_plain_prompt(docs, tokenizer)

    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    corpus_words = build_corpus_word_set(docs)
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=3,
        require_corpus=True, require_noun=True)
    profiles = classify_by_com_gap(profiles)

    # Relax if too strict
    if len(profiles) < 2:
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=2,
            require_corpus=False, require_noun=True)
        profiles = classify_by_com_gap(profiles)

    meta = [p.word for p in profiles if p.role == "meta"]
    sub = [p.word for p in profiles if p.role == "sub"]

    return {"meta": meta[:5], "sub": sub[:5], "n_profiles": len(profiles)}


# ── Evaluation per domain ─────────────────────────────────────────────

def evaluate_domain(
    domain_name: str,
    doc_texts: list[str],
    lens, lens_model, tokenizer, llm,
    n_clusters: int = 6,
    max_pairs_per_cluster: int = 6,
) -> dict:
    """Evaluate concept tree quality for one domain.

    For each cluster: extract meta/sub → generate all meta×sub pairs →
    LLM judge each pair → compute hierarchy quality score.
    """
    n = len(doc_texts)
    if n < 20:
        return {"domain": domain_name, "skipped": True}

    layer = lens.source_layers[-1]
    all_layers = lens.source_layers

    # L0 clustering
    prompts_resid = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer,
                                    prompts_resid, layer)
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    top_clusters = sorted(l0_valid.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]

    all_judgments = []

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]

        tree = extract_tree_for_cluster(
            lens, lens_model, tokenizer, docs, all_layers, "phase20")

        meta_list = tree["meta"]
        sub_list = tree["sub"]

        if not meta_list or not sub_list:
            continue

        # Generate pairs (limit to max_pairs)
        pairs = list(product(meta_list, sub_list))[:max_pairs_per_cluster]
        doc_context = docs[0][:500]

        for meta_c, sub_c in pairs:
            result = judge_hierarchy(meta_c, sub_c, doc_context, llm)
            result["domain"] = domain_name
            result["cluster_id"] = cid
            result["meta"] = meta_c
            result["sub"] = sub_c
            all_judgments.append(result)

            tag = "✓" if result["judgment"] in ("is_a", "part_of") else "✗"
            print(f"    [{domain_name} C{cid}] {tag} "
                  f"{meta_c} → {sub_c}: {result['judgment']}")

    # Compute scores
    n_total = len(all_judgments)
    n_is_a = sum(1 for j in all_judgments if j["judgment"] == "is_a")
    n_part_of = sum(1 for j in all_judgments if j["judgment"] == "part_of")
    n_related = sum(1 for j in all_judgments if j["judgment"] == "related")
    n_unrelated = sum(1 for j in all_judgments if j["judgment"] == "unrelated")
    n_error = sum(1 for j in all_judgments if j["judgment"] == "error")

    n_hierarchical = n_is_a + n_part_of
    hierarchy_rate = n_hierarchical / max(1, n_total)
    related_rate = n_related / max(1, n_total)

    return {
        "domain": domain_name,
        "n_clusters": len(top_clusters),
        "n_pairs_judged": n_total,
        "n_is_a": n_is_a,
        "n_part_of": n_part_of,
        "n_related": n_related,
        "n_unrelated": n_unrelated,
        "n_error": n_error,
        "hierarchy_rate": round(hierarchy_rate, 3),
        "related_rate": round(related_rate, 3),
        "judgments": all_judgments,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 70)
    print("Phase 22: Concept tree quality evaluation (LLM judge)")
    print("=" * 70)

    print("\n[1/3] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print(f"\n[2/3] Loading LLM judge...")
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    # Load domains
    print(f"\n[3/3] Evaluating concept tree quality across domains...")
    max_docs = 200
    domains = {}

    nf = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    domains["nfcorpus"] = [r[1] for r in nf]
    print(f"  nfcorpus: {len(domains['nfcorpus'])} docs")

    sf = load_scifact_docs(max_docs=max_docs)
    if sf:
        domains["scifact"] = sf
        print(f"  scifact: {len(sf)} docs")

    novels = load_graphrag_corpus("novel", max_docs=max_docs)
    if novels:
        domains["novel"] = novels
        print(f"  novel: {len(novels)} docs")

    medical = load_graphrag_corpus("medical", max_docs=max_docs)
    if medical:
        domains["medical_qa"] = medical
        print(f"  medical_qa: {len(medical)} docs")

    # Evaluate each domain
    all_results = {}
    for domain_name, docs in domains.items():
        print(f"\n{'='*70}")
        print(f"Evaluating [{domain_name}]...")
        result = evaluate_domain(domain_name, docs, lens, lens_model,
                                  tokenizer, llm, n_clusters=5,
                                  max_pairs_per_cluster=4)
        all_results[domain_name] = result

    # Cross-domain summary
    print(f"\n{'='*70}")
    print(f"CROSS-DOMAIN CONCEPT TREE QUALITY")
    print(f"{'='*70}")
    print(f"  {'domain':<15} {'pairs':>5} {'is_a':>5} {'part':>5} "
          f"{'rel':>5} {'unrel':>5} {'hier%':>6} {'rel%':>6}")
    print(f"  {'-'*56}")
    for domain, r in all_results.items():
        if r.get("skipped"):
            print(f"  {domain:<15}  SKIPPED")
            continue
        print(f"  {domain:<15} {r['n_pairs_judged']:>5} {r['n_is_a']:>5} "
              f"{r['n_part_of']:>5} {r['n_related']:>5} {r['n_unrelated']:>5} "
              f"{r['hierarchy_rate']:>5.0%} {r['related_rate']:>5.0%}")

    # Save
    out = {
        "method": "concept_tree_quality_llm_judge",
        "model": cand["name"],
        "domains": {d: {k: v for k, v in r.items() if k != "judgments"}
                     for d, r in all_results.items()},
        "all_judgments": {d: r.get("judgments", [])
                           for d, r in all_results.items()},
    }
    out_path = EXP / "phase22_tree_quality.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
