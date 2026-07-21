"""Phase 20: concern prompt + 全层 COM + corpus 验证 + POS 过滤。

Phase 18（纯读取+全层COM）：C101 完美但其他簇失败（纯读取在混杂文档上噪声大）
Phase 19（concern+band COM）：改善了不连贯文档但 band 砍掉了深层子概念
Phase 20 = 最优组合：
  - concern prompt（Phase 19 的概念定位能力）
  - 全层 COM（Phase 18 的深层子概念信号）
  - corpus 验证 + POS 过滤（对抗深层 prompt 污染，替代 band 限制）

核心洞察：Phase 19 失败因为用 band 限制来对抗污染——但污染的根因是
"concern prompt 的 prefill 词（concepts/discussed）在深层主导"。
正确的对抗方式不是截断层区间，而是过滤掉那些 prefill 词本身 + 非领域词。

三层过滤（替代 band 限制）：
  1. Corpus 验证：概念必须在文档中出现（排除 lens artifact + prefill 词）
  2. POS 过滤：排除动词-ing/-ed（方法词：assessed/evaluated/reported）
  3. Prefill 词黑名单：显式排除 prompt 自己的词（concepts/discussed/types）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase20_concern_full_com
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english, build_corpus_word_set,
    ConceptDepthProfile,
)
from experiments.phase16a_cross_domain_pos import classify_concept_pos

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Prefill words from the concern prompt that contaminate deep layers.
# These appear in the prompt itself and leak into the readout.
PREFILL_WORDS = {
    "concepts", "discussed", "discusses", "discuss", "discussing",
    "types", "aspects", "terms", "factors", "elements", "mentions",
    "references", "topics", "descriptions", "list", "include",
    "includes", "including", "involve", "involves", "involving",
    "cover", "covers", "covered", "covering",
    "address", "addresses", "addressed", "addressing",
    "mention", "mentioned", "mentioning",
    "describe", "describes", "described", "describing",
    "relate", "relates", "related", "relating",
    "focus", "focuses", "focused", "focusing",
    "explore", "explores", "explored", "exploring",
    "examine", "examines", "examined", "examining",
    "consider", "considers", "considered", "considering",
    "highlight", "highlights", "highlighted", "highlighting",
    "analyze", "analyzes", "analyzed", "analyzing",
    "investigate", "investigates", "investigated", "investigating",
    "report", "reports", "reported", "reporting",
    "present", "presents", "presented", "presenting",
    "provide", "provides", "provided", "providing",
    "demonstrate", "demonstrates", "demonstrated",
    "suggest", "suggests", "suggested",
    "indicate", "indicates", "indicated",
    "reveal", "reveals", "revealed",
    "show", "shows", "shown", "showing",
    "study", "studies", "studied",
    "find", "finds", "found", "finding", "findings",
}


def build_concern_prompt_full(docs: list[str], tokenizer) -> str:
    """Concern prompt (same as Phase 18/19). We use full-layer COM."""
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List 8 one-word concepts.\n\n{doc_block}")
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


def compute_concept_profiles_filtered(
    gradient: dict[int, list[dict]],
    corpus_words: set[str],
    min_layers: int = 3,
    require_corpus: bool = True,
    require_noun: bool = True,
) -> list[ConceptDepthProfile]:
    """Compute profiles with full-layer COM + triple filtering.

    Filters (replacing Phase 19's band restriction):
      1. ASCII English (filter multilingual artifacts)
      2. Not in STOP_WORDS_EXTENDED or PREFILL_WORDS
      3. Stability: >= min_layers appearances
      4. Must appear in >= 1 workspace layer (>= L10)
      5. Corpus verification (if require_corpus): must be a real word in docs
      6. POS filter (if require_noun): prefer nouns, reject pure verbs
    """
    word_layer_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for layer, words in gradient.items():
        for w in words:
            token = w["token"].lower()
            word_layer_probs[token].append((layer, w["prob"]))

    profiles = []
    for word, layer_probs in word_layer_probs.items():
        # Filter 1: ASCII
        if not is_ascii_english(word):
            continue
        # Filter 2: stopwords + prefill words
        if word in STOP_WORDS_EXTENDED or word in PREFILL_WORDS:
            continue

        layers_appeared = sorted(set(l for l, _ in layer_probs))
        n_layers = len(layers_appeared)

        # Filter 3: stability
        if n_layers < min_layers:
            continue
        # Filter 4: workspace presence
        if not any(l >= 10 for l in layers_appeared):
            continue

        # Filter 5: corpus verification
        in_corpus = word in corpus_words
        if require_corpus and not in_corpus:
            continue

        # Filter 6: POS — reject pure verbs (VBG/VBD), accept NN/NNS/UNK
        if require_noun:
            pos = classify_concept_pos(word)
            if pos in ("VBG", "VBD"):
                continue  # reject gerunds and past participles

        total_prob = sum(p for _, p in layer_probs)
        if total_prob <= 0:
            continue
        com = sum(l * p for l, p in layer_probs) / total_prob

        first = min(layers_appeared)
        last = max(layers_appeared)

        profiles.append(ConceptDepthProfile(
            word=word,
            layers=layers_appeared,
            probs=[p for _, p in sorted(layer_probs)],
            com=com,
            first_layer=first,
            last_layer=last,
            span=last - first,
            n_layers=n_layers,
            total_prob=total_prob,
            in_corpus=in_corpus,
        ))

    profiles.sort(key=lambda p: p.com)
    return profiles


def classify_by_com_gap(
    profiles: list[ConceptDepthProfile],
    com_gap_threshold: float = 2.0,
) -> list[ConceptDepthProfile]:
    """Classify meta/sub using COM gap detection (full-layer range)."""
    if len(profiles) <= 1:
        for p in profiles:
            p.role = "meta"
        return profiles

    coms = sorted(p.com for p in profiles)

    max_gap = 0
    gap_idx = len(coms) // 2
    for i in range(1, len(coms)):
        gap = coms[i] - coms[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    com_threshold = coms[gap_idx] if max_gap >= com_gap_threshold else coms[len(coms) // 2]

    for p in profiles:
        p.role = "meta" if p.com < com_threshold else "sub"

    return profiles


# ── Main ──────────────────────────────────────────────────────────────

def run_phase20(lens, lens_model, tokenizer, doc_texts: list[str],
                max_docs: int = 300, n_clusters: int = 12):
    print("Phase 20: concern prompt + full-layer COM + triple filter")
    print(f"  (concern for localization, full-layer COM, corpus+POS+prefill filter)")
    print(f"{'='*70}")

    n = len(doc_texts)
    layer = lens.source_layers[-1]
    all_layers = lens.source_layers

    # 1. L0 clustering
    print(f"\n[1/3] J-Lens residual clustering ({n} docs)...")
    prompts_resid = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer,
                                    prompts_resid, layer)
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    top_clusters = sorted(l0_valid.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(l0_valid)} clusters, analyzing top {len(top_clusters)}")

    # 2. For each cluster
    print(f"\n[2/3] Building concern + full-layer COM hierarchy...")
    results = []
    n_high_quality = 0
    n_any_hierarchy = 0

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]
        node_id = f"C{cid}"

        concern_prompt = build_concern_prompt_full(docs, tokenizer)
        gradient = extract_depth_gradient(
            lens, lens_model, tokenizer, concern_prompt,
            layers=all_layers, n_words=8, max_seq_len=512)

        corpus_words = build_corpus_word_set(docs)

        # Triple-filtered profiles with full-layer COM
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words,
            min_layers=3,
            require_corpus=True,
            require_noun=True,
        )
        profiles = classify_by_com_gap(profiles)

        # If too strict (nothing left), relax: allow non-corpus nouns
        if len(profiles) < 2:
            profiles_relaxed = compute_concept_profiles_filtered(
                gradient, corpus_words,
                min_layers=3,
                require_corpus=False,
                require_noun=True,
            )
            profiles_relaxed = classify_by_com_gap(profiles_relaxed)
            if len(profiles_relaxed) > len(profiles):
                profiles = profiles_relaxed

        meta_concepts = [p.word for p in profiles if p.role == "meta"]
        sub_concepts = [p.word for p in profiles if p.role == "sub"]

        n_verified = sum(1 for p in profiles if p.in_corpus)
        high_quality = len(meta_concepts) >= 2 and len(sub_concepts) >= 1 and n_verified >= 3
        any_hierarchy = len(meta_concepts) >= 1 and len(sub_concepts) >= 1
        if high_quality:
            n_high_quality += 1
        if any_hierarchy:
            n_any_hierarchy += 1

        # COM threshold
        metas = [p for p in profiles if p.role == "meta"]
        subs = [p for p in profiles if p.role == "sub"]
        com_threshold = 0.0
        if metas and subs:
            com_threshold = (max(p.com for p in metas) +
                             min(p.com for p in subs)) / 2

        results.append({
            "node_id": node_id,
            "n_docs": len(members),
            "meta_concepts": meta_concepts,
            "sub_concepts": sub_concepts,
            "com_threshold": round(com_threshold, 1),
            "high_quality": high_quality,
            "n_verified": n_verified,
            "profiles": [
                {"word": p.word, "com": round(p.com, 1),
                 "first": p.first_layer, "last": p.last_layer,
                 "n_layers": p.n_layers, "in_corpus": p.in_corpus,
                 "role": p.role, "layers": p.layers}
                for p in profiles
            ],
            "sample_doc": docs[0][:200],
        })

        # Print
        tag = "✓ HIGH-QUALITY" if high_quality else ("~ hierarchy" if any_hierarchy else "✗ flat")
        print(f"\n{'='*70}")
        print(f"[{node_id}] ({len(members)} docs) {tag}")
        print(f"  sample: {docs[0][:100]}...")
        if metas:
            print(f"\n  META (COM low, full-layer):")
            for p in profiles:
                if p.role == "meta":
                    c = "✓" if p.in_corpus else "✗"
                    print(f"    {p.word:20} COM={p.com:5.1f}  layers={p.n_layers:2}  "
                          f"corpus={c}  [{p.first_layer}-{p.last_layer}]")
        if subs:
            print(f"\n  SUB (COM high, full-layer):")
            for p in profiles:
                if p.role == "sub":
                    c = "✓" if p.in_corpus else "✗"
                    print(f"    {p.word:20} COM={p.com:5.1f}  layers={p.n_layers:2}  "
                          f"corpus={c}  [{p.first_layer}-{p.last_layer}]")

    # 3. Summary + 3-phase comparison
    print(f"\n{'='*70}")
    print(f"SUMMARY (Phase 20: concern + full-layer COM + triple filter)")
    print(f"{'='*70}")
    print(f"  Clusters analyzed:       {len(results)}")
    print(f"  High-quality:            {n_high_quality}/{len(results)} "
          f"({n_high_quality/max(1,len(results)):.0%})")
    print(f"  Any hierarchy:           {n_any_hierarchy}/{len(results)} "
          f"({n_any_hierarchy/max(1,len(results)):.0%})")

    print(f"\n  4-phase comparison:")
    print(f"    Phase 16 (prior expansion):     5% expand rate, prompt pollution")
    print(f"    Phase 18 (plain + full COM):    1/12 HQ (8%)  — C101 perfect")
    print(f"    Phase 19 (concern + band COM):  2/12 HQ (17%) — C101 regressed")
    print(f"    Phase 20 (concern + full COM):  {n_high_quality}/{len(results)} HQ "
          f"({n_high_quality/max(1,len(results)):.0%})")

    best = [r for r in results if r["high_quality"]]
    if best:
        print(f"\n  High-quality hierarchies:")
        for r in sorted(best, key=lambda x: len(x["sub_concepts"]), reverse=True):
            print(f"    [{r['node_id']}] "
                  f"meta={r['meta_concepts'][:4]} → sub={r['sub_concepts'][:4]}")

    # Save
    cand = detect_model()
    out = {
        "method": "concern_full_com_triple_filter",
        "model": cand["name"],
        "n_docs": n,
        "n_clusters_analyzed": len(results),
        "n_high_quality": n_high_quality,
        "n_any_hierarchy": n_any_hierarchy,
        "nodes": results,
    }
    out_path = EXP / "phase20_concern_full_com.json"
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

    print(f"\n[2/2] Running Phase 20...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase20(lens, lens_model, tokenizer, doc_texts, n_clusters=12)


if __name__ == "__main__":
    main()
