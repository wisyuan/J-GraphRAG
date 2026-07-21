"""Phase 19: 混合质心算法——concern prompt 定位 + workspace 中间层梯度分离。

Phase 18 的纯读取质心算法在 C101（营养）上完美工作，但在混杂文档上失败
（workspace 被品牌名/URL 占据）。

Phase 17 A/B 对比揭示了一个未被利用的发现：
  - concern prompt 在 workspace 中间层（L10-L21）和纯读取高度重叠（0.2-0.5）
  - concern prompt 的概念定位能力能把不连贯文档的 workspace 引向主题
  - 但 concern prompt 在深层（L22+）被 prompt 词汇污染

混合方案（结合两者优势）：
  1. 用 concern prompt 做 forward pass（引导 workspace 形成主题概念）
  2. 只从 workspace 中间层（L10-L21）提取概念（避开深层污染）
  3. 在这个干净区间内算 COM，做元/子概念分类

关键设计：为什么 L10-L21 是干净区间？
  - Phase 17 证明 A/B 重叠在此区间最高（0.2-0.5）→ 文档驱动
  - L22+ 重叠降到 0 → concern 的 prefill 词汇开始主导
  - 所以只取 L10-L21 = 保留 concern 的概念定位 + 排除 prompt 污染

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase19_hybrid_centroid
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
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

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Workspace band: Phase 17 proved L10-L21 is the clean (document-driven) zone
WORKSPACE_START = 10
WORKSPACE_END = 21   # inclusive


def build_concern_prompt_hybrid(docs: list[str], tokenizer) -> str:
    """Concern prompt for concept localization.

    Same as Phase 17's build_concern_prompt_multi but we only read the
    workspace mid-layers (L10-L21), NOT the deep layers where prefill
    words (concepts/discussed) dominate.
    """
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


# ── Modified COM computation: workspace band only ─────────────────────

def compute_concept_profiles_band(
    gradient: dict[int, list[dict]],
    corpus_words: set[str],
    min_layers: int = 3,
    band_start: int = WORKSPACE_START,
    band_end: int = WORKSPACE_END,
) -> list[ConceptDepthProfile]:
    """Compute depth profiles using ONLY the workspace band [band_start, band_end].

    Key difference from Phase 18: COM is computed only within L10-L21.
    This excludes deep-layer prompt pollution from the centroid calc.

    A concept must still appear in >= min_layers within the band.
    """
    # Aggregate within band only
    word_layer_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for layer, words in gradient.items():
        if not (band_start <= layer <= band_end):
            continue
        for w in words:
            token = w["token"].lower()
            word_layer_probs[token].append((layer, w["prob"]))

    profiles = []
    for word, layer_probs in word_layer_probs.items():
        if not is_ascii_english(word):
            continue
        if word in STOP_WORDS_EXTENDED:
            continue

        layers_appeared = sorted(set(l for l, _ in layer_probs))
        n_layers = len(layers_appeared)

        if n_layers < min_layers:
            continue

        total_prob = sum(p for _, p in layer_probs)
        if total_prob <= 0:
            continue
        # COM computed within band only
        com = sum(l * p for l, p in layer_probs) / total_prob

        first = min(layers_appeared)
        last = max(layers_appeared)
        in_corpus = word in corpus_words

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


def classify_concepts_band(
    profiles: list[ConceptDepthProfile],
    com_gap_threshold: float = 1.5,  # smaller gap threshold (band is narrower)
    min_corpus_verified: int = 2,
) -> list[ConceptDepthProfile]:
    """Classify meta/sub using COM within the workspace band.

    The band is narrower (L10-L21 = 12 layers vs full 27), so the COM
    range is compressed. We use a smaller gap threshold.
    """
    verified = [p for p in profiles if p.in_corpus]
    working_set = verified if len(verified) >= min_corpus_verified else profiles

    if len(working_set) <= 1:
        for p in working_set:
            p.role = "meta"
        return profiles

    coms = sorted(p.com for p in working_set)

    max_gap = 0
    gap_idx = len(coms) // 2
    for i in range(1, len(coms)):
        gap = coms[i] - coms[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    com_threshold = coms[gap_idx] if max_gap >= com_gap_threshold else coms[len(coms) // 2]

    for p in profiles:
        if p in working_set:
            p.role = "meta" if p.com < com_threshold else "sub"
        else:
            p.role = "noise"

    return profiles


# ── Main ──────────────────────────────────────────────────────────────

def run_phase19(lens, lens_model, tokenizer, doc_texts: list[str],
                max_docs: int = 300, n_clusters: int = 12):
    print("Phase 19: Hybrid centroid — concern prompt + workspace band COM")
    print(f"  (concern prompt for localization, L{WORKSPACE_START}-L{WORKSPACE_END} for COM)")
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

    # 2. For each cluster: concern prompt → band-restricted COM
    print(f"\n[2/3] Building hybrid centroid hierarchy per cluster...")
    results = []
    n_high_quality = 0
    n_any_hierarchy = 0

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]
        node_id = f"C{cid}"

        # Concern prompt forward pass (reads ALL layers, but we only use band)
        concern_prompt = build_concern_prompt_hybrid(docs, tokenizer)
        gradient = extract_depth_gradient(
            lens, lens_model, tokenizer, concern_prompt,
            layers=all_layers, n_words=8, max_seq_len=512)

        # Compute profiles restricted to workspace band
        corpus_words = build_corpus_word_set(docs)
        profiles = compute_concept_profiles_band(gradient, corpus_words)
        profiles = classify_concepts_band(profiles)

        meta_concepts = [p.word for p in profiles if p.role == "meta"]
        sub_concepts = [p.word for p in profiles if p.role == "sub"]
        noise = [p.word for p in profiles if p.role == "noise"]

        # Quality assessment
        n_verified_meta = sum(1 for p in profiles if p.role == "meta" and p.in_corpus)
        n_verified_sub = sum(1 for p in profiles if p.role == "sub" and p.in_corpus)
        high_quality = n_verified_meta >= 2 and n_verified_sub >= 1
        any_hierarchy = len(meta_concepts) >= 2 and len(sub_concepts) >= 1
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
            "noise": noise,
            "com_threshold": round(com_threshold, 1),
            "high_quality": high_quality,
            "n_verified_meta": n_verified_meta,
            "n_verified_sub": n_verified_sub,
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
            print(f"\n  META (COM low within L{WORKSPACE_START}-L{WORKSPACE_END}):")
            for p in profiles:
                if p.role == "meta":
                    c = "✓" if p.in_corpus else "✗"
                    print(f"    {p.word:20} COM={p.com:5.1f}  layers={p.n_layers}  "
                          f"corpus={c}  [{p.first_layer}-{p.last_layer}]")
        if subs:
            print(f"\n  SUB (COM high within L{WORKSPACE_START}-L{WORKSPACE_END}):")
            for p in profiles:
                if p.role == "sub":
                    c = "✓" if p.in_corpus else "✗"
                    print(f"    {p.word:20} COM={p.com:5.1f}  layers={p.n_layers}  "
                          f"corpus={c}  [{p.first_layer}-{p.last_layer}]")

    # 3. Summary + comparison with Phase 18
    print(f"\n{'='*70}")
    print(f"SUMMARY (Phase 19: hybrid concern + workspace band)")
    print(f"{'='*70}")
    print(f"  Clusters analyzed:       {len(results)}")
    print(f"  High-quality (≥2 verified meta + ≥1 verified sub): "
          f"{n_high_quality}/{len(results)} ({n_high_quality/max(1,len(results)):.0%})")
    print(f"  Any hierarchy (≥2 meta + ≥1 sub):                   "
          f"{n_any_hierarchy}/{len(results)} ({n_any_hierarchy/max(1,len(results)):.0%})")

    print(f"\n  vs Phase 18 (plain readout, full-layer COM):")
    print(f"    Phase 18 high-quality: 1/12 (8%)")
    print(f"    Phase 19 high-quality: {n_high_quality}/{len(results)} "
          f"({n_high_quality/max(1,len(results)):.0%})")

    # Best hierarchies
    best = [r for r in results if r["high_quality"]]
    if best:
        print(f"\n  High-quality hierarchies:")
        for r in sorted(best, key=lambda x: x["n_verified_sub"], reverse=True):
            print(f"    [{r['node_id']}] "
                  f"meta={r['meta_concepts'][:4]} → sub={r['sub_concepts'][:4]}")

    # Save
    cand = detect_model()
    out = {
        "method": "hybrid_centroid_concern_workspace_band",
        "model": cand["name"],
        "n_docs": n,
        "n_clusters_analyzed": len(results),
        "n_high_quality": n_high_quality,
        "n_any_hierarchy": n_any_hierarchy,
        "workspace_band": [WORKSPACE_START, WORKSPACE_END],
        "nodes": results,
    }
    out_path = EXP / "phase19_hybrid_centroid.json"
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

    print(f"\n[2/2] Running Phase 19 hybrid centroid...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase19(lens, lens_model, tokenizer, doc_texts, n_clusters=12)


if __name__ == "__main__":
    main()
