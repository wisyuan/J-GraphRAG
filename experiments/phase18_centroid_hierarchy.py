"""Phase 18: 概念层分布质心算法——深度梯度→概念层级树。

用户提出的算法（基于 Phase 17 数据验证后确定可行）：
  1. 全层次提取：纯读取文档，一次 forward pass 扫描全部 27 层
  2. 清除噪声：≥3 层稳定性过滤 + 语料验证（概念必须在文档中出现）
  3. 统计层分布：每个概念计算概率加权质心（COM = center of mass）
  4. 层级分类：COM 低（早期 workspace 形成）= 元概念；COM 高（晚期形成）= 子概念

核心信号：J-Lens workspace 层的概念形成有先后顺序。跨多个层稳定出现的
概念（如 nutrition 跨 L10-L21）比只在深层出现的概念（如 fibre 只在 L22-L26）
形成得更早、更基础——这对应概念树的父节点。

Phase 17 A/B 对比证明纯读取（不加 concern prompt）在 workspace 中间层
（L10-L21）最干净，深层（L22+）被 prompt 污染。因此本实验用纯读取。

算法对比：
  Phase 16: 先验展开（prompt 请求子概念）→ prompt 污染，失败
  Phase 17: 深度梯度发现（A/B 对比）→ 证明梯度存在
  Phase 18: 质心算法（纯读取 + 稳定性 + COM 排序）→ 从梯度提取层级树

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase18_centroid_hierarchy
"""
from __future__ import annotations

import json
import math
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
from experiments.phase17_multihop_depth_gradient import (
    extract_depth_gradient, build_plain_prompt,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── Extended STOP_WORDS (learned from Phase 17/18 noise patterns) ──────

# Phase 17 分析中发现的结构性噪声词——论文格式/方法论/连接词，不是领域概念
STOP_WORDS_EXTENDED = {
    # Base stopwords (from phase16a)
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study", "studies",
    "result", "results", "method", "patient", "patients", "treatment",
    "associated", "compared", "significantly", "clinical", "using", "data",
    "analysis", "research", "health", "disease", "medical", "group",
    "following", "above", "document", "documents", "discuss", "discusses",
    "related", "based", "summarized", "listed", "outlined", "recent",
    "several", "evidence", "suggesting", "characterized", "indicating",
    "primarily", "showed", "might", "show", "seem", "appear", "require",
    "occur", "arise", "include", "single", "similar", "three", "five",
    "once", "which", "these", "those", "their", "there", "where",
    "while", "about", "after", "before", "between", "during", "through",
    "without", "within", "because", "however", "although", "whether",
    "many", "much", "both", "other", "another", "same", "different",
    "important", "possible", "available", "specific", "particular",
    "general", "common", "rare", "high", "low", "large", "small",
    "first", "second", "last", "next", "new", "old",
    # Paper structure words (Phase 18 noise)
    "findings", "methodology", "participants", "subjects", "population",
    "design", "designs", "designed", "describing", "described",
    "conclusion", "question", "questions", "summary", "objective",
    "background", "purpose", "approach", "framework", "review",
    "assessment", "assessments", "evaluated", "evaluation", "evaluations",
    "reported", "reporting", "published", "investigated", "investigating",
    "conducted", "examined", "analyzed", "measured", "determined",
    "observed", "demonstrated", "indicated", "suggests", "revealed",
    "shown", "included", "involving", "selected", "randomized",
    "divided", "assigned", "recruited", "enrolled",
    # Generic research/limitation words
    "limited", "limitations", "available", "availability", "accessibility",
    "empirical", "experimental", "prospective", "retrospective",
    "standard", "standardized", "routine", "practice", "practical",
    "alternative", "application", "applications", "applied",
    "number", "numbers", "amount", "amounts", "levels", "values",
    "rate", "rates", "frequency", "content", "contents",
    "overall", "generally", "typically", "usually", "often",
    "however", "furthermore", "moreover", "additionally", "consequently",
    "given", "unless", "except", "among", "outside", "despite",
    "until", "since", "today", "currently",
}


# ── Concept with depth distribution ───────────────────────────────────

@dataclass
class ConceptDepthProfile:
    """A concept word's distribution across workspace layers."""
    word: str
    layers: list[int]          # which layers it appears in
    probs: list[float]         # probability at each layer
    com: float                 # center of mass (prob-weighted average layer)
    first_layer: int           # first appearance
    last_layer: int             # last appearance
    span: int                  # last - first
    n_layers: int              # how many layers it appears in
    total_prob: float          # sum of probabilities
    in_corpus: bool            # verified in cluster documents
    role: str = "unknown"      # "meta" / "sub" / "noise" (assigned later)


def build_corpus_word_set(doc_texts: list[str]) -> set[str]:
    """Build set of real words appearing in documents (for corpus verification)."""
    words = set()
    for text in doc_texts:
        for m in re.finditer(r'[a-zA-Z]{4,}', text):
            words.add(m.group().lower())
    return words


def is_ascii_english(word: str) -> bool:
    """Filter out non-English tokens (multilingual BPE artifacts).

    Lens sometimes produces tokens from other languages (usuarios, novità,
    männer). These are artifacts, not domain concepts. We require ASCII.
    """
    try:
        word.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def compute_concept_profiles(
    gradient: dict[int, list[dict]],
    corpus_words: set[str],
    min_layers: int = 3,
    min_workspace_layers: int = 1,
    workspace_onset: int = 10,
) -> list[ConceptDepthProfile]:
    """Compute depth profiles for all concept words in the gradient.

    Filters:
      1. Must appear in >= min_layers layers (stability)
      2. Must appear in >= min_workspace_layers layers >= workspace_onset
         (not just noise layers)
      3. Must be ASCII English (filter multilingual artifacts)
      4. Must not be in STOP_WORDS_EXTENDED
      5. Corpus verification tracked (in_corpus field)

    Returns list of ConceptDepthProfile, sorted by COM ascending.
    """
    # Aggregate: word -> [(layer, prob), ...]
    word_layer_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for layer, words in gradient.items():
        for w in words:
            token = w["token"].lower()
            word_layer_probs[token].append((layer, w["prob"]))

    profiles = []
    for word, layer_probs in word_layer_probs.items():
        # Filter 1: ASCII English
        if not is_ascii_english(word):
            continue
        # Filter 2: not stopword
        if word in STOP_WORDS_EXTENDED:
            continue

        layers_appeared = sorted(set(l for l, _ in layer_probs))
        n_layers = len(layers_appeared)

        # Filter 3: stability (>= min_layers)
        if n_layers < min_layers:
            continue

        # Filter 4: must appear in workspace
        workspace_layers = [l for l in layers_appeared if l >= workspace_onset]
        if len(workspace_layers) < min_workspace_layers:
            continue

        # Compute center of mass
        total_prob = sum(p for _, p in layer_probs)
        if total_prob <= 0:
            continue
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

    # Sort by COM ascending (early-forming concepts first)
    profiles.sort(key=lambda p: p.com)
    return profiles


def classify_concepts(
    profiles: list[ConceptDepthProfile],
    com_gap_threshold: float = 2.0,
    min_corpus_verified: int = 2,
) -> list[ConceptDepthProfile]:
    """Classify concepts as meta vs sub based on COM distribution.

    Algorithm (user's design):
      1. Keep only corpus-verified concepts (must appear in documents)
      2. If too few corpus-verified (< min_corpus_verified), keep all
      3. Look for a natural gap in COM values
      4. Concepts below gap = meta (early-forming, fundamental)
      5. Concepts above gap = sub (late-forming, specific)

    The gap is found by sorting COMs and finding the largest consecutive
    difference. If no gap > com_gap_threshold, use the median.
    """
    # Prefer corpus-verified concepts
    verified = [p for p in profiles if p.in_corpus]
    if len(verified) < min_corpus_verified:
        # Too few verified — use all profiles but mark corpus status
        working_set = profiles
    else:
        working_set = verified

    if len(working_set) <= 1:
        for p in working_set:
            p.role = "meta"
        return profiles

    coms = sorted(p.com for p in working_set)

    # Find largest gap in COM values
    max_gap = 0
    gap_idx = len(coms) // 2  # default: median split
    for i in range(1, len(coms)):
        gap = coms[i] - coms[i - 1]
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    # Use gap if it's significant, otherwise median
    if max_gap >= com_gap_threshold:
        com_threshold = coms[gap_idx]
    else:
        com_threshold = coms[len(coms) // 2]

    # Assign roles
    for p in profiles:
        if p in working_set:
            p.role = "meta" if p.com < com_threshold else "sub"
        else:
            # Not in working set (failed corpus verification with enough
            # verified concepts available) → mark as noise
            p.role = "noise"

    return profiles


# ── Tree construction ─────────────────────────────────────────────────

@dataclass
class ConceptNode:
    """A node in the depth-gradient concept tree."""
    node_id: str
    level: str                 # "meta" or "sub"
    doc_indices: list[int]
    meta_concepts: list[str]   # COM-low concepts
    sub_concepts: list[str]    # COM-high concepts
    profiles: list[dict]       # full depth profiles for audit
    com_threshold: float = 0.0
    children: list["ConceptNode"] = field(default_factory=list)
    sample_doc: str = ""


def build_concept_tree_from_gradient(
    gradient: dict[int, list[dict]],
    doc_texts: list[str],
    doc_indices: list[int],
    node_id: str,
) -> ConceptNode:
    """Build a concept tree node from a depth gradient.

    The tree is two-level by design:
      Level 0 (meta): concepts with low COM (early workspace formation)
      Level 1 (sub): concepts with high COM (late workspace formation)

    Deeper recursion would require sub-clustering within sub-concepts,
    which is future work (Phase 19+).
    """
    corpus_words = build_corpus_word_set(doc_texts)
    profiles = compute_concept_profiles(gradient, corpus_words)
    profiles = classify_concepts(profiles)

    meta_concepts = [p.word for p in profiles if p.role == "meta"]
    sub_concepts = [p.word for p in profiles if p.role == "sub"]

    com_threshold = 0.0
    metas = [p for p in profiles if p.role == "meta"]
    subs = [p for p in profiles if p.role == "sub"]
    if metas and subs:
        com_threshold = (max(p.com for p in metas) +
                         min(p.com for p in subs)) / 2

    return ConceptNode(
        node_id=node_id,
        level="root",
        doc_indices=doc_indices,
        meta_concepts=meta_concepts,
        sub_concepts=sub_concepts,
        profiles=[
            {
                "word": p.word,
                "com": round(p.com, 1),
                "first": p.first_layer,
                "last": p.last_layer,
                "n_layers": p.n_layers,
                "total_prob": round(p.total_prob, 4),
                "in_corpus": p.in_corpus,
                "role": p.role,
                "layers": p.layers,
            }
            for p in profiles
        ],
        com_threshold=round(com_threshold, 1),
        sample_doc=doc_texts[0][:200] if doc_texts else "",
    )


# ── Main experiment ───────────────────────────────────────────────────

def run_phase18(lens, lens_model, tokenizer, doc_texts: list[str],
                max_docs: int = 300, n_clusters: int = 12):
    print("Phase 18: Centroid-based concept hierarchy from depth gradient")
    print(f"  (plain readout → 27-layer sweep → COM ranking → meta/sub)")
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

    # 2. For each cluster: plain-readout depth gradient → centroid tree
    print(f"\n[2/3] Building centroid hierarchy per cluster...")
    results = []
    n_with_hierarchy = 0

    for idx, (cid, members) in enumerate(top_clusters):
        docs = [doc_texts[m] for m in members]
        node_id = f"C{cid}"

        # Plain readout: document text only, no concern prompt
        plain_prompt = build_plain_prompt(docs, tokenizer)
        gradient = extract_depth_gradient(
            lens, lens_model, tokenizer, plain_prompt,
            layers=all_layers, n_words=8, max_seq_len=512)

        # Build hierarchy
        node = build_concept_tree_from_gradient(
            gradient, docs, members, node_id)

        has_hierarchy = len(node.meta_concepts) >= 2 and len(node.sub_concepts) >= 1
        if has_hierarchy:
            n_with_hierarchy += 1

        results.append(node)

        # Print
        print(f"\n{'='*70}")
        print(f"[{node_id}] ({len(members)} docs) "
              f"{'✓ HIERARCHY' if has_hierarchy else '✗ flat'}")
        print(f"  sample: {docs[0][:100]}...")
        print(f"  COM threshold: {node.com_threshold}")
        print(f"\n  META concepts (COM low, early workspace):")
        for p in node.profiles:
            if p["role"] == "meta":
                corpus = "✓" if p["in_corpus"] else "✗"
                print(f"    {p['word']:20} COM={p['com']:5.1f}  "
                      f"layers={p['n_layers']:2}  corpus={corpus}  "
                      f"[{p['first']}-{p['last']}]")
        print(f"\n  SUB concepts (COM high, late workspace):")
        for p in node.profiles:
            if p["role"] == "sub":
                corpus = "✓" if p["in_corpus"] else "✗"
                print(f"    {p['word']:20} COM={p['com']:5.1f}  "
                      f"layers={p['n_layers']:2}  corpus={corpus}  "
                      f"[{p['first']}-{p['last']}]")
        noise = [p for p in node.profiles if p["role"] == "noise"]
        if noise:
            print(f"\n  NOISE (filtered): {[p['word'] for p in noise][:10]}")

    # 3. Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Clusters analyzed:      {len(results)}")
    print(f"  With hierarchy (≥2 meta + ≥1 sub): {n_with_hierarchy}/{len(results)} "
          f"({n_with_hierarchy/max(1,len(results)):.0%})")

    # Best hierarchies
    best = [r for r in results
            if len(r.meta_concepts) >= 2 and len(r.sub_concepts) >= 2]
    if best:
        print(f"\n  Best hierarchies:")
        for node in sorted(best, key=lambda x: len(x.sub_concepts), reverse=True)[:5]:
            print(f"    [{node.node_id}] "
                  f"meta={node.meta_concepts[:3]} → sub={node.sub_concepts[:3]}")

    # Save
    cand = detect_model()
    out = {
        "method": "centroid_depth_hierarchy",
        "model": cand["name"],
        "n_docs": n,
        "n_clusters_analyzed": len(results),
        "n_with_hierarchy": n_with_hierarchy,
        "n_total_layers": len(all_layers),
        "nodes": [
            {
                "node_id": n.node_id,
                "n_docs": len(n.doc_indices),
                "meta_concepts": n.meta_concepts,
                "sub_concepts": n.sub_concepts,
                "com_threshold": n.com_threshold,
                "profiles": n.profiles,
                "sample_doc": n.sample_doc,
            }
            for n in results
        ],
    }
    out_path = EXP / "phase18_centroid_hierarchy.json"
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

    print(f"\n[2/2] Running Phase 18 centroid hierarchy...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase18(lens, lens_model, tokenizer, doc_texts, n_clusters=12)


if __name__ == "__main__":
    main()
