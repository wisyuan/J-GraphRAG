"""Phase 16: 纯 J-Lens 阈值控制递归展开。

核心命题：不是所有概念簇都该展开。类比 HDBSCAN 只在密度支持时分簇，
J-Lens 展开也应只在概念质量足够时才递归。

Phase 16a 前置实验验证了跨域 POS 分布差异：
  - 医学域 noun=68%, verb=18%, frag=21%
  - 小说域 noun=50%, verb=50% (叙事性强)
  - 代码域 frag=100% (需 tree-sitter 预处理)

结论：单一 noun_ratio 阈值不够，需要组合信号。

停止准则（三道门槛，全部满足才展开）：
  门 1：corpus_hit_rate ≥ 0.4  — 概念词必须在节点文档中验证
  门 2：bpe_frag_ratio ≤ 0.3   — BPE 碎片占比不能太高
  门 3：verb_ratio ≤ 0.4       — 动词(叙事/泛化)占比不能太高

展开方式：先验条件展开（meta-concept 作 prompt prior），不重新聚类——
这是"纯 J-Lens"。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase16_threshold_expansion
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.partition import partition_hdbscan
from experiments.concept_quality import (
    optimize_concepts, build_corpus_term_freq, _get_wordnet_nouns,
)
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals, build_concern_prompt
from experiments.phase16a_cross_domain_pos import (
    classify_concept_pos, concept_quality_score, is_likely_bpe_fragment,
    extract_cluster_concepts, STOP_WORDS,
    ConceptQualityMetrics,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── Threshold configuration ───────────────────────────────────────────

# Calibrated from Phase 16a cross-domain POS data.
# A node is expanded ONLY if ALL three conditions are met.
THRESHOLDS = {
    "min_corpus_hit_rate": 0.4,   # 门 1: ≥40% concepts verified in node docs
    "max_bpe_frag_ratio": 0.3,    # 门 2: ≤30% BPE fragments
    "max_verb_ratio": 0.4,        # 门 3: ≤40% verbs (narrative/generic)
    "min_effective_concepts": 3,  # need ≥3 quality concepts to expand
    "max_depth": 3,               # recursion depth cap
}


@dataclass
class ThresholdNode:
    """A node in the threshold-controlled concept tree."""
    node_id: str
    depth: int
    doc_indices: list[int]
    concepts: list[str]           # raw J-Lens extracted concepts
    quality: dict                 # POS metrics dict
    children: list["ThresholdNode"] = field(default_factory=list)
    parent_id: str | None = None
    expand_decision: str = "unknown"  # "expanded" / "stopped:<reason>" / "leaf"
    stop_reason: str | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_indices)


# ── Expansion gate ────────────────────────────────────────────────────

def should_expand(
    metrics: ConceptQualityMetrics,
    thresholds: dict | None = None,
) -> tuple[bool, str]:
    """Decide whether to expand a node based on concept quality metrics.

    Three gates (all must pass):
      1. corpus_hit_rate ≥ min_corpus_hit_rate
      2. bpe_frag_ratio ≤ max_bpe_frag_ratio
      3. verb_ratio ≤ max_verb_ratio

    Plus a minimum effective concept count.

    Returns (should_expand, reason).
    """
    th = thresholds or THRESHOLDS

    if metrics.n_effective < th["min_effective_concepts"]:
        return False, f"too_few_effective ({metrics.n_effective}<{th['min_effective_concepts']})"

    if metrics.corpus_hit_ratio < th["min_corpus_hit_rate"]:
        return False, f"low_corpus_hit ({metrics.corpus_hit_ratio:.0%}<{th['min_corpus_hit_rate']:.0%})"

    if metrics.bpe_fragment_ratio > th["max_bpe_frag_ratio"]:
        return False, f"high_bpe_frag ({metrics.bpe_fragment_ratio:.0%}>{th['max_bpe_frag_ratio']:.0%})"

    if metrics.verb_ratio > th["max_verb_ratio"]:
        return False, f"high_verb_ratio ({metrics.verb_ratio:.0%}>{th['max_verb_ratio']:.0%})"

    return True, "expanded"


# ── Prior-conditioned sub-concept expansion ───────────────────────────

def expand_with_prior(
    lens, lens_model, tokenizer,
    meta_concept: str,
    doc_texts: list[str],
    layer: int,
    n_words: int = 8,
) -> list[str]:
    """Extract sub-concepts using meta-concept as prompt prior.

    This is the Phase 14b core logic formalized. Instead of re-clustering,
    we condition the J-Lens readout on the parent concept.

    Prompt design (learned from Phase 16 v1 failure):
      v1 prompt "What specific types or aspects..." → produced meta-words
      (aspects, terms, factors) because "types/aspects" primes the model
      to enumerate structural categories, not domain concepts.

      v2 prompt: "What are the key processes and mechanisms related to
      {meta_concept}?" → forces domain-specific nouns (progression,
      inhibition, synthesis) rather than meta-structural words.

    Returns list of sub-concept words.
    """
    if len(doc_texts) < 3:
        return []

    doc_block = "\n---\n".join(d[:300] for d in doc_texts[:6])
    user_msg = (
        f"The following documents are about {meta_concept}. "
        f"What are the key biological processes and mechanisms related to "
        f"{meta_concept} discussed in these texts? "
        f"List {n_words} specific domain terms.\n\n{doc_block}"
    )
    prefill = f"The key processes related to {meta_concept} are"
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

    # Additional stop words: meta-structural words that the model generates
    # when asked to "list" things, regardless of domain. These are NOT
    # domain concepts — they're structural/meta vocabulary.
    META_WORDS = {
        "aspects", "terms", "factors", "elements", "mentions", "references",
        "topics", "discussions", "descriptions", "factors", "issues",
        "points", "areas", "fields", "domains", "categories", "types",
        "kinds", "sorts", "forms", "versions", "methods", "approaches",
        "techniques", "procedures", "processes", "steps", "stages",
        "levels", "degrees", "ranges", "amounts", "numbers", "rates",
    }

    words, seen = [], set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_WORDS
                and low not in META_WORDS and low not in seen
                and low != meta_concept.lower()):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── Recursive tree builder ────────────────────────────────────────────

def build_threshold_tree(
    lens, lens_model, tokenizer,
    doc_texts: list[str],
    jlens_vecs: np.ndarray,
    layer: int,
    thresholds: dict | None = None,
) -> ThresholdNode:
    """Build a concept tree with threshold-controlled expansion.

    L0: J-Lens residual clustering → meta-concept clusters
    L1+: Prior-conditioned expansion (no re-clustering) with threshold gate

    The tree only expands nodes that pass the quality threshold. Nodes that
    fail are marked as leaves with a diagnostic stop_reason.
    """
    th = thresholds or THRESHOLDS
    node_counter = [0]

    def _quality_dict(m: ConceptQualityMetrics) -> dict:
        return {
            "n_concepts": m.n_concepts,
            "noun_ratio": round(m.noun_ratio, 2),
            "verb_ratio": round(m.verb_ratio, 2),
            "bpe_frag_ratio": round(m.bpe_fragment_ratio, 2),
            "corpus_hit_ratio": round(m.corpus_hit_ratio, 2),
            "n_effective": m.n_effective,
            "pos_counts": m.pos_counts,
        }

    def _build(doc_indices: list[int], depth: int, parent_id: str | None,
               cluster_vecs: np.ndarray | None = None) -> ThresholdNode:
        node_counter[0] += 1
        node_id = f"d{depth}_n{node_counter[0]}"

        docs = [doc_texts[i] for i in doc_indices]
        concepts = extract_cluster_concepts(
            lens, lens_model, tokenizer, docs, layer) if len(docs) >= 3 else []
        metrics = concept_quality_score(concepts, docs)

        node = ThresholdNode(
            node_id=node_id, depth=depth,
            doc_indices=doc_indices, concepts=concepts,
            quality=_quality_dict(metrics),
            parent_id=parent_id,
        )

        # Check depth cap
        if depth >= th["max_depth"]:
            node.expand_decision = "stopped"
            node.stop_reason = "max_depth"
            return node

        # Check expansion gate
        can_expand, reason = should_expand(metrics, th)
        if not can_expand:
            node.expand_decision = "stopped"
            node.stop_reason = reason
            return node

        # Expand: for each effective meta-concept, do prior-conditioned readout
        effective_concepts = metrics.effective_concepts[:5]  # top 5
        child_node_ids = []

        for mc in effective_concepts:
            # Prior-conditioned sub-concept extraction
            sub_concepts = expand_with_prior(
                lens, lens_model, tokenizer, mc, docs, layer)

            if len(sub_concepts) < 2:
                continue  # no sub-structure found

            # Check sub-concept quality (gate 3: diversity)
            sub_lower = [s.lower() for s in sub_concepts]
            unique_ratio = len(set(sub_lower)) / max(1, len(sub_lower))
            if unique_ratio < 0.5:
                continue  # sub-concepts too repetitive

            # Create child node
            node_counter[0] += 1
            child_id = f"d{depth+1}_n{node_counter[0]}"
            sub_metrics = concept_quality_score(sub_concepts, docs)
            child = ThresholdNode(
                node_id=child_id, depth=depth + 1,
                doc_indices=doc_indices,  # same docs (prior expansion doesn't split)
                concepts=sub_concepts,
                quality=_quality_dict(sub_metrics),
                parent_id=node_id,
                expand_decision="leaf",
                stop_reason="prior_expansion_leaf",
            )
            # Recurse into child if it passes the gate
            child_can_expand, child_reason = should_expand(sub_metrics, th)
            if child_can_expand and depth + 1 < th["max_depth"]:
                # For prior expansion, children share the same docs.
                # Further recursion would use sub-concepts as new priors.
                # This is where true depth happens.
                grand_children = []
                for sc in sub_metrics.effective_concepts[:3]:
                    gc_concepts = expand_with_prior(
                        lens, lens_model, tokenizer, sc, docs, layer, n_words=5)
                    if len(gc_concepts) >= 2:
                        node_counter[0] += 1
                        gc_id = f"d{depth+2}_n{node_counter[0]}"
                        gc_metrics = concept_quality_score(gc_concepts, docs)
                        gc = ThresholdNode(
                            node_id=gc_id, depth=depth + 2,
                            doc_indices=doc_indices,
                            concepts=gc_concepts,
                            quality=_quality_dict(gc_metrics),
                            parent_id=child_id,
                            expand_decision="leaf",
                            stop_reason="max_depth_reached",
                        )
                        grand_children.append(gc)
                child.children = grand_children
                if grand_children:
                    child.expand_decision = "expanded"

            node.children.append(child)

        if node.children:
            node.expand_decision = "expanded"
        else:
            node.expand_decision = "stopped"
            node.stop_reason = "no_valid_sub_concepts"

        return node

    # L0: cluster all docs by J-Lens residuals
    print(f"  L0: clustering {len(doc_texts)} docs by J-Lens residuals...")
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    print(f"  L0: {len(l0_valid)} clusters (≥5 docs)")

    # Build a root node containing all L0 clusters as children
    node_counter[0] += 1
    root = ThresholdNode(
        node_id="root", depth=0,
        doc_indices=list(range(len(doc_texts))),
        concepts=[], quality={},
        expand_decision="expanded",
    )

    for cid, members in sorted(l0_valid.items(),
                                key=lambda x: len(x[1]), reverse=True):
        child = _build(members, 1, "root")
        root.children.append(child)

    return root


# ── Tree printing & stats ─────────────────────────────────────────────

def print_tree(node: ThresholdNode, doc_texts: list[str], indent: str = ""):
    """Pretty-print the threshold-controlled tree."""
    if node.node_id != "root":
        concepts_str = ", ".join(node.concepts[:5])
        q = node.quality
        decision = node.expand_decision
        reason = f" ({node.stop_reason})" if node.stop_reason else ""
        print(f"{indent}[{node.node_id}] d{node.depth} ({node.n_docs}d) "
              f"[{decision}{reason}]: {concepts_str}")
        if q:
            print(f"{indent}  noun={q.get('noun_ratio',0):.0%} "
                  f"verb={q.get('verb_ratio',0):.0%} "
                  f"frag={q.get('bpe_frag_ratio',0):.0%} "
                  f"corpus={q.get('corpus_hit_ratio',0):.0%}")
        if node.n_docs > 0 and node.doc_indices:
            sample = doc_texts[node.doc_indices[0]][:80]
            print(f"{indent}  sample: {sample}...")
    for child in node.children:
        print_tree(child, doc_texts, indent + "  ")


def tree_stats(node: ThresholdNode) -> dict:
    """Compute tree statistics including stop-reason distribution."""
    stats = {
        "n_nodes": 0, "n_expanded": 0, "n_stopped": 0,
        "max_depth": 0, "stop_reasons": {},
        "nodes_at_depth": {},
    }

    def _walk(n):
        if n.node_id == "root":
            for c in n.children:
                _walk(c)
            return
        stats["n_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], n.depth)
        stats["nodes_at_depth"][n.depth] = stats["nodes_at_depth"].get(n.depth, 0) + 1
        if n.expand_decision == "expanded":
            stats["n_expanded"] += 1
        else:
            stats["n_stopped"] += 1
            reason = n.stop_reason or "unknown"
            # Simplify reason (remove numbers in parens)
            reason_key = re.sub(r'\s*\(.*\)', '', reason)
            stats["stop_reasons"][reason_key] = stats["stop_reasons"].get(reason_key, 0) + 1
        for c in n.children:
            _walk(c)

    _walk(node)
    stats["expand_rate"] = stats["n_expanded"] / max(1, stats["n_nodes"])
    return stats


# ── Main ──────────────────────────────────────────────────────────────

def run_phase16(lens, lens_model, tokenizer, doc_texts: list[str],
                max_docs: int = 300):
    print("Phase 16: Threshold-controlled pure J-Lens expansion")
    print(f"{'='*70}")
    print(f"\n  Thresholds: {THRESHOLDS}")

    n = len(doc_texts)
    layer = lens.source_layers[-1]

    # 1. Extract J-Lens residuals for L0 clustering
    print(f"\n[1/3] Extracting J-Lens residuals ({n} docs)...")
    prompts = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer, prompts, layer)
    print(f"  residuals: {jlens_vecs.shape}")

    # 2. Build threshold-controlled tree
    print(f"\n[2/3] Building threshold-controlled tree...")
    tree = build_threshold_tree(
        lens, lens_model, tokenizer, doc_texts, jlens_vecs, layer)

    # 3. Print tree + stats
    print(f"\n[3/3] Tree audit:")
    print(f"{'='*70}")
    print_tree(tree, doc_texts)

    stats = tree_stats(tree)
    print(f"\n{'='*70}")
    print(f"TREE STATISTICS")
    print(f"{'='*70}")
    print(f"  Total nodes:      {stats['n_nodes']}")
    print(f"  Expanded:         {stats['n_expanded']} ({stats['expand_rate']:.0%})")
    print(f"  Stopped:          {stats['n_stopped']}")
    print(f"  Max depth:        {stats['max_depth']}")
    print(f"  Nodes per depth:  {stats['nodes_at_depth']}")
    print(f"  Stop reasons:")
    for reason, count in sorted(stats["stop_reasons"].items(),
                                 key=lambda x: -x[1]):
        print(f"    {reason:30} {count}")

    # Serialize
    def _serialize(node):
        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "n_docs": node.n_docs,
            "concepts": node.concepts,
            "quality": node.quality,
            "expand_decision": node.expand_decision,
            "stop_reason": node.stop_reason,
            "children": [_serialize(c) for c in node.children],
            "sample": doc_texts[node.doc_indices[0]][:150] if node.doc_indices else "",
        }

    cand = detect_model()
    out = {
        "method": "threshold_controlled_jlens_expansion",
        "model": cand["name"],
        "layer": layer,
        "n_docs": n,
        "thresholds": THRESHOLDS,
        "tree_stats": stats,
        "tree": _serialize(tree),
    }
    out_path = EXP / "phase16_threshold_tree.json"
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

    print(f"\n[2/2] Wrapping + running threshold expansion...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    # Load NFCorpus (medical, primary domain)
    records = load_beir_fine_records("nfcorpus", max_docs=300)
    doc_texts = [r[1] for r in records]
    run_phase16(lens, lens_model, tokenizer, doc_texts)


if __name__ == "__main__":
    main()
