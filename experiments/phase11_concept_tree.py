"""Phase 11 v2: Recursive concept-tree expansion via re-clustering.

Phase 11 v1 showed that concept→subconcept readout fails at chunk level
(artifacts dominate). This version uses the correct approach: recursive
re-clustering + cluster-level concept extraction (Stage 5's validated method,
80% accuracy).

Tree construction:
  Level 0: all docs → bge-m3 embed → HDBSCAN → clusters
           → each cluster: J-Lens concept extraction (multi-doc prompt)
  Level 1: within each cluster → re-embed (sub-cluster docs) → HDBSCAN → sub-clusters
           → each sub-cluster: J-Lens concept extraction
  Level 2: repeat until clusters < min_size

This is structurally identical to Phase 1-3's build_concept_tree, but:
  - Node labels are J-Lens concept words (human-readable) not vector centroids
  - Phase 1-3 failed because vector-space expansion broke retrieval geometry;
    J-Lens expansion is conceptual (discrete words), not geometric (vectors)

Validation:
  1. Tree quality: are child concepts MORE SPECIFIC than parent?
  2. Retrieval: does tree-guided search beat flat graph propagation?

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase11_concept_tree
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.concept_quality import optimize_concepts, build_corpus_term_freq
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


@dataclass
class ConceptNode:
    """A node in the concept tree."""
    node_id: str
    depth: int
    doc_indices: list[int]
    concepts: list[str]  # J-Lens extracted concept words
    children: list["ConceptNode"] = field(default_factory=list)
    parent_id: str | None = None

    @property
    def n_docs(self) -> int:
        return len(self.doc_indices)


def extract_cluster_concepts(lens, lens_model, tokenizer,
                              docs: list[str], layer: int,
                              n_words: int = 5) -> list[str]:
    """Extract concept words for a cluster (Stage 5 validated method).

    Multi-doc prompt → stable residual → L26 readout → content-word filter.
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List {n_words} one-word concepts.\n\n{doc_block}")
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True, add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(20)

    STOP = {"the", "and", "for", "that", "with", "from", "this", "are", "was",
            "were", "been", "have", "has", "will", "would", "could", "should",
            "not", "but", "into", "also", "they", "them", "than", "then", "when",
            "what", "each", "more", "most", "some", "such", "only", "very", "just",
            "like", "concept", "concepts", "key", "main", "topic", "study", "studies",
            "result", "results", "method", "patient", "patients", "treatment",
            "associated", "compared", "significantly", "clinical", "using", "data",
            "analysis", "research", "health", "disease", "medical", "group",
            "following", "above", "document", "documents", "discuss", "discusses",
            "related", "based", "summarized", "listed", "outlined"}

    words = []
    seen = set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


def build_concept_tree(embed, lens, lens_model, tokenizer,
                        doc_texts: list[str],
                        max_depth: int = 3,
                        min_cluster_size: int = 5,
                        min_split_size: int = 15,
                        node_counter: list[int] = None) -> ConceptNode:
    """Recursively build a concept tree via re-clustering.

    At each level:
      1. Embed docs → HDBSCAN → clusters
      2. Each cluster → J-Lens concept extraction
      3. If cluster ≥ min_split_size and depth < max_depth: recurse into it
    """
    if node_counter is None:
        node_counter = [0]
    layer = lens.source_layers[-1]

    def _build(doc_indices: list[int], depth: int, parent_id: str | None) -> ConceptNode:
        node_counter[0] += 1
        node_id = f"d{depth}_n{node_counter[0]}"

        # Extract concepts for this node's docs
        docs = [doc_texts[i] for i in doc_indices]
        concepts = extract_cluster_concepts(lens, lens_model, tokenizer,
                                             docs, layer) if len(docs) >= 3 else []
        node = ConceptNode(
            node_id=node_id, depth=depth,
            doc_indices=doc_indices, concepts=concepts,
            parent_id=parent_id)

        # Try to split into sub-clusters
        if depth >= max_depth or len(doc_indices) < min_split_size:
            return node

        # Re-embed and cluster this subset
        sub_texts = [doc_texts[i] for i in doc_indices]
        sub_vecs = np.asarray(embed.embed(sub_texts), dtype=np.float32)
        sub_clusters = partition_hdbscan(sub_vecs.tolist())

        # Only recurse into clusters with ≥ min_cluster_size
        valid = {cid: members for cid, members in sub_clusters.items()
                 if len(members) >= min_cluster_size}
        if len(valid) < 2:
            return node  # can't split further

        for cid, members in valid.items():
            # map back to original doc indices
            child_indices = [doc_indices[m] for m in members]
            child = _build(child_indices, depth + 1, node_id)
            node.children.append(child)

        return node

    return _build(list(range(len(doc_texts))), 0, None)


def print_tree(node: ConceptNode, doc_texts: list[str], indent: str = ""):
    """Pretty-print the concept tree for manual audit."""
    print(f"{indent}[{node.node_id}] d{node.depth} ({node.n_docs} docs): {node.concepts}")
    if node.n_docs > 0:
        print(f"{indent}  sample: {doc_texts[node.doc_indices[0]][:80]}...")
    for child in node.children:
        print_tree(child, doc_texts, indent + "  ")


def tree_stats(node: ConceptNode) -> dict:
    """Compute tree statistics."""
    def _walk(n, depth, stats):
        stats["n_nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        stats["nodes_at_depth"][depth] = stats["nodes_at_depth"].get(depth, 0) + 1
        if not n.children:
            stats["n_leaves"] += 1
        for c in n.children:
            _walk(c, depth + 1, stats)
    stats = {"n_nodes": 0, "max_depth": 0, "n_leaves": 0, "nodes_at_depth": {}}
    _walk(node, 0, stats)
    return stats


def run_phase11(embed, lens, lens_model, tokenizer, max_docs=200):
    print("Phase 11 v2: Recursive concept-tree expansion")
    print(f"{'='*70}")

    # 1. Load corpus
    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    doc_texts = [r[1] for r in records]
    print(f"\n  {len(doc_texts)} NFCorpus documents")

    # 2. Build concept tree
    print(f"\n  Building concept tree (max_depth=3, min_split=15)...")
    tree = build_concept_tree(embed, lens, lens_model, tokenizer,
                               doc_texts, max_depth=3, min_split_size=15)

    # 3. Print tree for manual audit
    print(f"\n{'='*70}")
    print("CONCEPT TREE (manual audit: are children more specific than parent?)")
    print(f"{'='*70}")
    print_tree(tree, doc_texts)

    # 4. Stats
    stats = tree_stats(tree)
    print(f"\n{'='*70}")
    print(f"Tree stats: {stats}")

    # 5. Serialize
    def _serialize(node):
        return {
            "node_id": node.node_id,
            "depth": node.depth,
            "n_docs": node.n_docs,
            "concepts": node.concepts,
            "parent": node.parent_id,
            "children": [_serialize(c) for c in node.children],
            "sample_doc": doc_texts[node.doc_indices[0]][:150] if node.doc_indices else "",
        }

    out = {
        "method": "recursive_concept_tree",
        "n_docs": len(doc_texts),
        "tree_stats": stats,
        "tree": _serialize(tree),
    }
    out_path = EXP / "phase11_concept_tree.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Wrapping + building concept tree...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    embed = CachedBgeM3Provider()
    run_phase11(embed, lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
