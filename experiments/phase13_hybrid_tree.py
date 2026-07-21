"""Phase 13: Hybrid concept tree — J-Lens residual Level-0 + bge-m3 Level-1.

Validated architecture:
  Level 0: docs → J-Lens residual clustering → meta-concept groups
           (cluster inspection confirmed these are conceptually coherent)
  Level 1: within each meta-concept → bge-m3 clustering → sub-concepts
           (bge-m3 is better at fine-grained lexical distinction within a
           coherent concept group)
  Labels:  J-Lens concept extraction at both levels (human-readable)

This combines each method's strength:
  - J-Lens: concept aggregation (correct meta-concept grouping)
  - bge-m3: lexical distinction (fine sub-concept splitting)

vs Phase 11 v2 (bge-m3 at all levels): root was 'mortality/survival/death'
(too generic — bge-m3 mixed unrelated docs at Level 0).
vs J-Lens at all levels: Level 1 produces too few sub-clusters (concept
space is low-dimensional, doesn't distinguish sub-topics).

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase13_hybrid_tree
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
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens,
)
from experiments.phase10_jlens_stage6 import extract_residuals

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


@dataclass
class HybridNode:
    node_id: str
    level: int  # 0 = meta-concept, 1 = sub-concept
    doc_indices: list[int]
    concepts: list[str]
    clustering_method: str  # "jlens" or "bge-m3"
    children: list["HybridNode"] = field(default_factory=list)

    @property
    def n_docs(self) -> int:
        return len(self.doc_indices)


STOP = {"the","and","for","that","with","from","this","are","was","were","been",
        "have","has","will","would","could","should","not","but","into","also",
        "they","them","than","then","when","what","each","more","most","some",
        "such","only","very","just","like","concept","concepts","key","main",
        "topic","study","studies","result","results","method","patient","patients",
        "treatment","associated","compared","significantly","clinical","using",
        "data","analysis","research","health","disease","medical","group",
        "following","above","document","documents","discuss","discusses",
        "related","based","summarized","listed","outlined","recent","several",
        "evidence","suggesting","characterized","indicating","primarily",
        "showed","might","show","seem","appear","require","occur","arise",
        "include","single","similar","three","five","once"}


def extract_concepts(lens, lens_model, tokenizer, docs: list[str],
                     layer: int, n: int = 5) -> list[str]:
    """J-Lens cluster-level concept extraction (Stage 5 validated method)."""
    if len(docs) < 3:
        return []
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = f"What concepts does this text discuss? List {n} one-word concepts.\n\n{doc_block}"
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True, add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"
    lens_logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                                   positions=[-1], max_seq_len=512)
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(25)
    words, seen = [], set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if len(tok) >= 4 and tok.isalpha() and low not in STOP and low not in seen:
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n:
            break
    return words


def _topic_prompt(text: str, tokenizer) -> str:
    msg = f"What is the main topic? One word.\n\n{text[:500]}"
    prefill = "The main topic is"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role":"user","content":msg},{"role":"assistant","content":prefill}],
                tokenize=False, continue_final_message=True, add_generation_prompt=False)
        except:
            pass
    return f"{msg}\n{prefill}"


def run_phase13(embed, lens, lens_model, tokenizer, max_docs=300):
    print("Phase 13: Hybrid concept tree (J-Lens L0 + bge-m3 L1)")
    print(f"{'='*70}")

    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    doc_texts = [r[1] for r in records]
    n = len(doc_texts)
    print(f"\n  {n} NFCorpus documents")

    # 1. bge-m3 embeddings (for Level-1 clustering)
    print(f"  embedding (bge-m3)...", flush=True)
    bge_vecs = np.asarray(embed.embed(doc_texts), dtype=np.float32)

    # 2. J-Lens residuals (for Level-0 clustering)
    print(f"  extracting J-Lens residuals...", flush=True)
    layer = lens.source_layers[-1]
    prompts = [_topic_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer, prompts, layer, max_seq_len=256)

    # 3. Level 0: J-Lens residual clustering → meta-concepts
    print(f"\n  Level 0: J-Lens residual clustering...")
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items() if len(members) >= 5}
    print(f"    {len(l0_valid)} meta-concept clusters (≥5 docs)")

    # 4. Build tree
    tree_nodes = []
    node_counter = [0]

    for l0_cid, l0_members in sorted(l0_valid.items(), key=lambda x: len(x[1]), reverse=True):
        node_counter[0] += 1
        l0_node_id = f"L0_{node_counter[0]}"

        # L0 concept extraction
        l0_docs = [doc_texts[m] for m in l0_members]
        l0_concepts = extract_concepts(lens, lens_model, tokenizer, l0_docs, layer)
        l0_node = HybridNode(
            node_id=l0_node_id, level=0,
            doc_indices=l0_members, concepts=l0_concepts,
            clustering_method="jlens")
        tree_nodes.append(l0_node)

        # 5. Level 1: bge-m3 clustering within meta-concept
        if len(l0_members) >= 15:  # only split if enough docs
            sub_vecs = bge_vecs[l0_members]
            l1_clusters = partition_hdbscan(sub_vecs.tolist())
            l1_valid = {cid: members for cid, members in l1_clusters.items()
                        if len(members) >= 3}

            for l1_cid, l1_relative in sorted(l1_valid.items(),
                                                key=lambda x: len(x[1]), reverse=True):
                node_counter[0] += 1
                l1_members = [l0_members[m] for m in l1_relative]
                l1_docs = [doc_texts[m] for m in l1_members]
                l1_concepts = extract_concepts(lens, lens_model, tokenizer, l1_docs, layer)
                l1_node = HybridNode(
                    node_id=f"{l0_node_id}_L1_{l1_cid}", level=1,
                    doc_indices=l1_members, concepts=l1_concepts,
                    clustering_method="bge-m3")
                l0_node.children.append(l1_node)

    # 6. Print tree for audit
    print(f"\n{'='*70}")
    print("HYBRID CONCEPT TREE")
    print(f"{'='*70}")
    for node in tree_nodes:
        print(f"\n[{node.node_id}] L0 ({node.n_docs} docs, J-Lens): {node.concepts}")
        print(f"  sample: {doc_texts[node.doc_indices[0]][:100]}...")
        for child in node.children:
            print(f"  [{child.node_id}] L1 ({child.n_docs} docs, bge-m3): {child.concepts}")
            print(f"    sample: {doc_texts[child.doc_indices[0]][:90]}...")

    # 7. Stats
    n_l1 = sum(len(n.children) for n in tree_nodes)
    print(f"\n{'='*70}")
    print(f"Stats: {len(tree_nodes)} L0 nodes, {n_l1} L1 nodes")

    # 8. Serialize
    def _serialize(node):
        return {
            "node_id": node.node_id, "level": node.level,
            "n_docs": node.n_docs, "concepts": node.concepts,
            "method": node.clustering_method,
            "children": [_serialize(c) for c in node.children],
            "sample": doc_texts[node.doc_indices[0]][:150] if node.doc_indices else "",
        }

    out = {
        "method": "hybrid_concept_tree",
        "n_docs": n,
        "n_l0": len(tree_nodes),
        "n_l1": n_l1,
        "tree": [_serialize(n) for n in tree_nodes],
    }
    out_path = EXP / "phase13_hybrid_tree.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print("[1/2] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model, tokenizer = load_model(cand["local_model_dir"], use_4bit=cand["needs_4bit"])
    print(f"\n[2/2] Wrapping + building hybrid tree...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))
    embed = CachedBgeM3Provider()
    run_phase13(embed, lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
