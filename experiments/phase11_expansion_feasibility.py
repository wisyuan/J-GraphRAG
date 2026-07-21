"""Phase 11 feasibility test: J-Lens concept→subconcept recursive expansion.

Core question: can J-Lens read out subconcepts given a parent concept + docs?

If yes → recursive concept-tree expansion is viable (J-Lens's native
concern-coupling mode, not vector-space linear transforms that failed in
Phase 1-3). If no → stay with flat concept graph + optimization.

Test design:
  1. NFCorpus docs → bge-m3 cluster → pick stable clusters (Stage 5 validated)
  2. For each cluster, extract parent concept (Stage 5 method, L26 readout)
  3. Three expansion prompt modes, read out subconcepts:
     A. "The documents discuss [parent]. Subtopics are" (concept-conditioned)
     B. "The documents discuss [parent]. The specific [parent] types are" (type-conditioned)
     C. "These documents share the concept [parent]. More specific concepts are" (general)
  4. Human + LLM audit: are the subconcepts real and more specific than parent?

Also test: can we do 2 levels? parent → child → grandchild?

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase11_expansion_feasibility
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

STOP = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "sub", "topic", "type", "types", "specific",
    "document", "documents", "discuss", "discusses", "related", "based",
    "study", "studies", "result", "results", "method", "patient", "patients",
    "treatment", "associated", "compared", "significantly", "clinical", "using",
    "data", "analysis", "research", "health", "disease", "medical", "group",
    "following", "above", "main", "key", "subtopic", "subtopics", "category",
    "kind", "sort", "form", "kind", "part", "aspect",
}


def readout_tokens(lens, lens_model, tokenizer, prompt: str,
                   layer: int, top_n: int = 15) -> list[tuple[str, float]]:
    """J-Lens readout at last position, return (token, prob) pairs."""
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512,
    )
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(top_n)
    return [(tokenizer.decode([int(idx)]).strip(), float(p))
            for idx, p in zip(topk.indices.tolist(), topk.values.tolist())]


def filter_content(toks: list[tuple[str, float]], parent: str = "",
                   exclude: set = None) -> list[str]:
    """Filter to content words, exclude parent + stopwords."""
    exclude = exclude or set()
    words = []
    seen = set()
    for tok, _ in toks:
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP
                and low not in seen and low != parent.lower()
                and low not in exclude):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
    return words


def extract_parent_concept(lens, lens_model, tokenizer, docs: list[str],
                           layer: int) -> list[str]:
    """Extract parent concept words for a cluster (Stage 5 method)."""
    doc_block = "\n---\n".join(d[:400] for d in docs[:5])
    user_msg = f"What concepts does this text discuss? List 5 one-word concepts.\n\n{doc_block}"
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True, add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"
    toks = readout_tokens(lens, lens_model, tokenizer, prompt, layer)
    return filter_content(toks)


def expand_concept(lens, lens_model, tokenizer, parent: str, docs: list[str],
                   layer: int, mode: str = "A") -> list[str]:
    """Expand a parent concept into subconcepts via J-Lens readout.

    Three prompt modes test different conditioning strategies.
    """
    doc_block = "\n---\n".join(d[:400] for d in docs[:5])

    if mode == "A":
        # Concept-conditioned: "discuss [parent]. Subtopics are"
        user_msg = (f"These documents discuss {parent}. "
                    f"What are the specific subtopics within {parent}?\n\n{doc_block}")
        prefill = f"The specific subtopics within {parent} are"
    elif mode == "B":
        # Type-conditioned: "specific [parent] types are"
        user_msg = (f"These documents are about {parent}. "
                    f"What specific types or categories of {parent} are discussed?\n\n{doc_block}")
        prefill = f"The specific types of {parent} discussed are"
    else:
        # General: "more specific concepts are"
        user_msg = (f"These documents all relate to {parent}. "
                    f"What more specific concepts do they cover?\n\n{doc_block}")
        prefill = f"The more specific concepts are"

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True, add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    toks = readout_tokens(lens, lens_model, tokenizer, prompt, layer)
    return filter_content(toks, parent=parent, exclude={parent.lower()})


def run_feasibility(embed, lens, lens_model, tokenizer, max_docs=300):
    print("Phase 11: Recursive expansion feasibility test")
    print(f"{'='*70}")

    # 1. Load NFCorpus + cluster (same as Stage 5)
    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    doc_texts = [r[1] for r in records]
    print(f"\n  {len(doc_texts)} NFCorpus documents")

    vecs = np.asarray(embed.embed(doc_texts), dtype=np.float32)
    clusters = partition_hdbscan(vecs.tolist())
    # pick top 6 clusters by size (≥5 members for stable readout)
    top = sorted([(cid, m) for cid, m in clusters.items() if len(m) >= 5],
                 key=lambda x: len(x[1]), reverse=True)[:6]
    print(f"  {len(top)} clusters (≥5 members)")

    layer = lens.source_layers[-1]

    # 2. For each cluster: parent concept + 3 expansion modes
    results = {}
    for ci, (cid, members) in enumerate(top):
        docs = [doc_texts[m] for m in members]
        print(f"\n{'='*60}")
        print(f"Cluster {cid} ({len(members)} docs)")
        print(f"  sample: {docs[0][:100]}...")

        # Parent concept
        parents = extract_parent_concept(lens, lens_model, tokenizer, docs, layer)
        parent = parents[0] if parents else "?"
        print(f"  parent concept: {parents[:3]}")
        print(f"  using parent: {parent}")

        # 3 expansion modes
        cluster_result = {"parent": parent, "all_parents": parents[:5],
                          "n_docs": len(members), "sample": docs[0][:150]}
        for mode in ("A", "B", "C"):
            children = expand_concept(lens, lens_model, tokenizer, parent, docs, layer, mode)
            cluster_result[f"children_{mode}"] = children
            print(f"  mode {mode} children: {children}")

        # Depth-2 test: pick first child, expand again
        if cluster_result["children_A"]:
            child = cluster_result["children_A"][0]
            grandchildren = expand_concept(lens, lens_model, tokenizer, child, docs, layer, "A")
            cluster_result["grandchild_of"] = child
            cluster_result["grandchildren"] = grandchildren
            print(f"  depth-2: '{parent}' → '{child}' → {grandchildren}")

        results[str(cid)] = cluster_result

    # 3. Summary + manual audit guide
    print(f"\n{'='*70}")
    print("MANUAL AUDIT GUIDE")
    print(f"{'='*70}")
    print("For each cluster, check: are children MORE SPECIFIC than parent?")
    print("Are grandchildren MORE SPECIFIC than children?\n")
    for cid, r in results.items():
        print(f"Cluster {cid} ({r['n_docs']}d): parent={r['parent']}")
        print(f"  children: {r['children_A']}")
        if r.get("grandchildren"):
            print(f"  depth-2: {r['parent']} → {r['grandchild_of']} → {r['grandchildren']}")
        print()

    out_path = EXP / "phase11_expansion_feasibility.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"saved to {out_path}")
    return results


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Wrapping + running feasibility test...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    embed = CachedBgeM3Provider()
    run_feasibility(embed, lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
