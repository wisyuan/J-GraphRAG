"""Relation readout between concept pairs (J-Lens workspace readout).

Ported from:
- experiments/phase27_relation_readout.py — build_relation_prompt
  ("The relationship between A and B is", readout position -1), decode_topk,
  STOP_REL
- experiments/phase28_relation_graph.py — extract_relation, RELATION_TYPES
- experiments/phase40_grounding_test.py — select_relation_pairs (cluster-
  driven pair selection), find_pair_context (co-occurrence / concat context)
- experiments/phase30_cluster_relation_graph.py — cluster_concepts (embed-
  based KMeans clustering; dependency of select_relation_pairs)

The embedding used for pair-selection clustering is injected as an
EmbedProvider (providers/base.py protocol) — this module never imports bge.

Import-safe: torch and sklearn are imported lazily inside functions.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..providers.base import EmbedProvider

# Stop words for relation readout filtering (phase27)
STOP_REL = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study",
    "relationship", "between", "related", "based", "associated",
    "discussed", "discusses", "discuss", "discussing",
    "described", "describes", "describe", "describing",
    "shown", "shows", "show", "showing",
    "found", "finds", "finding", "findings",
    "reported", "reports", "report", "reporting",
    "include", "includes", "including",
    "involve", "involves", "involving",
    "cover", "covers", "covered",
    "focus", "focuses", "focused",
    "address", "addresses", "addressed",
    "explore", "explores", "explored",
    "examine", "examines", "examined",
    "consider", "considers", "considered",
    "analyze", "analyzes", "analyzed",
    "investigate", "investigates", "investigated",
    "highlight", "highlights", "highlighted",
    "demonstrate", "demonstrates", "demonstrated",
    "suggest", "suggests", "suggested",
    "indicate", "indicates", "indicated",
    "reveal", "reveals", "revealed",
    "present", "presents", "presented",
    "provide", "provides", "provided",
    "specific", "specifically", "particular",
    "various", "different", "certain", "general",
    "important", "possible", "available",
    "first", "second", "last", "new", "old",
    "however", "furthermore", "moreover", "additionally",
    "given", "unless", "except", "among", "despite",
    "until", "since", "today", "currently",
    "following", "above", "document", "documents",
    "text", "texts", "passage", "passages",
    "both", "either", "neither", "other", "another",
    "same", "similar", "different",
    "overall", "generally", "typically",
    "result", "results", "method", "methods",
    "patient", "patients", "treatment", "treatments",
    "clinical", "using", "data", "analysis",
    "research", "health", "disease", "medical",
    "group", "groups", "study", "studies",
}

# Meaningful relation words (phase28, from Phase 27 PoC analysis)
RELATION_TYPES = {
    "treatment", "treat", "treats", "treated", "treating", "therapy",
    "therapeutic", "therapies", "reatment",
    "cause", "causes", "caused", "causing", "causal", "caus",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "risk", "risks",
    "component", "contains", "source", "sources",
    "builds", "build", "strengthens", "strengthen",
    "spreads", "spread", "progression", "progress",
    "growth", "grows",
    "inhibits", "inhibit", "inhibition",
    "promotes", "promote",
    "manages", "manage",
    "removal", "removes", "remove",
    "associated", "association",
    "induces", "induce",
    "reduces", "reduce",
    "increases", "increase",
    "affects", "affect",
    "improves", "improve",
    "protects", "protect", "protection",
    "targets", "target",
    "kills", "kill",
    "supports", "support",
    "requires", "require",
    "produces", "produce",
    "regulates", "regulate", "regulated", "regulatory",
    "stimulates", "stimulate",
    "suppresses", "suppress",
    "essential", "crucial", "vital", "integral", "critical",
    "central",
}


def decode_topk(logits_row, tokenizer, n: int = 10, scan: int = 40):
    """Decode top-k content words from logits row."""
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(scan)
    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_REL
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n:
            break
    return results


def build_relation_prompt(doc_text: str, concept_a: str, concept_b: str,
                          tokenizer) -> str:
    """Dual-concept concern prompt for relation readout."""
    user_msg = (
        f"This text discusses {concept_a} and {concept_b}. "
        f"What is the relationship between {concept_a} and {concept_b}? "
        f"Answer with one word.\n\n{doc_text[:600]}"
    )
    prefill = f"The relationship between {concept_a} and {concept_b} is"
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


def extract_relation(
    lens, lens_model, tokenizer,
    concept_a: str, concept_b: str,
    doc_text: str,
    layer: int,
) -> tuple[str | None, float]:
    """Read relation type between two concepts via dual-concept concern.

    Returns (relation_word, probability) or (None, 0) if no meaningful relation.
    """
    prompt = build_relation_prompt(doc_text, concept_a, concept_b, tokenizer)
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)

    for w in words:
        if w["token"].lower() in RELATION_TYPES:
            return w["token"], w["prob"]
    # Return top word even if not in known set (for analysis)
    if words:
        return words[0]["token"], words[0]["prob"]
    return None, 0.0


# ── Pair selection + context (phase40 / phase30) ────────────────────────

def cluster_concepts(concepts: list[str], embedder: EmbedProvider,
                     n_clusters: int | None = None) -> dict:
    """Cluster concept words using dense embeddings + K-Means.

    Returns {cluster_id: {concepts, representative, centroid}}.
    Representative = concept closest to cluster centroid.
    """
    if len(concepts) <= 2:
        return {0: {"concepts": concepts, "representative": concepts[0],
                     "centroid": None}}

    from sklearn.cluster import KMeans

    # Embed concept words
    vecs = np.asarray(embedder.embed(concepts), dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / (norms + 1e-8)

    # Determine cluster count
    if n_clusters is None:
        n_clusters = max(2, min(len(concepts) // 3, 6))
    n_clusters = min(n_clusters, len(concepts) - 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs_norm)
    centroids = kmeans.cluster_centers_

    clusters = {}
    for cid in range(n_clusters):
        members = [concepts[i] for i in range(len(concepts)) if labels[i] == cid]
        if not members:
            continue

        # Find representative (closest to centroid)
        member_indices = [i for i in range(len(concepts)) if labels[i] == cid]
        member_vecs = vecs_norm[member_indices]
        distances = np.linalg.norm(member_vecs - centroids[cid], axis=1)
        rep_idx = member_indices[np.argmin(distances)]
        representative = concepts[rep_idx]

        clusters[cid] = {
            "concepts": members,
            "representative": representative,
            "centroid": centroids[cid].tolist(),
            "n": len(members),
        }

    return clusters


def select_relation_pairs(
    concepts: list[str], embedder: EmbedProvider, max_pairs: int = 0,
) -> tuple[list[tuple[str, str, str]], dict]:
    """Cluster-driven pair selection (phase 30 strategy).

    Intra-cluster: all pairs within each embedding/KMeans cluster.
    Inter-cluster: pairs of cluster representatives (bridges).
    Returns (pairs, clusters); pairs are (layer_type, concept_a, concept_b),
    deduplicated as undirected pairs, capped at max_pairs (0 = no cap).
    """
    clusters = cluster_concepts(concepts, embedder)
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(layer_type: str, a: str, b: str) -> None:
        key = (a, b) if a < b else (b, a)
        if a == b or key in seen:
            return
        seen.add(key)
        pairs.append((layer_type, a, b))

    for cluster in clusters.values():
        members = cluster["concepts"]
        for a, b in combinations(members, 2):
            _add("intra", a, b)
    reps = [clusters[cid]["representative"] for cid in sorted(clusters)]
    for a, b in combinations(reps, 2):
        _add("inter", a, b)

    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    return pairs, clusters


def find_pair_context(
    c_a: str,
    c_b: str,
    concept_chunks: dict[str, list[str]],
    chunk_text_map: dict[str, str],
) -> tuple[str | None, str]:
    """Find context for a concept pair.

    Prefer a chunk containing both concepts; otherwise concatenate the
    shortest chunk containing a with the shortest containing b.
    Returns (text, strategy) or (None, "missing").
    """
    chunks_a = [cid for cid in concept_chunks.get(c_a, []) if cid in chunk_text_map]
    chunks_b = [cid for cid in concept_chunks.get(c_b, []) if cid in chunk_text_map]
    common = set(chunks_a) & set(chunks_b)
    if common:
        cid = min(common, key=lambda c: len(chunk_text_map[c]))
        return chunk_text_map[cid], "shared"
    if chunks_a and chunks_b:
        ta = chunk_text_map[min(chunks_a, key=lambda c: len(chunk_text_map[c]))]
        tb = chunk_text_map[min(chunks_b, key=lambda c: len(chunk_text_map[c]))]
        return f"{ta}\n{tb}", "concat"
    return None, "missing"
