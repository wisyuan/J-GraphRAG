"""Dual-level retrieval — query vector replaces keyword extraction.

Adopted verbatim in algorithm from:

- ``lightrag_retrieve``: ``experiments/phase50_lightrag_j.py:293`` — local
  channel query x entity-embedding top-20, global channel query x
  relation-embedding top-20 with weight = sim x prob spread onto both
  endpoint entities, one-hop neighbor expansion decayed by 0.5, chunk score
  = MAX entity weight over the chunk's entities. **MAX aggregation must not
  be changed to SUM**: on the validated 44/232-entity graphs SUM collapsed
  to a query-independent hub ranking (multi-concept overview chunks always
  win; medical ACC 1.8%). MAX keeps the assembly faithful to LightRAG's
  per-element ordering.
- ``merge_topk``: ``experiments/phase50_lightrag_j.py:351`` — graph-ranked
  chunks first, naive-bge backfill (LightRAG hybrid mode), same-score ties
  broken by bge cosine.
- ``interleave_rankings``: ``experiments/phase53_textside_entities.py:782``
  — graph ranking and naive top-10 fill the final top-10 round-robin.

Seed queries go through VectorStore.query; one-hop expansion through
GraphStore.neighbors; the query embedding is produced by an injected
EmbedProvider (see ``GraphIndex.retrieve``).

Import-safe: pure CPU/numpy, no disk or model access at import time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .index import NS_ENTITIES, NS_RELATIONS

if TYPE_CHECKING:
    from .index import GraphIndex

# ── Validated hyperparameters (do not change without re-validation) ──────
ENT_TOP_K = 20        # local retrieval: query -> entity top-k
REL_TOP_K = 20        # global retrieval: query -> relation top-k
NEIGHBOR_DECAY = 0.5  # high-order relatedness: one-hop weight decay
TOP_K = 10            # final context size


def lightrag_retrieve(
    index: "GraphIndex",
    query_vec: np.ndarray,
    ent_top_k: int = ENT_TOP_K,
    rel_top_k: int = REL_TOP_K,
    neighbor_decay: float = NEIGHBOR_DECAY,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Dual-level retrieval → ({chunk_id: score}, debug).

    (ii) keyword matching: query vec vs entity texts (local) and relation
    texts (global); relation hits weight endpoint entities by sim x prob.
    (iii) high-order relatedness: 1-hop neighbors of hit entities and of
    hit relations' endpoints, weight decayed by ``neighbor_decay``.
    Chunk score = MAX entity weight over the chunk's entities (see module
    docstring — SUM is validated-bad).
    """
    qv = np.asarray(query_vec, dtype=np.float64).ravel()
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    ent_w: dict[str, float] = {}

    hit_entities = [(eid, s) for eid, s in
                    index.vectors.query(NS_ENTITIES, qv, ent_top_k) if s > 0]
    for eid, s in hit_entities:
        ent_w[eid] = s

    hit_relations = []
    for rid, s in index.vectors.query(NS_RELATIONS, qv, rel_top_k):
        if s <= 0:
            continue
        r = index.relations[int(rid)]
        w = float(s) * r["prob"]
        hit_relations.append({"text": r["text"], "sim": float(s),
                              "prob": r["prob"], "completed": r["completed"]})
        for ep in (str(r["a"]), str(r["b"])):
            ent_w[ep] = max(ent_w.get(ep, 0.0), w)

    for e, w in list(ent_w.items()):
        for nb, _edge_id, _ew in index.graph.neighbors(e):
            ent_w[nb] = max(ent_w.get(nb, 0.0), neighbor_decay * w)

    chunk_score: dict[str, float] = {}
    for e, w in ent_w.items():
        for cid in index.entities[int(e)]["chunks"]:
            if w > chunk_score.get(cid, 0.0):
                chunk_score[cid] = w
    debug = {
        "hit_entities": [(index.entities[int(eid)]["name"], round(s, 4))
                         for eid, s in hit_entities[:10]],
        "hit_relations": hit_relations[:10],
        "n_entities_weighted": len(ent_w),
    }
    return chunk_score, debug


def merge_topk(
    scores: dict[str, float],
    b0_ids: list[str],
    bge_sim: dict[str, float] | None = None,
    top_k: int = TOP_K,
) -> list[str]:
    """Graph-ranked chunks first, bge (naive) backfill — LightRAG hybrid mode.

    Chunks sharing the same max entity weight (e.g. all chunks of the top
    entity) are ordered by bge cosine to the query — deterministic and keeps
    the naive channel inside the graph-ranked prefix, as LightRAG's hybrid
    mode does within its token budget.
    """
    bge_sim = bge_sim or {}
    ranked = [cid for cid, _s in sorted(
        scores.items(), key=lambda x: (x[1], bge_sim.get(x[0], 0.0)),
        reverse=True)]
    merged = ranked[:top_k]
    for cid in b0_ids:
        if len(merged) >= top_k:
            break
        if cid not in merged:
            merged.append(cid)
    return merged[:top_k]


def interleave_rankings(
    graph_ranked: list[str],
    naive_ids: list[str],
    top_k: int = TOP_K,
) -> list[str]:
    """Round-robin interleave of graph ranking and naive top-k into top-k.

    Verbatim port of the phase53 hybrid arm (:782): each round takes the
    first not-yet-merged chunk from the graph pool, then from the naive
    pool, until ``top_k`` is filled.
    """
    merged: list[str] = []
    pools = [iter(graph_ranked), iter(naive_ids)]
    while len(merged) < top_k:
        progressed = False
        for pool in pools:
            if len(merged) >= top_k:
                break
            for cid in pool:
                if cid not in merged:
                    merged.append(cid)
                    progressed = True
                    break
        if not progressed:
            # Both pools exhausted with fewer unique chunks than top_k
            # (tiny corpora) — the experiment loop would spin forever here;
            # validated runs always had len(graph)+len(naive) >= top_k.
            break
    return merged


if __name__ == "__main__":  # pragma: no cover
    # CPU selftest with a hand-built toy index (no models involved).
    from .index import GraphIndex
    from .stores.local import JsonGraphStore, NumpyVectorStore

    class _MockEmbed:
        """One-hot registry embedder: each new text gets the next basis vector,
        so distinct texts are exactly orthogonal (deterministic sims)."""

        def __init__(self) -> None:
            self._reg: dict[str, int] = {}

        @property
        def dim(self) -> int:
            return 8

        def embed(self, texts: list[str]) -> list[list[float]]:
            out = []
            for t in texts:
                if t not in self._reg:
                    self._reg[t] = len(self._reg)
                v = [0.0] * self.dim
                v[self._reg[t]] = 1.0
                out.append(v)
            return out

    graph = JsonGraphStore()
    vectors = NumpyVectorStore()
    idx = GraphIndex(graph, vectors, {"c0": "alpha beta", "c1": "beta gamma",
                                      "c2": "unrelated"})
    idx.entities = [
        {"name": "alpha", "members": ["alpha"], "chunks": ["c0"],
         "roles": [], "text": "alpha"},
        {"name": "beta", "members": ["beta"], "chunks": ["c0", "c1"],
         "roles": [], "text": "beta"},
        {"name": "gamma", "members": ["gamma"], "chunks": ["c1"],
         "roles": [], "text": "gamma"},
    ]
    for i in range(3):
        graph.add_node(str(i), name=idx.entities[i]["name"])
    idx.relations = [{"a": 0, "b": 1, "relation": "related_to", "prob": 1.0,
                      "text": "alpha related to beta", "completed": False}]
    graph.add_edge("0", "1", weight=1.0, relation="related_to")

    embed = _MockEmbed()
    idx.build_embeddings(embed)
    idx.build_chunk_embeddings(embed)

    # Query vector identical to entity 0's stored vector → entity 0 wins.
    qv = np.asarray(embed.embed(["alpha"]), dtype=np.float64)[0]
    scores, debug = lightrag_retrieve(idx, qv)
    assert scores, "no chunks scored"
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    assert ranked[0][0] == "c0", f"expected c0 first, got {ranked}"
    # MAX aggregation: c0's score equals entity 0's weight (not a sum).
    assert abs(scores["c0"] - max(1.0, NEIGHBOR_DECAY * 1.0)) < 1e-9, scores
    # One-hop decay: gamma is NOT a neighbor of entity 0; beta (neighbor) got
    # decayed weight -> c1 scored via beta at 0.5.
    assert abs(scores["c1"] - NEIGHBOR_DECAY) < 1e-9, scores

    m = merge_topk(scores, ["c2"], top_k=2)
    assert m[0] == "c0" and len(m) == 2
    il = interleave_rankings(["g1", "g2", "g3"], ["n1", "n2"], top_k=4)
    assert il == ["g1", "n1", "g2", "n2"], il
    print("OK")
