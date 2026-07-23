"""High-level orchestration: chunks → GraphIndex.

``build_index`` wires the validated phases into one call:

1. lens.extract_concepts (concepts + roles + ws vectors, one forward pass)
2. extract/filter concept & role filtering (parallel-worktree module —
   lazily imported; falls back to a TODO-isolated no-op when unavailable)
3. extract/entities text-side entity detection (same lazy-isolation)
4. relation readout over per-chunk co-occurring concept pairs → edges
5. lens release (del + torch.cuda.empty_cache — the phase53 VRAM
   choreography at experiments/phase53_textside_entities.py:713: ALL lens
   readouts finish and the GPU model is freed BEFORE the embedding
   provider is constructed/loaded; on an 8GB box bge-m3 and the 4bit LLM
   must not co-reside)
6. index assembly + embeddings (entity/relation texts "name: role1, role2",
   chunk texts for the naive channel)

Default providers: lens = QwenJlensLensProvider (lazy import of
jgraphrag.providers.qwen_jlens with a clear error when missing), embed =
CachedBgeM3Provider (disk-cached bge-m3).

Import-safe: all heavy imports (torch, FlagEmbedding, extract/*,
providers/*) are function-local.
"""
from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any

from .index import GraphIndex

if TYPE_CHECKING:
    from .providers.base import ChunkExtraction, EmbedProvider, LensProvider
    from .stores.base import GraphStore, VectorStore


def _default_lens() -> "LensProvider":
    """Instantiate the validated default lens (lazy — may pull in torch)."""
    try:
        from .providers.qwen_jlens import QwenJlensLensProvider
    except ImportError as e:
        raise ImportError(
            "Default LensProvider unavailable: jgraphrag.providers.qwen_jlens "
            "could not be imported (Qwen2.5-7B + jlens backend). Either install "
            "its dependencies or pass an explicit lens= to build_index()."
        ) from e
    return QwenJlensLensProvider()


def _default_embed() -> "EmbedProvider":
    """Instantiate the validated default embedder (disk-cached bge-m3)."""
    from .providers.bge_m3 import CachedBgeM3Provider
    return CachedBgeM3Provider()


def _apply_filters(extractions: "list[ChunkExtraction]") -> "list[ChunkExtraction]":
    """Concept/role filtering — the validated Phase 25 configuration:
    DF >= 2 (owned here) + concept_ok / role_ok from extract/filter.

    Isolated into its own function because jgraphrag.extract is written by a
    parallel task — if it is unavailable we log a TODO and pass everything
    through, so the rest of the pipeline stays testable.
    """
    try:
        from .extract.filter import concept_ok, role_ok  # type: ignore[import-not-found]
    except ImportError:
        # TODO(extract/filter not yet landed): no filtering applied — every
        # extracted concept/role is kept. Swap in the real filter by landing
        # jgraphrag/extract/filter.py; no call-site changes needed.
        return extractions

    df: dict[str, int] = {}
    for ext in extractions:
        for concept in {c.lower() for c in ext.concepts}:
            df[concept] = df.get(concept, 0) + 1
    for ext in extractions:
        keep = [c for c in ext.concepts
                if df.get(c.lower(), 0) >= 2 and concept_ok(c)[0]]
        ext.concepts = keep
        kept = {c.lower() for c in keep}
        ext.roles = {c: [r for r in roles if role_ok(r)]
                     for c, roles in (ext.roles or {}).items()
                     if c.lower() in kept}
        ext.ws_vecs = {c: v for c, v in (ext.ws_vecs or {}).items()
                       if c.lower() in kept}
    return extractions


def _detect_textside_entities(
    chunks: dict[str, str],
    extractions: "list[ChunkExtraction]",
) -> dict[str, Any] | None:
    """Text-side entity detection → entity_cache for GraphIndex.augment.

    Same parallel-task isolation as _apply_filters: returns None (no
    augmentation) when jgraphrag.extract.entities is unavailable.
    """
    try:
        from .extract.entities import detect_entities  # type: ignore[import-not-found]
    except ImportError:
        # TODO(extract/entities not yet landed): skip text-side augmentation.
        return None
    concept_vocab = {c.lower() for ext in extractions for c in ext.concepts}
    return detect_entities("corpus", chunks, concept_vocab)


def _release_gpu_memory() -> None:
    """Free the GPU model before the embedding provider loads (phase53:713).

    Caller must drop its own lens reference first (``del lens``); this does
    the gc + cache flush. All lens readouts MUST be complete before this.
    """
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_relation_edges(
    lens: "LensProvider",
    extractions: "list[ChunkExtraction]",
    chunk_ids: list[str],
    chunks: dict[str, str],
) -> list[dict[str, Any]]:
    """Read a relation word + prob for each co-occurring concept pair.

    Pair selection: all concept pairs co-occurring in a chunk (deduped
    across chunks; context = the first chunk containing both), read via
    LensProvider.extract_relation.
    """
    pairs: dict[tuple[str, str], str] = {}  # (a, b) -> context chunk id
    for cid, ext in zip(chunk_ids, extractions):
        concepts = sorted({c.lower() for c in ext.concepts})
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                pairs.setdefault((concepts[i], concepts[j]), cid)

    edges: list[dict[str, Any]] = []
    for (a, b), cid in pairs.items():
        rel_word, prob = lens.extract_relation(chunks[cid], a, b)
        if rel_word:
            edges.append({"concept_a": a, "concept_b": b,
                          "relation": rel_word.lower(), "prob": float(prob)})
    return edges


def build_index(
    chunks: dict[str, str] | list[str],
    lens: "LensProvider | None" = None,
    embed: "EmbedProvider | None" = None,
    stores: "tuple[GraphStore, VectorStore] | None" = None,
) -> GraphIndex:
    """Build a GraphIndex from raw chunks (validated J-GraphRAG pipeline).

    ``chunks``: {chunk_id: text} or a plain list of texts (ids "0", "1", ...).
    ``lens`` / ``embed`` / ``stores``: optional provider/store injection;
    defaults are QwenJlensLensProvider, CachedBgeM3Provider and the local
    NumpyVectorStore + JsonGraphStore.
    """
    if isinstance(chunks, dict):
        chunk_map = dict(chunks)
    else:
        chunk_map = {str(i): t for i, t in enumerate(chunks)}
    chunk_ids = sorted(chunk_map)
    texts = [chunk_map[cid] for cid in chunk_ids]

    if stores is None:
        from .stores.local import JsonGraphStore, NumpyVectorStore
        graph: "GraphStore" = JsonGraphStore()
        vectors: "VectorStore" = NumpyVectorStore()
    else:
        graph, vectors = stores

    # Steps 1-4: all lens readouts (GPU) — concepts, filter, relations.
    if lens is None:
        lens = _default_lens()
    extractions = lens.extract_concepts(texts)
    extractions = _apply_filters(extractions)
    entity_cache = _detect_textside_entities(chunk_map, extractions)
    relation_edges = _build_relation_edges(lens, extractions, chunk_ids, chunk_map)

    # Step 5: release the GPU model BEFORE the embedding provider loads.
    del lens
    _release_gpu_memory()

    # Step 6: assembly + embeddings (CPU-embedder by default).
    if embed is None:
        embed = _default_embed()
    index = GraphIndex.from_extractions(
        extractions, chunk_ids, relation_edges, graph, vectors, chunk_map)
    if entity_cache:
        index.augment_with_entities(entity_cache)
    index.build_embeddings(embed)
    index.build_chunk_embeddings(embed)
    return index
