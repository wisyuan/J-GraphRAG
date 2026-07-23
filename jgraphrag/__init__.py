"""J-GraphRAG — zero-LLM-generation knowledge graph RAG.

Builds a knowledge graph from text chunks via Jacobian Lens workspace readout
(abstract layer: concepts/relations/roles) plus text-side rule-based entity
detection (surface layer), then retrieves with LightRAG-style dual-level
ranking (local entity + global relation seeds, one-hop expansion, interleaved
with a dense-embedding naive arm). No LLM ``generate()`` calls at build time.

Quick start::

    from jgraphrag import build_index

    index = build_index(chunks)            # chunks: {id: text} or [text]
    results = index.retrieve("your query") # ranked chunk ids (list[str])

Model and storage backends are pluggable — pass your own ``lens=`` /
``embed=`` (providers.base protocols) or ``stores=`` (stores.base protocols)
to ``build_index``. Defaults: Qwen2.5-7B 4bit + jlens (validated), bge-m3
embeddings, local numpy/JSON stores. Only the default lens backend is
validated end-to-end (see docs/j-graphrag-complete-method.md).
"""
from __future__ import annotations

from jgraphrag.index import GraphIndex, merge_entities
from jgraphrag.pipeline import build_index
from jgraphrag.providers.base import ChunkExtraction, EmbedProvider, LensProvider
from jgraphrag.retrieve import interleave_rankings, lightrag_retrieve, merge_topk
from jgraphrag.stores.base import GraphStore, VectorStore
from jgraphrag.stores.local import JsonGraphStore, NumpyVectorStore

__all__ = [
    "build_index",
    "GraphIndex",
    "merge_entities",
    "lightrag_retrieve",
    "merge_topk",
    "interleave_rankings",
    "ChunkExtraction",
    "LensProvider",
    "EmbedProvider",
    "VectorStore",
    "GraphStore",
    "NumpyVectorStore",
    "JsonGraphStore",
]

__version__ = "0.1.0"
