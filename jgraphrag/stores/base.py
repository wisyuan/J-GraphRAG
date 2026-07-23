"""Store protocols — the pluggable persistence backends of J-GraphRAG.

- ``VectorStore``: dense-vector read/write used for retrieval seeding
  (entity/relation text embeddings, chunk embeddings). Default:
  ``NumpyVectorStore`` (in-memory matmul + npz). The ``namespace`` argument
  maps to an external vector DB's collection.
- ``GraphStore``: node/edge read/write used for graph propagation
  (one-hop expansion, weight spreading). Default: ``JsonGraphStore``
  (adjacency dict + JSON). Compatible in shape with Neo4j-style adapters.

Retrieval and indexing code programs against these protocols only — swapping
in Qdrant/Neo4j later must not touch ``retrieve.py`` / ``index.py``.

Only the local implementations ship in this package; external adapters are
intended as optional extras.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    """Namespaced dense-vector store with top-k cosine query."""

    def upsert(
        self,
        namespace: str,
        ids: list[str],
        vecs: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or overwrite vectors (``vecs`` shape ``[len(ids), dim]``)."""
        ...

    def query(self, namespace: str, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Return ``[(id, cosine_score)]`` sorted by score, descending."""
        ...

    def get_payload(self, namespace: str, id: str) -> dict[str, Any] | None: ...

    def count(self, namespace: str) -> int: ...


@runtime_checkable
class GraphStore(Protocol):
    """Property-graph store: typed nodes/edges with adjacency traversal."""

    def add_node(self, node_id: str, **attrs: Any) -> None: ...

    def add_edge(self, a: str, b: str, weight: float = 1.0, **attrs: Any) -> str:
        """Add an undirected edge; returns its edge id."""
        ...

    def neighbors(self, node_id: str) -> list[tuple[str, str, float]]:
        """Return ``[(neighbor_id, edge_id, weight)]`` for all incident edges."""
        ...

    def get_node(self, node_id: str) -> dict[str, Any] | None: ...

    def get_edge(self, edge_id: str) -> dict[str, Any] | None: ...

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]: ...

    def iter_edges(self) -> Iterator[tuple[str, dict[str, Any]]]: ...
