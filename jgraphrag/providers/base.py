"""Provider protocols — the pluggable model backends of J-GraphRAG.

Two seams, both optional at call time (``build_index(chunks)`` falls back to
the default implementations):

- ``LensProvider``: the graph-building readout backend. The validated
  implementation is ``QwenJlensLensProvider`` (Qwen2.5-7B 4bit + Jacobian
  Lens workspace readout). The protocol also fits a plain LLM-API extraction
  backend (the component-level escape hatch, tech-report §8) or a different
  base model with a fitted lens. Only the Qwen+jlens backend is validated.
- ``EmbedProvider``: dense text embeddings used for retrieval seeding and
  entity/relation text vectors. Validated default: bge-m3.

Implementations must be import-safe: constructing a provider must not load
model weights — defer that to the first extract/embed call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ChunkExtraction:
    """Per-chunk readout product (Pass 1 + Pass 2 of the two-pass extraction)."""

    concepts: list[str]
    # concept -> role words (e.g. {"insulin": ["hormone", "treatment"]})
    roles: dict[str, list[str]] = field(default_factory=dict)
    # concept -> workspace conditional vector (E_ws), L2-normalized
    ws_vecs: dict[str, np.ndarray] = field(default_factory=dict)
    # concept -> W_U first-BPE-fragment row vector (kept for parity with the
    # validated pipeline artifacts; not used by the default index/retrieval)
    wu_vecs: dict[str, np.ndarray] = field(default_factory=dict)


@runtime_checkable
class LensProvider(Protocol):
    """Workspace-readout backend used at graph-build time (offline)."""

    def extract_concepts(self, chunks: list[str]) -> list[ChunkExtraction]:
        """Extract concepts (+ roles + ws vectors) for a batch of chunks."""
        ...

    def extract_relation(self, context: str, concept_a: str, concept_b: str) -> tuple[str, float]:
        """Read the relation word between two concepts in a context.

        Returns ``(relation_word, probability)``.
        """
        ...


@runtime_checkable
class EmbedProvider(Protocol):
    """Dense embedding backend used for retrieval seeding (online, cheap)."""

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; one ``dim``-vector per input, in order."""
        ...
