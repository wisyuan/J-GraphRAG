"""Shared test fixtures: deterministic mock providers (no model, no GPU)."""
from __future__ import annotations

import re
import zlib

import numpy as np
import pytest


class OneHotEmbed:
    """Registry embedder: each unseen text gets the next basis vector.

    Distinct texts are exactly orthogonal; identical texts get identical
    vectors — deterministic cosine sims for retrieval assertions.
    """

    def __init__(self, dim: int = 32) -> None:
        self._reg: dict[str, int] = {}
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            if t not in self._reg:
                self._reg[t] = len(self._reg)
            v = [0.0] * self._dim
            v[self._reg[t] % self._dim] = 1.0
            out.append(v)
        return out


class TokenEmbed:
    """Bag-of-words hashed embedder: texts sharing a token have sim > 0.

    Deterministic across processes (crc32, not builtin hash).
    """

    DIM = 64

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.DIM)
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                v[zlib.crc32(tok.encode("utf-8")) % self.DIM] += 1.0
            out.append(v.tolist())
        return out


@pytest.fixture
def onehot_embed() -> OneHotEmbed:
    return OneHotEmbed()


@pytest.fixture
def token_embed() -> TokenEmbed:
    return TokenEmbed()
