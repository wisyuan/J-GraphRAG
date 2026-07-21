"""CachedBgeM3Provider — transparent disk cache for bge-m3 embeddings.

Wraps `BgeM3Provider` with zero interface change: `embed(texts) -> list[list[float]]`,
1024-dim raw (un-normalized) vectors, in input order. Existing experiments swap
`BgeM3Provider()` → `CachedBgeM3Provider()` and get automatic persistence —
re-runs skip the ~2GB model entirely for texts seen before.

Cache key: sha256(text + model_id + device + fp16). Batch params (batch_size /
max_length) don't affect per-text vectors (bge-m3 is stateless across batch
members), so they're excluded from the key.

Format: a single .npz file (keys: (N,) <U64 hex hashes, vecs: (N,1024) float32).
Load on init (~1s for 10k entries), full-rewrite on each miss batch (~50ms for
10k). Single-process experiments — no concurrency concerns.

Usage:
    from experiments.embed_cache import CachedBgeM3Provider
    embed = CachedBgeM3Provider()          # drop-in replacement
    vecs = embed.embed(["hello", "world"]) # first call: model + cache write
    vecs = embed.embed(["hello"])          # second call: cache hit, no model
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from jgraphrag.embed import BgeM3Provider
from jgraphrag.config import BGE_M3_DEVICE

# Repo root = parents[1] from this file (experiments/ → python/ → lincle/ → crates/ → repo/)
_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_DIR = _REPO / "data" / ".embedcache"


def _cache_namespace() -> str:
    """Namespace string for cache keys — isolates by model config.

    Includes model_id + device + fp16 so that switching GPU↔CPU or fp16↔fp32
    doesn't return stale vectors from a different precision.
    """
    device = BGE_M3_DEVICE or "auto"
    return f"{BgeM3Provider.MODEL_ID}|device={device}|fp16=True"


def _hash_key(text: str) -> str:
    return hashlib.sha256(f"{text}|{_cache_namespace()}".encode("utf-8")).hexdigest()


class CachedBgeM3Provider:
    """BgeM3Provider with transparent disk caching (composition, not inheritance).

    Why composition: BgeM3Provider uses a module-level singleton (_BGEM3) for
    the model. Subclassing would entangle with that singleton. Composition lets
    us hold a BgeM3Provider instance and delegate .embed() to it on cache miss.
    """

    MODEL_ID = BgeM3Provider.MODEL_ID
    DIM = BgeM3Provider.DIM

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        provider: Optional[BgeM3Provider] = None,
    ) -> None:
        self._provider = provider or BgeM3Provider()
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"{_cache_namespace().replace('|', '_').replace('/', '_')}.npz"
        # hash hex → list[float]. Kept as Python lists to match BgeM3Provider's contract.
        self._cache: dict[str, list[float]] = {}
        self._dirty = False
        self._load()

    @property
    def model_id(self) -> str:
        return self._provider.model_id

    @property
    def dim(self) -> int:
        return self._provider.dim

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._cache)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Cache hits skip the model entirely; misses batch-call
        the underlying provider once, then persist. Returns raw 1024-dim vectors
        in input order (identical contract to BgeM3Provider.embed)."""
        results: list[Optional[list[float]]] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            key = _hash_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        if miss_texts:
            # Single batched call to the real model for all misses.
            new_vecs = self._provider.embed(miss_texts)
            for idx, text, vec in zip(miss_indices, miss_texts, new_vecs):
                key = _hash_key(text)
                self._cache[key] = vec
                results[idx] = vec
            self._dirty = True
            self._dump()  # persist immediately (crash-safe for single process)

        # All slots filled; satisfy type checker.
        return [r for r in results]  # type: ignore[list-item]

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single text."""
        return self.embed([text])[0]

    def flush(self) -> None:
        """Explicitly persist the cache. Called automatically on each miss,
        but exposed for scripts that want deterministic save points."""
        if self._dirty:
            self._dump()

    def _load(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            data = np.load(self._cache_file, allow_pickle=False)
            keys = data["keys"].astype(str)
            vecs = data["vecs"]  # (N, 1024) float32
            for key, vec in zip(keys, vecs):
                self._cache[key] = vec.tolist()
        except Exception:
            # Corrupt cache file — start fresh rather than crash.
            # The experiment will re-embed and rebuild.
            pass

    def _dump(self) -> None:
        if not self._cache:
            return
        keys = np.array(list(self._cache.keys()), dtype="<U64")
        vecs = np.array(list(self._cache.values()), dtype=np.float32)
        # Write to a temp file then rename, so a crash mid-write doesn't
        # corrupt the existing cache file. np.savez auto-appends .npz, so
        # we use a stem-based temp name and replace to the final .npz path.
        tmp = self._cache_file.with_suffix(".tmp.npz")
        np.savez(tmp, keys=keys, vecs=vecs)
        tmp.replace(self._cache_file)
        self._dirty = False

    def __del__(self) -> None:
        # Best-effort save on exit. Python doesn't guarantee __del__ is called,
        # but _dump() after each miss makes this a safety net, not the main path.
        try:
            if self._dirty:
                self._dump()
        except Exception:
            pass


def get_or_embed(
    texts: list[str],
    provider: Optional[BgeM3Provider] = None,
    cache_dir: Optional[Path] = None,
) -> list[list[float]]:
    """Functional interface: embed with caching using an existing or new provider.

    Convenience for scripts that don't want to manage a CachedBgeM3Provider
    instance. Creates a CachedBgeM3Provider wrapping the given provider (or a
    new BgeM3Provider), embeds, and returns. The cache persists across calls
    since it's keyed on disk.
    """
    cached = CachedBgeM3Provider(cache_dir=cache_dir, provider=provider)
    return cached.embed(texts)


if __name__ == "__main__":  # pragma: no cover
    # Smoke test: embed two texts, re-embed one (should hit cache), verify consistency.
    import os
    import sys

    # Load .env like other experiments expect.
    if not os.environ.get("LINCLE_BGE_M3_PATH"):
        print("WARNING: LINCLE_BGE_M3_PATH not set — relying on default BAAI/bge-m3", file=sys.stderr)

    p = CachedBgeM3Provider()
    print(f"cache loaded: {p.size} entries from {p._cache_file}")

    texts = ["hello world", "password hash bcrypt authentication"]
    print(f"embedding {len(texts)} texts (first call — may load model)...")
    v1 = p.embed(texts)
    assert len(v1) == 2 and len(v1[0]) == p.DIM, f"expected 2×{p.DIM}, got {len(v1)}×{len(v1[0])}"
    print(f"  dim={len(v1[0])}, cache size now={p.size}")

    # Second call: 'hello world' should be a cache hit.
    print("re-embedding 'hello world' (should be cache hit)...")
    v2 = p.embed(["hello world"])
    assert v2[0] == v1[0], "cache inconsistency: same text gave different vector!"
    print(f"  cache hit verified, cache size still={p.size}")

    print(f"OK — cache file: {p._cache_file}")
