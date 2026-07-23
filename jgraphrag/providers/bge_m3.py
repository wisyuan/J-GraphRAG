"""bge-m3 EmbedProvider implementations — the validated default embedding backend.

Adopted verbatim in behavior from:
- ``jgraphrag/embed.py`` (``BgeM3Provider``: lazy FlagEmbedding load, 1024-dim
  dense vectors, config-driven path/device).
- ``experiments/embed_cache.py`` (``CachedBgeM3Provider``: transparent disk
  cache — sha256(text + namespace) keys, npz under EMBED_CACHE_DIR, atomic
  tmp+rename writes, namespace includes device + fp16 so CPU/GPU caches are
  separate namespaces).

Import-safe: FlagEmbedding/torch are imported lazily inside ``_load_model``;
constructing either provider does not load model weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import BGE_M3_DEVICE, BGE_M3_PATH, EMBED_CACHE_DIR

# Lazy import — FlagEmbedding is a heavy dep; only fail if actually used.
_BGEM3 = None


def _load_model():
    global _BGEM3
    if _BGEM3 is None:
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "FlagEmbedding not installed. Run: pip install -U FlagEmbedding"
            ) from e
        kwargs = {"use_fp16": True}
        if BGE_M3_DEVICE:
            kwargs["devices"] = BGE_M3_DEVICE
        _BGEM3 = BGEM3FlagModel(BGE_M3_PATH, **kwargs)
    return _BGEM3


class BgeM3Provider:
    """bge-m3 embedding provider (1024-dim dense)."""

    MODEL_ID = "BAAI/bge-m3"
    DIM = 1024

    def __init__(self) -> None:
        # Don't load weights at construction — defer to first embed() call so
        # importing this module is cheap (tests/dry-runs that don't embed work).
        self._model = None

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def dim(self) -> int:
        return self.DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one 1024-dim vector per input, in order."""
        if self._model is None:
            self._model = _load_model()
        out = self._model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        # encode returns a dict-like with 'dense_vecs' as a numpy array [N, 1024].
        vecs = out["dense_vecs"]
        return vecs.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Convenience: embed a single text."""
        return self.embed([text])[0]


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

    Cache format: a single .npz file (keys: (N,) <U64 hex hashes, vecs:
    (N,1024) float32). Load on init, full-rewrite on each miss batch.
    """

    MODEL_ID = BgeM3Provider.MODEL_ID
    DIM = BgeM3Provider.DIM

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        provider: Optional[BgeM3Provider] = None,
    ) -> None:
        self._provider = provider or BgeM3Provider()
        self._cache_dir = Path(cache_dir) if cache_dir else Path(EMBED_CACHE_DIR)
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
            # The next embed will re-embed and rebuild.
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
