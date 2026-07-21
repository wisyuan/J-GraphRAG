"""BgeM3Provider — real embedding provider for M2.6.

Implements the same surface as the Rust `EmbeddingProvider` trait (embed a
batch of texts → list of vectors, each tagged with its EmbeddingSpace). Uses
FlagEmbedding's BGEM3FlagModel: 1024-dim dense vectors, 8192 max tokens,
multilingual (100+ languages).

Bridge to Rust: the PyO3 cdylib calls this provider's `embed()` via the python
adapter; the returned vectors carry model_id "BAAI/bge-m3" + dim 1024 so the
Spec's EmbeddingSpace metadata is populated (prevents model binding, §12.3).

NOTE: requires `pip install FlagEmbedding` and the bge-m3 weights (auto-downloaded
from HuggingFace hub on first use, or set LINCLE_BGE_M3_PATH to a local copy).
"""
from __future__ import annotations

from typing import Optional

from .config import BGE_M3_DEVICE, BGE_M3_PATH

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


# Standalone smoke test: python -m jgraphrag.embed
if __name__ == "__main__":  # pragma: no cover
    p = BgeM3Provider()
    v = p.embed_one("authentication login password")
    print(f"dim={len(v)}, first3={v[:3]}")
    assert len(v) == 1024, f"expected 1024-dim, got {len(v)}"
    print("OK")
