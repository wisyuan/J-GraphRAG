"""Configuration via environment variables — no secrets hard-coded.

This module only carries connection parameters for the *default* providers
(Qwen2.5-7B + jlens lens readout, bge-m3 embeddings). Custom providers are
injected by the caller and need none of these.

Note: the bge-m3 variables keep their historical ``LINCLE_BGE_M3_*`` names
(inherited from the Lincle project this repo was split from). The on-disk
embedding cache is namespaced by device, so switching ``LINCLE_BGE_M3_DEVICE``
triggers a full re-embed.
"""
from __future__ import annotations

import os

# --- Default EmbedProvider: bge-m3 ---
BGE_M3_MODEL_NAME = os.environ.get("LINCLE_BGE_M3_MODEL", "BAAI/bge-m3")
# Where to load weights from. Default = HuggingFace hub id; can be a local path.
BGE_M3_PATH = os.environ.get("LINCLE_BGE_M3_PATH", BGE_M3_MODEL_NAME)
# Device: "cuda" / "mps" / "cpu". Default "cpu" — on an 8GB GPU box bge-m3 must
# not co-reside with the 4bit LLM (they OOM together), so CPU is the safe default.
BGE_M3_DEVICE = os.environ.get("LINCLE_BGE_M3_DEVICE", "cpu")

# --- Default LensProvider: Qwen2.5-7B-Instruct (4bit) + Jacobian Lens ---
# HF hub id (used when no local weights dir is present).
QWEN_MODEL_ID = os.environ.get("JGRAPHRAG_QWEN_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
# Local weights dir; default matches the restore_env.sh symlink layout.
QWEN_MODEL_PATH = os.environ.get("JGRAPHRAG_QWEN_MODEL_PATH", "/tmp/qwen25-7b-it-weights")
# Local jlens lens dir; falls back to HF "neuronpedia/jacobian-lens" when absent.
JLENS_LENS_PATH = os.environ.get("JGRAPHRAG_JLENS_LENS_PATH", "/tmp/jlens-qwen25-7b-it")

# --- Embedding disk cache ---
EMBED_CACHE_DIR = os.environ.get("JGRAPHRAG_EMBED_CACHE_DIR", "data/.embedcache")


class ConfigError(RuntimeError):
    """Raised when a required config value is missing."""
