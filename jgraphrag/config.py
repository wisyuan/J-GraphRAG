"""Configuration via environment variables — no secrets hard-coded.

All provider credentials/paths are read from the environment at runtime. This
keeps secrets out of the repo (the .gitignore already excludes .env). Copy
`.env.example` to `.env` and fill in your values, then source it before running.

Resolution order: explicit env var → .env file (if python-dotenv available) →
default. Missing required vars raise a clear error at provider init, not later.
"""
from __future__ import annotations

import os

# --- Embedding (bge-m3) ---
BGE_M3_MODEL_NAME = os.environ.get("LINCLE_BGE_M3_MODEL", "BAAI/bge-m3")
# Where to load weights from. Default = HuggingFace hub id; can be a local path.
BGE_M3_PATH = os.environ.get("LINCLE_BGE_M3_PATH", BGE_M3_MODEL_NAME)
# Device: "cuda" / "mps" / "cpu". Default auto (let FlagEmbedding decide).
BGE_M3_DEVICE = os.environ.get("LINCLE_BGE_M3_DEVICE", None)  # None = auto

# --- LLM (DeepSeek, OpenAI-compatible) ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")  # deepseek-chat retired 2026-07 (API: use deepseek-v4-pro/flash)

# --- Dataset paths ---
PI_REPO_PATH = os.environ.get("LINCLE_PI_REPO", "")  # path to cloned earendil-works/pi
NOVEL_PATH = os.environ.get("LINCLE_NOVEL_PATH", "")  # path to novel .txt


class ConfigError(RuntimeError):
    """Raised when a required config value is missing."""


def require_pi_repo() -> str:
    if not PI_REPO_PATH:
        raise ConfigError(
            "LINCLE_PI_REPO not set. Clone earendil-works/pi and set this to its path. "
            "e.g. export LINCLE_PI_REPO=/tmp/pi-repo"
        )
    return PI_REPO_PATH


def require_deepseek_key() -> str:
    if not DEEPSEEK_API_KEY:
        raise ConfigError(
            "DEEPSEEK_API_KEY not set. Get one at https://platform.deepseek.com/api_keys "
            "and export it."
        )
    return DEEPSEEK_API_KEY


def require_novel_path() -> str:
    if not NOVEL_PATH:
        raise ConfigError(
            "LINCLE_NOVEL_PATH not set. Set this to a public-domain novel .txt file."
        )
    return NOVEL_PATH
