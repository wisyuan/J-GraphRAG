"""Lincle Python providers + dataset tooling (M2.6/M2.7 + A/B dataset gen).

These are pure-Python modules that implement the embedding/LLM providers and
generate the A/B dataset (ground-truth annotation sources). They are bridged
to the Rust core (spec::providers traits) via the PyO3 cdylib + env-var config.

Modules:
- embed: BgeM3Provider — real embedding (bge-m3, 1024-dim) for M2.6.
- llm:   DeepSeekProvider — real concern-inference LLM for M2.7.
- config: env-var based configuration (API keys, model paths) — no secrets
  hard-coded; read from environment at runtime.

Dataset generators (in datasets/):
- gen_source1_deterministic.py — source-1 queries (tree-sitter/grep, zero pollution).
- gen_source2_docs.py — source-2 queries (pi markdown docs, low pollution).
- load_novel.py — novel dataset loader/chunker (for bet #1b).
"""
