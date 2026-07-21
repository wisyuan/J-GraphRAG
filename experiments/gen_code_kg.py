"""Code-domain KG generator (annotation pipeline Step 1, bet#1a).

Builds a knowledge graph from a codebase via deterministic extraction:
  entities   = files + exported identifiers (functions/types/classes/consts)
  relations  = file IMPORTS file, file DEFINES identifier, identifier REFERENCES identifier

NO LLM, NO tree-sitter dependency — pure regex extraction. Zero pollution
(KG triples come from static text patterns, independent of the LLM under test).
This feeds Step 2 (paper/blog correction) and Step 3 (scope contraction → queries).

Output: JSON with {nodes: [...], edges: [...]} where each edge is a typed triple.

Usage:
    LINCLE_PI_REPO=/path/to/pi python -m datasets.gen_code_kg > code_kg.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.config import require_pi_repo

# Extraction patterns (regex — deterministic, no ML).
EXPORT_RE = re.compile(
    r"\bexport\s+(?:async\s+)?(?:function|class|const|let|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)"
)
IMPORT_RE = re.compile(
    r"""from\s+['"]([^'"]+)['"]"""
)
RELATIVE_IMPORT_RE = re.compile(r"^\.{1,2}/")
IDENT_USAGE_RE = re.compile(r"\b([A-Z][A-Za-z0-9_$]{2,})\b")  # Capitalized idents (types/classes)

# File types to index.
SOURCE_GLOBS = ("*.ts", "*.tsx", "*.js", "*.jsx", "*.py", "*.rs", "*.go", "*.java")
# Skip these dirs.
SKIP_DIRS = {"node_modules", "target", ".git", "dist", "build", ".venv", "__pycache__"}


def _is_source(p: Path) -> bool:
    return any(p.match(g) for g in SOURCE_GLOBS) and not any(
        part in SKIP_DIRS for part in p.parts
    )


def _resolve_import(imp: str, current_file: Path, repo: Path) -> str | None:
    """Resolve a relative import to a repo-relative path (best-effort)."""
    if not RELATIVE_IMPORT_RE.match(imp):
        return None  # bare/package import — keep as-is, not a file edge
    base = (current_file.parent / imp).resolve()
    # Try common extensions.
    for ext in ("", ".ts", ".tsx", ".js", "/index.ts", "/index.js"):
        cand = Path(str(base) + ext)
        try:
            rel = cand.relative_to(repo)
            return str(rel).replace("\\", "/")
        except ValueError:
            continue
    return None


def generate() -> dict:
    repo = Path(require_pi_repo())
    nodes: list[dict] = []
    edges: list[dict] = []

    # Collect all source files first.
    files = sorted(
        p for p in repo.rglob("*") if p.is_file() and _is_source(p.relative_to(repo))
    )
    file_set = {str(p.relative_to(repo)).replace("\\", "/") for p in files}

    # Node: each source file.
    for f in sorted(file_set):
        nodes.append({"id": f, "kind": "file", "label": f})

    # Extract per-file: exports + imports + ident usages.
    file_exports: dict[str, list[str]] = {}
    for p in files:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        exports = EXPORT_RE.findall(text)
        file_exports[rel] = exports
        for ident in exports:
            nodes.append({"id": f"{rel}::{ident}", "kind": "symbol", "label": ident, "file": rel})
            edges.append({
                "source": rel, "relation": "DEFINES", "target": f"{rel}::{ident}",
                "provenance": "regex:export",
            })
        # Imports → file-to-file edges.
        for imp in IMPORT_RE.findall(text):
            resolved = _resolve_import(imp, p, repo)
            if resolved and resolved in file_set:
                edges.append({
                    "source": rel, "relation": "IMPORTS", "target": resolved,
                    "provenance": "regex:import",
                })

    # Cross-file identifier references (who uses a symbol defined elsewhere).
    # Build a global symbol → defining-file index.
    symbol_to_def: dict[str, str] = {}
    for f, exps in file_exports.items():
        for sym in exps:
            # Last-writer wins; collisions are rare for exported names.
            symbol_to_def[sym] = f
    for p in files:
        rel = str(p.relative_to(repo)).replace("\\", "/")
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        seen: set[str] = set()
        for ident in IDENT_USAGE_RE.findall(text):
            if ident in seen:
                continue
            seen.add(ident)
            def_file = symbol_to_def.get(ident)
            if def_file and def_file != rel:
                edges.append({
                    "source": rel, "relation": "REFERENCES_SYMBOL",
                    "target": f"{def_file}::{ident}", "provenance": "regex:ident_usage",
                })

    return {"nodes": nodes, "edges": edges, "provenance": "deterministic_regex_extraction"}


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
