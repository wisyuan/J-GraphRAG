"""Source-1 dataset generator — deterministic queries on pi (zero pollution).

These queries' ground-truth comes from static analysis (grep/AST), NOT from an
LLM, so they cannot bias the A/B (which tests LLM-based concern matching). They
test the relational-recall baseline (axis-3-ish) and give the A/B a stable floor.

Output: a JSON list of {query, ground_truth_files, provenance} entries.

Usage:
    LINCLE_PI_REPO=/path/to/pi python -m datasets.gen_source1_deterministic > source1.json
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Allow running both as module and as script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.config import require_pi_repo


def _grep(pattern: str, repo: Path, glob: str = "*.ts") -> list[str]:
    """Return relative paths of files matching a ripgrep/grep pattern."""
    try:
        out = subprocess.run(
            ["rg", "-l", "--no-config", "-g", glob, pattern, str(repo)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except FileNotFoundError:
        # Fallback to grep if rg absent.
        out = subprocess.run(
            ["grep", "-rlE", "--include=" + glob, pattern, str(repo)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    files = []
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        rel = os.path.relpath(line.strip(), str(repo))
        if "node_modules" in rel:
            continue
        files.append(rel)
    return sorted(files)


def _unique_exports(repo: Path) -> list[str]:
    """Extract a sample of exported identifiers from pi (for 'where is X' queries)."""
    files = []
    for pkg in ["ai", "agent", "orchestrator", "tui"]:
        d = repo / "packages" / pkg / "src"
        if d.is_dir():
            files.extend(d.rglob("*.ts"))
    idents: set[str] = set()
    name_re = re.compile(r"\b(?:export|interface|type|class|function|const)\s+([A-Z][A-Za-z0-9_]+)")
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in name_re.finditer(text):
            idents.add(m.group(1))
    # Take a stable, diverse sample.
    return sorted(idents)[:25]


def generate() -> list[dict]:
    repo = Path(require_pi_repo())
    entries: list[dict] = []

    # --- Query family A: "where is X defined / exported?" ---
    for ident in _unique_exports(repo):
        files = _grep(rf"\b{re.escape(ident)}\b", repo)
        if 1 <= len(files) <= 15:
            entries.append({
                "query": f"Where is {ident} defined or used?",
                "ground_truth_files": files[:8],
                "provenance": "source1_deterministic:identifier_grep",
            })

    # --- Query family B: "which files import X?" ---
    for module in ["openai", "typebox", "../types", "../agent-loop"]:
        files = _grep(rf"from ['\"]([^'\"]*/)?{re.escape(module)}", repo)
        if files:
            entries.append({
                "query": f"Which files import {module}?",
                "ground_truth_files": files,
                "provenance": "source1_deterministic:import_grep",
            })

    # --- Query family C: "which files are in package X?" (containment, axis-1) ---
    for pkg in ["ai", "agent", "orchestrator", "tui", "coding-agent"]:
        d = repo / "packages" / pkg / "src"
        if d.is_dir():
            rels = sorted(
                os.path.relpath(str(f), str(repo))
                for f in d.rglob("*.ts")
            )
            entries.append({
                "query": f"What source files are in the {pkg} package?",
                "ground_truth_files": rels,
                "provenance": "source1_deterministic:package_containment",
            })

    return entries


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
