"""Source-2 dataset generator — queries derived from pi's markdown docs (low pollution).

These queries' ground-truth comes from pi's own documentation (README/AGENTS.md/
CONTRIBUTING.md + package READMEs), written by pi's authors — NOT by the LLM
under test. Low pollution because the docs are an independent source from the
concern-inference LLM.

Approach: extract headings + code references from markdown, turn each into a
"the docs say X lives in Y" query whose ground-truth is the referenced source file.

Usage:
    LINCLE_PI_REPO=/path/to/pi python -m datasets.gen_source2_docs > source2.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.config import require_pi_repo

# Match markdown headings and `packages/<pkg>/src/...` style path references.
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)
PATH_REF_RE = re.compile(r"`?(packages/[A-Za-z0-9_./-]+(?:\.ts|/src/[A-Za-z0-9_./-]+))`?")


def generate() -> list[dict]:
    repo = Path(require_pi_repo())
    md_files = sorted(
        p for p in repo.rglob("*.md")
        if "node_modules" not in str(p)
    )
    entries: list[dict] = []

    for md in md_files:
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel_md = os.path.relpath(str(md), str(repo))

        # Pair each heading with the source-file references under it.
        # Split doc into heading-sections.
        sections: list[tuple[str, str]] = []
        pos = 0
        current_heading = os.path.basename(rel_md).removesuffix(".md")
        current_body: list[str] = []
        for m in HEADING_RE.finditer(text):
            sections.append((current_heading, "".join(current_body)))
            current_heading = m.group(2).strip()
            current_body = []
        sections.append((current_heading, "".join(current_body)))

        for heading, body in sections:
            if not heading or len(heading) < 4:
                continue
            refs = sorted(set(PATH_REF_RE.findall(body)))
            # Keep only refs that look like source files.
            src_refs = [r for r in refs if r.endswith(".ts") or "/src/" in r]
            if not src_refs:
                continue
            # De-noise: cap at 8 ground-truth files.
            entries.append({
                "query": f"The docs describe: {heading}",
                "ground_truth_files": src_refs[:8],
                "provenance": f"source2_docs:{rel_md}",
                "note": "ground-truth from pi author-written docs, not the LLM under test",
            })

    return entries


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
