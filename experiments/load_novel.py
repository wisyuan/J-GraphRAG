"""Novel dataset loader for bet #1b (M2's second falsification domain).

A public-domain mid/long novel, chunked into chapters (each chapter = one Record,
analogous to one source file in the code dataset). Same bge-m3 + M2 infra reused;
only the Record granularity differs (chapter vs file).

The loader:
- Reads a plain-text novel (UTF-8).
- Splits into chapters by common heading patterns (第X章 / Chapter X / CHAPTER X /
  章节X / etc.). Falls back to fixed-size paragraph batches if no headings found.
- Emits a JSON list of {id, title, text} chapters ready for the SemanticProjector.

Source-3 human annotation (the bet#1b decision core) is done manually on top of
this chunked output — not generated here (would pollute).

Usage:
    LINCLE_NOVEL_PATH=/path/to/novel.txt python -m datasets.load_novel > novel_chunks.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jgraphrag.config import require_novel_path

# Heading patterns across languages/styles. Order matters (most specific first).
HEADING_PATTERNS = [
    re.compile(r"^第[一二三四五六七八九十百千零0-9]+[章节回卷][、.:\s]*(.*)$", re.MULTILINE),
    re.compile(r"^Chapter\s+(\d+)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^CHAPTER\s+([IVX0-9]+)\.?\s*(.*)$", re.MULTILINE),
    re.compile(r"^(?:\d+)[、.]\s*(.{4,40})$", re.MULTILINE),  # bare "1. 标题"
]


def chunk_novel(text: str) -> list[dict]:
    """Split a novel into chapter chunks."""
    # Try each heading pattern; use the first that yields >= 3 chapters.
    for pat in HEADING_PATTERNS:
        matches = list(pat.finditer(text))
        if len(matches) >= 3:
            chapters = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                if len(body) < 50:  # skip spurious short matches
                    continue
                chapters.append({
                    "id": len(chapters) + 1,
                    "title": _extract_title(m),
                    "text": body,
                })
            if len(chapters) >= 3:
                return chapters

    # Fallback: split into ~2000-char paragraph batches (no clear chapters).
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 100]
    batches = []
    batch = []
    size = 0
    for p in paras:
        batch.append(p)
        size += len(p)
        if size >= 2000:
            batches.append("\n\n".join(batch))
            batch = []
            size = 0
    if batch:
        batches.append("\n\n".join(batch))
    return [
        {"id": i + 1, "title": f"段 {i + 1}", "text": b}
        for i, b in enumerate(batches)
    ]


def _extract_title(m: re.Match) -> str:
    """Best-effort chapter title from a heading match."""
    groups = [g for g in m.groups() if g]
    if groups:
        title = groups[-1].strip().strip("。.、:：")
        if title:
            return title[:60]
    return m.group(0).strip()[:60]


def generate() -> list[dict]:
    path = Path(require_novel_path())
    text = path.read_text(encoding="utf-8", errors="ignore")
    chapters = chunk_novel(text)
    if len(chapters) < 5:
        sys.stderr.write(
            f"WARNING: only {len(chapters)} chunks detected; novel may be too short "
            "or have no recognizable chapter headings.\n"
        )
    return chapters


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
