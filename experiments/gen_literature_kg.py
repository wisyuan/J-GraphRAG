"""Step 2: Literature-correction of the novel KG (bet#1b, source-2+).

Consumes academic-literature full-text files (papers/criticism/Wikipedia saved
into experiments/m2/literature/) and rule-extracts chapter/character/theme
citations to CORRECT/SUPPLEMENT the auto-generated novel_kg.json.

ZERO POLLUTION: extraction is pure regex — no LLM reads or summarizes the
literature. We extract what human authors explicitly wrote ("see Chapter 12",
"in Ch. 3", "Darcy's proposal in Chapter 34") and turn those citations into KG
edges. The judgment is the human author's, not an LLM's.

Input:  a directory of literature files (.txt/.md/.html) + the base novel_kg.json
Output: a corrected novel_kg.json with supplemented edges (theme APPEARS_IN chapter,
        character DISCUSSED_IN chapter, etc.)

Usage:
    python -m experiments.gen_literature_kg \
        --literature-dir experiments/m2/literature \
        --base-kg experiments/m2/novel_kg.json \
        > experiments/m2/novel_kg_corrected.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Citation patterns (rule extraction, multi-language) ---
# English: "Chapter 12", "Ch. 3", "chapter 34", "in Ch. 5", "Chapter Twelve"
# Chinese: "第12章", "第三章", "第34回"
_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 61,
}

CN_NUM = {"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}

CHAP_RE_EN = re.compile(
    r"\b(?:Chapter|Chap\.?|Ch\.?)\s+("
    r"\d{1,3}|"
    + "|".join(_NUM_WORDS) +
    r")\b",
    re.IGNORECASE,
)
CHAP_RE_CN = re.compile(r"第([一二三四五六七八九十百\d]{1,4})[章回节卷]")

# Character names to look for near chapter citations (Pride & Prejudice).
# Extended by the novel KG's own character list at runtime.
DEFAULT_CHARACTERS = {
    "Elizabeth", "Darcy", "Jane", "Bingley", "Wickham", "Lydia",
    "Mr. Bennet", "Mrs. Bennet", "Mr. Collins", "Charlotte", "Lady Catherine",
    "Kitty", "Mary", "Mr. Darcy", "Miss Bennet", "Miss Elizabeth",
}

# Theme keywords (multi-angle coverage for concern ambiguity).
THEME_KEYWORDS = {
    "marriage": ["marriage", "wedlock", "matrimony", "engagement", "proposal"],
    "class": ["class", "rank", "station", "gentleman", "society", "breeding"],
    "pride": ["pride", "vanity", "arrogance", "conceit"],
    "prejudice": ["prejudice", "bias", "prejudgment", "first impression"],
    "feminism": ["woman", "women", "female", "independence", "agency", "feminist"],
    "irony": ["irony", "ironic", "satire", "satirical", "wit"],
    "narrative": ["narrator", "narrative", "free indirect", "focalization", "point of view"],
    "money": ["money", "income", "fortune", "wealth", "settlement", "pound"],
}


def _cn_num_to_int(s: str) -> int:
    """Convert Chinese numeral (三十四) or arabic to int. Best-effort."""
    if s.isdigit():
        return int(s)
    # Handle 十X / X十 / X十X patterns.
    total = 0
    if "十" in s:
        parts = s.split("十")
        tens = CN_NUM.get(parts[0], 1) if parts[0] else 1
        ones = CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    return CN_NUM.get(s, 0)


def _extract_chapter_citations(text: str) -> list[int]:
    """Return all chapter numbers cited in `text` (deduped, sorted)."""
    nums: set[int] = set()
    for m in CHAP_RE_EN.finditer(text):
        token = m.group(1).lower()
        if token.isdigit():
            n = int(token)
        else:
            n = _NUM_WORDS.get(token, 0)
        if 1 <= n <= 61:  # Pride & Prejudice has 61 chapters
            nums.add(n)
    for m in CHAP_RE_CN.finditer(text):
        n = _cn_num_to_int(m.group(1))
        if 1 <= n <= 120:
            nums.add(n)
    return sorted(nums)


def _extract_themes(text: str) -> set[str]:
    """Return theme labels whose keywords appear in `text`."""
    lower = text.lower()
    found: set[str] = set()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            found.add(theme)
    return found


def _extract_characters(text: str, known: set[str]) -> set[str]:
    """Return character names found in `text`."""
    found: set[str] = set()
    for name in known:
        if name in text:
            found.add(name)
    return found


def correct_kg(base_kg: dict, literature_dir: Path, characters: set[str]) -> dict:
    """Read all literature files, extract citations, supplement the base KG."""
    kg = json.loads(json.dumps(base_kg))  # deep copy
    known_characters = characters | DEFAULT_CHARACTERS
    new_edges: list[dict] = []
    provenance_count = 0

    for lit_file in sorted(literature_dir.iterdir()):
        if lit_file.suffix not in (".txt", ".md", ".html", ".text"):
            continue
        try:
            text = lit_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        src_name = lit_file.name

        # Split into paragraphs for finer-grained co-occurrence.
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            chapters = _extract_chapter_citations(para)
            if not chapters:
                continue
            themes = _extract_themes(para)
            chars = _extract_characters(para, known_characters)

            # theme → APPEARS_IN chapter (the core source-2+ contribution:
            # human authors linking abstract themes to specific chapters).
            for theme in themes:
                for ch in chapters:
                    new_edges.append({
                        "source": f"theme::{theme}",
                        "relation": "DISCUSSED_IN",
                        "target": f"chapter_{ch}",
                        "provenance": f"literature:{src_name}",
                    })
                    provenance_count += 1
            # character → DISCUSSED_IN chapter
            for char in chars:
                for ch in chapters:
                    new_edges.append({
                        "source": f"char::{char}",
                        "relation": "DISCUSSED_IN",
                        "target": f"chapter_{ch}",
                        "provenance": f"literature:{src_name}",
                    })
                    provenance_count += 1

    # Add new theme nodes (not in base KG).
    existing_ids = {n["id"] for n in kg["nodes"]}
    theme_nodes = set()
    for e in new_edges:
        if e["source"].startswith("theme::") and e["source"] not in existing_ids:
            theme_nodes.add(e["source"])
    for tid in sorted(theme_nodes):
        kg["nodes"].append({
            "id": tid, "kind": "theme",
            "label": tid.removeprefix("theme::"),
        })
        existing_ids.add(tid)

    # Merge edges (dedupe by (source, relation, target, provenance)).
    seen = {(e["source"], e["relation"], e["target"], e["provenance"]) for e in kg["edges"]}
    added = 0
    for e in new_edges:
        key = (e["source"], e["relation"], e["target"], e["provenance"])
        if key not in seen:
            kg["edges"].append(e)
            seen.add(key)
            added += 1

    kg["provenance"] = (
        f"{base_kg.get('provenance','')} + literature_correction "
        f"({added} new edges from {literature_dir})"
    )
    kg["literature_edges_added"] = added
    return kg


def main() -> int:
    ap = argparse.ArgumentParser(description="Step 2: correct novel KG with literature citations")
    ap.add_argument("--literature-dir", required=True, help="dir of literature full-text files")
    ap.add_argument("--base-kg", required=True, help="path to base novel_kg.json")
    ap.add_argument("--out", default="-", help="output path (- = stdout)")
    args = ap.parse_args()

    lit_dir = Path(args.literature_dir)
    if not lit_dir.is_dir():
        sys.stderr.write(f"ERROR: literature dir not found: {lit_dir}\n")
        return 1

    base_kg = json.loads(Path(args.base_kg).read_text(encoding="utf-8"))

    # Collect character names from base KG.
    characters: set[str] = set()
    for n in base_kg["nodes"]:
        if n.get("kind") == "character":
            characters.add(n["label"])

    corrected = correct_kg(base_kg, lit_dir, characters)
    out = json.dumps(corrected, indent=2, ensure_ascii=False)
    if args.out == "-":
        print(out)
    else:
        Path(args.out).write_text(out, encoding="utf-8")
    sys.stderr.write(
        f"Step 2 done: +{corrected['literature_edges_added']} literature edges. "
        f"Nodes: {len(corrected['nodes'])}, Edges: {len(corrected['edges'])}\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
