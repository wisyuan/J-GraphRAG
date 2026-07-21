"""Query generator from KG (annotation pipeline Step 3).

Consumes a KG (from gen_code_kg.py or gen_novel_kg.py) and emits (query,
ground_truth) trials by scope-contraction over the graph's relations. Queries
emerge mechanically from relations — no human authoring, no LLM.

Query families per relation type:
  code KG:
    DEFINES          → "Where is {symbol} defined?" gt=[defining file]
    IMPORTS          → "Which files does {file} import?" gt=[imported files]
    REFERENCES_SYMBOL→ "Where is {symbol} used?" gt=[referencing files]
  novel KG:
    APPEARS_IN(char) → "In which chapters does {character} appear?" gt=[chapters]
    INTERACTS_WITH   → "Who does {character} interact with?" gt=[co-characters]

These are zero-pollution (ground-truth derived deterministically from the KG,
which itself was extracted without LLM). They feed AbExperiment.trials directly.

Usage:
    python -m datasets.gen_queries_from_kg <kg.json> [--domain code|novel] > trials.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _node_label(kg: dict, node_id: str) -> str:
    for n in kg["nodes"]:
        if n["id"] == node_id:
            return n.get("label", node_id)
    return node_id


def _node_kind(kg: dict, node_id: str) -> str | None:
    for n in kg["nodes"]:
        if n["id"] == node_id:
            return n.get("kind")
    return None


def gen_code_queries(kg: dict) -> list[dict]:
    trials: list[dict] = []
    by_rel = defaultdict(list)
    for e in kg["edges"]:
        by_rel[e["relation"]].append(e)

    # DEFINES: group by symbol → "Where is X defined?" gt = file(s).
    defines = defaultdict(set)
    for e in by_rel["DEFINES"]:
        # source is file, target is file::symbol
        symbol = e["target"].split("::")[-1]
        defines[symbol].add(e["source"])
    for symbol, files in defines.items():
        if 1 <= len(files) <= 10:
            trials.append({
                "query": f"Where is {symbol} defined?",
                "ground_truth_files": sorted(files),
                "provenance": "kg:DEFINES",
            })

    # IMPORTS: group by source file → "What does X import?" gt = imported files.
    imports = defaultdict(set)
    for e in by_rel["IMPORTS"]:
        imports[e["source"]].add(e["target"])
    for src, targets in imports.items():
        if 1 <= len(targets) <= 10:
            trials.append({
                "query": f"What does {src} import?",
                "ground_truth_files": sorted(targets),
                "provenance": "kg:IMPORTS",
            })

    # REFERENCES_SYMBOL: "Where is X used?" gt = files referencing it.
    refs = defaultdict(set)
    for e in by_rel["REFERENCES_SYMBOL"]:
        symbol = e["target"].split("::")[-1]
        refs[symbol].add(e["source"])
    for symbol, files in refs.items():
        if 1 <= len(files) <= 10:
            trials.append({
                "query": f"Where is {symbol} used?",
                "ground_truth_files": sorted(files),
                "provenance": "kg:REFERENCES_SYMBOL",
            })
    return trials


def gen_novel_queries(kg: dict) -> list[dict]:
    trials: list[dict] = []
    by_rel = defaultdict(list)
    for e in kg["edges"]:
        by_rel[e["relation"]].append(e)

    # APPEARS_IN (character): "In which chapters does X appear?" gt = chapters.
    appears = defaultdict(set)
    for e in by_rel["APPEARS_IN"]:
        if e["source"].startswith("char::"):
            char = e["source"][len("char::"):]
            appears[char].add(e["target"])
    for char, chapters in appears.items():
        if 2 <= len(chapters) <= 20:
            trials.append({
                "query": f"In which chapters does {char} appear?",
                "ground_truth_files": sorted(chapters),
                "provenance": "kg:APPEARS_IN",
            })

    # INTERACTS_WITH: "Who does X interact with?" gt = co-characters.
    interacts = defaultdict(set)
    for e in by_rel["INTERACTS_WITH"]:
        a = e["source"][len("char::"):]
        b = e["target"][len("char::"):]
        interacts[a].add(b)
        interacts[b].add(a)
    for char, others in interacts.items():
        if 1 <= len(others) <= 10:
            trials.append({
                "query": f"Who does {char} interact with?",
                "ground_truth_files": sorted(f"char::{o}" for o in others),
                "provenance": "kg:INTERACTS_WITH",
            })

    # DISCUSSED_IN (theme→chapter, from Step 2 literature correction):
    # "In which chapters is theme X discussed?" gt = chapters.
    # ★ These are the concern-ambiguity queries — the bet#1b decision core.
    discussed = defaultdict(set)
    for e in by_rel["DISCUSSED_IN"]:
        if e["source"].startswith("theme::"):
            theme = e["source"][len("theme::"):]
            discussed[theme].add(e["target"])
    for theme, chapters in discussed.items():
        if 1 <= len(chapters) <= 20:
            trials.append({
                "query": f"In which chapters is the theme of {theme} discussed?",
                "ground_truth_files": sorted(chapters),
                "provenance": "kg:DISCUSSED_IN (literature-corrected)",
            })
    return trials


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kg_json", help="path to KG json (from gen_code_kg / gen_novel_kg)")
    ap.add_argument("--domain", choices=("code", "novel"), required=True)
    args = ap.parse_args()

    kg = json.loads(Path(args.kg_json).read_text(encoding="utf-8"))
    gen = gen_code_queries if args.domain == "code" else gen_novel_queries
    trials = gen(kg)
    print(json.dumps(trials, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
