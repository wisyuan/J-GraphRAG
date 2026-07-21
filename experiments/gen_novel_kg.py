"""Novel-domain KG generator (annotation pipeline Step 1, bet#1b).

Builds a knowledge graph from a novel via deterministic extraction:
  entities   = characters (NER) + chapters + locations (NER)
  relations  = character APPEARS_IN chapter, character INTERACTS_WITH character
               (co-occurrence within a chapter window), location APPEARS_IN chapter

NER uses spaCy — a fixed pretrained model, NOT the LLM under test. So this is
zero-pollution (extracting named entities is pattern recognition, not concern
judgment). spaCy is optional: if not installed, falls back to a name-list
heuristic (less accurate but still deterministic & pollution-free).

Feeds Step 2 (academic-literature correction: paper says "复仇 in 第3章" → add
edge theme复仇—APPEARS_IN—第3章) and Step 3 (scope contraction → queries).

Output: JSON {nodes, edges}.

Usage:
    LINCLE_NOVEL_PATH=/path/to/novel.txt python -m datasets.gen_novel_kg > novel_kg.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.load_novel import chunk_novel
from jgraphrag.config import require_novel_path

# spaCy is optional — degrade gracefully to name-list if absent.
_NLP = None


def _get_nlp():
    """Load a spaCy model for NER. Tries Chinese first (novels likely Chinese),
    then English. Returns None if spaCy unavailable."""
    global _NLP
    if _NLP is not None:
        return _NLP
    try:
        import spacy
    except ImportError:
        return None
    for model in ("zh_core_web_sm", "en_core_web_sm"):
        try:
            _NLP = spacy.load(model)
            return _NLP
        except OSError:
            continue
    return None


# Fallback name list for Chinese novels (common classical-character surnames +
# titles). Very lossy, but deterministic & pollution-free when spaCy is absent.
_CN_NAME_HINTS = (
    "宝玉 黎明 黛玉 宝钗 凤姐 贾母 贾政 王夫人 薛姨妈 湘云 探春 迎春 惜春 元春 "
    "秦可卿 尤二姐 尤三姐 贾琏 贾珍 贾蓉 贾环 贾兰 妙玉 晴雯 麝月 袭人 紫鹃 雪雁 "
    "贾赦 邢夫人 薛蟠 夏金桂 香菱 平儿 鸳鸯 香菱".split()
)


def _extract_entities(text: str) -> tuple[set[str], set[str]]:
    """Return (persons, locations) found in text. NER if spaCy available, else
    name-list heuristic."""
    nlp = _get_nlp()
    persons: set[str] = set()
    locations: set[str] = set()
    if nlp is not None:
        doc = nlp(text[:50000])  # cap to avoid OOM on huge chapters
        for ent in doc.ents:
            if ent.label_ in ("PERSON",):
                persons.add(ent.text.strip())
            elif ent.label_ in ("GPE", "LOC", "LOCATION"):
                locations.add(ent.text.strip())
    else:
        # Heuristic: substring match against the name hint list.
        for name in _CN_NAME_HINTS:
            if name in text:
                persons.add(name)
    # Filter trivially short / noisy entities.
    persons = {p for p in persons if len(p) >= 2}
    locations = {loc for loc in locations if len(loc) >= 2}
    return persons, locations


def generate() -> dict:
    path = Path(require_novel_path())
    text = path.read_text(encoding="utf-8", errors="ignore")
    chapters = chunk_novel(text)

    nodes: list[dict] = []
    edges: list[dict] = []

    # Node: each chapter.
    for ch in chapters:
        nodes.append({
            "id": f"chapter_{ch['id']}",
            "kind": "chapter",
            "label": ch["title"],
            "text_len": len(ch["text"]),
        })

    # Extract entities per chapter, build co-occurrence.
    char_chapters: dict[str, list[int]] = defaultdict(list)
    loc_chapters: dict[str, list[int]] = defaultdict(list)
    chapter_chars: dict[int, set[str]] = {}

    for ch in chapters:
        cid = ch["id"]
        persons, locations = _extract_entities(ch["text"])
        chapter_chars[cid] = persons
        for p in persons:
            char_chapters[p].append(cid)
        for loc in locations:
            loc_chapters[loc].append(cid)

    # Nodes for characters/locations.
    for name, cids in char_chapters.items():
        if len(cids) >= 2:  # appear in ≥2 chapters to be a real entity
            nodes.append({
                "id": f"char::{name}", "kind": "character", "label": name,
                "chapter_count": len(cids),
            })
    for name, cids in loc_chapters.items():
        if len(cids) >= 2:
            nodes.append({
                "id": f"loc::{name}", "kind": "location", "label": name,
                "chapter_count": len(cids),
            })

    # Edges: APPEARS_IN.
    for name, cids in char_chapters.items():
        if len(cids) < 2:
            continue
        for cid in cids:
            edges.append({
                "source": f"char::{name}", "relation": "APPEARS_IN",
                "target": f"chapter_{cid}", "provenance": "ner:cooccur",
            })
    for name, cids in loc_chapters.items():
        if len(cids) < 2:
            continue
        for cid in cids:
            edges.append({
                "source": f"loc::{name}", "relation": "APPEARS_IN",
                "target": f"chapter_{cid}", "provenance": "ner:cooccur",
            })

    # Edges: INTERACTS_WITH — two characters co-occur in ≥2 chapters.
    char_names = [n for n, cids in char_chapters.items() if len(cids) >= 2]
    for i, a in enumerate(char_names):
        a_set = set(char_chapters[a])
        for b in char_names[i + 1:]:
            shared = a_set & set(char_chapters[b])
            if len(shared) >= 2:
                edges.append({
                    "source": f"char::{a}", "relation": "INTERACTS_WITH",
                    "target": f"char::{b}", "provenance": "ner:cooccur",
                    "weight": len(shared),
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "provenance": "spaCy_NER_or_name_list_heuristic (deterministic, no LLM)",
        "spaCy_used": _get_nlp() is not None,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2, ensure_ascii=False))
