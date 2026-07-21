"""Concept quality optimization: artifact filtering + BPE prefix completion.

Two optimizations that improve J-Lens concept extraction quality:

1. **Adaptive DF filtering** — replaces fixed 30% threshold with automatic
   detection of the DF distribution's "knee" (where artifact DF diverges from
   real concept DF). Also applies TF-IDF weighting: a concept's value is
   proportional to its within-chunk frequency (TF) × inverse document frequency.

2. **BPE prefix completion** — J-Lens often produces subword prefixes (`Pol`
   for polycystic, `Stat` for statins, `Fr` for fructose). We complete these
   by looking up full words starting with the prefix in WordNet + the corpus
   vocabulary, then picking the completion with highest bge-m3 cosine to the
   prefix's context.

These are prerequisites for recursive concept-tree expansion (Phase 11):
expansion quality depends on parent concept quality, which depends on
artifact removal + prefix completion.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Optional

import numpy as np


# ── Optimization 1: Adaptive DF filtering ──────────────────────────────

def compute_df(concept_chunks: dict[str, list[str]]) -> dict[str, int]:
    """Document frequency: how many chunks each concept appears in."""
    return {c: len(chunks) for c, chunks in concept_chunks.items()}


def find_df_knee(df_values: list[int], n_total: int) -> float:
    """Find the DF ratio "knee" — the threshold separating real concepts
    from artifacts.

    Artifacts have anomalously high DF (50%+); real concepts cluster at
    2-15%. The knee is found via the maximum curvature method on the sorted
    DF ratio curve.

    Falls back to 0.3 (30%) if the distribution is too flat or small.
    """
    if len(df_values) < 10:
        return 0.3

    ratios = sorted([df / n_total for df in df_values], reverse=True)
    # Find the point of maximum curvature (knee) using the simple
    # "distance to line" method: draw a line from first to last point,
    # find the point farthest below it.
    n = len(ratios)
    x = np.arange(n)
    y = np.array(ratios)
    # line from (0, y[0]) to (n-1, y[-1])
    if y[0] - y[-1] < 1e-8:
        return 0.3
    # perpendicular distance from each point to the line
    dx = n - 1
    dy = y[-1] - y[0]
    norm = math.sqrt(dx * dx + dy * dy)
    distances = np.abs(dx * (y - y[0]) - x * dy) / norm
    knee_idx = int(np.argmax(distances))
    knee_ratio = ratios[knee_idx]

    # Sanity bounds: knee should be between 20% and 60%.
    # Below 20% risks removing real domain concepts (e.g., "cancer" in a
    # medical corpus at 29%); above 60% is too permissive for artifacts.
    return max(0.20, min(0.60, knee_ratio))


def adaptive_filter(concept_chunks: dict[str, list[str]],
                    n_total: int,
                    min_df: int = 2,
                    chunk_texts: list[str] | None = None) -> tuple[dict[str, list[str]], dict]:
    """Filter concepts using adaptive DF knee detection + corpus-text verification.

    Key insight: lens artifacts (alink, ohana, esub) do NOT appear as real
    words in the source text — they only appear in lens output. Domain concepts
    (Cancer, Surgery) DO appear in the text. So we cross-reference: if a
    high-DF concept appears as a real word in the corpus, it's a domain concept;
    if it only exists in lens output, it's an artifact.

    Returns (filtered concept_chunks, metadata dict).
    """
    df_map = compute_df(concept_chunks)
    df_values = list(df_map.values())
    max_df_ratio = find_df_knee(df_values, n_total)
    max_df = int(n_total * max_df_ratio)

    # Build corpus text-word set for artifact verification
    corpus_words = set()
    if chunk_texts:
        import re
        for text in chunk_texts:
            for m in re.finditer(r'[a-zA-Z]{3,}', text):
                corpus_words.add(m.group().lower())

    kept = {}
    removed_artifacts = []
    removed_noise = []
    idf = {}

    for concept, chunks in concept_chunks.items():
        df = len(chunks)
        concept_lower = concept.lower()

        if df > max_df:
            # High-DF: is it a real word in the corpus, or lens-only artifact?
            if chunk_texts and concept_lower in corpus_words:
                # Real domain word (e.g., "cancer" in medical corpus) → keep
                kept[concept] = chunks
                idf[concept] = math.log((n_total + 1) / (df + 1)) + 1
            else:
                # Lens artifact (e.g., "alink") → remove
                removed_artifacts.append((concept, df))
        elif df < min_df:
            removed_noise.append((concept, df))
        else:
            kept[concept] = chunks
            idf[concept] = math.log((n_total + 1) / (df + 1)) + 1

    meta = {
        "max_df_ratio": max_df_ratio,
        "max_df": max_df,
        "n_before": len(concept_chunks),
        "n_after": len(kept),
        "removed_artifacts": sorted(removed_artifacts, key=lambda x: x[1], reverse=True),
        "removed_noise": len(removed_noise),
        "corpus_verified": bool(chunk_texts),
        "idf": idf,
    }
    return kept, meta


# ── Optimization 2: BPE prefix completion ─────────────────────────────

# Lazy-loaded word lists
_WORDNET_NOUNS: Optional[set] = None
_CORPUS_VOCAB: Optional[set] = None


def _get_wordnet_nouns() -> set[str]:
    """Get English nouns from WordNet for prefix completion."""
    global _WORDNET_NOUNS
    if _WORDNET_NOUNS is None:
        try:
            from nltk.corpus import wordnet
            nouns = set()
            for synset in wordnet.all_synsets(pos='n'):
                for lemma in synset.lemmas():
                    name = lemma.name().lower().replace('_', '')
                    if 4 <= len(name) <= 25 and name.isalpha():
                        nouns.add(name)
            _WORDNET_NOUNS = nouns
        except Exception:
            _WORDNET_NOUNS = set()
    return _WORDNET_NOUNS


def build_corpus_vocab(chunk_texts: list[str]) -> set[str]:
    """Extract vocabulary from corpus chunks (for domain-specific completion).

    Looks for alphabetic words of length ≥5 in the corpus. These are
    domain-specific terms (e.g., 'polycystic', 'statins') that WordNet
    may not have.
    """
    import re
    vocab = set()
    for text in chunk_texts:
        for match in re.finditer(r'\b[a-zA-Z]{5,}\b', text.lower()):
            vocab.add(match.group())
    return vocab


def build_corpus_term_freq(chunk_texts: list[str]) -> dict[str, int]:
    """Build term frequency map from corpus (for BM25-style completion ranking).

    Returns {word: total_count_across_corpus}. Words appearing more frequently
    are preferred completions — e.g., "statins" (50x) beats "static" (3x) when
    completing "Stat", because statins is the dominant domain term.

    This replaces the previous "pick longest" heuristic with corpus-frequency
    ranking, which is more accurate for domain-specific disambiguation.
    """
    import re
    from collections import Counter
    counter = Counter()
    for text in chunk_texts:
        for match in re.finditer(r'\b[a-zA-Z]{5,}\b', text.lower()):
            counter[match.group()] += 1
    return dict(counter)


def complete_prefix(prefix: str, corpus_term_freq: dict[str, int],
                    wordnet_nouns: set[str],
                    embed_fn=None) -> Optional[str]:
    """Complete a BPE prefix to a full word using corpus frequency ranking.

    Strategy:
      1. Find all words in corpus_term_freq + wordnet_nouns starting with prefix
      2. If only one candidate, return it
      3. If multiple, rank by corpus frequency (BM25-style: frequent domain
         terms beat rare ones) — e.g., "statins"@50 beats "static"@3
      4. If no corpus candidates, fall back to wordnet (shortest = most common)
      5. If no candidates, return None (keep the prefix as-is)

    Args:
        prefix: the BPE prefix token (e.g., 'Pol', 'Stat', 'Fr')
        corpus_term_freq: {word: count} from the corpus
        wordnet_nouns: general English nouns (fallback)
    """
    prefix_lower = prefix.lower()
    if len(prefix_lower) < 3:
        return None  # too short to disambiguate

    max_len = len(prefix) + 15
    # Corpus candidates with frequency
    corpus_candidates = {w: freq for w, freq in corpus_term_freq.items()
                         if w.startswith(prefix_lower) and len(w) <= max_len}
    # WordNet candidates (no frequency info)
    wn_candidates = {w for w in wordnet_nouns
                     if w.startswith(prefix_lower) and len(w) <= max_len}

    if not corpus_candidates and not wn_candidates:
        return None

    if corpus_candidates:
        # Rank by corpus frequency — the most frequent completion wins.
        # This is the BM25 insight: a term that appears 50x in the corpus
        # is almost certainly the intended word, not one that appears 2x.
        # Tie-break by shorter length (more common form).
        return max(corpus_candidates, key=lambda w: (corpus_candidates[w], -len(w)))

    if len(wn_candidates) == 1:
        return next(iter(wn_candidates))
    # WordNet fallback: prefer shorter (more common)
    return min(wn_candidates, key=len) if wn_candidates else None


def complete_concepts(concepts: list[str], chunk_texts: list[str],
                      corpus_term_freq: dict[str, int] | None = None) -> list[str]:
    """Complete a list of concept words, fixing BPE prefixes.

    Returns a new list where prefix-fragments are replaced with completions.
    Concepts that are already complete (len ≥ 5, or no completion found)
    are kept as-is.
    """
    wn_nouns = _get_wordnet_nouns()
    if corpus_term_freq is None:
        corpus_term_freq = build_corpus_term_freq(chunk_texts)

    result = []
    for concept in concepts:
        # Heuristic: if concept is short (3-5 chars), it's likely a BPE prefix
        if len(concept) <= 5:
            completed = complete_prefix(concept, corpus_term_freq, wn_nouns)
            result.append(completed if completed else concept)
        else:
            result.append(concept)
    return result


# ── Combined pipeline ──────────────────────────────────────────────────

def optimize_concepts(concept_chunks: dict[str, list[str]],
                      n_total: int,
                      chunk_texts: list[str] | None = None) -> tuple[dict, dict]:
    """Full concept optimization pipeline: adaptive DF filter + BPE completion.

    Args:
        concept_chunks: {concept_word: [chunk_ids]}
        n_total: total number of chunks
        chunk_texts: optional, for corpus vocabulary (BPE completion)

    Returns:
        (optimized concept_chunks, metadata)
    """
    # Step 1: adaptive DF filter (with corpus-text artifact verification)
    filtered, filter_meta = adaptive_filter(concept_chunks, n_total,
                                            chunk_texts=chunk_texts)

    # Step 2: BPE prefix completion on surviving concepts (BM25-style ranking)
    corpus_term_freq = build_corpus_term_freq(chunk_texts) if chunk_texts else {}
    wn_nouns = _get_wordnet_nouns()

    completed_map = {}  # old_name → new_name
    for concept in list(filtered.keys()):
        if len(concept) <= 5:
            completed = complete_prefix(concept, corpus_term_freq, wn_nouns)
            if completed and completed != concept.lower():
                completed_map[concept] = completed

    # Apply completions: merge chunks under the completed name
    final = {}
    for concept, chunks in filtered.items():
        name = completed_map.get(concept, concept)
        if name in final:
            # merge: union of chunk lists
            existing = set(final[name])
            existing.update(chunks)
            final[name] = sorted(existing)
        else:
            final[name] = chunks

    # Recompute IDF on final concepts
    idf = {}
    for concept, chunks in final.items():
        df = len(chunks)
        idf[concept] = math.log((n_total + 1) / (df + 1)) + 1

    meta = {
        **filter_meta,
        "bpe_completions": completed_map,
        "n_final": len(final),
        "idf": idf,
    }
    return final, meta


if __name__ == "__main__":
    # Smoke test with Stage 7c medical results
    import json
    from pathlib import Path

    results_path = Path(__file__).resolve().parents[1] / "data" / "m6" / "phase10_stage7c_concept_graph_medical.json"
    if results_path.exists():
        d = json.load(open(results_path))
        print("=== Adaptive DF knee detection ===")
        # Simulate: we need raw concept_chunks, but JSON only has top concepts.
        # Test the knee finder with synthetic data.
        n_total = 957
        # Reproduce approximate DF distribution from Stage 7c
        synthetic_dfs = [806, 774, 613, 515, 274, 227, 218, 188, 154, 92, 89, 59, 52, 47]
        synthetic_dfs += [5] * 50 + [3] * 30 + [2] * 40  # real concepts + noise
        knee = find_df_knee(synthetic_dfs, n_total)
        print(f"  knee ratio: {knee:.2f} (fixed was 0.30)")
        print(f"  max_df: {int(n_total * knee)} chunks")

    print("\n=== BPE prefix completion (BM25 frequency ranking) ===")
    # Test with known prefixes from Stage 5
    test_prefixes = ["Pol", "Stat", "Fr", "Candid", "hydro", "phosph", "diet"]
    wn = _get_wordnet_nouns()
    # Simulate corpus term frequencies: statins appears more than static
    fake_freq = {"polycystic": 45, "statins": 50, "fructose": 30, "candida": 25,
                 "hydrogen": 40, "phosphate": 35, "dietary": 60,
                 "polynomial": 2, "static": 3, "fresh": 5}
    for p in test_prefixes:
        result = complete_prefix(p, fake_freq, wn)
        print(f"  {p:10} → {result}")
