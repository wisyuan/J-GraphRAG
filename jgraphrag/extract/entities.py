"""Text-side rule-based entity detection (pure CPU, zero model).

Ported from experiments/phase53_textside_entities.py (S1).

Background: Phase 52 was structurally FALSIFIED — Qwen2.5-7B 4bit forms no
readable workspace representation for rare proper names (Dozmare / Excalibur /
Princess Frederica / Baron von Pawel-Rammingen), so J-Lens entity-level
extraction is a dead end. But these names explicitly exist in the text, so
detection needs no model at all. This module extracts entities with pure
text-side rules:

- Capitalized spans: runs of 1-4 capitalized content words (internal
  connectors like of/von/der allowed); sentence-initial false positives are
  stripped via "lowercase-form corpus frequency vs. mid-sentence capitalized
  frequency" (the "The" of "The Princess" is ~never capitalized mid-sentence
  and high-frequency lowercase → stripped; "Princess" is capitalized
  mid-sentence often → kept).
- Term spans (medical supplement): lowercase multi-word noun phrases (2-3
  words), corpus frequency >= 2, tail-word verb/function-word blocklist.
- Filtering: stopwords, pure digits, length < 3, Roman numerals; deduped
  against the twopass concept vocab (after _stem normalization).

The output structure (entity_chunks / entity_frequency) mirrors the twopass
concept cache (concept_chunks / concept_frequency).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .filter import _stem

# ── Detection rule constants ─────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'’][A-Za-z]+)*|\d[\d,.]*")
# Sentence boundary: after sentence punctuation / newline (Gutenberg text
# treats newline as paragraph break)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…;:\"”’)])\s+|\n+")

CONNECTORS = {"of", "the", "de", "von", "van", "der", "den", "la", "le",
              "di", "da", "del", "du", "des", "y", "and", "mac", "san"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "could", "shall", "may", "might", "must", "of", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "our", "you",
    "your", "i", "me", "my", "him", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "where", "as", "us",
    "said", "say", "says", "one", "two", "also", "upon", "shall", "let",
    "mr", "mrs", "miss", "dr", "st",
}

# Sentence-initial / single-word capitalization false-positive blocklist
# (fallback beyond the lowercase-frequency rule)
CAP_BLOCKLIST = {
    "the", "this", "that", "these", "those", "there", "here", "then",
    "thus", "hence", "when", "where", "what", "which", "while", "who",
    "how", "why", "yes", "no", "indeed", "however", "moreover",
    "furthermore", "nevertheless", "meanwhile", "afterwards", "chapter",
    "section", "part", "book", "volume", "page", "contents", "index",
    "introduction", "preface", "appendix", "illustration", "fig",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "i", "it", "he", "she", "we", "they", "you", "but", "and", "for",
    "nor", "yet", "so", "as", "if", "though", "although", "because",
    "since", "until", "unless", "after", "before", "now", "soon", "later",
    "perhaps", "certainly", "surely", "really", "truly", "actually",
}

ROMAN_RE = re.compile(r"^[IVXLCDM]+$")

# Contraction forms (I'll/Don't etc. are not entities)
CONTRACTION_RE = re.compile(
    r"^(i|you|he|she|it|we|they|that|there|who|what|let|don|can|won|isn|"
    r"aren|didn|doesn|couldn|wouldn|shouldn|mustn|ain|shan)['’]", re.I)

# Gutenberg boilerplate / discourse-marker false positives (single-word
# capitalized span fallback blocklist)
CAP_BLOCKLIST |= {
    "footnote", "footnotes", "illustrations", "transcriber", "gutenberg",
    "well", "ver", "ebook", "ebooks", "www", "http", "https",
}

# Term-span tail-word blocklist (verbs / function words / generic nouns —
# a rule approximation without a POS tagger)
TERM_TAIL_BLOCK = STOPWORDS | {
    "get", "gets", "getting", "got", "make", "makes", "making", "made",
    "take", "takes", "taking", "took", "taken", "use", "uses", "using",
    "used", "know", "knows", "known", "find", "finds", "found", "ask",
    "asks", "tell", "tells", "told", "go", "goes", "going", "gone",
    "come", "comes", "came", "see", "sees", "saw", "seen", "look",
    "looks", "looked", "want", "wants", "need", "needs", "needed",
    "keep", "keeps", "help", "helps", "give", "gives", "given", "gave",
    "work", "works", "worked", "call", "called", "feel", "feels", "felt",
    "seem", "seems", "seemed", "become", "becomes", "became", "include",
    "includes", "including", "included", "follow", "follows", "following",
    "followed", "develop", "develops", "developing", "developed",
    "cause", "causes", "causing", "caused", "treat", "treats", "treated",
    "way", "ways", "thing", "things", "time", "times", "day", "days",
    "year", "years", "people", "person", "man", "men", "woman", "women",
    "something", "anything", "everything", "nothing", "lot", "lots",
    "kind", "kinds", "type", "types", "number", "part", "parts",
    "much", "many", "less", "least", "well", "better", "best", "worse",
    "worst", "long", "longer", "high", "higher", "highest", "low",
    "lower", "lowest", "early", "earlier", "late", "later", "new",
    "newer", "old", "older", "good", "bad", "big", "small", "large",
    "able", "likely", "unlikely", "possible", "impossible", "common",
    "rare", "similar", "different", "important", "available", "present",
    "certain", "several", "various", "single", "whole", "full", "total",
    "main", "major", "minor", "general", "specific", "particular",
    "according", "depending", "based", "related", "compared", "due",
    "per", "via", "etc", "eg", "ie",
}

MAX_CAP_WORDS = 4        # capitalized span: at most 4 content words (connectors not counted)
TERM_MIN_FREQ = 2        # term span corpus frequency lower bound
S1_TOP_PRINT = 20        # number of high-frequency entities printed for QC

# Phase 52 failure checklist (S1 QC targets — they explicitly exist in the
# text, so detection should be near 100%)
P52_CHECKLIST = {
    "medical": ["basal cell carcinoma", "fair skin", "organ transplant",
                "immune suppression"],
    "novel": ["princess frederica", "baron von pawel-rammingen", "arthur",
              "excalibur", "dozmare"],
}


# ── Text-side entity detection ───────────────────────────────────────────


def _tokenize_sentences(text: str) -> list[list[tuple[str, bool]]]:
    """Sentence split + tokenize. Returns [[(token, is_sentence_initial), ...], ...]."""
    sentences = []
    for sent in SENT_SPLIT_RE.split(text):
        toks = WORD_RE.findall(sent)
        if not toks:
            continue
        sentences.append([(t, i == 0) for i, t in enumerate(toks)])
    return sentences


def _is_cap_word(tok: str) -> bool:
    return (tok[0].isupper() and any(c.isalpha() for c in tok)
            and not CONTRACTION_RE.match(tok))


def _strip_possessive(tok: str) -> str:
    """Strip trailing possessive / lone quotes: Sterne's → Sterne."""
    t = re.sub(r"['’]s$", "", tok, flags=re.I)
    return t.rstrip("'’") or tok


def _word_stats(sentences: list[list[tuple[str, bool]]]) -> tuple[Counter, Counter]:
    """Per-word counts: lowercase occurrences, mid-sentence (non-initial)
    capitalized occurrences."""
    n_lower: Counter = Counter()
    n_cap_mid: Counter = Counter()
    for sent in sentences:
        for tok, is_initial in sent:
            if not any(c.isalpha() for c in tok):
                continue
            if tok.islower():
                n_lower[tok] += 1
            elif _is_cap_word(tok) and not is_initial:
                n_cap_mid[tok.lower()] += 1
    return n_lower, n_cap_mid


def _strip_sentence_initial(run: list[str], n_lower: Counter,
                            n_cap_mid: Counter) -> list[str]:
    """Strip sentence-initial false positives: words whose lowercase form is
    frequent and which are never capitalized mid-sentence."""
    out = list(run)
    while out:
        w = out[0]
        wl = w.lower()
        if wl in CONNECTORS:
            out.pop(0)
            continue
        if n_cap_mid.get(wl, 0) == 0 and n_lower.get(wl, 0) >= 2:
            out.pop(0)
            continue
        break
    return out


def _valid_cap_span(words: list[str]) -> bool:
    if not words:
        return False
    real = [w for w in words if w.lower() not in CONNECTORS]
    if not real:
        return False
    text = " ".join(words)
    if not any(c.isalpha() for c in text):
        return False
    if sum(c.isalpha() for c in text) < 3:
        return False
    if all(w.lower() in STOPWORDS for w in real):
        return False
    if len(real) == 1:
        w = real[0]
        if w.lower() in CAP_BLOCKLIST or w.lower() in STOPWORDS:
            return False
        if ROMAN_RE.match(w):
            return False
        if len(w) < 3 and not w.isupper():
            return False
    return True


def extract_cap_spans(sentences: list[list[tuple[str, bool]]],
                      n_lower: Counter, n_cap_mid: Counter) -> list[str]:
    """Extract capitalized spans (internal connectors allowed), with
    sentence-initial false-positive stripping."""
    spans: list[str] = []
    for sent in sentences:
        toks = [t for t, _ in sent]
        i = 0
        while i < len(toks):
            if not _is_cap_word(toks[i]):
                i += 1
                continue
            run = [toks[i]]
            n_real = 1
            j = i + 1
            while j < len(toks) and n_real < MAX_CAP_WORDS:
                if _is_cap_word(toks[j]):
                    run.append(toks[j])
                    n_real += 1
                    j += 1
                elif (toks[j].lower() in CONNECTORS and j + 1 < len(toks)
                      and _is_cap_word(toks[j + 1]) and n_real < MAX_CAP_WORDS):
                    run.append(toks[j])
                    j += 1
                else:
                    break
            # Drop dangling trailing connector
            while run and run[-1].lower() in CONNECTORS:
                run.pop()
            run = [_strip_possessive(w) for w in run]
            run = _strip_sentence_initial(run, n_lower, n_cap_mid)
            if _valid_cap_span(run):
                spans.append(" ".join(run))
            i = max(j, i + 1)
    return spans


def extract_term_spans(
    sentences: list[list[tuple[str, bool]]],
) -> tuple[Counter, set[str]]:
    """Lowercase multi-word noun-phrase candidates (2-3 words) + the set of
    n-grams that appeared with an initial capital.

    Returns (candidates, cap_grams): the latter rescues freq-1 n-grams that
    appeared capitalized (heading-style terms, e.g. the section heading
    "Immune suppression – People with ...").
    """
    cand: Counter = Counter()
    cap_grams: set[str] = set()
    for sent in sentences:
        toks_l = [t.lower() for t, _ in sent]
        toks_raw = [t for t, _ in sent]
        for n in (3, 2):
            for i in range(len(toks_l) - n + 1):
                gram = toks_l[i:i + n]
                if not all(re.fullmatch(r"[a-z]+", g) for g in gram):
                    continue
                if any(g in STOPWORDS or g in CONNECTORS for g in gram):
                    continue
                if gram[-1] in TERM_TAIL_BLOCK or gram[0] in TERM_TAIL_BLOCK:
                    continue
                if any(len(g) < 3 for g in gram):
                    continue
                key = " ".join(gram)
                cand[key] += 1
                if any(g[0].isupper() for g in toks_raw[i:i + n]):
                    cap_grams.add(key)
    return cand, cap_grams


def detect_entities(domain: str, corpus: dict[str, str],
                    concept_vocab: set[str]) -> dict:
    """Main flow: run rule detection over the whole corpus, return the entity
    cache structure."""
    print(f"  [{domain}] tokenizing {len(corpus)} chunks...", flush=True)
    chunk_sents = {cid: _tokenize_sentences(text)
                   for cid, text in corpus.items()}

    # Corpus-wide word stats (sentence-initial false-positive judgement +
    # term frequency)
    all_sents = [s for sents in chunk_sents.values() for s in sents]
    n_lower, n_cap_mid = _word_stats(all_sents)

    # Concept vocab normalization (_stem per-word merge) for dedup
    def _norm(s: str) -> str:
        return " ".join(_stem(w) for w in s.lower().split())
    concept_norm = {_norm(c) for c in concept_vocab}

    entity_chunks: dict[str, set[str]] = defaultdict(set)
    entity_surface: dict[str, Counter] = defaultdict(Counter)
    entity_type: dict[str, str] = {}

    for cid, sents in chunk_sents.items():
        for span in extract_cap_spans(sents, n_lower, n_cap_mid):
            key = span.lower()
            if _norm(key) in concept_norm:
                continue
            entity_chunks[key].add(cid)
            entity_surface[key][span] += 1
            entity_type[key] = "capitalized"

    if domain == "medical":
        term_cand, cap_grams = extract_term_spans(all_sents)
        # Keep freq>=2; keep freq==1 only if it appeared capitalized
        # (heading-style terms, e.g. "Immune suppression" — only 1 corpus
        # occurrence but a genuine domain term)
        kept = {g: c for g, c in term_cand.items()
                if c >= TERM_MIN_FREQ or g in cap_grams}
        for gram, cnt in kept.items():
            entity_surface[gram][gram] += cnt
        # Re-locate the chunks where each kept term appears
        for cid, sents in chunk_sents.items():
            text_l = " ".join(t for s in sents for t, _ in s).lower()
            for gram in kept:
                if f" {gram} " in f" {text_l} ":
                    if _norm(gram) in concept_norm:
                        continue
                    entity_chunks[gram].add(cid)
                    entity_type[gram] = "term"

    entity_freq = {e: sum(surf.values())
                   for e, surf in entity_surface.items() if entity_chunks[e]}
    entity_chunks = {e: sorted(cids) for e, cids in entity_chunks.items() if cids}
    entity_freq = {e: f for e, f in entity_freq.items() if e in entity_chunks}
    display = {e: surf.most_common(1)[0][0]
               for e, surf in entity_surface.items() if e in entity_chunks}

    # QC 1: Phase 52 failure-checklist hit rate (containment matching:
    # "Princess Frederica of Hanover" covers "princess frederica")
    checklist = {}
    for target in P52_CHECKLIST.get(domain, []):
        hit = any(target == e or target in e or e in target
                  for e in entity_chunks)
        checklist[target] = hit
    n_hit = sum(checklist.values())
    rate = n_hit / len(checklist) if checklist else None

    top = sorted(entity_freq.items(), key=lambda x: -x[1])[:S1_TOP_PRINT]
    print(f"  [{domain}] entities={len(entity_chunks)} "
          f"(cap={sum(1 for t in entity_type.values() if t == 'capitalized')}, "
          f"term={sum(1 for t in entity_type.values() if t == 'term')})")
    print(f"  [{domain}] Phase52 checklist: {n_hit}/{len(checklist)} "
          f"→ {rate:.0%}" if rate is not None else "")
    for t, ok in checklist.items():
        print(f"    {'✓' if ok else '✗'} {t}")
    print(f"  [{domain}] top-{S1_TOP_PRINT} frequent entities (manual QC):")
    for e, f in top:
        print(f"    {f:5d}  {display[e]}")

    return {
        "domain": domain,
        "n_chunks": len(corpus),
        "n_entities": len(entity_chunks),
        "entity_chunks": entity_chunks,
        "entity_frequency": entity_freq,
        "entity_display": display,
        "entity_type": entity_type,
        "qc": {
            "phase52_checklist": checklist,
            "checklist_hit_rate": rate,
            "top_entities": [[display[e], f] for e, f in top],
        },
    }
