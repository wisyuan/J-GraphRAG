"""Cache filtering rules and shared filter constants (pure CPU, no model).

Ported from:
- experiments/phase39b_filter_cache.py   — concept_ok / role_ok
- experiments/phase20_concern_full_com.py — PREFILL_WORDS
- experiments/phase18_centroid_hierarchy.py — STOP_WORDS_EXTENDED,
  is_ascii_english, build_corpus_word_set, ConceptDepthProfile
- experiments/phase16a_cross_domain_pos.py — STOP_WORDS, classify_concept_pos
  (suffix-heuristic POS, plus its suffix tables)
- experiments/concept_quality.py — _get_wordnet_nouns, build_corpus_term_freq,
  complete_prefix (BM25-style BPE prefix completion)
- experiments/phase39_two_pass_cache.py — ROLE_STOP, _stem

The validated Phase 25 concept filter configuration is:
  1. DF >= 2 (applied by the caller, which owns the DF counts)
  2. ASCII English
  3. not in PREFILL_WORDS / STOP_WORDS_EXTENDED
  4. POS not VBG/VBD (verb forms)
Role filtering appends: POS not RB* (adverbs) and not in ROLE_STOP.

WARNING: every constant here is part of the validated filter configuration.
Dropping or altering one changes the extraction product vs. the final-review
(Phase 50/53) artifacts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# ── Prefill-word blacklist (phase20) ────────────────────────────────────

# Prefill words from the concern prompt that contaminate deep layers.
# These appear in the prompt itself and leak into the readout.
PREFILL_WORDS = {
    "concepts", "discussed", "discusses", "discuss", "discussing",
    "types", "aspects", "terms", "factors", "elements", "mentions",
    "references", "topics", "descriptions", "list", "include",
    "includes", "including", "involve", "involves", "involving",
    "cover", "covers", "covered", "covering",
    "address", "addresses", "addressed", "addressing",
    "mention", "mentioned", "mentioning",
    "describe", "describes", "described", "describing",
    "relate", "relates", "related", "relating",
    "focus", "focuses", "focused", "focusing",
    "explore", "explores", "explored", "exploring",
    "examine", "examines", "examined", "examining",
    "consider", "considers", "considered", "considering",
    "highlight", "highlights", "highlighted", "highlighting",
    "analyze", "analyzes", "analyzed", "analyzing",
    "investigate", "investigates", "investigated", "investigating",
    "report", "reports", "reported", "reporting",
    "present", "presents", "presented", "presenting",
    "provide", "provides", "provided", "providing",
    "demonstrate", "demonstrates", "demonstrated",
    "suggest", "suggests", "suggested",
    "indicate", "indicates", "indicated",
    "reveal", "reveals", "revealed",
    "show", "shows", "shown", "showing",
    "study", "studies", "studied",
    "find", "finds", "found", "finding", "findings",
}

# ── Stopword lists (phase16a / phase18) ─────────────────────────────────

# Base stopwords (phase16a) — used by the depth-gradient layer decoder.
STOP_WORDS = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study", "studies",
    "result", "results", "method", "patient", "patients", "treatment",
    "associated", "compared", "significantly", "clinical", "using", "data",
    "analysis", "research", "health", "disease", "medical", "group",
    "following", "above", "document", "documents", "discuss", "discusses",
    "related", "based", "summarized", "listed", "outlined", "recent",
    "several", "evidence", "suggesting", "characterized", "indicating",
    "primarily", "showed", "might", "show", "seem", "appear", "require",
    "occur", "arise", "include", "single", "similar", "three", "five",
    "once", "which", "these", "those", "their", "there", "where",
    "while", "about", "after", "before", "between", "during", "through",
    "without", "within", "because", "however", "although", "whether",
    "many", "much", "both", "other", "another", "same", "different",
    "important", "possible", "available", "specific", "particular",
    "general", "common", "rare", "high", "low", "large", "small",
    "first", "second", "last", "next", "new", "old",
}

# Structural noise words found in the Phase 17/18 analysis — paper
# formatting / methodology / connective words, not domain concepts.
STOP_WORDS_EXTENDED = {
    # Base stopwords (from phase16a)
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study", "studies",
    "result", "results", "method", "patient", "patients", "treatment",
    "associated", "compared", "significantly", "clinical", "using", "data",
    "analysis", "research", "health", "disease", "medical", "group",
    "following", "above", "document", "documents", "discuss", "discusses",
    "related", "based", "summarized", "listed", "outlined", "recent",
    "several", "evidence", "suggesting", "characterized", "indicating",
    "primarily", "showed", "might", "show", "seem", "appear", "require",
    "occur", "arise", "include", "single", "similar", "three", "five",
    "once", "which", "these", "those", "their", "there", "where",
    "while", "about", "after", "before", "between", "during", "through",
    "without", "within", "because", "however", "although", "whether",
    "many", "much", "both", "other", "another", "same", "different",
    "important", "possible", "available", "specific", "particular",
    "general", "common", "rare", "high", "low", "large", "small",
    "first", "second", "last", "next", "new", "old",
    # Paper structure words (Phase 18 noise)
    "findings", "methodology", "participants", "subjects", "population",
    "design", "designs", "designed", "describing", "described",
    "conclusion", "question", "questions", "summary", "objective",
    "background", "purpose", "approach", "framework", "review",
    "assessment", "assessments", "evaluated", "evaluation", "evaluations",
    "reported", "reporting", "published", "investigated", "investigating",
    "conducted", "examined", "analyzed", "measured", "determined",
    "observed", "demonstrated", "indicated", "suggests", "revealed",
    "shown", "included", "involving", "selected", "randomized",
    "divided", "assigned", "recruited", "enrolled",
    # Generic research/limitation words
    "limited", "limitations", "available", "availability", "accessibility",
    "empirical", "experimental", "prospective", "retrospective",
    "standard", "standardized", "routine", "practice", "practical",
    "alternative", "application", "applications", "applied",
    "number", "numbers", "amount", "amounts", "levels", "values",
    "rate", "rates", "frequency", "content", "contents",
    "overall", "generally", "typically", "usually", "often",
    "however", "furthermore", "moreover", "additionally", "consequently",
    "given", "unless", "except", "among", "outside", "despite",
    "until", "since", "today", "currently",
}

# ── Role-word blocklist + stemming (phase39) ────────────────────────────

# Role-specific blocklist: generic template words the model emits at prefill
# positions regardless of document content (Phase 35-37's "template cloze"
# failure mode). Distinct from phase35's STOP (which targets prompt words).
ROLE_STOP = {
    "type", "types", "aspect", "aspects", "basic", "basics", "part", "parts",
    "kind", "kinds", "form", "forms", "thing", "things", "way", "ways",
    "overview", "summary", "example", "examples", "detail", "details",
    "role", "roles", "feature", "features", "element", "elements",
    "component", "components", "category", "categories", "variety", "varieties",
}


def _stem(word: str) -> str:
    """Crude singular form for dedupe keys only (not for output)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


# ── Corpus verification helpers (phase18) ───────────────────────────────

@dataclass
class ConceptDepthProfile:
    """A concept word's distribution across workspace layers."""
    word: str
    layers: list[int]          # which layers it appears in
    probs: list[float]         # probability at each layer
    com: float                 # center of mass (prob-weighted average layer)
    first_layer: int           # first appearance
    last_layer: int             # last appearance
    span: int                  # last - first
    n_layers: int              # how many layers it appears in
    total_prob: float          # sum of probabilities
    in_corpus: bool            # verified in cluster documents
    role: str = "unknown"      # "meta" / "sub" / "noise" (assigned later)


def build_corpus_word_set(doc_texts: list[str]) -> set[str]:
    """Build set of real words appearing in documents (for corpus verification)."""
    words = set()
    for text in doc_texts:
        for m in re.finditer(r'[a-zA-Z]{4,}', text):
            words.add(m.group().lower())
    return words


def is_ascii_english(word: str) -> bool:
    """Filter out non-English tokens (multilingual BPE artifacts).

    Lens sometimes produces tokens from other languages (usuarios, novità,
    männer). These are artifacts, not domain concepts. We require ASCII.
    """
    try:
        word.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


# ── POS classification (phase16a; suffix heuristic, no NLTK data needed) ─

# Noun suffixes: strong signal that a word is a noun
NOUN_SUFFIXES = (
    'tion', 'ment', 'ness', 'ity', 'sion', 'osis', 'oma', 'ism', 'ist',
    'logy', 'pathy', 'emia', 'uria', 'graphy', 'plasia', 'rrhea',
    'cy', 'ce', 'cracy', 'hood', 'ship', 'dom',
)

# Adjective suffixes
ADJ_SUFFIXES = (
    'ful', 'less', 'ous', 'ive', 'able', 'ible', 'ical', 'ish',
    'like', 'most', 'ward',
)

# Adjective suffixes that need length check (short words like "al", "ic")
ADJ_SUFFIXES_SHORT = ('al', 'ic', 'id', 'an', 'ar', 'en', 'or')


def classify_concept_pos(word: str) -> str:
    """Suffix-heuristic POS classification.

    Returns one of:
      NN   — noun (strong suffix like -tion, -ment)
      NNS  — plural noun (-s)
      VBG  — verb gerund/participle (-ing)
      VBD  — verb past tense (-ed)
      JJ   — adjective (-ful, -ous, -ive, ...)
      UNK  — unknown (could be a root noun like "Cancer" or BPE fragment)

    The UNK category is the ambiguous one — resolved by corpus verification
    in `concept_quality_score`.
    """
    w = word.lower()
    if len(w) < 3:
        return 'SHORT'

    # Verb forms (check before noun, since -ing/-ed can overlap)
    if w.endswith('ing') and len(w) > 4:
        return 'VBG'
    if w.endswith('ed') and len(w) > 3 and not w.endswith('eed'):
        # "eed" words like "need", "seed", "feed" are usually not VBD
        # but "treated", "linked" are
        return 'VBD'

    # Strong noun suffixes
    for suffix in NOUN_SUFFIXES:
        if w.endswith(suffix):
            return 'NN'

    # Plural (but not ss, us, is, os — those are singular)
    if w.endswith('s') and not w.endswith(('ss', 'us', 'is', 'os', 'xs')):
        return 'NNS'

    # Adjective suffixes (long ones first)
    for suffix in ADJ_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return 'JJ'
    for suffix in ADJ_SUFFIXES_SHORT:
        if w.endswith(suffix) and len(w) > 5:
            return 'JJ'

    return 'UNK'


# ── BPE prefix completion (concept_quality) ─────────────────────────────

# Lazy-loaded word lists
_WORDNET_NOUNS: Optional[set] = None


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


def build_corpus_term_freq(chunk_texts: list[str]) -> dict[str, int]:
    """Build term frequency map from corpus (for BM25-style completion ranking).

    Returns {word: total_count_across_corpus}. Words appearing more frequently
    are preferred completions — e.g., "statins" (50x) beats "static" (3x) when
    completing "Stat", because statins is the dominant domain term.

    This replaces the previous "pick longest" heuristic with corpus-frequency
    ranking, which is more accurate for domain-specific disambiguation.
    """
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


# ── Validated filter rules (phase39b) ───────────────────────────────────

def concept_ok(word: str) -> tuple[bool, str]:
    """Phase 25 concept filter. Returns (keep, reason_if_dropped)."""
    w = word.lower()
    if not is_ascii_english(w):
        return False, "non_ascii"
    if w in PREFILL_WORDS or w in STOP_WORDS_EXTENDED:
        return False, "prefill_stopword"
    pos = classify_concept_pos(w)
    if pos in ("VBG", "VBD"):
        return False, "verb_form"
    return True, ""


def role_ok(word: str) -> bool:
    """Role filter: concept rules + adverbs + ROLE_STOP (stem-aware)."""
    w = word.lower()
    ok, _ = concept_ok(w)
    if not ok:
        return False
    if w in ROLE_STOP or _stem(w) in ROLE_STOP:
        return False
    if classify_concept_pos(w).startswith("RB"):
        return False
    return True
