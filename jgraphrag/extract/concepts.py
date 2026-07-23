"""Pass 1 concept extraction (J-Lens workspace readout).

Ported from:
- experiments/phase31_full_pipeline_cache.py — extract_concepts_full_pipeline
  (main path: concern prompt → 27-layer depth gradient → triple filter →
  BM25 BPE completion → <2 fallback)
- experiments/phase20_concern_full_com.py — build_concern_prompt_full,
  compute_concept_profiles_filtered
- experiments/phase17_multihop_depth_gradient.py — extract_depth_gradient
  (+ its per-layer top-k decoder)
- experiments/phase10_jlens_stage7c.py — extract_chunk_concepts (fallback
  single-layer readout) + STOP_CONCEPTS

Import-safe: torch is imported lazily inside functions; importing this module
touches no disk/model/GPU.
"""
from __future__ import annotations

from collections import defaultdict

from .filter import (
    ConceptDepthProfile,
    PREFILL_WORDS,
    STOP_WORDS,
    STOP_WORDS_EXTENDED,
    classify_concept_pos,
    complete_prefix,
    is_ascii_english,
)

# Stop words for the Stage 7c fallback single-layer readout.
STOP_CONCEPTS = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "function", "def", "return", "class", "import",
    "concept", "concepts", "key", "main", "topic", "also", "they", "them",
    "than", "then", "when", "what", "each", "more", "most", "some", "such",
    "only", "very", "just", "like", "question", "list", "following",
    "above", "based", "study", "studies", "result", "results", "method",
    "methods", "patient", "patients", "group", "groups", "treatment",
    "associated", "compared", "significantly", "clinical", "using",
    "data", "analysis", "research", "health", "disease", "medical",
    # filter generic medical-paper boilerplate that appears in every chunk
}


# ── Concern prompt (phase20) ────────────────────────────────────────────

def build_concern_prompt_full(docs: list[str], tokenizer) -> str:
    """Concern prompt (same as Phase 18/19). We use full-layer COM."""
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List 8 one-word concepts.\n\n{doc_block}")
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            pass
    return f"{user_msg}\n{prefill}"


# ── Depth gradient (phase17) ────────────────────────────────────────────

def _decode_layer_topk(
    logits_row,
    tokenizer,
    n_words: int = 8,
    topk_scan: int = 40,
) -> list[dict]:
    """Decode top-k content words from one layer's logits row.

    Returns list of {token, prob} for content words (filtered).
    """
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(topk_scan)

    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_WORDS
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n_words:
            break
    return results


def extract_depth_gradient(
    lens, lens_model, tokenizer,
    prompt: str,
    layers: list[int] | None = None,
    n_words: int = 5,
    max_seq_len: int = 512,
) -> dict[int, list[dict]]:
    """Extract concept words at ALL layers in one forward pass.

    This is the core multihop primitive: a single `lens.apply(layers=all,
    positions=[-1])` gives the concept readout at every depth. The depth
    gradient (how concepts change across layers) is the concept hierarchy.

    Args:
        prompt: the full prompt string (can be plain text OR concern prompt)
        layers: which layers to read. None = all source_layers.
        n_words: max content words per layer

    Returns:
        {layer_int: [{token, prob}, ...]} for each requested layer.
    """
    if layers is None:
        layers = lens.source_layers

    lens_logits, model_logits, _ = lens.apply(
        lens_model, prompt,
        layers=layers,
        positions=[-1],
        max_seq_len=max_seq_len,
    )

    gradient = {}
    for layer in layers:
        gradient[layer] = _decode_layer_topk(
            lens_logits[layer][0], tokenizer, n_words=n_words)

    return gradient


# ── Triple filter (phase20) ─────────────────────────────────────────────

def compute_concept_profiles_filtered(
    gradient: dict[int, list[dict]],
    corpus_words: set[str],
    min_layers: int = 3,
    require_corpus: bool = True,
    require_noun: bool = True,
) -> list[ConceptDepthProfile]:
    """Compute profiles with full-layer COM + triple filtering.

    Filters (replacing Phase 19's band restriction):
      1. ASCII English (filter multilingual artifacts)
      2. Not in STOP_WORDS_EXTENDED or PREFILL_WORDS
      3. Stability: >= min_layers appearances
      4. Must appear in >= 1 workspace layer (>= L10)
      5. Corpus verification (if require_corpus): must be a real word in docs
      6. POS filter (if require_noun): prefer nouns, reject pure verbs
    """
    word_layer_probs: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for layer, words in gradient.items():
        for w in words:
            token = w["token"].lower()
            word_layer_probs[token].append((layer, w["prob"]))

    profiles = []
    for word, layer_probs in word_layer_probs.items():
        # Filter 1: ASCII
        if not is_ascii_english(word):
            continue
        # Filter 2: stopwords + prefill words
        if word in STOP_WORDS_EXTENDED or word in PREFILL_WORDS:
            continue

        layers_appeared = sorted(set(l for l, _ in layer_probs))
        n_layers = len(layers_appeared)

        # Filter 3: stability
        if n_layers < min_layers:
            continue
        # Filter 4: workspace presence
        if not any(l >= 10 for l in layers_appeared):
            continue

        # Filter 5: corpus verification
        in_corpus = word in corpus_words
        if require_corpus and not in_corpus:
            continue

        # Filter 6: POS — reject pure verbs (VBG/VBD), accept NN/NNS/UNK
        if require_noun:
            pos = classify_concept_pos(word)
            if pos in ("VBG", "VBD"):
                continue  # reject gerunds and past participles

        total_prob = sum(p for _, p in layer_probs)
        if total_prob <= 0:
            continue
        com = sum(l * p for l, p in layer_probs) / total_prob

        first = min(layers_appeared)
        last = max(layers_appeared)

        profiles.append(ConceptDepthProfile(
            word=word,
            layers=layers_appeared,
            probs=[p for _, p in sorted(layer_probs)],
            com=com,
            first_layer=first,
            last_layer=last,
            span=last - first,
            n_layers=n_layers,
            total_prob=total_prob,
            in_corpus=in_corpus,
        ))

    profiles.sort(key=lambda p: p.com)
    return profiles


# ── Fallback single-layer readout (stage7c) ─────────────────────────────

def extract_chunk_concepts(lens, lens_model, tokenizer, chunk_text: str,
                           n_words: int = 5) -> list[str]:
    """Extract concept words from a document chunk (long prompt → stable residual).

    Uses the same concern-coupled pattern as Stage 5 (which achieved 80%
    accuracy on NFCorpus clusters). The chunk text provides enough context
    for the model to form stable concept representations, avoiding the
    short-prompt instability that sank Stage 7b (1.4%).
    """
    import torch

    user_msg = (
        f"What concepts does this text discuss? List {n_words} one-word concepts.\n\n"
        f"{chunk_text[:800]}"
    )
    prefill = "The concepts discussed are"
    prompt = f"{user_msg}\n{prefill}"
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass

    last = lens.source_layers[-1]
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[last],
        positions=[-1], max_seq_len=512,
    )

    probs = torch.softmax(lens_logits[last][0].float(), dim=-1)
    topk = probs.topk(30)  # over-extract, then filter

    words = []
    seen = set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_CONCEPTS
                and low not in seen):
            # reject mixed-case BPE fragments (like Stage 5 filter)
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── Full pipeline (phase31) ─────────────────────────────────────────────

def extract_concepts_full_pipeline(
    lens, lens_model, tokenizer,
    chunk_text: str,
    all_layers: list[int],
    corpus_words: set[str],
    corpus_freq: dict[str, int],
    wn_nouns: set[str],
) -> dict:
    """Full Phase 25 pipeline for a single chunk.

    1. concern prompt → full-layer depth gradient
    2. compute_concept_profiles_filtered (stability + corpus + POS)
    3. BM25 BPE completion
    4. Return filtered concepts + COM profiles

    This replaces Stage 7c's single-layer extract_chunk_concepts.
    """
    # Step 1: concern prompt + full depth gradient (1 forward pass, all 27 layers)
    prompt = build_concern_prompt_full([chunk_text], tokenizer)
    gradient = extract_depth_gradient(
        lens, lens_model, tokenizer, prompt,
        layers=all_layers, n_words=8, max_seq_len=512)

    # Step 2: Phase 20 triple filter — strict first, then relax
    # Strict: require_corpus=True (only concepts verified in the chunk text)
    # This filters out lens artifacts (amac/emonic/alink) that aren't real words
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=2,
        require_corpus=True, require_noun=True)

    # Relax: if too strict (< 2 concepts survived), allow non-corpus nouns
    if len(profiles) < 2:
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=2,
            require_corpus=False, require_noun=True)
        # Filter manually: only keep if in corpus (final safety net)
        profiles = [p for p in profiles if p.in_corpus]

    # Collect concept words from profiles
    concepts_raw = [p.word for p in profiles]

    # Fallback: if too few after strict filtering, use single-layer L26 readout
    # (same as Stage 7c — this is the proven fallback from Phase 25 benchmark)
    if len(concepts_raw) < 2:
        concepts_raw = extract_chunk_concepts(
            lens, lens_model, tokenizer, chunk_text, n_words=5)
        # Re-run corpus verification on fallback concepts
        concepts_raw = [c for c in concepts_raw if c.lower() in corpus_words]

    # Step 3: BM25 BPE completion
    concepts_completed = []
    for c in concepts_raw[:8]:
        if len(c) <= 5:
            result = complete_prefix(c, corpus_freq, wn_nouns)
            concepts_completed.append(result if result else c)
        else:
            concepts_completed.append(c)

    # Deduplicate
    seen = set()
    concepts_final = []
    for c in concepts_completed:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            concepts_final.append(c)

    # Build COM profiles for output (diagnostic)
    com_profiles = []
    for p in profiles:
        com_profiles.append({
            "word": p.word,
            "com": round(p.com, 1),
            "n_layers": p.n_layers,
            "first": p.first_layer,
            "last": p.last_layer,
            "in_corpus": p.in_corpus,
        })

    # Raw gradient summary (top-3 per layer, for diagnostics)
    gradient_summary = {}
    for layer in all_layers:
        words = gradient.get(layer, [])
        gradient_summary[str(layer)] = [
            {"token": w["token"], "prob": w["prob"]}
            for w in words[:3]
        ]

    return {
        "concepts": concepts_final[:8],
        "n_concepts": len(concepts_final),
        "com_profiles": com_profiles,
        "gradient_summary": gradient_summary,
    }
