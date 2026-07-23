"""Pass 2 role expansion + workspace vector collection (J-Lens).

Ported from:
- experiments/phase39_two_pass_cache.py — pass2_role_expansion,
  find_concept_positions, get_wu_first_fragment_vec (reversed prefill
  "The concepts are: {...}", one forward pass yields top-5 role words +
  ActivationRecorder residual → lens.transport → 3584-dim L2-normalized
  ws vector + wu_vec)
- experiments/phase35_prefill_position_scan.py — STOP, decode_topk_custom

Mechanism (phase39 docstring): jlens's lens.apply() returns only unembedded
logits and does not expose the residual, so we hook the layer with
jlens.hooks.ActivationRecorder and call lens.transport(residual, layer) —
exactly what apply() does internally (record → transport → unembed) — but a
single forward yields both the logits (role decoding) and the transported
residuals (vector collection).

Import-safe: torch and jlens are imported lazily inside functions.
"""
from __future__ import annotations

import numpy as np

from .filter import ROLE_STOP, _stem

MAX_PREFILL_CONCEPTS = 8
N_ROLE_DECODE = 8   # decode top-8 per concept position
N_ROLE_KEEP = 5     # keep top-5 after filtering

# Decode-time stoplist for role readout (phase35's STOP — targets prompt
# words and generic template continuations).
STOP = {
    "the","and","for","that","with","from","this","are","was","were","been",
    "have","has","will","would","could","should","not","but","into","also",
    "they","them","than","then","when","what","each","more","most","some",
    "such","only","very","just","like","which","can","all","other","including",
    "those","a","an","of","to","in","is","by","on","or","as","at","its",
    "concept","concepts","key","main","topic","study","studies","result",
    "results","method","patient","patients","treatment","associated","compared",
    "significantly","clinical","using","data","analysis","research","health",
    "disease","medical","group","based","related","following","above","document",
    "documents","discuss","discusses","listed","summarized","outlined",
    "described","describes","shown","shows","found","reported","include",
    "includes","including","involve","cover","covers","focus","focuses",
    "address","addresses","explore","explores","examine","examines","consider",
    "analyzes","investigate","highlight","demonstrate","suggest","indicates",
    "reveal","present","provides","specific","specifically","particular",
    "various","different","certain","general","important","possible",
    "available","first","second","last","new","however","furthermore",
    "moreover","additionally","given","unless","except","among","despite",
    "until","since","today","currently","text","texts","passage","context",
    "excerpt","snippet","outlined","vided","supplied","mentioned","provided",
    "following","segment","fragments","article","paragraph","description",
    "information","discussion","discussions","scope","extract","section",
    "regarding","certainly","indeed","within","according","illustr","quite",
    "here","prim","actually","interestingly","while","although","throughout",
}


def decode_topk_custom(logits_row, tokenizer, n=15, scan=50):
    """Decode top-k content words."""
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(scan)
    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n:
            break
    return results


def find_concept_positions(tokenizer, prompt: str,
                           concepts: list[str]) -> dict[str, int]:
    """Locate each concept's token position in the prefill region.

    Reproduces Phase 35's two-stage matching: exact/normalized match first,
    then first-BPE-fragment prefix match (for multi-token concepts).
    Returns {concept: position} (position = first fragment token index).
    """
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    token_texts = [tokenizer.decode([t]) for t in ids]

    # Find prefill start (right after "concepts are")
    prefill_start = None
    for i, _t in enumerate(token_texts):
        if "concepts are" in "".join(token_texts[max(0, i - 3):i + 1]).lower():
            prefill_start = i + 1
            break
    if prefill_start is None:
        return {}

    concept_positions: dict[str, int] = {}
    # Pass A: exact or containment match
    for i in range(prefill_start, len(token_texts)):
        tok_text = token_texts[i].strip().rstrip(",").lower()
        for c in concepts:
            if c in concept_positions:
                continue
            if tok_text == c.lower() or c.lower() in tok_text:
                concept_positions[c] = i
                break
    # Pass B: first-fragment prefix match (multi-token concepts)
    for i in range(prefill_start, len(token_texts)):
        tok = token_texts[i].strip().lower()
        for c in concepts:
            if c in concept_positions:
                continue
            cl = c.lower()
            if tok == cl or (len(cl) > 3 and cl.startswith(tok) and len(tok) >= 3):
                concept_positions[c] = i
                break
    return concept_positions


def pass2_role_expansion(lens, lens_model, tokenizer, chunk_text: str,
                         concepts: list[str], corpus_words: set[str],
                         layer: int
                         ) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Phase 35 reversed-prefill role expansion in ONE forward pass.

    Returns (roles, ws_vecs):
      roles:   {concept: [role words]} — top-5 filtered role words
      ws_vecs: {concept: np.ndarray[d_model]} — L2-normalized transported
               residual (J_l @ h) at the concept's prefill position

    jlens note: lens.apply() returns only unembedded logits; to get the
    residual itself we hook the layer with jlens.hooks.ActivationRecorder and
    call lens.transport(residual, layer) — identical math to what apply()
    does internally (record → transport → unembed), but a single forward
    yields both the logits (role decoding) and the residuals (vectors).
    """
    import torch

    roles: dict[str, list[str]] = {}
    ws_vecs: dict[str, np.ndarray] = {}
    if not concepts:
        return roles, ws_vecs

    prefill_concepts = concepts[:MAX_PREFILL_CONCEPTS]
    concept_str = ", ".join(prefill_concepts)
    user_msg = f"What concepts does this text discuss?\n\n{chunk_text[:400]}"
    prefill = f"The concepts are: {concept_str}"

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    concept_positions = find_concept_positions(tokenizer, prompt, prefill_concepts)
    if not concept_positions:
        return roles, ws_vecs

    positions_to_read = sorted(set(concept_positions.values()))

    # One forward pass: record residual at `layer`, then transport + unembed
    from jlens.hooks import ActivationRecorder
    input_ids = lens_model.encode(prompt, max_length=1024)
    seq_len = input_ids.shape[1]
    positions_to_read = [p for p in positions_to_read if p < seq_len]
    if not positions_to_read:
        return roles, ws_vecs

    with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as rec:
        lens_model.forward(input_ids)
    # rec.activations[layer]: [1, seq_len, d_model]
    resid = rec.activations[layer][0, positions_to_read].float()  # [n_pos, d]
    transported = lens.transport(resid, layer)      # [n_pos, d_model] final-layer basis
    logits = lens_model.unembed(transported).float().cpu()  # [n_pos, vocab]

    pos_to_idx = {p: i for i, p in enumerate(positions_to_read)}
    chunk_concepts_lower = {c.lower() for c in prefill_concepts}
    chunk_concept_stems = {_stem(c) for c in prefill_concepts}

    for concept, pos in sorted(concept_positions.items(), key=lambda x: x[1]):
        idx = pos_to_idx.get(pos)
        if idx is None:
            continue

        # --- role words: decode top-8, filter, keep top-5 ---
        candidates = decode_topk_custom(logits[idx], tokenizer,
                                        n=N_ROLE_DECODE, scan=40)
        concept_roles: list[str] = []
        seen_stems: set[str] = set()
        for w in candidates:
            tok = w["token"]
            low = tok.lower()
            stem = _stem(tok)
            if low in chunk_concepts_lower or stem in chunk_concept_stems:
                continue                         # not the concept itself / sibling concept
            if low in ROLE_STOP or stem in ROLE_STOP:
                continue                         # generic template word
            if low not in corpus_words:          # must be a real corpus word
                continue
            if stem in seen_stems:               # inflection dedupe (Types vs Type)
                continue
            seen_stems.add(stem)
            concept_roles.append(tok)
            if len(concept_roles) >= N_ROLE_KEEP:
                break
        roles[concept] = concept_roles

        # --- workspace vector: L2-normalized transported residual ---
        vec = transported[idx].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            ws_vecs[concept] = vec / norm

    return roles, ws_vecs


def get_wu_first_fragment_vec(concept: str, lm_head_wu: np.ndarray,
                              tokenizer) -> np.ndarray | None:
    """W_U row vector of the concept's first BPE fragment (L2-normalized).

    Follows Phase 38's get_bpe_fragment_vectors but uses only the first
    fragment (core semantic direction).
    """
    token_ids = tokenizer.encode(concept, add_special_tokens=False)
    if not token_ids:
        return None
    vec = lm_head_wu[token_ids[0]].astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)
