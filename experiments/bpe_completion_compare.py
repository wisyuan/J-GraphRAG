"""BPE prefix completion: autoregressive (model.generate) vs BM25 (corpus freq).

Motivation
----------
J-Lens concept readout produces BPE subword prefixes (`Stat`, `phosph`,
`Hydro`) instead of full words. Current fix: BM25 corpus-frequency lookup
(`concept_quality.complete_prefix`) — pick the word starting with the prefix
that appears most often in the corpus.

Problem: BM25 is global — it picks the same completion regardless of which
document produced the prefix. If a medical corpus has "status"@75 and
"statins"@3, BM25 always picks "status" even when the source document is
about statins.

Alternative: let the model complete the prefix itself via `model.generate`.
The model sees the document context and can pick the contextually-correct
continuation. This needs no corpus frequency table, no WordNet, no lens —
just the LM head that already knows "Stat" in a statins document → "ins".

Experiment
----------
1. Load corpus documents (NFCorpus + medical).
2. J-Lens readout per document → find short tokens (BPE prefix candidates).
3. For each prefix, run BOTH methods:
   a. BM25: `complete_prefix(prefix, corpus_term_freq, wordnet)`
   b. Autoregressive: `model.generate(prompt_with_prefix)` → decode continuation
4. LLM judge: which completion better describes the document's topic?
5. Report accuracy + case-by-case comparison.

No lens needed for the completion itself (we use model.generate, not lens
transport). The lens is only used to find which prefixes to test.

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.bpe_completion_compare
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.concept_quality import (
    complete_prefix, build_corpus_term_freq, _get_wordnet_nouns,
)
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── Autoregressive completion ─────────────────────────────────────────

def complete_prefix_autoregressive(
    prefix: str,
    doc_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 5,
    max_doc_len: int = 500,
) -> str | None:
    """Complete a BPE prefix by letting the model generate the continuation.

    The model sees the document context + the prefix, then generates the rest
    of the word. This is context-aware: "Stat" in a statins doc → "ins", but
    "Stat" in a statistics doc → "istics".

    Uses the chat template with assistant prefill — the model continues
    writing the prefix word, rather than generating a new word from scratch.

    Returns the full completed word (prefix + continuation), or None if
    the model generated something that doesn't form a clean word.
    """
    import torch

    prefix_clean = prefix.strip()
    doc_excerpt = doc_text[:max_doc_len].strip()

    user_msg = (
        f"What is the main topic of this text? Complete the word.\n\n"
        f"{doc_excerpt}\n\n"
        f"The main topic is: {prefix_clean}"
    )

    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefix_clean}],
            tokenize=False,
            continue_final_message=True,
            add_generation_prompt=False,
        )
    except Exception:
        prompt = f"{user_msg}"

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (after the input)
    new_token_ids = output[0][input_ids.shape[1]:]
    raw_continuation = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    # Clean: take only alphabetic chars until first non-alpha
    # (model may add space, punctuation, or start a new word)
    continuation = ""
    for ch in raw_continuation:
        if ch.isalpha():
            continuation += ch
        else:
            break

    if not continuation:
        return None

    full_word = prefix_clean + continuation

    # Sanity: the completed word should be longer than the prefix
    if len(full_word) <= len(prefix_clean):
        return None

    return full_word


# ── Test case collection ──────────────────────────────────────────────

def collect_bpe_prefixes(
    lens, lens_model, tokenizer,
    doc_texts: list[str],
    max_docs: int = 40,
    layer: int | None = None,
) -> list[dict]:
    """Run J-Lens readout on documents, collect BPE prefix candidates.

    A BPE prefix candidate is a readout token that:
    - is short (3-7 chars)
    - is alphabetic
    - starts with uppercase or lowercase letter
    - is not a known full English word (checked via simple heuristic: if
      it appears verbatim in the corpus as a standalone word, it's complete)

    Returns list of {doc_idx, doc_text, prefix, layer} dicts.
    """
    if layer is None:
        layer = lens.source_layers[-1]

    results = []
    n = min(max_docs, len(doc_texts))

    for i in range(n):
        text = doc_texts[i]
        # Build concern prompt (same as Stage 6)
        user_msg = (
            f"What is the main topic of this document? Answer in one word.\n\n"
            f"{text[:600]}"
        )
        prefill = "The main topic is"
        try:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            prompt = f"{user_msg}\n{prefill}"

        # J-Lens readout at the readout layer
        try:
            lens_logits, _, _ = lens.apply(
                lens_model, prompt,
                layers=[layer],
                positions=[-1],
                max_seq_len=384,
            )
        except Exception as e:
            print(f"    skip doc {i}: lens.apply failed: {e}", flush=True)
            continue

        # Top-10 tokens
        import torch
        probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
        topk = probs.topk(10)

        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            token_str = tokenizer.decode([idx]).strip()
            # Filter: short alphabetic tokens (potential BPE prefixes)
            if 3 <= len(token_str) <= 7 and token_str.isalpha() and token_str.isascii():
                results.append({
                    "doc_idx": i,
                    "doc_text": text,
                    "prefix": token_str,
                    "prob": p,
                    "layer": layer,
                })

        if (i + 1) % 10 == 0:
            print(f"    scanned {i+1}/{n} docs, {len(results)} prefix candidates",
                  flush=True)

    return results


def is_complete_word(token: str, corpus_term_freq: dict[str, int]) -> bool:
    """Check if a token is already a complete word in the corpus.

    If the token appears as-is in the corpus word frequency map (case-
    insensitive), it's likely a complete word, not a BPE prefix.
    """
    return token.lower() in corpus_term_freq


# ── Comparison ────────────────────────────────────────────────────────

def run_comparison(
    prefix_cases: list[dict],
    corpus_term_freq: dict[str, int],
    model,
    tokenizer,
    llm=None,
    max_cases: int = 30,
) -> dict:
    """For each BPE prefix: run BM25 + autoregressive, compare.

    If llm is provided, use it to judge which completion is better given
    the document context. Otherwise, just report both completions side by side.
    """
    wn_nouns = _get_wordnet_nouns()
    results = []

    seen = set()  # deduplicate (doc_idx, prefix)
    for case in prefix_cases:
        key = (case["doc_idx"], case["prefix"])
        if key in seen:
            continue
        seen.add(key)
        if len(results) >= max_cases:
            break

        prefix = case["prefix"]
        doc_text = case["doc_text"]

        # Skip if token is already a complete word
        if is_complete_word(prefix, corpus_term_freq):
            continue

        # Method 1: BM25 corpus frequency
        bm25_result = complete_prefix(prefix, corpus_term_freq, wn_nouns)

        # Method 2: Autoregressive
        auto_result = complete_prefix_autoregressive(
            prefix, doc_text, model, tokenizer,
        )

        # If both are None, skip (no completion possible)
        if not bm25_result and not auto_result:
            continue

        entry = {
            "prefix": prefix,
            "doc_excerpt": doc_text[:200].replace("\n", " "),
            "bm25": bm25_result,
            "autoregressive": auto_result,
            "prob": case.get("prob", 0),
        }

        # LLM judge (if available)
        if llm and (bm25_result or auto_result):
            winner = llm_judge_completion(
                prefix, doc_text, bm25_result, auto_result, llm,
            )
            entry["judge"] = winner
            print(f"  {prefix:10} BM25={str(bm25_result):20} "
                  f"Auto={str(auto_result):20} judge={winner}",
                  flush=True)
        else:
            print(f"  {prefix:10} BM25={str(bm25_result):20} "
                  f"Auto={str(auto_result):20}",
                  flush=True)

        results.append(entry)

    # Summary
    if llm:
        bm25_wins = sum(1 for r in results if r.get("judge") == "bm25")
        auto_wins = sum(1 for r in results if r.get("judge") == "autoregressive")
        ties = sum(1 for r in results if r.get("judge") == "tie")
        neither = sum(1 for r in results if r.get("judge") == "neither")
        n = len(results)
        summary = {
            "n_cases": n,
            "bm25_wins": bm25_wins,
            "auto_wins": auto_wins,
            "ties": ties,
            "neither": neither,
            "bm25_rate": bm25_wins / n if n else 0,
            "auto_rate": auto_wins / n if n else 0,
        }
    else:
        summary = {"n_cases": len(results)}

    return {"cases": results, "summary": summary}


def llm_judge_completion(
    prefix: str,
    doc_text: str,
    bm25_result: str | None,
    auto_result: str | None,
    llm,
) -> str:
    """Ask LLM: given the document, which completion of the prefix is better?

    Returns: "bm25", "autoregressive", "tie", or "neither".
    """
    doc_excerpt = doc_text[:600].replace("\n", " ").strip()

    options = []
    if bm25_result:
        options.append(f"A) {bm25_result}")
    if auto_result:
        options.append(f"B) {auto_result}")

    if not options:
        return "neither"
    if not bm25_result:
        return "autoregressive"
    if not auto_result:
        return "bm25"
    if bm25_result.lower() == auto_result.lower():
        return "tie"

    prompt = (
        f'A concept extraction system produced the prefix "{prefix}" as a topic '
        f"descriptor for the following document. Two methods completed this prefix "
        f"to a full word:\n\n"
        f"  A) {bm25_result}\n"
        f"  B) {auto_result}\n\n"
        f'Document excerpt:\n"{doc_excerpt}"\n\n'
        f"Which completion better describes the document's actual topic? "
        f"Answer with exactly one word: A, B, tie, or neither."
    )

    try:
        resp = llm.complete(prompt, max_tokens=10, temperature=0.0)
        answer = resp.strip().lower()
        if answer.startswith("a"):
            return "bm25"
        elif answer.startswith("b"):
            return "autoregressive"
        elif "tie" in answer:
            return "tie"
        else:
            return "neither"
    except Exception:
        return "neither"


# ── Corpus loading ────────────────────────────────────────────────────

def load_nfcorpus_docs(max_docs: int = 100) -> list[str]:
    """Load NFCorpus documents from BEIR cache."""
    from experiments.corpus_loader import load_beir_fine_records
    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    return [r[1] for r in records]


def load_medical_docs(max_docs: int = 50) -> list[str]:
    """Load medical corpus from GraphRAG-Bench, split into chunks."""
    path = Path("/tmp/graphrag-bench/medical.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    full_text = data[0]["context"]

    # Split into ~1000-char chunks at sentence boundaries
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > 1200 and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())

    return chunks[:max_docs]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="BPE completion: autoregressive vs BM25 comparison",
    )
    ap.add_argument("--max-docs", type=int, default=40,
                    help="max documents to scan for BPE prefixes")
    ap.add_argument("--max-cases", type=int, default=25,
                    help="max prefix cases to compare")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip LLM judge (just report both completions)")
    ap.add_argument("--corpus", choices=["nfcorpus", "medical", "both"],
                    default="both")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 70)
    print("BPE Completion: Autoregressive (model.generate) vs BM25 (corpus freq)")
    print("=" * 70)

    # Load model
    print("\n[1/4] Loading model...")
    cand = detect_model()
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    # Load lens (for prefix collection only)
    print("\n[2/4] Loading lens (for BPE prefix collection)...")
    lens = load_lens(cand["local_lens_path"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    # LLM judge
    llm = None
    if not args.no_judge:
        try:
            from jgraphrag.llm import DeepSeekProvider
            llm = DeepSeekProvider()
            print("  LLM judge: DeepSeek (enabled)")
        except Exception as e:
            print(f"  LLM judge: disabled ({e})")

    # Load corpus
    print(f"\n[3/4] Loading corpus ({args.corpus})...")
    all_docs = []
    corpus_labels = []
    if args.corpus in ("nfcorpus", "both"):
        nf = load_nfcorpus_docs(max_docs=args.max_docs)
        all_docs.extend(nf)
        corpus_labels.extend(["nfcorpus"] * len(nf))
        print(f"  nfcorpus: {len(nf)} docs")
    if args.corpus in ("medical", "both"):
        med = load_medical_docs(max_docs=args.max_docs)
        all_docs.extend(med)
        corpus_labels.extend(["medical"] * len(med))
        print(f"  medical: {len(med)} docs")

    if not all_docs:
        print("ERROR: no documents loaded")
        return

    # Build corpus term frequency (for BM25)
    corpus_term_freq = build_corpus_term_freq(all_docs)
    print(f"  corpus vocab: {len(corpus_term_freq)} unique words")

    # Collect BPE prefixes via J-Lens
    print(f"\n[4/4] Collecting BPE prefixes via J-Lens readout...")
    layer = lens.source_layers[-1]
    prefix_cases = collect_bpe_prefixes(
        lens, lens_model, tokenizer, all_docs,
        max_docs=len(all_docs), layer=layer,
    )
    print(f"  collected {len(prefix_cases)} prefix candidates")

    # Deduplicate by prefix (keep highest-prob doc for each unique prefix)
    best_per_prefix = {}
    for case in prefix_cases:
        p = case["prefix"]
        if p not in best_per_prefix or case["prob"] > best_per_prefix[p]["prob"]:
            best_per_prefix[p] = case

    unique_prefixes = sorted(best_per_prefix.values(),
                             key=lambda x: x["prob"], reverse=True)
    print(f"  unique prefixes: {len(unique_prefixes)}")

    # Run comparison
    print(f"\n{'=' * 70}")
    print(f"COMPARISON RESULTS")
    print(f"{'=' * 70}")

    result = run_comparison(
        unique_prefixes, corpus_term_freq,
        model, tokenizer, llm,
        max_cases=args.max_cases,
    )

    # Print summary
    summary = result["summary"]
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    if "bm25_wins" in summary:
        n = summary["n_cases"]
        print(f"  Total cases:      {n}")
        print(f"  BM25 wins:        {summary['bm25_wins']} ({summary['bm25_rate']:.0%})")
        print(f"  Autoregressive:   {summary['auto_wins']} ({summary['auto_rate']:.0%})")
        print(f"  Ties:             {summary['ties']}")
        print(f"  Neither:          {summary['neither']}")
    else:
        print(f"  Total cases: {summary['n_cases']} (no LLM judge)")

    # Save
    cand = detect_model()
    out = {
        "method": "bpe_completion_autoregressive_vs_bm25",
        "model": cand["name"],
        "layer": layer,
        "corpus": args.corpus,
        "n_docs": len(all_docs),
        "n_prefixes_total": len(prefix_cases),
        "n_prefixes_unique": len(unique_prefixes),
        **result,
    }
    out_path = EXP / "bpe_completion_compare.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
