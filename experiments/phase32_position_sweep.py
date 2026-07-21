"""Phase 32: 生成序列 per-position J-Lens 分析。

前置实验：让 LLM 对文档做一次完整回答（generate ~30 tokens），
在每个生成位置的 J-Lens workspace 读取概念，和该位置的实际 token 对照。

验证两个问题：
1. 哪个位置的 lens 概念和文档核心概念匹配度最高（贡献度分布）
2. position=-1（prefill 末尾）是否已覆盖了文档的核心概念

设计：
  - 对每篇文档：concern prompt → generate 30 tokens
  - 在每个位置（-1, 0, 1, ..., 29）读 J-Lens top-k
  - 记录：position, actual_token, lens_topk_concepts
  - 分析：per-position 概念和文档核心词的匹配度

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase32_position_sweep
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english, build_corpus_word_set,
)
from experiments.phase20_concern_full_com import PREFILL_WORDS
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

N_GENERATE = 30
N_WORDS_PER_POS = 8
SCAN_PER_POS = 40
BLACKLIST = STOP_WORDS_EXTENDED | PREFILL_WORDS


def decode_lens_topk(logits_row, tokenizer, n=8, scan=40):
    """Decode top-k content words from one position's lens logits."""
    probs = torch.softmax(logits_row.float(), dim=-1)
    topk = probs.topk(scan)
    results = []
    seen = set()
    for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha()
                and is_ascii_english(low)
                and low not in BLACKLIST
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                results.append({"token": tok, "prob": round(p, 4)})
        if len(results) >= n:
            break
    return results


def generate_with_lens_trace(lens, lens_model, tokenizer, model, prompt,
                              n_new_tokens=N_GENERATE):
    """Generate tokens while recording J-Lens at every position.

    Uses model.generate(output_hidden_states=True) to get proper generation
    with KV cache, then reads J-Lens from the hidden states at each
    generated position.

    Returns:
        generated_tokens: list of decoded token strings
        lens_trace: list of [{position, actual_token, lens_concepts}]
    """
    all_layers = lens.source_layers
    read_layer = all_layers[-1]  # L26

    # Map read_layer to hidden_states index
    # hidden_states is a tuple of (n_layers+1) tensors: [embedding, layer1_out, ..., layerN_out]
    # read_layer=26 → hidden_states[27] (index = layer+1 because embedding is index 0)
    # But for Qwen, the model has 28 layers (0-27). hidden_states has 29 entries.
    # read_layer 26 → hidden_states index 27
    hs_idx = read_layer + 1  # +1 because hidden_states[0] is embedding output

    model_dtype = next(model.lm_head.parameters()).dtype

    # Read J-Lens at position=-1 (prefill end) first
    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[read_layer],
        positions=[-1], max_seq_len=512)
    prefill_concepts = decode_lens_topk(
        lens_logits[read_layer][0], tokenizer, N_WORDS_PER_POS, SCAN_PER_POS)

    lens_trace = [{
        "position": -1,
        "actual_token": "(prefill end)",
        "lens_concepts": prefill_concepts,
    }]

    # Generate with hidden states
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=n_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

    # Extract generated token IDs
    generated_ids = output.sequences[0][input_ids.shape[1]:]
    generated_tokens = [tokenizer.decode([tid]).strip() for tid in generated_ids]

    # Read J-Lens at each generated position from hidden_states
    # output.hidden_states is a tuple of length n_new_tokens
    # Each element is a tuple of (n_layers+1) tensors, each [batch, seq_len, hidden_dim]
    # For generation step i, hidden_states[i][hs_idx] has shape [1, cur_len, hidden_dim]
    # The LAST position of each step's hidden state is the generated token's position
    for step in range(min(n_new_tokens, len(output.hidden_states))):
        step_hidden = output.hidden_states[step]
        if step < len(step_hidden):
            # Get the residual at read_layer for the last position
            h = step_hidden[hs_idx][0, -1].float()  # [hidden_dim], float32 for transport
            h_transport = lens.transport(h, read_layer)
            lens_logits_row = model.lm_head(h_transport.to(model_dtype))
            lens_concepts = decode_lens_topk(
                lens_logits_row, tokenizer, N_WORDS_PER_POS, SCAN_PER_POS)

            lens_trace.append({
                "position": step,
                "actual_token": generated_tokens[step] if step < len(generated_tokens) else "?",
                "lens_concepts": lens_concepts,
            })

    return generated_tokens, lens_trace


def analyze_trace(lens_trace, doc_text, corpus_words):
    """Analyze per-position lens concepts vs document core concepts.

    Question 1: which position has highest overlap with document keywords?
    Question 2: does position=-1 cover most document concepts?
    """
    # Extract document keywords (top frequency content words)
    words = re.findall(r'[a-zA-Z]{5,}', doc_text.lower())
    from collections import Counter
    word_freq = Counter(words)
    # Top 20 document keywords (excluding stopwords)
    doc_keywords = set()
    for w, _ in word_freq.most_common(50):
        if w not in BLACKLIST and w in corpus_words:
            doc_keywords.add(w)
            if len(doc_keywords) >= 20:
                break

    # For each position, compute overlap with doc keywords
    per_pos = []
    for entry in lens_trace:
        lens_words = {c["token"].lower() for c in entry["lens_concepts"]}
        overlap = lens_words & doc_keywords
        per_pos.append({
            "position": entry["position"],
            "actual_token": entry["actual_token"],
            "lens_concepts": [c["token"] for c in entry["lens_concepts"]],
            "overlap_count": len(overlap),
            "overlap_words": sorted(overlap),
        })

    # Question 1: position with max overlap
    best_pos = max(per_pos, key=lambda x: x["overlap_count"])

    # Question 2: does position=-1 cover most concepts?
    pos_neg1 = per_pos[0]  # position=-1
    all_other_concepts = set()
    for entry in per_pos[1:]:  # skip -1
        all_other_concepts.update(entry["lens_concepts"])
    neg1_concepts = set(pos_neg1["lens_concepts"])
    concepts_only_in_generation = all_other_concepts - neg1_concepts

    return {
        "doc_keywords": sorted(doc_keywords)[:20],
        "per_position": per_pos,
        "best_position": best_pos,
        "pos_neg1_coverage": {
            "neg1_concepts": sorted(neg1_concepts),
            "concepts_only_in_generation": sorted(concepts_only_in_generation),
            "neg1_covers_pct": round(len(neg1_concepts) / max(1, len(all_other_concepts)) * 100, 0),
        },
    }


def run_experiment(lens, lens_model, tokenizer, model):
    print("Phase 32: Generate sequence per-position J-Lens analysis")
    print(f"{'='*70}")

    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    chunks = list(corpus.values())

    # Build corpus word set
    corpus_words = set()
    for text in chunks[:200]:
        for m in re.finditer(r'[a-zA-Z]{4,}', text):
            corpus_words.add(m.group().lower())

    # Test on 5 chunks at different positions
    test_indices = [0, 50, 100, 200, 400]
    all_results = []

    for idx in test_indices:
        doc = chunks[idx]
        print(f"\n{'='*60}")
        print(f"Chunk {idx}: {doc[:80]}...")

        # Build concern prompt
        user_msg = (f"What concepts does this text discuss? "
                    f"List 8 one-word concepts.\n\n{doc[:600]}")
        prefill = "The concepts discussed are"
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)

        # Generate with lens trace
        print(f"  Generating {N_GENERATE} tokens with per-position lens...")
        gen_tokens, lens_trace = generate_with_lens_trace(
            lens, lens_model, tokenizer, model, prompt)

        gen_text = " ".join(gen_tokens)
        print(f"  Generated: {gen_text[:100]}")

        # Analyze
        analysis = analyze_trace(lens_trace, doc, corpus_words)

        # Print per-position table
        print(f"\n  {'pos':>4}  {'actual_token':15}  {'lens top-5':40}  {'overlap':>10}")
        print(f"  {'-'*75}")
        for entry in analysis["per_position"]:
            lens_str = ", ".join(entry["lens_concepts"][:5])
            ov_str = ", ".join(entry["overlap_words"][:3]) if entry["overlap_words"] else ""
            print(f"  {entry['position']:>4}  {entry['actual_token']:15}  "
                  f"{lens_str:40}  {ov_str}")

        print(f"\n  Document keywords: {analysis['doc_keywords']}")
        print(f"  Best position: {analysis['best_position']['position']} "
              f"(overlap={analysis['best_position']['overlap_count']})")
        print(f"  Position -1 coverage: {analysis['pos_neg1_coverage']['neg1_covers_pct']:.0f}%")
        print(f"    -1 concepts: {analysis['pos_neg1_coverage']['neg1_concepts']}")
        print(f"    Only in generation: {analysis['pos_neg1_coverage']['concepts_only_in_generation']}")

        all_results.append({
            "chunk_index": idx,
            "doc_excerpt": doc[:200],
            "generated_text": gen_text[:200],
            "doc_keywords": analysis["doc_keywords"],
            "per_position": analysis["per_position"],
            "best_position": analysis["best_position"],
            "pos_neg1_coverage": analysis["pos_neg1_coverage"],
        })

    # Summary across all chunks
    print(f"\n{'='*70}")
    print(f"CROSS-CHUNK SUMMARY")
    print(f"{'='*70}")

    # Q1: average overlap by position
    print(f"\n  Q1: Average doc-keyword overlap by position:")
    max_pos = max(r["best_position"]["position"] for r in all_results
                  if isinstance(r["best_position"]["position"], int))
    # Collect overlap counts per position across chunks
    pos_overlaps = defaultdict(list)
    for r in all_results:
        for entry in r["per_position"]:
            pos_overlaps[entry["position"]].append(entry["overlap_count"])

    print(f"  {'position':>8}  {'avg overlap':>12}  {'max overlap':>12}")
    print(f"  {'-'*35}")
    for pos in sorted(pos_overlaps.keys()):
        vals = pos_overlaps[pos]
        print(f"  {pos:>8}  {np.mean(vals):>10.1f}  {max(vals):>12}")

    # Q2: does -1 cover most concepts?
    print(f"\n  Q2: Does position=-1 cover most document concepts?")
    for r in all_results:
        cov = r["pos_neg1_coverage"]
        print(f"    Chunk {r['chunk_index']}: "
              f"{cov['neg1_covers_pct']:.0f}% coverage, "
              f"{len(cov['concepts_only_in_generation'])} concepts only in generation")

    avg_cov = np.mean([r["pos_neg1_coverage"]["neg1_covers_pct"]
                       for r in all_results])
    print(f"\n  Average -1 coverage: {avg_cov:.0f}%")
    if avg_cov > 70:
        print(f"  → Position -1 already covers most concepts. "
              f"Generation adds little new signal.")
    else:
        print(f"  → Generation positions reveal concepts NOT in position -1. "
              f"Multi-position readout adds value.")

    # Save
    out = {
        "method": "per_position_lens_analysis",
        "n_generate": N_GENERATE,
        "n_chunks_tested": len(all_results),
        "results": all_results,
        "summary": {
            "avg_neg1_coverage_pct": round(float(avg_cov), 0),
        },
    }
    out_path = EXP / "phase32_position_sweep.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running per-position lens analysis...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_experiment(lens, lens_model, tokenizer, model)


if __name__ == "__main__":
    main()
