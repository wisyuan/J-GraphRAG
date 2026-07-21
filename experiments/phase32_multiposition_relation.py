"""Phase 32 PoC: 多位置 J-Lens 语境关系读出。

假设：position=-1 读出字典关系；生成过程中每个 position 的 J-Lens
反映语境绑定的关系演化。

验证：同一对概念，不同文档语境下，生成 5 个 token 的 J-Lens 序列是否不同。

方法：
1. 双概念 prompt → generate 5 tokens（不采样，greedy）
2. 在 prompt 末尾 + 每个生成 token 位置读 J-Lens workspace
3. 对比同一概念对不同文档的 J-Lens 序列

关键问题：position > -1 的 J-Lens 是否读出不同的关系词？

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase32_multiposition_relation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase27_relation_readout import decode_topk, STOP_REL
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def read_multiposition_jlens(
    lens, lens_model, tokenizer, model,
    prompt: str, n_generate: int = 5,
    layers: list[int] | None = None,
) -> list[dict]:
    """Read J-Lens at prompt-end + each generated position.

    1. Forward pass on prompt → read at position=-1
    2. Generate n tokens greedily
    3. For each generated position, forward pass on extended sequence → read

    Returns [{position, generated_token, jlens_top5: [...]}, ...]
    """
    if layers is None:
        layers = [lens.source_layers[-1]]  # L26

    results = []

    # Step 1: Read at prompt end (position=-1)
    lens_logits, model_logits, input_ids = lens.apply(
        lens_model, prompt, layers=layers, positions=[-1], max_seq_len=512)

    for layer in layers:
        words = decode_topk(lens_logits[layer][0], tokenizer, n=8, scan=50)
        results.append({
            "position": -1,
            "layer": layer,
            "generated_token": "(prompt end)",
            "jlens_top8": [{"token": w["token"], "prob": w["prob"]} for w in words],
        })

    # Step 2: Generate n tokens greedily
    input_tensor = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=n_generate,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )

    # Decode generated tokens
    generated_ids = output.sequences[0][input_tensor.shape[1]:]
    generated_tokens = []
    for tid in generated_ids:
        tok = tokenizer.decode([tid])
        generated_tokens.append(tok)

    # Step 3: Read J-Lens at each generated position
    # Build the full sequence (prompt + generated tokens) and read at each new position
    full_ids = output.sequences[0]  # [prompt + generated]

    for gen_idx in range(n_generate):
        # Position in the full sequence
        pos = input_tensor.shape[1] + gen_idx

        # Forward pass up to this position
        truncated = full_ids[:pos + 1].unsqueeze(0)

        # Use lens.apply with the reconstructed prompt
        truncated_text = tokenizer.decode(truncated[0], skip_special_tokens=False)

        try:
            lens_logits, _, _ = lens.apply(
                lens_model, truncated_text,
                layers=layers, positions=[-1], max_seq_len=512)

            for layer in layers:
                words = decode_topk(lens_logits[layer][0], tokenizer, n=8, scan=50)
                results.append({
                    "position": gen_idx,
                    "layer": layer,
                    "generated_token": generated_tokens[gen_idx],
                    "jlens_top8": [{"token": w["token"], "prob": w["prob"]} for w in words],
                })
        except Exception as e:
            results.append({
                "position": gen_idx,
                "layer": layers[0],
                "generated_token": generated_tokens[gen_idx],
                "error": str(e)[:100],
            })

    return results


def build_dual_concept_prompt(doc_text, concept_a, concept_b, tokenizer):
    user_msg = (
        f"This text discusses {concept_a} and {concept_b}. "
        f"Describe their relationship in this context.\n\n"
        f"{doc_text[:500]}\n\n"
        f"The relationship between {concept_a} and {concept_b} in this text is"
    )
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": ""}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        except Exception:
            pass
    return f"{user_msg} "


def find_docs_for_concepts(corpus, concept_a, concept_b, n=2):
    """Find n documents mentioning both concepts, with different contexts."""
    docs_both = []
    docs_a_only = []
    for cid, text in corpus.items():
        cl = text.lower()
        has_a = concept_a in cl
        has_b = concept_b in cl
        if has_a and has_b:
            docs_both.append((cid, text))
        elif has_a:
            docs_a_only.append((cid, text))

    # Return diverse documents (prefer different first 50 chars)
    results = []
    seen_starts = set()
    for cid, text in docs_both:
        start = text[:50]
        if start not in seen_starts:
            seen_starts.add(start)
            results.append((cid, text))
        if len(results) >= n:
            break

    # Fill with docs_a_only if needed
    while len(results) < n and docs_a_only:
        cid, text = docs_a_only.pop(0)
        results.append((cid, text))

    return results[:n]


def run_poc(lens, lens_model, tokenizer, model):
    print("Phase 32 PoC: Multi-position J-Lens contextual relation readout")
    print(f"{'='*70}")

    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    corpus_dict = {cid: text for cid, text in list(corpus.items())[:500]}

    # Test pairs
    TEST_PAIRS = [
        ("cancer", "surgery"),
        ("cancer", "radiation"),
        ("tumor", "diagnosis"),
        ("blood", "cancer"),
    ]

    layer = lens.source_layers[-1]
    all_results = []

    for concept_a, concept_b in TEST_PAIRS:
        docs = find_docs_for_concepts(corpus_dict, concept_a, concept_b, n=2)

        if len(docs) < 2:
            print(f"\n  [{concept_a} + {concept_b}] SKIP (need 2 docs, found {len(docs)})")
            continue

        print(f"\n{'='*60}")
        print(f"  Concept pair: {concept_a} + {concept_b}")
        print(f"{'='*60}")

        pair_data = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "contexts": [],
        }

        for doc_idx, (cid, doc_text) in enumerate(docs):
            prompt = build_dual_concept_prompt(doc_text, concept_a, concept_b, tokenizer)
            positions = read_multiposition_jlens(
                lens, lens_model, tokenizer, model, prompt,
                n_generate=5, layers=[layer])

            # Print
            context_label = f"Context {doc_idx+1}"
            print(f"\n  [{context_label}] doc: {doc_text[:80]}...")
            print(f"  {'pos':>4}  {'gen_tok':12} {'J-Lens top-5':50}")
            print(f"  {'-'*70}")
            for pos in positions:
                if "error" in pos:
                    print(f"  {pos['position']:>4}  {pos.get('generated_token',''):12} ERROR: {pos['error'][:40]}")
                    continue
                words_str = ", ".join(f"{w['token']}({w['prob']:.2f})"
                                       for w in pos["jlens_top8"][:5])
                print(f"  {pos['position']:>4}  {pos['generated_token']:12} {words_str}")

            # Collect all J-Lens words across positions
            all_words = set()
            for pos in positions:
                if "jlens_top8" in pos:
                    for w in pos["jlens_top8"]:
                        all_words.add(w["token"].lower())

            pair_data["contexts"].append({
                "label": context_label,
                "doc_excerpt": doc_text[:150],
                "positions": positions,
                "all_jlens_words": sorted(all_words),
            })

        # Compare across contexts
        if len(pair_data["contexts"]) >= 2:
            words_1 = set(pair_data["contexts"][0]["all_jlens_words"])
            words_2 = set(pair_data["contexts"][1]["all_jlens_words"])
            overlap = words_1 & words_2
            unique_1 = words_1 - words_2
            unique_2 = words_2 - words_1

            print(f"\n  Context comparison:")
            print(f"    Context 1 all words ({len(words_1)}): {sorted(words_1)[:15]}")
            print(f"    Context 2 all words ({len(words_2)}): {sorted(words_2)[:15]}")
            print(f"    Overlap: {len(overlap)} {sorted(overlap)[:10]}")
            print(f"    Unique to ctx1: {sorted(unique_1)[:10]}")
            print(f"    Unique to ctx2: {sorted(unique_2)[:10]}")

            # Compare position=-1 (dictionary) vs position>0 (contextual)
            for ctx in pair_data["contexts"]:
                pos_minus1 = [p for p in ctx["positions"] if p["position"] == -1]
                pos_after = [p for p in ctx["positions"] if p["position"] >= 0]

                if pos_minus1 and pos_after:
                    w0 = set(w["token"].lower() for w in pos_minus1[0]["jlens_top8"][:5])
                    w_after = set()
                    for p in pos_after:
                        if "jlens_top8" in p:
                            w_after.update(w["token"].lower() for w in p["jlens_top8"][:5])
                    new_in_generation = w_after - w0
                    print(f"\n    [{ctx['label']}] pos=-1 → {sorted(w0)[:5]}")
                    print(f"    [{ctx['label']}] new in gen → {sorted(new_in_generation)[:10]}")

            pair_data["comparison"] = {
                "overlap": sorted(overlap),
                "unique_ctx1": sorted(unique_1)[:10],
                "unique_ctx2": sorted(unique_2)[:10],
                "n_overlap": len(overlap),
                "n_unique": len(unique_1) + len(unique_2),
            }

        all_results.append(pair_data)

    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")

    n_different = 0
    n_same = 0
    for r in all_results:
        if "comparison" in r:
            c = r["comparison"]
            if c["n_unique"] > c["n_overlap"]:
                n_different += 1
                verdict = "DIFFERENT (context matters)"
            else:
                n_same += 1
                verdict = "same"
            print(f"  {r['concept_a']:10} + {r['concept_b']:10}: "
                  f"overlap={c['n_overlap']}, unique={c['n_unique']} → {verdict}")

    total = n_different + n_same
    if total > 0:
        print(f"\n  {n_different}/{total} pairs show context-dependent relations")
        if n_different > n_same:
            print(f"  → Multi-position readout captures context-specific relations!")
        else:
            print(f"  → Relations remain context-independent.")

    # Save
    cand = detect_model()
    out = {
        "method": "multiposition_relation_poc",
        "model": cand["name"],
        "n_pairs": len(all_results),
        "n_context_dependent": n_different,
        "results": all_results,
    }
    out_path = EXP / "phase32_multiposition_relation_poc.json"
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

    print(f"\n[2/2] Running multi-position relation PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(lens, lens_model, tokenizer, model)


if __name__ == "__main__":
    main()
