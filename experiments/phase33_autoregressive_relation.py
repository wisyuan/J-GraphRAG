"""Phase 33 PoC: 自回归关系判定——生成序列 + workspace 轨迹分析。

用户猜想：关系不是单次 workspace 读出的一个词，而是生成序列中
跨步稳定的概念轨迹。让模型先生成几个 token，在 J-Lens 序列中找
描述关系的概念词。

流程（每对概念 A, B）：
1. prompt = "The relationship between {A} and {B} is"
2. model.generate(max_new_tokens=5, do_sample=False)
   → 生成 token 序列 [t1, t2, ..., t5]
3. 对每个生成步 i，读 workspace:
   ws_i = lens logits at step i
4. 在 workspace 序列中找:
   - 跨步稳定的词（≥2 步出现 → 强信号）
   - 超过概率阈值的词
   → 这些词描述了 A 和 B 的关系

对比：
  Phase 27: 单次 workspace 读出（静态）
  Phase 33: 生成序列的 workspace 轨迹（动态）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase33_autoregressive_relation
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase27_relation_readout import decode_topk, STOP_REL
from experiments.phase28_relation_graph import RELATION_TYPES

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def generate_with_workspace_trace(
    model, tokenizer, lens, lens_model,
    prompt: str,
    max_new_tokens: int = 5,
    n_workspace_words: int = 15,
) -> dict:
    """Generate tokens while reading workspace at each step.

    Returns:
        generated_tokens: list of decoded tokens
        workspace_trace: list of per-step workspace top-k
        workspace_stable: words appearing in ≥2 steps
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    n_input = input_ids.shape[1]

    generated_tokens = []
    workspace_trace = []

    # We need to hook into the generation loop to read workspace at each step
    # Strategy: generate one token at a time, read workspace, repeat
    current_ids = input_ids

    for step in range(max_new_tokens):
        with torch.no_grad():
            # Forward pass to get logits
            outputs = model(current_ids, use_cache=True)

            # Read workspace via J-Lens at the last position
            # We need to use the lens on the current forward pass
            # lens.apply does its own forward pass, so we call it separately
            # with the current token sequence
            current_prompt = tokenizer.decode(current_ids[0], skip_special_tokens=False)

            try:
                layer = lens.source_layers[-1]
                lens_logits, _, _ = lens.apply(
                    lens_model, current_prompt,
                    layers=[layer],
                    positions=[-1],
                    max_seq_len=512,
                )
                ws_words = decode_topk(
                    lens_logits[layer][0], tokenizer,
                    n=n_workspace_words, scan=40)
                workspace_trace.append(ws_words)
            except Exception:
                workspace_trace.append([])

            # Greedy decode next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            # Stop if EOS
            if next_token_id.item() == tokenizer.eos_token_id:
                break

            generated_tokens.append(tokenizer.decode(next_token_id[0]))
            current_ids = torch.cat([current_ids, next_token_id], dim=-1)

            # Clean up cache to save VRAM
            del outputs

    # Find workspace words that are stable across steps (≥2)
    word_step_counts = defaultdict(int)
    for step_words in workspace_trace:
        seen_this_step = set()
        for w in step_words:
            tok = w["token"].lower()
            if tok not in seen_this_step:
                word_step_counts[tok] += 1
                seen_this_step.add(tok)

    workspace_stable = [
        (word, count) for word, count in word_step_counts.items()
        if count >= 2
    ]
    workspace_stable.sort(key=lambda x: -x[1])

    # Also find workspace words that appear at any step with high probability
    workspace_high_prob = []
    for step_words in workspace_trace:
        for w in step_words:
            if w["prob"] >= 0.05:  # 5% threshold
                workspace_high_prob.append((w["token"].lower(), w["prob"], step))
    # Deduplicate, keep max prob
    prob_seen = {}
    for word, prob, step in workspace_high_prob:
        if word not in prob_seen or prob > prob_seen[word][0]:
            prob_seen[word] = (prob, step)
    workspace_high_prob = sorted(prob_seen.items(), key=lambda x: -x[1][0])

    return {
        "generated_tokens": generated_tokens,
        "generated_text": "".join(generated_tokens).strip(),
        "workspace_trace": workspace_trace,
        "workspace_stable": workspace_stable,
        "workspace_high_prob": [(w, p[0], p[1]) for w, p in workspace_high_prob[:15]],
    }


def build_relation_prompt(concept_a: str, concept_b: str, doc_text: str,
                          tokenizer) -> str:
    """Dual-concept relation prompt."""
    user_msg = (
        f"This text discusses {concept_a} and {concept_b}. "
        f"What is the relationship between {concept_a} and {concept_b}? "
        f"Answer with one word.\n\n{doc_text[:500]}"
    )
    prefill = f"The relationship between {concept_a} and {concept_b} is"
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


# Test cases (same as Phase 27)
TEST_PAIRS = [
    ("cancer", "chemotherapy", "treatment/therapy"),
    ("cancer", "surgery", "treatment/removal"),
    ("smoking", "cancer", "causes/risk"),
    ("tumor", "growth", "causes/involves"),
    ("insulin", "diabetes", "treats/manages"),
    ("calcium", "bones", "builds/strengthens"),
    ("cancer", "metastasis", "spreads/progression"),
    ("diet", "nutrition", "related/part_of"),
    ("blood", "heart", "pumps/circulates"),
    ("diagnosis", "treatment", "leads_to/precedes"),
]


def run_poc(model, tokenizer, lens, lens_model):
    print("Phase 33 PoC: Autoregressive relation via workspace trace")
    print(f"{'='*70}")

    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    chunks = list(corpus.values())
    layer = lens.source_layers[-1]

    results = []

    for concept_a, concept_b, expected in TEST_PAIRS:
        # Find a chunk containing both
        doc_text = ""
        for chunk in chunks:
            cl = chunk.lower()
            if concept_a in cl and concept_b in cl:
                doc_text = chunk
                break
        if not doc_text:
            for chunk in chunks:
                if concept_a in chunk.lower():
                    doc_text = chunk
                    break
        if not doc_text:
            doc_text = chunks[0]

        prompt = build_relation_prompt(concept_a, concept_b, doc_text, tokenizer)

        # Phase 33: autoregressive with workspace trace
        trace_data = generate_with_workspace_trace(
            model, tokenizer, lens, lens_model, prompt,
            max_new_tokens=5, n_workspace_words=15)

        # Phase 27 baseline: single workspace readout
        from experiments.phase27_relation_readout import build_relation_prompt as build_rel
        single_prompt = build_rel(doc_text, concept_a, concept_b, tokenizer)
        lens_logits, _, _ = lens.apply(
            lens_model, single_prompt, layers=[layer],
            positions=[-1], max_seq_len=512)
        single_words = decode_topk(lens_logits[layer][0], tokenizer, n=10, scan=50)

        # Analyze
        gen_text = trace_data["generated_text"]
        stable_words = trace_data["workspace_stable"]
        high_prob = trace_data["workspace_high_prob"]

        # Check for relation words
        single_rel = [w["token"] for w in single_words[:5]
                      if w["token"].lower() in RELATION_TYPES]
        stable_rel = [w for w, c in stable_words if w in RELATION_TYPES]
        gen_rel = [w for w in gen_text.lower().split()
                   if w in RELATION_TYPES and len(w) >= 3]
        high_prob_rel = [w for w, p, s in high_prob if w in RELATION_TYPES]

        print(f"\n  [{concept_a} + {concept_b}] expected: {expected}")
        print(f"    Generated:     '{gen_text}'")
        print(f"    Phase27 single: {[w['token'] for w in single_words[:5]]}")
        print(f"      → known rel: {single_rel}")
        print(f"    Stable (≥2 steps): {[(w,c) for w,c in stable_words[:8]]}")
        print(f"      → known rel: {stable_rel}")
        print(f"    High-prob (≥5%): {[(w,f'{p:.2f}') for w,p,s in high_prob[:5]]}")
        print(f"      → known rel: {high_prob_rel}")

        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "expected": expected,
            "generated_text": gen_text,
            "phase27_top5": [w["token"] for w in single_words[:5]],
            "phase27_known_rel": single_rel,
            "workspace_stable": stable_words[:10],
            "stable_known_rel": stable_rel,
            "high_prob_words": high_prob[:10],
            "high_prob_known_rel": high_prob_rel,
            "generated_known_rel": gen_rel,
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"ANALYSIS")
    print(f"{'='*70}")

    methods = {
        "Phase27 (single)": lambda r: r["phase27_known_rel"],
        "Stable (≥2 steps)": lambda r: r["stable_known_rel"],
        "High-prob (≥5%)": lambda r: r["high_prob_known_rel"],
        "Generated tokens": lambda r: r["generated_known_rel"],
    }

    for name, extractor in methods.items():
        n_with_rel = sum(1 for r in results if extractor(r))
        print(f"  {name:25}: {n_with_rel}/{len(results)} pairs have known relation words")

    # Best combo: stable OR high_prob
    n_combined = sum(1 for r in results
                     if r["stable_known_rel"] or r["high_prob_known_rel"])
    print(f"  {'Stable OR High-prob':25}: {n_combined}/{len(results)} pairs")

    # Save
    cand = detect_model()
    out = {
        "method": "autoregressive_relation_poc",
        "model": cand["name"],
        "n_test_pairs": len(results),
        "results": results,
    }
    out_path = EXP / "phase33_autoregressive_relation_poc.json"
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

    print(f"\n[2/2] Running autoregressive relation PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(model, tokenizer, lens, lens_model)


if __name__ == "__main__":
    main()
