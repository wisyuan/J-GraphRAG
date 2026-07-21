"""Phase 34: 完整生成过程的位置贡献度分析。

让模型完整生成一个关系答案（~30 tokens），记录每一步的：
1. 生成的 token
2. J-Lens workspace top-k
3. workspace 中领域词的比例和概率

目标：找到"内容转折点"——workspace 从结构词主导切换到领域词主导的位置。
如果能找到稳定的转折点，就可以只读转折点之后的几步，大幅减少 forward pass。

分析指标（每步）：
  - domain_ratio: 领域词占 top-k 的比例
  - domain_prob_sum: 领域词的概率总和
  - top1_domain: top-1 是否是领域词
  - novel_domain_words: 该步首次出现的领域词数量

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase34_full_generation_trace
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase27_relation_readout import decode_topk

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"

# Structural / prompt words that dominate early generation steps
STRUCTURAL = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then",
    "what", "each", "more", "most", "some", "such", "only", "very",
    "just", "like", "concept", "concepts", "key", "main", "topic",
    "study", "studies", "result", "results", "method", "patient",
    "patients", "treatment", "associated", "compared", "significantly",
    "clinical", "using", "data", "analysis", "research", "health",
    "disease", "medical", "group", "based", "related", "following",
    "above", "document", "documents", "discuss", "discusses",
    "relationship", "between", "discussed", "discusses", "discuss",
    "described", "describes", "describe", "shown", "shows", "show",
    "found", "finds", "reported", "reports", "report", "include",
    "includes", "including", "involve", "involves", "cover", "covers",
    "focus", "focuses", "address", "addresses", "explore", "explores",
    "examine", "examines", "consider", "considers", "analyze",
    "investigate", "highlight", "highlights", "demonstrate",
    "suggest", "suggests", "indicate", "indicates", "reveal",
    "present", "provides", "specific", "specifically", "particular",
    "various", "different", "certain", "general", "important",
    "possible", "available", "first", "second", "last", "new",
    "however", "furthermore", "moreover", "additionally",
    "given", "unless", "except", "among", "despite", "until",
    "since", "today", "currently", "text", "texts", "passage",
    "context", "contexts", "contextual", "excerpt", "excerpts",
    "snippet", "snippets", "outlined", "vided", "supplied",
    "mentioned", "provided", "following", "segment", "segments",
    "fragment", "fragments", "article", "paragraph", "description",
    "information", "discussion", "discussions", "scope", "extract",
    "section", "regarding", "certainly", "indeed", "within",
    "according", "illustr", "quite", "here", "prim", "actually",
    "interestingly", "while", "although", "throughout",
    "understanding", "urnished", "offered", "conveyed",
    "significant", "relation", "nection", "sist", "disc",
    "htags", "diesem", "apesh", "olicited", "terms", "gratis",
    "reviewing", "contexto", "texto", "primary",
    # Punctuation/structural tokens
    "In", "The", "This", "These", "Those", "Their", "There",
    "When", "What", "Each", "More", "Most", "Some", "Such",
    "Only", "However", "Indeed", "Regarding", "Certainly",
    "According", "Illustr", "Actually", "Interestingly",
    "Throughout", "Understanding", "Prim", "Quite", "Here",
}

# Known relation type words
RELATION_WORDS = {
    "treatment", "treat", "treats", "treated", "treating", "therapy",
    "therapeutic", "therapies", "reatment", "trat",
    "cause", "causes", "caused", "causing", "causal", "caus",
    "prevent", "prevents", "prevented", "preventing", "prevention",
    "risk", "component", "contains", "source", "builds", "build",
    "strengthens", "spreads", "spread", "progression", "growth",
    "inhibits", "promotes", "manages", "removal", "associated",
    "induces", "reduces", "increases", "affects", "improves",
    "protects", "targets", "kills", "supports", "requires",
    "produces", "regulates", "stimulates", "suppresses",
    "essential", "crucial", "vital", "integral", "critical", "central",
    "circulation", "circulated", "circ", "pumps", "pumping",
    "sequential", "guided", "dependent", "maintenance",
}


def is_domain_word(word: str) -> bool:
    """Check if a word is domain-specific (not structural)."""
    w = word.lower()
    return (len(w) >= 4 and w not in STRUCTURAL
            and w.isalpha()
            and not w.startswith(("http", "www", "@@")))


def full_generation_trace(
    model, tokenizer, lens, lens_model,
    prompt: str,
    max_new_tokens: int = 30,
    n_workspace_words: int = 15,
) -> dict:
    """Full generation with workspace read at every step.

    Records per-step: generated token, workspace top-k, domain metrics.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    current_ids = input_ids

    steps = []
    seen_domain_words = set()

    for step in range(max_new_tokens):
        # Read workspace at current position
        current_text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
        layer = lens.source_layers[-1]

        try:
            lens_logits, _, _ = lens.apply(
                lens_model, current_text,
                layers=[layer], positions=[-1],
                max_seq_len=512)
            ws_words = decode_topk(
                lens_logits[layer][0], tokenizer,
                n=n_workspace_words, scan=50)
        except Exception:
            ws_words = []

        # Analyze workspace
        all_ws = [(w["token"], w["prob"]) for w in ws_words]
        domain_ws = [(t, p) for t, p in all_ws if is_domain_word(t)]
        relation_ws = [(t, p) for t, p in all_ws
                       if t.lower() in RELATION_WORDS]
        novel_domain = [t for t, _ in domain_ws if t.lower() not in seen_domain_words]
        seen_domain_words.update(t.lower() for t, _ in domain_ws)

        domain_ratio = len(domain_ws) / max(1, len(all_ws))
        domain_prob_sum = sum(p for _, p in domain_ws)
        top1_domain = len(domain_ws) > 0 and domain_ws[0] == all_ws[0]

        # Generate next token
        with torch.no_grad():
            outputs = model(current_ids, use_cache=True)
            next_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_logits, dim=-1).unsqueeze(0)

        gen_token = tokenizer.decode(next_token_id[0])

        steps.append({
            "step": step,
            "generated_token": gen_token,
            "workspace_top5": [{"token": t, "prob": round(p, 4)} for t, p in all_ws[:5]],
            "domain_words": [t for t, _ in domain_ws[:5]],
            "relation_words": [t for t, _ in relation_ws[:5]],
            "novel_domain_words": novel_domain[:5],
            "domain_ratio": round(domain_ratio, 3),
            "domain_prob_sum": round(domain_prob_sum, 4),
            "top1_is_domain": top1_domain,
        })

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        current_ids = torch.cat([current_ids, next_token_id], dim=-1)
        del outputs

    # Find "content transition point": first step where domain_ratio > 0.3
    transition_step = None
    for s in steps:
        if s["domain_ratio"] > 0.3 or s["domain_prob_sum"] > 0.1:
            transition_step = s["step"]
            break

    # Find peak domain step: highest domain_prob_sum
    peak_step = max(steps, key=lambda s: s["domain_prob_sum"])

    return {
        "generated_text": "".join(s["generated_token"] for s in steps).strip(),
        "n_steps": len(steps),
        "transition_step": transition_step,
        "peak_step": peak_step["step"],
        "peak_domain_prob": peak_step["domain_prob_sum"],
        "steps": steps,
        "all_domain_words": sorted(seen_domain_words),
        "all_relation_words": sorted(
            w.lower() for s in steps for w in s["relation_words"]),
    }


def build_relation_prompt(concept_a: str, concept_b: str, doc_text: str,
                          tokenizer) -> str:
    user_msg = (
        f"This text discusses {concept_a} and {concept_b}. "
        f"What is the relationship between {concept_a} and {concept_b}? "
        f"Answer in one sentence.\n\n{doc_text[:500]}"
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


TEST_PAIRS = [
    ("cancer", "chemotherapy"),
    ("cancer", "surgery"),
    ("smoking", "cancer"),
    ("tumor", "growth"),
    ("insulin", "diabetes"),
    ("calcium", "bones"),
    ("blood", "heart"),
    ("diagnosis", "treatment"),
]


def run_experiment(model, tokenizer, lens, lens_model):
    print("Phase 34: Full generation workspace trace — position contribution")
    print(f"{'='*70}")

    corpus, _ = load_graphrag_bench("medical", max_queries=1)
    chunks = list(corpus.values())

    all_results = []

    for concept_a, concept_b in TEST_PAIRS:
        # Find doc with both concepts
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
        trace = full_generation_trace(
            model, tokenizer, lens, lens_model, prompt,
            max_new_tokens=30, n_workspace_words=15)

        print(f"\n  [{concept_a} + {concept_b}]")
        print(f"    Generated: '{trace['generated_text'][:80]}'")
        print(f"    Steps: {trace['n_steps']}, "
              f"transition: step {trace['transition_step']}, "
              f"peak: step {trace['peak_step']} "
              f"(prob={trace['peak_domain_prob']:.3f})")
        print(f"    Domain words: {trace['all_domain_words'][:10]}")
        print(f"    Relation words: {trace['all_relation_words'][:10]}")

        # Per-step breakdown
        print(f"\n    {'step':>4} {'gen':>8} {'dom%':>5} {'domΣ':>7} {'top3_workspace':>40} {'domain':>30}")
        for s in trace["steps"]:
            ws3 = " ".join(f"{w['token']}({w['prob']:.2f})" for w in s["workspace_top5"][:3])
            dom = ", ".join(s["domain_words"][:4])
            marker = " ★" if s["domain_prob_sum"] > 0.1 else ""
            print(f"    {s['step']:>4} {s['generated_token']:>8} "
                  f"{s['domain_ratio']:>4.0%} {s['domain_prob_sum']:>6.3f} "
                  f"{ws3:>40} {dom:>30}{marker}")

        all_results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            **trace,
        })

    # Cross-pair analysis
    print(f"\n{'='*70}")
    print(f"CROSS-PAIR ANALYSIS")
    print(f"{'='*70}")

    # When does domain content peak?
    peak_steps = [r["peak_step"] for r in all_results]
    transition_steps = [r["transition_step"] for r in all_results if r["transition_step"] is not None]

    print(f"\n  Transition step (domain_ratio > 0.3):")
    print(f"    Values: {transition_steps}")
    print(f"    Mean: {np.mean(transition_steps):.1f}, Median: {np.median(transition_steps):.0f}")

    print(f"\n  Peak step (max domain_prob_sum):")
    print(f"    Values: {peak_steps}")
    print(f"    Mean: {np.mean(peak_steps):.1f}, Median: {np.median(peak_steps):.0f}")

    # Per-step aggregate: average domain_prob_sum across all pairs
    print(f"\n  Per-step average domain_prob_sum (across {len(all_results)} pairs):")
    max_steps = max(r["n_steps"] for r in all_results)
    for step in range(min(max_steps, 30)):
        values = []
        for r in all_results:
            if step < len(r["steps"]):
                values.append(r["steps"][step]["domain_prob_sum"])
        if values:
            avg = np.mean(values)
            bar = "█" * int(avg * 50)
            print(f"    step {step:>2}: {avg:.3f} {bar}")

    # Unique domain words discovered per step (cumulative)
    print(f"\n  Cumulative unique domain words discovered:")
    cumulative = set()
    for step in range(min(max_steps, 30)):
        new_words = set()
        for r in all_results:
            if step < len(r["steps"]):
                new_words.update(r["steps"][step]["novel_domain_words"])
        cumulative.update(new_words)
        print(f"    step {step:>2}: +{len(new_words):>2} new → {len(cumulative):>3} total")

    # Save
    cand = detect_model()
    out = {
        "method": "full_generation_position_contribution",
        "model": cand["name"],
        "n_pairs": len(all_results),
        "transition_steps": transition_steps,
        "peak_steps": peak_steps,
        "results": all_results,
    }
    out_path = EXP / "phase34_full_generation_trace.json"
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

    print(f"\n[2/2] Running full generation trace...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_experiment(model, tokenizer, lens, lens_model)


if __name__ == "__main__":
    main()
