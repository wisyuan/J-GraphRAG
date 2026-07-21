"""Phase 10 Stage 2 — J-Lens 静态读出质量验证（条件 A）。

Stage 1（基础设施）通过后，本脚本验证核心命题：J-Lens 能否把簇质心残差
读出为人类可读的概念词——比 Phase 9/9b 的编码器 MLM head 读出好多少。

流程（条件 A：静态读出，无关切耦合）：
  1. pi-code FineRecords → bge-m3 嵌入 → HDBSCAN 簇
  2. 每簇：把簇内文档拼成一个 prompt → Qwen3-4B forward → J-Lens 读最后位置
     的多层残差 → top-k token = 概念词候选
  3. LLM judge（DeepSeek）：簇文档 vs 读出的概念词，是否准确？
  4. 对比 Phase 9/9b（编码器 MLM head）的 LLM judge 准确率

门控（Stage 2 → Stage 3）：
  - 如果 J-Lens 读出 LLM judge 准确率 > Phase 9/9b + 15% margin
    → J-Lens 静态读出已足够，产品路线成立
  - 否则 → 进 Stage 3（关切耦合读出）

为什么"拼文档当 prompt"是合法的静态读出：
  J-Lens 的原生用法就是 prompt-driven（论文的 modulation 实验就是给 prompt
  读残差）。无显式 query 时，文档本身就是"关切"——模型读到这批代码自然会
  激活相关概念（password/hash/auth 等）。这是条件 A（最弱条件），Stage 3
  再加显式 query 强化关切。

关键参数：
  - positions=[-1]：读最后 token 位置（概念最丰富处）
  - layers：读多个 source layer，取"最早收敛到稳定概念"的层（中间层常比
    final 更早形成概念——这是 J-Lens 相对 logit-lens 的优势）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage2 --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols
from experiments.partition import partition_hdbscan
from experiments.phase9_xlmr_vec2vec import judge_readout  # reuse Phase 9 judge
from jgraphrag.llm import DeepSeekProvider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Reuse Stage 1's model/lens loading config
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens,
)
from experiments.phase10_jlens_stage1 import _model_dir_complete


def build_cluster_prompt(members: list[str], member_texts: list[str],
                         max_chars: int = 1000, tokenizer=None) -> str:
    """Build a concern-coupled concept-readout prompt for one cluster.

    For instruction-tuned models: use the chat template. The user message asks
    for the shared concept; we provide an assistant prefill "The shared concept
    is" and read out at the last token of the prefill. The model's residual at
    that position is "reaching for" the concept word that completes the sentence.

    For base models (no chat template / tokenizer=None): fall back to a raw
    concern-prefixed prompt (weaker — Stage 2 showed base models can't abstract).

    Design rationale (validated across Stage 2 iterations):
      - Suffix-only ("docs... The concept is") → contamination (model predicts
        "concept"/"conceptual" family, the suffix's own continuation).
      - Prefix-only ("Identify the theme: docs") → model reads docs as code,
        reads out code tokens.
      - Chat template + assistant prefill "The shared concept is" → for IT
        models, the prefill forces the model to commit to a concept word at the
        readout position. This is the J-Lens native pattern (the prompt IS the
        concern; the prefill localizes where the concept must form).
    """
    parts = []
    total = 0
    for text in member_texts:
        chunk = text[:300]
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    doc_block = "\n---\n".join(parts)

    user_msg = (
        "Several code snippets from the same cluster are shown below. "
        "What single technical concept or theme do they all share? "
        "Answer with just one or two words.\n\n"
        f"{doc_block}"
    )
    assistant_prefill = "The shared concept is"

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_prefill}],
                tokenize=False,
                continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass  # fall through to raw
    # Base model fallback (or chat template unavailable)
    return f"{user_msg}\n\n{assistant_prefill}"


def readout_cluster(lens, lens_model, tokenizer, prompt: str,
                    sample_layers: list[int], top_k: int = 10) -> dict:
    """J-Lens readout for one cluster's prompt.

    Returns per-layer top-k tokens at the last position. The caller picks which
    layer's readout to use as the cluster label (heuristic: the earliest layer
    whose top-1 token is a "content word", not punctuation/stopword).
    """
    lens_logits, model_logits, _ = lens.apply(
        lens_model, prompt,
        layers=sample_layers,
        positions=[-1],
        max_seq_len=1024,
    )
    import torch
    result = {"by_layer": {}, "model_final": []}
    # model final
    probs = torch.softmax(model_logits[0].float(), dim=-1)
    topk = probs.topk(top_k)
    result["model_final"] = [
        {"token": tokenizer.decode([int(i)]).strip(), "prob": float(p)}
        for i, p in zip(topk.indices.tolist(), topk.values.tolist())
    ]
    for layer in sample_layers:
        probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
        topk = probs.topk(top_k)
        result["by_layer"][str(layer)] = [
            {"token": tokenizer.decode([int(i)]).strip(), "prob": float(p)}
            for i, p in zip(topk.indices.tolist(), topk.values.tolist())
        ]
    return result


def _is_content_token(t: str) -> bool:
    """A token is a content word if: alphabetic, len ≥4, not a stopword, not a
    bare code keyword, and not a likely BPE fragment.

    BPE fragment heuristic: a real English content word rarely has weird
    capitalization patterns like 'NCY', 'iare', 'tsy'. We accept lowercase,
    Capitalized, or ALLCAPS (≥5 chars) but reject mixed-case short tokens that
    look like subword fragments."""
    STOP = {"the", "and", "for", "that", "with", "from", "this", "are", "was",
            "were", "been", "have", "has", "will", "would", "could", "should",
            "not", "but", "into", "function", "def", "return", "class", "import",
            "const", "let", "var", "void", "null", "true", "false", "await",
            "async", "new", "typeof", "instanceof", "else", "while",
            "console", "expect", "string", "number", "object", "array",
            "also", "they", "them", "than", "then", "when", "what", "each",
            "more", "most", "some", "such", "only", "very", "just", "like"}
    if len(t) < 4 or not t.isalpha() or t.lower() in STOP:
        return False
    # Reject mixed-case fragments (e.g. 'NCY', 'iare', 'tsy', 'ugin')
    # Accept: lowercase, Titlecase (first cap), ALLCAPS ≥5
    if t.islower():
        return True
    if t[0].isupper() and t[1:].islower():  # Titlecase
        return True
    if t.isupper() and len(t) >= 5:  # ALLCAPS acronym
        return True
    return False  # mixed-case = likely BPE fragment


def pick_best_layer_readout(by_layer: dict, sample_layers: list[int]) -> list[str]:
    """Pick the LATE layer with the most content tokens in its top-5.

    Stage 1 showed that on Qwen3-1.7B (28 layers), concepts only crystallize at
    layers ≥20; layers 0-13 produce BPE fragments (punctuation, CJK quotes).
    So we scan from LATE→early and pick the first layer with ≥2 content tokens.
    If no layer has ≥2 content tokens, return the best available (deduplicated).
    """
    best_content: list[str] = []
    best_n = -1
    for layer in reversed(sample_layers):  # late → early
        toks = [t["token"] for t in by_layer[str(layer)][:5]]
        content = [t for t in toks if _is_content_token(t)]
        # deduplicate case-insensitively, preserve order
        seen = set()
        content = [t for t in content if not (t.lower() in seen or seen.add(t.lower()))]
        if len(content) > best_n:
            best_n = len(content)
            best_content = content
        if best_n >= 2:
            return best_content[:5]
    return best_content[:5] if best_content else [t["token"] for t in by_layer[str(sample_layers[-1])][:5]]


def run_stage2(embed, lens, lens_model, tokenizer, llm, repo_path: str,
               cand_name: str = "?", max_files: int = 50, n_clusters: int = 12):
    print("Phase 10 Stage 2: J-Lens static readout quality")
    print(f"{'='*70}")

    # 1. Load corpus + embed + cluster (same as Phase 9 for comparability)
    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")
    print(f"\n  {len(fine_texts)} FineRecords from {len(files)} files")

    print(f"  embedding (bge-m3, for clustering)...", end="", flush=True)
    fine_vecs = np.asarray(embed.embed(fine_texts), dtype=np.float32)
    print(f" done")

    cluster_members = partition_hdbscan(fine_vecs.tolist())
    top_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(cluster_members)} clusters; using top {len(top_clusters)}")

    # 2. Sample layers spread across the network
    sample_layers = sorted({
        lens.source_layers[0],
        lens.source_layers[len(lens.source_layers) // 4],
        lens.source_layers[len(lens.source_layers) // 2],
        lens.source_layers[3 * len(lens.source_layers) // 4],
        lens.source_layers[-1],
    })
    print(f"  sampling layers: {sample_layers}")

    # 3. Per-cluster readout + judge
    results = {}
    judge_scores = []
    for ci, (cid, members) in enumerate(top_clusters):
        member_texts = [fine_texts[m] for m in members]
        prompt = build_cluster_prompt(members, member_texts, tokenizer=tokenizer)
        print(f"\n  Cluster {cid} ({len(members)} members):")
        print(f"    sample: {member_texts[0][:80]!r}...")

        readout = readout_cluster(lens, lens_model, tokenizer, prompt, sample_layers)
        concept_tokens = pick_best_layer_readout(readout["by_layer"], sample_layers)
        print(f"    concept tokens: {concept_tokens}")
        print(f"    model final top-3: {[t['token'] for t in readout['model_final'][:3]]}")

        # LLM judge
        judged = judge_readout(concept_tokens, member_texts[:3], llm)
        print(f"    judge: accurate={judged.get('accurate')} conf={judged.get('confidence')}")
        judge_scores.append(1.0 if judged.get("accurate") else 0.0)

        results[str(cid)] = {
            "n_members": len(members),
            "concept_tokens": concept_tokens,
            "model_final_top3": [t["token"] for t in readout["model_final"][:3]],
            "lens_by_layer": readout["by_layer"],
            "sample_docs": [t[:120] for t in member_texts[:3]],
            "judge": judged,
        }

    # 4. Summary + gate
    accuracy = float(np.mean(judge_scores)) if judge_scores else 0.0
    print(f"\n{'='*70}")
    print(f"  J-Lens static readout LLM-judge accuracy: {accuracy:.1%} ({len(judge_scores)} clusters)")

    # Load Phase 9/9b for comparison (both encoder-MLM-head readouts, both
    # failed at 0% LLM-judge accuracy). J-Lens just needs to beat 0%.
    phase9_path = EXP / "phase9_xlmr_vec2vec.json"
    phase9b_path = EXP / "phase9b_per_position.json"
    comparison = {}
    if phase9_path.exists():
        p9 = json.loads(phase9_path.read_text())
        acc = p9.get("accuracy", {})
        # take the best of the three pooling methods
        vals = [v for v in acc.values() if isinstance(v, (int, float))]
        if vals:
            comparison["phase9_xlmr_best"] = max(vals)
    if phase9b_path.exists():
        p9b = json.loads(phase9b_path.read_text())
        vals = [s["accuracy"] for s in p9b.get("strategies", {}).values()
                if isinstance(s, dict) and isinstance(s.get("accuracy"), (int, float))]
        if vals:
            comparison["phase9b_perpos_best"] = max(vals)

    print(f"\n  Comparison (LLM-judge accuracy):")
    print(f"    J-Lens (this):      {accuracy:.1%}")
    for label, acc in comparison.items():
        if acc is not None:
            delta = accuracy - acc
            print(f"    {label:20} {acc:.1%}  (Δ {delta:+.1%})")

    # Gate decision
    gate_margin = 0.15
    best_phase9 = max((a for a in comparison.values() if a is not None), default=0.0)
    gate_pass = accuracy >= best_phase9 + gate_margin
    print(f"\n  GATE (J-Lens > best Phase9 + {gate_margin:.0%}): "
          f"{'PASS' if gate_pass else 'FAIL'} → {'Stage 3 needed' if not gate_pass else 'static readout sufficient'}")

    out = {
        "method": "jlens_static_readout",
        "model": cand_name,
        "n_clusters": len(results),
        "summary": {"judge_accuracy": accuracy},
        "comparison_phase9": comparison,
        "gate_pass": gate_pass,
        "clusters": results,
    }
    out_path = EXP / "phase10_stage2_jlens_readout.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 2: J-Lens readout quality")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-files", type=int, default=50)
    ap.add_argument("--n-clusters", type=int, default=12)
    ap.add_argument("--model-id", default=None,
                    help="force a specific model path/id (skip auto-detect)")
    ap.add_argument("--no-4bit", action="store_true",
                    help="force bf16 (for ≤2B models)")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/4] Loading lens + resolving model...")
    if args.model_id:
        # Forced model: find matching candidate by path prefix, or build ad-hoc
        cand = None
        for c in __import__("experiments.phase10_jlens_stage1", fromlist=["CANDIDATES"]).CANDIDATES:
            if c["local_model_dir"] in args.model_id or c["model_id"] in args.model_id:
                cand = c
                break
        if cand is None:
            cand = {"name": "custom", "local_model_dir": args.model_id,
                    "local_lens_path": "", "model_id": args.model_id, "needs_4bit": not args.no_4bit}
        use_4bit = cand["needs_4bit"] and not args.no_4bit
    else:
        cand = detect_model()
        use_4bit = cand["needs_4bit"]
    lens = load_lens(cand["local_lens_path"] or None)
    print(repr(lens))

    print(f"\n[2/4] Loading model ({cand['name']}, {'4-bit' if use_4bit else 'bf16'})...")
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=use_4bit)

    print(f"\n[3/4] Wrapping with jlens.from_hf...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    print(f"\n[4/4] Running Stage 2 experiment...")
    embed = CachedBgeM3Provider()
    llm = DeepSeekProvider()
    run_stage2(embed, lens, lens_model, tokenizer, llm, args.repo,
               cand["name"], args.max_files, args.n_clusters)


if __name__ == "__main__":
    main()
