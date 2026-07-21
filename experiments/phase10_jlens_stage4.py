"""Phase 10 Stage 4 — tree-sitter 结构摘要 + J-Lens 读出（A/B/C 注释模式对比）。

Stage 3 在 raw code 上 J-Lens 读出 70% 真实概念但 BPE 碎片噪声 ~30%。
本阶段验证：tree-sitter 结构摘要（准自然语言）能否消除碎片噪声，提升
概念读出质量。并测试注释提取（A 无注释 / B docstring / C 全注释）的增益。

流程：
  1. pi-code .ts 文件 → tree-sitter 结构摘要（3 种模式）
  2. bge-m3 嵌入摘要 → HDBSCAN 簇（文件粒度，不是 FineRecord 粒度）
  3. 每簇：摘要拼成 prompt → 7B-it J-Lens 读出 → top-k 概念词
  4. LLM judge + 人工审计引导
  5. 对比：Stage 3 raw code vs Stage 4 tree-sitter (A/B/C)

关键假设：
  - tree-sitter 摘要的符号名（AuthService, validateToken）是程序员显式写的
    概念词，LLM 读到它们时残差流自然激活概念 → BPE 碎片应大幅减少
  - docstring（B）可能增益（更多概念上下文）或噪声（TODO/FIXME）
  - 让数据决定，不假设

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage4 --repo /tmp/pi-repo
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
from experiments.partition import partition_hdbscan
from experiments.treesitter_summary import build_structural_summary, detect_language
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage2 import (
    readout_cluster, pick_best_layer_readout, judge_readout,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

COMMENT_MODES = ("none", "docstring", "all")


def collect_source_files(repo_path: str, max_files: int = 80) -> list[Path]:
    """Collect source files tree-sitter can parse (TS/JS/Py/Rust/Go)."""
    repo = Path(repo_path)
    files = []
    for p in sorted(repo.rglob("*")):
        if "node_modules" in str(p) or ".git" in str(p):
            continue
        if p.is_file() and detect_language(p) is not None:
            files.append(p)
            if len(files) >= max_files:
                break
    return files


def run_stage4(embed, lens, lens_model, tokenizer, llm, repo_path: str,
               max_files: int = 80, n_clusters: int = 10):
    print("Phase 10 Stage 4: tree-sitter structural summary + J-Lens readout")
    print(f"{'='*70}")

    # 1. Collect source files
    files = collect_source_files(repo_path, max_files)
    print(f"\n  {len(files)} source files (tree-sitter parseable)")

    # 2. For each comment mode: build summaries, embed, cluster, readout
    all_results = {}
    cand = detect_model()

    for mode in COMMENT_MODES:
        print(f"\n{'='*70}")
        print(f"COMMENT MODE: {mode}")
        print(f"{'='*70}")

        # Build structural summaries
        summaries = []
        valid_files = []
        for f in files:
            try:
                summary = build_structural_summary(f, max_symbols=30, comments=mode)
                if summary and len(summary) > 20:  # skip trivially small
                    summaries.append(summary)
                    valid_files.append(f)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
        print(f"  {len(summaries)} valid summaries (mode={mode})")

        if len(summaries) < n_clusters:
            print(f"  WARNING: only {len(summaries)} summaries, need ≥{n_clusters}")
            continue

        # Embed + cluster (on the structural summary text, not raw code)
        print(f"  embedding summaries (bge-m3)...", end="", flush=True)
        vecs = np.asarray(embed.embed(summaries), dtype=np.float32)
        print(f" done")

        cluster_members = partition_hdbscan(vecs.tolist())
        # filter to clusters with ≥3 members
        valid_clusters = {cid: members for cid, members in cluster_members.items()
                          if len(members) >= 3}
        top_clusters = sorted(valid_clusters.items(),
                              key=lambda x: len(x[1]), reverse=True)[:n_clusters]
        print(f"  {len(valid_clusters)} clusters (≥3 members); using top {len(top_clusters)}")

        # Per-cluster J-Lens readout + judge
        mode_results = {}
        judge_scores = []
        for ci, (cid, members) in enumerate(top_clusters):
            member_summaries = [summaries[m] for m in members]
            # Build the readout prompt: combine cluster's summaries + concern
            # (reuse Stage 2's concern-coupling via chat template)
            prompt = _build_summary_cluster_prompt(member_summaries, tokenizer)

            # Sample layers spread across the network
            sample_layers = sorted({
                lens.source_layers[0],
                lens.source_layers[len(lens.source_layers) // 3],
                lens.source_layers[2 * len(lens.source_layers) // 3],
                lens.source_layers[-1],
            })

            readout = readout_cluster(lens, lens_model, tokenizer, prompt,
                                      sample_layers)
            concept_tokens = pick_best_layer_readout(readout["by_layer"], sample_layers)

            # Judge
            # For judge context, show the actual symbol names (compact)
            judge_context = [s[:300] for s in member_summaries[:3]]
            judged = judge_readout(concept_tokens, judge_context, llm)
            judge_scores.append(1.0 if judged.get("accurate") else 0.0)

            print(f"\n  Cluster {cid} ({len(members)} files, mode={mode}):")
            print(f"    sample file: {valid_files[members[0]].name}")
            print(f"    concept tokens: {concept_tokens}")
            print(f"    judge: accurate={judged.get('accurate')} conf={judged.get('confidence')}")

            mode_results[str(cid)] = {
                "n_files": len(members),
                "sample_file": valid_files[members[0]].name,
                "concept_tokens": concept_tokens,
                "model_final_top3": [t["token"] for t in readout["model_final"][:3]],
                "lens_by_layer": readout["by_layer"],
                "judge": judged,
            }

        accuracy = float(np.mean(judge_scores)) if judge_scores else 0.0
        print(f"\n  MODE={mode} LLM-judge accuracy: {accuracy:.1%} ({len(judge_scores)} clusters)")

        all_results[mode] = {
            "n_summaries": len(summaries),
            "n_clusters": len(mode_results),
            "accuracy": accuracy,
            "clusters": mode_results,
        }

    # 3. Cross-mode comparison
    print(f"\n{'='*70}")
    print(f"CROSS-MODE COMPARISON (LLM-judge accuracy)")
    print(f"{'='*70}")
    for mode in COMMENT_MODES:
        if mode in all_results:
            acc = all_results[mode]["accuracy"]
            print(f"  {mode:12}: {acc:.1%}")

    # Load Stage 3 raw-code baseline for comparison
    stage3_path = EXP / "phase10_stage2_jlens_readout.json"
    if stage3_path.exists():
        s3 = json.loads(stage3_path.read_text())
        s3_acc = s3.get("summary", {}).get("judge_accuracy", 0)
        print(f"  {'raw_code (Stage 3)':12}: {s3_acc:.1%}  ← baseline")

    out = {
        "method": "treesitter_summary_jlens",
        "model": cand["name"],
        "repo": str(repo_path),
        "modes": all_results,
    }
    out_path = EXP / "phase10_stage4_treesitter.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def _build_summary_cluster_prompt(member_summaries: list[str], tokenizer) -> str:
    """Build a concern-coupled prompt from cluster's structural summaries.

    Uses the chat template (same as Stage 2/3) with an assistant prefill that
    forces concept formation at the readout position.
    """
    # Cap total length
    parts = []
    total = 0
    for s in member_summaries:
        chunk = s[:500]
        if total + len(chunk) > 1500:
            remaining = 1500 - total
            if remaining > 50:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    doc_block = "\n---\n".join(parts)

    user_msg = (
        "Several code structure summaries from the same cluster are shown below. "
        "Each summary lists the symbols (classes, functions, interfaces) in a source "
        "file. What single technical concept or theme do these files share? "
        "Answer with just one or two words.\n\n"
        f"{doc_block}"
    )
    assistant_prefill = "The shared concept is"

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False,
            )
        except Exception:
            pass
    return f"{user_msg}\n\n{assistant_prefill}"


def main():
    ap = argparse.ArgumentParser(description="Phase 10 Stage 4: tree-sitter + J-Lens")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-files", type=int, default=80)
    ap.add_argument("--n-clusters", type=int, default=10)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/4] Loading lens + model...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    print(repr(lens))
    model_src = cand["local_model_dir"] if _model_dir_complete(cand["local_model_dir"]) else cand["model_id"]
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/4] Wrapping with jlens.from_hf...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    print(repr(lens_model))

    print(f"\n[3/4] Loading embed + LLM...")
    embed = CachedBgeM3Provider()
    llm = __import__("jgraphrag.llm", fromlist=["DeepSeekProvider"]).DeepSeekProvider()

    print(f"\n[4/4] Running Stage 4 (tree-sitter × {len(COMMENT_MODES)} comment modes)...")
    run_stage4(embed, lens, lens_model, tokenizer, llm, args.repo,
               args.max_files, args.n_clusters)


if __name__ == "__main__":
    main()
