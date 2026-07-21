"""Phase 10 Stage 5 — J-Lens 在自然语言域的稳定性验证（NFCorpus）。

Stage 4 在代码域发现 lens artifact（`epam`/`围绕`/`一致好评` 跨簇重复），
推测由 tree-sitter 摘要的结构同质性触发。本阶段在 NFCorpus（医学摘要，
自然语言散文）上验证：自然语言的句子多样性是否避开 lens artifact？

如果 Stage 5 J-Lens 读出干净（无 artifact）→ 确认 lens artifact 是代码域
特有问题，J-Lens 在自然语言域可用。
如果 Stage 5 仍有 artifact → lens 本身有稳定性问题，与输入模态无关。

流程（同 Stage 3/4，但自然语言域 + 无 tree-sitter）：
  1. NFCorpus 文档 → bge-m3 嵌入 → HDBSCAN 簇
  2. 每簇：文档拼成 prompt（chat template + assistant prefill）→ 7B-it J-Lens
  3. LLM judge + 人工审计
  4. 对比 Stage 4 代码域的 lens artifact 频率

关键诊断指标：
  - artifact 频率：`epam`/`围绕`/`一致好评`/`NCY` 等跨簇重复 token 出现率
  - 概念准确率：L26 top-5 中真实医学概念（disease/treatment/gene）的比例

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase10_jlens_stage5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.embed_cache import CachedBgeM3Provider
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage2 import (
    readout_cluster, pick_best_layer_readout, judge_readout,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

# Known lens artifacts from Stage 4 (code domain) — check if they recur
_KNOWN_ARTIFACTS = {"epam", "围绕", "一致好评", "条评论", "NCY", "agina",
                    "iare", "inox", "odzi", "zych", "alink"}


def run_stage5(embed, lens, lens_model, tokenizer, llm, max_docs: int = 200,
               n_clusters: int = 10):
    print("Phase 10 Stage 5: J-Lens on natural language (NFCorpus)")
    print(f"{'='*70}")

    # 1. Load NFCorpus
    records = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    # records: list of (id, text). Use the text.
    doc_ids = [r[0] for r in records]
    doc_texts = [r[1] for r in records]
    print(f"\n  {len(doc_texts)} NFCorpus documents (medical abstracts)")

    # 2. Embed + cluster
    print(f"  embedding (bge-m3)...", end="", flush=True)
    vecs = np.asarray(embed.embed(doc_texts), dtype=np.float32)
    print(f" done")

    cluster_members = partition_hdbscan(vecs.tolist())
    valid_clusters = {cid: members for cid, members in cluster_members.items()
                      if len(members) >= 3}
    top_clusters = sorted(valid_clusters.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"  {len(valid_clusters)} clusters (≥3 members); using top {len(top_clusters)}")

    # 3. Sample layers
    sample_layers = sorted({
        lens.source_layers[0],
        lens.source_layers[len(lens.source_layers) // 3],
        lens.source_layers[2 * len(lens.source_layers) // 3],
        lens.source_layers[-1],
    })

    # 4. Per-cluster readout + judge
    results = {}
    judge_scores = []
    all_l26_tokens = []  # for artifact frequency analysis

    for ci, (cid, members) in enumerate(top_clusters):
        member_texts = [doc_texts[m] for m in members]
        prompt = _build_nl_cluster_prompt(member_texts, tokenizer)

        readout = readout_cluster(lens, lens_model, tokenizer, prompt, sample_layers)
        concept_tokens = pick_best_layer_readout(readout["by_layer"], sample_layers)

        # Collect L26 (last source layer) tokens for artifact analysis
        last_layer = str(sample_layers[-1])
        l26_toks = [t["token"] for t in readout["by_layer"][last_layer][:5]]
        all_l26_tokens.extend(l26_toks)

        judged = judge_readout(concept_tokens, member_texts[:3], llm)
        judge_scores.append(1.0 if judged.get("accurate") else 0.0)

        print(f"\n  Cluster {cid} ({len(members)} docs):")
        print(f"    sample: {member_texts[0][:90]!r}...")
        print(f"    concept tokens: {concept_tokens}")
        print(f"    L{last_layer} raw: {l26_toks}")
        print(f"    judge: accurate={judged.get('accurate')} conf={judged.get('confidence')}")

        results[str(cid)] = {
            "n_docs": len(members),
            "concept_tokens": concept_tokens,
            "l26_raw": l26_toks,
            "model_final_top3": [t["token"] for t in readout["model_final"][:3]],
            "lens_by_layer": readout["by_layer"],
            "sample_docs": [t[:120] for t in member_texts[:3]],
            "judge": judged,
        }

    # 5. Artifact analysis
    accuracy = float(np.mean(judge_scores)) if judge_scores else 0.0
    token_freq = Counter(all_l26_tokens)
    artifact_hits = sum(token_freq.get(a, 0) for a in _KNOWN_ARTIFACTS)
    artifact_rate = artifact_hits / len(all_l26_tokens) if all_l26_tokens else 0.0

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  LLM-judge accuracy: {accuracy:.1%} ({len(judge_scores)} clusters)")
    print(f"  Artifact rate: {artifact_rate:.1%} ({artifact_hits}/{len(all_l26_tokens)} L26 tokens)")
    print(f"  Most common L26 tokens: {token_freq.most_common(10)}")
    print(f"\n  Known artifacts found:")
    for a in _KNOWN_ARTIFACTS:
        if a in token_freq:
            print(f"    {a!r}: {token_freq[a]} times")

    # Cross-domain comparison
    stage4_path = EXP / "phase10_stage4_treesitter.json"
    stage3_path = EXP / "phase10_stage2_jlens_readout.json"
    print(f"\n  Cross-domain comparison (LLM-judge accuracy):")
    print(f"    Stage 5 NFCorpus (NL):     {accuracy:.1%}")
    if stage4_path.exists():
        s4 = json.loads(stage4_path.read_text())
        s4_accs = [m["accuracy"] for m in s4.get("modes", {}).values()]
        if s4_accs:
            print(f"    Stage 4 tree-sitter (code): {max(s4_accs):.1%}")
    if stage3_path.exists():
        s3 = json.loads(stage3_path.read_text())
        print(f"    Stage 3 raw code:          {s3.get('summary',{}).get('judge_accuracy',0):.1%}")

    cand = detect_model()
    out = {
        "method": "jlens_natural_language_nfcorpus",
        "model": cand["name"],
        "n_docs": len(doc_texts),
        "n_clusters": len(results),
        "summary": {"judge_accuracy": accuracy, "artifact_rate": artifact_rate},
        "token_frequency": dict(token_freq.most_common(20)),
        "clusters": results,
    }
    out_path = EXP / "phase10_stage5_nfcorpus.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def _build_nl_cluster_prompt(member_texts: list[str], tokenizer) -> str:
    """Build a concern-coupled prompt from cluster's natural language docs.

    Same chat-template + assistant-prefill pattern as Stage 2-4. The docs are
    medical abstracts (natural language sentences), so no tree-sitter needed.
    """
    parts = []
    total = 0
    for text in member_texts:
        chunk = text[:400]
        if total + len(chunk) > 1400:
            remaining = 1400 - total
            if remaining > 50:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    doc_block = "\n---\n".join(parts)

    user_msg = (
        "Several document abstracts from the same cluster are shown below. "
        "What single medical or scientific concept or topic do they all share? "
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
    ap = argparse.ArgumentParser(description="Phase 10 Stage 5: J-Lens on NFCorpus")
    ap.add_argument("--max-docs", type=int, default=200)
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
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()

    print(f"\n[4/4] Running Stage 5 (NFCorpus natural language)...")
    run_stage5(embed, lens, lens_model, tokenizer, llm,
               args.max_docs, args.n_clusters)


if __name__ == "__main__":
    main()
