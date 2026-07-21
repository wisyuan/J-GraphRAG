"""Phase 35 PoC: Prefill position 扫描——反转 prompt 结构。

之前：[文档] + [问题 prefill "Concepts are"] → 读 position -1
现在：[问题] + [文档概念 prefill "Concepts: cancer, tumor, surgery..."]
      → 扫描 prefill 中每个概念词所在 position 的 workspace

两步流程：
  Step 1: 标准 concern prompt → position -1 → top-25 候选概念
  Step 2: 构造反转 prompt：
    User: "List the concepts in this document."
    Assistant: "The concepts are: cancer, tumor, surgery, blood, ..."
                                        ↑      ↑      ↑
                                        pos 0  pos 1  pos 2
    → lens.apply(positions=[0,1,2,...]) 读每个概念 position 的 workspace

每个 position 的 workspace 反映模型对该概念的内部表示。
对比 position -1 的单次读出，看是否能提取更多/更精确的概念。

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase35_prefill_position_scan
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
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20_concern_full_com import build_concern_prompt_full
from experiments.phase20_concern_full_com import compute_concept_profiles_filtered
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"

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


def run_poc(lens, lens_model, tokenizer):
    print("Phase 35 PoC: Prefill position scan")
    print(f"{'='*70}")

    # Try GraphRAG-Bench first, fallback to NFCorpus
    try:
        corpus, _ = load_graphrag_bench("medical", max_queries=1)
        chunks = list(corpus.values())
        source = "GraphRAG-Bench medical"
    except Exception:
        from experiments.corpus_loader import load_beir_fine_records
        records = load_beir_fine_records("nfcorpus", max_docs=200)
        chunks = [r[1] for r in records]
        source = "NFCorpus"
    print(f"  Source: {source} ({len(chunks)} chunks)")
    all_layers = lens.source_layers
    layer = all_layers[-1]

    # Test on 5 chunks
    test_chunks = [chunks[i] for i in [0, min(50,len(chunks)-1), min(100,len(chunks)-1), min(150,len(chunks)-1), min(180,len(chunks)-1)]]

    for ci, chunk in enumerate(test_chunks):
        print(f"\n{'='*60}")
        print(f"Chunk {ci}: {chunk[:80]}...")
        print(f"{'='*60}")

        # Step 1: Standard position -1 readout (get candidate concepts)
        import re
        corpus_words = set()
        for c in chunks[:200]:
            for m in re.finditer(r'[a-zA-Z]{4,}', c):
                corpus_words.add(m.group().lower())

        prompt_v1 = build_concern_prompt_full([chunk], tokenizer)
        lens_logits_v1, _, _ = lens.apply(
            lens_model, prompt_v1,
            layers=all_layers, positions=[-1], max_seq_len=512)

        # Get top-25 from all layers at position -1
        all_candidates = set()
        for l in all_layers:
            words = decode_topk_custom(lens_logits_v1[l][0], tokenizer, n=25, scan=60)
            for w in words:
                all_candidates.add(w["token"])

        # Filter: corpus-verified candidates
        verified = sorted([c for c in all_candidates if c.lower() in corpus_words])[:10]

        print(f"\n  Step 1 (position -1, top-25/layer):")
        print(f"    Raw candidates: {len(all_candidates)}")
        print(f"    Corpus-verified: {verified}")

        if len(verified) < 2:
            print(f"    Too few verified — skipping prefill scan")
            continue

        # Step 2: Build reversed prompt with verified concepts as prefill
        # Format: User asks for concepts, Assistant lists them
        concept_str = ", ".join(verified[:8])
        user_msg = f"What concepts does this text discuss?\n\n{chunk[:400]}"
        prefill = f"The concepts are: {concept_str}"

        if hasattr(tokenizer, "apply_chat_template"):
            prompt_v2 = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": prefill}],
                tokenize=False, continue_final_message=True,
                add_generation_prompt=False)
        else:
            prompt_v2 = f"{user_msg}\n{prefill}"

        # Tokenize to find positions of concept words in the prefill
        # We need to find where each concept token is in the full prompt
        full_ids = tokenizer.encode(prompt_v2, return_tensors="pt").to(
            lens_model._hf_model.device if hasattr(lens_model, '_hf_model') else "cuda")

        # Decode each token to find concept positions
        token_texts = []
        for i in range(full_ids.shape[1]):
            tok = tokenizer.decode([full_ids[0, i].item()])
            token_texts.append(tok)

        # Find positions of concept words (after "The concepts are:")
        prefill_start = None
        for i, t in enumerate(token_texts):
            if "concepts are" in "".join(token_texts[max(0,i-3):i+1]).lower():
                prefill_start = i + 1
                break

        # Find each concept's position
        concept_positions = {}
        if prefill_start:
            for i in range(prefill_start, len(token_texts)):
                tok_text = token_texts[i].strip().rstrip(",").lower()
                for c in verified[:8]:
                    if tok_text == c.lower() or c.lower() in tok_text:
                        if c not in concept_positions:
                            concept_positions[c] = i
                        break

        print(f"\n  Step 2 (prefill position scan):")
        print(f"    Prefill: \"{prefill[:80]}...\"")
        print(f"    Concept positions: {concept_positions}")

        if not concept_positions:
            print(f"    No concept positions found — trying alternative search")
            # Alternative: find by matching decoded text
            concept_list_str = concept_str.lower()
            for i in range(prefill_start or 0, len(token_texts)):
                tok = token_texts[i].strip().lower()
                for c in verified[:8]:
                    cl = c.lower()
                    if tok == cl or (len(cl) > 3 and cl.startswith(tok) and len(tok) >= 3):
                        concept_positions[c] = i
                        break

            print(f"    Alternative positions: {concept_positions}")

        # Step 3: Read workspace at each concept position
        positions_to_read = sorted(set(concept_positions.values()))
        if not positions_to_read:
            print(f"    No positions to read")
            continue

        # lens.apply returns {layer: [n_positions, vocab]} tensor
        # positions_to_read = [139, 141, 143] → tensor[3, vocab]
        lens_logits_v2, _, _ = lens.apply(
            lens_model, prompt_v2,
            layers=[layer],
            positions=positions_to_read,
            max_seq_len=1024,
        )

        # For each concept position, read workspace
        # lens_logits_v2[layer] is [n_positions, vocab]
        # Index 0 = first position in positions_to_read, etc.
        print(f"\n    Per-position workspace (layer {layer}):")
        pos_to_idx = {p: i for i, p in enumerate(positions_to_read)}
        for concept, pos in sorted(concept_positions.items(), key=lambda x: x[1]):
            idx = pos_to_idx.get(pos)
            if idx is not None and layer in lens_logits_v2:
                words = decode_topk_custom(
                    lens_logits_v2[layer][idx], tokenizer, n=8, scan=40)
                ws_str = ", ".join(f"{w['token']}({w['prob']:.2f})" for w in words[:5])
                print(f"      pos={pos:>3} [{concept:12}]: {ws_str}")

        # Step 4: Compare — what does prefill scan give us that position -1 doesn't?
        prefill_words = set()
        for pos in positions_to_read:
            idx = pos_to_idx.get(pos)
            if idx is not None and layer in lens_logits_v2:
                words = decode_topk_custom(
                    lens_logits_v2[layer][idx], tokenizer, n=10, scan=40)
                for w in words:
                    prefill_words.add(w["token"].lower())

        # Words only in prefill scan (not in position -1)
        v1_words = set()
        for l in all_layers:
            words = decode_topk_custom(lens_logits_v1[l][0], tokenizer, n=10, scan=40)
            for w in words:
                v1_words.add(w["token"].lower())

        novel = prefill_words - v1_words
        overlap = prefill_words & v1_words

        print(f"\n    Comparison:")
        print(f"      Position -1 unique words: {len(v1_words)}")
        print(f"      Prefill scan unique words: {len(prefill_words)}")
        print(f"      Novel (only in prefill): {len(novel)} → {sorted(novel)[:10]}")
        print(f"      Overlap: {len(overlap)} → {sorted(overlap)[:10]}")


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    print(f"\n[2/2] Running prefill position scan PoC...")
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    run_poc(lens, lens_model, tokenizer)


if __name__ == "__main__":
    main()
