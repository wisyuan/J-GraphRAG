"""Phase 20b: 质心算法跨域验证。

Phase 20 在 NFCorpus（医学营养）上达到 100% 覆盖率。但 NFCorpus 的特点是
概念词密集、文档结构化强。需要在其他域上验证：
  - novel（小说）：叙事性，动词多（Phase 16a verb=50%）
  - medical（GraphRAG-Bench 医学QA）：医学术语密集
  - scifact（科学论文摘要）：学术写作，名词为主

核心问题：Phase 20 的 concern+全层COM+三重过滤在非 NFCorpus 域上：
  1. 覆盖率是否还能保持 100%？
  2. 高质量层级比例如何变化？
  3. 不同域的 meta→sub 层级质量特征？

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase20b_cross_domain
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
from experiments.corpus_loader import load_beir_fine_records
from experiments.partition import partition_hdbscan
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals, build_concern_prompt
from experiments.phase17_multihop_depth_gradient import extract_depth_gradient
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
    classify_by_com_gap, build_corpus_word_set,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── Corpus loaders ────────────────────────────────────────────────────

def load_graphrag_corpus(name: str, max_docs: int = 200) -> list[str]:
    """Load GraphRAG-Bench corpus (novel or medical), split into chunks."""
    path = Path(f"/tmp/graphrag-bench/Datasets/Corpus/{name}.json")
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    full_text = data[0]["context"] if isinstance(data, list) else data["context"]
    # Split into ~800-char chunks at sentence boundaries
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > 800 and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks[:max_docs]


def load_scifact_docs(max_docs: int = 200) -> list[str]:
    """Load SciFact from BEIR."""
    records = load_beir_fine_records("scifact", max_docs=max_docs)
    return [r[1] for r in records]


# ── Per-domain run ────────────────────────────────────────────────────

def run_domain(
    domain_name: str,
    doc_texts: list[str],
    lens, lens_model, tokenizer,
    n_clusters: int = 8,
) -> dict:
    """Run Phase 20 algorithm on one domain. Returns summary stats."""
    n = len(doc_texts)
    if n < 20:
        print(f"\n  [{domain_name}] skipped (only {n} docs)")
        return {"domain": domain_name, "n_docs": n, "skipped": True}

    layer = lens.source_layers[-1]
    all_layers = lens.source_layers

    print(f"\n  [{domain_name}] {n} docs")

    # L0 clustering
    print(f"    residuals...", flush=True)
    prompts_resid = [build_concern_prompt(t, tokenizer) for t in doc_texts]
    jlens_vecs = extract_residuals(lens, lens_model, tokenizer,
                                    prompts_resid, layer)
    l0_clusters = partition_hdbscan(jlens_vecs.tolist())
    l0_valid = {cid: members for cid, members in l0_clusters.items()
                if len(members) >= 5}
    top_clusters = sorted(l0_valid.items(),
                          key=lambda x: len(x[1]), reverse=True)[:n_clusters]
    print(f"    {len(l0_valid)} clusters, analyzing top {len(top_clusters)}")

    if not top_clusters:
        return {"domain": domain_name, "n_docs": n, "n_clusters": 0,
                "n_high_quality": 0, "n_any_hierarchy": 0,
                "hierarchies": []}

    # Per-cluster hierarchy
    hierarchies = []
    n_high_quality = 0
    n_any_hierarchy = 0

    for cid, members in top_clusters:
        docs = [doc_texts[m] for m in members]

        concern_prompt = build_concern_prompt_full(docs, tokenizer)
        gradient = extract_depth_gradient(
            lens, lens_model, tokenizer, concern_prompt,
            layers=all_layers, n_words=8, max_seq_len=512)

        corpus_words = build_corpus_word_set(docs)
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=3,
            require_corpus=True, require_noun=True)
        profiles = classify_by_com_gap(profiles)

        # Relax if too strict
        if len(profiles) < 2:
            profiles = compute_concept_profiles_filtered(
                gradient, corpus_words, min_layers=3,
                require_corpus=False, require_noun=True)
            profiles = classify_by_com_gap(profiles)

        meta = [p.word for p in profiles if p.role == "meta"]
        sub = [p.word for p in profiles if p.role == "sub"]
        n_verified = sum(1 for p in profiles if p.in_corpus)

        hq = len(meta) >= 2 and len(sub) >= 1 and n_verified >= 3
        any_h = len(meta) >= 1 and len(sub) >= 1
        if hq:
            n_high_quality += 1
        if any_h:
            n_any_hierarchy += 1

        hierarchies.append({
            "cluster_id": cid,
            "n_docs": len(members),
            "meta": meta[:5],
            "sub": sub[:5],
            "high_quality": hq,
            "n_verified": n_verified,
            "sample": docs[0][:120],
        })

        tag = "✓HQ" if hq else ("~" if any_h else "✗")
        print(f"    C{cid} ({len(members)}d) {tag}: "
              f"meta={meta[:3]} → sub={sub[:3]}")

    n_analyzed = len(top_clusters)
    result = {
        "domain": domain_name,
        "n_docs": n,
        "n_clusters": n_analyzed,
        "n_high_quality": n_high_quality,
        "n_any_hierarchy": n_any_hierarchy,
        "high_quality_rate": round(n_high_quality / max(1, n_analyzed), 2),
        "any_hierarchy_rate": round(n_any_hierarchy / max(1, n_analyzed), 2),
        "hierarchies": hierarchies,
    }
    print(f"    → HQ {n_high_quality}/{n_analyzed} ({result['high_quality_rate']:.0%}), "
          f"any {n_any_hierarchy}/{n_analyzed} ({result['any_hierarchy_rate']:.0%})")
    return result


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 70)
    print("Phase 20b: Cross-domain centroid algorithm validation")
    print("=" * 70)

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    # Load all domains
    print(f"\n[2/2] Loading corpora + running per-domain...")
    max_docs = 200
    domains = {}

    # NFCorpus (medical nutrition)
    nf = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    domains["nfcorpus"] = [r[1] for r in nf]
    print(f"  nfcorpus: {len(domains['nfcorpus'])} docs")

    # SciFact (scientific papers)
    sf = load_scifact_docs(max_docs=max_docs)
    if sf:
        domains["scifact"] = sf
        print(f"  scifact: {len(sf)} docs")

    # Novel (narrative)
    novels = load_graphrag_corpus("novel", max_docs=max_docs)
    if novels:
        domains["novel"] = novels
        print(f"  novel: {len(novels)} docs")

    # Medical (GraphRAG-Bench)
    medical = load_graphrag_corpus("medical", max_docs=max_docs)
    if medical:
        domains["medical_qa"] = medical
        print(f"  medical_qa: {len(medical)} docs")

    # Run each domain
    all_results = {}
    for domain_name, docs in domains.items():
        result = run_domain(domain_name, docs, lens, lens_model, tokenizer)
        all_results[domain_name] = result

    # Cross-domain comparison
    print(f"\n{'='*70}")
    print("CROSS-DOMAIN COMPARISON")
    print(f"{'='*70}")
    print(f"  {'domain':<15} {'n_docs':>6} {'n_cls':>5} {'HQ':>5} {'any':>5} "
          f"{'HQ%':>5} {'any%':>5}")
    print(f"  {'-'*52}")
    for domain, r in all_results.items():
        if r.get("skipped"):
            print(f"  {domain:<15}  SKIPPED")
            continue
        print(f"  {domain:<15} {r['n_docs']:>6} {r['n_clusters']:>5} "
              f"{r['n_high_quality']:>5} {r['n_any_hierarchy']:>5} "
              f"{r['high_quality_rate']:>4.0%} {r['any_hierarchy_rate']:>4.0%}")

    # Best hierarchies per domain
    print(f"\n  Best hierarchies per domain:")
    for domain, r in all_results.items():
        if r.get("skipped"):
            continue
        hqs = [h for h in r.get("hierarchies", []) if h["high_quality"]]
        if hqs:
            for h in hqs[:2]:
                print(f"    [{domain}] C{h['cluster_id']} ({h['n_docs']}d): "
                      f"meta={h['meta'][:3]} → sub={h['sub'][:3]}")
        else:
            # Show best available
            anys = [h for h in r.get("hierarchies", [])
                    if h["meta"] and h["sub"]]
            if anys:
                h = anys[0]
                print(f"    [{domain}] C{h['cluster_id']} ({h['n_docs']}d): "
                      f"meta={h['meta'][:3]} → sub={h['sub'][:3]}")

    # Save
    out = {
        "method": "cross_domain_centroid_validation",
        "model": cand["name"],
        "max_docs_per_domain": max_docs,
        "domains": all_results,
    }
    out_path = EXP / "phase20b_cross_domain.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
