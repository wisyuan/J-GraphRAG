"""Phase 16a (前置实验): 跨域概念词性分布分析。

用户洞察：之前的阈值设计基于医学域数据（noun_ratio 为主），但不同域的概念词
词性分布不同——小说域动词多（叙事性），代码域动名词混合。如果阈值不校准，
可能在小说域过严（什么都不展开）或过松（展开 garbage）。

本实验：在 NFCorpus（医学）、novel（小说）、code（pi-code）三域上跑统一的
J-Lens 概念提取，统计 POS 分布差异，验证阈值是否需要域自适应。

关键问题：
  1. 三域的概念词 POS 分布是否真的有显著差异？
  2. noun_ratio 阈值 0.4 在三域上分别意味着什么？
  3. 是否需要域自适应阈值，还是固定值即可？

方法论（诚实）：
  - 后缀启发式 POS 分类（无 NLTK 数据依赖），分 NN/NNS/VBG/VBD/JJ/UNK
  - 语料验证：概念词是否在原文档中出现（区分"UNK 但真实"如 Cancer 和"UNK 且碎片"如 alink）
  - 输出三域对比表 + 阈值敏感性分析

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase16a_cross_domain_pos
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.corpus_loader import load_beir_fine_records
from experiments.partition import partition_hdbscan
from experiments.concept_quality import (
    optimize_concepts, build_corpus_term_freq, build_corpus_vocab,
    _get_wordnet_nouns, complete_prefix,
)
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase10_jlens_stage6 import extract_residuals, build_concern_prompt

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)


# ── POS classification (suffix heuristic, no NLTK data needed) ────────

# Noun suffixes: strong signal that a word is a noun
NOUN_SUFFIXES = (
    'tion', 'ment', 'ness', 'ity', 'sion', 'osis', 'oma', 'ism', 'ist',
    'logy', 'pathy', 'emia', 'uria', 'graphy', 'plasia', 'rrhea',
    'cy', 'ce', 'cracy', 'hood', 'ship', 'dom',
)

# Adjective suffixes
ADJ_SUFFIXES = (
    'ful', 'less', 'ous', 'ive', 'able', 'ible', 'ical', 'ish',
    'like', 'most', 'ward',
)

# Adjective suffixes that need length check (short words like "al", "ic")
ADJ_SUFFIXES_SHORT = ('al', 'ic', 'id', 'an', 'ar', 'en', 'or')


def classify_concept_pos(word: str) -> str:
    """Suffix-heuristic POS classification.

    Returns one of:
      NN   — noun (strong suffix like -tion, -ment)
      NNS  — plural noun (-s)
      VBG  — verb gerund/participle (-ing)
      VBD  — verb past tense (-ed)
      JJ   — adjective (-ful, -ous, -ive, ...)
      UNK  — unknown (could be a root noun like "Cancer" or BPE fragment)

    The UNK category is the ambiguous one — resolved by corpus verification
    in `concept_quality_score`.
    """
    w = word.lower()
    if len(w) < 3:
        return 'SHORT'

    # Verb forms (check before noun, since -ing/-ed can overlap)
    if w.endswith('ing') and len(w) > 4:
        return 'VBG'
    if w.endswith('ed') and len(w) > 3 and not w.endswith('eed'):
        # "eed" words like "need", "seed", "feed" are usually not VBD
        # but "treated", "linked" are
        return 'VBD'

    # Strong noun suffixes
    for suffix in NOUN_SUFFIXES:
        if w.endswith(suffix):
            return 'NN'

    # Plural (but not ss, us, is, os — those are singular)
    if w.endswith('s') and not w.endswith(('ss', 'us', 'is', 'os', 'xs')):
        return 'NNS'

    # Adjective suffixes (long ones first)
    for suffix in ADJ_SUFFIXES:
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return 'JJ'
    for suffix in ADJ_SUFFIXES_SHORT:
        if w.endswith(suffix) and len(w) > 5:
            return 'JJ'

    return 'UNK'


def is_likely_bpe_fragment(word: str, corpus_words: set[str] | None = None) -> bool:
    """Heuristic: is this token a BPE fragment rather than a real word?

    A BPE fragment typically:
    - Is short (≤ 5 chars)
    - Does NOT appear in the corpus as a real word
    - Has unusual letter patterns

    If corpus_words is provided, we can definitively check.
    """
    w = word.lower()
    if len(w) <= 4:
        if corpus_words is not None:
            return w not in corpus_words
        return True
    if corpus_words is not None:
        # 5-6 char token not in corpus = suspicious
        if len(w) <= 6 and w not in corpus_words:
            return True
    return False


# ── Concept quality scoring ───────────────────────────────────────────

@dataclass
class ConceptQualityMetrics:
    """POS + corpus-verification metrics for a set of concept words."""
    n_concepts: int
    pos_counts: dict[str, int]  # {'NN': 3, 'VBG': 2, ...}
    noun_ratio: float           # (NN + NNS + corpus_verified_UNK) / total
    verb_ratio: float           # (VBG + VBD) / total
    adj_ratio: float            # JJ / total
    bpe_fragment_ratio: float   # fragments / total
    corpus_hit_ratio: float     # concepts found in corpus text / total
    effective_concepts: list[str]  # concepts that pass quality filter
    n_effective: int


def concept_quality_score(
    concepts: list[str],
    doc_texts: list[str] | None = None,
) -> ConceptQualityMetrics:
    """Score concept quality across POS + corpus-verification dimensions.

    This is the core signal for the expansion stopping criterion. The key
    insight: a cluster worth expanding has concepts that are:
    1. Mostly nouns (domain entities), not verbs/adjectives (narrative/generic)
    2. Verifiable in the corpus (real words, not BPE fragments)
    3. Diverse (not all variants of the same fragment)

    **Corpus verification is the primary gate.** POS is secondary. This is
    because BPE fragments like "rosis" match the noun suffix "-osis" — only
    corpus verification can distinguish "pathogenesis" (real, in corpus) from
    "rosis" (BPE fragment, not in corpus).

    The "effective noun ratio" counts words that are BOTH in the corpus AND
    classified as nouns (by suffix OR by being a root word in corpus). This
    catches root nouns like "Cancer", "Surgery", "Blood" that lack suffixes.
    """
    n = len(concepts)
    if n == 0:
        return ConceptQualityMetrics(
            n_concepts=0, pos_counts={}, noun_ratio=0, verb_ratio=0,
            adj_ratio=0, bpe_fragment_ratio=0, corpus_hit_ratio=0,
            effective_concepts=[], n_effective=0,
        )

    # Build corpus word set for verification
    corpus_words: set[str] = set()
    if doc_texts:
        for text in doc_texts:
            for m in re.finditer(r'[a-zA-Z]{3,}', text):
                corpus_words.add(m.group().lower())

    pos_counts: Counter = Counter()
    fragments = 0
    corpus_hits = 0
    effective = []

    for concept in concepts:
        pos = classify_concept_pos(concept)
        in_corpus = concept.lower() in corpus_words

        # Primary gate: corpus verification.
        # A concept NOT in the corpus is either a BPE fragment or too
        # abstract — either way, it's a weak concept.
        if not in_corpus:
            # Check if it's a BPE fragment (short + not in corpus)
            if is_likely_bpe_fragment(concept, corpus_words):
                pos_counts['FRAG'] = pos_counts.get('FRAG', 0) + 1
                fragments += 1
            else:
                pos_counts[pos] += 1
            continue  # not effective

        # In corpus → real word
        corpus_hits += 1
        pos_counts[pos] += 1

        # Classify: is this a noun-like concept or verb/adjective?
        if pos in ('NN', 'NNS'):
            effective.append(concept)
        elif pos == 'UNK':
            # Root noun in corpus (Cancer, Surgery, Blood, tumor) → effective
            # These are the strongest concepts!
            effective.append(concept)
        elif pos in ('VBG', 'VBD', 'JJ'):
            # Verbs/adjectives in corpus — real words but less concept-like.
            # Include them but they won't boost noun_ratio.
            effective.append(concept)

    n_eff = len(effective)
    n_verified = corpus_hits  # concepts that passed corpus gate
    # Effective noun ratio: noun-like verified concepts / all verified
    noun_like = sum(
        1 for c in effective
        if classify_concept_pos(c) in ('NN', 'NNS', 'UNK')
    )
    effective_noun_ratio = noun_like / max(1, n_verified)

    return ConceptQualityMetrics(
        n_concepts=n,
        pos_counts=dict(pos_counts),
        noun_ratio=effective_noun_ratio,
        verb_ratio=(pos_counts.get('VBG', 0) + pos_counts.get('VBD', 0)) / n,
        adj_ratio=pos_counts.get('JJ', 0) / n,
        bpe_fragment_ratio=fragments / n,
        corpus_hit_ratio=corpus_hits / n,
        effective_concepts=effective,
        n_effective=n_eff,
    )


# ── Concept extraction (cluster-level, Stage 5 validated) ─────────────

STOP_WORDS = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was",
    "were", "been", "have", "has", "will", "would", "could", "should",
    "not", "but", "into", "also", "they", "them", "than", "then", "when",
    "what", "each", "more", "most", "some", "such", "only", "very", "just",
    "like", "concept", "concepts", "key", "main", "topic", "study", "studies",
    "result", "results", "method", "patient", "patients", "treatment",
    "associated", "compared", "significantly", "clinical", "using", "data",
    "analysis", "research", "health", "disease", "medical", "group",
    "following", "above", "document", "documents", "discuss", "discusses",
    "related", "based", "summarized", "listed", "outlined", "recent",
    "several", "evidence", "suggesting", "characterized", "indicating",
    "primarily", "showed", "might", "show", "seem", "appear", "require",
    "occur", "arise", "include", "single", "similar", "three", "five",
    "once", "which", "these", "those", "their", "there", "where",
    "while", "about", "after", "before", "between", "during", "through",
    "without", "within", "because", "however", "although", "whether",
    "many", "much", "both", "other", "another", "same", "different",
    "important", "possible", "available", "specific", "particular",
    "general", "common", "rare", "high", "low", "large", "small",
    "first", "second", "last", "next", "new", "old",
}


def extract_cluster_concepts(
    lens, lens_model, tokenizer,
    docs: list[str], layer: int,
    n_words: int = 8,
) -> list[str]:
    """J-Lens cluster-level concept extraction (Stage 5 method).

    Multi-doc concern prompt → L26 readout → top-k → content-word filter.
    """
    if len(docs) < 3:
        return []
    doc_block = "\n---\n".join(d[:400] for d in docs[:8])
    user_msg = (f"What concepts does this text discuss? "
                f"List {n_words} one-word concepts.\n\n{doc_block}")
    prefill = "The concepts discussed are"
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True,
            add_generation_prompt=False)
    else:
        prompt = f"{user_msg}\n{prefill}"

    lens_logits, _, _ = lens.apply(
        lens_model, prompt, layers=[layer],
        positions=[-1], max_seq_len=512)
    probs = torch.softmax(lens_logits[layer][0].float(), dim=-1)
    topk = probs.topk(30)

    words, seen = [], set()
    for idx in topk.indices.tolist():
        tok = tokenizer.decode([idx]).strip()
        low = tok.lower()
        if (len(tok) >= 4 and tok.isalpha() and low not in STOP_WORDS
                and low not in seen):
            if tok.islower() or (tok[0].isupper() and tok[1:].islower()):
                seen.add(low)
                words.append(tok)
        if len(words) >= n_words:
            break
    return words


# ── Cross-domain experiment ───────────────────────────────────────────

def load_novel_docs(max_docs: int = 200) -> list[str]:
    """Load novel corpus from GraphRAG-Bench."""
    # Try multiple possible paths (location varies by download method)
    for path in [Path("/tmp/graphrag-bench/Datasets/Corpus/novel.json"),
                 Path("/tmp/graphrag-bench/novel.json")]:
        if path.exists() and path.stat().st_size > 1000:
            break
    else:
        return []
    data = json.loads(path.read_text())
    full_text = data[0]["context"] if isinstance(data, list) else data["context"]
    # Split into ~1500-char chunks
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > 1500 and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent
    if current.strip():
        chunks.append(current.strip())
    return chunks[:max_docs]


def load_code_docs(max_docs: int = 200) -> list[str]:
    """Load code corpus (pi-code) via FineRecord loader."""
    try:
        from experiments.fine_record_dispersion import split_file_to_symbols
    except ImportError:
        return []
    # Find code files
    code_files = []
    for pat in ['**/*.py', '**/*.rs', '**/*.ts', '**/*.js']:
        code_files.extend(Path(REPO).glob(pat))
    # Filter out venv and node_modules
    code_files = [f for f in code_files
                  if '.venv' not in str(f) and 'node_modules' not in str(f)
                  and '/target/' not in str(f)][:max_docs]
    docs = []
    for f in code_files[:max_docs]:
        try:
            text = f.read_text(errors='ignore')[:2000]
            docs.append(text)
        except Exception:
            continue
    return docs


def run_cross_domain_pos(
    lens, lens_model, tokenizer, embed,
    domains: dict[str, list[str]],
    layer: int | None = None,
) -> dict:
    """Run concept extraction on multiple domains, compare POS distributions.

    For each domain:
      1. Embed docs → bge-m3 clustering → clusters
      2. For each cluster: J-Lens concept extraction
      3. Score concept quality (POS + corpus verification)
      4. Aggregate domain-level POS distribution

    Returns per-domain stats + cross-domain comparison.
    """
    if layer is None:
        layer = lens.source_layers[-1]

    domain_results = {}

    for domain_name, doc_texts in domains.items():
        n_docs = len(doc_texts)
        if n_docs < 10:
            print(f"\n  [{domain_name}] skipped (only {n_docs} docs)")
            continue

        print(f"\n  [{domain_name}] {n_docs} docs")

        # Cluster
        print(f"    embedding...", flush=True)
        vecs = np.asarray(embed.embed(doc_texts), dtype=np.float32)
        clusters = partition_hdbscan(vecs.tolist())
        # Keep clusters with ≥ 5 docs
        valid = {cid: members for cid, members in clusters.items()
                 if len(members) >= 5}
        print(f"    {len(valid)} clusters (≥5 docs)", flush=True)

        if not valid:
            continue

        # Concept extraction per cluster
        all_concepts = []
        cluster_details = []
        for cid, members in sorted(valid.items(),
                                     key=lambda x: len(x[1]), reverse=True)[:15]:
            member_docs = [doc_texts[m] for m in members]
            concepts = extract_cluster_concepts(
                lens, lens_model, tokenizer, member_docs, layer)
            # Score quality
            metrics = concept_quality_score(concepts, member_docs)
            all_concepts.extend(concepts)
            cluster_details.append({
                "cluster_id": cid,
                "n_docs": len(members),
                "concepts": concepts,
                "pos_counts": metrics.pos_counts,
                "noun_ratio": round(metrics.noun_ratio, 2),
                "verb_ratio": round(metrics.verb_ratio, 2),
                "bpe_fragment_ratio": round(metrics.bpe_fragment_ratio, 2),
                "corpus_hit_ratio": round(metrics.corpus_hit_ratio, 2),
            })
            print(f"    C{cid} ({len(members)}d): {concepts}")
            print(f"      noun={metrics.noun_ratio:.0%} "
                  f"verb={metrics.verb_ratio:.0%} "
                  f"frag={metrics.bpe_fragment_ratio:.0%} "
                  f"corpus_hit={metrics.corpus_hit_ratio:.0%}")

        # Domain-level aggregation
        domain_metrics = concept_quality_score(all_concepts, doc_texts)
        domain_results[domain_name] = {
            "n_docs": n_docs,
            "n_clusters": len(valid),
            "n_concepts": len(all_concepts),
            "pos_distribution": domain_metrics.pos_counts,
            "noun_ratio": round(domain_metrics.noun_ratio, 3),
            "verb_ratio": round(domain_metrics.verb_ratio, 3),
            "adj_ratio": round(domain_metrics.adj_ratio, 3),
            "bpe_fragment_ratio": round(domain_metrics.bpe_fragment_ratio, 3),
            "corpus_hit_ratio": round(domain_metrics.corpus_hit_ratio, 3),
            "clusters": cluster_details,
        }

    return domain_results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("=" * 70)
    print("Phase 16a: Cross-domain concept POS distribution analysis")
    print("=" * 70)

    # Load model + lens
    print("\n[1/3] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    # Load embedder
    from experiments.embed_cache import CachedBgeM3Provider
    embed = CachedBgeM3Provider()

    # Load domains
    print(f"\n[2/3] Loading corpora...")
    max_docs = 200  # keep small for speed

    domains = {}

    # NFCorpus (medical)
    nf = load_beir_fine_records("nfcorpus", max_docs=max_docs)
    domains["nfcorpus_medical"] = [r[1] for r in nf]
    print(f"  nfcorpus: {len(domains['nfcorpus_medical'])} docs")

    # Novel
    novels = load_novel_docs(max_docs=max_docs)
    if novels:
        domains["novel"] = novels
        print(f"  novel: {len(novels)} docs")

    # Code
    code = load_code_docs(max_docs=max_docs)
    if code:
        domains["code"] = code
        print(f"  code: {len(code)} docs")

    # Run cross-domain analysis
    print(f"\n[3/3] Running cross-domain POS analysis...")
    layer = lens.source_layers[-1]
    results = run_cross_domain_pos(
        lens, lens_model, tokenizer, embed, domains, layer)

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("CROSS-DOMAIN POS DISTRIBUTION COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'domain':<20} {'n_conc':>7} {'noun%':>7} {'verb%':>7} "
          f"{'adj%':>7} {'frag%':>7} {'corpus%':>8}")
    print(f"  {'-' * 68}")
    for domain, stats in results.items():
        print(f"  {domain:<20} {stats['n_concepts']:>7} "
              f"{stats['noun_ratio']:>6.0%} {stats['verb_ratio']:>6.0%} "
              f"{stats['adj_ratio']:>6.0%} {stats['bpe_fragment_ratio']:>6.0%} "
              f"{stats['corpus_hit_ratio']:>7.0%}")

    # Threshold sensitivity analysis
    print(f"\n{'=' * 70}")
    print("THRESHOLD SENSITIVITY: noun_ratio ≥ 0.4")
    print(f"{'=' * 70}")
    for domain, stats in results.items():
        clusters = stats.get("clusters", [])
        n_pass = sum(1 for c in clusters if c["noun_ratio"] >= 0.4)
        n_total = len(clusters)
        print(f"  {domain:<20} {n_pass}/{n_total} clusters pass "
              f"({n_pass/max(1,n_total):.0%})")

    # Save
    out = {
        "method": "cross_domain_pos_distribution",
        "model": cand["name"],
        "layer": layer,
        "max_docs_per_domain": max_docs,
        "domains": results,
    }
    out_path = EXP / "phase16a_cross_domain_pos.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")


if __name__ == "__main__":
    main()
