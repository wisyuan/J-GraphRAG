"""Phase 52: J-Lens 专名级实体提取（Phase 50 失败分析的直接后续）。

Phase 50（LightRAG 纯 J-Lens 替换）保持率未达 0.9 的瓶颈：词表只有主题级
概念（cancer/marriage），缺命名实体（basal cell carcinoma、Princess
Frederica、Arthur/Excalibur）。failure_analysis 的三类失败案例全部指向
「topical concept vs named entity」的粒度错位。

理论依据（J-space 论文 transformer-circuits.pub/2026/workspace）：
  1. 内部推理例：多跳问题中模型自发把中间实体放进 workspace —— 实体级表示存在。
  2. paired-question 协议：什么问题决定什么内容进入 J-space。隐式使用（cloze）
     不把属性标签放进 workspace；显式命名问题才放。
  3. J-lens 单 token 限制：多词实体以核心 token 读出，用语料 bigram/trigram
     共现统计合并还原（complete_prefix 思路从 BPE 前缀推广到短语）。

S1 实测教训（本仓 phase52 诊断，详见报告 JSON 的 s1 字段）：
  - 复数列举 prompt（"List the specific people..."）+ prefill 位置读出：
    workspace 被列表格式 token（":" "**" "‘"）占据，实体完全读不出。
  - 单数显式命名问题（"What is the most specific named entity...?"）：
    深层（L24-26）稳定读出主导实体（basal 0.31 / Queen 0.85）——
    paired-question 协议的直接验证。
  - 句末多位置读出（positions=各句末 token）：同一次 forward 内，
    每个句末位置的 workspace 持有该句的局部实体/术语
    （fair/skin/exposure 在风险因子句末读出；phase32 也已发现
    生成位置能读出 -1 没有的概念）——这是零额外 forward 的召回来源。

最终管线（每 chunk 1 次 forward，多位置读出）：
  单数显式命名问题（"What is the most specific named entity...?"，
    chat template + assistant prefill）
  → positions=[chunk 内各句末 token, -1] × layers L16-26 一次 lens.apply
  → 深层（L≥22）聚合 harvest（top-8/层，min_prob，ASCII/STOP/话语词过滤）
  → 稳定性（≥2 个 (pos,layer) 命中或单点 prob≥0.3）
  → chunk 本地验证（complete_prefix 兜底 BPE 片段，如 tre→tregeagle）
  → bigram 合并 + 左右扩展（basal→basal cell carcinoma；span 间隔
    只允许 of/von/der 类贵族小品词——"skin cancer are diagnosed" 这类
    跨助动词合并在语法上被禁止；句读破折号为硬边界）
  → 独立单 token 收紧：仅保留 chunk 内字面全大写缩写（BCC/UV）与
    内部大写专名（重复大写 ≥2 或非 WordNet 词——Frederica/Dozmare
    不是英语词）；独立小写词一律丢弃（S2 第一轮 precision=10% 的主因）

阶段：
  S1 prompt/读出策略对比（v1=单数问题+句末多位置，v2=复数问题+句末多
     位置；medical 20 chunks 冒烟，打印 5 个 chunk 人检）
  S2 能力判决：覆盖率（Phase 50 缺失实体清单，≥60% 过线）
              + 精度（medical 200 chunks 抽样，DeepSeek judge ≤250 次，≥70% 过线）
  S3 全量提取 → concept_cache/entity_cache_{domain}.json（断点续跑）
  S4 终审：实体节点并入图（contains 边接 chunk，实体-实体关系暂缺）
     → 重跑 phase50 ah 臂（medical 56 + novel 48，DeepSeek judge）
     → 保持率判决（medical 目标 ≥0.9，novel 目标稳定 >0.9）

实验结论（2026-07-21，四轮迭代后的最终判决，详见输出 JSON conclusion）：
  **FALSIFIED @ S2**。覆盖率 overall=0.30（medical 0.50：basal cell
  carcinoma/fair skin 可回收；novel 0.17：仅 Cornwall），精度 0.196，
  双门槛（0.60/0.70）均未过线，S3/S4 未执行。瓶颈是结构性的：
  罕见多 token 专名（Dozmare/Excalibur/Frederica/Pawel-Rammingen）在
  单数/复数/类型化问题 × -1/句末多位置的全部组合下均未进入 workspace
  （7B 4bit 对低频专名不形成可读出表示）；paired-question 协议本身
  被证实（显式命名问题确实读出主导实体 basal/Queen @L24-26），但
  可达实体限于高频显著者，无法补齐 Phase 50 的保持率缺口。

运行：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -c "import experiments.phase52_entity_extraction"      # 零副作用
    python -m experiments.phase52_entity_extraction --stage s1 --smoke
    python -m experiments.phase52_entity_extraction --stage s2
    python -m experiments.phase52_entity_extraction --stage s3 --domain all
    python -m experiments.phase52_entity_extraction --stage s4 --domain all
    python -m experiments.phase52_entity_extraction --stage all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
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
from experiments.phase16a_cross_domain_pos import classify_concept_pos
from experiments.concept_quality import (
    complete_prefix, build_corpus_term_freq, _get_wordnet_nouns,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"
OUT_PATH = EXP / "phase52_entity_extraction.json"
S4_OUT_PATH = EXP / "phase52_lightrag_j_entities.json"

# ── S1: 两个读出策略变体（都是显式命名问题——paired-question 协议；
#    v1 句末多位置（召回导向），v2 -1 单位置（主导实体，诊断基线）） ──────

PROMPT_VARIANTS = {
    "v1": {
        "desc": "singular naming question + sentence-end multi-position "
                "readout (1 forward/chunk)",
        "user": ("What is the most specific named entity, proper noun, or "
                 "technical term mentioned in this text?\n\n{chunk}"),
        "prefill": "The most specific named entity mentioned in this "
                   "text is",
        "positions": "sentence_ends",
    },
    "v2": {
        "desc": "plural naming question + sentence-end multi-position "
                "readout (diagnostic contrast)",
        "user": ("What specific named entities (people, places, "
                 "organizations, proper nouns, specific technical terms) "
                 "are mentioned in this text?\n\n{chunk}"),
        "prefill": "The specific named entities mentioned in this text "
                   "include",
        "positions": "sentence_ends",
    },
}

# Readout layers: apply() reads L16-26 in one pass; harvest aggregates the
# deep workspace layers (entity signal observed at L22-26 in diagnostics).
LAYERS_READ = list(range(16, 27))
LAYERS_DEEP = 22
HARVEST_TOPK = 8
HARVEST_MIN_PROB = 0.02
STABLE_MIN_HITS = 2        # ≥2 (position, layer) appearances, or ...
STABLE_SINGLE_PROB = 0.3   # ... one strong single readout

# Entity-prompt contamination + discourse words that leak into sentence-end
# readouts (observed in diagnostics: poor/meanwhile/likely/his...).
ENTITY_PREFILL_WORDS = {
    "entities", "entity", "mentioned", "mentioning", "specific",
    "specifically", "names", "name", "named", "titles", "title", "proper",
    "terms", "term", "text", "people", "person", "persons", "places",
    "place", "organizations", "organization", "nouns", "noun", "list",
    "listed", "include", "includes", "following", "locations", "location",
    "mentions", "reference", "references", "examples", "example",
    "another", "most",
}
DISCOURSE_WORDS = {
    "his", "her", "its", "their", "our", "your", "him", "them", "she",
    "meanwhile", "thus", "besides", "however", "indeed", "certainly",
    "unfortunately", "plus", "yes", "isn", "don", "how", "look", "come",
    "born", "unlike", "user", "data", "poor", "likely", "certain",
    "strange", "anxious", "placid", "older", "weak", "aging",
}

BAD_WORDS = (STOP_WORDS_EXTENDED | PREFILL_WORDS | ENTITY_PREFILL_WORDS
             | DISCOURSE_WORDS)

# Function words, two roles:
#  - SPAN_FUNCTION_WORDS: noble/possessive particles that may appear INSIDE
#    a multi-word entity span ("Princess Frederica of Hanover",
#    "Baron von Pawel-Rammingen").
#  - AUX_WORDS: copula/auxiliaries/prepositions — skipped when building the
#    content-word stream (so they never extend a span), and NOT allowed
#    inside a span gap ("skin cancer are diagnosed" is not an entity).
SPAN_FUNCTION_WORDS = {
    "of", "the", "de", "von", "van", "der", "den", "du", "la", "le", "el",
    "di", "da", "des", "del", "und", "al", "bin", "ibn", "fitz", "ten",
    "ter",
}
AUX_WORDS = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "will", "would", "can", "could", "may", "might", "shall",
    "should", "must", "do", "does", "did",
    "and", "a", "an", "in", "on", "upon", "to",
}
FUNCTION_WORDS = SPAN_FUNCTION_WORDS | AUX_WORDS

# S2 覆盖率目标：Phase 50 failure_analysis 的缺失实体清单
# （experiments/m6/phase50_lightrag_j.json failure_analysis.cases[*].cause）
MISSING_ENTITIES = {
    "medical": [
        "basal cell carcinoma",   # Medical-604c9d44: BCC 风险因子 chunk 无锚点
        "fair skin",
        "organ transplant",
        "immune suppression",
    ],
    "novel": [
        "Princess Frederica",          # Novel-74440a6a: 谁娶了她
        "Baron von Pawel-Rammingen",   # 同上（答案实体）
        "Arthur",                      # Novel-613d4e81: Cornwall×亚瑟王
        "Excalibur",
        "Dozmare",
        "Cornwall",
    ],
}

COVERAGE_GATE = 0.60
PRECISION_GATE = 0.70
PRECISION_MAX_JUDGE_CALLS = 250


# ── Prompt 构建（chat 结构参考 build_concern_prompt_full） ──────────────


def build_entity_prompt(chunk_text: str, tokenizer, variant: str = "v1") -> str:
    """Explicit naming-question prompt (paired-question protocol)."""
    v = PROMPT_VARIANTS[variant]
    user_msg = v["user"].format(chunk=chunk_text)
    prefill = v["prefill"]
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


def readout_positions(
    tokenizer, prompt: str, chunk_text: str, variant: str,
    max_seq_len: int = 512,
) -> list[int]:
    """Token positions to read out.

    v1: every sentence-end token inside the chunk region + [-1].
    v2: [-1] only (dominant-entity baseline).
    """
    if PROMPT_VARIANTS[variant]["positions"] == "last_only":
        return [-1]
    enc = tokenizer(prompt, return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    start_char = prompt.find(chunk_text[:80])
    if start_char < 0:
        return [-1]
    pos = []
    for i, (s, _e) in enumerate(offsets):
        if s < start_char or i >= max_seq_len - 1:
            continue
        tok_text = prompt[s:_e]
        if re.search(r'[.!?]["\')\]]*$', tok_text):
            pos.append(i)
    return pos + [-1]


# ── 多位置深层 harvest（结构对齐 phase17 extract_depth_gradient 的
#    lens.apply 调用，但 positions 为句末列表、decode 保留大写专名） ──────


def _decode_row_entity(
    logits_row: torch.Tensor,
    tokenizer,
    topk: int = HARVEST_TOPK,
    min_prob: float = HARVEST_MIN_PROB,
) -> list[tuple[str, float]]:
    """Top-k entity-candidate tokens from one (layer, position) logits row.

    Keeps capitalized proper nouns and all-caps acronyms (BCC), len>=3,
    ASCII alpha; rejects stopword/prefill/discourse contamination.
    """
    probs = torch.softmax(logits_row.float(), dim=-1)
    tk = probs.topk(topk)
    out = []
    for idx, p in zip(tk.indices.tolist(), tk.values.tolist()):
        tok = tokenizer.decode([int(idx)]).strip()
        low = tok.lower()
        if (len(tok) >= 3 and tok.isalpha() and is_ascii_english(tok)
                and low not in BAD_WORDS and float(p) >= min_prob):
            out.append((low, float(p)))
    return out


def harvest_entities(
    lens, lens_model, tokenizer,
    prompt: str,
    positions: list[int],
    layers: list[int] | None = None,
    max_seq_len: int = 512,
) -> dict[str, dict]:
    """One forward pass; aggregate entity candidates over (position, layer).

    Returns {token_lower: {total_prob, max_prob, n_hits, layers, positions}}
    aggregated over deep layers (>= LAYERS_DEEP) at all readout positions.
    """
    layers = layers or LAYERS_READ
    lens_logits, _model_logits, _ = lens.apply(
        lens_model, prompt,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
    )
    agg: dict[str, dict] = {}
    deep = [l for l in layers if l >= LAYERS_DEEP]
    for layer in deep:
        for pi, pos in enumerate(positions):
            for low, p in _decode_row_entity(lens_logits[layer][pi],
                                             tokenizer):
                a = agg.setdefault(low, {"total_prob": 0.0, "max_prob": 0.0,
                                         "n_hits": 0, "layers": set(),
                                         "positions": set()})
                a["total_prob"] += p
                a["max_prob"] = max(a["max_prob"], p)
                a["n_hits"] += 1
                a["layers"].add(layer)
                a["positions"].add(pos)
    for a in agg.values():
        a["total_prob"] = round(a["total_prob"], 4)
        a["max_prob"] = round(a["max_prob"], 4)
        a["layers"] = sorted(a["layers"])
        a["positions"] = sorted(a["positions"])
    return agg


# ── 过滤：稳定性 + chunk 本地验证 + BM25 补全 + POS 放宽 ────────────────


def filter_entity_tokens(
    agg: dict[str, dict],
    chunk_words: set[str],
    corpus_freq: dict[str, int],
    wn_nouns: set[str],
    max_keep: int = 16,
) -> list[dict]:
    """Stability + chunk-local verification + relaxed POS.

    - stability: >= STABLE_MIN_HITS (pos,layer) hits, or one readout with
      prob >= STABLE_SINGLE_PROB
    - POS relaxed: reject VBG/VBD only for lowercase tokens (proper nouns
      like "Reading" must not be killed by the suffix heuristic)
    - chunk-local verification: token (lowercased) must be a real word in
      THIS chunk; BPE fragments get one complete_prefix rescue attempt
      (e.g. "tre" -> "tregeagle"), the completion must appear in the chunk.
    """
    out = []
    for low, a in agg.items():
        if a["n_hits"] < STABLE_MIN_HITS and a["max_prob"] < STABLE_SINGLE_PROB:
            continue
        word = low
        if word not in chunk_words:
            # BPE-fragment rescue, chunk-local first: among corpus words
            # starting with the prefix, prefer those appearing in THIS
            # chunk ("bas" -> "basal", not the globally-frequent "based"),
            # then fall back to the global complete_prefix ranking.
            local = [w for w in corpus_freq
                     if w.startswith(word) and w in chunk_words]
            if local:
                word = max(local, key=lambda w: corpus_freq[w])
            else:
                comp = complete_prefix(word, corpus_freq, wn_nouns)
                if comp and comp in chunk_words:
                    word = comp
                else:
                    continue
        # Post-completion: the rescued word must not be a stopword either
        # (e.g. "treat" -> "treatment" would smuggle a stopword back in)
        if word in BAD_WORDS:
            continue
        # POS filter on the FINAL word (after completion rescue): reject
        # clear gerunds/participles; proper nouns like "Reading" are rare
        # enough in these corpora that the suffix heuristic is net-positive
        pos = classify_concept_pos(word)
        if pos in ("VBG", "VBD"):
            continue
        out.append({
            "word": word,
            "total_prob": a["total_prob"],
            "max_prob": a["max_prob"],
            "n_hits": a["n_hits"],
            "layers": a["layers"],
        })
    out.sort(key=lambda x: -x["total_prob"])
    return out[:max_keep]


# ── 多词实体合并（J-lens 单 token 限制的语料统计还原） ────────────────────


def tokenize_content_words(text: str) -> list[tuple[str, int, int]]:
    """Word tokens minus function words: [(word, start, end)]."""
    return [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"[A-Za-z][A-Za-z'’\-]*", text)
        if m.group().lower() not in FUNCTION_WORDS and len(m.group()) >= 2
    ]


def build_ngram_stats(
    chunk_texts: list[str],
) -> tuple[Counter, Counter, Counter]:
    """Corpus content-word unigram/bigram/trigram counts (function words
    skipped).

    Content-word adjacency tolerates function words inside entity spans:
    "Princess Frederica of Hanover" yields the content bigram
    (frederica, hanover).
    """
    uni: Counter = Counter()
    bi: Counter = Counter()
    tri: Counter = Counter()
    for t in chunk_texts:
        cw = [w.lower() for w, _, _ in tokenize_content_words(t)]
        uni.update(cw)
        for i in range(len(cw) - 1):
            bi[(cw[i], cw[i + 1])] += 1
        for i in range(len(cw) - 2):
            tri[(cw[i], cw[i + 1], cw[i + 2])] += 1
    return uni, bi, tri


_SENT_END = re.compile(r'[.!?]')
# Hard span boundaries: commas/colons/parens/quotes/dashes also break an
# entity span ("skin cancer, also" / "Age – Risk" must not merge).
_SPAN_BREAK = re.compile(r'[.!?,.;:()"“”\[\]{}–—\-…]')


def _gap_ok(gap: str) -> bool:
    """A span gap may contain only whitespace + SPAN_FUNCTION_WORDS
    (" of ", " von der ") — no punctuation, no auxiliaries."""
    if _SPAN_BREAK.search(gap):
        return False
    words = re.findall(r"[a-zA-Z]+", gap)
    return all(w.lower() in SPAN_FUNCTION_WORDS for w in words)


def merge_multiword(
    tokens_lower: list[str],
    chunk_text: str,
    bigram_freq: Counter,
    trigram_freq: Counter,
    unigram_freq: Counter,
    bigram_min: int = 2,
    extend_min: int = 3,
    extend_ratio: float = 0.05,
    max_span_words: int = 5,
) -> tuple[list[str], set[str]]:
    """Merge/extend single tokens into multi-word entities via corpus stats.

    Two modes from an extracted token at content position j:
      merge   — next content word is ALSO extracted and adjacent in this
                chunk (both sides J-lens-confirmed; corpus bigram NOT
                required — single-attested pairs like "fair skin" would
                otherwise be unmergeable);
      extend  — next content word is not extracted but the bigram is both
                frequent (>= extend_min) AND specific to this pair
                (bigram(a,b)/unigram(b) >= extend_ratio — recovers
                "basal cell carcinoma" from "basal" alone, while generic
                continuations like "cancer is" fail the ratio test).
    Extension stops at sentence/clause boundaries and max_span_words.
    Spans are literal text slices (original casing, function words inside
    kept: "Baron von Pawel-Rammingen").

    Returns (multiword_spans, absorbed_token_lowers).
    """
    content = tokenize_content_words(chunk_text)
    lows = [w.lower() for w, _, _ in content]
    tokset = set(tokens_lower)
    used = [False] * len(content)
    spans: list[str] = []
    seen_spans: set[str] = set()
    absorbed: set[str] = set()

    i = 0
    while i < len(content):
        if used[i] or lows[i] not in tokset:
            i += 1
            continue
        chain = [i]
        # Left extension: pull in a specific left neighbor
        # ("frederica" -> "Princess Frederica of Hanover").
        while len(chain) < max_span_words and chain[0] - 1 >= 0:
            left = chain[0] - 1
            if used[left]:
                break
            gap = chunk_text[content[left][2]:content[chain[0]][1]]
            if not _gap_ok(gap):
                break
            pair = (lows[left], lows[chain[0]])
            n_pair = bigram_freq.get(pair, 0)
            if (lows[left] not in BAD_WORDS
                    and n_pair >= extend_min
                    and n_pair / max(1, unigram_freq.get(lows[left], 0))
                    >= extend_ratio):
                chain.insert(0, left)
            else:
                break
        # Right merge/extension.
        j = chain[-1]
        while len(chain) < max_span_words:
            nxt = None
            if j + 1 < len(content) and not used[j + 1]:
                gap = chunk_text[content[j][2]:content[j + 1][1]]
                if _gap_ok(gap):
                    pair = (lows[j], lows[j + 1])
                    n_pair = bigram_freq.get(pair, 0)
                    if lows[j + 1] in tokset:
                        # merge: both sides J-lens-extracted AND adjacent in
                        # this chunk — enough evidence on its own; the
                        # corpus bigram may be single-attested ("fair skin"
                        # appears once corpus-wide) and is NOT required.
                        nxt = j + 1
                    elif (n_pair >= extend_min
                          and lows[j + 1] not in BAD_WORDS
                          and n_pair / max(1, unigram_freq.get(
                              lows[j + 1], 0)) >= extend_ratio):
                        nxt = j + 1  # extension: word need not be extracted
            if nxt is None:
                break
            chain.append(nxt)
            j = nxt
        if len(chain) >= 2:
            start = content[chain[0]][1]
            end = content[chain[-1]][2]
            span = chunk_text[start:end].strip()
            if span.lower() not in seen_spans:
                seen_spans.add(span.lower())
                spans.append(span)
            for k in range(chain[0], chain[-1] + 1):
                used[k] = True
                if lows[k] in tokset:
                    absorbed.add(lows[k])
            i = chain[-1] + 1
        else:
            i += 1
    return spans, absorbed


def surface_form(word_lower: str, chunk_text: str) -> str:
    """Original casing of a word's first occurrence in the chunk."""
    m = re.search(rf"\b{re.escape(word_lower)}\b", chunk_text, re.IGNORECASE)
    return m.group() if m else word_lower


def count_interior_capitals(word_lower: str, chunk_text: str) -> int:
    """Number of capitalized occurrences NOT at a sentence start —
    proper-noun evidence ("Princess" mid-sentence vs "The" after a period).
    """
    n = 0
    for m in re.finditer(rf"\b{re.escape(word_lower)}\b", chunk_text):
        before = chunk_text[:m.start()].rstrip()
        if before and not _SENT_END.search(before[-1:]):
            n += 1
    return n


_WN_LEMMAS: set[str] | None = None


def _get_wordnet_lemmas() -> set[str]:
    """All WordNet lemma names (any POS) — the 'common English' test for
    standalone capitalized singles. Rare proper names (Frederica, Dozmare)
    are NOT WordNet words; common nouns heading a sentence (Age, Signs,
    Radiation) are."""
    global _WN_LEMMAS
    if _WN_LEMMAS is None:
        try:
            from nltk.corpus import wordnet
            _WN_LEMMAS = {
                lm.name().lower()
                for lm in wordnet.all_lemma_names()
            }
        except Exception:
            _WN_LEMMAS = set()
    return _WN_LEMMAS


def appears_all_caps(word_lower: str, chunk_text: str) -> bool:
    """True if the word appears in ALL-CAPS in the chunk (BCC, UV)."""
    return bool(re.search(rf"\b{re.escape(word_lower.upper())}\b",
                          chunk_text))


# ── 单 chunk 完整提取（1 次 forward，多位置读出） ─────────────────────────


def extract_entities_chunk(
    lens, lens_model, tokenizer,
    chunk_text: str,
    corpus_freq: dict[str, int],
    wn_nouns: set[str],
    bigram_freq: Counter,
    trigram_freq: Counter,
    unigram_freq: Counter,
    top_freq_words: set[str],
    variant: str = "v1",
) -> dict:
    """Full entity pipeline for one chunk (1 forward pass)."""
    prompt = build_entity_prompt(chunk_text, tokenizer, variant)
    positions = readout_positions(tokenizer, prompt, chunk_text, variant)
    agg = harvest_entities(lens, lens_model, tokenizer, prompt, positions)

    chunk_words = build_corpus_word_set([chunk_text])
    # build_corpus_word_set keeps len>=4; entities allow len 3 — but only
    # STANDALONE 3-letter words (\b bounded; a bare "[a-zA-Z]{3}" matches
    # prefixes inside longer words and would falsely verify fragments
    # like "bas", blocking the completion rescue).
    for m in re.finditer(r"\b[a-zA-Z]{3}\b", chunk_text):
        chunk_words.add(m.group().lower())

    profiles = filter_entity_tokens(agg, chunk_words, corpus_freq, wn_nouns)
    tokens = [p["word"] for p in profiles]

    spans, absorbed = merge_multiword(
        tokens, chunk_text, bigram_freq, trigram_freq, unigram_freq)

    entities = list(spans)
    for t in tokens:
        if t in absorbed:
            continue
        # Standalone single-token keep rules (strict — S2 round 1 showed
        # standalone generic words destroy precision):
        #  - ALL-CAPS acronym appearing verbatim in the chunk (BCC, UV); or
        #  - capitalized in chunk interior AND (recurring capitalized >= 2,
        #    or not a WordNet word at all — rare proper names like
        #    Frederica/Dozmare are not in the English lexicon)
        #  Standalone lowercase singles are dropped: they were the dominant
        #  precision-10% noise source (options/treat/harm/exposure).
        if appears_all_caps(t, chunk_text):
            entities.append(surface_form(t, chunk_text).upper())
            continue
        n_caps = count_interior_capitals(t, chunk_text)
        if n_caps >= 2 or (n_caps >= 1 and t not in _get_wordnet_lemmas()):
            entities.append(surface_form(t, chunk_text))

    # Dedupe (case-insensitive), multiword first
    seen = set()
    final = []
    for e in entities:
        el = e.lower()
        if el not in seen:
            seen.add(el)
            final.append(e)

    return {
        "entities": final,
        "n_entities": len(final),
        "n_multiword": len(spans),
        "token_profiles": profiles,
    }


# ── 资源准备 ─────────────────────────────────────────────────────────────


def load_model_stack():
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    return lens, lens_model, tokenizer, cand


class CorpusResources:
    """Per-domain corpus statistics for completion + multiword merging."""

    def __init__(self, chunk_texts: list[str]) -> None:
        self.corpus_freq = build_corpus_term_freq(chunk_texts)
        self.wn_nouns = _get_wordnet_nouns()
        self.unigram_freq, self.bigram_freq, self.trigram_freq = \
            build_ngram_stats(chunk_texts)
        # top-200 most frequent corpus words: too generic to stand alone
        self.top_freq_words = {
            w for w, _ in Counter(self.corpus_freq).most_common(200)}


def _extractor(stack, res: CorpusResources, variant: str):
    lens, lens_model, tokenizer = stack[0], stack[1], stack[2]

    def extract(chunk_text: str) -> dict:
        return extract_entities_chunk(
            lens, lens_model, tokenizer, chunk_text,
            res.corpus_freq, res.wn_nouns,
            res.bigram_freq, res.trigram_freq, res.unigram_freq,
            res.top_freq_words, variant)
    return extract


# ── S1: 变体对比 + 冒烟 ──────────────────────────────────────────────────


def run_s1(stack, domain: str = "medical", max_chunks: int = 20,
           n_print: int = 5) -> dict:
    print(f"\n{'='*70}\nS1: variant comparison ({domain}, "
          f"{max_chunks} chunks smoke)\n{'='*70}")
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())[:max_chunks]
    res = CorpusResources([t for _, t in chunk_items])

    out = {"domain": domain, "n_chunks": len(chunk_items),
           "variant_desc": {k: v["desc"] for k, v in PROMPT_VARIANTS.items()},
           "diagnosis_note": (
               "复数列举 prompt + prefill -1 读出在全层被格式 token "
               "(':','**','‘') 占据（S1 第一次运行 avg_entities=0.1）；"
               "单数显式命名问题在 L24-26 稳定读出主导实体；"
               "句末多位置在一次 forward 内给出句级局部实体。"
               "v1=句末多位置, v2=-1 单位置（对照）。"),
           "variants": {}}
    for variant in PROMPT_VARIANTS:
        extract = _extractor(stack, res, variant)
        vres = {"chunks": {}, "stats": {}}
        tot = cap = multi = 0
        for cid, text in chunk_items:
            r = extract(text)
            ents = r["entities"]
            vres["chunks"][cid] = ents
            tot += len(ents)
            multi += r["n_multiword"]
            cap += sum(1 for e in ents
                       if " " not in e and not e.islower())
        n = max(1, len(chunk_items))
        proper_rate = (cap + multi) / max(1, tot)
        vres["stats"] = {
            "avg_entities": round(tot / n, 2),
            "n_multiword": multi,
            "n_capitalized_single": cap,
            "proper_rate": round(proper_rate, 3),
        }
        out["variants"][variant] = vres
        print(f"\n── variant {variant} ({PROMPT_VARIANTS[variant]['desc']})")
        print(f"   avg_entities={tot/n:.1f}  multiword={multi}  "
              f"capitalized={cap}  proper_rate={proper_rate:.2f}")
        # 冒烟：打印前 n_print 个 chunk 的实体列表供人工检查
        for cid, text in chunk_items[:n_print]:
            print(f"\n   [{cid}] {text[:90]}...")
            print(f"     → {vres['chunks'][cid]}")

    # Winner: recall first (avg entities), then proper_rate
    def score(v: str) -> tuple[float, float]:
        s = out["variants"][v]["stats"]
        return (s["avg_entities"] * (0.5 + s["proper_rate"]),
                s["proper_rate"])
    winner = max(PROMPT_VARIANTS, key=score)
    out["winner"] = winner
    out["winner_rationale"] = (
        f"{winner} wins on recall*properness score "
        f"{score(winner)} vs {score([v for v in PROMPT_VARIANTS if v != winner][0])}. "
        "Smoke samples printed above for manual template-pollution check.")
    print(f"\nS1 winner: {winner}")
    return out


# ── S2: 能力判决（覆盖率 + 精度） ─────────────────────────────────────────


def entity_recovered(target: str, extracted: list[str]) -> bool:
    """Multi-word targets need the full span (or all content words);
    single-word targets allow substring (arthur ⊂ 'King Arthur')."""
    t = target.lower()
    twords = [w for w in re.findall(r"[a-z0-9]+", t)
              if w not in FUNCTION_WORDS]
    multiword_target = len(twords) > 1
    for e in extracted:
        el = e.lower()
        if t in el:
            return True
        ewords = set(re.findall(r"[a-z0-9]+", el))
        if multiword_target:
            if twords and all(w in ewords for w in twords):
                return True
        else:
            if el in t or t in el:
                return True
    return False


def run_s2_coverage(stack, variant: str, domain: str,
                    max_chunks_per_entity: int = 5) -> dict:
    """Can the pipeline recover Phase 50's missing entities on the chunks
    that actually contain them?"""
    targets = MISSING_ENTITIES[domain]
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    res = CorpusResources(list(corpus.values()))
    extract = _extractor(stack, res, variant)

    per_entity = []
    n_recovered = 0
    for target in targets:
        pat = re.compile(re.escape(target), re.IGNORECASE)
        hit_cids = [cid for cid, t in corpus.items() if pat.search(t)]
        if not hit_cids:
            # multi-word fallback: all content words present
            twords = [w for w in re.findall(r"[a-z0-9]+", target.lower())
                      if w not in FUNCTION_WORDS]
            hit_cids = [
                cid for cid, t in corpus.items()
                if all(re.search(rf"\b{re.escape(w)}", t, re.IGNORECASE)
                       for w in twords)
            ]
        hit_cids = hit_cids[:max_chunks_per_entity]
        recovered = False
        chunk_results = {}
        for cid in hit_cids:
            r = extract(corpus[cid])
            chunk_results[cid] = r["entities"]
            if entity_recovered(target, r["entities"]):
                recovered = True
        per_entity.append({
            "entity": target, "n_chunks_with_entity": len(hit_cids),
            "recovered": recovered, "extractions": chunk_results,
        })
        n_recovered += recovered
        print(f"  [{domain}] {target!r}: "
              f"{'RECOVERED' if recovered else 'missed'} "
              f"({len(hit_cids)} chunks checked)")
        for cid, ents in list(chunk_results.items())[:2]:
            print(f"      {cid}: {ents}")

    coverage = n_recovered / max(1, len(targets))
    return {
        "domain": domain, "targets": per_entity,
        "n_recovered": n_recovered, "n_targets": len(targets),
        "coverage": round(coverage, 3),
        "gate": COVERAGE_GATE, "passed": coverage >= COVERAGE_GATE,
    }


def run_s2_precision(stack, variant: str, domain: str = "medical",
                     max_chunks: int = 200,
                     max_judge: int = PRECISION_MAX_JUDGE_CALLS) -> dict:
    """DeepSeek judges each extracted entity: real proper noun / specific
    term actually appearing in the chunk?"""
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())[:max_chunks]
    res = CorpusResources([t for _, t in chunk_items])
    extract = _extractor(stack, res, variant)

    instances: list[dict] = []
    for i, (cid, text) in enumerate(chunk_items):
        r = extract(text)
        for e in r["entities"]:
            instances.append({"cid": cid, "entity": e,
                              "multiword": " " in e})
        if (i + 1) % 50 == 0:
            print(f"    [{domain}] extract {i+1}/{len(chunk_items)}",
                  flush=True)

    rng = np.random.default_rng(0)
    if len(instances) > max_judge:
        idx = rng.choice(len(instances), size=max_judge, replace=False)
        judged = [instances[int(i)] for i in idx]
    else:
        judged = instances

    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _judge(inst):
        llm = DeepSeekProvider()
        text = corpus[inst["cid"]]  # full chunk (~1200 chars): entities
        # near the end must not be judged "not in text" unfairly
        prompt = (
            f"Text: {text}\n\n"
            f"Term: {inst['entity']}\n\n"
            f"Is \"{inst['entity']}\" a specific named entity (person, "
            f"place, organization, work title) or a specific technical / "
            f"proper term that actually appears verbatim in the text "
            f"above? Generic common nouns, topic words, or words NOT "
            f"appearing in the text do NOT count. Answer YES or NO only."
        )
        try:
            msg = llm.complete(prompt, max_tokens=10)
            resp = msg.content if hasattr(msg, "content") else str(msg)
            verdict = resp.strip().upper().startswith("YES")
        except Exception:
            verdict = None
        return {**inst, "judge": verdict}

    results_j: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_judge, inst) for inst in judged]
        done = 0
        for f in as_completed(futures):
            results_j.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"    judge {done}/{len(judged)}", flush=True)

    valid = [r for r in results_j if r["judge"] is not None]
    n_yes = sum(1 for r in valid if r["judge"])
    precision = n_yes / max(1, len(valid))
    return {
        "domain": domain, "n_chunks": len(chunk_items),
        "n_instances": len(instances), "n_judged": len(valid),
        "n_yes": n_yes, "precision": round(precision, 3),
        "gate": PRECISION_GATE, "passed": precision >= PRECISION_GATE,
        "judged_sample": results_j[:60],
    }


def granularity_comparison(domain: str, entity_samples: dict,
                           n_show: int = 8) -> list[dict]:
    """Side-by-side: topic-concept cache (twopass) vs entity extraction on
    the same chunks — the granularity gap Phase 50 hit."""
    cache_path = CACHE_DIR / f"concept_cache_{domain}_twopass.json"
    if not cache_path.exists():
        return []
    topic = json.loads(cache_path.read_text())
    rows = []
    for cid, ents in list(entity_samples.items())[:n_show]:
        rows.append({
            "cid": cid,
            "topic_concepts": topic["chunks"].get(cid, {}).get("concepts"),
            "entities": ents,
        })
    return rows


def run_s2(stack, variant: str, max_chunks: int = 200) -> dict:
    print(f"\n{'='*70}\nS2: capability verdict (variant={variant})\n{'='*70}")
    cov_m = run_s2_coverage(stack, variant, "medical")
    cov_n = run_s2_coverage(stack, variant, "novel")
    prec = run_s2_precision(stack, variant, "medical",
                            max_chunks=max_chunks)

    samples = {}
    for t in cov_m["targets"]:
        samples.update(t["extractions"])
    contrast = granularity_comparison("medical", samples)
    for row in contrast:
        print(f"  granularity [{row['cid']}]")
        print(f"    topic:   {row['topic_concepts']}")
        print(f"    entity:  {row['entities']}")

    # Gate per the task: a single line over the whole missing-entity list
    # (≥60% recovered), plus precision ≥70%. Per-domain numbers reported.
    n_rec = cov_m["n_recovered"] + cov_n["n_recovered"]
    n_tot = cov_m["n_targets"] + cov_n["n_targets"]
    coverage_overall = round(n_rec / max(1, n_tot), 3)
    passed = coverage_overall >= COVERAGE_GATE and prec["passed"]
    out = {
        "variant": variant,
        "coverage_overall": coverage_overall,
        "coverage_medical": cov_m, "coverage_novel": cov_n,
        "precision_medical": prec,
        "granularity_contrast": contrast,
        "passed": passed,
    }
    print(f"\nS2 verdict: coverage overall={coverage_overall} "
          f"(medical={cov_m['coverage']} novel={cov_n['coverage']}, "
          f"gate {COVERAGE_GATE}), "
          f"precision={prec['precision']} (gate {PRECISION_GATE}) "
          f"→ {'PASS' if passed else 'FAIL'}")
    return out


# ── S3: 全量提取（断点续跑） ─────────────────────────────────────────────


def run_s3(stack, variant: str, domain: str, max_chunks: int = 0) -> dict:
    out_path = CACHE_DIR / f"entity_cache_{domain}.json"
    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    chunk_items = list(corpus.items())
    if max_chunks > 0:
        chunk_items = chunk_items[:max_chunks]

    cache: dict
    if out_path.exists():
        cache = json.loads(out_path.read_text())
        cache.setdefault("chunks", {})
    else:
        cache = {
            "domain": domain,
            "model": detect_model()["name"],
            "prompt_variant": variant,
            "pipeline": ("phase52 entity (explicit-naming question + "
                         "sentence-end multi-position deep-layer readout + "
                         "stability/chunk-local verification + BM25 "
                         "completion + bigram merge/extend)"),
            "chunks": {},
        }
    todo = [(cid, t) for cid, t in chunk_items
            if cid not in cache["chunks"]]
    print(f"\n  [{domain}] {len(chunk_items)} chunks total, "
          f"{len(cache['chunks'])} cached, {len(todo)} to extract")

    if todo:
        res = CorpusResources([t for _, t in chunk_items])
        extract = _extractor(stack, res, variant)
        t_start = time.perf_counter()
        times = []
        for i, (cid, text) in enumerate(todo):
            t0 = time.perf_counter()
            r = extract(text)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            cache["chunks"][cid] = {
                "entities": r["entities"],
                "n_entities": r["n_entities"],
                "n_multiword": r["n_multiword"],
                "text_excerpt": text[:100],
                "extraction_time_s": round(t1 - t0, 4),
            }
            done = len(cache["chunks"])
            if done % 100 == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / (i + 1) * (len(todo) - i - 1)
                cache["partial"] = True
                out_path.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"    [{domain}] {done}/{len(chunk_items)} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
        cache["avg_per_chunk_s"] = round(float(np.mean(times)), 4)

    cache["n_chunks"] = len(cache["chunks"])
    entity_chunks: dict[str, list[str]] = defaultdict(list)
    freq: Counter = Counter()
    n_multi = 0
    for cid, cdata in cache["chunks"].items():
        n_multi += cdata.get("n_multiword", 0)
        for e in cdata["entities"]:
            el = e.lower()
            entity_chunks[el].append(cid)
            freq[el] += 1
    cache["entity_chunks"] = dict(entity_chunks)
    cache["entity_frequency"] = dict(freq.most_common(40))
    cache["n_unique_entities"] = len(freq)
    cache["n_multiword_instances"] = n_multi
    cache["partial"] = len(cache["chunks"]) < len(chunk_items)
    out_path.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"  [{domain}] done: {cache['n_chunks']} chunks, "
          f"{cache['n_unique_entities']} unique entities "
          f"({cache['n_multiword_instances']} multiword instances)")
    print(f"  [{domain}] top entities: {list(freq.most_common(15))}")
    print(f"  saved → {out_path}")
    return {"domain": domain, "n_chunks": cache["n_chunks"],
            "n_unique_entities": cache["n_unique_entities"],
            "n_multiword_instances": cache["n_multiword_instances"],
            "top_entities": list(freq.most_common(20)),
            "avg_per_chunk_s": cache.get("avg_per_chunk_s"),
            "cache_path": str(out_path)}


# ── S4: 终审（实体入图，重跑 phase50 ah 臂） ─────────────────────────────


def build_augmented_index(domain: str):
    """LightRAG 索引 + 实体节点（与原概念节点共存）。

    概念侧复用 phase50 的 merge_entities/_stem 合并；实体节点按 _stem
    分组（无 ws 向量，跳过余弦合并）；contains 边 = concept_chunks ∪
    entity_chunks；关系边只存在于概念-概念之间（实体-实体关系暂缺，
    按任务书先靠概念层关系）。
    """
    from experiments import phase50_lightrag_j as p50
    from experiments.phase39_two_pass_cache import _stem
    from experiments.phase41_retrieval_equivalence import load_phase41_inputs

    cache, vecs, relations = load_phase41_inputs(domain)
    ent_cache = json.loads(
        (CACHE_DIR / f"entity_cache_{domain}.json").read_text())

    # ── concept-side entities (identical to p50.LightRagIndex) ──
    concepts = [str(c) for c in vecs["concepts"]]
    groups = p50.merge_entities(concepts, vecs["ws_vec"])
    entities: list[dict] = []
    canon_of: dict[str, int] = {}
    for g in groups:
        members = [concepts[i] for i in g]
        rep = max(g, key=lambda i: (int(vecs["count"][i]), -i))
        chunks: set[str] = set()
        for m in members:
            chunks.update(cache["concept_chunks"].get(m, []))
        entities.append({"name": concepts[rep], "members": members,
                         "chunks": sorted(chunks), "roles": set(),
                         "text": "", "kind": "concept"})
        for m in members:
            canon_of[m] = len(entities) - 1
    for cid, cdata in cache["chunks"].items():
        for raw_concept, role_list in (cdata.get("roles") or {}).items():
            idx = canon_of.get(raw_concept.lower())
            if idx is not None:
                entities[idx]["roles"].update(role_list)

    # ── entity nodes (new) ──
    n_concept_nodes = len(entities)
    stem_groups: dict[str, list[str]] = defaultdict(list)
    for name in ent_cache["entity_chunks"]:
        stem_groups[_stem(name)].append(name)
    for _st, names in sorted(stem_groups.items()):
        rep = max(names, key=lambda n: len(ent_cache["entity_chunks"][n]))
        chunks: set[str] = set()
        for n in names:
            chunks.update(ent_cache["entity_chunks"][n])
        entities.append({"name": rep, "members": sorted(names),
                         "chunks": sorted(chunks), "roles": set(),
                         "text": rep, "kind": "entity"})

    for ent in entities:
        ent["roles"] = sorted(ent["roles"])
        if not ent["text"]:
            ent["text"] = (f"{ent['name']}: {', '.join(ent['roles'])}"
                           if ent["roles"] else ent["name"])

    # ── relations: concept-concept only ──
    rel_list: list[dict] = []
    adj: dict[int, set[int]] = defaultdict(set)
    for e in (relations or {"edges": []})["edges"]:
        a = canon_of.get(e["concept_a"].lower())
        b = canon_of.get(e["concept_b"].lower())
        if a is None or b is None or a == b:
            continue
        rel_list.append({
            "a": a, "b": b, "relation": str(e["relation"]),
            "prob": float(e["prob"]),
            "text": f"{entities[a]['name']} {e['relation']} "
                    f"{entities[b]['name']}",
            "completed": False,
        })
        adj[a].add(b)
        adj[b].add(a)

    class _Index:
        pass
    index = _Index()
    index.entities = entities
    index.relations = rel_list
    index.adj = {k: sorted(v) for k, v in adj.items()}
    index.canon_of = canon_of
    index.n_concept_nodes = n_concept_nodes
    index.ent_emb = None
    index.rel_emb = None

    def build_embeddings(embed_fn):
        ent = np.asarray(embed_fn([e["text"] for e in entities]),
                         dtype=np.float64)
        index.ent_emb = ent / np.where(
            np.linalg.norm(ent, axis=1, keepdims=True) > 0,
            np.linalg.norm(ent, axis=1, keepdims=True), 1.0)
        if rel_list:
            rel = np.asarray(embed_fn([r["text"] for r in rel_list]),
                             dtype=np.float64)
            index.rel_emb = rel / np.where(
                np.linalg.norm(rel, axis=1, keepdims=True) > 0,
                np.linalg.norm(rel, axis=1, keepdims=True), 1.0)
        else:
            index.rel_emb = np.zeros((0, ent.shape[1]))
    index.build_embeddings = build_embeddings
    return index, cache


def run_s4(domain: str, max_queries: int = 0) -> dict:
    """Final verdict: rerun phase50's ah arm with the entity-augmented
    graph. b0/factor reused from phase50 (same questions, same judge
    chain) — only the graph channel changes."""
    from experiments import phase50_lightrag_j as p50
    from experiments.phase41_retrieval_equivalence import (
        load_corpus_texts, TOP_K,
    )
    from experiments.phase26_acc_eval import (
        generate_answer, judge_answer_correctness,
    )

    phase50 = json.loads((EXP / "phase50_lightrag_j.json").read_text())
    p50_summary = phase50["domains"][domain]["summary"]
    factor = p50_summary["factor"]
    lb_lightrag = p50_summary["leaderboard"]["lightrag_acc_pct"]
    p50_ah = p50_summary["arms"].get("ah", {})

    index, cache = build_augmented_index(domain)
    corpus = load_corpus_texts(domain, cache)
    n_q = p50.FULL_QUERIES[domain] if max_queries == 0 else max_queries
    _corpus, questions = load_graphrag_bench(domain, n_q)

    from experiments.embed_cache import CachedBgeM3Provider
    embed_fn = CachedBgeM3Provider().embed

    chunk_ids = sorted(cache["chunks"].keys())
    chunk_emb = np.asarray(
        embed_fn([corpus[cid] for cid in chunk_ids]), dtype=np.float64)
    chunk_emb = chunk_emb / np.where(
        np.linalg.norm(chunk_emb, axis=1, keepdims=True) > 0,
        np.linalg.norm(chunk_emb, axis=1, keepdims=True), 1.0)

    print(f"  [{domain}] building embeddings for "
          f"{len(index.entities)} entities "
          f"({index.n_concept_nodes} concept + "
          f"{len(index.entities) - index.n_concept_nodes} entity nodes), "
          f"{len(index.relations)} relations...")
    index.build_embeddings(embed_fn)

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    # Phase 1: retrieval — ah arm (graph/naive round-robin interleave,
    # identical to phase50's arm "ah")
    contexts: list[dict] = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi] / (np.linalg.norm(query_emb[qi]) + 1e-12)
        q_sims = chunk_emb @ qv
        b0_ids = [chunk_ids[j] for j in np.argsort(-q_sims)[:TOP_K]]
        bge_sim = {chunk_ids[j]: float(q_sims[j])
                   for j in range(len(chunk_ids))}
        scores, dbg = p50.lightrag_retrieve(index, qv)
        graph_ranked = [cid for cid, _s in sorted(
            scores.items(), key=lambda x: (x[1], bge_sim.get(x[0], 0.0)),
            reverse=True)]
        merged: list[str] = []
        pools = [iter(graph_ranked), iter(b0_ids)]
        while len(merged) < TOP_K:
            for pool in pools:
                if len(merged) >= TOP_K:
                    break
                for cid in pool:
                    if cid not in merged:
                        merged.append(cid)
                        break
        contexts.append({
            "qid": q.get("id", str(qi)), "level": q.get("level"),
            "question": q["question"], "answer": q.get("answer", ""),
            "ranked_ah": merged, "debug": dbg,
        })
        if (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)

    # Phase 2: DeepSeek answer + judge (same chain as phase50)
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi_entry):
        qi, entry = qi_entry
        llm = DeepSeekProvider()
        ctx = " ".join(corpus[cid] for cid in entry["ranked_ah"])
        ans = generate_answer(entry["question"], ctx, llm)
        acc = bool(judge_answer_correctness(
            entry["question"], ans, entry["answer"], llm))
        return qi, {"qid": entry["qid"], "level": entry["level"],
                    "question": entry["question"], "acc": acc,
                    "answer": ans[:300], "ranked": entry["ranked_ah"]}

    per_query: list[dict | None] = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_eval, (i, e)) for i, e in enumerate(contexts)]
        done = 0
        for f in as_completed(futures):
            qi, rec = f.result()
            per_query[qi] = rec
            done += 1
            if done % 10 == 0:
                print(f"    eval {done}/{len(contexts)}", flush=True)
    per_query = [r for r in per_query if r is not None]

    accs = [1.0 if r["acc"] else 0.0 for r in per_query]
    acc = float(np.mean(accs)) if accs else None
    by_level = {}
    for lv in p50.LEVELS:
        lv_accs = [1.0 if r["acc"] else 0.0
                   for r in per_query if r["level"] == lv]
        by_level[lv] = {"acc": float(np.mean(lv_accs)) if lv_accs else None,
                        "n": len(lv_accs)}
    retention = (acc * factor * 100.0 / lb_lightrag
                 if acc is not None else None)
    verdict = ("retained(>=0.9)" if retention is not None and retention >= 0.9
               else "not_retained")

    out = {
        "domain": domain,
        "arm": "ah_with_entities",
        "n_entities_total": len(index.entities),
        "n_concept_nodes": index.n_concept_nodes,
        "n_entity_nodes": len(index.entities) - index.n_concept_nodes,
        "n_relations": len(index.relations),
        "acc": acc, "by_level": by_level, "n": len(per_query),
        "factor_from_phase50": factor,
        "lb_lightrag": lb_lightrag,
        "retention_vs_leaderboard_lightrag": retention,
        "verdict": verdict,
        "phase50_ah_baseline": {
            "acc": p50_ah.get("acc"),
            "retention": p50_ah.get("retention_vs_leaderboard_lightrag"),
        },
        "per_query": per_query,
    }
    print(f"\n  [{domain}] ah+entities: ACC={acc:.3f} "
          f"retention={retention:.3f} → {verdict}")
    print(f"  [{domain}] phase50 ah baseline: "
          f"ACC={p50_ah.get('acc'):.3f} "
          f"retention={p50_ah.get('retention_vs_leaderboard_lightrag'):.3f}")
    return out


# ── 主流程与门控 ─────────────────────────────────────────────────────────


def _load_out() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {}


def _save_out(out: dict) -> None:
    OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  saved → {OUT_PATH}")


def resolve_variant(args, out: dict) -> str:
    if args.variant:
        return args.variant
    s1 = out.get("s1", {})
    if s1.get("winner"):
        return s1["winner"]
    raise SystemExit(
        "No prompt variant available: run --stage s1 first or pass "
        "--variant {v1,v2}")


def _load_s4_out() -> dict:
    if S4_OUT_PATH.exists():
        return json.loads(S4_OUT_PATH.read_text())
    return {}


def main() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(
        description="Phase 52: J-Lens 专名级实体提取")
    ap.add_argument("--stage", default="all",
                    choices=["s1", "s2", "s3", "s4", "all"])
    ap.add_argument("--domain", default="all",
                    choices=["medical", "novel", "all"])
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="0 = 全量（S1 冒烟默认 20，S2 精度默认 200）")
    ap.add_argument("--smoke", action="store_true",
                    help="S1 冒烟模式（20 chunks，打印 5 个）")
    ap.add_argument("--variant", default=None, choices=["v1", "v2"],
                    help="覆盖 S1 获胜变体（默认读 phase52 json 的 s1.winner）")
    args = ap.parse_args()

    out = _load_out()
    out["method"] = "phase52_entity_extraction"
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stages = ["s1", "s2", "s3", "s4"] if args.stage == "all" else [args.stage]
    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]

    stack = None
    if any(s in ("s1", "s2", "s3") for s in stages):
        print("[init] Loading model + lens...")
        stack = load_model_stack()

    if "s1" in stages:
        s1 = run_s1(stack, domain="medical",
                    max_chunks=args.max_chunks or 20, n_print=5)
        out["s1"] = s1
        _save_out(out)

    if "s2" in stages:
        variant = resolve_variant(args, out)
        s2 = run_s2(stack, variant, max_chunks=args.max_chunks or 200)
        out["s2"] = s2
        _save_out(out)
        if not s2["passed"]:
            out["conclusion"] = {
                "verdict": "FALSIFIED",
                "reason": ("S2 gate failed — J-Lens entity-level readout "
                           "does not meet coverage/precision thresholds; "
                           "S3/S4 skipped. See s2 for details."),
            }
            _save_out(out)
            print("\nS2 gate FAILED — stopping before S3/S4. "
                  "Entity-level extraction: FALSIFIED at this gate.")
            return

    if "s3" in stages:
        variant = resolve_variant(args, out)
        out.setdefault("s3", {})
        for domain in domains:
            s3 = run_s3(stack, variant, domain, max_chunks=args.max_chunks)
            out["s3"][domain] = s3
            _save_out(out)

    if "s4" in stages:
        s4_all = _load_s4_out()
        s4_all["method"] = "phase52_lightrag_j_entities"
        s4_all["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        s4_all.setdefault("domains", {})
        for domain in domains:
            cache_path = CACHE_DIR / f"entity_cache_{domain}.json"
            if not cache_path.exists():
                print(f"  [{domain}] entity cache missing — run s3 first; "
                      f"skipping s4 for {domain}")
                continue
            print(f"\n── S4 final verdict: {domain} ──")
            s4 = run_s4(domain, max_queries=0)
            s4_all["domains"][domain] = s4
            S4_OUT_PATH.write_text(
                json.dumps(s4_all, indent=2, ensure_ascii=False))
            print(f"  saved → {S4_OUT_PATH}")
        if s4_all["domains"]:
            print(f"\n{'='*70}\nS4 终审对比表（ah 臂保持率，有/无实体）"
                  f"\n{'='*70}")
            print(f"  {'domain':<10} {'phase50 ah':>12} "
                  f"{'ah+entities':>12} {'verdict':>18}")
            for domain, s4 in s4_all["domains"].items():
                base = s4["phase50_ah_baseline"]["retention"]
                new = s4["retention_vs_leaderboard_lightrag"]
                print(f"  {domain:<10} {base:>12.3f} {new:>12.3f} "
                      f"{s4['verdict']:>18}")
            out["s4_summary"] = {
                d: {
                    "phase50_ah_retention": s["phase50_ah_baseline"]
                    ["retention"],
                    "ah_entities_retention":
                        s["retention_vs_leaderboard_lightrag"],
                    "verdict": s["verdict"],
                } for d, s in s4_all["domains"].items()
            }
            _save_out(out)


if __name__ == "__main__":
    main()
