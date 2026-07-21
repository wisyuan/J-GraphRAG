"""Phase 53: 文本侧实体检测 + 图增强 + LightRAG-J 终审。

背景：Phase 52 结构性 FALSIFIED——Qwen2.5-7B 4bit 对低频专名
（Dozmare/Excalibur/Princess Frederica/Baron von Pawel-Rammingen）不形成
可读出的 workspace 表示，J-Lens 实体级提取无解。但这些专名**显式存在于
文本中**，检测不需要模型生成。本实验改用纯文本侧规则（零模型）提取实体
入图，重跑 LightRAG-J 混合臂（phase50 ah 口径）做保持率终审。

S1 文本侧实体检测（CPU，零模型）
  - 大写 span：连续大写开头词序列（1-4 个实词，允许 of/von/der 等内部
    连接词）；句首假阳性用「小写形式在语料中的频率 vs 句中大写频率」排除
    （"The Princess" 的 The 句中大写次数≈0 且小写高频 → 剥掉；Princess
    句中大写高频 → 保留）。
  - 术语 span（medical 补充）：小写多词名词短语（2-3 词），语料频率 ≥2，
    尾词动词/虚词黑名单过滤。
  - 过滤：停用词、纯数字、长度<3、罗马数字；与 twopass 概念词表
    （_stem 归并后）去重。
  - 产出 entity_cache_textside_{domain}.json（entity_chunks/entity_frequency，
    结构对齐 twopass 的 concept_chunks/concept_frequency）。
  - 质检：Phase 52 失败清单检出率（medical 4 + novel 5 个目标实体，
    它们就在文本里，应接近 100%）+ 各域 top-20 高频实体人工判断。

S2 关系读出对罕见实体的有效性（GPU，小规模）
  假设：模型读不出罕见专名 token，但能读出两个罕见实体间的**关系词**
  （关系词是常见词汇）。从 novel 取含罕见实体的共现 chunk 构造 8-10 个
  实体对（罕见-罕见、罕见-常见），用 phase27 build_relation_prompt 原样
  读出，DeepSeek judge 判正确性。≥60% → 成立（S3 实体间关系走 J-Lens
  读出）；<60% → 如实记录，S3 走共现默认边。

S3 图增强 + 终审（DeepSeek judge）
  实体节点入图（实体→entity_chunks contains 边）；实体-实体边按 S2 判决
  （成立=J-Lens 读出 top 实体对，否则共现默认边）；实体-概念边=共现。
  重跑 phase50 的 ah 臂（图/naive 交错混合，medical 56 + novel 48），
  换算系数复用 phase50 的 factor（medical 1.005 / novel 0.885）。
  终审判决：保持率 ≥0.9 双双达标？对照 phase50 ah = medical 0.842 /
  novel 0.900。

运行：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -c "import experiments.phase53_textside_entities"   # 零副作用
    python -m experiments.phase53_textside_entities --stage s1 --domain both
    python -m experiments.phase53_textside_entities --stage s2
    python -m experiments.phase53_textside_entities --stage s3 --domain both
    python -m experiments.phase53_textside_entities --stage all --domain both
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase39_two_pass_cache import _stem
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, TOP_K,
)
from experiments.phase27_relation_readout import (
    build_relation_prompt, decode_topk,
)
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.phase50_lightrag_j import (
    LightRagIndex, lightrag_retrieve, leaderboard_mean,
    LEADERBOARD_ACC, LEVELS, FULL_QUERIES, ENT_TOP_K, REL_TOP_K,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"
OUT_PATH = EXP / "phase53_textside_entities.json"

# phase50 终审换算系数（同查询集同 judge 的 B0 锚点，本实验复用不重算）
PHASE50_FACTOR = {"medical": 1.005, "novel": 0.885}
# phase50 对照数字（保持率终审的基准行）
PHASE50_B0 = {"medical": 0.607, "novel": 0.542}
PHASE50_AH = {"medical": {"acc": 0.536, "retention": 0.842},
              "novel": {"acc": 0.458, "retention": 0.900}}

# Phase 52 失败清单（S1 质检目标——它们显式存在于文本，应接近 100% 检出）
P52_CHECKLIST = {
    "medical": ["basal cell carcinoma", "fair skin", "organ transplant",
                "immune suppression"],
    "novel": ["princess frederica", "baron von pawel-rammingen", "arthur",
              "excalibur", "dozmare"],
}

# ── S1 检测规则常量 ────────────────────────────────────────────────────
WORD_RE = re.compile(r"[A-Za-z]+(?:[-'’][A-Za-z]+)*|\d[\d,.]*")
# 句子边界：句读/换行后大写（Gutenberg 文本换行即段落）
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…;:\"”’)])\s+|\n+")

CONNECTORS = {"of", "the", "de", "von", "van", "der", "den", "la", "le",
              "di", "da", "del", "du", "des", "y", "and", "mac", "san"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "once", "here", "there", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did", "doing",
    "would", "could", "shall", "may", "might", "must", "of", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "our", "you",
    "your", "i", "me", "my", "him", "this", "that", "these", "those",
    "what", "which", "who", "whom", "how", "why", "where", "as", "us",
    "said", "say", "says", "one", "two", "also", "upon", "shall", "let",
    "mr", "mrs", "miss", "dr", "st",
}

# 句首/单词大写假阳性黑名单（小写高频规则之外的兜底）
CAP_BLOCKLIST = {
    "the", "this", "that", "these", "those", "there", "here", "then",
    "thus", "hence", "when", "where", "what", "which", "while", "who",
    "how", "why", "yes", "no", "indeed", "however", "moreover",
    "furthermore", "nevertheless", "meanwhile", "afterwards", "chapter",
    "section", "part", "book", "volume", "page", "contents", "index",
    "introduction", "preface", "appendix", "illustration", "fig",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday",
    "i", "it", "he", "she", "we", "they", "you", "but", "and", "for",
    "nor", "yet", "so", "as", "if", "though", "although", "because",
    "since", "until", "unless", "after", "before", "now", "soon", "later",
    "perhaps", "certainly", "surely", "really", "truly", "actually",
}

ROMAN_RE = re.compile(r"^[IVXLCDM]+$")

# 缩略形式（I'll/Don’t 等非实体）
CONTRACTION_RE = re.compile(
    r"^(i|you|he|she|it|we|they|that|there|who|what|let|don|can|won|isn|"
    r"aren|didn|doesn|couldn|wouldn|shouldn|mustn|ain|shan)['’]", re.I)

# Gutenberg 样板/话语标记假阳性（单词大写 span 兜底黑名单）
CAP_BLOCKLIST |= {
    "footnote", "footnotes", "illustrations", "transcriber", "gutenberg",
    "well", "ver", "ebook", "ebooks", "www", "http", "https",
}

# 术语 span 尾词黑名单（动词/虚词/泛名词，无 POS tagger 的规则近似）
TERM_TAIL_BLOCK = STOPWORDS | {
    "get", "gets", "getting", "got", "make", "makes", "making", "made",
    "take", "takes", "taking", "took", "taken", "use", "uses", "using",
    "used", "know", "knows", "known", "find", "finds", "found", "ask",
    "asks", "tell", "tells", "told", "go", "goes", "going", "gone",
    "come", "comes", "came", "see", "sees", "saw", "seen", "look",
    "looks", "looked", "want", "wants", "need", "needs", "needed",
    "keep", "keeps", "help", "helps", "give", "gives", "given", "gave",
    "work", "works", "worked", "call", "called", "feel", "feels", "felt",
    "seem", "seems", "seemed", "become", "becomes", "became", "include",
    "includes", "including", "included", "follow", "follows", "following",
    "followed", "develop", "develops", "developing", "developed",
    "cause", "causes", "causing", "caused", "treat", "treats", "treated",
    "way", "ways", "thing", "things", "time", "times", "day", "days",
    "year", "years", "people", "person", "man", "men", "woman", "women",
    "something", "anything", "everything", "nothing", "lot", "lots",
    "kind", "kinds", "type", "types", "number", "part", "parts",
    "much", "many", "less", "least", "well", "better", "best", "worse",
    "worst", "long", "longer", "high", "higher", "highest", "low",
    "lower", "lowest", "early", "earlier", "late", "later", "new",
    "newer", "old", "older", "good", "bad", "big", "small", "large",
    "able", "likely", "unlikely", "possible", "impossible", "common",
    "rare", "similar", "different", "important", "available", "present",
    "certain", "several", "various", "single", "whole", "full", "total",
    "main", "major", "minor", "general", "specific", "particular",
    "according", "depending", "based", "related", "compared", "due",
    "per", "via", "etc", "eg", "ie",
}

MAX_CAP_WORDS = 4        # 大写 span 最多 4 个实词（连接词不计）
TERM_MIN_FREQ = 2        # 术语 span 语料频率下限
S1_TOP_PRINT = 20        # 质检打印的高频实体数


# ── S1 文本侧实体检测 ──────────────────────────────────────────────────


def _tokenize_sentences(text: str) -> list[list[tuple[str, bool]]]:
    """切句 + 切词。返回 [[(token, is_sentence_initial), ...], ...]。"""
    sentences = []
    for sent in SENT_SPLIT_RE.split(text):
        toks = WORD_RE.findall(sent)
        if not toks:
            continue
        sentences.append([(t, i == 0) for i, t in enumerate(toks)])
    return sentences


def _is_cap_word(tok: str) -> bool:
    return (tok[0].isupper() and any(c.isalpha() for c in tok)
            and not CONTRACTION_RE.match(tok))


def _strip_possessive(tok: str) -> str:
    """去掉词尾所有格/孤立引号：Sterne’s → Sterne。"""
    t = re.sub(r"['’]s$", "", tok, flags=re.I)
    return t.rstrip("'’") or tok


def _word_stats(sentences: list[list[tuple[str, bool]]]) -> tuple[Counter, Counter]:
    """逐词统计：小写出现次数、句中大写出现次数（非句首）。"""
    n_lower: Counter = Counter()
    n_cap_mid: Counter = Counter()
    for sent in sentences:
        for tok, is_initial in sent:
            if not any(c.isalpha() for c in tok):
                continue
            if tok.islower():
                n_lower[tok] += 1
            elif _is_cap_word(tok) and not is_initial:
                n_cap_mid[tok.lower()] += 1
    return n_lower, n_cap_mid


def _strip_sentence_initial(run: list[str], n_lower: Counter,
                            n_cap_mid: Counter) -> list[str]:
    """剥掉句首假阳性词：小写形式高频且句中从不大写的词。"""
    out = list(run)
    while out:
        w = out[0]
        wl = w.lower()
        if wl in CONNECTORS:
            out.pop(0)
            continue
        if n_cap_mid.get(wl, 0) == 0 and n_lower.get(wl, 0) >= 2:
            out.pop(0)
            continue
        break
    return out


def _valid_cap_span(words: list[str]) -> bool:
    if not words:
        return False
    real = [w for w in words if w.lower() not in CONNECTORS]
    if not real:
        return False
    text = " ".join(words)
    if not any(c.isalpha() for c in text):
        return False
    if sum(c.isalpha() for c in text) < 3:
        return False
    if all(w.lower() in STOPWORDS for w in real):
        return False
    if len(real) == 1:
        w = real[0]
        if w.lower() in CAP_BLOCKLIST or w.lower() in STOPWORDS:
            return False
        if ROMAN_RE.match(w):
            return False
        if len(w) < 3 and not w.isupper():
            return False
    return True


def extract_cap_spans(sentences: list[list[tuple[str, bool]]],
                      n_lower: Counter, n_cap_mid: Counter) -> list[str]:
    """提取大写 span（允许内部连接词），含句首假阳性剥离。"""
    spans: list[str] = []
    for sent in sentences:
        toks = [t for t, _ in sent]
        i = 0
        while i < len(toks):
            if not _is_cap_word(toks[i]):
                i += 1
                continue
            run = [toks[i]]
            n_real = 1
            j = i + 1
            while j < len(toks) and n_real < MAX_CAP_WORDS:
                if _is_cap_word(toks[j]):
                    run.append(toks[j])
                    n_real += 1
                    j += 1
                elif (toks[j].lower() in CONNECTORS and j + 1 < len(toks)
                      and _is_cap_word(toks[j + 1]) and n_real < MAX_CAP_WORDS):
                    run.append(toks[j])
                    j += 1
                else:
                    break
            # 去掉结尾悬挂的连接词
            while run and run[-1].lower() in CONNECTORS:
                run.pop()
            run = [_strip_possessive(w) for w in run]
            run = _strip_sentence_initial(run, n_lower, n_cap_mid)
            if _valid_cap_span(run):
                spans.append(" ".join(run))
            i = max(j, i + 1)
    return spans


def extract_term_spans(
    sentences: list[list[tuple[str, bool]]],
) -> tuple[Counter, set[str]]:
    """小写多词名词短语候选（2-3 词）+ 出现过首字母大写的 n-gram 集合。

    返回 (candidates, cap_grams)：后者用于捞出 freq-1 但以大写形式出现
    的标题型术语（如 "Immune suppression – People with ..." 的小节标题）。
    """
    cand: Counter = Counter()
    cap_grams: set[str] = set()
    for sent in sentences:
        toks_l = [t.lower() for t, _ in sent]
        toks_raw = [t for t, _ in sent]
        for n in (3, 2):
            for i in range(len(toks_l) - n + 1):
                gram = toks_l[i:i + n]
                if not all(re.fullmatch(r"[a-z]+", g) for g in gram):
                    continue
                if any(g in STOPWORDS or g in CONNECTORS for g in gram):
                    continue
                if gram[-1] in TERM_TAIL_BLOCK or gram[0] in TERM_TAIL_BLOCK:
                    continue
                if any(len(g) < 3 for g in gram):
                    continue
                key = " ".join(gram)
                cand[key] += 1
                if any(g[0].isupper() for g in toks_raw[i:i + n]):
                    cap_grams.add(key)
    return cand, cap_grams


def detect_entities(domain: str, corpus: dict[str, str],
                    concept_vocab: set[str]) -> dict:
    """S1 主流程：对全语料跑规则检测，返回实体缓存结构。"""
    print(f"  [{domain}] tokenizing {len(corpus)} chunks...", flush=True)
    chunk_sents = {cid: _tokenize_sentences(text)
                   for cid, text in corpus.items()}

    # 全语料词统计（句首假阳性判定 + 术语频率）
    all_sents = [s for sents in chunk_sents.values() for s in sents]
    n_lower, n_cap_mid = _word_stats(all_sents)

    # 概念词表归一化（_stem 逐词归并）用于去重
    def _norm(s: str) -> str:
        return " ".join(_stem(w) for w in s.lower().split())
    concept_norm = {_norm(c) for c in concept_vocab}

    entity_chunks: dict[str, set[str]] = defaultdict(set)
    entity_surface: dict[str, Counter] = defaultdict(Counter)
    entity_type: dict[str, str] = {}

    for cid, sents in chunk_sents.items():
        for span in extract_cap_spans(sents, n_lower, n_cap_mid):
            key = span.lower()
            if _norm(key) in concept_norm:
                continue
            entity_chunks[key].add(cid)
            entity_surface[key][span] += 1
            entity_type[key] = "capitalized"

    if domain == "medical":
        term_cand, cap_grams = extract_term_spans(all_sents)
        # freq>=2 保留；freq==1 仅当以大写形式出现过（标题型术语，
        # 如 "Immune suppression"——语料仅 1 处但确为领域术语）
        kept = {g: c for g, c in term_cand.items()
                if c >= TERM_MIN_FREQ or g in cap_grams}
        for gram, cnt in kept.items():
            entity_surface[gram][gram] += cnt
        # 重新定位每个保留术语出现的 chunk
        for cid, sents in chunk_sents.items():
            text_l = " ".join(t for s in sents for t, _ in s).lower()
            for gram in kept:
                if f" {gram} " in f" {text_l} ":
                    if _norm(gram) in concept_norm:
                        continue
                    entity_chunks[gram].add(cid)
                    entity_type[gram] = "term"

    entity_freq = {e: sum(surf.values())
                   for e, surf in entity_surface.items() if entity_chunks[e]}
    entity_chunks = {e: sorted(cids) for e, cids in entity_chunks.items() if cids}
    entity_freq = {e: f for e, f in entity_freq.items() if e in entity_chunks}
    display = {e: surf.most_common(1)[0][0]
               for e, surf in entity_surface.items() if e in entity_chunks}

    # 质检 1：Phase 52 失败清单检出率（包含式匹配：
    # "Princess Frederica of Hanover" 覆盖 "princess frederica"）
    checklist = {}
    for target in P52_CHECKLIST.get(domain, []):
        hit = any(target == e or target in e or e in target
                  for e in entity_chunks)
        checklist[target] = hit
    n_hit = sum(checklist.values())
    rate = n_hit / len(checklist) if checklist else None

    top = sorted(entity_freq.items(), key=lambda x: -x[1])[:S1_TOP_PRINT]
    print(f"  [{domain}] entities={len(entity_chunks)} "
          f"(cap={sum(1 for t in entity_type.values() if t == 'capitalized')}, "
          f"term={sum(1 for t in entity_type.values() if t == 'term')})")
    print(f"  [{domain}] Phase52 checklist: {n_hit}/{len(checklist)} "
          f"→ {rate:.0%}" if rate is not None else "")
    for t, ok in checklist.items():
        print(f"    {'✓' if ok else '✗'} {t}")
    print(f"  [{domain}] top-{S1_TOP_PRINT} frequent entities (人工质检):")
    for e, f in top:
        print(f"    {f:5d}  {display[e]}")

    return {
        "domain": domain,
        "n_chunks": len(corpus),
        "n_entities": len(entity_chunks),
        "entity_chunks": entity_chunks,
        "entity_frequency": entity_freq,
        "entity_display": display,
        "entity_type": entity_type,
        "qc": {
            "phase52_checklist": checklist,
            "checklist_hit_rate": rate,
            "top_entities": [[display[e], f] for e, f in top],
        },
    }


def run_s1(domains: list[str]) -> dict:
    out = {}
    for domain in domains:
        cache, _vecs, _rel = load_phase41_inputs(domain)
        corpus = load_corpus_texts(domain, cache)
        result = detect_entities(domain, corpus, set(cache["concept_chunks"]))
        path = CACHE_DIR / f"entity_cache_textside_{domain}.json"
        path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
        print(f"  saved → {path}")
        out[domain] = result["qc"] | {"n_entities": result["n_entities"]}
    return out


# ── S2 关系读出对罕见实体的有效性 ──────────────────────────────────────

# novel 罕见实体候选（S2 实体对池；从 Phase 52 失败清单 + 亚瑟王传奇主线）
RARE_POOL = ["dozmare", "excalibur", "arthur", "king arthur", "launcelot",
             "sir launcelot", "guenever", "queen guenever", "merlin",
             "princess frederica", "baron von pawel-rammingen", "cornwall",
             "tintagel", "camelot", "mordred", "sir bedivere", "cornwall"]
# 目标实体对（罕见-罕见、罕见-常见搭配；实际存在性按共现 chunk 过滤）
S2_TARGET_PAIRS = [
    ("excalibur", "dozmare"),                    # 罕见-罕见
    ("princess frederica", "baron von pawel-rammingen"),  # 罕见-罕见
    ("excalibur", "arthur"),                     # 罕见-常见
    ("arthur", "launcelot"),
    ("arthur", "guenever"),
    ("arthur", "merlin"),
    ("launcelot", "guenever"),
    ("excalibur", "launcelot"),
    ("arthur", "cornwall"),
    ("merlin", "cornwall"),
]


def _find_cochunk(corpus: dict[str, str], a: str, b: str) -> str | None:
    for cid, text in corpus.items():
        tl = text.lower()
        if a in tl and b in tl:
            return text
    return None


def judge_relation_words(text: str, a: str, b: str,
                         words: list[str], llm) -> tuple[bool, str]:
    """DeepSeek judge：读出的关系词是否准确描述文本中 A-B 的关系。"""
    prompt = (
        f"Text excerpt:\n{text[:1200]}\n\n"
        f"In this text, what is the relationship between \"{a}\" and "
        f"\"{b}\"? A model proposed these candidate relation words: "
        f"{', '.join(words)}.\n\n"
        f"Does ANY of these words correctly characterize the actual "
        f"relationship between \"{a}\" and \"{b}\" as described in the "
        f"text? Answer with only YES or NO."
    )
    try:
        msg = llm.complete(prompt, max_tokens=10)
        resp = msg.content if hasattr(msg, "content") else str(msg)
        return resp.strip().upper().startswith("YES"), resp.strip()
    except Exception as e:
        return False, f"[ERROR: {e}]"


def run_s2() -> dict:
    from jgraphrag.llm import DeepSeekProvider
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )

    cache, _vecs, _rel = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)

    pairs = []
    for a, b in S2_TARGET_PAIRS:
        text = _find_cochunk(corpus, a, b)
        if text is not None:
            pairs.append((a, b, text))
        else:
            print(f"  skip (no co-occurrence chunk): {a} + {b}")
    print(f"  {len(pairs)} entity pairs with co-occurring chunks")

    print("  loading model + lens (GPU)...", flush=True)
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    layer = lens.source_layers[-1]

    llm = DeepSeekProvider()
    results = []
    n_correct = 0
    for a, b, text in pairs:
        prompt = build_relation_prompt(text, a, b, tokenizer)
        lens_logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                                       positions=[-1], max_seq_len=512)
        words = decode_topk(lens_logits[layer][0], tokenizer, n=5)
        cand_words = [w["token"] for w in words]
        ok, raw = judge_relation_words(text, a, b, cand_words, llm)
        n_correct += ok
        print(f"  [{'✓' if ok else '✗'}] {a} + {b} → {cand_words}")
        results.append({"a": a, "b": b, "relation_words": words,
                        "judge": ok, "judge_raw": raw,
                        "doc_excerpt": text[:200]})

    acc = n_correct / len(results) if results else 0.0
    verdict = acc >= 0.6
    print(f"\n  S2 判决: {n_correct}/{len(results)} = {acc:.0%} "
          f"→ 关系读出对罕见实体{'成立 (≥60%)' if verdict else '不成立 (<60%)'}")
    return {"domain": "novel", "model": cand["name"], "pairs": results,
            "n_pairs": len(results), "n_correct": n_correct,
            "accuracy": acc, "gate": 0.6,
            "readout_valid_for_rare_entities": verdict}


# ── S3 图增强 + 终审 ───────────────────────────────────────────────────

MAX_EE_EDGES = 20000     # 实体-实体共现默认边上限
MAX_EC_EDGES = 20000     # 实体-概念共现边上限
MIN_COOCC = 2            # 共现边最小次数（freq-1 实体对噪声过大）
S2_READOUT_TOP_PAIRS = 100  # S2 成立时 J-Lens 读出的 top 实体对数


def _norm_concept(s: str) -> str:
    return " ".join(_stem(w) for w in s.lower().split())


def augment_index(index: LightRagIndex, entity_cache: dict,
                  readout_edges: list[dict] | None = None) -> dict:
    """把文本侧实体节点与边挂进 LightRagIndex（就地修改）。返回统计。"""
    base_n = len(index.entities)
    ent_chunks = entity_cache["entity_chunks"]
    display = entity_cache["entity_display"]
    ent_freq = entity_cache["entity_frequency"]
    new_idx: dict[str, int] = {}

    for name in sorted(ent_chunks):
        new_idx[name] = len(index.entities)
        index.entities.append({
            "name": display.get(name, name), "members": [name],
            "chunks": sorted(ent_chunks[name]), "roles": [],
            "text": display.get(name, name),
        })

    adj = defaultdict(set)
    for a, nbs in index.adj.items():
        adj[a].update(nbs)

    # 实体-实体边
    n_ee = 0
    if readout_edges is not None:
        # S2 成立：J-Lens 读出的 top 实体对边
        for e in readout_edges:
            a, b = new_idx.get(e["a"]), new_idx.get(e["b"])
            if a is None or b is None or a == b:
                continue
            index.relations.append({
                "a": a, "b": b, "relation": e["relation"],
                "prob": e["prob"],
                "text": f"{index.entities[a]['name']} {e['relation']} "
                        f"{index.entities[b]['name']}",
                "completed": True,
            })
            adj[a].add(b)
            adj[b].add(a)
            n_ee += 1
    else:
        # 共现默认边：chunk 内实体对（每 chunk 取全局频率 top-15 控制组合爆炸）
        cooc: Counter = Counter()
        cid_ents: dict[str, list[str]] = defaultdict(list)
        for name, cids in ent_chunks.items():
            for cid in cids:
                cid_ents[cid].append(name)
        for cid, names in cid_ents.items():
            names = sorted(names, key=lambda n: -ent_freq.get(n, 0))[:15]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    cooc[(names[i], names[j])] += 1
        pairs = [(c, a, b) for (a, b), c in cooc.items() if c >= MIN_COOCC]
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:MAX_EE_EDGES]
        c_max = pairs[0][0] if pairs else 1
        for c, a, b in pairs:
            ia, ib = new_idx[a], new_idx[b]
            index.relations.append({
                "a": ia, "b": ib, "relation": "related_to",
                "prob": c / c_max,
                "text": f"{index.entities[ia]['name']} related to "
                        f"{index.entities[ib]['name']}",
                "completed": True,
            })
            adj[ia].add(ib)
            adj[ib].add(ia)
            n_ee += 1

    # 实体-概念共现边（按 chunk 正向展开，避免 entity×concept 全对枚举）
    chunk_concepts: dict[str, list[int]] = defaultdict(list)
    for ci in range(base_n):
        for cid in index.entities[ci]["chunks"]:
            chunk_concepts[cid].append(ci)
    ec_cooc: Counter = Counter()
    cid_ents_all: dict[str, list[str]] = defaultdict(list)
    for name, cids in ent_chunks.items():
        for cid in cids:
            cid_ents_all[cid].append(name)
    for cid, names in cid_ents_all.items():
        cis = chunk_concepts.get(cid)
        if not cis:
            continue
        for name in names:
            for ci in cis:
                ec_cooc[(name, ci)] += 1
    ec_items = [(k, v) for k, v in ec_cooc.items() if v >= MIN_COOCC]
    ec_pairs = sorted(ec_items, key=lambda x: -x[1])[:MAX_EC_EDGES]
    ec_max = ec_pairs[0][1] if ec_pairs else 1
    for (name, ci), ov in ec_pairs:
        ei = new_idx[name]
        index.relations.append({
            "a": ei, "b": ci, "relation": "related_to",
            "prob": ov / ec_max,
            "text": f"{index.entities[ei]['name']} related to "
                    f"{index.entities[ci]['name']}",
            "completed": True,
        })
        adj[ei].add(ci)
        adj[ci].add(ei)

    index.adj = {k: sorted(v) for k, v in adj.items()}
    return {"n_base_entities": base_n, "n_new_entities": len(new_idx),
            "n_ee_edges": n_ee, "n_ec_edges": len(ec_pairs)}


def jlens_readout_top_pairs(entity_cache: dict, top_n: int) -> list[dict]:
    """S2 成立时：对共现 top 实体对做 J-Lens 关系读出，产出带关系词的边。"""
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )

    ent_chunks = entity_cache["entity_chunks"]
    ent_freq = entity_cache["entity_frequency"]
    cooc: Counter = Counter()
    cid_ents: dict[str, list[str]] = defaultdict(list)
    for name, cids in ent_chunks.items():
        for cid in cids:
            cid_ents[cid].append(name)
    chunk_text: dict[str, str] = {}
    for cid, names in cid_ents.items():
        names = sorted(names, key=lambda n: -ent_freq.get(n, 0))[:15]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cooc[(names[i], names[j], cid)] += 1
    pair_best: dict[tuple[str, str], tuple[int, str]] = {}
    for (a, b, cid), c in cooc.items():
        key = (a, b)
        if key not in pair_best or c > pair_best[key][0]:
            pair_best[key] = (c, cid)
    top = sorted(pair_best.items(), key=lambda x: -x[1][0])[:top_n]

    # 需要 chunk 原文
    cache, _v, _r = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)

    print(f"  J-Lens readout for top-{len(top)} entity pairs (GPU)...",
          flush=True)
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    layer = lens.source_layers[-1]

    edges = []
    c_max = top[0][1][0] if top else 1
    for (a, b), (c, cid) in top:
        text = corpus[cid]
        prompt = build_relation_prompt(text, a, b, tokenizer)
        lens_logits, _, _ = lens.apply(lens_model, prompt, layers=[layer],
                                       positions=[-1], max_seq_len=512)
        words = decode_topk(lens_logits[layer][0], tokenizer, n=3)
        rel = words[0]["token"].lower() if words else "related_to"
        edges.append({"a": a, "b": b, "relation": rel, "prob": c / c_max})
    # 释放 GPU：后续 bge 嵌入需要显存
    del lens_model, model
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return edges


def run_s3_domain(domain: str, s2_verdict: dict | None,
                  max_queries: int = 0, verbose: bool = True) -> dict:
    """概念+实体增强图的 ah 臂（phase50 口径）+ DeepSeek 终审。"""
    cache, vecs, relations = load_phase41_inputs(domain)
    corpus = load_corpus_texts(domain, cache)
    n_q = FULL_QUERIES[domain] if max_queries == 0 else max_queries
    _corpus, questions = load_graphrag_bench(domain, n_q)

    entity_path = CACHE_DIR / f"entity_cache_textside_{domain}.json"
    entity_cache = json.loads(entity_path.read_text())

    # J-Lens 读出先行：此时 bge 尚未加载，GPU 空闲；读出后立即释放显存。
    # 实体间边：S2 成立（仅 novel 有 GPU 读出产物）→ J-Lens 读出；
    # 否则共现默认边
    readout_edges = None
    if (s2_verdict and s2_verdict.get("readout_valid_for_rare_entities")
            and domain == "novel"):
        readout_edges = jlens_readout_top_pairs(
            entity_cache, S2_READOUT_TOP_PAIRS)

    from experiments.embed_cache import CachedBgeM3Provider
    embed_fn = CachedBgeM3Provider().embed

    chunk_ids = sorted(cache["chunks"].keys())
    chunk_emb = np.asarray(
        embed_fn([corpus[cid] for cid in chunk_ids]), dtype=np.float64)
    chunk_emb = chunk_emb / np.where(
        np.linalg.norm(chunk_emb, axis=1, keepdims=True) > 0,
        np.linalg.norm(chunk_emb, axis=1, keepdims=True), 1.0)

    index = LightRagIndex(cache, vecs, relations)
    n_base_rel = len(index.relations)

    stats = augment_index(index, entity_cache, readout_edges)
    stats["edge_mode"] = ("jlens_readout" if readout_edges is not None
                          else "cooccurrence_default")
    if verbose:
        print(f"  [{domain}] {len(chunk_ids)} chunks, "
              f"{stats['n_base_entities']} concept-entities "
              f"+ {stats['n_new_entities']} text-side entities, "
              f"{n_base_rel} base relations "
              f"+{stats['n_ee_edges']} ee +{stats['n_ec_edges']} ec "
              f"({stats['edge_mode']}), {len(questions)} queries", flush=True)

    index.build_embeddings(embed_fn)
    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    # Phase 1: retrieval —— phase50 ah 臂（图/naive round-robin 交错）
    contexts = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi] / (np.linalg.norm(query_emb[qi]) + 1e-12)
        q_sims = chunk_emb @ qv
        b0_ids = [chunk_ids[j] for j in np.argsort(-q_sims)[:TOP_K]]
        bge_sim = {chunk_ids[j]: float(q_sims[j])
                   for j in range(len(chunk_ids))}
        scores, dbg = lightrag_retrieve(index, qv)
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
        contexts.append({"qid": q.get("id", str(qi)), "level": q.get("level"),
                         "question": q["question"],
                         "answer": q.get("answer", ""), "ranked": merged,
                         "debug": dbg})
        if verbose and (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)

    # Phase 2: DeepSeek answer + judge（phase26 链，与 phase50 同构）
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi_entry):
        qi, entry = qi_entry
        llm = DeepSeekProvider()
        ctx = " ".join(corpus[cid] for cid in entry["ranked"])
        ans = generate_answer(entry["question"], ctx, llm)
        acc = bool(judge_answer_correctness(
            entry["question"], ans, entry["answer"], llm))
        return qi, {"qid": entry["qid"], "level": entry["level"],
                    "question": entry["question"], "acc": acc,
                    "answer": ans[:300], "ranked": entry["ranked"]}

    per_query: list[dict | None] = [None] * len(contexts)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_eval, (i, e)) for i, e in enumerate(contexts)]
        done = 0
        for future in as_completed(futures):
            qi, rec = future.result()
            per_query[qi] = rec
            done += 1
            if verbose and done % 10 == 0:
                print(f"    eval {done}/{len(contexts)}", flush=True)
    per_query = [r for r in per_query if r is not None]

    accs = [1.0 if r["acc"] else 0.0 for r in per_query]
    acc = float(np.mean(accs)) if accs else None
    by_level = {}
    for lv in LEVELS:
        lv_accs = [1.0 if r["acc"] else 0.0
                   for r in per_query if r["level"] == lv]
        by_level[lv] = {"acc": float(np.mean(lv_accs)) if lv_accs else None,
                        "n": len(lv_accs)}

    factor = PHASE50_FACTOR[domain]
    lb_lightrag = leaderboard_mean(domain, "LightRAG")
    retention = acc * factor * 100.0 / lb_lightrag if acc is not None else None
    verdict = ("retained(>=0.9)" if retention is not None and retention >= 0.9
               else "not_retained")
    if verbose:
        print(f"  [{domain}] entity-augmented ah: ACC={acc:.3f} "
              f"retention={retention:.3f} (factor={factor}, "
              f"lb={lb_lightrag:.1f}%) → {verdict}")
    return {"domain": domain, "arm": "ah_entity_graph", "acc": acc,
            "by_level": by_level, "n": len(per_query),
            "factor": factor, "leaderboard_lightrag_acc_pct": lb_lightrag,
            "retention_vs_leaderboard_lightrag": retention,
            "verdict": verdict, "graph_stats": stats,
            "n_base_relations": n_base_rel, "per_query": per_query}


def run_s3(domains: list[str], out: dict, max_queries: int = 0) -> dict:
    """逐域运行并立即落盘（单域崩了不丢另一域的结果）。"""
    s2_verdict = out.get("s2")
    if s2_verdict is None:
        print("  warn: no persisted S2 verdict found; "
              "entity-entity edges default to co-occurrence")
    s3 = out.setdefault("s3", {})
    for domain in domains:
        print(f"\n── S3 {domain} ──")
        s3[domain] = run_s3_domain(domain, s2_verdict, max_queries)
        out["final"] = build_final_table(out)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  saved [{domain}] → {OUT_PATH}", flush=True)
    return s3


# ── 汇总与终审判决 ─────────────────────────────────────────────────────


def build_final_table(out: dict) -> dict:
    """终审对比表：B0 / 概念图 ah / 概念+实体图 ah × 两域。"""
    table = {}
    for domain, s3 in out.get("s3", {}).items():
        p50 = PHASE50_AH[domain]
        b0 = PHASE50_B0[domain]
        factor = PHASE50_FACTOR[domain]
        lb = leaderboard_mean(domain, "LightRAG")
        table[domain] = {
            "b0_phase50": {
                "acc": b0,
                "retention": b0 * factor * 100.0 / lb},
            "concept_graph_ah_phase50": {
                "acc": p50["acc"], "retention": p50["retention"]},
            "concept_plus_entity_graph_ah_phase53": {
                "acc": s3["acc"],
                "retention": s3["retention_vs_leaderboard_lightrag"],
                "by_level": {lv: v["acc"]
                             for lv, v in s3["by_level"].items()},
                "graph_stats": s3["graph_stats"],
                "verdict": s3["verdict"]},
        }
    both = all(
        t["concept_plus_entity_graph_ah_phase53"]["retention"] is not None
        and t["concept_plus_entity_graph_ah_phase53"]["retention"] >= 0.9
        for t in table.values()) if len(table) == 2 else False
    verdict = (
        "文本侧实体补齐了 LightRAG-J 的保持率缺口（双域 ≥0.9）"
        if both else
        "文本侧实体未能（完全）补齐 LightRAG-J 的保持率缺口——见终审表")
    return {"table": table, "both_domains_retained": both,
            "one_line_verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 53 文本侧实体检测 + 图增强 + LightRAG-J 终审")
    ap.add_argument("--stage", default="all",
                    choices=["s1", "s2", "s3", "all"])
    ap.add_argument("--domain", default="both",
                    choices=["medical", "novel", "both"])
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = 全量（medical 56 / novel 48），调试用")
    args = ap.parse_args()

    domains = ["medical", "novel"] if args.domain == "both" else [args.domain]
    out = {}
    if OUT_PATH.exists():
        out = json.loads(OUT_PATH.read_text())
    out.setdefault("method", "phase53_textside_entities")
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.stage in ("s1", "all"):
        print("── S1 文本侧实体检测 ──")
        out["s1"] = run_s1(domains)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if args.stage in ("s2", "all"):
        print("── S2 关系读出对罕见实体的有效性 ──")
        out["s2"] = run_s2()
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if args.stage in ("s3", "all"):
        run_s3(domains, out, max_queries=args.max_queries)
        out["final"] = build_final_table(out)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\n── 终审 ──")
        print(json.dumps(out["final"], indent=2, ensure_ascii=False))
    print(f"saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
