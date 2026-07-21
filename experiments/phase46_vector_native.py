"""Phase 46: J-GraphRAG 向量原生能力探针（纯 CPU / 网络，禁止加载 Qwen）。

核心问题（与 prompt 读出路线对偶）：J-Lens 用 prompt 读出的能力，有多少
本来就线性可及地存在于概念几何（ws 向量 / M 矩阵 / W 图）里？

四个探针：
  P1 几何消歧：词形对（same，_stem 构造）/ 近义对（near，DeepSeek 预标注）
     / 无关对（diff，随机 + DeepSeek 确认）三分类。特征：(a) ws 余弦、
     (b) M 行余弦（语境分布相似度，M 构建口径镜像 phase41 build_m_matrix）、
     (c) 联合（逻辑回归 + 留一交叉验证）。基线：bge 余弦（同测试对）。
     判决：三分类准确率 / 宏 F1，几何 vs bge。
     （义项检测加分项跳过：npz 只有均值向量，无 per-chunk ws 向量。）
  P2 几何属性：top-15 DF 概念，chunk 集合内共现的次概念/角色词按
     频次 × ws 余弦 排序取 top-8 作为"几何属性"；DeepSeek judge precision；
     对照臂用 bge 余弦替代 ws。判决：precision + 两臂重合度。
  P3 概念代数：A:B::C:? 在 ws 空间（vec(B)-vec(A)+vec(C) top-k）。
     测试例：relations 同关系类型边对 + 词形代数（tumor:tumors ::
     surgery:surgeries 类）。判决：top-1/top-5 命中率 vs 随机基线。
  P4 多跳评分：A→B→C 真路径 vs 伪路径（A→B 边存在、B→C 不存在）。
     评分：W 幂次传播 (W²)[A,C]、ws 链式余弦乘积、vec(A)+vec(C) 与
     2vec(B) 的距离。判决：区分真/伪路径的 AUC。

复用（只 import 不改；均为函数级 lazy import，模块 import 本身零重依赖）：
  - phase39_two_pass_cache._stem（词形归并）
  - phase41_retrieval_equivalence 的 M 矩阵口径（此处镜像实现，见
    build_m_matrix docstring）
  - embed_cache.CachedBgeM3Provider（bge 基线，强制 CPU）
  - jgraphrag.llm.DeepSeekProvider（judge，用法同 phase26_acc_eval）

DeepSeek 预算：P1 预标注 ~4 次 + P2 judge ~15 次 ≈ 20 次（上限 200，
LLMBudget 硬截断）。

运行：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.phase46_vector_native --probe all --domain medical
    python -m experiments.phase46_vector_native --probe p1   # 单探针
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# bge 基线强制 CPU（GPU 被另一条实验线占用）；必须在 jgraphrag 首次
# import 前设定，因此所有 jgraphrag / experiments 重依赖均为函数级 lazy import。
os.environ.setdefault("LINCLE_BGE_M3_DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"
OUT_PATH = EXP / "phase46_vector_native.json"

BM25_K1 = 1.2
BM25_B = 0.75

# P1/P3 词形对不足 8 对时的下限（任务书允许放宽到 ≥5 并在报告注明）
MIN_PAIRS_RELAXED = 5
MIN_PAIRS_TARGET = 8


# ── 数据加载（契约同 phase41.load_phase41_inputs） ──────────────────────


def load_inputs(domain: str, cache_dir: Path = CACHE_DIR) -> tuple[dict, dict, dict]:
    cache_path = cache_dir / f"concept_cache_{domain}_twopass.json"
    vecs_path = cache_dir / f"concept_vecs_{domain}.npz"
    rel_path = cache_dir / f"relations_{domain}.json"
    for p in (cache_path, vecs_path, rel_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (phase39/40 outputs)")
    cache = json.loads(cache_path.read_text())
    npz = np.load(vecs_path, allow_pickle=False)
    ws = npz["ws_vec"].astype(np.float64)
    ws = ws / np.where(np.linalg.norm(ws, axis=1, keepdims=True) > 0,
                       np.linalg.norm(ws, axis=1, keepdims=True), 1.0)
    vecs = {
        "concepts": [str(c) for c in npz["concepts"]],
        "ws_vec": ws,
        "wu_vec": npz["wu_vec"].astype(np.float64),
        "count": npz["count"].astype(int),
    }
    relations = json.loads(rel_path.read_text())
    return cache, vecs, relations


def load_chunk_texts(domain: str, cache: dict) -> dict[str, str]:
    """Full chunk texts keyed by cache chunk ids (phase41.load_corpus_texts)."""
    from experiments.phase4_dig_graphragbench import load_graphrag_bench

    corpus, _ = load_graphrag_bench(domain, max_queries=1)
    return {cid: t for cid, t in corpus.items() if cid in cache["chunks"]}


def build_m_matrix(
    cache: dict, chunk_texts: dict[str, str], row_concepts: list[str],
) -> tuple[np.ndarray, list[str]]:
    """M[i,j] = IDF(concept_i) x BM25_tf(concept_i, chunk_j).

    Mirrors phase41.build_m_matrix's formula (itself = phase38.build_matrices):
    IDF is the phase25 smoothed log((N+1)/(df+1))+1 over cache concept_chunks;
    BM25 tf = regex word-count in full chunk text (min 1, k1=1.2, b=0.75,
    length-normalized by avg chunk length). Rows follow row_concepts (npz
    vocab); columns are sorted cache chunk ids. phase41 builds rows for all
    graph concepts via ConceptGraph; here only the npz vocab rows are needed,
    same per-cell formula.
    """
    concept_chunks = cache["concept_chunks"]
    chunk_ids = sorted(cache["chunks"].keys())
    chunk_index = {cid: j for j, cid in enumerate(chunk_ids)}
    n_total = len(chunk_ids)
    idf = {c: math.log((n_total + 1) / (len(cids) + 1)) + 1
           for c, cids in concept_chunks.items()}
    # chunk lengths / avg (same as ConceptGraph.compute_tf)
    chunk_len = {cid: len(chunk_texts.get(cid, "").split()) for cid in chunk_ids}
    avg_len = sum(chunk_len.values()) / max(len(chunk_ids), 1)

    m = np.zeros((len(row_concepts), len(chunk_ids)), dtype=np.float64)
    for i, concept in enumerate(row_concepts):
        for cid in concept_chunks.get(concept, []):
            j = chunk_index.get(cid)
            if j is None:
                continue
            text = chunk_texts.get(cid, "").lower()
            tf = len(re.findall(r"\b" + re.escape(concept.lower()) + r"\b",
                                text)) or 1
            dl = chunk_len.get(cid, 0) or avg_len
            norm = dl / max(avg_len, 1)
            tf_w = tf * (BM25_K1 + 1) / (tf + BM25_K1 * (1 - BM25_B + BM25_B * norm))
            m[i, j] = idf.get(concept, 1.0) * tf_w
    return m, chunk_ids


def cos_matrix(vecs: np.ndarray) -> np.ndarray:
    """Pairwise cosine for L2-normalized rows (defensively re-normalized)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    v = vecs / np.where(norms > 0, norms, 1.0)
    return v @ v.T


def _stem(word: str) -> str:
    """Re-export of phase39_two_pass_cache._stem (lazy import, unmodified)."""
    from experiments.phase39_two_pass_cache import _stem as stem_impl

    return stem_impl(word)


# ── DeepSeek 预算器 ─────────────────────────────────────────────────────


class LLMBudget:
    """DeepSeek call wrapper with a hard budget (usage pattern of phase26)."""

    def __init__(self, max_calls: int = 200) -> None:
        self.max_calls = max_calls
        self.n_calls = 0
        self._llm = None

    def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        if self.n_calls >= self.max_calls:
            raise RuntimeError(
                f"DeepSeek budget exhausted ({self.max_calls} calls)")
        if self._llm is None:
            from jgraphrag.llm import DeepSeekProvider

            self._llm = DeepSeekProvider()
        self.n_calls += 1
        msg = self._llm.complete(prompt, max_tokens=max_tokens)
        content = msg.content if hasattr(msg, "content") else str(msg)
        if getattr(msg, "is_error", False):
            raise RuntimeError(f"DeepSeek error: {msg.error_message}")
        return content


def _parse_json_block(text: str):
    """Extract the first JSON array/object from an LLM reply."""
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# ── P1 几何消歧 ─────────────────────────────────────────────────────────


def morphological_pairs(concepts: list[str]) -> list[tuple[str, str]]:
    """Same-stem surface-variant pairs via phase39._stem (label=same)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in concepts:
        groups[_stem(c)].append(c)
    pairs = []
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.append((g[i], g[j]))
    return sorted(pairs)


def label_pairs_with_llm(
    pairs: list[tuple[str, str]], domain: str, llm: LLMBudget,
    batch_size: int = 20,
) -> dict[tuple[str, str], str]:
    """DeepSeek 预标注：NEAR（近义/强相关）vs DIFF（无关）。批量 JSON 输出。"""
    labeled: dict[tuple[str, str], str] = {}
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        lines = "\n".join(f"{k + 1}. \"{a}\" vs \"{b}\""
                          for k, (a, b) in enumerate(batch))
        prompt = (
            f"These are concept terms extracted from a {domain} corpus "
            "(cancer-information texts). For each pair, decide whether the "
            "two terms are NEAR (synonymous or strongly semantically related "
            "in this corpus, e.g. chemo/chemotherapy, tumor/carcinoma) or "
            "DIFF (distinct, unrelated concepts, e.g. tumor/pregnancy).\n\n"
            f"{lines}\n\n"
            'Reply with ONLY a JSON array of labels, one per pair, in order, '
            'e.g. ["NEAR", "DIFF", ...].'
        )
        resp = llm.complete(prompt, max_tokens=400)
        labels = _parse_json_block(resp)
        if not isinstance(labels, list) or len(labels) != len(batch):
            # fallback: mark all as unparsable → skip batch
            print(f"    WARNING: label batch {start} unparsable, skipped")
            continue
        for (a, b), lab in zip(batch, labels):
            lab = str(lab).strip().upper()
            if lab.startswith("NEAR"):
                labeled[(a, b)] = "near"
            elif lab.startswith("DIFF"):
                labeled[(a, b)] = "diff"
    return labeled


def _loo_logreg(X: np.ndarray, y: np.ndarray) -> dict:
    """LOO cross-validated balanced logistic regression → metrics dict."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score)
    from sklearn.model_selection import LeaveOneOut, cross_val_predict

    preds = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        X, y, cv=LeaveOneOut())
    rep = classification_report(y, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro",
                                   zero_division=0)),
        "per_class": {lab: {"precision": rep[lab]["precision"],
                            "recall": rep[lab]["recall"],
                            "f1": rep[lab]["f1-score"]}
                      for lab in ("same", "near", "diff") if lab in rep},
        "preds": preds,
    }


def probe_p1(domain: str, llm: LLMBudget, seed: int = 0,
             verbose: bool = True) -> dict:
    """P1 几何消歧：ws / M 行 / 联合 vs bge，三分类（same/near/diff）。"""
    cache, vecs, relations = load_inputs(domain)
    concepts = vecs["concepts"]
    n = len(concepts)
    ws_cos = cos_matrix(vecs["ws_vec"])
    rng = np.random.default_rng(seed)

    # ── 测试对构造 ──
    same_pairs = morphological_pairs(concepts)
    same_set = {frozenset(p) for p in same_pairs}

    # near 候选：ws 余弦中位区间（排除词形对）；diff 候选：随机低余弦对
    all_pairs = [(concepts[i], concepts[j])
                 for i in range(n) for j in range(i + 1, n)
                 if frozenset((concepts[i], concepts[j])) not in same_set]
    cos_vals = np.array([ws_cos[concepts.index(a), concepts.index(b)]
                         for a, b in all_pairs])
    order = np.argsort(-cos_vals)
    # near 候选：余弦最高的前 40（DeepSeek 过滤出真正近义对）
    near_cand = [all_pairs[k] for k in order[:40]]
    # diff 候选：余弦最低段随机取 30
    low = order[int(0.5 * len(order)):]
    low = low[rng.permutation(len(low))[:30]]
    diff_cand = [all_pairs[k] for k in low]

    if verbose:
        print(f"  P1 [{domain}] vocab={n}, morphological(same) pairs="
              f"{len(same_pairs)}: {same_pairs}")
        print(f"  P1 DeepSeek 预标注: {len(near_cand)} near 候选 + "
              f"{len(diff_cand)} diff 候选 "
              f"(~{math.ceil(len(near_cand)/20) + math.ceil(len(diff_cand)/20)} calls)")
    labeled = label_pairs_with_llm(near_cand + diff_cand, domain, llm)
    near_pairs = sorted([p for p, lab in labeled.items() if lab == "near"])
    diff_pairs = sorted([p for p, lab in labeled.items() if lab == "diff"])
    # diff 对数量与 near 对齐（不超过 2x near，避免类别失衡过大）
    if len(diff_pairs) > 2 * max(len(near_pairs), MIN_PAIRS_RELAXED):
        diff_pairs = [diff_pairs[k] for k in rng.permutation(len(diff_pairs))
                      [:2 * max(len(near_pairs), MIN_PAIRS_RELAXED)]]

    pairs = same_pairs + near_pairs + diff_pairs
    y = np.array(["same"] * len(same_pairs) + ["near"] * len(near_pairs)
                 + ["diff"] * len(diff_pairs))
    relaxed_note = None
    for label, plist in (("same", same_pairs), ("near", near_pairs),
                         ("diff", diff_pairs)):
        if len(plist) < MIN_PAIRS_TARGET:
            relaxed_note = (
                f"label '{label}' 只有 {len(plist)} 对 (<{MIN_PAIRS_TARGET})，"
                f"按任务书放宽到 ≥{MIN_PAIRS_RELAXED}")
            if len(plist) < MIN_PAIRS_RELAXED:
                relaxed_note += " —— 低于下限，结果仅供参考"
    if verbose:
        print(f"  P1 pairs: same={len(same_pairs)} near={len(near_pairs)} "
              f"diff={len(diff_pairs)}  ({relaxed_note or 'all ≥8'})")

    # ── 特征 ──
    cidx = {c: i for i, c in enumerate(concepts)}
    chunk_texts = load_chunk_texts(domain, cache)
    m, _cids = build_m_matrix(cache, chunk_texts, concepts)
    m_cos = cos_matrix(m)

    from experiments.embed_cache import CachedBgeM3Provider

    bge = np.asarray(CachedBgeM3Provider().embed(concepts), dtype=np.float64)
    bge_cos = cos_matrix(bge)

    def _feat(mat):
        return np.array([mat[cidx[a], cidx[b]] for a, b in pairs])

    f_ws, f_m, f_bge = _feat(ws_cos), _feat(m_cos), _feat(bge_cos)
    feats = {
        "ws_cos": f_ws.reshape(-1, 1),
        "m_row_cos": f_m.reshape(-1, 1),
        "joint_ws+m": np.column_stack([f_ws, f_m]),
        "bge_cos": f_bge.reshape(-1, 1),
    }
    results = {}
    for name, X in feats.items():
        r = _loo_logreg(X, y)
        results[name] = {k: v for k, v in r.items() if k != "preds"}
        if verbose:
            pc = r["per_class"]
            print(f"    {name:<12} LOO acc={r['accuracy']:.3f}  "
                  f"macroF1={r['macro_f1']:.3f}  "
                  + "  ".join(f"{lab}:F1={pc[lab]['f1']:.2f}"
                             for lab in ("same", "near", "diff")
                             if lab in pc))

    # 混淆明细（联合特征）
    joint_preds = _loo_logreg(feats["joint_ws+m"], y)["preds"]
    per_pair = [
        {"a": a, "b": b, "label": lab, "pred": pred,
         "ws_cos": round(float(w), 4), "m_cos": round(float(mm), 4),
         "bge_cos": round(float(bg), 4)}
        for (a, b), lab, pred, w, mm, bg in zip(
            pairs, y, joint_preds, f_ws, f_m, f_bge)
    ]
    # 每类特征均值（几何结构的直接刻画）
    class_means = {}
    for lab in ("same", "near", "diff"):
        mask = y == lab
        if mask.any():
            class_means[lab] = {
                "ws_cos": round(float(f_ws[mask].mean()), 4),
                "m_cos": round(float(f_m[mask].mean()), 4),
                "bge_cos": round(float(f_bge[mask].mean()), 4),
            }
    geo_best = max(("ws_cos", "m_row_cos", "joint_ws+m"),
                   key=lambda k: results[k]["macro_f1"])
    verdict = ("geometry_wins" if results[geo_best]["macro_f1"]
               > results["bge_cos"]["macro_f1"] else "bge_wins")
    if verbose:
        print(f"  P1 verdict: {verdict} (best geometry={geo_best} "
              f"F1={results[geo_best]['macro_f1']:.3f} vs bge "
              f"F1={results['bge_cos']['macro_f1']:.3f})")
    return {
        "n_pairs": {k: len(v) for k, v in
                    (("same", same_pairs), ("near", near_pairs),
                     ("diff", diff_pairs))},
        "relaxed_note": relaxed_note,
        "selection_note": (
            "near 候选取自 ws 余弦 top-40、diff 候选取自余弦下半区随机 30，"
            "再经 DeepSeek 预标注——候选构造本身以 ws 为条件，对 ws 臂存在"
            "选择偏差（near/diff 的 ws 余弦被拉近），解读需谨慎"),
        "polysemy_bonus": "skipped: npz 只有均值向量，无 per-chunk ws 向量",
        "class_means": class_means,
        "results": results, "verdict": verdict, "per_pair": per_pair,
    }


# ── P2 几何属性 ─────────────────────────────────────────────────────────


def geometric_attributes(
    c: str, cache: dict, vecs: dict, sim_vec: dict[str, float] | None,
    ws_cos: np.ndarray, cidx: dict[str, int], top_n: int = 8,
) -> list[dict]:
    """概念 c 的几何属性候选：chunk 集合内共现次概念 + c 的角色词。

    评分 = 共现频次 × 相似度。ws 臂：次概念直接用 ws 余弦；角色词无 ws
    向量，用"角色条件向量"= c 带该角色的 chunks 内共现概念的 ws 均值，
    再与 c 取余弦（纯几何，不用 bge）。bge 臂由调用方传入 sim_vec
    （candidate → bge 余弦）。
    """
    concept_chunks = cache["concept_chunks"]
    chunks = cache["chunks"]
    cids = concept_chunks.get(c, [])
    c_stem = _stem(c)
    vocab = set(vecs["concepts"])

    cooccur: Counter = Counter()
    roles: Counter = Counter()
    role_ctx: dict[str, Counter] = defaultdict(Counter)
    for cid in cids:
        entry = chunks.get(cid, {})
        others = {str(x).lower() for x in entry.get("concepts", [])} & vocab
        others.discard(c)
        others = {o for o in others if _stem(o) != c_stem}
        for o in others:
            cooccur[o] += 1
        role_map = entry.get("roles", {})
        for surface, rlist in role_map.items():
            if str(surface).lower() == c:
                for r in rlist:
                    r = str(r).strip().lower()
                    if not r or _stem(r) == c_stem:
                        continue
                    roles[r] += 1
                    for o in others:
                        role_ctx[r][o] += 1

    ws_c = vecs["ws_vec"][cidx[c]]
    scored = []
    for term, freq in cooccur.items():
        sim = float(ws_cos[cidx[term], cidx[c]]) if sim_vec is None \
            else sim_vec.get(term, 0.0)
        scored.append({"term": term, "kind": "concept", "freq": freq,
                       "sim": sim, "score": freq * sim})
    for r, freq in roles.items():
        if sim_vec is None:
            ctx = role_ctx[r]
            if ctx:
                vecs_stack = np.stack([vecs["ws_vec"][cidx[o]] for o in ctx])
                cond = vecs_stack.mean(axis=0)
                denom = (np.linalg.norm(cond) * np.linalg.norm(ws_c)) or 1.0
                sim = float(cond @ ws_c / denom)
            else:
                sim = 0.0
        else:
            sim = sim_vec.get(r, 0.0)
        scored.append({"term": r, "kind": "role", "freq": freq,
                       "sim": sim, "score": freq * sim})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_n]


def judge_attributes(
    c: str, terms: list[str], domain: str, llm: LLMBudget,
) -> dict[str, bool]:
    """DeepSeek 逐属性判定（一次调用判一个概念的全集，YES/NO）。"""
    lines = "\n".join(f"{k + 1}. {t}" for k, t in enumerate(terms))
    prompt = (
        f"Corpus: {domain} texts (cancer-information book). Concept: \"{c}\".\n"
        "Below are candidate attribute/aspect terms claimed to be genuine "
        "attributes of this concept AS USED in this corpus (e.g. for "
        "\"cancer\", \"diagnosis\" and \"chemotherapy\" are genuine; "
        "\"pregnancy\" is not).\n\n"
        f"{lines}\n\n"
        "For EACH numbered term reply YES (genuine attribute/aspect) or NO. "
        'Reply with ONLY a JSON array like ["YES", "NO", ...], in order.'
    )
    resp = llm.complete(prompt, max_tokens=300)
    labels = _parse_json_block(resp)
    out = {}
    if isinstance(labels, list) and len(labels) == len(terms):
        for t, lab in zip(terms, labels):
            out[t] = str(lab).strip().upper().startswith("YES")
    else:
        print(f"    WARNING: judge reply for '{c}' unparsable "
              f"({len(labels) if isinstance(labels, list) else '?'} vs "
              f"{len(terms)}), counting all as NO")
        out = {t: False for t in terms}
    return out


def probe_p2(domain: str, llm: LLMBudget, top_concepts: int = 15,
             top_n: int = 8, verbose: bool = True) -> dict:
    """P2 几何属性：ws 臂 vs bge 臂，DeepSeek judge precision + 重合度。"""
    cache, vecs, relations = load_inputs(domain)
    concepts = vecs["concepts"]
    cidx = {c: i for i, c in enumerate(concepts)}
    ws_cos = cos_matrix(vecs["ws_vec"])
    order = np.argsort(-vecs["count"])[:top_concepts]
    top = [concepts[i] for i in order]

    if verbose:
        print(f"  P2 [{domain}] top-{top_concepts} DF concepts: {top}")
        print(f"  P2 DeepSeek judge: {len(top)} calls "
              f"(每概念一次调用同时判两臂并集)")

    ws_attrs = {c: geometric_attributes(c, cache, vecs, None, ws_cos, cidx,
                                        top_n) for c in top}

    # bge 臂：同一候选池（全部共现次概念 + 角色词），用 bge 余弦重打分
    from experiments.embed_cache import CachedBgeM3Provider

    embed = CachedBgeM3Provider().embed
    full_attrs = {c: geometric_attributes(c, cache, vecs, None, ws_cos, cidx,
                                          top_n=10**9) for c in top}
    cand_strings = sorted({d["term"] for c in top for d in full_attrs[c]}
                          | set(top))
    cand_emb = np.asarray(embed(cand_strings), dtype=np.float64)
    cand_norm = cand_emb / np.where(
        np.linalg.norm(cand_emb, axis=1, keepdims=True) > 0,
        np.linalg.norm(cand_emb, axis=1, keepdims=True), 1.0)
    sidx = {s: i for i, s in enumerate(cand_strings)}

    bge_attrs = {}
    for c in top:
        sim_vec = {t: float(cand_norm[sidx[t]] @ cand_norm[sidx[c]])
                   for t in cand_strings if t != c}
        bge_attrs[c] = geometric_attributes(c, cache, vecs, sim_vec, ws_cos,
                                            cidx, top_n)

    # judge：每概念一次调用，判两臂并集
    per_concept = {}
    for c in top:
        terms_ws = [d["term"] for d in ws_attrs[c]]
        terms_bge = [d["term"] for d in bge_attrs[c]]
        union = list(dict.fromkeys(terms_ws + terms_bge))
        verdicts = judge_attributes(c, union, domain, llm)
        p_ws = (float(np.mean([verdicts[t] for t in terms_ws]))
                if terms_ws else None)
        p_bge = (float(np.mean([verdicts[t] for t in terms_bge]))
                 if terms_bge else None)
        overlap = len(set(terms_ws) & set(terms_bge)) / max(top_n, 1)
        per_concept[c] = {
            "df": int(vecs["count"][cidx[c]]),
            "ws_top": ws_attrs[c], "bge_top": bge_attrs[c],
            "judge": verdicts,
            "precision_ws": p_ws, "precision_bge": p_bge,
            "overlap_top8": overlap,
        }
        if verbose:
            print(f"    {c:<14} P_ws={p_ws:.2f} P_bge={p_bge:.2f} "
                  f"overlap={overlap:.2f}")

    agg = {
        "precision_ws": float(np.mean([v["precision_ws"] for v in
                                       per_concept.values()
                                       if v["precision_ws"] is not None])),
        "precision_bge": float(np.mean([v["precision_bge"] for v in
                                        per_concept.values()
                                        if v["precision_bge"] is not None])),
        "mean_overlap_top8": float(np.mean([v["overlap_top8"] for v in
                                            per_concept.values()])),
    }
    agg["verdict"] = ("ws_geometry_viable"
                      if agg["precision_ws"] >= agg["precision_bge"]
                      else "bge_better")
    if verbose:
        print(f"  P2 verdict: {agg['verdict']}  P_ws="
              f"{agg['precision_ws']:.3f} P_bge={agg['precision_bge']:.3f} "
              f"overlap={agg['mean_overlap_top8']:.3f}")
    return {"top_concepts": top, "aggregate": agg, "per_concept": per_concept}


# ── P3 概念代数 ─────────────────────────────────────────────────────────


def analogy_rank(
    a: str, b: str, c: str, ws: np.ndarray, cidx: dict[str, int],
    exclude: set[str],
) -> list[str]:
    """vec(B)-vec(A)+vec(C) 对词表余弦排序（排除 A/B/C），返回概念序列。"""
    v = ws[cidx[b]] - ws[cidx[a]] + ws[cidx[c]]
    norm = np.linalg.norm(v) or 1.0
    sims = ws @ (v / norm)
    concepts = list(cidx.keys())
    return [concepts[k] for k in np.argsort(-sims)
            if concepts[k] not in exclude]


def probe_p3(domain: str, seed: int = 0, max_rel_analogies: int = 40,
             verbose: bool = True) -> dict:
    """P3 概念代数：关系类比 + 词形代数，ws 空间 vec(B)-vec(A)+vec(C)。"""
    cache, vecs, relations = load_inputs(domain)
    concepts = vecs["concepts"]
    n = len(concepts)
    cidx = {c: i for i, c in enumerate(concepts)}
    ws = vecs["ws_vec"]
    rng = np.random.default_rng(seed)
    vocab_set = set(concepts)

    # 关系类比：同关系类型的有向边对 (A→B, C→D)，期望 D
    by_rel: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in relations["edges"]:
        a, b, r = e["concept_a"], e["concept_b"], e["relation"]
        if a in vocab_set and b in vocab_set and a != b:
            by_rel[r].append((a, b))
    rel_analogies = []
    for r, edges in sorted(by_rel.items()):
        for i in range(len(edges)):
            for j in range(len(edges)):
                if i == j:
                    continue
                (a, b), (c, d) = edges[i], edges[j]
                if len({a, b, c, d}) < 4:
                    continue
                rel_analogies.append(
                    {"a": a, "b": b, "c": c, "d": d, "kind": f"relation:{r}"})
    if len(rel_analogies) > max_rel_analogies:
        pick = rng.permutation(len(rel_analogies))[:max_rel_analogies]
        rel_analogies = [rel_analogies[k] for k in sorted(pick)]

    # 词形代数：词形对两两组合 (a:b :: c:d)，期望 d
    mpairs = morphological_pairs(concepts)
    morph_analogies = []
    for i in range(len(mpairs)):
        for j in range(len(mpairs)):
            if i == j:
                continue
            (a, b), (c, d) = mpairs[i], mpairs[j]
            if len({a, b, c, d}) < 4:
                continue
            morph_analogies.append(
                {"a": a, "b": b, "c": c, "d": d, "kind": "morph"})

    def _eval(analogies: list[dict]) -> dict:
        top1 = top5 = 0
        details = []
        for an in analogies:
            ranked = analogy_rank(an["a"], an["b"], an["c"], ws, cidx,
                                  {an["a"], an["b"], an["c"]})
            hit1 = ranked[0] == an["d"]
            hit5 = an["d"] in ranked[:5]
            top1 += hit1
            top5 += hit5
            details.append({**an, "top5": ranked[:5],
                            "hit1": bool(hit1), "hit5": bool(hit5)})
        m = max(len(analogies), 1)
        return {"n": len(analogies), "top1": top1 / m, "top5": top5 / m,
                "details": details}

    rel_res = _eval(rel_analogies)
    morph_res = _eval(morph_analogies)
    chance1 = 1.0 / (n - 3)
    chance5 = 5.0 / (n - 3)
    n_all = len(rel_analogies) + len(morph_analogies)
    overall = {
        "n": n_all,
        "top1": (rel_res["top1"] * rel_res["n"]
                 + morph_res["top1"] * morph_res["n"]) / max(n_all, 1),
        "top5": (rel_res["top5"] * rel_res["n"]
                 + morph_res["top5"] * morph_res["n"]) / max(n_all, 1),
    }
    out = {
        "n_vocab": n,
        "chance_top1": chance1, "chance_top5": chance5,
        "relation_analogies": rel_res,
        "morph_analogies": morph_res,
        "overall": overall,
    }
    out["verdict"] = ("linear_analogy_holds"
                      if overall["top5"] >= 3 * chance5
                      else "no_linear_analogy_structure")
    if verbose:
        print(f"  P3 [{domain}] relation analogies n={rel_res['n']} "
              f"top1={rel_res['top1']:.3f} top5={rel_res['top5']:.3f} | "
              f"morph n={morph_res['n']} top1={morph_res['top1']:.3f} "
              f"top5={morph_res['top5']:.3f} | chance5={chance5:.3f}")
        print(f"  P3 verdict: {out['verdict']}")
    return out


# ── P4 多跳评分 ─────────────────────────────────────────────────────────


def probe_p4(domain: str, seed: int = 0, max_paths: int = 40,
             verbose: bool = True) -> dict:
    """P4 多跳评分：真 A→B→C 路径 vs 伪路径，W² / ws 链式 / 中点距 AUC。"""
    from sklearn.metrics import roc_auc_score

    cache, vecs, relations = load_inputs(domain)
    concepts = vecs["concepts"]
    n = len(concepts)
    cidx = {c: i for i, c in enumerate(concepts)}
    ws = vecs["ws_vec"]
    ws_cos = cos_matrix(ws)
    rng = np.random.default_rng(seed)
    vocab_set = set(concepts)

    w = np.zeros((n, n), dtype=np.float64)
    for e in relations["edges"]:
        a, b = e["concept_a"], e["concept_b"]
        if a in vocab_set and b in vocab_set and a != b:
            w[cidx[a], cidx[b]] = max(w[cidx[a], cidx[b]], float(e["prob"]))
    edge_set = {(i, j) for i in range(n) for j in range(n) if w[i, j] > 0}

    # 真路径：A→B 且 B→C（A≠C）；伪路径：A→B 存在、B→C 不存在（且 A→C 无边）
    real, seen = [], set()
    for (i, j) in sorted(edge_set):
        for k in range(n):
            if k != i and w[j, k] > 0:
                key = (i, j, k)
                if key not in seen:
                    seen.add(key)
                    real.append(key)
    if len(real) > max_paths:
        real = [real[t] for t in sorted(rng.permutation(len(real))[:max_paths])]
    real_set = set(real)
    pseudo = []
    attempts = 0
    while len(pseudo) < len(real) and attempts < 10000:
        attempts += 1
        i, j = list(edge_set)[rng.integers(len(edge_set))]
        k = int(rng.integers(n))
        if k == i or w[j, k] > 0 or w[i, k] > 0 or (i, j, k) in real_set:
            continue
        if (i, j, k) in pseudo:
            continue
        pseudo.append((i, j, k))

    w2 = w @ w

    def _scores(paths):
        s_w2, s_chain, s_mid = [], [], []
        for (i, j, k) in paths:
            s_w2.append(float(w2[i, k]))
            s_chain.append(float(max(ws_cos[i, j], 0.0)
                                 * max(ws_cos[j, k], 0.0)))
            s_mid.append(float(-np.linalg.norm(ws[i] + ws[k] - 2 * ws[j])))
        return s_w2, s_chain, s_mid

    r_w2, r_chain, r_mid = _scores(real)
    p_w2, p_chain, p_mid = _scores(pseudo)
    y = [1] * len(real) + [0] * len(pseudo)
    aucs = {}
    for name, r_s, p_s in (("w2_propagation", r_w2, p_w2),
                           ("ws_chain_cos", r_chain, p_chain),
                           ("ws_midpoint_dist", r_mid, p_mid)):
        vals = r_s + p_s
        aucs[name] = (float(roc_auc_score(y, vals))
                      if len(set(vals)) > 1 else None)
    verdict = ("geometry_scores_paths"
               if (aucs["ws_chain_cos"] or 0) >= 0.7 else
               "w2_only" if (aucs["w2_propagation"] or 0) >= 0.7
               else "weak_discrimination")
    if verbose:
        print(f"  P4 [{domain}] real={len(real)} pseudo={len(pseudo)}  "
              f"AUC: " + "  ".join(f"{k}={v:.3f}" if v is not None
                                   else f"{k}=n/a" for k, v in aucs.items()))
        print(f"  P4 verdict: {verdict}")
    return {
        "n_real": len(real), "n_pseudo": len(pseudo), "auc": aucs,
        "verdict": verdict,
        "real_paths": [[concepts[i], concepts[j], concepts[k]]
                       for (i, j, k) in real],
        "pseudo_paths": [[concepts[i], concepts[j], concepts[k]]
                         for (i, j, k) in pseudo],
    }


# ── 输出合并 + CLI ──────────────────────────────────────────────────────


def merge_out(domain: str, probe: str, result: dict,
              out_path: Path = OUT_PATH) -> None:
    out = {}
    if out_path.exists():
        out = json.loads(out_path.read_text())
    out.setdefault("method", "phase46_vector_native")
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out.setdefault("probes", {}).setdefault(domain, {})[probe] = result
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"  saved {probe}/{domain} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 46 向量原生能力探针")
    ap.add_argument("--domain", default="medical",
                    choices=["medical", "novel"])
    ap.add_argument("--probe", default="all",
                    choices=["p1", "p2", "p3", "p4", "all"])
    ap.add_argument("--max-llm-calls", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    llm = LLMBudget(max_calls=args.max_llm_calls)
    probes = ["p1", "p2", "p3", "p4"] if args.probe == "all" else [args.probe]
    print(f"Phase 46 vector-native probes: domain={args.domain} "
          f"probes={probes} seed={args.seed}")
    print(f"DeepSeek 预算: ≤{args.max_llm_calls} calls "
          f"(预估 P1 ~6 + P2 ~15 ≈ 21; P3/P4 不用 LLM)")
    t0 = time.time()
    for p in probes:
        print(f"\n── {p.upper()} ──")
        if p == "p1":
            merge_out(args.domain, p, probe_p1(args.domain, llm, args.seed))
        elif p == "p2":
            merge_out(args.domain, p, probe_p2(args.domain, llm))
        elif p == "p3":
            merge_out(args.domain, p, probe_p3(args.domain, args.seed))
        elif p == "p4":
            merge_out(args.domain, p, probe_p4(args.domain, args.seed))
    print(f"\nDone in {time.time() - t0:.1f}s; DeepSeek calls used: "
          f"{llm.n_calls}")


if __name__ == "__main__":
    main()
