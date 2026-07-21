"""Phase 51: HippoRAG2 的 J-Lens 替换实验——保持率终审第二弹。

核心问题：HippoRAG2（arXiv 2502.14802v2，GraphRAG-Bench 排行榜医疗/小说
双域 SOTA 之一）的离线索引靠 LLM OpenIE 抽三元组、在线检索靠 LLM 做
recognition memory 过滤。本实验把离线图资产整体替换为我们已有的 J-Lens
读出产物 + Phase 53 文本侧实体（零 Qwen、全缓存），保留 HippoRAG2 的
query-to-triple → recognition memory → PPR 在线检索算法（纯矩阵运算，
我们的主场），看能保持原版排行榜性能的几成（保持率 ≥ 0.9 即成立）。

替换对应关系（HippoRAG2 组件 → 我们的产物）：
  OpenIE 三元组提取   → relations_{domain}.json（概念间带类型词边，248/308）
                        + Phase 53 实体共现边（augment_index 共现默认边，
                        ee/ec 各封顶 20000，MIN_COOCC=2）
  实体节点            → concept_cache twopass 概念（50/244，phase50 归并后
                        44/232）+ entity_cache_textside（11530/16186，入图前
                        按 _stem 归并——HippoRAG 实体侧同义边全对计算太贵，
                        任务书指定用 stem 归并替代）
  同义词边            → 概念侧 ws_vec 余弦 ≥ 0.8（HippoRAG 默认阈值，
                        phase46 P1 已验证此路），仅入图不进 fact store
                        （与官方 add_synonymy_edges 一致）
  passage 节点 + contains 边 → chunks + concept_chunks/entity_chunks
                        （dense-sparse 集成）
  嵌入模型            → bge-m3（CachedBgeM3Provider，CPU，磁盘缓存；
                        原版用 NV-Embed-v2，差异由换算系数吸收）

在线检索（对照官方 repo OSU-NLP-Group/HippoRAG 逐点重实现，对应清单见
输出 JSON 的 fidelity 字段）：
  1. query-to-triple：整查询 bge 嵌入 vs 全部三元组（"subject relation
     object" 文本）嵌入余弦，取 top-5（linking_top_k=5，官方默认）
  2. recognition memory：DeepSeek 从 top-5 三元组中保留与查询相关的
     （官方用 LLM/DSPy filter，是原方案的在线 LLM 步骤，保留不算替换失败；
     臂 B 记录"无过滤"对照）
  3. 种子重置分：保留三元组的 subject/object phrase 节点（分 = 平均三元组
     相似度 ÷ 该节点 chunk 数 df，再按出现次数平均，取 top-5 phrase——
     graph_search_with_fact_entities 原逻辑）+ **全部 passage 节点**
     （分 = min-max 归一化的 bge 相似度 × passage_node_weight）
  4. PPR：幂迭代 p ← d·M·p + (1-d)·s，M = 度归一化加权邻接（无向），
     d = damping = 0.5（官方 BaseConfig 默认，**不是**教科书 0.85）；
     权重：关系边 = prob、contains 边 = 1.0、同义边 = 余弦（官方同）
  5. passage 按 PPR 分排序取 top-10 进 context；无三元组存活时回退纯
     dense（官方 retrieve() 的 fallback 行为）

臂：
  a   完整替换（recognition memory 开，passage_node_weight=0.05 官方默认）
  b   臂 A 无 recognition memory（消融在线 LLM 过滤的贡献）
  c   臂 A + passage_node_weight 扫描 {0.3, 0.5, 0.8}（medical 上扫，
      选优后跑 novel；臂 A 本身即 0.05 扫描点）

换算（排行榜参照系，复用 phase50 同跑 B0 锚，不重跑）：
  保持率 = ACC × factor / 排行榜 HippoRAG2 ACC
  factor = phase50 的 B0 锚（medical 1.005 / novel 0.885）
  排行榜 = graphrag_bench_leaderboard.json 的 hipporag2 行
  （medical 均值 64.85 = 66.28/61.98/63.08/68.05；
   novel 均值 56.48 = 60.14/53.38/64.10/48.28）
内部对照（直接引用，不重跑）：phase50 B0（0.607/0.542）、
phase53 LightRAG-J ah（保持率 1.011/1.104）。

评测：DeepSeek generate_answer + judge_answer_correctness（phase26 链），
ACC 总分 + L1-L4 分级。medical 56 题 + novel 48 题（phase41 口径）。

运行：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1 LINCLE_BGE_M3_DEVICE=cpu
    python -c "import experiments.phase51_hipporag_j"           # 零副作用
    python -m experiments.phase51_hipporag_j --domain medical --arm all --smoke 4
    python -m experiments.phase51_hipporag_j --domain medical --arm all
    python -m experiments.phase51_hipporag_j --domain novel --arm all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase39_two_pass_cache import _stem
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, TOP_K,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.phase50_lightrag_j import (
    LightRagIndex, LEVELS, FULL_QUERIES,
)
from experiments.phase53_textside_entities import (
    augment_index, PHASE50_FACTOR, PHASE50_B0,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
CACHE_DIR = EXP / "concept_cache"
OUT_PATH = EXP / "phase51_hipporag_j.json"

# ── HippoRAG2 官方默认超参（OSU-NLP-Group/HippoRAG, BaseConfig） ─────────
LINKING_TOP_K = 5         # linking_top_k：top-5 三元组 + top-5 phrase 种子
DAMPING = 0.5             # damping：PPR 沿边概率（官方默认 0.5，非 0.85）
PASSAGE_NODE_WEIGHT = 0.05  # passage_node_weight：passage 重置分权重因子
SYNONYM_WS_COS = 0.8      # synonymy_edge_sim_threshold：同义边余弦阈值
PPR_MAX_ITER = 200
PPR_TOL = 1e-9
C_WEIGHT_SWEEP = [0.3, 0.5, 0.8]   # 臂 C 扫描（任务书指定；臂 A = 0.05 点）

# GraphRAG-Bench 排行榜 HippoRAG2 行（graphrag_bench_leaderboard.json，
# arXiv 2506.05690 Table 2，GPT-4o-mini 评测）。
LEADERBOARD_HIPPORAG2 = {
    "medical": {"L1": 66.28, "L2": 61.98, "L3": 63.08, "L4": 68.05},
    "novel": {"L1": 60.14, "L2": 53.38, "L3": 64.10, "L4": 48.28},
}
# phase53 终审对照行（直接引用，不重跑）
PHASE53_AH = {"medical": {"acc": 0.643, "retention": 1.011},
              "novel": {"acc": 0.563, "retention": 1.104}}


def leaderboard_hipporag2_mean(domain: str) -> float:
    vals = LEADERBOARD_HIPPORAG2[domain]
    return float(np.mean([vals[lv] for lv in LEVELS]))


# ── 图构建：概念 + 文本侧实体（_stem 归并） + 同义边 + contains ──────────


def stem_merge_entity_cache(entity_cache: dict) -> dict:
    """按 _stem 归并文本侧实体（替代 HippoRAG 实体侧同义边全对计算）。"""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in entity_cache["entity_chunks"]:
        key = " ".join(_stem(w) for w in name.split())
        groups[key].append(name)
    chunks, freq, display = {}, {}, {}
    for key, names in groups.items():
        rep = max(names,
                  key=lambda n: entity_cache["entity_frequency"].get(n, 0))
        cids: set[str] = set()
        for n in names:
            cids.update(entity_cache["entity_chunks"][n])
        chunks[key] = sorted(cids)
        freq[key] = sum(entity_cache["entity_frequency"].get(n, 0)
                        for n in names)
        display[key] = entity_cache["entity_display"].get(rep, rep)
    return {"entity_chunks": chunks, "entity_frequency": freq,
            "entity_display": display}


class HippoRagJIndex:
    """HippoRAG2 索引的 J-Lens 替换版。

    节点：phrase（概念实体 + 文本侧实体）+ passage（chunks）。
    边：关系三元组（prob 加权，对称）/ contains（1.0）/ 概念同义（ws 余弦）。
    triples：query-to-triple 用的全部三元组（去重文本）。
    """

    def __init__(self, cache: dict, vecs: dict, relations: dict,
                 entity_cache: dict) -> None:
        index = LightRagIndex(cache, vecs, relations)
        self.base_n = len(index.entities)   # 概念实体数（归并后）
        merged = stem_merge_entity_cache(entity_cache)
        stats = augment_index(index, merged, readout_edges=None)
        self.index = index
        self.aug_stats = stats
        n_phrase = len(index.entities)

        chunk_ids = sorted(cache["chunks"].keys())
        self.chunk_ids = chunk_ids
        chunk_pos = {cid: j for j, cid in enumerate(chunk_ids)}
        n_passage = len(chunk_ids)
        self.n_phrase = n_phrase
        self.n_passage = n_passage
        n_nodes = n_phrase + n_passage

        # ── 邻接（对称加权，csr；重复边求和） ──
        rows, cols, vals = [], [], []

        def _add(i: int, j: int, w: float) -> None:
            rows.append(i); cols.append(j); vals.append(w)
            rows.append(j); cols.append(i); vals.append(w)

        for r in index.relations:
            _add(r["a"], r["b"], max(float(r["prob"]), 1e-3))
        # contains：phrase → passage（HippoRAG passage 边权重 1.0）
        for p, ent in enumerate(index.entities):
            for cid in ent["chunks"]:
                j = chunk_pos.get(cid)
                if j is not None:
                    _add(p, n_phrase + j, 1.0)
        # 概念同义边：ws 余弦 >= SYNONYM_WS_COS（仅概念侧，图边非三元组）
        concepts = [str(c) for c in vecs["concepts"]]
        npz_index = {c: i for i, c in enumerate(concepts)}
        ws = vecs["ws_vec"].astype(np.float64)
        ws = ws / np.where(np.linalg.norm(ws, axis=1, keepdims=True) > 0,
                           np.linalg.norm(ws, axis=1, keepdims=True), 1.0)
        rep_vec = np.zeros((self.base_n, ws.shape[1]))
        for i in range(self.base_n):
            mem = [npz_index[m] for m in index.entities[i]["members"]
                   if m in npz_index]
            if mem:
                v = ws[mem].mean(axis=0)
                rep_vec[i] = v / (np.linalg.norm(v) + 1e-12)
        sim = rep_vec @ rep_vec.T
        ii, jj = np.where(np.triu(sim >= SYNONYM_WS_COS, k=1))
        n_syn = 0
        for a, b in zip(ii.tolist(), jj.tolist()):
            _add(a, b, float(sim[a, b]))
            n_syn += 1
        self.n_synonym_edges = n_syn

        A = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes),
                       dtype=np.float64)
        A.sum_duplicates()
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            dinv = np.where(deg > 0, 1.0 / np.where(deg > 0, deg, 1.0), 0.0)
        # 列随机转移矩阵 M[:, j] = A[:, j] / deg[j]（从 j 走向邻居）
        self.transition = A @ csr_matrix(
            (dinv, (np.arange(n_nodes), np.arange(n_nodes))),
            shape=(n_nodes, n_nodes))

        # phrase 节点的 df（contains chunk 数，phrase 种子衰减用）
        self.phrase_df = np.array(
            [max(1, len(ent["chunks"])) for ent in index.entities],
            dtype=np.float64)

        # ── 三元组（query-to-triple fact store，按文本去重） ──
        triples: list[dict] = []
        seen: set[str] = set()
        for r in index.relations:
            a, b = r["a"], r["b"]
            text = f"{index.entities[a]['name']} {r['relation']} " \
                   f"{index.entities[b]['name']}"
            if text in seen:
                continue
            seen.add(text)
            triples.append({"a": a, "b": b, "relation": r["relation"],
                            "text": text})
        self.triples = triples
        self.triple_emb: np.ndarray | None = None
        self.chunk_emb: np.ndarray | None = None

    def build_embeddings(self, embed_fn, corpus: dict[str, str]) -> None:
        tri = np.asarray(embed_fn([t["text"] for t in self.triples]),
                         dtype=np.float64)
        self.triple_emb = tri / np.where(
            np.linalg.norm(tri, axis=1, keepdims=True) > 0,
            np.linalg.norm(tri, axis=1, keepdims=True), 1.0)
        ch = np.asarray(embed_fn([corpus[cid] for cid in self.chunk_ids]),
                        dtype=np.float64)
        self.chunk_emb = ch / np.where(
            np.linalg.norm(ch, axis=1, keepdims=True) > 0,
            np.linalg.norm(ch, axis=1, keepdims=True), 1.0)

    # ── PPR（igraph personalized_pagerank 等价幂迭代） ──

    def ppr(self, reset: np.ndarray, damping: float = DAMPING) -> np.ndarray:
        s = reset.copy()
        total = s.sum()
        if total <= 0:
            return np.zeros(self.n_passage)
        s /= total
        p = s.copy()
        for _ in range(PPR_MAX_ITER):
            p_new = damping * (self.transition @ p) + (1.0 - damping) * s
            if np.abs(p_new - p).sum() < PPR_TOL:
                p = p_new
                break
            p = p_new
        return p[self.n_phrase:]

    # ── 在线检索（query-to-triple → [recognition] → 种子 → PPR） ──

    def query_to_triple(self, qv: np.ndarray) -> list[tuple[int, float]]:
        sims = self.triple_emb @ qv
        top = np.argsort(-sims)[:LINKING_TOP_K]
        return [(int(i), float(sims[i])) for i in top]

    def phrase_seed_weights(
        self, kept: list[tuple[int, float]],
    ) -> np.ndarray:
        """graph_search_with_fact_entities 原逻辑：分/df 累加→按次数平均
        →取 top-LINKING_TOP_K phrase。"""
        w = np.zeros(self.n_phrase)
        occurs = np.zeros(self.n_phrase)
        for ti, score in kept:
            t = self.triples[ti]
            for node in (t["a"], t["b"]):
                w[node] += score / self.phrase_df[node]
                occurs[node] += 1
        w = np.divide(w, occurs, out=np.zeros_like(w), where=occurs > 0)
        if len(kept) > 0 and (w > 0).sum() > LINKING_TOP_K:
            thresh = np.sort(w[w > 0])[-LINKING_TOP_K]
            w = np.where(w >= thresh, w, 0.0)
        return w

    def passage_seed_weights(self, qv: np.ndarray,
                             weight: float) -> tuple[np.ndarray, np.ndarray]:
        sims = self.chunk_emb @ qv
        lo, hi = float(sims.min()), float(sims.max())
        norm = (sims - lo) / (hi - lo) if hi > lo else np.zeros_like(sims)
        return norm * weight, sims

    def retrieve(
        self, qv: np.ndarray, kept: list[tuple[int, float]],
        passage_weight: float,
    ) -> tuple[list[str], dict]:
        """kept = recognition 后保留的 [(triple_idx, score)]（臂 B = 全 top-5）。
        空 → 官方 fallback：纯 dense 排序。"""
        pw, sims = self.passage_seed_weights(qv, passage_weight)
        dense_ranked = [self.chunk_ids[j]
                        for j in np.argsort(-sims)[:TOP_K]]
        if not kept:
            return dense_ranked, {"fallback": "dense_no_facts"}
        reset = np.concatenate([self.phrase_seed_weights(kept), pw])
        doc_scores = self.ppr(reset)
        top = np.argsort(-doc_scores)[:TOP_K]
        ranked = [self.chunk_ids[j] for j in top]
        seed_phrases = sorted(
            ((self.index.entities[i]["name"], float(reset[i]))
             for i in range(self.n_phrase) if reset[i] > 0),
            key=lambda x: -x[1])[:LINKING_TOP_K]
        debug = {
            "seed_phrases": [(n, round(s, 5)) for n, s in seed_phrases],
            "top_passage_scores": [round(float(doc_scores[j]), 8)
                                   for j in top[:5]],
        }
        return ranked, debug


# ── recognition memory（DeepSeek 在线过滤，官方 rerank_facts 的对应物） ──


def recognition_memory(query: str, candidates: list[tuple[int, float]],
                       triples: list[dict], llm) -> tuple[list[tuple[int, float]], str]:
    """DeepSeek 从 top-5 三元组中保留与查询相关的。返回 (kept, llm_raw)。"""
    lines = [f"{k + 1}. ({triples[ti]['text']})"
             for k, (ti, _s) in enumerate(candidates)]
    prompt = (
        f"Query: {query}\n\n"
        f"Candidate facts:\n" + "\n".join(lines) + "\n\n"
        "Which of these facts are relevant to answering the query? "
        "Reply with only the numbers of the relevant facts, comma-separated "
        "(e.g. \"1,3\"). Reply \"none\" if none are relevant."
    )
    try:
        msg = llm.complete(prompt, max_tokens=20)
        raw = msg.content if hasattr(msg, "content") else str(msg)
    except Exception as e:  # API 故障时保守保留全部（退化为臂 B 行为）
        return list(candidates), f"[ERROR: {e}]"
    kept_idx = []
    for tok in raw.replace(" ", "").split(","):
        if tok.isdigit() and 1 <= int(tok) <= len(candidates):
            kept_idx.append(int(tok) - 1)
    kept = [candidates[k] for k in sorted(set(kept_idx))]
    return kept, raw.strip()


# ── 单域运行 ─────────────────────────────────────────────────────────────


def arm_keys(arms: list[str], domain: str, out: dict,
             c_sweep_all: bool = False) -> list[str]:
    """展开臂名为具体配置键。臂 C：medical 全扫；novel 用 medical 选优
    （--c-sweep-all 时 novel 也全扫，用于核查 0.881 临界值是否为权重假象）。"""
    keys = []
    for arm in arms:
        if arm == "c":
            if domain == "medical" or c_sweep_all:
                keys += [f"c_w{w}" for w in C_WEIGHT_SWEEP]
            else:
                keys.append(f"c_w{best_sweep_weight(out)}")
        else:
            keys.append(arm)
    return keys


def best_sweep_weight(out: dict) -> float:
    """从已落盘的 medical 扫描结果选最优 passage_node_weight（缺省 0.5）。"""
    try:
        res = out["domains"]["medical"]["results"]
        best, best_acc = 0.5, -1.0
        for key, r in res.items():
            if key.startswith("c_w") and r.get("acc") is not None \
                    and r["acc"] > best_acc:
                best, best_acc = float(key[3:]), r["acc"]
        return best
    except (KeyError, TypeError, ValueError):
        return 0.5


def key_config(key: str) -> tuple[bool, float]:
    """配置键 → (recognition 开关, passage_node_weight)。"""
    if key == "a":
        return True, PASSAGE_NODE_WEIGHT
    if key == "b":
        return False, PASSAGE_NODE_WEIGHT
    return True, float(key[3:])


def run_domain(
    domain: str,
    arms: list[str],
    out: dict,
    max_queries: int = 0,
    smoke: bool = False,
    verbose: bool = True,
    c_sweep_all: bool = False,
) -> dict:
    cache, vecs, relations = load_phase41_inputs(domain)
    corpus = load_corpus_texts(domain, cache)
    n_q = FULL_QUERIES[domain] if max_queries == 0 else max_queries
    _corpus, questions = load_graphrag_bench(domain, n_q)
    if smoke:
        questions = questions[:smoke]

    entity_path = CACHE_DIR / f"entity_cache_textside_{domain}.json"
    entity_cache = json.loads(entity_path.read_text())

    hg = HippoRagJIndex(cache, vecs, relations, entity_cache)
    if verbose:
        print(f"  [{domain}] {hg.n_passage} passages, {hg.base_n} concept "
              f"+ {hg.n_phrase - hg.base_n} text-side phrases, "
              f"{len(hg.triples)} unique triples, "
              f"{hg.n_synonym_edges} synonym edges, "
              f"{len(questions)} queries", flush=True)

    from experiments.embed_cache import CachedBgeM3Provider
    embed_fn = CachedBgeM3Provider().embed
    t0 = time.time()
    hg.build_embeddings(embed_fn, corpus)
    if verbose:
        print(f"  [{domain}] embeddings ready ({time.time() - t0:.0f}s)",
              flush=True)
    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    keys = arm_keys(arms, domain, out, c_sweep_all=c_sweep_all)
    need_recognition = any(k != "b" for k in keys)

    # Phase 1: retrieval（recognition memory 是唯一的在线 LLM 步骤）
    llm = None
    contexts: list[dict] = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi] / (np.linalg.norm(query_emb[qi]) + 1e-12)
        candidates = hg.query_to_triple(qv)
        entry = {"qid": q.get("id", str(qi)), "level": q.get("level"),
                 "question": q["question"], "answer": q.get("answer", ""),
                 "ranked": {}, "debug": {
                     "candidates": [
                         {"text": hg.triples[ti]["text"], "sim": round(s, 4)}
                         for ti, s in candidates]}}
        kept = None
        if need_recognition:
            if llm is None:
                from jgraphrag.llm import DeepSeekProvider
                llm = DeepSeekProvider()
            kept, raw = recognition_memory(
                q["question"], candidates, hg.triples, llm)
            entry["debug"]["recognition"] = {
                "kept": [hg.triples[ti]["text"] for ti, _s in kept],
                "llm_raw": raw}
        for key in keys:
            use_rec, weight = key_config(key)
            k = kept if use_rec else list(candidates)
            ranked, dbg = hg.retrieve(qv, k or [], weight)
            entry["ranked"][key] = ranked
            entry["debug"].setdefault("arms", {})[key] = dbg
        contexts.append(entry)
        if verbose and (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)

    if smoke:
        return {"domain": domain, "smoke": True, "retrieval": contexts,
                "n_phrase": hg.n_phrase, "n_passage": hg.n_passage,
                "n_triples": len(hg.triples),
                "n_synonym_edges": hg.n_synonym_edges,
                "aug_stats": hg.aug_stats}

    # Phase 2: DeepSeek answer + judge（phase26 链，与 phase50/53 同构）
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi_entry):
        qi, entry = qi_entry
        llm = DeepSeekProvider()
        rec = {"qid": entry["qid"], "level": entry["level"],
               "question": entry["question"], "answer": entry["answer"],
               "methods": {}}
        for key in keys:
            ctx = " ".join(corpus[cid] for cid in entry["ranked"][key])
            ans = generate_answer(entry["question"], ctx, llm)
            acc = bool(judge_answer_correctness(
                entry["question"], ans, entry["answer"], llm))
            rec["methods"][key] = {"acc": acc, "answer": ans[:300],
                                   "ranked": entry["ranked"][key]}
        rec["debug"] = entry["debug"]
        return qi, rec

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

    results = {}
    for key in keys:
        accs = [1.0 if r["methods"][key]["acc"] else 0.0 for r in per_query]
        by_level = {}
        for lv in LEVELS:
            lv_accs = [1.0 if r["methods"][key]["acc"] else 0.0
                       for r in per_query if r["level"] == lv]
            by_level[lv] = {"acc": float(np.mean(lv_accs)) if lv_accs else None,
                            "n": len(lv_accs)}
        results[key] = {"acc": float(np.mean(accs)) if accs else None,
                        "by_level": by_level, "n": len(per_query)}

    if verbose:
        print(f"\n  [{domain}] {'arm':<8} {'ACC':>7} "
              f"{'L1':>7} {'L2':>7} {'L3':>7} {'L4':>7}")
        for key in keys:
            r = results[key]
            cells = [f"{r['by_level'][lv]['acc']:.2f}"
                     if r["by_level"][lv]["acc"] is not None else "n/a"
                     for lv in LEVELS]
            print(f"  {'':<9} {key:<8} {r['acc']:.3f} "
                  f"{cells[0]:>7} {cells[1]:>7} {cells[2]:>7} {cells[3]:>7}")
    return {"domain": domain, "results": results, "per_query": per_query,
            "n_phrase": hg.n_phrase, "n_passage": hg.n_passage,
            "n_triples": len(hg.triples),
            "n_synonym_edges": hg.n_synonym_edges,
            "aug_stats": {k: v for k, v in hg.aug_stats.items()}}


# ── 换算与汇总 ───────────────────────────────────────────────────────────


def summarize(domain: str, results: dict) -> dict:
    factor = PHASE50_FACTOR[domain]
    lb = leaderboard_hipporag2_mean(domain)
    summary = {"leaderboard": {
        "source": "graphrag_bench_leaderboard.json hipporag2 "
                  "(GraphRAG-Bench arXiv 2506.05690 Table 2)",
        "hipporag2_acc_pct": lb,
        "hipporag2_by_level_pct": LEADERBOARD_HIPPORAG2[domain],
    }, "factor": factor, "factor_source": "phase50 B0 锚（复用，不重跑）",
        "controls_quoted": {
            "b0_phase50_acc": PHASE50_B0[domain],
            "lightrag_j_ah_phase53": PHASE53_AH[domain],
        }, "arms": {}}
    for key, r in results.items():
        acc = r["acc"]
        retention = (acc * factor * 100.0 / lb
                     if acc is not None else None)
        summary["arms"][key] = {
            "acc": acc,
            "retention_vs_leaderboard_hipporag2": retention,
            "verdict": ("retained(>=0.9)"
                        if retention is not None and retention >= 0.9
                        else "not_retained"),
        }
    return summary


FIDELITY = [
    {"point": "query-to-triple：整查询嵌入 vs 全部三元组文本嵌入 top-5",
     "status": "一致", "note": "linking_top_k=5 官方默认；嵌入 bge-m3 替代 "
     "NV-Embed-v2（差异由换算系数吸收）"},
    {"point": "recognition memory：LLM 过滤 top-5 三元组",
     "status": "一致", "note": "官方 DSPy filter → DeepSeek 单 prompt 判留；"
     "臂 B 消融"},
    {"point": "phrase 种子分 = 平均三元组分 ÷ df，top-5 phrase",
     "status": "一致", "note": "graph_search_with_fact_entities 原逻辑"},
    {"point": "全部 passage 入重置，分 = min-max 归一 dense 相似度 × "
     "passage_node_weight", "status": "一致",
     "note": "官方默认 0.05（臂 A）；臂 C 扫 {0.3,0.5,0.8}"},
    {"point": "PPR：无向加权图，damping=0.5，幂迭代/度归一",
     "status": "近似", "note": "官方 igraph prpack；本实现 scipy 幂迭代 "
     "（同一不动点，tol=1e-9）。damping 官方默认 0.5 而非任务书估计的 0.85"},
    {"point": "边权：关系=prob（官方=共现计数）、contains=1.0、同义=余弦",
     "status": "近似", "note": "我们的关系边无原始计数，prob 是同量级替代"},
    {"point": "无三元组存活 → 回退纯 dense", "status": "一致",
     "note": "官方 retrieve() fallback"},
    {"point": "passage top-k 进 context", "status": "近似",
     "note": "官方 qa_top_k=5/retrieval_top_k=200；本实验 top-10 与 "
     "phase50/53 口径一致（同 judge 同 context 预算才可比）"},
    {"point": "同义边：实体嵌入 KNN ≥0.8", "status": "近似",
     "note": "概念侧 ws_vec ≥0.8；文本侧实体按任务书用 _stem 归并替代"},
]


def build_final(out: dict) -> dict:
    """终审对比表 + 判决 + 失败案例（arm a 答错的前 3 题/域）。"""
    table = {}
    for domain, d in out.get("domains", {}).items():
        factor = PHASE50_FACTOR[domain]
        lb = leaderboard_hipporag2_mean(domain)
        # 回填 gold answer（旧版 per_query 未存 answer 字段）
        gold_of = {}
        try:
            _c, qs = load_graphrag_bench(domain, FULL_QUERIES[domain])
            gold_of = {q.get("id", str(i)): q.get("answer", "")
                       for i, q in enumerate(qs)}
        except Exception:
            pass
        for rec in d.get("per_query", []):
            if not rec.get("answer") and rec["qid"] in gold_of:
                rec["answer"] = gold_of[rec["qid"]]
        arms = {}
        for key, r in d["results"].items():
            if r.get("acc") is None:
                continue
            ret = r["acc"] * factor * 100.0 / lb
            arms[key] = {"acc": r["acc"], "retention": round(ret, 3),
                         "retained": ret >= 0.9}
        fails = []
        for rec in d.get("per_query", []):
            m = rec.get("methods", {}).get("a")
            if m and not m["acc"]:
                fails.append({"qid": rec["qid"], "level": rec["level"],
                              "question": rec["question"],
                              "gold": rec.get("answer", "")[:200],
                              "generated": m["answer"][:200]})
        table[domain] = {
            "b0_phase50": {"acc": PHASE50_B0[domain]},
            "lightrag_j_ah_phase53": PHASE53_AH[domain],
            "hipporag_j_arms": arms,
            "best_arm": max(arms, key=lambda k: arms[k]["retention"])
            if arms else None,
            "arm_a_failures": fails[:3],
        }
    best_ret = {dom: (t["hipporag_j_arms"].get(t["best_arm"], {})
                      .get("retention") or 0.0)
                for dom, t in table.items() if t.get("best_arm")}
    both = len(table) == 2 and all(
        any(a["retained"] for a in t["hipporag_j_arms"].values())
        for t in table.values())
    verdict = (
        "HippoRAG-J 保持率双域 ≥0.9（经 passage_node_weight 调优）"
        if both else
        "HippoRAG-J 未能在双域同时达到保持率 0.9——见各域臂表")
    return {"table": table, "best_retention_by_domain": best_ret,
            "both_domains_retained": both, "one_line_verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 51 HippoRAG2 J-Lens 替换实验")
    ap.add_argument("--domain", default="medical",
                    choices=["medical", "novel", "both"])
    ap.add_argument("--arm", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = 全量（medical 56 / novel 48）")
    ap.add_argument("--smoke", type=int, default=0,
                    help="只跑检索不跑 LLM 评测（前 N 题，不落盘）")
    ap.add_argument("--c-sweep-all", action="store_true",
                    help="novel 域臂 C 也全扫 {0.3,0.5,0.8}（默认只用 "
                         "medical 选优权重）")
    ap.add_argument("--finalize", action="store_true",
                    help="不重跑实验，仅从已落盘结果重建 final 汇总表")
    args = ap.parse_args()

    arms = ["a", "b", "c"] if args.arm == "all" else [args.arm]
    domains = ["medical", "novel"] if args.domain == "both" else [args.domain]

    out = {}
    if OUT_PATH.exists() and not args.smoke:
        out = json.loads(OUT_PATH.read_text())
    out.setdefault("method", "phase51_hipporag_j")
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["config"] = {
        "linking_top_k": LINKING_TOP_K, "damping": DAMPING,
        "passage_node_weight": PASSAGE_NODE_WEIGHT,
        "synonym_ws_cos": SYNONYM_WS_COS, "c_weight_sweep": C_WEIGHT_SWEEP,
        "top_k": TOP_K, "arms": arms,
    }
    out["fidelity"] = FIDELITY

    if args.finalize:
        out["final"] = build_final(out)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(json.dumps(out["final"], indent=2, ensure_ascii=False))
        return

    for domain in domains:
        print(f"\n── {domain} ──")
        res = run_domain(domain, arms, out, max_queries=args.max_queries,
                         smoke=args.smoke, c_sweep_all=args.c_sweep_all)
        if args.smoke:
            for entry in res["retrieval"][:3]:
                print(f"  {entry['qid']} [{entry['level']}] {entry['question'][:60]}")
                print(f"    candidates="
                      f"{[c['text'] for c in entry['debug']['candidates']]}")
                if "recognition" in entry["debug"]:
                    print(f"    recognition kept="
                          f"{entry['debug']['recognition']['kept']}")
                for key, dbg in entry["debug"].get("arms", {}).items():
                    print(f"    {key}: seeds={dbg.get('seed_phrases')}"
                          f" ranked={entry['ranked'][key][:5]}")
            continue
        prev = out.setdefault("domains", {}).get(domain, {})
        merged_results = dict(prev.get("results", {}))
        merged_results.update(res["results"])
        prev_pq = {r["qid"]: r for r in prev.get("per_query", [])}
        for rec in res["per_query"]:
            if rec["qid"] in prev_pq:
                prev_pq[rec["qid"]]["methods"].update(rec["methods"])
                prev_pq[rec["qid"]]["debug"] = rec["debug"]
            else:
                prev_pq[rec["qid"]] = rec
        merged_pq = list(prev_pq.values())
        out["domains"][domain] = {
            "results": merged_results,
            "n_phrase": res["n_phrase"], "n_passage": res["n_passage"],
            "n_triples": res["n_triples"],
            "n_synonym_edges": res["n_synonym_edges"],
            "aug_stats": res["aug_stats"],
            "summary": summarize(domain, merged_results),
            "per_query": merged_pq,
        }
        s = out["domains"][domain]["summary"]
        print(f"  factor = {s['factor']} (phase50 B0 锚), "
              f"leaderboard HippoRAG2 = "
              f"{s['leaderboard']['hipporag2_acc_pct']:.2f}%")
        for key, a in s["arms"].items():
            r = a["retention_vs_leaderboard_hipporag2"]
            print(f"  arm {key}: ACC={a['acc']:.3f} retention={r:.3f} "
                  f"→ {a['verdict']}")
        out["final"] = build_final(out)
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
