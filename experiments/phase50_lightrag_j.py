"""Phase 50: LightRAG 的 J-Lens 替换实验——纯替换能否保住排行榜性能。

核心问题：LightRAG（arXiv 2410.05779，GraphRAG-Bench 引用的官方版本；
任务书中写的 2410.12838 是另一篇论文）把实体/关系提取与查询关键词提取
都交给 LLM generate()。本实验把这两处 LLM 依赖整体替换为我们已有的
J-Lens 读出产物（零 API、本地 7B 预提取缓存），保留 LightRAG 的图结构
与 dual-level 检索算法，在 GraphRAG-Bench 上看能保持原版的几成性能
（保持率 >= 0.9 即成立）。

替换对应关系（LightRAG 组件 → 我们的产物）：
  R(·) 实体提取      → concept_cache_{domain}_twopass.json 的 50/244 概念
  P(·) 实体画像 K-V  → 概念名（key）+ roles 列表（value/description）
  R(·) 关系提取      → relations_{domain}.json 的边（含 relation 类型词与 prob）
  D(·) 去重合并      → _stem 词形归并 + ws_vec 余弦 >= 0.95 的近义合并
                       （phase44 结论的廉价复现，不重跑 GPU 判定）
  查询关键词提取     → 不抽关键词，query 直接 bge 嵌入后同时充当
                       local（实体匹配）与 global（关系匹配）键
  实体/关系向量库    → bge-m3（CachedBgeM3Provider，CPU，磁盘缓存；
                       原版用 OpenAI 嵌入，差异由换算系数吸收）
  高阶关联 (iii)     → 命中实体/关系端点的一跳邻居（衰减 0.5）
  上下文组装         → chunk 继承其最优命中图元素的权重（MAX 聚合，对应
                       LightRAG 按实体/关系顺序收集 text units），同分按
                       bge 余弦排序，top-10，bge naive 回填（hybrid 模式）。
                       注：初版用 SUM 聚合，在 44 实体的极小图上退化为
                       查询无关 hub 排序（medical ACC 1.8%），已修正。

臂：
  b0  纯 bge chunk 检索（对照，phase41 口径）
  a   纯替换：上述 LightRAG 重实现
  b   替换+增强：臂 A 的图上加 phase47 CP 张量补全的高分缺失边
      （rank8/lam1.0 = phase47 S2 medical 最优配置，AUC 0.813；
      补全边数 <= 原边数 20%；novel 复用同配置，phase47 未跑 novel）
      注：任务书提到的 V3 关系细化边在 relations_{domain}.json 中不存在
      单独版本（mcos 是同规模另一抽样），按任务书指示用现有边即可。

换算（排行榜参照系，原版本地不可跑——Fast-GraphRAG 在同环境 7B 下已崩盘）：
  保持率 = ACC(臂) × factor / 排行榜 LightRAG ACC
  factor = 排行榜 RAG(w/o rerank) ACC / 本跑 B0 ACC（同查询集同 judge，
           phase26b 锚点：medical 61.0 / 57.1 ≈ 1.07，本跑重算）
  排行榜数字 = GraphRAG-Bench 论文 Table 2（GPT-4o-mini 评测，
           arXiv 2506.05690v3，https://arxiv.org/html/2506.05690v3），
           总 ACC = L1-L4 四级均值（与 61.0 锚点口径一致）。

评测：DeepSeek generate_answer + judge_answer_correctness（phase26 链），
ACC 总分 + L1-L4 分级。medical 56 题 + novel 48 题（phase41/42 口径）。

运行：
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -c "import experiments.phase50_lightrag_j"           # 零副作用
    python -m experiments.phase50_lightrag_j --domain medical --arm all --smoke 4
    python -m experiments.phase50_lightrag_j --domain medical --arm all
    python -m experiments.phase50_lightrag_j --domain novel --arm all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase39_two_pass_cache import _stem
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, TOP_K,
)
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase50_lightrag_j.json"

# ── LightRAG 重实现超参（对应官方实现默认值，见模块 docstring） ─────────
ENT_TOP_K = 20          # local 检索：query→实体 top-k（官方 QueryParam top_k 量级）
REL_TOP_K = 20          # global 检索：query→关系 top-k
NEIGHBOR_DECAY = 0.5    # 高阶关联 (iii)：一跳邻居权重衰减
MERGE_WS_COS = 0.95     # D(·) 去重：ws 余弦合并阈值（极保守，仅近同义）
CP_RANK, CP_LAM, CP_RESTARTS = 8, 1.0, 5   # phase47 S2 最优配置
CP_MAX_FRAC = 0.2       # 张量补全边数上限（相对原边数）

FULL_QUERIES = {"medical": 56, "novel": 48}

# GraphRAG-Bench 论文 Table 2（Generation ACC, GPT-4o-mini 评测）。
# 来源: https://arxiv.org/html/2506.05690v3 (arXiv 2506.05690v3, §4.1 Table 2)
LEADERBOARD_URL = "https://arxiv.org/html/2506.05690v3"
LEADERBOARD_ACC = {
    "medical": {
        "LightRAG": {"L1": 63.32, "L2": 61.32, "L3": 63.14, "L4": 67.91},
        "RAG_wo_rerank": {"L1": 63.72, "L2": 57.61, "L3": 63.72, "L4": 58.94},
    },
    "novel": {
        "LightRAG": {"L1": 58.62, "L2": 49.07, "L3": 48.85, "L4": 23.80},
        "RAG_wo_rerank": {"L1": 58.76, "L2": 41.35, "L3": 50.08, "L4": 41.52},
    },
}
LEVELS = ["L1", "L2", "L3", "L4"]


def leaderboard_mean(domain: str, method: str) -> float:
    vals = LEADERBOARD_ACC[domain][method]
    return float(np.mean([vals[lv] for lv in LEVELS]))


# ── D(·) 实体去重合并：_stem 归并 + ws 余弦阈值 ─────────────────────────


def merge_entities(
    concepts: list[str], ws_vec: np.ndarray, threshold: float = MERGE_WS_COS,
) -> list[list[int]]:
    """Union-find 合并：同 _stem 组，或 ws 余弦 >= threshold。返回成员下标组。"""
    n = len(concepts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_stem: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(concepts):
        by_stem[_stem(c)].append(i)
    for idxs in by_stem.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    ws = ws_vec.astype(np.float64)
    ws = ws / np.where(np.linalg.norm(ws, axis=1, keepdims=True) > 0,
                       np.linalg.norm(ws, axis=1, keepdims=True), 1.0)
    sim = ws @ ws.T
    ii, jj = np.where(np.triu(sim >= threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: min(g))


# ── LightRAG 索引（图 + 向量库） ─────────────────────────────────────────


class LightRagIndex:
    """LightRAG 索引的 J-Lens 替换版：实体/关系表 + bge 向量 + 一跳邻接。

    entities: [{name, members, chunks, roles, text}]
    relations: [{a, b, relation, prob, text}]（a/b 为合并后实体下标）
    """

    def __init__(self, cache: dict, vecs: dict, relations: dict,
                 extra_edges: list[dict] | None = None) -> None:
        concepts = [str(c) for c in vecs["concepts"]]
        npz_index = {c: i for i, c in enumerate(concepts)}
        groups = merge_entities(concepts, vecs["ws_vec"])
        canon_of: dict[str, int] = {}   # 原概念 → 合并实体下标
        entities: list[dict] = []
        for g in groups:
            members = [concepts[i] for i in g]
            rep = max(g, key=lambda i: (int(vecs["count"][i]), -i))
            chunks: set[str] = set()
            roles: set[str] = set()
            for m in members:
                chunks.update(cache["concept_chunks"].get(m, []))
            entities.append({
                "name": concepts[rep], "members": members,
                "chunks": sorted(chunks), "roles": roles, "text": "",
            })
            for m in members:
                canon_of[m] = len(entities) - 1
        # roles 汇总（chunks 的 roles 键是原始大小写概念名，统一小写后匹配）
        for cid, cdata in cache["chunks"].items():
            for raw_concept, role_list in (cdata.get("roles") or {}).items():
                ent_idx = canon_of.get(raw_concept.lower())
                if ent_idx is not None:
                    entities[ent_idx]["roles"].update(role_list)
        for ent in entities:
            ent["roles"] = sorted(ent["roles"])
            ent["text"] = (f"{ent['name']}: {', '.join(ent['roles'])}"
                           if ent["roles"] else ent["name"])

        rel_list: list[dict] = []
        adj: dict[int, set[int]] = defaultdict(set)
        for e in relations["edges"]:
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
        for e in extra_edges or []:
            a, b = e["a"], e["b"]
            rel_list.append({
                "a": a, "b": b, "relation": e["relation"],
                "prob": e["prob"],
                "text": f"{entities[a]['name']} {e['relation']} "
                        f"{entities[b]['name']}",
                "completed": True,
            })
            adj[a].add(b)
            adj[b].add(a)

        self.entities = entities
        self.relations = rel_list
        self.adj = {k: sorted(v) for k, v in adj.items()}
        self.canon_of = canon_of
        self.ent_emb: np.ndarray | None = None
        self.rel_emb: np.ndarray | None = None

    def build_embeddings(self, embed_fn) -> None:
        ent = np.asarray(embed_fn([e["text"] for e in self.entities]),
                         dtype=np.float64)
        rel = np.asarray(embed_fn([r["text"] for r in self.relations]),
                         dtype=np.float64)
        self.ent_emb = ent / np.where(
            np.linalg.norm(ent, axis=1, keepdims=True) > 0,
            np.linalg.norm(ent, axis=1, keepdims=True), 1.0)
        self.rel_emb = rel / np.where(
            np.linalg.norm(rel, axis=1, keepdims=True) > 0,
            np.linalg.norm(rel, axis=1, keepdims=True), 1.0)


# ── 张量补全（phase47 CP 重构，臂 B 用） ─────────────────────────────────


def tensor_completion_edges(
    domain: str, index: LightRagIndex, max_new: int,
) -> list[dict]:
    """Fit CP on the relation tensor; return top-scored missing edges.

    Score normalized to (0,1] by the max completion score so completed edges
    enter retrieval with prob-comparable weights. Edge concepts are mapped
    through the index's merged-entity canonicalization; self-loops dropped.
    """
    from experiments.phase47_tensor_multihop import (
        build_tensor, fit_cp_best, cp_reconstruct,
    )

    T, concepts, buckets, _edges, _W, _bm = build_tensor(domain)
    rng = np.random.default_rng(0)
    factors, err = fit_cp_best(T, CP_RANK, CP_LAM, CP_RESTARTS, rng)
    t_hat = cp_reconstruct(factors)
    observed = T > 0
    n = T.shape[0]
    cand = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = int(np.argmax(t_hat[i, :, j]))
            s = float(t_hat[i, r, j])
            if s > 0 and not observed[i, r, j]:
                cand.append((s, concepts[i], buckets[r], concepts[j]))
    cand.sort(key=lambda x: -x[0])
    out = []
    seen = set()
    s_max = cand[0][0] if cand else 1.0
    for s, a, bucket, b in cand:
        ea, eb = index.canon_of.get(a), index.canon_of.get(b)
        if ea is None or eb is None or ea == eb or (ea, eb) in seen:
            continue
        seen.add((ea, eb))
        out.append({"a": ea, "b": eb, "relation": bucket,
                    "prob": s / s_max, "score": s})
        if len(out) >= max_new:
            break
    print(f"  CP completion: recon_err={err:.4f}, {len(out)} new edges "
          f"(cap {max_new}, rank={CP_RANK} lam={CP_LAM})")
    return out


# ── LightRAG dual-level 检索（query 嵌入直替关键词提取） ─────────────────


def lightrag_retrieve(
    index: LightRagIndex, query_vec: np.ndarray,
) -> tuple[dict[str, float], dict]:
    """Dual-level retrieval → ({cid: score}, debug).

    (ii) keyword matching: query vec vs entity texts (local) and relation
    texts (global); relation hits weight endpoint entities by sim x prob.
    (iii) high-order relatedness: 1-hop neighbors of hit entities and of
    hit relations' endpoints, weight decayed by NEIGHBOR_DECAY.
    Chunk score = MAX entity weight over the chunk's entities — the chunk
    inherits the rank of its best-matching graph element (LightRAG collects
    text units in the order of the entities/relations they came from).
    NOTE: the first run of this phase used SUM aggregation instead; on our
    tiny 44/232-entity graph that collapsed to a query-independent hub
    ranking (multi-concept overview chunks always win; medical ACC 1.8%).
    MAX keeps the assembly faithful to LightRAG's per-element ordering.
    """
    qv = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    ent_w: dict[int, float] = {}

    sims = index.ent_emb @ qv
    top_e = np.argsort(-sims)[:ENT_TOP_K]
    hit_entities = [(int(i), float(sims[i])) for i in top_e if sims[i] > 0]
    for i, s in hit_entities:
        ent_w[i] = s

    rsims = index.rel_emb @ qv if len(index.relations) else np.zeros(0)
    hit_relations = []
    if len(index.relations):
        top_r = np.argsort(-rsims)[:REL_TOP_K]
        for ri in top_r:
            if rsims[ri] <= 0:
                continue
            r = index.relations[int(ri)]
            w = float(rsims[ri]) * r["prob"]
            hit_relations.append({"text": r["text"], "sim": float(rsims[ri]),
                                  "prob": r["prob"], "completed": r["completed"]})
            for ep in (r["a"], r["b"]):
                ent_w[ep] = max(ent_w.get(ep, 0.0), w)

    for e, w in list(ent_w.items()):
        for nb in index.adj.get(e, []):
            ent_w[nb] = max(ent_w.get(nb, 0.0), NEIGHBOR_DECAY * w)

    chunk_score: dict[str, float] = defaultdict(float)
    for e, w in ent_w.items():
        for cid in index.entities[e]["chunks"]:
            if w > chunk_score[cid]:
                chunk_score[cid] = w
    debug = {
        "hit_entities": [(index.entities[i]["name"], round(s, 4))
                         for i, s in hit_entities[:10]],
        "hit_relations": hit_relations[:10],
        "n_entities_weighted": len(ent_w),
    }
    return dict(chunk_score), debug


def merge_topk(scores: dict[str, float], b0_ids: list[str],
               bge_sim: dict[str, float] | None = None,
               top_k: int = TOP_K) -> list[str]:
    """Graph-ranked chunks first, bge (naive) backfill — LightRAG hybrid mode.

    Chunks sharing the same max entity weight (e.g. all chunks of the top
    entity) are ordered by bge cosine to the query — deterministic and keeps
    the naive channel inside the graph-ranked prefix, as LightRAG's hybrid
    mode does within its token budget.
    """
    bge_sim = bge_sim or {}
    ranked = [cid for cid, _s in sorted(
        scores.items(), key=lambda x: (x[1], bge_sim.get(x[0], 0.0)),
        reverse=True)]
    merged = ranked[:top_k]
    for cid in b0_ids:
        if len(merged) >= top_k:
            break
        if cid not in merged:
            merged.append(cid)
    return merged[:top_k]


# ── 单域运行 ─────────────────────────────────────────────────────────────


def run_domain(
    domain: str,
    arms: list[str],
    max_queries: int = 0,
    smoke: bool = False,
    verbose: bool = True,
) -> dict:
    cache, vecs, relations = load_phase41_inputs(domain)
    corpus = load_corpus_texts(domain, cache)
    n_q = FULL_QUERIES[domain] if max_queries == 0 else max_queries
    _corpus, questions = load_graphrag_bench(domain, n_q)
    if smoke:
        questions = questions[:smoke]

    from experiments.embed_cache import CachedBgeM3Provider
    embed_fn = CachedBgeM3Provider().embed

    chunk_ids = sorted(cache["chunks"].keys())
    chunk_emb = np.asarray(
        embed_fn([corpus[cid] for cid in chunk_ids]), dtype=np.float64)
    chunk_emb = chunk_emb / np.where(
        np.linalg.norm(chunk_emb, axis=1, keepdims=True) > 0,
        np.linalg.norm(chunk_emb, axis=1, keepdims=True), 1.0)

    index = LightRagIndex(cache, vecs, relations)
    n_orig_rel = len(index.relations)
    if "b" in arms:
        max_new = int(len(relations["edges"]) * CP_MAX_FRAC)
        extra = tensor_completion_edges(domain, index, max_new)
        index_b = LightRagIndex(cache, vecs, relations, extra_edges=extra)
    else:
        extra, index_b = [], None
    if verbose:
        print(f"  [{domain}] {len(chunk_ids)} chunks, "
              f"{len(index.entities)} entities "
              f"(from {len(vecs['concepts'])} concepts), "
              f"{n_orig_rel} relations (+{len(extra)} completed), "
              f"{len(questions)} queries, arms={arms}")

    if "a" in arms or "ah" in arms:
        index.build_embeddings(embed_fn)
    if index_b is not None:
        index_b.build_embeddings(embed_fn)

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    # Phase 1: retrieval (local, no LLM)
    contexts: list[dict] = []
    for qi, q in enumerate(questions):
        qv = query_emb[qi] / (np.linalg.norm(query_emb[qi]) + 1e-12)
        q_sims = chunk_emb @ qv
        b0_ids = [chunk_ids[j] for j in np.argsort(-q_sims)[:TOP_K]]
        bge_sim = {chunk_ids[j]: float(q_sims[j])
                   for j in range(len(chunk_ids))}
        entry = {"qid": q.get("id", str(qi)), "level": q.get("level"),
                 "question": q["question"], "answer": q.get("answer", ""),
                 "ranked": {"b0": b0_ids}, "debug": {}}
        for arm, idx in (("a", index), ("b", index_b)):
            if arm not in arms or idx is None:
                continue
            scores, dbg = lightrag_retrieve(idx, qv)
            entry["ranked"][arm] = merge_topk(scores, b0_ids, bge_sim)
            entry["debug"][arm] = dbg
        if "ah" in arms:
            # Diagnostic hybrid: round-robin interleave of graph-ranked and
            # naive bge chunks (LightRAG's default mode concatenates BOTH
            # graph-derived text units and naive vector hits in the context;
            # arm a lets the graph channel monopolize the top-10 budget).
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
            entry["ranked"]["ah"] = merged
            entry["debug"]["ah"] = dbg
        contexts.append(entry)
        if verbose and (qi + 1) % 20 == 0:
            print(f"    retrieval {qi + 1}/{len(questions)}", flush=True)

    if smoke:
        return {"domain": domain, "smoke": True, "retrieval": contexts,
                "n_entities": len(index.entities),
                "n_relations": n_orig_rel, "n_completed": len(extra)}

    # Phase 2: LLM evaluation (DeepSeek answer + judge)
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval(qi_entry):
        qi, entry = qi_entry
        llm = DeepSeekProvider()
        rec = {"qid": entry["qid"], "level": entry["level"],
               "question": entry["question"], "methods": {}}
        for arm in arms:
            ranked = entry["ranked"][arm]
            ctx = " ".join(corpus[cid] for cid in ranked)
            ans = generate_answer(entry["question"], ctx, llm)
            acc = bool(judge_answer_correctness(
                entry["question"], ans, entry["answer"], llm))
            rec["methods"][arm] = {"acc": acc, "answer": ans[:300],
                                   "ranked": ranked,
                                   "debug": entry["debug"].get(arm)}
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
    for arm in arms:
        accs = [1.0 if r["methods"][arm]["acc"] else 0.0 for r in per_query]
        by_level = {}
        for lv in LEVELS:
            lv_accs = [1.0 if r["methods"][arm]["acc"] else 0.0
                       for r in per_query if r["level"] == lv]
            by_level[lv] = {"acc": float(np.mean(lv_accs)) if lv_accs else None,
                            "n": len(lv_accs)}
        results[arm] = {"acc": float(np.mean(accs)) if accs else None,
                        "by_level": by_level, "n": len(per_query)}

    if verbose:
        print(f"\n  [{domain}] {'arm':<6} {'ACC':>7} "
              f"{'L1':>7} {'L2':>7} {'L3':>7} {'L4':>7}")
        for arm in arms:
            r = results[arm]
            cells = [f"{r['by_level'][lv]['acc']:.2f}"
                     if r["by_level"][lv]["acc"] is not None else "n/a"
                     for lv in LEVELS]
            print(f"  {'':<9} {arm:<6} {r['acc']:.3f} "
                  f"{cells[0]:>7} {cells[1]:>7} {cells[2]:>7} {cells[3]:>7}")
    return {"domain": domain, "results": results, "per_query": per_query,
            "n_entities": len(index.entities), "n_relations": n_orig_rel,
            "n_completed": len(extra),
            "completed_edges": [
                {"a": index.entities[e["a"]]["name"],
                 "b": index.entities[e["b"]]["name"],
                 "relation": e["relation"], "prob": round(e["prob"], 4)}
                for e in extra]}


# ── 换算与汇总 ───────────────────────────────────────────────────────────


def summarize(domain_out: dict) -> dict:
    domain = domain_out["domain"]
    results = domain_out["results"]
    lb_lightrag = leaderboard_mean(domain, "LightRAG")
    lb_rag = leaderboard_mean(domain, "RAG_wo_rerank")
    b0_acc = results["b0"]["acc"]
    factor = lb_rag / 100.0 / b0_acc if b0_acc else None
    summary = {"leaderboard": {
        "source": f"GraphRAG-Bench paper Table 2 (arXiv 2506.05690v3), {LEADERBOARD_URL}",
        "lightrag_acc_pct": lb_lightrag,
        "lightrag_by_level_pct": LEADERBOARD_ACC[domain]["LightRAG"],
        "rag_wo_rerank_acc_pct": lb_rag,
        "rag_wo_rerank_by_level_pct": LEADERBOARD_ACC[domain]["RAG_wo_rerank"],
    }, "b0_acc": b0_acc, "factor": factor, "arms": {}}
    for arm in ("a", "b", "ah"):
        if arm not in results:
            continue
        acc = results[arm]["acc"]
        retention = (acc * factor * 100.0 / lb_lightrag
                     if factor and acc is not None else None)
        summary["arms"][arm] = {
            "acc": acc,
            "retention_vs_leaderboard_lightrag": retention,
            "verdict": ("retained(>=0.9)" if retention is not None and retention >= 0.9
                        else "not_retained"),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 50 LightRAG J-Lens 替换实验")
    ap.add_argument("--domain", default="medical",
                    choices=["medical", "novel", "both"])
    ap.add_argument("--arm", default="all",
                    choices=["a", "b", "b0", "ah", "all"])
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = 全量（medical 56 / novel 48）")
    ap.add_argument("--smoke", type=int, default=0,
                    help="只跑检索不跑 LLM 评测（前 N 题，不落盘）")
    args = ap.parse_args()

    arms = ["b0", "a", "b"] if args.arm == "all" else [args.arm]
    domains = ["medical", "novel"] if args.domain == "both" else [args.domain]

    out = {}
    if OUT_PATH.exists() and not args.smoke:
        out = json.loads(OUT_PATH.read_text())
    out.setdefault("method", "phase50_lightrag_j")
    out["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out["leaderboard_source"] = LEADERBOARD_URL
    out["config"] = {
        "ent_top_k": ENT_TOP_K, "rel_top_k": REL_TOP_K,
        "neighbor_decay": NEIGHBOR_DECAY, "merge_ws_cos": MERGE_WS_COS,
        "cp_rank": CP_RANK, "cp_lam": CP_LAM, "cp_max_frac": CP_MAX_FRAC,
        "top_k": TOP_K, "arms": arms, "chunk_aggregation": "max",
    }
    out["notes"] = (
        "First run used SUM chunk aggregation and degenerated to a "
        "query-independent hub ranking on our tiny entity graph "
        "(medical arm-a ACC 0.018, novel 0.146). Switched to MAX "
        "aggregation (chunk inherits its best-matching graph element's "
        "weight, bge tie-break) — faithful to LightRAG's per-element "
        "text-unit ordering.")

    for domain in domains:
        print(f"\n── {domain} ──")
        res = run_domain(domain, arms, max_queries=args.max_queries,
                         smoke=args.smoke)
        if args.smoke:
            for entry in res["retrieval"][:3]:
                print(f"  {entry['qid']} [{entry['level']}] "
                      f"b0={entry['ranked']['b0'][:3]}")
                for arm in ("a", "b"):
                    if arm in entry["ranked"]:
                        dbg = entry["debug"][arm]
                        print(f"    {arm}: ents={dbg['hit_entities'][:5]}")
                        print(f"       rels={[h['text'] for h in dbg['hit_relations'][:5]]}")
                        print(f"       ranked={entry['ranked'][arm][:5]}")
            continue
        # Merge into any existing domain entry (single-arm re-runs keep the
        # other arms' results); per_query merged by qid, methods updated.
        prev = out.setdefault("domains", {}).get(domain, {})
        merged_results = dict(prev.get("results", {}))
        merged_results.update(res["results"])
        prev_pq = {r["qid"]: r for r in prev.get("per_query", [])}
        for rec in res["per_query"]:
            if rec["qid"] in prev_pq:
                prev_pq[rec["qid"]]["methods"].update(rec["methods"])
            else:
                prev_pq[rec["qid"]] = rec
        merged_pq = list(prev_pq.values())
        res_for_summary = {"domain": domain, "results": merged_results}
        out["domains"][domain] = {
            "results": merged_results,
            "n_entities": res["n_entities"],
            "n_relations": res["n_relations"],
            "n_completed": res["n_completed"] or prev.get("n_completed", 0),
            "completed_edges": (res["completed_edges"]
                                or prev.get("completed_edges", [])),
            "summary": summarize(res_for_summary),
            "per_query": merged_pq,
        }
        s = out["domains"][domain]["summary"]
        print(f"  factor = {s['factor']:.4f} "
              f"(leaderboard RAG {s['leaderboard']['rag_wo_rerank_acc_pct']:.1f}% "
              f"/ our B0 {s['b0_acc'] * 100:.1f}%)")
        for arm, a in s["arms"].items():
            r = a["retention_vs_leaderboard_lightrag"]
            print(f"  arm {arm}: ACC={a['acc']:.3f} retention={r:.3f} "
                  f"→ {a['verdict']}")
        OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"  saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
