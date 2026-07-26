"""Phase 59: 增量建图行为漂移——真增量追加 vs 全量重建（产品化决策输入，不进论文）。

来源：AGENTS.md §6 条目 9（dev 设计讨论 2026-07-23）。dev v1 的 insert 语义
= 追加语料 + 全量重建；本实验验证真增量（新 chunk 直接入图、统计增量累加）
相对全量重建的漂移程度。SUPPORTED 则把增量姿势带回 dev v2。

协议（novel，20 本书按书切分：前 16 本=旧库，后 4 本=新追加——模拟新产品
文档到达场景；概念产物复用 twopass 缓存，纯 CPU 零模型）：
  G_full = 全部 20 本全量重建（ground truth）
  G_inc  = 旧库 16 本建图（DF≥2 剪枝在旧统计下执行）→ 追加 4 本
    变体 tombstone：旧库被剪概念保留墓碑，追加后 DF 达标可复活
    变体 no-tombstone：剪掉即删除，不复活

判决指标（预设，先于运行定义）：
  主判决（tombstone 变体 vs G_full）：
    节点 Jaccard ≥ 0.98 且 检索 top-10 重叠 ≥ 0.95 且 Kendall τ ≥ 0.90
    → 增量姿势 SUPPORTED
  次要：no-tombstone 的节点损失（墓碑机制的代价量化）；top-100 共现对
    选择漂移（J-Lens 读出边集合的稳定性）；IDF 陈旧 vs 刷新的检索漂移

用法：
  python -m experiments.phase59_incremental_graph --selftest   # 合成数据
  python -m experiments.phase59_incremental_graph              # 全量（CPU）
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_questions, TOP_K,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase59_incremental_graph.json"

DF_MIN = 2                 # 与生产管线一致的 DF 剪枝阈值
GATE_JACCARD = 0.98
GATE_OVERLAP = 0.95
GATE_TAU = 0.90
TOP_PAIRS = 100            # 对应 phase53 的 J-Lens 读出边选择规模


# ── 图构建原语（全部基于 concept_chunks 纯统计）──


def split_books(chunk_ids: list[str], frac_old: float = 0.8
                ) -> tuple[set[str], set[str]]:
    books = sorted({cid.split("::")[0] for cid in chunk_ids})
    n_old = int(len(books) * frac_old)
    old_books = set(books[:n_old])
    old = {cid for cid in chunk_ids if cid.split("::")[0] in old_books}
    return old, set(chunk_ids) - old


def df_map(concept_chunks: dict, chunk_subset: set[str] | None) -> Counter:
    """概念 DF（可选限定 chunk 子集统计）。"""
    df = Counter()
    for concept, cids in concept_chunks.items():
        n = len(cids if chunk_subset is None
                else [c for c in cids if c in chunk_subset])
        if n:
            df[concept] = n
    return df


def node_set(df: Counter, chunk_subset: set[str] | None = None,
             concept_chunks: dict | None = None) -> set[str]:
    """DF≥2 剪枝后的节点集。chunk_subset 限定统计范围（旧库）。"""
    if chunk_subset is None:
        return {c for c, n in df.items() if n >= DF_MIN}
    out = set()
    for c, n in df.items():
        cids = [x for x in concept_chunks[c] if x in chunk_subset]
        if len(cids) >= DF_MIN:
            out.add(c)
    return out


def cooc_edges(concept_chunks: dict, nodes: set[str],
               chunk_subset: set[str] | None) -> set[tuple[str, str]]:
    """共现边：共享 ≥1 chunk 的节点对。"""
    cid_concepts: dict[str, list[str]] = defaultdict(list)
    for c in nodes:
        for cid in concept_chunks[c]:
            if chunk_subset is None or cid in chunk_subset:
                cid_concepts[cid].append(c)
    edges = set()
    for cid, cs in cid_concepts.items():
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                edges.add(tuple(sorted((cs[i], cs[j]))))
    return edges


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 1.0


def idf_vec(concepts: list[str], df: Counter, n_docs: int) -> dict[str, float]:
    return {c: float(np.log(n_docs / max(1, df.get(c, 0)))) for c in concepts}


def retrieve_topk(question: str, chunk_concepts: dict[str, set[str]],
                  idf: dict[str, float], k: int) -> list[str]:
    """q×M 概念传播：问题文本匹配概念 → chunk 打分 = Σ idf(matched∈chunk)。"""
    ql = question.lower()
    matched = [c for c in idf if len(c) >= 3 and c.lower() in ql]
    scores = {cid: sum(idf.get(c, 0.0) for c in cs if c in matched)
              for cid, cs in chunk_concepts.items()}
    return [cid for cid, _s in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def kendall_tau(a: list[str], b: list[str]) -> float:
    from scipy.stats import kendalltau
    union = list(dict.fromkeys(a + b))
    ra = {cid: i for i, cid in enumerate(a)}
    rb = {cid: i for i, cid in enumerate(b)}
    xa = [ra.get(c, len(a)) for c in union]
    xb = [rb.get(c, len(b)) for c in union]
    tau, _p = kendalltau(xa, xb)
    return float(tau)


# ── 主流程 ──


def run(domain: str = "novel") -> dict:
    cache, _vecs, _rel = load_phase41_inputs(domain)
    concept_chunks = {c: list(cids) for c, cids in cache["concept_chunks"].items()}
    chunk_ids = sorted(cache["chunks"].keys())
    chunk_concepts: dict[str, set[str]] = {cid: set() for cid in chunk_ids}
    for c, cids in concept_chunks.items():
        for cid in cids:
            if cid in chunk_concepts:
                chunk_concepts[cid].add(c)

    old_chunks, new_chunks = split_books(chunk_ids)
    print(f"  {domain}: {len(chunk_ids)} chunks = {len(old_chunks)} old "
          f"+ {len(new_chunks)} new", flush=True)

    df_full = df_map(concept_chunks, None)
    df_old = df_map(concept_chunks, old_chunks)

    # 节点集
    nodes_full = {c for c, n in df_full.items() if n >= DF_MIN}
    nodes_old_pruned = {c for c, n in df_old.items() if n >= DF_MIN}
    # tombstone：旧库被剪节点在追加后若全量 DF 达标则复活 → 等价于 nodes_full
    nodes_inc_tomb = set(nodes_full)
    # no-tombstone：旧库剪掉即消失；新 chunk 里的新概念按追加后 DF 判
    concepts_in_new = {c for c in concept_chunks
                       if any(cid in new_chunks for cid in concept_chunks[c])}
    nodes_inc_notomb = nodes_old_pruned | {
        c for c in concepts_in_new if df_full[c] >= DF_MIN}

    # 共现边
    edges_full = cooc_edges(concept_chunks, nodes_full, None)
    edges_inc_tomb = cooc_edges(concept_chunks, nodes_inc_tomb, None)
    edges_inc_notomb = cooc_edges(concept_chunks, nodes_inc_notomb, None)

    # top-pair 选择漂移（J-Lens 读出边集合选择）
    def top_pairs(chunk_subset):
        pair_cnt = Counter()
        for c in nodes_full:
            for cid in concept_chunks[c]:
                if chunk_subset is not None and cid not in chunk_subset:
                    continue
                pair_cnt[c] += 1  # placeholder, replaced below
        # 真实共现对计数
        cid_concepts: dict[str, list[str]] = defaultdict(list)
        for c in nodes_full:
            for cid in concept_chunks[c]:
                if chunk_subset is None or cid in chunk_subset:
                    cid_concepts[cid].append(c)
        pc = Counter()
        for cid, cs in cid_concepts.items():
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pc[tuple(sorted((cs[i], cs[j])))] += 1
        return {p for p, _n in pc.most_common(TOP_PAIRS)}

    pairs_full = top_pairs(None)
    pairs_old = top_pairs(old_chunks)

    # 检索漂移（IDF 陈旧 vs 刷新），48 题
    questions = load_questions(domain, 48)
    idf_stale = idf_vec(list(nodes_full), df_old, len(old_chunks))
    idf_fresh = idf_vec(list(nodes_full), df_full, len(chunk_ids))
    overlaps, taus = [], []
    for q in questions:
        top_full = retrieve_topk(q["question"], chunk_concepts, idf_fresh, TOP_K)
        top_stale = retrieve_topk(q["question"], chunk_concepts, idf_stale, TOP_K)
        overlaps.append(len(set(top_full) & set(top_stale)) / TOP_K)
        taus.append(kendall_tau(top_full, top_stale))

    result = {
        "domain": domain,
        "split": {"old_chunks": len(old_chunks), "new_chunks": len(new_chunks)},
        "nodes": {
            "full": len(nodes_full),
            "old_pruned": len(nodes_old_pruned),
            "inc_tombstone": len(nodes_inc_tomb),
            "inc_no_tombstone": len(nodes_inc_notomb),
            "jaccard_tombstone_vs_full": jaccard(nodes_inc_tomb, nodes_full),
            "jaccard_no_tombstone_vs_full": jaccard(nodes_inc_notomb, nodes_full),
            "resurrected": len(nodes_full - nodes_old_pruned),
            "lost_without_tombstone": len(nodes_full - nodes_inc_notomb),
        },
        "edges": {
            "full": len(edges_full),
            "jaccard_tombstone_vs_full": jaccard(edges_inc_tomb, edges_full),
            "jaccard_no_tombstone_vs_full": jaccard(edges_inc_notomb, edges_full),
        },
        "top_pairs_selection": {
            "jaccard_old_vs_full": jaccard(pairs_old, pairs_full),
            "note": "J-Lens 读出边（top-100 共现对）的选择集稳定性",
        },
        "retrieval_drift_stale_vs_fresh_idf": {
            "top10_overlap_mean": float(np.mean(overlaps)),
            "kendall_tau_mean": float(np.mean(taus)),
            "n_questions": len(questions),
        },
    }
    # 主判决（tombstone 变体）
    nj = result["nodes"]["jaccard_tombstone_vs_full"]
    ov = result["retrieval_drift_stale_vs_fresh_idf"]["top10_overlap_mean"]
    tau = result["retrieval_drift_stale_vs_fresh_idf"]["kendall_tau_mean"]
    ok = (nj >= GATE_JACCARD and ov >= GATE_OVERLAP and tau >= GATE_TAU)
    result["verdict"] = {
        "gates": {"node_jaccard": GATE_JACCARD, "overlap": GATE_OVERLAP,
                  "tau": GATE_TAU},
        "measured": {"node_jaccard": nj, "overlap": ov, "tau": tau},
        "incremental_supported": bool(ok),
        "note": "检索漂移行是 IDF 陈旧 vs 刷新——增量系统不刷新 IDF 时的真实漂移",
    }
    return result


def _selftest():
    """合成小语料：split/df/节点/边/Jaccard/检索/τ。"""
    chunk_ids = [f"Book-{i}::chunk_0" for i in range(10)]
    old, new = split_books(chunk_ids, 0.8)
    assert len(old) == 8 and len(new) == 2
    cc = {"alpha": ["Book-0::chunk_0", "Book-1::chunk_0", "Book-9::chunk_0"],
          "beta": ["Book-0::chunk_0", "Book-2::chunk_0"],
          "gamma": ["Book-9::chunk_0"]}
    df_full = df_map(cc, None)
    df_old = df_map(cc, old)
    assert df_full["alpha"] == 3 and df_old["alpha"] == 2
    assert df_full["gamma"] == 1  # 全量也不达标
    nodes_full = {c for c, n in df_full.items() if n >= DF_MIN}
    assert nodes_full == {"alpha", "beta"}
    nodes_old = {c for c, n in df_old.items() if n >= DF_MIN}
    assert nodes_old == {"alpha", "beta"}  # alpha 旧库恰好达标
    edges = cooc_edges(cc, nodes_full, None)
    assert ("alpha", "beta") in edges
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    # 检索：stale idf 下 alpha 的 idf 不同但排序一致
    chunk_concepts = {cid: set() for cid in chunk_ids}
    for c, cids in cc.items():
        for cid in cids:
            chunk_concepts[cid].add(c)
    idf_a = idf_vec(list(nodes_full), df_old, 8)
    idf_b = idf_vec(list(nodes_full), df_full, 10)
    ta = retrieve_topk("alpha and beta", chunk_concepts, idf_a, 2)
    tb = retrieve_topk("alpha and beta", chunk_concepts, idf_b, 2)
    assert ta[0] == tb[0] == "Book-0::chunk_0"
    assert kendall_tau(["a", "b"], ["a", "b"]) == 1.0
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--domain", default="novel")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    result = run(args.domain)
    json.dump(result, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
