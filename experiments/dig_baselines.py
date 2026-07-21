"""Phase 4-dig 的 3 个 dig baseline。

D0: flat dig（纯余弦最近邻，当前产品 dig related）
D1: 概念树图传播 dig（Sigmoid 图 + GraphRAG 传播，原始向量不展开）
D2: 确定性 KG 图遍历 dig（code_kg.json REFERENCES/IMPORTS 多跳，仅 pi-code）

核心命题：D1/D2 在 bridge 问题（跨文档关联）上胜过 D0，
在 comparison 问题（对照）上持平。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    DEVICE = None


def _to_tensor(arr):
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)

def _normalize_t(t):
    norm = t.norm(dim=-1, keepdim=True)
    return t / norm.clamp(min=1e-8)

def _cosine_np(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ── D0: Flat Dig ────────────────────────────────────────────────────────

class FlatDig:
    """纯余弦最近邻 = 当前产品 dig related mode。Baseline。"""

    name = "D0-flat"

    def fit(self, corpus_emb, corpus_ids):
        self.corpus_ids = corpus_ids
        self.c_norm = _normalize_t(_to_tensor(corpus_emb))

    def search(self, query_emb, query_ids, top_k=100):
        q_norm = _normalize_t(_to_tensor(query_emb))
        scores = (q_norm @ self.c_norm.T).cpu().numpy()
        results = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(scores[i])[::-1][:top_k]
            results[qid] = {self.corpus_ids[j]: float(scores[i, j]) for j in top_idx}
        return results


# ── D1: Concept Tree Graph Propagation Dig ──────────────────────────────

class ConceptTreeDig:
    """概念树 Sigmoid 图 + GraphRAG 传播 dig（v2: 级联扩展，不重排）。

    ★ v2 修正（Phase 4-dig 首轮教训）：
    v1 用 geometric mean 重排全库 → centroid 信号稀释 individual cosine → 有害。
    v2 级联扩展：B0 召回 top-K → 图传播只追加跨簇邻居（扩展召回集）→ 不重排已有结果。

    流程：
      1. B0 召回 top-recall_k（原始余弦，不重排）
      2. 在 top-recall_k 内，找 query 最近的簇 → 沿 Sigmoid 图找邻居簇
      3. 邻居簇的成员追加到候选集（用 individual cosine 评分，但低于原始召回）
      4. 返回：原始召回 top-k + 跨簇邻居追加

    图传播只**增加召回**（找到 D0 漏掉的跨簇文档），不**破坏排序**。
    """

    name = "D1-concept-tree"

    def __init__(self, target_edges_per_cluster=6.5, recall_k=100, expand_weight=0.8):
        self.target_epc = target_edges_per_cluster
        self.recall_k = recall_k
        self.expand_weight = expand_weight  # 跨簇邻居的分数 = cosine * expand_weight

    def fit(self, corpus_emb, corpus_ids):
        from sklearn.cluster import HDBSCAN

        self.corpus_ids = corpus_ids
        self.corpus_emb = corpus_emb

        labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(corpus_emb)
        self.labels = labels

        self.cluster_members = {}
        self.centroids = {}
        for label in sorted(set(labels)):
            if label < 0:
                continue
            mask = labels == label
            members = np.where(mask)[0]
            self.cluster_members[label] = members
            self.centroids[label] = corpus_emb[members].mean(axis=0)

        self.cluster_ids = sorted(self.centroids.keys())

        if len(self.cluster_ids) < 2:
            self.c_norm = _normalize_t(_to_tensor(corpus_emb))
            self._degenerate = True
            return
        self._degenerate = False

        self._build_sigmoid_graph()
        self.c_norm = _normalize_t(_to_tensor(corpus_emb))

    def _build_sigmoid_graph(self):
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)

        sims = []
        for i, ca in enumerate(cluster_ids):
            for cb in cluster_ids[i + 1:]:
                sims.append(_cosine_np(self.centroids[ca], self.centroids[cb]))

        if len(sims) < 2:
            self.graph = {c: [] for c in cluster_ids}
            return

        n_target_edges = int(self.target_epc * n_clusters / 2)
        n_target_edges = max(1, min(n_target_edges, len(sims)))
        sorted_sims = np.sort(sims)[::-1]
        theta = float(sorted_sims[min(n_target_edges - 1, len(sorted_sims) - 1)])

        k = 20.0
        self.graph = defaultdict(list)
        for i, ca in enumerate(cluster_ids):
            for cb in cluster_ids[i + 1:]:
                raw_sim = _cosine_np(self.centroids[ca], self.centroids[cb])
                transformed = 1.0 / (1.0 + math.exp(-k * (raw_sim - theta)))
                if transformed >= 0.5:
                    self.graph[ca].append((cb, transformed))
                    self.graph[cb].append((ca, transformed))
        self.graph = dict(self.graph)

    def search(self, query_emb, query_ids, top_k=100):
        if self._degenerate:
            q_norm = _normalize_t(_to_tensor(query_emb))
            scores = (q_norm @ self.c_norm.T).cpu().numpy()
            results = {}
            for i, qid in enumerate(query_ids):
                top_idx = np.argsort(scores[i])[::-1][:top_k]
                results[qid] = {self.corpus_ids[j]: float(scores[i, j]) for j in top_idx}
            return results

        results = {}
        for qi, qid in enumerate(query_ids):
            q = query_emb[qi]
            scores_dict = self._cascade_expand_one(q, top_k)
            results[qid] = scores_dict
        return results

    def _cascade_expand_one(self, q, top_k):
        """★ v2 级联扩展：B0 召回 + 跨簇邻居追加（不重排）。"""

        # Step 1: B0 召回 top-recall_k（原始余弦）
        q_t = _normalize_t(_to_tensor(q).unsqueeze(0))
        individual_sims = (q_t @ self.c_norm.T).squeeze(0).cpu().numpy()

        recall_k = min(self.recall_k, len(individual_sims))
        recall_indices = np.argsort(individual_sims)[::-1][:recall_k]

        # 原始召回的分数（保持不变）
        scores = {}
        for idx in recall_indices:
            scores[self.corpus_ids[idx]] = float(individual_sims[idx])

        # Step 2: 找 query 最近的簇 → 沿 Sigmoid 图找邻居簇
        best_cluster = max(self.cluster_ids,
                           key=lambda c: _cosine_np(q, self.centroids[c]))

        neighbor_clusters = set()
        for neighbor, _ in self.graph.get(best_cluster, []):
            neighbor_clusters.add(neighbor)

        # Step 3: 邻居簇的成员追加到候选集（用 individual cosine * expand_weight）
        for nc in neighbor_clusters:
            for member_idx in self.cluster_members.get(nc, []):
                cid = self.corpus_ids[member_idx]
                if cid not in scores:
                    # 追加：分数 = cosine * expand_weight（低于原始召回，但进入候选集）
                    scores[cid] = float(individual_sims[member_idx]) * self.expand_weight

        # Step 4: 全部候选按分数排序，返回 top_k
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_scores)


# ── D2: Deterministic KG Graph Traversal Dig ────────────────────────────

class KGDig:
    """code_kg.json 的 REFERENCES_SYMBOL/IMPORTS 图遍历。

    对 B0 召回的 top-K 文件，沿 KG 边扩展 1-hop，找跨文件关联。
    仅适用于代码域（有 code_kg.json）。
    """

    name = "D2-kg"

    def __init__(self, expand_k=20, max_hop=1, expand_weight=0.5):
        self.expand_k = expand_k      # B0 召回多少文档后扩展
        self.max_hop = max_hop         # KG 扩展跳数
        self.expand_weight = expand_weight  # KG 扩展文档的分数权重

    def fit(self, corpus_emb, corpus_ids, code_kg_path=None):
        self.corpus_ids = corpus_ids
        self.corpus_id_set = set(corpus_ids)
        self.c_norm = _normalize_t(_to_tensor(corpus_emb))

        # 加载 KG 构建 file→file 邻接表
        self.adjacency = defaultdict(set)  # {file: {related_files}}

        if code_kg_path and Path(code_kg_path).exists():
            kg = json.loads(Path(code_kg_path).read_text())
            for edge in kg.get("edges", []):
                src = edge["source"]
                tgt = edge["target"]
                relation = edge.get("relation", "")

                if relation == "IMPORTS":
                    # file → file
                    self.adjacency[src].add(tgt)
                    self.adjacency[tgt].add(src)
                elif relation in ("REFERENCES_SYMBOL", "DEFINES"):
                    # file → symbol → 从 symbol 提取所属 file
                    if "::" in tgt:
                        tgt_file = tgt.rsplit("::", 1)[0]
                        if tgt_file != src:
                            self.adjacency[src].add(tgt_file)
                            self.adjacency[tgt_file].add(src)

        # 构建 corpus_id → 在 corpus 中的 index 映射
        # corpus_id 格式对 pi-code 是 "{path}::{name}"
        # KG 边是 file-level，需要从 FineRecord id 映射回 file
        self.fine_to_file = {}
        for cid in corpus_ids:
            if "::" in cid:
                self.fine_to_file[cid] = cid.rsplit("::", 1)[0]
            else:
                self.fine_to_file[cid] = cid

        # file → FineRecord ids
        self.file_to_fines = defaultdict(list)
        for cid, fpath in self.fine_to_file.items():
            self.file_to_fines[fpath].append(cid)

    def search(self, query_emb, query_ids, top_k=100):
        q_norm = _normalize_t(_to_tensor(query_emb))

        results = {}
        for qi, qid in enumerate(query_ids):
            q = q_norm[qi:qi+1]
            base_scores = (q @ self.c_norm.T).squeeze(0).cpu().numpy()

            # B0 召回 top-expand_k
            expand_idx = np.argsort(base_scores)[::-1][:self.expand_k]

            # 收集扩展候选（KG 1-hop）
            expanded = set(expand_idx.tolist())
            for idx in expand_idx:
                cid = self.corpus_ids[idx]
                fpath = self.fine_to_file.get(cid, "")
                for related_file in self.adjacency.get(fpath, set()):
                    for related_cid in self.file_to_fines.get(related_file, []):
                        if related_cid in self.corpus_id_set:
                            expanded.add(self.corpus_ids.index(related_cid))

            # 排序：原始召回用原始分数，KG 扩展用降权分数
            scores = {}
            for idx in expand_idx:
                scores[self.corpus_ids[idx]] = float(base_scores[idx])

            for cid_idx in expanded:
                cid = self.corpus_ids[cid_idx]
                if cid not in scores:
                    scores[cid] = float(base_scores[cid_idx]) * self.expand_weight

            # top_k
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            results[qid] = dict(sorted_scores)

        return results
