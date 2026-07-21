"""多层级概念树构建（残差 + HDBSCAN 递归）。

Phase 2 证实：残差递归 self-similar（persistence=1.03），
每层只需矩阵减法 + HDBSCAN，不需要协方差分解。

树结构：
  depth 0: 全局 HDBSCAN → 顶层簇 C_k^(1)
  depth 1: 每个顶层簇内 残差 + HDBSCAN → 子簇 C_{k,j}^(2)
  depth 2: 每个子簇内 残差 + HDBSCAN → 子子簇 C_{k,j,l}^(3)
  直到 d_max 或终止条件
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClusterNode:
    """概念树的一个簇节点。"""
    cluster_id: str               # 层级 ID，如 "5", "5.2", "5.2.1"
    depth: int                    # 0=顶层，1=子层...
    member_indices: list[int]     # 在原始 vecs 中的下标
    centroid: np.ndarray          # 质心（原始空间）
    children: list["ClusterNode"] = field(default_factory=list)
    is_compressed: bool = False   # global_dist ≈ 0

    @property
    def n_members(self) -> int:
        return len(self.member_indices)


@dataclass
class ConceptTree:
    """多层级概念树。"""
    roots: list[ClusterNode] = field(default_factory=list)
    all_nodes: list[ClusterNode] = field(default_factory=list)

    def nodes_at_depth(self, depth: int) -> list[ClusterNode]:
        return [n for n in self.all_nodes if n.depth == depth]

    @property
    def max_depth(self) -> int:
        return max((n.depth for n in self.all_nodes), default=0)

    def depth_distribution(self) -> dict[int, int]:
        dist = defaultdict(int)
        for n in self.all_nodes:
            dist[n.depth] += 1
        return dict(dist)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _mean_pairwise_distance_np(vecs: np.ndarray) -> float:
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = vecs / norms
    cos_matrix = normalized @ normalized.T
    iu = np.triu_indices(n, k=1)
    return float((1.0 - cos_matrix[iu]).mean())


COMPRESSED_THRESHOLD = 1e-6
MIN_SUBCLUSTER = 4


def build_concept_tree(
    vecs: np.ndarray,
    d_max: int = 3,
    min_subcluster: int = MIN_SUBCLUSTER,
) -> ConceptTree:
    """递归构建多层级概念树。

    Args:
        vecs: (N, 1024) 原始嵌入向量
        d_max: 最大深度（0=顶层聚类，1=子层...）
        min_subcluster: 子簇最小成员数
    Returns:
        ConceptTree
    """
    from sklearn.cluster import HDBSCAN

    tree = ConceptTree()
    labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(vecs)

    for label in sorted(set(labels)):
        if label < 0:
            continue  # noise
        members = np.where(labels == label)[0]
        if len(members) < min_subcluster:
            continue
        centroid = vecs[members].mean(axis=0)
        node = ClusterNode(
            cluster_id=str(int(label)),
            depth=0,
            member_indices=members.tolist(),
            centroid=centroid,
        )
        tree.roots.append(node)
        tree.all_nodes.append(node)

        _recursive_split(vecs, node, d_max, min_subcluster, prefix=str(int(label)))

    return tree


def _recursive_split(
    vecs: np.ndarray,
    node: ClusterNode,
    d_max: int,
    min_subcluster: int,
    prefix: str,
):
    """递归分裂一个节点：残差 + HDBSCAN。"""
    if node.depth >= d_max:
        return
    if node.n_members < min_subcluster:
        return

    members = vecs[node.member_indices]

    # 检查是否完全压缩
    global_dist = _mean_pairwise_distance_np(members)
    if global_dist < COMPRESSED_THRESHOLD:
        node.is_compressed = True
        return

    # 残差（去均值）
    residual = members - members.mean(axis=0)

    # 在残差空间 HDBSCAN
    from sklearn.cluster import HDBSCAN
    try:
        sub_labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(residual)
    except Exception:
        return

    sub_cluster_idx = 0
    for label in sorted(set(sub_labels)):
        if label < 0:
            continue
        sub_members_local = np.where(sub_labels == label)[0]
        if len(sub_members_local) < min_subcluster:
            continue

        # 映射回原始 vecs 的全局下标
        global_indices = [node.member_indices[i] for i in sub_members_local]
        sub_vecs = vecs[global_indices]
        centroid = sub_vecs.mean(axis=0)

        child = ClusterNode(
            cluster_id=f"{prefix}.{sub_cluster_idx}",
            depth=node.depth + 1,
            member_indices=global_indices,
            centroid=centroid,
        )
        node.children.append(child)
        tree = None  # all_nodes 由调用者管理（这里通过 node.children 链接）
        sub_cluster_idx += 1

        _recursive_split(vecs, child, d_max, min_subcluster, child.cluster_id)


def build_sigmoid_graph(
    nodes: list[ClusterNode],
    theta: float,
    k: float = 20.0,
) -> dict[str, list[tuple[str, float]]]:
    """同一层簇间的 Sigmoid 加权图。

    复用 concept_tree_validation.py:build_sigmoid_graph 的逻辑，
    但 theta 从 bootstrap 统计化来。

    Returns: {cluster_id: [(neighbor_id, weight), ...]}
    """
    edges: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for i, na in enumerate(nodes):
        for nb in nodes[i + 1:]:
            raw_sim = _cosine(na.centroid, nb.centroid)
            transformed = 1.0 / (1.0 + math.exp(-k * (raw_sim - theta)))
            if transformed >= 0.5:  # edge_threshold 固定 0.5（等价于 raw_sim >= theta）
                edges[na.cluster_id].append((nb.cluster_id, transformed))
                edges[nb.cluster_id].append((na.cluster_id, transformed))
    return dict(edges)
