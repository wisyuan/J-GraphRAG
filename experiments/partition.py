"""C_k 范畴构造——嵌入分形 Phase 1 的两种操作化方式。

C_k 是概念性的范畴（literature §3.5），实现方式随意。本模块提供两种：
  (a) partition_hdbscan         — 空间邻近聚类（Phase 0 已用，互斥划分）
  (b) partition_component_activation — 单分量高激活（文献理论本意，重叠划分）

两者都返回 dict[int, list[int]]：{范畴标签: [成员在 vecs 中的下标]}。
HDBSCAN 排除 noise(-1)；component-activation 选方差 top-k 维度。
"""
from __future__ import annotations

import numpy as np


def partition_hdbscan(
    vecs: list[list[float]],
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> dict[int, list[int]]:
    """HDBSCAN 聚类构造范畴（互斥划分）。

    复用 Phase 0 的参数：min_cluster_size=3, min_samples=2, metric="cosine"。
    返回 {label: [member_indices]}，排除 noise(-1)。
    """
    from sklearn.cluster import HDBSCAN

    data = np.asarray(vecs, dtype=np.float64)
    labels = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="cosine",
    ).fit_predict(data)

    clusters: dict[int, list[int]] = {}
    for i, label in enumerate(labels):
        if label >= 0:  # 排除 noise
            clusters.setdefault(int(label), []).append(i)
    return clusters


def partition_component_activation(
    vecs: list[list[float]],
    n_dims: int | None = None,
    top_fraction: float = 0.2,
) -> dict[int, list[int]]:
    """单分量高激活构造范畴（重叠划分，文献理论本意）。

    对方差最大的前 n_dims 个维度 d，定义：
        C_d = { 文本 i : vecs[i][d] 排在该维度的 top top_fraction 分位 }

    一个文本可属于多个 C_d（范畴可重叠），这是和 HDBSCAN 的本质区别。

    Args:
        vecs: 嵌入向量 (N, 1024)
        n_dims: 选多少个方差最大的维度建范畴。None=自动（min(50, N//10)）。
        top_fraction: 每维度取激活最高的多少比例文本。0.2=前20%。
    """
    data = np.asarray(vecs, dtype=np.float64)
    n_texts, total_dims = data.shape

    if n_dims is None:
        n_dims = min(50, max(1, n_texts // 10))
    n_dims = min(n_dims, total_dims)

    # 按方差排序选 top-k 维度（和 Phase 6 J-Space 思路一致）。
    variances = data.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:n_dims]

    threshold_count = max(1, int(n_texts * top_fraction))
    clusters: dict[int, list[int]] = {}
    for rank, dim in enumerate(top_dims):
        # 该维度激活值最高的 threshold_count 个文本的下标
        top_indices = np.argsort(data[:, dim])[::-1][:threshold_count]
        clusters[int(dim)] = top_indices.tolist()
    return clusters


if __name__ == "__main__":  # pragma: no cover
    # 冒烟测试：用随机向量验证两种构造都能产出非空范畴。
    rng = np.random.default_rng(42)
    fake = rng.standard_normal((100, 1024)).tolist()

    h = partition_hdbscan(fake)
    print(f"HDBSCAN: {len(h)} clusters, sizes: {sorted(len(v) for v in h.values())[:10]}...")

    c = partition_component_activation(fake, n_dims=10, top_fraction=0.2)
    print(f"Component-activation: {len(c)} categories, "
          f"sizes: {sorted(len(v) for v in c.values())[:10]}...")
    # 验证重叠性：一个文本可属于多个范畴
    all_member_sets = [set(v) for v in c.values()]
    overlap = len(set.intersection(*all_member_sets)) if all_member_sets else 0
    print(f"  (overlap check: {overlap} texts in ALL categories — nonzero means highly overlapping)")
