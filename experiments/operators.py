"""展开算子——嵌入分形 Phase 1 的核心。

每个算子接收一个范畴内的成员向量 (n, 1024)，返回展开后的子向量 (n, dim')。
dim' 可能 ≠ 1024（如 PCA 降到 50），但 mean_pairwise_distance 不关心维度。

四个算子覆盖完整光谱：
  1. op_mahalanobis  — 条件马氏变换 s_k = Σ^{-1/2}(v - μ)，理论核心
  2. op_residual     — 残差嵌入 r_k = v - μ，弱基线
  3. op_local_pca    — 局部 PCA 展开，降维聚焦簇内差异
  4. op_llm_reembed  — LLM 精细重嵌入，ceiling（需 embed+llm provider）
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def op_mahalanobis(members: np.ndarray, ridge: float = 1e-5, n_dims: int | None = None) -> np.ndarray:
    """条件马氏变换：s_k(x) = Σ_k^{-1/2} (v(x) - μ_k)。

    白化（whitening）：去均值 + 用条件协方差的逆平方根缩放。
    展开被全局嵌入压缩的簇内精细方差结构。

    Args:
        members: (n, 1024) 范畴内成员向量
        ridge: 岭正则化系数（1024 维 + 小样本 → Σ 秩亏损，必须正则化）
        n_dims: 只在方差最大的前 n_dims 维上做变换（降维加速）。
                None=全 1024 维（慢，7.7s/call）；50=快 ~400x（和 J-Space 思路一致）。
    Returns:
        (n, dim') 白化后的子向量。dim' = n_dims 或 1024。
    """
    mu = members.mean(axis=0)
    centered = members - mu

    if n_dims is not None and n_dims < members.shape[1]:
        # 选方差最大的 n_dims 维（和 Phase 6 J-Space 识别思路一致）
        variances = members.var(axis=0)
        top_dims = np.argsort(variances)[::-1][:n_dims]
        centered = centered[:, top_dims]  # (n, n_dims)

    if members.shape[0] < 2:
        return centered
    sigma = np.cov(centered, rowvar=False)
    sigma_reg = sigma + ridge * np.eye(sigma.shape[0])

    sigma_inv_sqrt = sqrtm(np.linalg.pinv(sigma_reg))
    if np.iscomplexobj(sigma_inv_sqrt):
        sigma_inv_sqrt = np.real(sigma_inv_sqrt)

    return centered @ sigma_inv_sqrt.T


def op_residual(members: np.ndarray) -> np.ndarray:
    """残差嵌入：r_k(x) = v(x) - μ_k。

    最简单的算子（类似 batch norm 去均值）。
    移除簇共同分量，保留个体差异。不放大方差，只移除均值。
    作为弱基线——如果残差都比全局好，说明去均值本身就有效。
    """
    mu = members.mean(axis=0)
    return members - mu


def op_local_pca(members: np.ndarray, n_components: int = 50) -> np.ndarray:
    """局部 PCA 展开：簇内 PCA，投影到 top-k 主成分子空间。

    聚焦簇内最大方差方向——这些方向代表"簇内细粒度差异"，
    正是被全局嵌入压缩的子概念结构。

    Args:
        members: (n, 1024)
        n_components: 保留多少主成分（1024→n_components）
    Returns:
        (n, n_components) 主成分坐标
    """
    from sklearn.decomposition import PCA

    n_components = min(n_components, members.shape[0], members.shape[1])
    if n_components < 1:
        n_components = 1
    pca = PCA(n_components=n_components)
    return pca.fit_transform(members)  # (n, k)，已自动中心化


def op_llm_reembed(
    member_texts: list[str],
    embed,
    llm,
) -> list[list[float]]:
    """LLM 精细重嵌入（ceiling baseline）。

    逐字复用 concept_tree_validation.py:experiment2_fine_reembedding 的 prompt：
    对每个成员，LLM 生成"什么让这个符号独特"的对比描述，bge-m3 重嵌入。
    这是 Phase 0 已验证 1.88x 的上限。

    Args:
        member_texts: 范畴内成员的原始文本
        embed: CachedBgeM3Provider（或任何有 .embed(list[str]) 的 provider）
        llm: DeepSeekProvider（或有 .complete(prompt, max_tokens) 的 provider）
    Returns:
        list[list[float]] 重嵌入向量（长度 = len(member_texts)）
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _gen_fine_desc(idx_and_text):
        idx, text = idx_and_text
        # 取同簇其他成员作对比上下文（最多2个，各截500字符）。
        others = [t for j, t in enumerate(member_texts) if j != idx][:3]
        contrast = "\n---\n".join(others[:2])
        prompt = (
            f"Below is one code symbol from a group of related symbols.\n"
            f"Describe what makes THIS symbol UNIQUE compared to the others.\n"
            f"Focus on its specific functionality, not general patterns.\n\n"
            f"This symbol:\n{text[:500]}\n\n"
            f"Other symbols in the same group:\n{contrast[:500]}\n\n"
            f"Unique aspect (1-2 sentences):"
        )
        msg = llm.complete(prompt, 128)
        return idx, msg.content if not msg.is_error else text[:200]

    fine_descs: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_gen_fine_desc, (i, member_texts[i])): i
            for i in range(len(member_texts))
        }
        for future in as_completed(futures):
            idx, desc = future.result()
            fine_descs[idx] = desc

    desc_texts = [fine_descs.get(i, member_texts[i][:200]) for i in range(len(member_texts))]
    return embed.embed(desc_texts)


# ── 纯数学算子的注册表（phase1_bakeoff 用名字查函数）──────────────────────
MATH_OPERATORS = {
    "mahalanobis": (op_mahalanobis, {"ridge": 1e-5, "n_dims": 50}),
    "residual": (op_residual, {}),
    "local_pca": (op_local_pca, {"n_components": 50}),
}
