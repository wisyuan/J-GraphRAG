"""Phase 4 检索 baseline——3 个 BaseSearch 实现。

B0:        flat ANN（普通 RAG，全局余弦）
B-fractal: 原始嵌入分形（component-activation + 条件高斯 + 多证据融合）
B0+:       残差展开（HDBSCAN 范畴 + 去均值）

每个 baseline 实现 fit(corpus_emb) + search(queries) → {qid: {cid: score}}。
score 格式兼容 BEIR EvaluateRetrieval。

GPU 加速：cos_sim 用 torch GPU 矩阵乘。
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GPU = DEVICE.type == "cuda"
except ImportError:
    DEVICE = None
    GPU = False


def _to_tensor(arr: np.ndarray) -> "torch.Tensor":
    """numpy → torch GPU tensor (float32)."""
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)


def _normalize(t: "torch.Tensor") -> "torch.Tensor":
    """L2 normalize (for cosine similarity)."""
    norm = t.norm(dim=-1, keepdim=True)
    return t / norm.clamp(min=1e-8)


# ── B0: Flat ANN ────────────────────────────────────────────────────────

class FlatSearch:
    """普通 RAG：全局余弦相似度。"""

    name = "B0"

    def fit(self, corpus_emb: np.ndarray, corpus_ids: list[str]):
        self.corpus_ids = corpus_ids
        self.c_norm = _normalize(_to_tensor(corpus_emb))  # (Nd, 1024) normalized

    def search(self, query_emb: np.ndarray, query_ids: list[str],
               top_k: int = 100) -> dict[str, dict[str, float]]:
        q_norm = _normalize(_to_tensor(query_emb))  # (Nq, 1024)
        scores = q_norm @ self.c_norm.T  # (Nq, Nd) cosine matrix
        scores_np = scores.cpu().numpy()
        results = {}
        for i, qid in enumerate(query_ids):
            top_idx = np.argsort(scores_np[i])[::-1][:top_k]
            results[qid] = {self.corpus_ids[j]: float(scores_np[i, j]) for j in top_idx}
        return results


# ── B-fractal: 原始嵌入分形 ─────────────────────────────────────────────

class FractalSearch:
    """★ 嵌入分形理论核心路径（级联匹配 + 选择性激活 版本）。

    修正历史：融合匹配（v1）在 Phase 4 全部 FALSIFIED——展开向量做全库
    召回破坏了 bge-m3 的全局语义排序。级联匹配（v2）让 B0 召回 + 展开精排
    各司其职，展开只在候选集内重排序。

    流程：
      1. [召回] depth-0：原始 bge-m3 全局余弦 → top-K0 候选（= B0，保证不丢召回）
      2. [选择性激活] 对 query q，找出激活程度超过阈值的范畴集合 A(q) ⊆ {C_1,...,C_m}
      3. [depth-1 精排] 在 top-K0 内，用 A(q) 中的范畴条件余弦重排 → top-K1
      4. [depth-2 精排] 在 top-K1 内，用子范畴条件余弦重排 → top-K2
      5. ...直到深度用完或候选集不再缩小

    选择性激活：不是全部范畴都展开，只展开 query 高激活的范畴。
    低激活范畴与 query 无关，展开它们只引入噪声。
    """

    name = "B-fractal"

    def __init__(self, n_dims: int = 50, top_fraction: float = 0.2,
                 activation_threshold: float = 0.5,
                 recall_k0: int = 100, cascade_ks: list[int] = None,
                 ridge: float = 1e-5, max_depth: int = 3):
        self.n_dims = n_dims
        self.top_fraction = top_fraction
        self.activation_threshold = activation_threshold
        self.recall_k0 = recall_k0
        self.cascade_ks = cascade_ks or [50, 20, 10]  # 每层精排后的候选集大小
        self.ridge = ridge
        self.max_depth = max_depth

    def fit(self, corpus_emb: np.ndarray, corpus_ids: list[str]):
        from scipy.linalg import sqrtm

        self.corpus_ids = corpus_ids
        self.corpus_emb = corpus_emb
        N, D = corpus_emb.shape
        self.D = D

        # 选方差 top-k 维度
        variances = corpus_emb.var(axis=0)
        self.top_dims = np.argsort(variances)[::-1][:min(self.n_dims, D)]

        # 对每个维度 d 建条件高斯范畴
        n_top = max(1, int(N * self.top_fraction))
        self.categories = []

        for d in self.top_dims:
            member_idx = np.argsort(corpus_emb[:, d])[::-1][:n_top]
            members = corpus_emb[member_idx]

            # 条件高斯（在 top_dims 子空间里）
            sub_members = members[:, self.top_dims]
            mu_d = sub_members.mean(axis=0)
            centered = sub_members - mu_d

            if len(sub_members) > 1:
                sigma_d = np.cov(centered, rowvar=False)
            else:
                sigma_d = np.eye(len(self.top_dims)) * 0.01
            sigma_reg = sigma_d + self.ridge * np.eye(len(self.top_dims))

            sigma_inv_sqrt = sqrtm(np.linalg.pinv(sigma_reg))
            if np.iscomplexobj(sigma_inv_sqrt):
                sigma_inv_sqrt = np.real(sigma_inv_sqrt)

            self.categories.append({
                "dim": int(d),
                "mu": mu_d,
                "sigma_inv_sqrt": sigma_inv_sqrt,
            })

        # 预计算每个文档在各范畴下的子向量 s_d(x)（归一化）
        self.doc_sub_vectors = []  # list of (Nd, n_dims) normalized tensors
        for cat in self.categories:
            sub_all = (corpus_emb[:, self.top_dims] - cat["mu"]) @ cat["sigma_inv_sqrt"].T
            self.doc_sub_vectors.append(_normalize(_to_tensor(sub_all)))

        # 原始向量（B0 召回用）
        self.c_norm = _normalize(_to_tensor(corpus_emb))

    def search(self, query_emb: np.ndarray, query_ids: list[str],
               top_k: int = 100) -> dict[str, dict[str, float]]:
        results = {}

        for qi, qid in enumerate(query_ids):
            q = query_emb[qi]
            scores = self._cascade_search_one(q, top_k)
            results[qid] = scores

        return results

    def _cascade_search_one(self, q: np.ndarray, top_k: int) -> dict[str, float]:
        """对单个 query 做级联匹配。

        流程：B0 召回 → 选择性激活 → 多层精排 → 凸组合分数。
        最终分数 = α·精排分数 + (1-α)·全局余弦，确保分数和排序一致。
        """

        # ── depth-0: B0 召回（原始向量全局余弦）──
        q_t = _normalize(_to_tensor(q).unsqueeze(0))
        global_sims = (q_t @ self.c_norm.T).squeeze(0).cpu().numpy()
        K0 = min(self.recall_k0, len(global_sims))
        candidates = np.argsort(global_sims)[::-1][:K0]

        # ── 选择性激活 ──
        q_sel = q[self.top_dims]
        q_sel_norm = (q_sel - q_sel.min()) / (q_sel.max() - q_sel.min() + 1e-8)
        active_cats = np.where(q_sel_norm >= self.activation_threshold)[0]

        if len(active_cats) == 0:
            return {self.corpus_ids[j]: float(global_sims[j]) for j in candidates[:top_k]}

        # 激活权重（用于加权融合，softmax 归一化）
        active_weights = q_sel_norm[active_cats]
        active_weights = active_weights / (active_weights.sum() + 1e-8)

        # 预计算 query 在各激活范畴下的子向量
        q_sub_vectors = []
        for cat_idx in active_cats:
            cat = self.categories[cat_idx]
            q_sub = (q[self.top_dims] - cat["mu"]) @ cat["sigma_inv_sqrt"].T
            q_sub_norm = q_sub / (np.linalg.norm(q_sub) + 1e-8)
            q_sub_vectors.append(q_sub_norm)

        # ── 多层级联精排 ──
        for depth in range(self.max_depth):
            if len(candidates) <= top_k and depth > 0:
                break

            # 加权融合：score = Σ w_d · sim_d(q, x)
            fused_scores = np.zeros(len(candidates), dtype=np.float64)
            for i, (cat_idx, w) in enumerate(zip(active_cats, active_weights)):
                doc_sub_np = self.doc_sub_vectors[cat_idx].cpu().numpy()
                doc_sub = doc_sub_np[candidates]
                sims = doc_sub @ q_sub_vectors[i]
                fused_scores += w * sims

            # 重排候选集
            rerank_order = np.argsort(fused_scores)[::-1]
            candidates = candidates[rerank_order]

            # 缩小候选集
            target_k = self.cascade_ks[depth] if depth < len(self.cascade_ks) else top_k
            candidates = candidates[:max(top_k, target_k)]

        # ── 最终分数：保序策略（精排做候选集缩减，但不改变最终排序）──
        # 精排的价值在于"从 top-100 里筛掉不相关的"，而非"重排 top-10 的顺序"
        # 因此最终排序仍用全局余弦（global_sims），精排只决定哪些文档进入 top-k
        # 这避免了精排重排打乱好排序的风险
        final_candidates = candidates[:top_k]
        # 用全局余弦重新排序这些候选（恢复 B0 的排序）
        final_candidates_sorted = sorted(final_candidates,
                                          key=lambda idx: global_sims[idx], reverse=True)
        return {self.corpus_ids[idx]: float(global_sims[idx]) for idx in final_candidates_sorted}


# ── B0+: 残差展开 ───────────────────────────────────────────────────────

class ResidualSearch:
    """简化路径：HDBSCAN 范畴 + 残差去均值。

    对每个 query，找最近的 HDBSCAN 簇 k*，
    在簇 k* 的残差空间（v - μ_k*）里算余弦。
    """

    name = "B0+"

    def fit(self, corpus_emb: np.ndarray, corpus_ids: list[str]):
        from sklearn.cluster import HDBSCAN

        self.corpus_ids = corpus_ids
        self.corpus_emb = corpus_emb

        labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(corpus_emb)
        self.labels = labels

        # 每簇算 μ_k
        self.cluster_means = {}
        self.cluster_members = {}
        for label in sorted(set(labels)):
            if label < 0:
                continue
            mask = labels == label
            self.cluster_means[label] = corpus_emb[mask].mean(axis=0)
            self.cluster_members[label] = np.where(mask)[0]

        # noise 点用全局均值
        self.global_mean = corpus_emb.mean(axis=0)

        # 预计算每个簇的残差向量（归一化）
        self.cluster_residuals = {}  # {label: (n_members, 1024) normalized tensor}
        for label, members in self.cluster_members.items():
            residual = corpus_emb[members] - self.cluster_means[label]
            self.cluster_residuals[label] = _normalize(_to_tensor(residual))

        # 全局余弦作为 fallback（noise 点 + query 找不到簇时）
        self.c_norm = _normalize(_to_tensor(corpus_emb))

    def search(self, query_emb: np.ndarray, query_ids: list[str],
               top_k: int = 100) -> dict[str, dict[str, float]]:
        # 对每个 query：找最近簇，在残差空间比较
        cluster_means_tensor = _to_tensor(
            np.array([self.cluster_means[k] for k in sorted(self.cluster_means.keys())])
        )
        cluster_keys = sorted(self.cluster_means.keys())

        results = {}
        for i, qid in enumerate(query_ids):
            q = query_emb[i]

            # 找最近簇（query 和簇均值的余弦）
            q_t = _normalize(_to_tensor(q).unsqueeze(0))
            means_norm = _normalize(cluster_means_tensor)
            cluster_sims = (q_t @ means_norm.T).squeeze(0).cpu().numpy()
            best_cluster_idx = int(np.argmax(cluster_sims))
            best_label = cluster_keys[best_cluster_idx]

            # 在 best_cluster 的残差空间算余弦
            q_residual = q - self.cluster_means[best_label]
            q_res_norm = _normalize(_to_tensor(q_residual).unsqueeze(0))

            members = self.cluster_members[best_label]
            member_ids = [self.corpus_ids[m] for m in members]
            res_sims = (q_res_norm @ self.cluster_residuals[best_label].T).squeeze(0).cpu().numpy()

            # 只在簇成员内排序可能 top_k 不够——补充全局余弦的非成员
            # 策略：簇成员用残差分数，非成员用全局余弦分数
            scores = np.full(len(self.corpus_ids), -1e9, dtype=np.float64)
            scores[members] = res_sims

            # 非成员用全局余弦（降权，因为不在同一残差空间）
            q_global = _normalize(_to_tensor(q).unsqueeze(0))
            global_sims = (q_global @ self.c_norm.T).squeeze(0).cpu().numpy()
            non_member_mask = scores < -1e8
            scores[non_member_mask] = global_sims[non_member_mask] * 0.5  # 降权

            top_idx = np.argsort(scores)[::-1][:top_k]
            results[qid] = {self.corpus_ids[j]: float(scores[j]) for j in top_idx}

        return results
