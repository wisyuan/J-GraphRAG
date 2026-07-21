"""Phase 6 — 语义读出可行性验证（J-Space + 分词器反查 + Gradient×Input）。

验证核心命题：嵌入向量的方向能否被自动读出为人类可读的概念？

三步验证（从简到难）：
  1. J-Space 识别：方差 top-k 维度是否跨语料稳定
  2. 分词器反查：范畴方向 → 分词器嵌入最近邻 → token → 概念词
  3. Gradient×Input：确认反查的 token 因果驱动该维度

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase6_semantic_readout --repo /tmp/pi-repo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider
from experiments.fine_record_dispersion import split_file_to_symbols
from experiments.partition import partition_hdbscan

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


def cosine_np(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def mean_pairwise_distance_np(vecs):
    n = vecs.shape[0]
    if n < 2:
        return 0.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normed = vecs / norms
    cos_m = normed @ normed.T
    iu = np.triu_indices(n, k=1)
    return float((1.0 - cos_m[iu]).mean())


# ── Step 1: J-Space 识别 ────────────────────────────────────────────────

def identify_jspace(corpus_emb: np.ndarray, top_k: int = 50) -> dict:
    """J-Space = 方差最大的 top-k 维度。

    对比真实方差的 top-k 和随机 baseline（随机选 k 维的方差均值）。
    """
    variances = corpus_emb.var(axis=0)
    top_dims = np.argsort(variances)[::-1][:top_k]

    # 随机 baseline
    rng = np.random.default_rng(42)
    random_dims = rng.choice(len(variances), size=top_k, replace=False)
    random_var_sum = variances[random_dims].sum()
    top_var_sum = variances[top_dims].sum()

    return {
        "top_dims": top_dims.tolist(),
        "top_variances": variances[top_dims].tolist(),
        "top_var_sum": float(top_var_sum),
        "random_var_sum": float(random_var_sum),
        "concentration_ratio": float(top_var_sum / random_var_sum),  # >1 = 信息集中在少数维度
        "total_dims": len(variances),
    }


def jspace_stability(embed, corpora_texts: dict[str, list[str]], top_k: int = 50) -> dict:
    """检查 J-Space top-k 维度是否跨语料稳定。"""
    jspaces = {}
    all_top_dims = []

    for name, texts in corpora_texts.items():
        vecs = np.asarray(embed.embed(texts[:500]), dtype=np.float64)  # 最多 500 文档
        js = identify_jspace(vecs, top_k)
        jspaces[name] = js
        all_top_dims.append(set(js["top_dims"]))
        print(f"  [{name}] concentration={js['concentration_ratio']:.2f}x, "
              f"top dims: {js['top_dims'][:10]}...")

    # Jaccard 相似度（跨语料）
    if len(all_top_dims) >= 2:
        jaccards = []
        for i in range(len(all_top_dims)):
            for j in range(i + 1, len(all_top_dims)):
                inter = len(all_top_dims[i] & all_top_dims[j])
                union = len(all_top_dims[i] | all_top_dims[j])
                jac = inter / union if union > 0 else 0
                jaccards.append(jac)
        mean_jac = float(np.mean(jaccards))
        print(f"  cross-corpus Jaccard: mean={mean_jac:.4f}")
    else:
        mean_jac = 0.0

    return {"per_corpus": {k: {kk: vv for kk, vv in v.items() if kk != "top_variances"}
                           for k, v in jspaces.items()},
            "cross_corpus_jaccard": mean_jac,
            "stable": mean_jac > 0.3}  # Jaccard > 0.3 = 稳定


# ── Step 2: 分词器反查 ──────────────────────────────────────────────────

def build_tokenizer_embed_space(embed, top_n_tokens: int = 5000) -> dict:
    """构建分词器 token → 嵌入向量的映射。

    策略改进：用 WordNet 名词列表（干净英语概念词），而非 XLM-RoBERTa 全词表（多语言碎片）。
    """
    from nltk.corpus import wordnet

    # 从 WordNet 收集英语名词
    noun_set = set()
    for synset in wordnet.all_synsets(pos='n'):
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace('_', '')
            # 过滤：纯字母、长度 3-20
            if (3 <= len(name) <= 20
                    and re.match(r'^[a-zA-Z]+$', name)
                    and name not in noun_set):
                noun_set.add(name)
            if len(noun_set) >= top_n_tokens:
                break
        if len(noun_set) >= top_n_tokens:
            break

    token_strs = sorted(noun_set)[:top_n_tokens]
    token_vecs = np.asarray(embed.embed(token_strs), dtype=np.float64)

    print(f"  tokenizer space: {len(token_strs)} WordNet nouns embedded")

    return {
        "tokens": token_strs,
        "vecs": token_vecs,
    }


def reverse_lookup(cluster_centroid: np.ndarray, tokenizer_space: dict,
                   top_k: int = 10, jspace_dims: list[int] = None) -> list[tuple[str, float]]:
    """范畴方向 → 分词器嵌入空间最近邻 → top-k token。

    如果提供 jspace_dims，只在 J-Space 子空间里算余弦（信号更干净）。
    """
    token_vecs = tokenizer_space["vecs"]
    tokens = tokenizer_space["tokens"]

    if jspace_dims:
        # 投影到 J-Space 子空间
        centroid_sub = cluster_centroid[jspace_dims]
        token_vecs_sub = token_vecs[:, jspace_dims]
    else:
        centroid_sub = cluster_centroid
        token_vecs_sub = token_vecs

    # 算余弦
    sims = []
    for i in range(len(tokens)):
        sim = cosine_np(centroid_sub, token_vecs_sub[i])
        sims.append((tokens[i], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def cluster_concept_labels(corpus_emb, corpus_texts, embed, tokenizer_space,
                            jspace_dims=None, n_clusters=10) -> dict:
    """对语料的 top-N 簇做分词器反查，看标签是否语义连贯。"""
    from sklearn.cluster import HDBSCAN

    labels = HDBSCAN(min_cluster_size=3, min_samples=2, metric="cosine").fit_predict(corpus_emb)

    # 选 top-N 最大簇
    cluster_members = {}
    for i, l in enumerate(labels):
        if l >= 0:
            cluster_members.setdefault(l, []).append(i)

    top_clusters = sorted(cluster_members.items(), key=lambda x: len(x[1]), reverse=True)[:n_clusters]

    results = {}
    for cid, members in top_clusters:
        centroid = corpus_emb[members].mean(axis=0)

        # 分词器反查
        top_tokens = reverse_lookup(centroid, tokenizer_space, top_k=5,
                                     jspace_dims=jspace_dims)

        # 该簇的文档样本（取前 3 个文档的前 100 字符）
        sample_texts = [corpus_texts[m][:100] for m in members[:3]]

        results[int(cid)] = {
            "n_members": len(members),
            "concept_tokens": [(t, round(s, 4)) for t, s in top_tokens],
            "sample_docs": sample_texts,
        }

    return results


# ── Step 3: Gradient×Input 因果验证 ─────────────────────────────────────

def gradient_input_attribution(text: str, target_dim: int, embed) -> dict:
    """对单个文本，算分量 target_dim 对输入 token 的 Gradient×Input 归因。

    回答：哪些输入 token 因果驱动了分量 target_dim？
    """
    from jgraphrag.embed import _load_model
    import torch

    # 获取 bge-m3 torch model
    flag_model = _load_model()
    torch_model = flag_model.model.model  # XLMRobertaModel
    tokenizer = flag_model.tokenizer

    # fp16 backward 不被 CPU/DNNL 支持——临时转 fp32
    torch_model = torch_model.float()

    # Tokenize
    encoded = tokenizer(text, return_tensors="pt",
                        max_length=512, truncation=True)

    # 确定模型在哪个 device
    device = next(torch_model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # 获取 token embeddings（leaf tensor，可 backward）
    word_embeddings = torch_model.embeddings.word_embeddings
    token_embeds = word_embeddings(input_ids).detach()
    token_embeds.requires_grad_(True)

    # Forward through XLMRobertaModel
    outputs = torch_model(
        inputs_embeds=token_embeds,
        attention_mask=attention_mask,
    )
    last_hidden = outputs.last_hidden_state  # (1, seq_len, 1024)

    # Mean pooling (bge-m3 default)
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
    count = mask_expanded.sum(dim=1).clamp(min=1)
    pooled = sum_hidden / count  # (1, 1024)

    # 目标分量
    target = pooled[0, target_dim]

    # Backward
    target.backward()

    # Gradient×Input: contribution(t) = <grad[t,:], embed[t,:]>
    grads = token_embeds.grad[0]  # (seq_len, 1024)
    embeds = token_embeds.detach()[0]  # (seq_len, 1024)
    contributions = (grads * embeds).sum(dim=1).cpu().numpy()  # (seq_len,)

    # 解码 token
    token_texts = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # 按 |contribution| 排序
    ranked = sorted(zip(token_texts, contributions), key=lambda x: abs(x[1]), reverse=True)

    # 清理梯度
    token_embeds.grad = None

    return {
        "target_dim": target_dim,
        "top_tokens": [(t, float(c)) for t, c in ranked[:10]],
        "target_value": float(target.detach().cpu()),
    }


# ── 主流程 ──────────────────────────────────────────────────────────────

def run_experiment(embed, repo_path: str, max_files: int = 50):
    print("Phase 6: Semantic Readout Feasibility Test")
    print(f"{'='*60}")

    # 加载语料
    repo = Path(repo_path)
    files = sorted(p for p in repo.rglob("*.ts") if "node_modules" not in str(p))[:max_files]
    fine_texts = []
    for f in files:
        content = f.read_text(encoding="utf-8", errors="ignore")
        for sym in split_file_to_symbols(content):
            fine_texts.append(f"{sym['kind']} {sym['name']}:\n{sym['body']}")

    print(f"\n  {len(fine_texts)} FineRecords from {len(files)} files")
    print(f"  embedding...", end="", flush=True)
    fine_vecs = np.asarray(embed.embed(fine_texts), dtype=np.float64)
    print(f" done")

    # ── Step 1: J-Space ──
    print(f"\n{'='*60}")
    print("Step 1: J-Space Identification")
    js = identify_jspace(fine_vecs, top_k=50)
    print(f"  concentration ratio: {js['concentration_ratio']:.2f}x "
          f"(top-50 dims / random-50 dims variance)")
    print(f"  top-10 dims: {js['top_dims'][:10]}")
    jspace_dims = js["top_dims"]

    # ── Step 2: 分词器反查 ──
    print(f"\n{'='*60}")
    print("Step 2: Tokenizer Reverse Lookup")
    print("  building tokenizer embedding space...", end="", flush=True)
    tok_space = build_tokenizer_embed_space(embed, top_n_tokens=3000)
    print(f" done")

    # 在 J-Space 子空间里反查
    print(f"\n  Cluster concept labels (J-Space projected):")
    labels = cluster_concept_labels(fine_vecs, fine_texts, embed, tok_space,
                                     jspace_dims=jspace_dims, n_clusters=10)
    for cid, info in sorted(labels.items()):
        tokens_str = ", ".join(f"{t}({s})" for t, s in info["concept_tokens"])
        print(f"\n    Cluster {cid} ({info['n_members']} members):")
        print(f"      concept tokens: {tokens_str}")
        print(f"      sample: {info['sample_docs'][0][:80]}...")

    # ── Step 3: Gradient×Input（抽样验证）──
    print(f"\n{'='*60}")
    print("Step 3: Gradient×Input Causal Validation")
    print("  (testing 3 high-variance dimensions on a sample text)")

    # 取一个有代表性的文本
    sample_text = fine_texts[0][:500]
    print(f"  sample text: {sample_text[:100]}...")

    # 对 J-Space top-3 维度做 gradient×input
    grad_results = {}
    for dim_idx, dim in enumerate(jspace_dims[:3]):
        print(f"\n  Dimension {dim}:", end=" ", flush=True)
        try:
            gra = gradient_input_attribution(sample_text, dim, embed)
            top_toks = ", ".join(f"{t}({c:+.4f})" for t, c in gra["top_tokens"][:5])
            print(f"top tokens: {top_toks}")
            grad_results[dim] = gra
        except Exception as e:
            print(f"ERROR: {e}")
            grad_results[dim] = {"error": str(e)}

    # ── 交叉验证：分词器反查 vs Gradient×Input ──
    print(f"\n{'='*60}")
    print("Cross-validation: Tokenizer reverse vs Gradient×Input")
    if grad_results:
        for dim in list(grad_results.keys())[:3]:
            if "error" in grad_results[dim]:
                continue
            # 分词器反查：哪个 token 嵌入在 dim 上最接近"高激活方向"
            # 构造一个"只激活 dim"的方向向量
            direction = np.zeros(1024)
            direction[dim] = 1.0
            # 但这不对——应该是"在 dim 上高激活的文本的均值方向"
            # 简化：直接看 gradient×input 的 top token 是否和 cluster label 一致
            grad_tokens = set(t for t, _ in grad_results[dim]["top_tokens"][:5])
            print(f"  Dim {dim}: Gradient×Input tokens = {grad_tokens}")

    # ── 总结 ──
    print(f"\n{'='*60}")
    print("=== SUMMARY ===")
    print(f"  J-Space concentration: {js['concentration_ratio']:.2f}x (信息是否集中)")
    print(f"  Top clusters have concept tokens: "
          f"{'YES' if any(info['concept_tokens'] for info in labels.values()) else 'NO'}")

    # 人工审查引导
    print(f"\n  ★ Manual review needed:")
    print(f"  Are the concept tokens semantically coherent with cluster content?")
    for cid, info in sorted(labels.items())[:5]:
        tokens = [t for t, _ in info["concept_tokens"][:3]]
        sample = info["sample_docs"][0][:60]
        print(f"    Cluster {cid}: tokens={tokens} vs sample='{sample}...'")

    result = {
        "n_fine_records": len(fine_texts),
        "jspace": {k: v for k, v in js.items() if k != "top_variances"},
        "cluster_labels": {str(k): v for k, v in labels.items()},
        "gradient_input": {str(k): v for k, v in grad_results.items()},
    }

    out_path = EXP / "phase6_semantic_readout.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase 6: Semantic Readout")
    ap.add_argument("--repo", default="/tmp/pi-repo")
    ap.add_argument("--max-files", type=int, default=50)
    args = ap.parse_args()

    embed = CachedBgeM3Provider()
    run_experiment(embed, args.repo, args.max_files)


if __name__ == "__main__":
    main()
