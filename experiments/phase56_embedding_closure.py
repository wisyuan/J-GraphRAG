"""Phase 56: 去嵌入闭环——Qwen ws 向量替代 bge 种子的检索质量判决。

背景（tech-report §9.4 搁置项）：管线中 bge-m3 是唯一外来模型，去掉它则
"一个 7B 模型完成全部工作"的闭环成立。与 Phase 42 证伪的界限：42 是符号层
概念路由；本实验是稠密向量层种子替换。先验：Phase 40/48 E_ws 去偏后 ≈ bge。

路线 a（已证伪 2026-07-24）：chunk 向量 = 概念 ws_vec 聚合——死于覆盖空洞
（244 概念仅覆盖 25% chunk），与 ws 几何无关。

路线 b（一次 forward 双产出，引入用户推测）：
  dense  = transported 残差全位置 mean-pool
  sparse = lens logits 全位置 max-pool → relu → top-{10,25,50} 实词 token
           （SPLADE 式稀疏编码；用户推测：有效概念密度最高位置不一定是 -1
           → 全位置 max-pool；J-Space 论文称有效概念 ≤25 → 分档验证 25 是否瓶颈）

判决框架（与用户共同定稿 2026-07-24——不判"与 bge 是否相同"，判真实覆盖）：
  主判决：LLM-judged evidence coverage（llm_judge_evidence_recall，全 48 题配对），
    coverage(Qwen) >= coverage(bge) - 0.05 → SUPPORTED；< -0.10 → FALSIFIED；其间 INCONCLUSIVE
  次要① 优先度：recall(top-3)/recall(top-10)——覆盖相同但排得靠后则实际效用差
  次要② 分裂样本：|Δcoverage| 最大 8 题，LLM 评审员盲推（A/B 随机化）+
    用户审核推荐合理性（导出 data/m6/phase56_review.md）
  ACC 降级参考（本实验不跑生成侧）

用法：
  python -m experiments.phase56_embedding_closure --selftest   # mock，零 GPU 零 API
  python -m experiments.phase56_embedding_closure --variants   # 路线 a 诊断（已完成）
  python -m experiments.phase56_embedding_closure --encode --encode-limit 20  # 编码冒烟
  python -m experiments.phase56_embedding_closure --encode     # 全量编码（GPU ~2h）
  python -m experiments.phase56_embedding_closure --evalb [--smoke]  # 路线 b 判决（已完成）
  python -m experiments.phase56_embedding_closure --anchored   # 56b：实体锚定读出（双域）

Phase 56b（实体锚定读出，双域判别——用户假设 2026-07-26）：
  假设：workspace 内容以当前上下文为边界——query 侧只能读出问题承载的 +
  模型参数化记忆能联想的概念；语料私有内容词在 query 侧不可能显现。
  判别预测：实体锚定读出的 coverage 缺口（bge − qwen）在 medical（医学通识
  = 模型有参数化知识）应显著小于 novel（语料私有场景）。
  预设判决：gap_novel − gap_medical ≥ 0.15 → 假设 SUPPORTED（闭环按域划界）；
  两域同败（gap 均 >0.10）→ 闭环全面 FALSIFIED；
  某域 anchored ≥ bge − 0.05 → 该域闭环意外的成立。
  设计依据：读出定律（锚定实体位置）+ 原论文 paired-question 协议。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase4_dig_graphragbench import (
    BENCH_DIR, llm_judge_evidence_recall,
)
from experiments.phase41_retrieval_equivalence import (
    load_phase41_inputs, load_corpus_texts, load_questions, TOP_K,
)

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
OUT_PATH = EXP / "phase56_embedding_closure.json"
REVIEW_PATH = EXP / "phase56_review.md"
ROUTE_B_DENSE = EXP / "concept_cache" / "chunk_dense_ws.npz"
ROUTE_B_SPARSE = EXP / "concept_cache" / "chunk_sparse_ws.json"

# 稀疏向量 token 过滤（保留实词）
_STOP = set("""a an the and or but if then of in on at to for with by from as is are was were be been
it its this that these those i you he she we they his her their our my your me him us them not no
""".split())
# concern prompt 模板词黑名单（冒烟实测：模板位置读出攻占稀疏向量 top 位，
# 即 Phase 16/24/42/43 的"模板续词"模式）
_TEMPLATE_STOP = set("""what concepts concept does this text discuss list one word words one-word
name named names -word 名称 should regarding tasked these discuss. concepts.
""".split())
SPARSE_TOPK_STORE = 50   # 落盘保留 top-50；评测分 10/25/50 三档

# 主判决阈值（见模块 docstring）
GATE_SUPPORT = -0.05
GATE_FALSIFY = -0.10
N_REVIEW = 8
CONCERN_PROMPT = "What concepts does this text discuss? List 8 one-word concepts.\n\n{text}"


def build_chunk_ws_vecs(cache: dict, vecs: dict,
                        idf: bool = False) -> dict[str, np.ndarray]:
    """chunk → 其概念 ws_vec 均值（L2 归一）。无概念的 chunk 得零向量（排最后）。
    idf=True 时按 log(N/df) 加权，抑制高频通用概念主导。"""
    concepts = vecs["concepts"]
    ws = vecs["ws_vec"]
    cidx = {c: i for i, c in enumerate(concepts)}
    n_chunks = len(cache["chunks"])
    chunk_concepts: dict[str, list[int]] = {}
    for concept, cids in cache["concept_chunks"].items():
        i = cidx.get(concept)
        if i is None:
            continue
        w = float(np.log(n_chunks / max(1, len(cids)))) if idf else 1.0
        for cid in cids:
            chunk_concepts.setdefault(cid, []).append((i, w))
    out = {}
    for cid in cache["chunks"]:
        entries = chunk_concepts.get(cid)
        if not entries:
            out[cid] = np.zeros(ws.shape[1])
            continue
        idxs, weights = zip(*entries)
        v = (ws[list(idxs)] * np.asarray(weights)[:, None]).sum(axis=0)
        n = np.linalg.norm(v)
        out[cid] = v / n if n > 0 else v
    return out


def concept_agg_query_vec(question: str, vecs: dict) -> np.ndarray:
    """问题文本中提到的图概念的 ws_vec 聚合（零模型变体）。"""
    ws = vecs["ws_vec"]
    ql = question.lower()
    idxs = [i for i, c in enumerate(vecs["concepts"])
            if len(c) >= 3 and c.lower() in ql]
    if not idxs:
        return np.zeros(ws.shape[1])
    v = ws[idxs].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _encode_impl(questions: list[str], modes: list[str]) -> dict[str, np.ndarray]:
    import torch
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )
    import jlens
    from jlens.hooks import ActivationRecorder

    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    layer = lens.source_layers[-1]

    out: dict[str, list] = {m: [] for m in modes}
    for q in questions:
        text_by_mode = {
            "concern": CONCERN_PROMPT.format(text=q),
            "plain": q,
            "meanpool": q,
        }
        for mode in modes:
            msgs = [{"role": "user", "content": text_by_mode[mode]}]
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
            input_ids = lens_model.encode(prompt, max_length=1024)
            with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as rec:
                lens_model.forward(input_ids)
            resid = rec.activations[layer][0].float()  # bf16→fp32（transport 的 J_bar 是 float32）
            transported = lens.transport(resid, layer)
            if mode == "meanpool":
                v = transported.mean(dim=0).float().cpu().numpy()
            else:
                v = transported[-1].float().cpu().numpy()
            out[mode].append(v / (np.linalg.norm(v) + 1e-12))
    del lens_model, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return {m: np.asarray(vs) for m, vs in out.items()}


def encode_queries_ws(questions: list[str], mode: str = "concern") -> np.ndarray:
    """query 的 transported residual 向量（L2 归一）。加载并释放 Qwen。

    mode:
      concern  — concern prompt（"What concepts..."）position -1
      plain    — 纯问题文本 position -1
      meanpool — 纯问题文本，全部位置均值
    """
    return _encode_impl(questions, [mode])[mode]


# ── 路线 b：一次 forward 双产出（dense mean-pool + sparse top-k logits）──


def _valid_sparse_token(tok: str) -> bool:
    t = tok.strip().lower()
    if len(t) < 2 or t in _STOP or t in _TEMPLATE_STOP:
        return False
    if t.startswith(("-", "'", "’")):  # BPE 碎片（-word / 's）
        return False
    return sum(ch.isalpha() for ch in t) >= 2


def _encode_texts(lens, lens_model, tokenizer, texts: list[str],
                  tag: str) -> tuple[np.ndarray, list[dict]]:
    """已加载模型上的编码循环（dense mean-pool + sparse top-k logits）。"""
    import torch
    from jlens.hooks import ActivationRecorder
    layer = lens.source_layers[-1]

    dense_list: list[np.ndarray] = []
    sparse_list: list[dict] = []
    for i, text in enumerate(texts):
        body = CONCERN_PROMPT.format(text=text[:6000])
        msgs = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        input_ids = lens_model.encode(prompt, max_length=1024)
        # 位置窗口：只在实际文本 span 之后池化，跳过模板指令位置
        # （冒烟实测：模板位置读出的是"模板续词"而非内容概念）
        prefix = CONCERN_PROMPT.split("{text}")[0]
        prefix_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix}], tokenize=False,
            add_generation_prompt=False)
        text_start = len(tokenizer(prefix_prompt)["input_ids"])
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as rec:
            lens_model.forward(input_ids)
        resid = rec.activations[layer][0].float()  # [seq, d]，bf16→fp32
        transported = lens.transport(resid, layer)
        v = transported.mean(dim=0).cpu().numpy()
        dense_list.append(v / (np.linalg.norm(v) + 1e-12))
        # 分片 unembed（整段 [seq, vocab] fp32 会爆 8GB 显存）
        pooled = None
        start = min(text_start, transported.shape[0] - 1)
        for s in range(start, transported.shape[0], 64):
            lg = lens_model.unembed(transported[s:s + 64]).float()
            mx = lg.max(dim=0).values
            pooled = mx if pooled is None else torch.maximum(pooled, mx)
        pooled = torch.relu(pooled).cpu()
        top = pooled.topk(300)
        entries: dict[str, float] = {}
        for score, tid in zip(top.values.tolist(), top.indices.tolist()):
            tok = tokenizer.decode([tid]).strip().lower()
            if not _valid_sparse_token(tok) or tok in entries:
                continue
            entries[tok] = float(score)
            if len(entries) >= SPARSE_TOPK_STORE:
                break
        sparse_list.append(entries)
        if (i + 1) % 100 == 0:
            print(f"    encode[{tag}] {i + 1}/{len(texts)}", flush=True)
    return np.asarray(dense_list), sparse_list


def encode_pass_dense_sparse(texts: list[str], tag: str) -> tuple[np.ndarray, list[dict]]:
    """路线 b 编码（单次加载，单批文本）。注意：8GB 显存同一进程不能加载
    两次 4bit 模型——多批文本必须走 _encode_texts 共享会话。"""
    import torch
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )
    import jlens

    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    out = _encode_texts(lens, lens_model, tokenizer, texts, tag)
    del lens_model, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return out


def sparse_score(q_sparse: dict, c_sparse: dict, topk: int) -> float:
    """SPLADE 式点积：共享 token 的 q_score × c_score（chunk 侧截 top-k）。"""
    items = sorted(c_sparse.items(), key=lambda x: -x[1])[:topk]
    return sum(q_sparse.get(t, 0.0) * s for t, s in items)


def sparse_topk_ranking(q_sparse: dict, chunk_ids: list[str],
                        sparse_chunks: list[dict], k: int,
                        topk_tokens: int) -> list[str]:
    scores = [(sparse_score(q_sparse, sparse_chunks[i], topk_tokens), cid)
              for i, cid in enumerate(chunk_ids)]
    scores.sort(key=lambda x: -x[0])
    return [cid for _s, cid in scores[:k]]


# ── Phase 56b：实体锚定读出（query 侧）──

_CHAT_FILLER = {"user", "system", "assistant", "im_start", "im_end"}


def _anchor_positions(tokenizer, input_ids) -> list[int]:
    """内容词/实体位置：decoded token 去空格后 ≥4 字母且非停用/模板填充，
    或大写开头 ≥3 字母（实体）。读出定律：锚定实体位置。"""
    pos = []
    for i, tid in enumerate(input_ids[0].tolist()):
        tok = tokenizer.decode([tid])
        t = tok.strip()
        tl = t.lower()
        if tl in _CHAT_FILLER or tl in _STOP or tl in _TEMPLATE_STOP:
            continue
        n_alpha = sum(ch.isalpha() for ch in t)
        if n_alpha >= 4 or (t[:1].isupper() and n_alpha >= 3):
            pos.append(i)
    return pos


def encode_queries_anchored(lens, lens_model, tokenizer,
                            questions: list[str], tag: str) -> list[dict]:
    """锚定读出：纯问题文本，在内容词/实体位置读 lens logits → max-pool
    → relu → top-50 实词（= 实体名的参数化联想 + 问题承载概念）。"""
    import torch
    from jlens.hooks import ActivationRecorder
    layer = lens.source_layers[-1]
    out: list[dict] = []
    for i, q in enumerate(questions):
        msgs = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        input_ids = lens_model.encode(prompt, max_length=512)
        anchors = _anchor_positions(tokenizer, input_ids)
        if not anchors:
            anchors = [input_ids.shape[1] - 1]
        with torch.no_grad(), ActivationRecorder(lens_model.layers, at=[layer]) as rec:
            lens_model.forward(input_ids)
        resid = rec.activations[layer][0].float()
        transported = lens.transport(resid, layer)
        pooled = None
        for p in anchors:
            lg = lens_model.unembed(transported[p:p + 1]).float()[0]
            pooled = lg if pooled is None else torch.maximum(pooled, lg)
        pooled = torch.relu(pooled).cpu()
        top = pooled.topk(300)
        entries: dict[str, float] = {}
        for score, tid in zip(top.values.tolist(), top.indices.tolist()):
            tok = tokenizer.decode([tid]).strip().lower()
            if not _valid_sparse_token(tok) or tok in entries:
                continue
            entries[tok] = float(score)
            if len(entries) >= SPARSE_TOPK_STORE:
                break
        out.append(entries)
        if (i + 1) % 20 == 0:
            print(f"    anchored[{tag}] {i + 1}/{len(questions)}", flush=True)
    return out


def topk_by_cosine(qv: np.ndarray, mat_ids: list[str],
                   mat: np.ndarray, k: int) -> list[str]:
    sims = mat @ qv
    return [mat_ids[i] for i in np.argsort(-sims)[:k]]


def verdict_from_coverage(cov_qwen: float, cov_bge: float) -> str:
    diff = cov_qwen - cov_bge
    if diff >= GATE_SUPPORT:
        return f"SUPPORTED(diff={diff:+.3f} >= {GATE_SUPPORT})"
    if diff < GATE_FALSIFY:
        return f"FALSIFIED(diff={diff:+.3f} < {GATE_FALSIFY})"
    return f"INCONCLUSIVE(diff={diff:+.3f} 介于 {GATE_FALSIFY}~{GATE_SUPPORT})"


def run_coverage_eval(questions: list[dict], corpus: dict, ranked: dict,
                      llm, limit: int | None = None) -> list[dict]:
    """每题 × 每臂 × {top-10, top-3} 的 evidence coverage。"""
    qs = questions[:limit] if limit else questions

    def _one(args):
        q, arm, cut = args
        ctx = " ".join(corpus[cid] for cid in ranked[q["id"]][arm][:cut])
        ev = q.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        return q["id"], arm, cut, llm_judge_evidence_recall(
            q["question"], ctx, ev, llm)

    rows: dict[str, dict] = {q["id"]: {"qid": q["id"], "level": q.get("level"),
                                       "question": q["question"]}
                             for q in qs}
    tasks = [(q, arm, cut) for q in qs for arm in ["qwen", "bge"]
             for cut in (TOP_K, 3)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_one, t) for t in tasks]
        done = 0
        for f in as_completed(futures):
            qid, arm, cut, rec = f.result()
            rows[qid][f"{arm}_cov{cut}"] = rec
            done += 1
            if done % 20 == 0:
                print(f"    coverage {done}/{len(tasks)}", flush=True)
    return [rows[q["id"]] for q in qs]


def analyze(rows: list[dict]) -> dict:
    def m(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    cov_q, cov_b = m("qwen_cov10"), m("bge_cov10")
    return {
        "n": len(rows),
        "qwen_cov10": cov_q, "bge_cov10": cov_b,
        "qwen_cov3": m("qwen_cov3"), "bge_cov3": m("bge_cov3"),
        "qwen_priority": (m("qwen_cov3") or 0) / cov_q if cov_q else None,
        "bge_priority": (m("bge_cov3") or 0) / cov_b if cov_b else None,
        "verdict": verdict_from_coverage(cov_q or 0, cov_b or 0),
    }


def build_review_file(rows: list[dict], ranked: dict, corpus: dict,
                      questions: list[dict], llm) -> None:
    """分裂样本 → LLM 评审员盲推（A/B 随机化）→ markdown 供用户审核。"""
    qmap = {q["id"]: q for q in questions}
    sampled = sorted(
        (r for r in rows if r.get("qwen_cov10") is not None),
        key=lambda r: -abs(r["qwen_cov10"] - r["bge_cov10"]))[:N_REVIEW]
    rng = np.random.RandomState(56)
    lines = ["# Phase 56 分裂样本人工审核（LLM 初审 + 用户复核）\n",
             "每题两臂 top-10 以 A/B 盲标展示，LLM 评审员给出推荐与理由。",
             "A/B 真实身份见文末密钥。\n"]
    key = {}
    for r in sampled:
        q = qmap[r["qid"]]
        arms = ["qwen", "bge"]
        if rng.rand() < 0.5:
            arms = arms[::-1]
        key[r["qid"]] = {"A": arms[0], "B": arms[1]}
        ev = q.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        blocks = {}
        for label, arm in zip(["A", "B"], arms):
            chunks = "\n".join(
                f"  [{i+1}] {corpus[cid][:200]}..."
                for i, cid in enumerate(ranked[r["qid"]][arm]))
            blocks[label] = chunks
        review_prompt = (
            f"Question: {q['question']}\n\n"
            f"Evidence needed:\n" + "\n".join(f"- {e}" for e in ev) +
            f"\n\nRetrieval A top-10 chunks:\n{blocks['A']}\n\n"
            f"Retrieval B top-10 chunks:\n{blocks['B']}\n\n"
            "Which retrieval set is more useful for answering the question "
            "completely and faithfully? Reply with ONLY JSON: "
            '{"better": "A"|"B"|"tie", "rationale": "<one sentence>"}')
        msg = llm.complete(review_prompt, max_tokens=800, thinking=False)
        text = msg.content if hasattr(msg, "content") else str(msg)
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        rec = json.loads(m.group(0)) if m else {"better": "?", "rationale": text[:120]}
        lines += [
            f"\n## {r['qid']}（coverage: qwen={r['qwen_cov10']:.2f} bge={r['bge_cov10']:.2f}）",
            f"**Q**: {q['question']}",
            f"**Evidence**: {'; '.join(str(e) for e in ev)}",
            f"\n**Retrieval A**:\n{blocks['A']}",
            f"\n**Retrieval B**:\n{blocks['B']}",
            f"\n**LLM 推荐**: {rec.get('better')} — {rec.get('rationale')}",
            "\n**你的判断**（合理？）: ",
        ]
    lines += ["\n---\n## A/B 密钥\n"] + [
        f"- {qid}: A={v['A']}, B={v['B']}" for qid, v in key.items()]
    REVIEW_PATH.write_text("\n".join(lines))
    print(f"  review file -> {REVIEW_PATH}")


def _selftest():
    """聚合 + 检索 + 判决逻辑，合成数据，零 GPU 零 API。"""
    cache = {"chunks": {"c1": "", "c2": "", "c3": ""},
             "concept_chunks": {"x": ["c1", "c2"], "y": ["c2"]}}
    vecs = {"concepts": ["x", "y"], "count": np.array([2, 1]),
            "wu_vec": np.zeros((2, 3)),
            "ws_vec": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
    cv = build_chunk_ws_vecs(cache, vecs)
    assert np.allclose(cv["c1"], [1, 0, 0])
    assert np.allclose(cv["c3"], [0, 0, 0])
    exp = np.array([1, 1, 0]) / np.sqrt(2)
    assert np.allclose(cv["c2"], exp)
    ids = ["c1", "c2", "c3"]
    mat = np.stack([cv[c] for c in ids])
    assert topk_by_cosine(np.array([0.0, 1.0, 0.0]), ids, mat, 1) == ["c2"]
    assert "SUPPORTED" in verdict_from_coverage(0.60, 0.64)
    assert "INCONCLUSIVE" in verdict_from_coverage(0.55, 0.62)
    assert "FALSIFIED" in verdict_from_coverage(0.50, 0.62)
    print("selftest OK")


def run_variants(questions: list[dict], corpus: dict, chunk_ids: list[str],
                 cache: dict, vecs: dict) -> None:
    """query 4 变体 × chunk 2 变体的诊断矩阵（4 题，重叠率 + coverage）。"""
    from experiments.embed_cache import CachedBgeM3Provider
    from jgraphrag.llm import DeepSeekProvider

    qs = questions[:4]
    qvecs = _encode_impl([q["question"] for q in qs],
                         ["concern", "plain", "meanpool"])
    qvecs["concept-agg"] = np.stack(
        [concept_agg_query_vec(q["question"], vecs) for q in qs])
    chunk_vars = {
        "agg": build_chunk_ws_vecs(cache, vecs, idf=False),
        "agg+idf": build_chunk_ws_vecs(cache, vecs, idf=True),
    }

    embed_fn = CachedBgeM3Provider().embed
    bge_mat = np.asarray(embed_fn([corpus[cid] for cid in chunk_ids]),
                         dtype=np.float64)
    norms = np.linalg.norm(bge_mat, axis=1, keepdims=True)
    bge_mat = bge_mat / np.where(norms > 0, norms, 1.0)
    bge_q = np.asarray(embed_fn([q["question"] for q in qs]), dtype=np.float64)
    bge_ranked = [topk_by_cosine(bge_q[i] / (np.linalg.norm(bge_q[i]) + 1e-12),
                                 chunk_ids, bge_mat, TOP_K)
                  for i in range(len(qs))]

    llm = DeepSeekProvider()
    print(f"  {'query-mode':<12} {'chunk':<9} {'overlap':<8} {'cov10':<6}")
    for cname, cws in chunk_vars.items():
        ws_mat = np.stack([cws[cid] for cid in chunk_ids])
        for qname, qv_all in qvecs.items():
            ovs, covs = [], []
            for i, q in enumerate(qs):
                ranked = topk_by_cosine(qv_all[i], chunk_ids, ws_mat, TOP_K)
                ovs.append(len(set(ranked) & set(bge_ranked[i])) / TOP_K)
                ev = q.get("evidence") or []
                if isinstance(ev, str):
                    ev = [ev]
                ctx = " ".join(corpus[cid] for cid in ranked)
                covs.append(llm_judge_evidence_recall(q["question"], ctx, ev, llm))
            print(f"  {qname:<12} {cname:<9} {np.mean(ovs):<8.3f} "
                  f"{np.mean(covs):<6.3f}", flush=True)


def run_encode(questions: list[dict], corpus: dict, chunk_ids: list[str],
               limit: int | None = None) -> None:
    """路线 b 编码 pass（GPU，单会话）：语料 dense+sparse 落盘 + query 同构编码。
    同一进程只加载一次 4bit 模型（8GB 铁律）。"""
    import torch
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )
    import jlens

    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    ids = chunk_ids[:limit] if limit else chunk_ids
    dense, sparse = _encode_texts(lens, lens_model, tokenizer,
                                  [corpus[cid] for cid in ids], "corpus")
    ROUTE_B_DENSE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ROUTE_B_DENSE, chunk_ids=ids, dense=dense)
    json.dump(dict(zip(ids, sparse)), open(ROUTE_B_SPARSE, "w"))
    print(f"  corpus encoded: dense {dense.shape}, sparse -> {ROUTE_B_SPARSE}",
          flush=True)

    q_dense, q_sparse = _encode_texts(
        lens, lens_model, tokenizer, [q["question"] for q in questions], "query")
    np.savez_compressed(EXP / "concept_cache" / "query_dense_ws.npz",
                        qids=[q["id"] for q in questions], dense=q_dense)
    json.dump({q["id"]: s for q, s in zip(questions, q_sparse)},
              open(EXP / "concept_cache" / "query_sparse_ws.json", "w"))
    print("  query encoded", flush=True)

    del lens_model, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def run_eval_routb(questions: list[dict], corpus: dict, chunk_ids: list[str],
                   smoke: bool = False) -> None:
    """路线 b 评测：dense / sparse-top{10,25,50} vs bge 的 coverage 判决。"""
    from experiments.embed_cache import CachedBgeM3Provider
    from jgraphrag.llm import DeepSeekProvider

    dz = np.load(ROUTE_B_DENSE)
    enc_ids = [str(c) for c in dz["chunk_ids"]]
    dense_mat = dz["dense"].astype(np.float64)
    sparse_chunks = json.load(open(ROUTE_B_SPARSE))
    sparse_list = [sparse_chunks[cid] for cid in enc_ids]
    qz = np.load(EXP / "concept_cache" / "query_dense_ws.npz")
    q_dense = qz["dense"].astype(np.float64)
    q_sparse = json.load(open(EXP / "concept_cache" / "query_sparse_ws.json"))

    qs = questions[:4] if smoke else questions
    embed_fn = CachedBgeM3Provider().embed
    bge_mat = np.asarray(embed_fn([corpus[cid] for cid in enc_ids]),
                         dtype=np.float64)
    norms = np.linalg.norm(bge_mat, axis=1, keepdims=True)
    bge_mat = bge_mat / np.where(norms > 0, norms, 1.0)
    bge_q = np.asarray(embed_fn([q["question"] for q in qs]), dtype=np.float64)

    qid_to_qi = {q["id"]: i for i, q in enumerate(questions)}
    encoders = ["bge", "dense", "sparse10", "sparse25", "sparse50"]
    ranked: dict[str, dict] = {}
    for q in qs:
        qi = qid_to_qi[q["id"]]
        r = {"bge": topk_by_cosine(
            bge_q[qi] / (np.linalg.norm(bge_q[qi]) + 1e-12),
            enc_ids, bge_mat, TOP_K)}
        r["dense"] = topk_by_cosine(q_dense[qi], enc_ids, dense_mat, TOP_K)
        for topk in (10, 25, 50):
            r[f"sparse{topk}"] = sparse_topk_ranking(
                q_sparse[q["id"]], enc_ids, sparse_list, TOP_K, topk)
        ranked[q["id"]] = r

    llm = DeepSeekProvider()
    rows = []
    for q in qs:
        row = {"qid": q["id"], "level": q.get("level"), "question": q["question"]}
        ev = q.get("evidence") or []
        if isinstance(ev, str):
            ev = [ev]
        for enc in encoders:
            ctx = " ".join(corpus[cid] for cid in ranked[q["id"]][enc])
            row[f"{enc}_cov10"] = llm_judge_evidence_recall(
                q["question"], ctx, ev, llm)
            ctx3 = " ".join(corpus[cid] for cid in ranked[q["id"]][enc][:3])
            row[f"{enc}_cov3"] = llm_judge_evidence_recall(
                q["question"], ctx3, ev, llm)
        rows.append(row)
        print(f"    judged {len(rows)}/{len(qs)}", flush=True)

    bge_cov = float(np.mean([r["bge_cov10"] for r in rows]))
    table = {}
    for enc in encoders:
        cov = float(np.mean([r[f"{enc}_cov10"] for r in rows]))
        cov3 = float(np.mean([r[f"{enc}_cov3"] for r in rows]))
        table[enc] = {"cov10": cov, "cov3": cov3,
                      "priority": cov3 / cov if cov else None,
                      "verdict_vs_bge": verdict_from_coverage(cov, bge_cov)}
    report = {"method": "Phase 56 路线 b：Qwen 自编码（dense/sparse）vs bge",
              "n": len(rows), "encoders": table,
              "ranked": ranked, "per_query": rows}
    out = EXP / ("phase56_routeb_smoke.json" if smoke else "phase56_routeb.json")
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(table, ensure_ascii=False, indent=1))
    print(f"saved -> {out}")


def run_56b() -> None:
    """56b 双域判别：medical chunk 编码 + 双域锚定 query + coverage 判决。"""
    import torch
    from experiments.phase10_jlens_stage1 import (
        detect_model, load_model, load_lens, _model_dir_complete,
    )
    import jlens
    from experiments.embed_cache import CachedBgeM3Provider
    from jgraphrag.llm import DeepSeekProvider

    # ── 数据 ──
    domains = {}
    for dom, n_q in [("novel", 48), ("medical", 56)]:
        cache, _v, _r = load_phase41_inputs(dom)
        corpus = load_corpus_texts(dom, cache)
        chunk_ids = sorted(cache["chunks"].keys())
        questions = load_questions(dom, n_q)
        gold = {q["id"]: q for q in json.load(open(BENCH_DIR / f"{dom}_questions.json"))}
        for q in questions:
            g = gold.get(q["id"], {})
            q["evidence"] = g.get("evidence") or []
        domains[dom] = {"corpus": corpus, "chunk_ids": chunk_ids,
                        "questions": questions}

    # ── 单会话 GPU：medical chunk 编码 + 双域锚定 query ──
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    m = domains["medical"]
    _dense_m, sparse_m = _encode_texts(
        lens, lens_model, tokenizer,
        [m["corpus"][cid] for cid in m["chunk_ids"]], "medical-corpus")
    json.dump(dict(zip(m["chunk_ids"], sparse_m)),
              open(EXP / "concept_cache" / "chunk_sparse_ws_medical.json", "w"))

    anchored = {}
    for dom in ["novel", "medical"]:
        qs = domains[dom]["questions"]
        anchored[dom] = dict(zip(
            [q["id"] for q in qs],
            encode_queries_anchored(lens, lens_model, tokenizer,
                                    [q["question"] for q in qs], dom)))
    json.dump(anchored, open(EXP / "concept_cache" / "query_anchored_ws.json", "w"))
    del lens_model, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ── 检索 + coverage 判决 ──
    novel_sparse = json.load(open(ROUTE_B_SPARSE))
    llm = DeepSeekProvider()
    embed_fn = CachedBgeM3Provider().embed
    report: dict = {"method": "Phase 56b: 实体锚定读出双域判别（用户假设）",
                    "gates": {"hypothesis_gap_diff": 0.15, "falsify_gap": 0.10,
                              "closure": -0.05},
                    "domains": {}}
    # novel bge 基线复用路线 b 评测结果
    prev = json.load(open(EXP / "phase56_routeb.json"))
    bge_prev = {r["qid"]: r for r in prev["per_query"]}

    for dom in ["novel", "medical"]:
        d = domains[dom]
        chunk_ids = d["chunk_ids"]
        sparse_chunks = ([novel_sparse[cid] for cid in chunk_ids]
                         if dom == "novel"
                         else [sparse_m[chunk_ids.index(cid)]
                               for cid in chunk_ids])
        bge_mat = np.asarray(embed_fn([d["corpus"][cid] for cid in chunk_ids]),
                             dtype=np.float64)
        norms = np.linalg.norm(bge_mat, axis=1, keepdims=True)
        bge_mat = bge_mat / np.where(norms > 0, norms, 1.0)
        bge_q = np.asarray(embed_fn([q["question"] for q in d["questions"]]),
                           dtype=np.float64)

        rows = []
        for qi, q in enumerate(d["questions"]):
            ev = q.get("evidence") or []
            if isinstance(ev, str):
                ev = [ev]
            row = {"qid": q["id"], "level": q.get("level")}
            # anchored arm
            ranked_a = sparse_topk_ranking(
                anchored[dom][q["id"]], chunk_ids, sparse_chunks, TOP_K, 25)
            ctx = " ".join(d["corpus"][cid] for cid in ranked_a)
            row["anch_cov10"] = llm_judge_evidence_recall(
                q["question"], ctx, ev, llm)
            # bge arm（novel 复用旧判）
            if dom == "novel" and q["id"] in bge_prev:
                row["bge_cov10"] = bge_prev[q["id"]]["bge_cov10"]
            else:
                ranked_b = topk_by_cosine(
                    bge_q[qi] / (np.linalg.norm(bge_q[qi]) + 1e-12),
                    chunk_ids, bge_mat, TOP_K)
                ctxb = " ".join(d["corpus"][cid] for cid in ranked_b)
                row["bge_cov10"] = llm_judge_evidence_recall(
                    q["question"], ctxb, ev, llm)
            rows.append(row)
            print(f"    56b[{dom}] {len(rows)}/{len(d['questions'])}", flush=True)

        bge_cov = float(np.mean([r["bge_cov10"] for r in rows]))
        anc_cov = float(np.mean([r["anch_cov10"] for r in rows]))
        report["domains"][dom] = {
            "n": len(rows), "bge_cov10": bge_cov, "anchored_cov10": anc_cov,
            "gap": bge_cov - anc_cov, "per_query": rows}

    gn = report["domains"]["novel"]["gap"]
    gm = report["domains"]["medical"]["gap"]
    if gn - gm >= 0.15:
        v = (f"用户假设 SUPPORTED（gap_novel−gap_medical = {gn - gm:+.3f} ≥ 0.15）"
             "——闭环按域划界：知识密集域可用，语料私有域不可用")
    elif gn > 0.10 and gm > 0.10:
        v = f"闭环全面 FALSIFIED（两域 gap 均 >0.10：novel {gn:.3f}, medical {gm:.3f}）"
    else:
        v = f"INCONCLUSIVE/域闭环（novel gap {gn:.3f}, medical gap {gm:.3f}）"
    report["verdict"] = v
    out = EXP / "phase56b_anchored.json"
    json.dump(report, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in d.items() if kk != "per_query"}
                      for k, d in report["domains"].items()},
                     ensure_ascii=False, indent=1))
    print("verdict:", v)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="4 题全链路")
    ap.add_argument("--variants", action="store_true",
                    help="失败模式诊断：query 4 变体 × chunk 2 变体（4 题）")
    ap.add_argument("--encode", action="store_true",
                    help="路线 b 编码 pass（GPU，语料+query 落盘缓存）")
    ap.add_argument("--encode-limit", type=int, default=0,
                    help="--encode 冒烟：只编前 N 个 chunk")
    ap.add_argument("--evalb", action="store_true",
                    help="路线 b 评测（需先 --encode 全量）")
    ap.add_argument("--anchored", action="store_true",
                    help="Phase 56b：实体锚定读出双域判别")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    # ── 数据与向量（CPU）──
    cache, vecs, _rel = load_phase41_inputs("novel")
    corpus = load_corpus_texts("novel", cache)
    chunk_ids = sorted(cache["chunks"].keys())

    questions = load_questions("novel", 48)
    gold = {q["id"]: q for q in json.load(open(BENCH_DIR / "novel_questions.json"))}
    for q in questions:
        g = gold.get(q["id"], {})
        q["evidence"] = g.get("evidence") or []
        q["answer"] = g.get("answer", "")
    print(f"{len(questions)} questions, {len(chunk_ids)} chunks", flush=True)

    if args.variants:
        run_variants(questions, corpus, chunk_ids, cache, vecs)
        return

    if args.encode:
        run_encode(questions, corpus, chunk_ids,
                   limit=args.encode_limit or None)
        return

    if args.evalb:
        run_eval_routb(questions, corpus, chunk_ids, smoke=args.smoke)
        return

    if args.anchored:
        run_56b()
        return

    chunk_ws = build_chunk_ws_vecs(cache, vecs)
    ws_mat = np.stack([chunk_ws[cid] for cid in chunk_ids])

    # ── query 编码（GPU，一次性）──
    q_vecs = encode_queries_ws([q["question"] for q in questions])

    # ── bge 基线 ──
    from experiments.embed_cache import CachedBgeM3Provider
    embed_fn = CachedBgeM3Provider().embed
    bge_mat = np.asarray(embed_fn([corpus[cid] for cid in chunk_ids]),
                         dtype=np.float64)
    norms = np.linalg.norm(bge_mat, axis=1, keepdims=True)
    bge_mat = bge_mat / np.where(norms > 0, norms, 1.0)
    bge_q = np.asarray(embed_fn([q["question"] for q in questions]),
                       dtype=np.float64)

    # ── 检索（两臂 top-10）──
    ranked: dict[str, dict] = {}
    for qi, q in enumerate(questions):
        ranked[q["id"]] = {
            "qwen": topk_by_cosine(q_vecs[qi], chunk_ids, ws_mat, TOP_K),
            "bge": topk_by_cosine(
                bge_q[qi] / (np.linalg.norm(bge_q[qi]) + 1e-12),
                chunk_ids, bge_mat, TOP_K),
        }
    overlap = [len(set(r["qwen"]) & set(r["bge"])) / TOP_K
               for r in ranked.values()]
    print(f"  seed top-10 overlap (参考): {np.mean(overlap):.3f}", flush=True)

    # ── coverage 判决（API）──
    from jgraphrag.llm import DeepSeekProvider
    llm = DeepSeekProvider()
    rows = run_coverage_eval(questions, corpus, ranked, llm,
                             limit=4 if args.smoke else None)
    report = {
        "method": "Phase 56: Qwen ws 向量 vs bge 种子的 evidence coverage 判决",
        "gates": {"support": GATE_SUPPORT, "falsify": GATE_FALSIFY},
        "seed_overlap_top10": float(np.mean(overlap)),
        "analysis": analyze(rows),
        "ranked": ranked,
        "per_query": rows,
    }
    if not args.smoke:
        build_review_file(rows, ranked, corpus, questions, llm)
        json.dump(report, open(OUT_PATH, "w"), ensure_ascii=False, indent=1)
        print(f"saved -> {OUT_PATH}")
    print(json.dumps(report["analysis"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
