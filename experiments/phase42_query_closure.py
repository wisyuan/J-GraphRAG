"""Phase 42: J-GraphRAG 查询侧闭环——查询脱离 bge-m3 进入概念空间。

核心问题：查询能否像文档一样经两步法（Pass 1 概念提取 + 条件向量）进入
概念空间，用余弦匹配替代 bge 检索？文档侧早已是闭环（phase39/39b：chunk →
概念 → ws 向量 → 词表），本实验把同一管线搬到查询侧。

每查询 4 个检索臂 + 2 个对照：

  Pass 1（概念提取，复用 phase31 管线但拆开重做）：
    concern prompt（phase20 build_concern_prompt_full）→ 单次 forward 手工
    记录全层残差（jlens ActivationRecorder，镜像 lens.apply 的
    record → transport → unembed），同一 forward 同时产出：
      a) 全层 depth gradient（位置 -1）→ 三重过滤 + BM25 补全（与
         extract_concepts_full_pipeline 逐行一致）→ concept_ok（phase39b）
      b) 臂 A 向量：查询文本 span 内概念词 token 位置的 last-source-layer
         残差，transport 后 L2 归一（零额外 forward）
    唯一例外：phase31 的 single-layer fallback（extract_chunk_concepts）
    触发时会多 1 次 forward——计数进 meta.n_fallback_forwards。

  臂 B 向量（+1 forward）：phase39.pass2_role_expansion 的反转 prefill，
    读 prefill 概念位置的 ws 向量（与文档侧 ws_vec 同一构造）。

  匹配：查询概念向量（A 或 B）对词表 ws_vec（concept_vecs_{domain}.npz）
    做余弦，每个查询概念取 top-3 词表概念、max(cos,0) 为权，求和聚合成
    概念空间 q 向量（L2 归一），再映射到 M 的概念轴（字符串对齐）。

  检索臂：
    q_ws_A / q_ws_B : chunk 得分 = q × M（M 与 phase41 同一构建，
                      IDF×BM25 概念×chunk，来自 RetrievalBase）
    hybrid          : z(bge 余弦) + 0.3 × z(q_ws_B × M)
  对照：
    B0              : bge 直接检索（CachedBgeM3Provider）
    graph           : phase41 flat 图传播（seed=bge top-5，merge_ranking）

指标：
  检索层——每臂 vs B0 / vs graph 的 top-10 重叠率 + Kendall τ
    （phase41 的 topk_overlap / kendall_tau_scores，口径一致）。
  答案层——q_ws_A / q_ws_B / hybrid / B0 各取 top-10 chunk 作 context，
    phase26 generate_answer + judge_answer_correctness（DeepSeek），
    ACC 总分 + 分级（L1-L4）。--smoke 跳过。
  诊断——每查询导出提取概念 + 匹配到的 top-3 词表概念（含余弦）+
    臂 A/B 向量在同一概念上的余弦一致性。

产出：experiments/m6/phase42_query_closure_{domain}.json（--smoke 不写文件）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    python -m experiments.phase42_query_closure --smoke
    python -m experiments.phase42_query_closure --domain medical
    python -m experiments.phase42_query_closure --domain all --max-queries 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase10_jlens_stage1 import (
    detect_model, load_model, load_lens, _model_dir_complete,
)
from experiments.phase17_multihop_depth_gradient import _decode_layer_topk
from experiments.phase18_centroid_hierarchy import build_corpus_word_set
from experiments.phase20_concern_full_com import (
    build_concern_prompt_full, compute_concept_profiles_filtered,
)
from experiments.concept_quality import (
    complete_prefix, build_corpus_term_freq, _get_wordnet_nouns,
)
from experiments.phase39_two_pass_cache import pass2_role_expansion
from experiments.phase39b_filter_cache import concept_ok
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness
from experiments.phase41_retrieval_equivalence import (
    EXP, TOP_K, SEED_K,
    RetrievalBase, load_phase41_inputs, load_corpus_texts,
    propagate_scored, merge_ranking, scores_to_dict,
    topk_overlap, kendall_tau_scores,
)

TOP_MATCH = 3          # vocab concepts kept per query concept
HYBRID_ALPHA = 0.3     # weight of concept-space score in the hybrid arm
MAX_SEQ_LEN = 512      # same as extract_depth_gradient
# Balanced per-level subsample size per domain (medical = phase41's
# FULL_BENCH_QUERIES; novel = 12/level x 4 levels, phase41 Part A convention).
DEFAULT_QUERIES = {"medical": 56, "novel": 48}

ARMS = ["q_ws_A", "q_ws_B", "hybrid"]
EVAL_ARMS = ["q_ws_A", "q_ws_B", "hybrid", "B0"]  # answer-layer arms
LEVELS = ["L1", "L2", "L3", "L4"]


# ── Pass 1, rebuilt: one forward → gradient (concepts) + arm-A residuals ──


def pass1_query(lens, lens_model, tokenizer, query_text: str,
                all_layers: list[int], corpus_words: set[str],
                corpus_freq: dict[str, int], wn_nouns: set[str]) -> dict:
    """Phase 25/31 extraction on a query, with arm-A vectors from the same forward.

    Mirrors extract_concepts_full_pipeline step-for-step, except the depth
    gradient comes from a manually-driven forward (ActivationRecorder at all
    source layers) so the last-source-layer residuals at the concept token
    positions inside the query span are available too (arm A, zero extra
    forward). The single-layer fallback (rare) costs one extra forward, same
    as phase31.

    Returns {concepts, raw_map, ws_vecs, positions, n_fallback_forwards,
             prompt, offset_mapping}:
      concepts:  final concept list (completed + concept_ok filtered)
      raw_map:   {final_concept: pre-completion raw word} (for diagnostics)
      ws_vecs:   {final_concept: L2-normalized transported residual} (arm A)
      positions: {final_concept: token position in prompt} (diagnostic)
    """
    from jlens.hooks import ActivationRecorder

    prompt = build_concern_prompt_full([query_text], tokenizer)
    # Tokenize once with offsets; the same ids go into the forward so token
    # positions and char spans are guaranteed aligned.
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=MAX_SEQ_LEN, return_offsets_mapping=True)
    input_ids = enc.input_ids.to(lens_model.input_device)
    offsets = enc.offset_mapping[0].tolist()
    seq_len = input_ids.shape[1]

    last_source = all_layers[-1]
    with torch.no_grad(), ActivationRecorder(lens_model.layers,
                                             at=list(all_layers)) as rec:
        lens_model.forward(input_ids)
    acts = {i: rec.activations[i].detach() for i in all_layers}

    # Depth gradient at position -1 (identical math to lens.apply)
    gradient = {}
    for layer in all_layers:
        resid = acts[layer][0, seq_len - 1].float()          # [d_model]
        logits = lens_model.unembed(lens.transport(resid, layer)).float().cpu()
        gradient[layer] = _decode_layer_topk(logits, tokenizer, n_words=8)

    # Phase 20 triple filter — strict first, then relax (phase31 lines verbatim)
    profiles = compute_concept_profiles_filtered(
        gradient, corpus_words, min_layers=2,
        require_corpus=True, require_noun=True)
    if len(profiles) < 2:
        profiles = compute_concept_profiles_filtered(
            gradient, corpus_words, min_layers=2,
            require_corpus=False, require_noun=True)
        profiles = [p for p in profiles if p.in_corpus]
    concepts_raw = [p.word for p in profiles]

    n_fallback = 0
    if len(concepts_raw) < 2:
        # phase31 fallback: single-layer L26 readout (1 extra forward)
        from experiments.phase10_jlens_stage7c import extract_chunk_concepts
        n_fallback = 1
        concepts_raw = extract_chunk_concepts(
            lens, lens_model, tokenizer, query_text, n_words=5)
        concepts_raw = [c for c in concepts_raw if c.lower() in corpus_words]

    # BM25 BPE completion + dedupe (phase31), keeping the raw word per concept
    raw_map: dict[str, str] = {}
    seen: set[str] = set()
    concepts_final: list[str] = []
    for c in concepts_raw[:8]:
        completed = c
        if len(c) <= 5:
            result = complete_prefix(c, corpus_freq, wn_nouns)
            if result:
                completed = result
        cl = completed.lower()
        if cl not in seen:
            seen.add(cl)
            concepts_final.append(completed)
            raw_map[completed] = c

    # phase39b concept filter (drops prefill template words like "listed")
    concepts_kept = [c for c in concepts_final[:8] if concept_ok(c)[0]]

    # Arm A: locate each kept concept inside the query span, read the residual
    ws_vecs, positions = _arm_a_vectors(
        lens, acts[last_source], last_source, offsets, prompt, query_text,
        concepts_kept, raw_map, seq_len)

    return {
        "concepts": concepts_kept,
        "raw_map": raw_map,
        "ws_vecs": ws_vecs,
        "positions": positions,
        "n_fallback_forwards": n_fallback,
    }


def _arm_a_vectors(lens, last_layer_act: torch.Tensor, last_layer: int,
                   offsets: list[list[int]], prompt: str, query_text: str,
                   concepts: list[str], raw_map: dict[str, str],
                   seq_len: int) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Transported last-source-layer residuals at concept token positions.

    Location: char-offset search for the concept (or its pre-completion raw
    form, e.g. "chemo" for "chemotherapy") inside the query-text span of the
    prompt; the token whose offset span covers the match start is the read
    position (first-fragment convention, same as phase39's prefill locating).
    """
    ws_vecs: dict[str, np.ndarray] = {}
    positions: dict[str, int] = {}
    if not concepts:
        return ws_vecs, positions

    query_slice = query_text[:400]  # build_concern_prompt_full truncates at 400
    qstart = prompt.find(query_slice)
    if qstart < 0:
        return ws_vecs, positions
    qend = qstart + len(query_slice)
    prompt_lower = prompt.lower()

    def char_to_token(char_idx: int) -> int | None:
        for pos, (s, e) in enumerate(offsets):
            if pos >= seq_len:
                break
            if s <= char_idx < e:
                return pos
        return None

    read_positions: dict[str, int] = {}
    for concept in concepts:
        for form in (concept, raw_map.get(concept, concept)):
            idx = prompt_lower.find(form.lower(), qstart, qend)
            if idx >= 0:
                pos = char_to_token(idx)
                if pos is not None:
                    read_positions[concept] = pos
                    break

    if not read_positions:
        return ws_vecs, positions

    pos_list = sorted(set(read_positions.values()))
    resid = last_layer_act[0, pos_list].float()               # [n_pos, d]
    transported = lens.transport(resid, last_layer)           # final-layer basis
    pos_to_idx = {p: i for i, p in enumerate(pos_list)}
    for concept, pos in read_positions.items():
        vec = transported[pos_to_idx[pos]].float().cpu().numpy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            ws_vecs[concept] = vec / norm
            positions[concept] = pos
    return ws_vecs, positions


# ── Concept-space matching + retrieval arms ─────────────────────────────


def match_vocab_concepts(ws_vecs: dict[str, np.ndarray],
                         vocab_ws_norm: np.ndarray,
                         vocab_concepts: list[str],
                         top_match: int = TOP_MATCH
                         ) -> tuple[dict[str, list[dict]], np.ndarray]:
    """Cosine-match query concept vectors against the vocab ws_vec table.

    Returns ({query_concept: [{concept, cosine} x top_match]}, q):
    q = sum over query concepts of max(cos,0)-weighted top-match indicator
    vectors over the vocab axis (NOT yet L2-normalized — caller normalizes
    after mapping to the M axis).
    """
    matches: dict[str, list[dict]] = {}
    q = np.zeros(len(vocab_concepts), dtype=np.float64)
    for qc, vec in ws_vecs.items():
        cos = vocab_ws_norm @ vec
        top_idx = np.argsort(cos)[::-1][:top_match]
        entries = []
        for i in top_idx:
            c = float(cos[i])
            entries.append({"concept": vocab_concepts[i], "cosine": round(c, 4)})
            q[i] += max(c, 0.0)
        matches[qc] = entries
    return matches, q


def q_to_m_axis(q_vocab: np.ndarray, vocab_concepts: list[str],
                concept_index: dict[str, int], n_concepts: int) -> np.ndarray:
    """Map a vocab-axis q vector onto M's concept axis (string alignment)."""
    q = np.zeros(n_concepts, dtype=np.float64)
    for i, c in enumerate(vocab_concepts):
        j = concept_index.get(c)
        if j is not None and q_vocab[i] != 0:
            q[j] = q_vocab[i]
    norm = np.linalg.norm(q)
    return q / norm if norm > 0 else q


def zscore(v: np.ndarray) -> np.ndarray:
    std = float(v.std())
    if std <= 0:
        return np.zeros_like(v)
    return (v - float(v.mean())) / std


def topk_by_score(score_vec: np.ndarray, chunk_ids: list[str],
                  k: int = TOP_K) -> list[str]:
    order = np.argsort(score_vec)[::-1][:k]
    return [chunk_ids[i] for i in order]


# ── Per-domain run ───────────────────────────────────────────────────────


def run_domain(domain: str, lens, lens_model, tokenizer,
               max_queries: int = 0, smoke: bool = False,
               out_dir: Path = EXP, verbose: bool = True) -> dict:
    from experiments.embed_cache import CachedBgeM3Provider

    cache, vecs, relations = load_phase41_inputs(domain)
    chunk_text_map = load_corpus_texts(domain, cache)
    n = DEFAULT_QUERIES.get(domain, 56) if max_queries == 0 else max_queries
    if smoke:
        # load the full default set, then take the literal first 5 —
        # load_graphrag_bench(5) would balance to 1/level = 4 questions
        _, questions_all = load_graphrag_bench(
            domain, DEFAULT_QUERIES.get(domain, 56))
        questions = questions_all[:5]
    else:
        _, questions_all = load_graphrag_bench(domain, n)
        questions = questions_all

    # Corpus resources for Pass 1 filtering / BM25 completion (corpus-side,
    # exactly as the document pipeline — solves chemo→chemotherapy matching)
    chunk_texts_list = list(chunk_text_map.values())
    corpus_words = build_corpus_word_set(chunk_texts_list)
    corpus_freq = build_corpus_term_freq(chunk_texts_list)
    wn_nouns = _get_wordnet_nouns()

    embed_fn = CachedBgeM3Provider().embed
    base = RetrievalBase(cache, vecs, relations, chunk_text_map, embed_fn)

    # Vocab ws table, row-normalized (npz rows are means of normalized
    # instance vectors — not unit norm)
    vocab_concepts = [str(c) for c in vecs["concepts"]]
    vocab_ws = vecs["ws_vec"]
    norms = np.linalg.norm(vocab_ws, axis=1, keepdims=True)
    vocab_ws_norm = vocab_ws / np.where(norms > 0, norms, 1.0)

    all_layers = lens.source_layers
    last_layer = all_layers[-1]

    if verbose:
        print(f"  [{domain}] {len(base.chunk_ids)} chunks, "
              f"{len(base.concepts)} concepts, {len(questions)} queries, "
              f"vocab ws rows: {len(vocab_concepts)}")

    query_emb = np.asarray(
        embed_fn([q["question"] for q in questions]), dtype=np.float64)

    per_query: list[dict] = []
    n_fallback_total = 0
    t_start = time.perf_counter()

    for qi, q in enumerate(questions):
        t0 = time.perf_counter()
        qtext = q["question"]

        # Pass 1: concepts + arm-A vectors (1 forward, +1 if fallback fires)
        p1 = pass1_query(lens, lens_model, tokenizer, qtext, all_layers,
                         corpus_words, corpus_freq, wn_nouns)
        n_fallback_total += p1["n_fallback_forwards"]

        # Arm B: reversed-prefill ws vectors (+1 forward)
        _roles, ws_b = pass2_role_expansion(
            lens, lens_model, tokenizer, qtext, p1["concepts"],
            corpus_words, last_layer)

        # Match into the vocab concept space
        matches_a, q_vocab_a = match_vocab_concepts(
            p1["ws_vecs"], vocab_ws_norm, vocab_concepts)
        matches_b, q_vocab_b = match_vocab_concepts(
            ws_b, vocab_ws_norm, vocab_concepts)
        q_a = q_to_m_axis(q_vocab_a, vocab_concepts,
                          base.concept_index, len(base.concepts))
        q_b = q_to_m_axis(q_vocab_b, vocab_concepts,
                          base.concept_index, len(base.concepts))

        # A/B vector consistency on shared concepts
        shared = sorted(set(p1["ws_vecs"]) & set(ws_b))
        ab_cos = {c: round(float(p1["ws_vecs"][c] @ ws_b[c]), 4)
                  for c in shared}

        # Controls
        qv = query_emb[qi]
        seed_ids, b0_ids = base.seed_and_b0(qv)
        bge_cos = (base.chunk_emb @ (qv / (np.linalg.norm(qv) + 1e-8)))
        b0_scores = {cid: float(s) for cid, s in zip(base.chunk_ids, bge_cos)
                     if s > 0}
        graph_scores = propagate_scored(base.graph, seed_ids)
        graph_rank = merge_ranking(seed_ids, graph_scores, b0_ids)

        # Arms
        scores_a = q_a @ base.m
        scores_b = q_b @ base.m
        hybrid = zscore(bge_cos) + HYBRID_ALPHA * zscore(scores_b)
        arm_scores = {
            "q_ws_A": scores_to_dict(scores_a, base.chunk_ids),
            "q_ws_B": scores_to_dict(scores_b, base.chunk_ids),
        }
        rankings = {
            "q_ws_A": topk_by_score(scores_a, base.chunk_ids),
            "q_ws_B": topk_by_score(scores_b, base.chunk_ids),
            "hybrid": topk_by_score(hybrid, base.chunk_ids),
            "B0": b0_ids,
        }

        # Retrieval metrics vs B0 and vs graph
        metrics = {}
        for arm in ARMS:
            sc = (arm_scores[arm] if arm in arm_scores
                  else {cid: float(hybrid[j])
                        for j, cid in enumerate(base.chunk_ids)})
            metrics[arm] = {
                "vs_b0": {
                    "top10_overlap": topk_overlap(rankings[arm], rankings["B0"]),
                    "kendall_tau": kendall_tau_scores(sc, b0_scores),
                },
                "vs_graph": {
                    "top10_overlap": topk_overlap(rankings[arm], graph_rank),
                    "kendall_tau": kendall_tau_scores(sc, graph_scores),
                },
            }

        rec = {
            "qid": q.get("id", str(qi)),
            "level": q.get("level"),
            "question": qtext,
            "answer": q.get("answer", ""),
            "concepts": p1["concepts"],
            "raw_map": p1["raw_map"],
            "arm_a_positions": p1["positions"],
            "n_arm_a_vectors": len(p1["ws_vecs"]),
            "n_arm_b_vectors": len(ws_b),
            "matches_A": matches_a,
            "matches_B": matches_b,
            "ab_cosine": ab_cos,
            "ab_cosine_mean": (round(float(np.mean(list(ab_cos.values()))), 4)
                               if ab_cos else None),
            "rankings": rankings,
            "graph_ranking": graph_rank,
            "metrics": metrics,
            "time_s": round(time.perf_counter() - t0, 3),
        }
        per_query.append(rec)
        if verbose:
            print(f"    [{domain}] {qi + 1}/{len(questions)} "
                  f"concepts={len(p1['concepts'])} "
                  f"vecA={len(p1['ws_vecs'])} vecB={len(ws_b)} "
                  f"({rec['time_s']}s)", flush=True)

    elapsed = time.perf_counter() - t_start

    # ── Answer layer (skipped in --smoke) ──
    acc_results = None
    if not smoke:
        acc_results = _eval_answers(per_query, base, questions, verbose)

    # ── Aggregate retrieval metrics ──
    retrieval_summary = {}
    for arm in ARMS:
        agg = {}
        for ctrl in ("vs_b0", "vs_graph"):
            ovs = [r["metrics"][arm][ctrl]["top10_overlap"] for r in per_query]
            taus = [r["metrics"][arm][ctrl]["kendall_tau"] for r in per_query
                    if r["metrics"][arm][ctrl]["kendall_tau"] is not None]
            agg[ctrl] = {
                "mean_top10_overlap": round(float(np.mean(ovs)), 4),
                "mean_kendall_tau": (round(float(np.mean(taus)), 4)
                                     if taus else None),
                "n_tau": len(taus),
            }
        retrieval_summary[arm] = agg

    if verbose:
        print(f"\n  [{domain}] retrieval summary ({len(per_query)} queries)")
        print(f"  {'arm':<10} {'ov/B0':>7} {'tau/B0':>7} "
              f"{'ov/graph':>9} {'tau/graph':>9}")
        for arm in ARMS:
            a = retrieval_summary[arm]
            fmt = lambda x: f"{x:.3f}" if x is not None else "n/a"
            print(f"  {arm:<10} {fmt(a['vs_b0']['mean_top10_overlap']):>7} "
                  f"{fmt(a['vs_b0']['mean_kendall_tau']):>7} "
                  f"{fmt(a['vs_graph']['mean_top10_overlap']):>9} "
                  f"{fmt(a['vs_graph']['mean_kendall_tau']):>9}")
        if acc_results:
            print(f"\n  [{domain}] ACC summary")
            for arm in EVAL_ARMS:
                r = acc_results[arm]
                lv = " ".join(f"{k}={v['acc']:.2f}" if v["acc"] is not None
                              else f"{k}=n/a"
                              for k, v in r["by_level"].items())
                acc_s = f"{r['acc']:.3f}" if r["acc"] is not None else "n/a"
                print(f"  {arm:<10} ACC={acc_s}  {lv}")

    out = {
        "method": "phase42_query_closure",
        "domain": domain,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "top_k": TOP_K, "seed_k": SEED_K, "top_match": TOP_MATCH,
            "hybrid_alpha": HYBRID_ALPHA, "hybrid_q": "q_ws_B",
            "max_queries": max_queries, "smoke": smoke,
            "match_weight": "max(cosine, 0)",
        },
        "n_chunks": len(base.chunk_ids),
        "n_concepts": len(base.concepts),
        "n_vocab_ws_rows": len(vocab_concepts),
        "n_queries": len(per_query),
        "retrieval_summary": retrieval_summary,
        "acc": acc_results,
        "per_query": per_query,
        "meta": {
            "model": detect_model()["name"],
            "n_fallback_forwards": n_fallback_total,
            "elapsed_s": round(elapsed, 1),
            "s_per_query": round(elapsed / max(len(per_query), 1), 2),
        },
    }

    if not smoke:
        out_path = out_dir / f"phase42_query_closure_{domain}.json"
        # strip bulky per-query answers from disk copy? keep everything.
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        if verbose:
            print(f"  saved to {out_path}")
    return out


def _eval_answers(per_query: list[dict], base: RetrievalBase,
                  questions: list[dict], verbose: bool) -> dict:
    """ACC for EVAL_ARMS: top-10 context → generate_answer → judge (DeepSeek)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from jgraphrag.llm import DeepSeekProvider

    n_calls = 2 * len(EVAL_ARMS) * len(per_query)
    if verbose:
        print(f"  answer layer: ~{n_calls} DeepSeek calls "
              f"(answer + judge x {len(EVAL_ARMS)} arms x {len(per_query)} queries)")

    contexts = {}
    for rec in per_query:
        contexts[rec["qid"]] = {
            arm: " ".join(base.chunk_text_map[cid]
                          for cid in rec["rankings"][arm])
            for arm in EVAL_ARMS
        }

    def _eval(rec):
        llm = DeepSeekProvider()
        res = {}
        for arm in EVAL_ARMS:
            acc = None
            if rec["answer"]:
                ans = generate_answer(rec["question"], contexts[rec["qid"]][arm],
                                      llm)
                acc = bool(judge_answer_correctness(
                    rec["question"], ans, rec["answer"], llm))
            res[arm] = acc
        return rec["qid"], res

    acc_by_qid: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_eval, rec) for rec in per_query]
        done = 0
        for fut in as_completed(futures):
            qid, res = fut.result()
            acc_by_qid[qid] = res
            done += 1
            if verbose and done % 10 == 0:
                print(f"    eval {done}/{len(per_query)}", flush=True)

    for rec in per_query:
        rec["acc"] = acc_by_qid.get(rec["qid"], {})

    results = {}
    for arm in EVAL_ARMS:
        accs = [1.0 if rec["acc"].get(arm) else 0.0
                for rec in per_query if rec["acc"].get(arm) is not None]
        by_level = {}
        for lv in LEVELS:
            rs = [r for r in per_query if r["level"] == lv]
            lv_accs = [1.0 if r["acc"].get(arm) else 0.0
                       for r in rs if r["acc"].get(arm) is not None]
            by_level[lv] = {
                "acc": float(np.mean(lv_accs)) if lv_accs else None,
                "n": len(rs),
            }
        results[arm] = {
            "acc": float(np.mean(accs)) if accs else None,
            "by_level": by_level,
            "n": len(per_query),
        }
    return results


# ── Smoke diagnostics ────────────────────────────────────────────────────


def print_smoke_report(out: dict):
    """Human-inspectable per-query dump for --smoke runs."""
    n_extract_fail = 0
    for rec in out["per_query"]:
        print(f"\n{'=' * 74}")
        print(f"Q ({rec['qid']}, {rec['level']}): {rec['question']}")
        print(f"  concepts ({len(rec['concepts'])}): {rec['concepts']}")
        print(f"  raw→completed: {rec['raw_map']}")
        if not rec["concepts"]:
            n_extract_fail += 1
        for arm, key in (("A", "matches_A"), ("B", "matches_B")):
            print(f"  arm {arm} matches (top-{TOP_MATCH} vocab):")
            for qc, entries in rec[key].items():
                s = ", ".join(f"{e['concept']}({e['cosine']:.3f})"
                              for e in entries)
                print(f"    {qc}: {s}")
        if rec["ab_cosine"]:
            examples = list(rec["ab_cosine"].items())[:2]
            ex_s = ", ".join(f"{c}={v:.3f}" for c, v in examples)
            print(f"  A-vs-B vector cosine: mean={rec['ab_cosine_mean']} "
                  f"(e.g. {ex_s})")
        for arm in EVAL_ARMS:
            print(f"  top-5 [{arm}]: {rec['rankings'][arm][:5]}")
    print(f"\n{'=' * 74}")
    print(f"smoke: {len(out['per_query'])} queries, "
          f"extraction failures (empty concepts): {n_extract_fail}")
    print(f"meta: {out['meta']}")


# ── main ─────────────────────────────────────────────────────────────────


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ap = argparse.ArgumentParser(
        description="Phase 42: query-side closure — queries enter concept "
                    "space via the two-step pipeline, cosine replaces bge")
    ap.add_argument("--domain", default="all",
                    choices=["medical", "novel", "all"])
    ap.add_argument("--max-queries", type=int, default=0,
                    help="0 = domain default (medical 56, novel 48)")
    ap.add_argument("--smoke", action="store_true",
                    help="medical, first 5 queries, diagnostics only "
                         "(no LLM judge, no output file)")
    args = ap.parse_args()

    print("=" * 60)
    print("Phase 42: query-side closure (query → concept space, no bge)")
    print("=" * 60)

    print("\n[1/2] Loading model + lens...")
    cand = detect_model()
    lens = load_lens(cand["local_lens_path"])
    model_src = (cand["local_model_dir"]
                 if _model_dir_complete(cand["local_model_dir"])
                 else cand["model_id"])
    model, tokenizer = load_model(model_src, use_4bit=cand["needs_4bit"])

    import jlens
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)

    print("\n[2/2] Query-closure runs...")
    if args.smoke:
        out = run_domain("medical", lens, lens_model, tokenizer,
                         max_queries=5, smoke=True)
        print_smoke_report(out)
        return

    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]
    for domain in domains:
        run_domain(domain, lens, lens_model, tokenizer,
                   max_queries=args.max_queries)


if __name__ == "__main__":
    main()
