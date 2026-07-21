"""Phase 26: Fast-GraphRAG benchmark——同 Qwen + 同 bge-m3 公平对比。

用 Fast-GraphRAG 框架（LLM generate() 提取 + PageRank 检索）跑 GraphRAG-Bench
medical，和 J-GraphRAG 基线对比 ACC + 建图时间 + VRAM。

前置条件：本地 LLM 服务已启动（phase26_local_llm_server.py --port 8000）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.phase26_fast_graphrag
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase4_dig_graphragbench import load_graphrag_bench
from experiments.phase26_acc_eval import generate_answer, judge_answer_correctness

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"
EXP.mkdir(parents=True, exist_ok=True)

TOP_K = 10
LLM_BASE_URL = "http://127.0.0.1:8000/v1"
LLM_API_KEY = "dummy"
LLM_MODEL = "qwen2.5-7b-instruct"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024


def run_fast_graphrag(domain: str = "medical",
                      max_queries: int = 28,
                      max_chunks: int = 200):
    print(f"Phase 26: Fast-GraphRAG benchmark")
    print(f"  domain={domain}, LLM={LLM_MODEL}, embed={EMBED_MODEL}")
    print(f"  LLM endpoint: {LLM_BASE_URL}")
    print(f"{'='*70}")

    # 1. Load corpus
    print(f"\n[1/4] Loading corpus...")
    corpus_chunks_dict, questions = load_graphrag_bench(domain, max_queries)
    chunk_items = list(corpus_chunks_dict.items())[:max_chunks]
    chunk_ids = [cid for cid, _ in chunk_items]
    chunk_text_map = {cid: text for cid, text in chunk_items}
    print(f"  {len(chunk_ids)} chunks, {len(questions)} questions")

    # 2. Configure Fast-GraphRAG
    print(f"\n[2/4] Configuring Fast-GraphRAG...")
    from fast_graphrag import GraphRAG
    from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService
    import instructor

    DOMAIN = "Medical and healthcare information"
    ENTITY_TYPES = ["disease", "treatment", "medication", "symptom",
                     "body_part", "procedure", "test", "patient_group"]

    working_dir = str(REPO / "data" / "m6" / "fast_graphrag_workspace")
    os.makedirs(working_dir, exist_ok=True)

    grag = GraphRAG(
        working_dir=working_dir,
        domain=DOMAIN,
        example_queries="What is the most common type of skin cancer?",
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=OpenAILLMService(
                model=LLM_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                mode=instructor.Mode.JSON,
            ),
            embedding_service=OpenAIEmbeddingService(
                model=EMBED_MODEL,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                embedding_dim=EMBED_DIM,
            ),
        ),
    )
    print(f"  Fast-GraphRAG configured (workspace: {working_dir})")

    # 3. Insert chunks (TIMED — this is the "graph build" phase)
    print(f"\n[3/4] Inserting {len(chunk_ids)} chunks (timed)...")
    t_start = time.perf_counter()

    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    async def insert_all():
        for i, cid in enumerate(chunk_ids):
            text = chunk_text_map[cid]
            try:
                await grag.insert(text)
            except Exception as e:
                print(f"    chunk {i} insert error: {e}", flush=True)
            if (i + 1) % 20 == 0:
                elapsed = time.perf_counter() - t_start
                print(f"    {i+1}/{len(chunk_ids)} inserted ({elapsed:.0f}s)", flush=True)

    asyncio.get_event_loop().run_until_complete(insert_all())

    t_build = time.perf_counter() - t_start
    avg_build = t_build / len(chunk_ids)
    print(f"  Build time: {t_build:.1f}s ({avg_build:.3f}s/chunk)")

    # 4. Query + evaluate
    print(f"\n[4/4] Querying + evaluating ({len(questions)} questions)...")
    from jgraphrag.llm import DeepSeekProvider
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Fast-GraphRAG query is async — use nest_asyncio for nested loops
    async def query_one(question):
        try:
            result = await grag.query(question)
            return result.response if hasattr(result, 'response') else str(result)
        except Exception as e:
            return f"[ERROR: {e}]"

    def _eval_query(q_data):
        q = q_data
        level = q["level"]
        question = q["question"]
        gold_answer = q.get("answer", "")

        # Fast-GraphRAG query
        fgr_response = asyncio.get_event_loop().run_until_complete(query_one(question))

        # FGR generates its own answer from its retrieval
        fgr_answer = fgr_response.strip()

        # ACC
        llm_local = DeepSeekProvider()
        fgr_acc = judge_answer_correctness(question, fgr_answer, gold_answer, llm_local)

        # Also get B0 for comparison
        # B0 is already computed in phase26_acc_eval baseline — skip here

        return {
            "level": level,
            "question": question[:80],
            "fgr_answer": fgr_answer[:150],
            "gold_answer": gold_answer[:150],
            "fgr_acc": fgr_acc,
        }

    results_list = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_eval_query, q): q for q in questions if q.get("answer")}
        done = 0
        for future in as_completed(futures):
            try:
                results_list.append(future.result())
            except Exception as e:
                print(f"    eval error: {e}", flush=True)
            done += 1
            if done % 10 == 0:
                print(f"    {done}/{len(questions)}", flush=True)

    # Summary
    acc_values = [1.0 if r["fgr_acc"] else 0.0 for r in results_list]
    by_level = defaultdict(list)
    for r in results_list:
        by_level[r["level"]].append(1.0 if r["fgr_acc"] else 0.0)

    overall_acc = np.mean(acc_values) if acc_values else 0

    print(f"\n{'='*70}")
    print(f"FAST-GRAPHRAG RESULTS ({domain})")
    print(f"{'='*70}")
    print(f"  ACC: {overall_acc:.1%} ({sum(acc_values)}/{len(acc_values)})")
    print(f"  Build time: {t_build:.1f}s ({avg_build:.3f}s/chunk)")
    print(f"  Per-level ACC:")
    for lv in ["L1", "L2", "L3", "L4"]:
        s = by_level.get(lv, [])
        if s:
            print(f"    {lv}: {np.mean(s):.1%} ({sum(s)}/{len(s)})")

    # Save
    out = {
        "method": "fast_graphrag",
        "domain": domain,
        "llm": LLM_MODEL,
        "embedding": EMBED_MODEL,
        "n_chunks": len(chunk_ids),
        "n_questions": len(results_list),
        "efficiency": {
            "build_time_s": round(t_build, 1),
            "build_time_per_chunk_s": round(avg_build, 4),
        },
        "results": {
            "acc": float(overall_acc),
            "acc_by_level": {lv: (float(np.mean(by_level[lv]))
                                  if by_level.get(lv) else None)
                             for lv in ["L1","L2","L3","L4"]},
            "n": len(acc_values),
        },
        "per_query": results_list,
    }
    out_path = EXP / f"phase26_fast_graphrag_{domain}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n  saved to {out_path}")
    return out


def main():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Verify LLM server is running
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{LLM_BASE_URL.replace('/v1','')}/health", timeout=5)
        print(f"LLM server health: {resp.read().decode()}")
    except Exception as e:
        print(f"ERROR: LLM server not running at {LLM_BASE_URL}. Start it first:")
        print(f"  python -m experiments.phase26_local_llm_server --port 8000")
        return

    run_fast_graphrag(domain="medical", max_queries=28, max_chunks=50)


if __name__ == "__main__":
    main()
