"""Phase 39b: twopass 缓存后处理——套用 Phase 25 验证过的过滤配置。

背景：phase39 的 Pass 1 直接复用 phase31 的 extract_concepts_full_pipeline，
其输出未经过 Phase 25 的下游过滤（DF≥2 + prefill 黑名单 + POS 动词拒绝），
导致 "listed"/"summarized"/"hinted" 等 prefill 模板词进入概念集
（旧 concept_cache_{domain}_full.json 同样如此——Phase 25 当年是靠
phase25_filter_with_bpe 在分析时剔除的）。

本脚本对 concept_cache_{domain}_twopass.json + concept_vecs_{domain}.npz
做确定性后处理，规则与 phase25_filter_with_bpe 严格一致（概念层），
角色词另加 RB 副词拒绝和 phase39 的 ROLE_STOP：

  概念过滤（= Phase 25 配置）：
    1. DF ≥ 2
    2. ASCII English
    3. 不在 PREFILL_WORDS / STOP_WORDS_EXTENDED
    4. POS 不是 VBG/VBD（动词形式）

  角色过滤（在上述基础上追加）：
    5. POS 不是 RB*（副词：below/briefly/under 等）
    6. 不在 ROLE_STOP（type/basics/part 等通用模板词）

产出（原文件备份为 *_raw.json / *_raw.npz）：
  - concept_cache_{domain}_twopass.json（过滤后，含 filter_stats）
  - concept_vecs_{domain}.npz（剔除被过滤概念的行）

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    python -m experiments.phase39b_filter_cache [--domain all]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.phase20_concern_full_com import PREFILL_WORDS
from experiments.phase18_centroid_hierarchy import (
    STOP_WORDS_EXTENDED, is_ascii_english,
)
from experiments.phase16a_cross_domain_pos import classify_concept_pos
from experiments.phase39_two_pass_cache import ROLE_STOP, _stem

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6" / "concept_cache"


def concept_ok(word: str) -> tuple[bool, str]:
    """Phase 25 concept filter. Returns (keep, reason_if_dropped)."""
    w = word.lower()
    if not is_ascii_english(w):
        return False, "non_ascii"
    if w in PREFILL_WORDS or w in STOP_WORDS_EXTENDED:
        return False, "prefill_stopword"
    pos = classify_concept_pos(w)
    if pos in ("VBG", "VBD"):
        return False, "verb_form"
    return True, ""


def role_ok(word: str) -> bool:
    """Role filter: concept rules + adverbs + ROLE_STOP (stem-aware)."""
    w = word.lower()
    ok, _ = concept_ok(w)
    if not ok:
        return False
    if w in ROLE_STOP or _stem(w) in ROLE_STOP:
        return False
    if classify_concept_pos(w).startswith("RB"):
        return False
    return True


def filter_domain(domain: str) -> None:
    cache_path = EXP / f"concept_cache_{domain}_twopass.json"
    npz_path = EXP / f"concept_vecs_{domain}.npz"
    cache = json.loads(cache_path.read_text())

    # Pass A: decide kept concepts (DF>=2 + phase25 rules)
    concept_chunks_raw = cache["concept_chunks"]
    kept_concepts: dict[str, list[str]] = {}
    removed: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for concept, chunks in concept_chunks_raw.items():
        df = len(chunks)
        if df < 2:
            removed["too_rare"].append((concept, df))
            continue
        ok, reason = concept_ok(concept)
        if ok:
            kept_concepts[concept] = chunks
        else:
            removed[reason].append((concept, df))

    # Pass B: rewrite per-chunk concepts + roles
    n_roles_before = n_roles_after = 0
    for cid, data in cache["chunks"].items():
        new_concepts = [c for c in data["concepts"]
                        if c.lower() in kept_concepts]
        new_roles: dict[str, list[str]] = {}
        for concept, roles in data.get("roles", {}).items():
            if concept.lower() not in kept_concepts:
                continue
            n_roles_before += len(roles)
            filtered = [r for r in roles if role_ok(r)]
            # inflection dedupe within the concept
            seen: set[str] = set()
            deduped = []
            for r in filtered:
                s = _stem(r)
                if s not in seen:
                    seen.add(s)
                    deduped.append(r)
            n_roles_after += len(deduped)
            if deduped:
                new_roles[concept] = deduped
        data["concepts"] = new_concepts
        data["roles"] = new_roles
        data["n_concepts"] = len(new_concepts)

    # Pass C: rebuild aggregates (only count concepts surviving per-chunk)
    concept_chunks: dict[str, list[str]] = defaultdict(list)
    for cid, data in cache["chunks"].items():
        for c in data["concepts"]:
            concept_chunks[c.lower()].append(cid)
    concept_freq = Counter({c: len(v) for c, v in concept_chunks.items()})
    role_freq: Counter = Counter()
    for data in cache["chunks"].values():
        for roles in data["roles"].values():
            role_freq.update(r.lower() for r in roles)

    cache["concept_chunks"] = dict(concept_chunks)
    cache["concept_frequency"] = dict(concept_freq.most_common(40))
    cache["n_unique_concepts"] = len(concept_freq)
    cache["role_frequency"] = dict(role_freq.most_common(40))
    cache["n_unique_roles"] = len(role_freq)
    cache["filter_stats"] = {
        "config": "phase25 (DF>=2 + ASCII + PREFILL_WORDS/STOP_WORDS_EXTENDED "
                  "+ POS !VBG/VBD); roles += !RB + ROLE_STOP",
        "concepts_before": len(concept_chunks_raw),
        "concepts_after": len(concept_chunks),
        "removed": {k: sorted(v, key=lambda x: -x[1])[:20]
                    for k, v in removed.items()},
        "roles_before": n_roles_before,
        "roles_after": n_roles_after,
    }

    # Pass D: filter npz rows
    z = np.load(npz_path, allow_pickle=True)
    concepts_arr = z["concepts"]
    keep_mask = np.array([c in concept_chunks for c in concepts_arr])
    np.savez(
        npz_path.with_suffix(".filtered.npz"),
        concepts=concepts_arr[keep_mask],
        wu_vec=z["wu_vec"][keep_mask],
        ws_vec=z["ws_vec"][keep_mask],
        count=z["count"][keep_mask],
    )

    # Backup originals, then replace
    shutil.copy2(cache_path, cache_path.with_suffix(".raw.json"))
    shutil.copy2(npz_path, npz_path.with_suffix(".raw.npz"))
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    npz_path.with_suffix(".filtered.npz").replace(npz_path)

    print(f"[{domain}] concepts: {len(concept_chunks_raw)} -> "
          f"{len(concept_chunks)} | roles: {n_roles_before} -> {n_roles_after}")
    print(f"[{domain}] removed: "
          f"{ {k: len(v) for k, v in removed.items()} }")
    print(f"[{domain}] top concepts: {list(concept_freq.most_common(10))}")
    print(f"[{domain}] top roles: {list(role_freq.most_common(10))}")
    print(f"[{domain}] kept npz rows: {int(keep_mask.sum())}/{len(keep_mask)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="all",
                    choices=["medical", "novel", "all"])
    args = ap.parse_args()
    domains = ["medical", "novel"] if args.domain == "all" else [args.domain]
    for domain in domains:
        filter_domain(domain)


if __name__ == "__main__":
    main()
