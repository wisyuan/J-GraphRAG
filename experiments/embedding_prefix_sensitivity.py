"""bge-m3 对前缀词汇量的敏感度实验（二分搜索阈值）。

目的：验证"基于范畴先验的重嵌入"路径是否可行。
如果少量先验词汇就能引起嵌入质变 → Phase 6 概念标签 + 重嵌入可行（零 LLM）。
如果需要大量词汇或无阈值 → bge-m3 对前缀不敏感，需 LLM 重嵌入。

方法：
  1. 准备跨领域概念词库（每个概念 → N 个近义词/相关词）
  2. 对同一文本，逐步增加前缀词汇量
  3. 测量 embedding 偏移：1 - cosine(embed(prefixed), embed(original))
  4. 二分搜索偏移量的"质变阈值"

输出：词汇量 → 偏移量曲线 + 质变阈值。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.embed_cache import CachedBgeM3Provider

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "data" / "m6"


# ── 概念词库（跨 4 领域 + WordNet 补充）─────────────────────────────────

CONCEPT_BANK = {
    # 代码域
    "authentication": ["auth", "login", "password", "credential", "session", "token",
                        "verify", "identity", "access", "security", "hash", "bcrypt",
                        "oauth", "jwt", "encrypt"],
    "error_handling": ["error", "exception", "catch", "throw", "try", "fail",
                        "crash", "retry", "timeout", "abort", "raise", "handle",
                        "recover", "rollback", "validate"],
    # 医学域（NFCorpus）
    "nutrition": ["diet", "vitamin", "nutrient", "protein", "metabolism", "food",
                   "supplement", "mineral", "calorie", "fiber", "antioxidant", "deficiency",
                   "malnutrition", "obesity", "health"],
    "cardiovascular": ["heart", "blood", "vessel", "artery", "cholesterol", "statin",
                        "cardiac", "hypertension", "circulation", "lipid", "stroke", "pressure",
                        "coronary", "ventricular", "cardiovascular"],
    # 科学域（SciFact）
    "machine_learning": ["model", "training", "neural", "network", "learning", "dataset",
                          "feature", "classification", "regression", "optimization", "gradient",
                          "accuracy", "loss", "epoch", "inference"],
    "genomics": ["gene", "DNA", "genome", "sequence", "mutation", "expression", "protein",
                 "RNA", "chromosome", "allele", "transcript", "sequencing", "genetic",
                 "epigenetic", "hereditary"],
    # 业务域（enterprise）
    "customer_service": ["customer", "refund", "delivery", "order", "complaint", "support",
                          "satisfaction", "delay", "cancel", "invoice", "warranty", "return",
                          "feedback", "escalate", "resolve"],
}


def build_prefix(words: list[str], n: int) -> str:
    """构建前缀：取前 n 个词，逗号分隔。"""
    if n == 0:
        return ""
    return ", ".join(words[:n]) + ": "


def cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


# ── 主实验 ──────────────────────────────────────────────────────────────

def run_experiment(embed):
    # 测试文本：每个概念配一个该领域的真实文本片段
    test_texts = {
        "authentication": "function verifyPassword(user, input) { const hash = bcrypt.hash(input); return user.hash === hash; }",
        "error_handling": "try { await db.query(sql); } catch (e) { logger.error(e); await retry(3); }",
        "nutrition": "Recent studies suggest that high-fiber diets may reduce the risk of colorectal cancer through improved gut microbiome diversity.",
        "cardiovascular": "Statin therapy has been shown to reduce LDL cholesterol levels and decrease cardiovascular mortality in high-risk patients.",
        "machine_learning": "We propose a novel transformer architecture that achieves state-of-the-art performance on the GLUE benchmark with 40% fewer parameters.",
        "genomics": "CRISPR-Cas9 gene editing enables precise modification of genomic sequences, offering potential treatments for inherited genetic disorders.",
        "customer_service": "Customer A reported a delayed delivery of order #12345. The refund was processed according to the 30-day policy.",
    }

    results = {}

    for concept, text in test_texts.items():
        words = CONCEPT_BANK[concept]
        print(f"\n  [{concept}] {len(words)} prefix words available")

        # 原始嵌入（基准）
        base_vec = embed.embed_one(text)

        # 逐步增加前缀词数：0, 1, 2, 3, 5, 8, 10, 15
        word_counts = [0, 1, 2, 3, 5, 8, 10, 15]
        offsets = []

        for n in word_counts:
            prefix = build_prefix(words, n)
            prefixed = f"{prefix}{text}"
            vec = embed.embed_one(prefixed)
            offset = 1.0 - cosine(vec, base_vec)
            offsets.append({"n_words": n, "offset": float(offset),
                            "cosine_to_base": float(cosine(vec, base_vec))})

        results[concept] = {
            "text_preview": text[:80],
            "n_prefix_words_total": len(words),
            "offset_curve": offsets,
        }

        # 打印曲线
        for entry in offsets:
            bar = "█" * int(entry["offset"] * 100)
            print(f"    n={entry['n_words']:2d}: offset={entry['offset']:.4f} "
                  f"(cos={entry['cosine_to_base']:.4f}) {bar}")

    # ── 二分搜索质变阈值 ──
    print(f"\n{'='*60}")
    print("=== Binary search for phase-change threshold ===")
    print("定义：offset > 0.1 = 质变（嵌入方向显著改变）")

    thresholds = {}
    for concept, data in results.items():
        curve = data["offset_curve"]
        threshold = None
        for i in range(1, len(curve)):
            if curve[i]["offset"] > 0.1 and curve[i-1]["offset"] <= 0.1:
                threshold = curve[i]["n_words"]
                break
        if threshold is None:
            # 检查是否从未超过 0.1
            max_offset = max(e["offset"] for e in curve)
            if max_offset < 0.1:
                threshold = "never exceeds 0.1"
            else:
                threshold = f"≤{curve[1]['n_words']} (immediate)"

        thresholds[concept] = threshold
        print(f"  {concept:25s}: threshold = {threshold}")

    # ── 交叉概念测试（不相关词汇前缀）──
    print(f"\n{'='*60}")
    print("=== Cross-concept control: unrelated prefix words ===")
    # 用 authentication 的词给 nutrition 文本加前缀
    text = test_texts["nutrition"]
    base_vec = embed.embed_one(text)
    unrelated_words = CONCEPT_BANK["authentication"]
    related_words = CONCEPT_BANK["nutrition"]

    for n in [3, 8, 15]:
        # 相关前缀
        rel_prefix = build_prefix(related_words, n)
        rel_vec = embed.embed_one(f"{rel_prefix}{text}")
        rel_offset = 1.0 - cosine(rel_vec, base_vec)

        # 不相关前缀
        unrel_prefix = build_prefix(unrelated_words, n)
        unrel_vec = embed.embed_one(f"{unrel_prefix}{text}")
        unrel_offset = 1.0 - cosine(unrel_vec, base_vec)

        print(f"  n={n:2d}: related={rel_offset:.4f} vs unrelated={unrel_offset:.4f} "
              f"(ratio={rel_offset/unrel_offset:.2f}x)" if unrel_offset > 0 else
              f"  n={n:2d}: related={rel_offset:.4f} vs unrelated={unrel_offset:.4f}")

    return {"concepts": results, "thresholds": thresholds}


def main():
    embed = CachedBgeM3Provider()
    print("bge-m3 Prefix Sensitivity Experiment")
    print(f"Testing {len(CONCEPT_BANK)} concepts × 7 prefix word counts")

    results = run_experiment(embed)

    out_path = EXP / "prefix_sensitivity.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
