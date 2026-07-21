"""反义词反向研究：用对立概念反推正向条件信息。

核心假设：如果反义词前缀把嵌入推向方向 d_anti，
那么 d_pos（正向条件方向）可能 ≈ −d_anti（对称性）。
如果是这样，可以用反义词偏移反推正向信息：
    v_enhanced ≈ v_original − α · (v_anti − v_original)

验证步骤：
  1. 对每个概念，构建近义词集和反义词集
  2. 测量：v_base, v_syn（近义词前缀）, v_anti（反义词前缀）
  3. 计算：d_syn = v_syn − v_base, d_anti = v_anti − v_base
  4. 检验：cos(d_syn, d_anti) 是否 < 0（对称）？
  5. 如果对称：反推 v_enhanced = v_base − α·d_anti，测它和 v_syn 的相似度
  6. 如果反推方向接近 v_syn → 技巧成立
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

# 近义词 + 反义词配对（跨领域）
CONCEPT_PAIRS = {
    "authentication": {
        "synonyms": ["auth", "login", "password", "credential", "session", "token",
                      "verify", "identity", "access", "security", "hash", "bcrypt"],
        "antonyms": ["logout", "signout", "disconnect", "revoke", "invalidate", "block",
                      "deny", "reject", "forbid", "ban", "expire", "remove"],
        "text": "function verifyPassword(user, input) { const hash = bcrypt.hash(input); return user.hash === hash; }",
    },
    "nutrition": {
        "synonyms": ["diet", "vitamin", "nutrient", "protein", "metabolism", "food",
                      "supplement", "mineral", "fiber", "antioxidant", "health", "nutrition"],
        "antonyms": ["malnutrition", "starvation", "deficiency", "toxin", "poison",
                      "junk", "fastfood", "deprivation", "fasting", "hunger", "empty", "unhealthy"],
        "text": "Recent studies suggest that high-fiber diets may reduce the risk of colorectal cancer through improved gut microbiome diversity.",
    },
    "machine_learning": {
        "synonyms": ["model", "training", "neural", "network", "learning", "dataset",
                      "feature", "classification", "optimization", "gradient", "accuracy", "inference"],
        "antonyms": ["random", "untrained", "static", "fixed", "memorize", "overfit",
                      "noise", "misclassification", "divergence", "loss", "error", "guess"],
        "text": "We propose a novel transformer architecture that achieves state-of-the-art performance on the GLUE benchmark.",
    },
    "cardiovascular": {
        "synonyms": ["heart", "blood", "vessel", "artery", "circulation", "cardiac",
                      "cholesterol", "statin", "lipid", "coronary", "pressure", "cardiovascular"],
        "antonyms": ["brain", "bone", "muscle", "skin", "liver", "kidney",
                      "nerve", "lung", "stomach", "intestine", "spine", "respiratory"],
        "text": "Statin therapy has been shown to reduce LDL cholesterol levels and decrease cardiovascular mortality.",
    },
    "customer_service": {
        "synonyms": ["customer", "refund", "delivery", "order", "support",
                      "satisfaction", "resolve", "service", "help", "assist", "quality", "care"],
        "antonyms": ["ignore", "charge", "withhold", "cancel", "refuse",
                      "complaint", "neglect", "abandon", "deny", "reject", "fail", "harm"],
        "text": "Customer A reported a delayed delivery of order #12345. The refund was processed according to the 30-day policy.",
    },
}


def build_prefix(words, n):
    if n == 0:
        return ""
    return ", ".join(words[:n]) + ": "


def cosine_np(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_experiment(embed):
    results = {}

    for concept, data in CONCEPT_PAIRS.items():
        text = data["text"]
        syns = data["synonyms"]
        antis = data["antonyms"]

        print(f"\n  [{concept}]")

        v_base = np.asarray(embed.embed_one(text))
        results[concept] = {"text_preview": text[:60]}

        # 对不同前缀词数测方向对称性
        direction_results = []
        for n in [1, 3, 5, 8, 12]:
            if n > min(len(syns), len(antis)):
                continue

            v_syn = np.asarray(embed.embed_one(f"{build_prefix(syns, n)}{text}"))
            v_anti = np.asarray(embed.embed_one(f"{build_prefix(antis, n)}{text}"))

            d_syn = v_syn - v_base     # 正向偏移方向
            d_anti = v_anti - v_base   # 反向偏移方向

            # 核心检验：d_syn 和 d_anti 是否对称（cos < 0）？
            cos_directions = cosine_np(d_syn, d_anti)

            # 偏移幅度
            offset_syn = 1.0 - cosine_np(v_syn, v_base)
            offset_anti = 1.0 - cosine_np(v_anti, v_base)

            # 反推：v_enhanced = v_base − α · d_anti
            # 试多个 α 值，看哪个 α 让 v_enhanced 最接近 v_syn
            best_alpha = 0
            best_sim = -1
            for alpha in np.arange(0.0, 3.0, 0.1):
                v_enhanced = v_base - alpha * d_anti
                sim = cosine_np(v_enhanced, v_syn)
                if sim > best_sim:
                    best_sim = sim
                    best_alpha = alpha

            # 也测：v_enhanced 和 v_base 的偏移（反推是否产生了有意义的偏移）
            v_enhanced_best = v_base - best_alpha * d_anti
            offset_enhanced = 1.0 - cosine_np(v_enhanced_best, v_base)

            entry = {
                "n_words": n,
                "offset_syn": float(offset_syn),
                "offset_anti": float(offset_anti),
                "cos_syn_anti_directions": float(cos_directions),
                "best_alpha": float(best_alpha),
                "enhanced_to_syn_similarity": float(best_sim),
                "offset_enhanced": float(offset_enhanced),
            }
            direction_results.append(entry)

            sym_label = "对称✓" if cos_directions < -0.1 else ("正交~" if abs(cos_directions) < 0.1 else "同向✗")
            print(f"    n={n:2d}: offset syn={offset_syn:.4f} anti={offset_anti:.4f} | "
                  f"dir cos={cos_directions:+.4f} {sym_label} | "
                  f"α*={best_alpha:.1f}, enhanced→syn={best_sim:.4f}")

        results[concept]["direction_analysis"] = direction_results

    # ── 总结 ──
    print(f"\n{'='*60}")
    print("=== Summary: antonym-synonym direction symmetry ===")

    all_cos = []
    for concept, data in results.items():
        for entry in data["direction_analysis"]:
            all_cos.append(entry["cos_syn_anti_directions"])

    all_cos_arr = np.array(all_cos)
    print(f"  direction cos(d_syn, d_anti) across all concepts/word-counts:")
    print(f"    mean={all_cos_arr.mean():.4f}, std={all_cos_arr.std():.4f}")
    print(f"    min={all_cos_arr.min():.4f}, max={all_cos_arr.max():.4f}")
    print(f"    fraction < -0.1 (symmetric): {(all_cos_arr < -0.1).mean():.1%}")
    print(f"    fraction > 0.1 (same direction): {(all_cos_arr > 0.1).mean():.1%}")

    if all_cos_arr.mean() < -0.1:
        print("\n  ★ 对称性成立：反义词偏移方向和近义词偏移方向相反")
        print("    → 反推技巧可行：v_enhanced = v_base − α·d_anti 可近似正向条件")
    elif all_cos_arr.mean() > 0.1:
        print("\n  ✗ 同向：反义词和近义词把嵌入推向相同方向（都偏离原文本）")
        print("    → 反推技巧不可行")
    else:
        print("\n  ~ 正交：反义词和近义词偏移方向不相关")
        print("    → 反推技巧效果有限")

    # enhanced_to_syn 的平均相似度
    all_enhanced_sim = [e["enhanced_to_syn_similarity"]
                        for data in results.values()
                        for e in data["direction_analysis"]]
    print(f"\n  enhanced→syn similarity: mean={np.mean(all_enhanced_sim):.4f}")
    print(f"  (如果 > 0.95，反推方向非常接近真实正向方向)")

    return results


def main():
    embed = CachedBgeM3Provider()
    print("Antonym Reverse Experiment: Can antonyms reveal positive conditional info?")
    print(f"Testing {len(CONCEPT_PAIRS)} concepts × 5 word counts")

    results = run_experiment(embed)

    out_path = EXP / "antonym_reverse.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
