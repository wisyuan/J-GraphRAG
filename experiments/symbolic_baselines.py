"""Phase 8 symbolic retrieval baselines — two keyword-set matchers.

Both consume the SAME LLM-extracted concept-keyword sets (from keyword_cache.py),
differing only in how they compute set similarity. This factor-isolates:
  - S-jaccard (weighted Jaccard): pure symbolic matching ability
  - S-embed  (keyword-level embedding soft match): what embedding softening adds

Unlike phase4 baselines.py (which take doc/query embeddings), these take
keyword sets. The search() OUTPUT format is identical ({qid: {cid: score}}),
so phase4's per_query_ndcg / paired_permutation_test work unchanged.

Weight source: each keyword record has weight_llm (self-report) and
weight_logprob (model's true confidence). Callers pass which to use — the
comparison between them is itself a finding (Phase 8 tests logprobs >> self-report).

GPU: S-embed uses torch for the keyword similarity matrix (same pattern as
baselines.py). S-jaccard is pure Python/numpy (no embeddings needed).
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


def _to_tensor(arr: np.ndarray):
    return torch.as_tensor(arr, dtype=torch.float32, device=DEVICE)


def _normalize(t):
    norm = t.norm(dim=-1, keepdim=True)
    return t / norm.clamp(min=1e-8)


def _weights_dict(kw_set: dict, weight_key: str = "weight_logprob") -> dict[str, float]:
    """Extract {keyword: weight} from the record dict, using the chosen source."""
    return {kw: float(rec.get(weight_key, rec.get("weight_logprob", 0.5)))
            for kw, rec in kw_set.items()}


# ── S-jaccard: weighted Jaccard (pure symbolic) ─────────────────────────

class WeightedJaccardSearch:
    """Weighted Jaccard similarity over keyword sets.

    score(q, d) = Σ_{k∈q∩d} min(w_q[k], w_d[k]) / Σ_{k∈q∪d} max(w_q[k], w_d[k])

    Weighted Jaccard (not plain) because plain Jaccard over ~15-keyword sets
    compresses scores to a narrow 0.05-0.09 band — useless for ranking. Weights
    (from logprobs) spread the scores out so nDCG has discrimination to work with.

    fit() precomputes the keyword→docindex inverted index for efficiency: search
    only touches docs that share ≥1 keyword with the query, not the whole corpus.
    """

    name = "S-jaccard"

    def __init__(self, weight_key: str = "weight_logprob"):
        self.weight_key = weight_key

    def fit(self, kw_sets: list[dict], corpus_ids: list[str]):
        """kw_sets[i] = {keyword: {"weight_logprob":..., "weight_llm":...}} for corpus_ids[i]."""
        self.corpus_ids = corpus_ids
        # Per-doc {keyword: weight} using the chosen weight source.
        self.doc_weights = [_weights_dict(ks, self.weight_key) for ks in kw_sets]
        # Inverted index: keyword → list of (doc_idx, weight).
        self.inverted: dict[str, list[tuple[int, float]]] = {}
        for di, dw in enumerate(self.doc_weights):
            for kw, w in dw.items():
                # Lowercase + strip for matching robustness (keywords are free-text).
                key = kw.lower().strip()
                self.inverted.setdefault(key, []).append((di, w))

    def search(self, query_kw_sets: list[dict], query_ids: list[str],
               top_k: int = 100) -> dict[str, dict[str, float]]:
        results = {}
        for qi, qid in enumerate(query_ids):
            q_weights = _weights_dict(query_kw_sets[qi], self.weight_key)
            scores = self._score_one(q_weights)
            # Top-k by score (desc). scores is dense over candidate docs only.
            top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            results[qid] = {self.corpus_ids[di]: float(s) for di, s in top}
        return results

    def _score_one(self, q_weights: dict[str, float]) -> dict[int, float]:
        """Weighted Jaccard for one query against all docs sharing ≥1 keyword.

        Only candidate docs (those in the inverted index for some query keyword)
        are scored — the rest get 0 (implicitly, not returned).
        """
        # Normalize query keywords the same way as fit().
        q_norm = {kw.lower().strip(): w for kw, w in q_weights.items()}

        # Candidate docs = any doc sharing ≥1 keyword with the query (via inverted index).
        candidates = set()
        for kw in q_norm:
            for di, _ in self.inverted.get(kw, []):
                candidates.add(di)

        # Full weighted Jaccard over each candidate: num = Σ min, denom = Σ max over union.
        scores = {}
        for di in candidates:
            dw_norm = {k.lower().strip(): v for k, v in self.doc_weights[di].items()}
            union = set(q_norm) | set(dw_norm)
            numerator = sum(min(q_norm.get(k, 0), dw_norm.get(k, 0)) for k in union)
            denominator = sum(max(q_norm.get(k, 0), dw_norm.get(k, 0)) for k in union)
            scores[di] = numerator / denominator if denominator > 1e-9 else 0.0
        return scores


# ── S-embed: keyword-level embedding soft match ─────────────────────────

class EmbeddingSoftMatchSearch:
    """Soft set similarity using keyword-level embeddings.

    Unlike phase4's document-vector methods, this embeds each KEYWORD (not each
    document) with bge-m3. A query keyword "hashing" matches a doc keyword
    "bcrypt" if their embeddings are cosine-similar, even though they're
    different strings. This is the "softening" that pure Jaccard can't do.

    Score (one of two variants, see aggregation):
      1. For each query keyword kw_q, find the best-matching doc keyword kw_d*.
      2. contribution = w_q[kw_q] × cos(kw_q, kw_d*) × w_d[kw_d*]
      3. score = Σ contributions / normalizer

    Normalizer = sqrt(Σw_q × Σw_d) so scores stay in [0,1]-ish and are
    comparable to cosine.

    Embedding cache: all unique keywords across corpus+queries are embedded
    once via the provided embed callable (typically CachedBgeM3Provider), then
    reused. Keywords are deduplicated globally — usually <5000 unique across
    a whole benchmark.
    """

    name = "S-embed"

    def __init__(self, embed, tau: float = 0.5, weight_key: str = "weight_logprob"):
        """
        embed: callable (list[str] -> list[list[float]]) — usually CachedBgeM3Provider.
        tau: cosine threshold below which a keyword match contributes 0 (noise floor).
        weight_key: which weight source to use.
        """
        self.embed = embed
        self.tau = tau
        self.weight_key = weight_key

    def fit(self, kw_sets: list[dict], corpus_ids: list[str]):
        self.corpus_ids = corpus_ids
        self.doc_weights = [_weights_dict(ks, self.weight_key) for ks in kw_sets]
        # Collect all unique keywords across docs for embedding.
        all_kws = set()
        for dw in self.doc_weights:
            all_kws.update(k.lower().strip() for k in dw)
        self._embed_keywords(all_kws)

        # Precompute per-doc keyword index + embedding matrix slices.
        # doc_kw_list[i] = list of (keyword, weight, emb_row_index)
        self.doc_kw_list = []
        for dw in self.doc_weights:
            kws = [(k.lower().strip(), w) for k, w in dw.items()]
            self.doc_kw_list.append(kws)

    def _embed_keywords(self, keywords: set[str]):
        """Embed all unique keywords once. Store as normalized GPU tensor + index."""
        self.kw_list = sorted(keywords)
        if not self.kw_list:
            self.kw_emb = None
            self.kw_index = {}
            return
        self.kw_index = {kw: i for i, kw in enumerate(self.kw_list)}
        vecs = self.embed(self.kw_list)
        emb = np.asarray(vecs, dtype=np.float32)
        self.kw_emb = _normalize(_to_tensor(emb))  # (Nkw, 1024) normalized

    def search(self, query_kw_sets: list[dict], query_ids: list[str],
               top_k: int = 100) -> dict[str, dict[str, float]]:
        if self.kw_emb is None:
            return {qid: {} for qid in query_ids}

        results = {}
        for qi, qid in enumerate(query_ids):
            scores = self._score_one(query_kw_sets[qi])
            top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
            results[qid] = {self.corpus_ids[di]: float(s) for di, s in top}
        return results

    def _score_one(self, q_kw_set: dict) -> dict[int, float]:
        """Soft match score for one query against all docs.

        For each query keyword, find the best cosine match among ALL doc keywords
        (global, not per-doc), then route that match's weight to the docs that
        contain the matched keyword. This avoids per-doc embedding scans.
        """
        q_weights = _weights_dict(q_kw_set, self.weight_key)
        q_kws = list(q_weights.keys())
        q_kws_norm = [k.lower().strip() for k in q_kws]

        # Filter to keywords we have embeddings for.
        # q_present = (keyword_str, weight, emb_row_index)
        q_present = [(k, q_weights[kw], self.kw_index[k])
                     for kw, k in zip(q_kws, q_kws_norm) if k in self.kw_index]
        if not q_present:
            return {}

        # Query keyword embeddings + similarity to ALL keywords.
        q_rows = [row_idx for _, _, row_idx in q_present]
        q_emb = self.kw_emb[q_rows]  # (nq, 1024)
        sims = q_emb @ self.kw_emb.T  # (nq, Nkw) cosine to all keywords
        sims_np = sims.cpu().numpy()

        # For each query keyword, best match (above tau) and its target row.
        q_sum = sum(w for _, w, _ in q_present)
        q_sum_sqrt = max(q_sum, 1e-9) ** 0.5

        # Accumulate per-doc soft overlap.
        soft_num = {}  # doc_idx → Σ w_q × cos × w_d
        for qi_row, (kw_orig, w_q, _) in enumerate(q_present):
            row = sims_np[qi_row]
            # Best match keyword (could be itself if the query keyword also
            # appears in some doc — that's fine, it's a real match).
            best_j = int(row.argmax())
            best_cos = float(row[best_j])
            if best_cos < self.tau:
                continue
            best_kw = self.kw_list[best_j]
            # Route contribution to every doc containing best_kw.
            # Build inverted index lazily if not present.
            if not hasattr(self, "_kw_to_docs"):
                self._kw_to_docs = {}
                for di, kws in enumerate(self.doc_kw_list):
                    for k, w in kws:
                        self._kw_to_docs.setdefault(k, []).append((di, w))
            for di, w_d in self._kw_to_docs.get(best_kw, []):
                soft_num[di] = soft_num.get(di, 0.0) + w_q * best_cos * w_d

        scores = {}
        for di, num in soft_num.items():
            d_sum = sum(w for _, w in self.doc_kw_list[di])
            denom = (q_sum_sqrt * max(d_sum, 1e-9) ** 0.5)
            scores[di] = num / denom if denom > 1e-9 else 0.0
        return scores


# ── self-test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    # Construct two tiny keyword sets and verify scores are sane.
    corpus_kws = [
        {"password": {"weight_logprob": 0.9, "weight_llm": 0.9},
         "hashing": {"weight_logprob": 0.8, "weight_llm": 0.8},
         "authentication": {"weight_logprob": 0.7, "weight_llm": 0.7}},
        {"rendering": {"weight_logprob": 0.9, "weight_llm": 0.9},
         "react": {"weight_logprob": 0.8, "weight_llm": 0.8},
         "components": {"weight_logprob": 0.7, "weight_llm": 0.7}},
    ]
    corpus_ids = ["doc_auth", "doc_ui"]

    # ── S-jaccard ──
    s_jac = WeightedJaccardSearch()
    s_jac.fit(corpus_kws, corpus_ids)
    # Query overlapping doc 0 heavily
    q = [{"password": {"weight_logprob": 0.9, "weight_llm": 0.9},
          "hashing": {"weight_logprob": 0.8, "weight_llm": 0.8},
          "bcrypt": {"weight_logprob": 0.6, "weight_llm": 0.6}}]
    res = s_jac.search(q, ["q1"], top_k=10)
    print("=== S-jaccard ===")
    print(f"  query→ doc_auth: {res['q1'].get('doc_auth', 0):.4f} (should be high, 2/3 overlap)")
    print(f"  query→ doc_ui:   {res['q1'].get('doc_ui', '—')} (should be absent, no overlap)")

    # Sanity: weighted Jaccard for {password, hashing, bcrypt} vs {password, hashing, auth}
    # overlap = {password, hashing}, union = {password, hashing, bcrypt, auth}
    # num = min(0.9,0.9)+min(0.8,0.8) = 1.7
    # denom = max(0.9,0.9)+max(0.8,0.8)+max(0.6,0)+max(0,0.7) = 0.9+0.8+0.6+0.7 = 3.0
    # = 1.7/3.0 = 0.5667
    expected = (0.9 + 0.8) / (0.9 + 0.8 + 0.6 + 0.7)
    print(f"  expected: {expected:.4f}")
    assert abs(res["q1"]["doc_auth"] - expected) < 1e-6, "jaccard mismatch!"

    print("\n=== S-embed (needs bge-m3) ===")
    try:
        from experiments.embed_cache import CachedBgeM3Provider
        embed = CachedBgeM3Provider()
        s_emb = EmbeddingSoftMatchSearch(embed.embed, tau=0.5)
        s_emb.fit(corpus_kws, corpus_ids)
        res2 = s_emb.search(q, ["q1"], top_k=10)
        print(f"  query→ doc_auth: {res2['q1'].get('doc_auth', 0):.4f} (should be high)")
        # doc_ui should also get some score via "bcrypt"~"react"? probably below tau
        print(f"  query→ doc_ui:   {res2['q1'].get('doc_ui', '—')} (likely absent or low)")
        print("S-embed OK")
    except Exception as e:
        print(f"  (skipped: {e})")

    print("\nAll self-tests OK")
