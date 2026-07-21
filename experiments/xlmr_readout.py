"""XLM-RoBERTa-large MLM head readout + vec2vec translation (Phase 9).

§12.3 showed all three readout methods fail on bge-m3 (contrastive-tuned). §12.1
diagnosed the root cause: bge-m3 has NO MLM head, and its hidden state geometry
was reshaped by contrastive learning away from base XLM-RoBERTa. This module
tests the user's proposal: use base XLM-RoBERTa-large (WITH MLM head, NOT
contrastive-tuned) for concept readout, and bridge it to bge-m3 space via vec2vec.

Three readout paths (all on the SAME HDBSCAN clusters for fair comparison):
  1. XLM-R MLM head direct readout (this module — expected to succeed)
  2. bge-m3 reverse-lookup (phase6, known to fail — negative control)
  3. vec2vec: bge-m3 centroid → Procrustes M → XLM-R space → MLM head (test 2)

Cache: XLM-R hidden states persist to .xlmrcache/ (XLM-R inference is slower
than bge-m3, caching is critical for iteration).

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    export HF_HUB_DISABLE_XET=1
    from experiments.xlmr_readout import XLMRReadout
    r = XLMRReadout()
    tokens = r.readout_one("password hashing function")
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_DIR = _REPO / "data" / ".xlmrcache"

XLMR_MODEL_ID = "xlm-roberta-large"
XLMR_DIM = 1024


class XLMRReadout:
    """base XLM-RoBERTa-large with MLM head for concept readout.

    Lazy-loads the model on first use. Supports three pooling strategies
    (CLS / mean / per-position) — the data decides which works, because the
    MLM head was designed for single masked positions, not pooled vectors.
    """

    def __init__(self, model_id: str = XLMR_MODEL_ID,
                 cache_dir: Optional[Path] = None,
                 device: Optional[str] = None) -> None:
        self.model_id = model_id
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"hidden_states_{model_id.replace('/', '_')}.json"
        self._cache: dict[str, list[float]] = {}  # hash → pooled mean vector
        self._model = None
        self._tokenizer = None
        self._lm_head = None
        # Device resolution (deferred — only needed at load time)
        self._device = device
        self._load_cache()

    # ── model loading ────────────────────────────────────────────────────

    def _ensure_model(self):
        """Lazy-load XLM-RoBERTa-large ForMaskedLM (encoder + MLM head)."""
        if self._model is not None:
            return
        from transformers import XLMRobertaForMaskedLM, AutoTokenizer
        import torch

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"  loading {self.model_id} (XLMRobertaForMaskedLM) on {self._device}...", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = XLMRobertaForMaskedLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16 if self._device == "cuda" else torch.float32
        ).to(self._device).eval()
        # The MLM head: lm_head (dense + layer_norm + decoder→250002 vocab)
        self._lm_head = self._model.lm_head
        print(f"  done. vocab={self._tokenizer.vocab_size}, lm_head ready", flush=True)

    # ── hidden states ────────────────────────────────────────────────────

    def get_hidden_states(self, texts: list[str], pooling: str = "mean",
                          max_length: int = 512, batch_size: int = 8) -> np.ndarray:
        """Compute pooled hidden states for a batch of texts.

        pooling:
          "mean"  — attention-masked mean of last_hidden_state (matches bge-m3)
          "cls"   — <s> position (RoBERTa CLS)
          "per_pos" — NO pooling; returns (N, seq_len, dim) for per-position readout.
                      Caller must aggregate logits downstream.

        Returns (N, dim) for mean/cls, or (N, seq_len, dim) for per_pos.
        Cached for mean/cls (per_pos is too large to cache).
        """
        if pooling in ("mean", "cls"):
            # Try cache (only for mean/cls, keyed on text+pooling)
            results = [None] * len(texts)
            miss_idx, miss_texts = [], []
            for i, t in enumerate(texts):
                key = self._hash(t, pooling)
                if key in self._cache:
                    results[i] = np.asarray(self._cache[key], dtype=np.float32)
                else:
                    miss_idx.append(i)
                    miss_texts.append(t)
            if not miss_texts:
                return np.stack(results)
        else:
            miss_idx, miss_texts = list(range(len(texts))), list(texts)
            results = [None] * len(texts)

        # Compute misses
        import torch
        self._ensure_model()

        for batch_start in range(0, len(miss_texts), batch_size):
            batch = miss_texts[batch_start:batch_start + batch_size]
            encoded = self._tokenizer(
                batch, return_tensors="pt", max_length=max_length,
                truncation=True, padding=True,
            )
            input_ids = encoded["input_ids"].to(self._device)
            attention_mask = encoded["attention_mask"].to(self._device)

            with torch.no_grad():
                # Forward through encoder only (XLMRobertaForMaskedLM.roberta).
                # Use the roberta backbone directly to get hidden states without
                # running the lm_head (cheaper, and we call lm_head separately).
                encoder = getattr(self._model, "roberta", None) or self._model.model
                outputs = encoder(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                last_hidden = outputs.last_hidden_state  # (B, seq, 1024) fp16

                if pooling == "mean":
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (last_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif pooling == "cls":
                    pooled = last_hidden.float()[:, 0, :]  # <s> position
                else:  # per_pos — keep full sequence
                    pooled = last_hidden.float()  # (B, seq, 1024)

            pooled_np = pooled.cpu().numpy()
            for j, vec in enumerate(pooled_np):
                gi = miss_idx[batch_start + j]
                results[gi] = vec
                if pooling in ("mean", "cls"):
                    self._cache[self._hash(miss_texts[batch_start + j], pooling)] = vec.tolist()

        if pooling in ("mean", "cls"):
            self._dump_cache()
            return np.stack(results)
        else:
            return np.stack(results)  # (N, seq, dim)

    # ── MLM head readout ─────────────────────────────────────────────────

    def readout_via_mlm_head(self, hidden_states: np.ndarray, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """Pass hidden states through the MLM head, decode top-k tokens per vector.

        hidden_states: (N, 1024) — pooled vectors (mean/cls) OR (N, 1024) centroids.
        Returns list of N × [(token_str, logit), ...] (top_k per input).
        """
        import torch
        self._ensure_model()

        # Match the lm_head's dtype (fp16 on GPU) to avoid dtype-mismatch crash.
        head_dtype = next(self._lm_head.parameters()).dtype
        h = torch.as_tensor(hidden_states, dtype=head_dtype, device=self._device)
        # lm_head: dense → activation → layer_norm → decoder
        with torch.no_grad():
            logits = self._lm_head(h)  # (N, 250002)
        # Top-k per row
        top_v, top_i = logits.topk(top_k, dim=-1)
        top_v_np = top_v.cpu().numpy()
        top_i_np = top_i.cpu().numpy()

        results = []
        for n in range(top_i_np.shape[0]):
            tokens = []
            for j in range(top_k):
                tok_id = int(top_i_np[n, j])
                tok_str = self._tokenizer.convert_ids_to_tokens(tok_id)
                tokens.append((tok_str, float(top_v_np[n, j])))
            results.append(tokens)
        return results

    def readout_centroid(self, centroid: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        """Convenience: readout for a single centroid vector."""
        return self.readout_via_mlm_head(centroid.reshape(1, -1), top_k)[0]

    # ── vec2vec (Procrustes) ─────────────────────────────────────────────

    def learn_vec2vec(self, h_bge: np.ndarray, h_xlmr: np.ndarray) -> np.ndarray:
        """Learn linear map M: h_bge → h_xlmr via orthogonal Procrustes (SVD closed form).

        h_bge: (N, 1024) bge-m3 vectors. h_xlmr: (N, 1024) XLM-R pooled vectors.
        Returns M: (1024, 1024) such that M @ h_bge ≈ h_xlmr.

        Orthogonal Procrustes: M* = U V^T where U Σ V^T = SVD(h_bge^T @ h_xlmr).
        Constrains M to rotations (no scaling/shear) — most stable cross-model map.
        """
        # Center both (translation invariance)
        bge_c = h_bge - h_bge.mean(0, keepdims=True)
        xlmr_c = h_xlmr - h_xlmr.mean(0, keepdims=True)
        # SVD of cross-covariance
        M_approx = bge_c.T @ xlmr_c  # (1024, 1024)
        U, S, Vt = np.linalg.svd(M_approx, full_matrices=False)
        M = U @ Vt  # (1024, 1024) orthogonal
        return M

    def translate(self, h_bge: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply vec2vec map. h_bge: (N,1024) or (1024,). Returns same shape."""
        return (M @ h_bge.T).T if h_bge.ndim == 2 else M @ h_bge

    def translation_residual(self, h_bge: np.ndarray, h_xlmr: np.ndarray, M: np.ndarray) -> float:
        """Mean relative residual ‖M·h_bge − h_xlmr‖ / ‖h_xlmr‖."""
        pred = self.translate(h_bge, M)
        num = np.linalg.norm(pred - h_xlmr, axis=-1)
        den = np.linalg.norm(h_xlmr, axis=-1).clip(min=1e-8)
        return float(num.mean() / den.mean())

    # ── cache I/O (mirrors embed_cache.py) ───────────────────────────────

    def _hash(self, text: str, pooling: str) -> str:
        return hashlib.sha256(
            f"{text}|{self.model_id}|{pooling}".encode("utf-8")
        ).hexdigest()

    def _load_cache(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            data = json.loads(self._cache_file.read_text(encoding="utf-8"))
            # Backward-compat: vectors stored as lists
            self._cache = {k: v for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            self._cache = {}

    def _dump_cache(self) -> None:
        if not self._cache:
            return
        tmp = self._cache_file.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._cache_file)

    def flush(self) -> None:
        self._dump_cache()


# ── smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import os, sys

    r = XLMRReadout()
    print(f"cache: {len(r._cache)} entries @ {r._cache_file}")

    samples = [
        "export async function authenticateUser(email, password) { return bcrypt.compare(password, hash); }",
        "const Component = () => { return <div className='header'>Hello</div>; }",
        "SELECT users.id, orders.total FROM users JOIN orders ON users.id = orders.user_id;",
    ]
    print(f"\n=== {len(samples)} sample texts, mean pooling ===")
    h = r.get_hidden_states(samples, pooling="mean")
    print(f"  hidden states shape: {h.shape}")

    print(f"\n=== MLM head readout (top-8 tokens per text) ===")
    readouts = r.readout_via_mlm_head(h, top_k=8)
    for i, (text, toks) in enumerate(zip(samples, readouts)):
        print(f"\n  [{i}] text: {text[:70]}...")
        for tok, logit in toks:
            print(f"      {tok:<20s} logit={logit:+.3f}")

    print("\n=== CLS pooling comparison (first text) ===")
    h_cls = r.get_hidden_states([samples[0]], pooling="cls")
    toks_cls = r.readout_via_mlm_head(h_cls, top_k=5)[0]
    for tok, logit in toks_cls:
        print(f"      {tok:<20s} logit={logit:+.3f}")

    print("\nOK" if readouts else "FAIL: no readout")
