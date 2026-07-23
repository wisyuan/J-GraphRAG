"""Default LensProvider: Qwen2.5-7B-Instruct (4bit) + Jacobian Lens.

Ported from:
- experiments/phase10_jlens_stage1.py — _model_dir_complete / load_model /
  load_lens logic, adapted so that paths come from jgraphrag.config (no
  hard-coded /tmp paths, no module-level disk probing, no EXP.mkdir).

GPU constraints (8GB VRAM iron rules, inherited from the experiment env):
  - A process can load the 4bit model only ONCE — CUDA memory is not
    released, and a second load fails with "modules dispatched on CPU or
    disk". Run multiple domains in separate processes.
  - bge-m3 and Qwen must NOT co-reside on the GPU (they OOM together). Run
    the EmbedProvider on CPU (LINCLE_BGE_M3_DEVICE=cpu) or after this
    provider's process has finished.

The model is loaded lazily: constructing the provider touches nothing; the
first extract_concepts / extract_relation call triggers loading.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..config import JLENS_LENS_PATH, QWEN_MODEL_ID, QWEN_MODEL_PATH
from ..extract.concepts import extract_concepts_full_pipeline
from ..extract.filter import (
    _get_wordnet_nouns,
    build_corpus_term_freq,
    build_corpus_word_set,
)
from ..extract.relations import extract_relation as _read_relation
from ..extract.roles import get_wu_first_fragment_vec, pass2_role_expansion
from .base import ChunkExtraction

LENS_REPO = "neuronpedia/jacobian-lens"


def _model_dir_complete(d: str) -> bool:
    """A model dir is complete if config.json + index exist AND the total size
    of all safetensors shards matches the index's metadata.total_size (within
    1% to tolerate rounding). This catches partial downloads."""
    p = Path(d)
    if not (p / "config.json").exists():
        return False
    idx = p / "model.safetensors.index.json"
    if not idx.exists():
        return False
    meta = json.loads(idx.read_text())
    expected_total = meta.get("metadata", {}).get("total_size")
    if not expected_total:
        return False
    actual_total = sum(s.stat().st_size for s in p.glob("*.safetensors"))
    # require ≥99% of expected total (the index total_size is sum of raw bytes;
    # safetensors files include a small header overhead, so be generous)
    return actual_total >= expected_total * 0.99


def _lens_hf_filename(model_id: str) -> str:
    """HF repo filename of the pre-fitted lens for the given model id.

    Mirrors the phase10_jlens_stage1 pattern:
    ``{name}/jlens/Salesforce-wikitext/{ModelClass}_jacobian_lens.pt``.
    """
    model_tail = model_id.split("/")[-1]
    name = "qwen2.5-7b-it" if model_tail == "Qwen2.5-7B-Instruct" \
        else model_tail.lower()
    return f"{name}/jlens/Salesforce-wikitext/{model_tail}_jacobian_lens.pt"


def load_model(model_src: str, use_4bit: bool = True):
    """Load model in 4-bit NF4 (for >4B models) or fp16/bf16 (for ≤2B).

    4-bit fits 8GB VRAM: ~2.5GB weights + ~1GB activations + ~0.5GB J matrices
    + ~1GB overhead. lm_head stays fp16 (not quantized) so unembed is clean.

    J-Lens compatibility: jlens.HFLensModel.forward() hooks each layer's output
    residual stream. 4-bit only changes weight *storage* (dequantized on-the-fly
    during matmul); the residual stream output is still fp16/bf32. The J_l
    transport and lm_head unembed are unaffected.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  loading {model_src} ({'4-bit NF4' if use_4bit else 'bf16'})...", flush=True)
    # If model_src is a local directory, force local_files_only to avoid
    # network validation (transformers 5.x re-checks even for cached files)
    is_local = os.path.isdir(model_src)
    tok_kwargs = dict(local_files_only=True) if is_local else {}
    tokenizer = AutoTokenizer.from_pretrained(model_src, **tok_kwargs)
    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if is_local:
        kwargs["local_files_only"] = True
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_src, **kwargs)
    model.eval()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"  loaded. VRAM: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB",
          flush=True)
    return model, tokenizer


def load_lens(lens_path: str | None = None, model_id: str = QWEN_MODEL_ID):
    """Load the pre-fitted Jacobian lens.

    Resolution order: explicit/local lens_path → download from the HF
    neuronpedia/jacobian-lens repo.
    """
    import jlens

    if lens_path:
        p = Path(lens_path)
        if p.is_dir():
            # Accept a lens *directory* (e.g. /tmp/jlens-qwen25-7b-it) and pick
            # the .pt checkpoint inside it.
            pts = sorted(p.glob("*_jacobian_lens.pt")) or sorted(p.glob("*.pt"))
            p = pts[0] if pts else p / "__missing__"
        if p.exists():
            print(f"  loading lens from {p}", flush=True)
            return jlens.JacobianLens.load(str(p))
    print(f"  downloading lens from {LENS_REPO}", flush=True)
    return jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=_lens_hf_filename(model_id))


class QwenJlensLensProvider:
    """LensProvider backed by Qwen2.5-7B-Instruct 4bit + Jacobian Lens.

    Construction loads nothing. The first extract call resolves the model
    source (local weights dir from config if complete, else the HF hub id),
    loads the 4bit model + lens, and wraps them with jlens.from_hf.
    """

    def __init__(
        self,
        model_path: str = QWEN_MODEL_PATH,
        model_id: str = QWEN_MODEL_ID,
        lens_path: str = JLENS_LENS_PATH,
        use_4bit: bool = True,
    ):
        self.model_path = model_path
        self.model_id = model_id
        self.lens_path = lens_path
        self.use_4bit = use_4bit
        self._lens = None
        self._lens_model = None
        self._model = None
        self._tokenizer = None

    # ── lazy loading ────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._lens_model is not None:
            return
        import jlens

        self._lens = load_lens(self.lens_path, model_id=self.model_id)
        model_src = (self.model_path
                     if _model_dir_complete(self.model_path)
                     else self.model_id)
        self._model, self._tokenizer = load_model(
            model_src, use_4bit=self.use_4bit)
        self._lens_model = jlens.from_hf(
            self._model, self._tokenizer, force_bos=False)

    # ── LensProvider protocol ───────────────────────────────────────────

    def extract_concepts(self, chunks: list[str]) -> list[ChunkExtraction]:
        """Two-pass extraction per chunk: Pass 1 concepts (phase25 full
        pipeline) + Pass 2 role expansion / ws vectors (phase35/39).

        Each returned ChunkExtraction also carries ``wu_vecs``
        ({concept: W_U first-fragment vector, L2-normalized}).
        """
        self._ensure_loaded()
        lens, lens_model, tokenizer = self._lens, self._lens_model, self._tokenizer
        all_layers = lens.source_layers
        last_layer = all_layers[-1]

        # Corpus resources for BM25 completion + corpus verification
        corpus_words = build_corpus_word_set(chunks)
        corpus_freq = build_corpus_term_freq(chunks)
        wn_nouns = _get_wordnet_nouns()

        # W_U (lm_head) rows for first-fragment vectors: [vocab, 3584]
        lm_head_wu = (self._model.get_output_embeddings().weight
                      .detach().float().cpu().numpy())

        results: list[ChunkExtraction] = []
        for chunk_text in chunks:
            # Pass 1: Phase 25 full pipeline (identical to phase31 cache)
            p1 = extract_concepts_full_pipeline(
                lens, lens_model, tokenizer,
                chunk_text, all_layers,
                corpus_words, corpus_freq, wn_nouns)
            concepts = p1["concepts"]

            # Pass 2: role expansion (1 extra forward)
            roles, ws_vecs = pass2_role_expansion(
                lens, lens_model, tokenizer, chunk_text,
                concepts, corpus_words, last_layer)

            # wu_vec: concept first BPE fragment's W_U row (L2-normalized)
            wu_vecs: dict[str, np.ndarray] = {}
            for concept in {c.lower() for c in concepts}:
                wu = get_wu_first_fragment_vec(concept, lm_head_wu, tokenizer)
                if wu is not None:
                    wu_vecs[concept] = wu
            results.append(ChunkExtraction(
                concepts=concepts, roles=roles, ws_vecs=ws_vecs, wu_vecs=wu_vecs))
        return results

    def extract_relation(self, context: str, concept_a: str,
                         concept_b: str) -> tuple[str, float]:
        """Read the relation word between two concepts in a context.

        Returns ``(relation_word, probability)``; ``("", 0.0)`` means the
        readout produced no usable word (the caller should skip the edge,
        as phase40 does).
        """
        self._ensure_loaded()
        layer = self._lens.source_layers[-1]
        rel_word, rel_prob = _read_relation(
            self._lens, self._lens_model, self._tokenizer,
            concept_a, concept_b, context, layer)
        if rel_word is None:
            return "", 0.0
        return rel_word.lower(), rel_prob
