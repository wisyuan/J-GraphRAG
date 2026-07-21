"""CachedKeywordExtractor — LLM concept-keyword extraction with logprob-derived weights.

Phase 8 helper: for each text (doc or query), ask DeepSeek to extract 10-15
concept keywords, then derive each keyword's weight from the model's own logprobs
(geometric mean of token probabilities) rather than the LLM's self-reported
numbers. Self-reported weights are poorly calibrated (a known LLM weakness);
logprobs are the model's true internal confidence.

Two weight sources are kept (both persisted):
  - "weight_llm": the number the LLM wrote in the JSON (self-report)
  - "weight_logprob": exp(mean(token_logprobs)) for that keyword's tokens (true)

Downstream baselines pick whichever they want; the comparison is itself a finding.

Cache design mirrors embed_cache.py: sha256 key, atomic tmp+replace, JSON on disk.
This is the first concurrent LLM-calling code in the repo — ThreadPoolExecutor
wraps the synchronous provider at the batch level.

Usage:
    cd crates/lincle/python
    source .venv/bin/activate; set -a; . .env; set +a
    from experiments.keyword_cache import CachedKeywordExtractor
    ext = CachedKeywordExtractor()
    kws = ext.extract_keywords("some text")  # {"keyword": {"weight_llm":.., "weight_logprob":..}}
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from jgraphrag.llm import DeepSeekProvider

# Repo root = parents[1] from this file (experiments/ → python/ → lincle/ → crates/ → repo/)
_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_DIR = _REPO / "data" / ".kwcache"

# Bumped whenever the prompt changes — invalidates all cached entries.
PROMPT_VERSION = "v2"  # v2: split doc/query prompts to fix intent-vs-content mismatch

_MAX_INPUT_CHARS = 1500  # match run_m2_ab.generate_doc2query
_MAX_TOKENS = 512
_MAX_KEYWORDS = 15
_MIN_KEYWORDS = 8
_LOGPROB_TOP = 3  # top_logprobs to request (we only use the chosen token's lp)


def _build_prompt(text: str, mode: str = "doc") -> str:
    """Concept-extraction prompt, split by text type.

    v1 bug: one prompt for both docs and queries → queries extracted INTENT
    words ("definition location", "code navigation") while docs extracted
    CONTENT words ("agent loop", "event emission"). Zero overlap → 0.06 nDCG.

    v2 fix: both modes now extract ENTITIES/CONCEPTS the text *mentions*,
    not what it "is about" or "is asking for". A query "Where is AgentEventSink
    defined?" should yield ["AgentEventSink", "definition", ...] — the entity
    name, matching what the target doc also mentions — NOT "definition location".
    Both sides land in the same vocabulary space (entity/concept names).
    """
    if mode == "query":
        instruction = (
            f"Extract {_MIN_KEYWORDS}-{_MAX_KEYWORDS} keywords from this search query. "
            "Focus on the SPECIFIC ENTITIES, CONCEPTS, and TECHNICAL TERMS the query "
            "mentions or asks about (e.g. function names, type names, domain concepts, "
            "technologies). These are the terms a relevant document would also contain.\n\n"
            "Do NOT extract meta-words about the query itself (e.g. 'definition location', "
            "'code search', 'where is') — extract WHAT the query is about, not THAT it is a query.\n\n"
            f"Query:\n{text[:_MAX_INPUT_CHARS]}\n\n"
        )
    else:  # doc
        instruction = (
            f"Extract {_MIN_KEYWORDS}-{_MAX_KEYWORDS} keywords from this text. "
            "Focus on the SPECIFIC ENTITIES, CONCEPTS, and TECHNICAL TERMS the text "
            "is about (e.g. function names, type names, domain concepts, technologies, "
            "APIs). These are the terms a query about this topic would also mention.\n\n"
            "Do NOT extract generic words ('function', 'code', 'text') — extract the "
            "SPECIFIC nouns and technical terms.\n\n"
            f"Text:\n{text[:_MAX_INPUT_CHARS]}\n\n"
        )
    return (
        instruction
        + 'Each keyword gets a weight 0.0-1.0 reflecting how central it is. '
        'Respond with ONLY a JSON object, no prose, no markdown fences: '
        '{"keyword one": 0.9, "keyword two": 0.7}'
    )


def _parse_keywords(content: str, logprob_items: Optional[list] = None) -> list[dict]:
    """Parse the LLM JSON response into keyword records.

    Returns a list of {"keyword": str, "weight_llm": float, "weight_logprob": float}.
    logprob_items is the list of ChoiceLogprob objects (resp.choices[0].logprobs.content);
    when present we map token positions to keywords to derive weight_logprob.

    Mapping strategy: the JSON string is reconstructed from the token stream, so
    we can recover which token positions belong to each keyword by re-tokenizing
    the JSON text against the recorded token strings. We use json tokenizer-level
    alignment: walk the tokens, find the quoted-key segments.
    """
    # ── Step 1: clean the content (strip markdown fences if present) ──
    text = content.strip()
    fence = re.match(r"^```(?:json)?\s*", text, re.IGNORECASE)
    if fence:
        text = re.sub(r"\s*```\s*$", "", text[fence.end():])

    # ── Step 2: parse the JSON object ──
    raw_kws: dict[str, float] = {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            raw_kws = {str(k): float(v) for k, v in obj.items()
                       if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0}
    except (json.JSONDecodeError, ValueError):
        # Fallback: extract "key": number pairs via regex.
        for k, v in re.findall(r'"([^"]{2,60})"\s*:\s*([0-9]*\.?[0-9]+)', text):
            try:
                w = float(v)
                if 0.0 <= w <= 1.0:
                    raw_kws[k] = w
            except ValueError:
                pass

    if not raw_kws:
        return []

    # ── Step 3: derive weight_logprob by mapping tokens to keywords ──
    lp_weight_map = _map_logprobs_to_keywords(content, logprob_items)

    records = []
    for kw, w_llm in raw_kws.items():
        w_lp = lp_weight_map.get(kw)
        records.append({
            "keyword": kw,
            "weight_llm": w_llm,
            "weight_logprob": w_lp if w_lp is not None else w_llm,  # fall back to self-report
        })
    return records


def _map_logprobs_to_keywords(content: str, logprob_items: Optional[list]) -> dict[str, float]:
    """Map each keyword in the JSON to a logprob-derived weight.

    Strategy: reconstruct the content from tokens, tracking absolute char spans
    of each token. Then find each quoted JSON key ("...") in the content, and
    for each, collect the logprobs of tokens whose char span overlaps the key's
    interior. Weight = exp(mean(those token logprobs)) — geometric mean prob.
    """
    if not logprob_items:
        return {}

    # Build (token, char_start, char_end, logprob) by concatenating tokens.
    spans = []  # list of (token_str, start, end, logprob)
    pos = 0
    for item in logprob_items:
        tok = item.token if hasattr(item, "token") else item.get("token", "")
        if not tok:
            continue
        lp = item.logprob if hasattr(item, "logprob") else item.get("logprob", 0.0)
        spans.append((tok, pos, pos + len(tok), float(lp)))
        pos += len(tok)

    if pos == 0 or pos != len(content):
        # Token reconstruction drifted from content — alignment unreliable.
        # This happens occasionally with special tokens; bail to self-report.
        return {}

    # Find quoted JSON keys: scan content for "..." that precede a colon.
    # We look for quoted strings that are followed (after optional whitespace) by ':'.
    result = {}
    for m in re.finditer(r'"((?:[^"\\]|\\.){1,60})"\s*:', content):
        key_text = _decode_json_string(m.group(1))
        key_start = m.start(1)
        key_end = m.end(1)
        # Collect logprobs of tokens overlapping [key_start, key_end).
        lps = [lp for (tok, s, e, lp) in spans if e > key_start and s < key_end]
        if not lps:
            continue
        # Geometric mean = exp(mean(logprobs)). Clamp to [0,1].
        geo_mean = math.exp(sum(lps) / len(lps))
        result[key_text] = min(1.0, max(0.0, geo_mean))
    return result


def _decode_json_string(s: str) -> str:
    """Decode escape sequences in a quoted JSON string fragment."""
    try:
        return json.loads('"' + s + '"')
    except (json.JSONDecodeError, ValueError):
        return s


class CachedKeywordExtractor:
    """DeepSeekProvider wrapper with disk caching + logprob weights.

    Composition over inheritance (same pattern as CachedBgeM3Provider). The
    underlying provider is lazy-imported so importing this module without the
    openai package configured won't crash until first use.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        provider: Optional[DeepSeekProvider] = None,
        model: Optional[str] = None,
        max_workers: int = 8,
        max_retries: int = 3,
    ) -> None:
        self._provider = provider or DeepSeekProvider(model=model) if model else DeepSeekProvider()
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self._cache_dir / f"keywords_{self._provider.model}_{PROMPT_VERSION}.json"
        # text_hash → list[record] (records = dicts from _parse_keywords)
        self._cache: dict[str, list[dict]] = {}
        self._dirty = False
        self._max_workers = max_workers
        self._max_retries = max_retries
        self._load()

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def size(self) -> int:
        return len(self._cache)

    # ── single-text API ──────────────────────────────────────────────────

    def extract_keywords(self, text: str, mode: str = "doc") -> dict[str, dict]:
        """Extract keywords for one text. Returns {keyword: {"weight_llm", "weight_logprob"}}.

        mode: "doc" or "query" — uses different prompts so both sides land in the
        same vocabulary space (entity/concept names). See _build_prompt.

        Cache hit skips the LLM entirely. Miss → call DeepSeek with logprobs,
        parse, persist.
        """
        key = _hash_key(text, self._provider.model, mode)
        if key in self._cache:
            return {r["keyword"]: r for r in self._cache[key]}

        records = self._call_with_retry(text, mode)
        self._cache[key] = records
        self._dirty = True
        self._dump()
        return {r["keyword"]: r for r in records}

    # ── batch API (concurrent) ───────────────────────────────────────────

    def extract_keywords_batch(
        self, texts: list[str], desc: str = "", mode: str = "doc"
    ) -> list[dict[str, dict]]:
        """Concurrent batch extraction. Returns one dict per input text, in order.

        mode: "doc" or "query" (all texts in a batch use the same mode —
        a batch is either all-docs or all-queries).

        Cache hits are resolved first (no API call); misses are dispatched to a
        thread pool. This is the first concurrent LLM caller in the repo.
        """
        results: list[Optional[dict[str, dict]]] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []

        for i, t in enumerate(texts):
            key = _hash_key(t, self._provider.model, mode)
            if key in self._cache:
                results[i] = {r["keyword"]: r for r in self._cache[key]}
            else:
                miss_idx.append(i)
                miss_texts.append(t)

        if not miss_texts:
            return results  # type: ignore[return-value]

        n_miss = len(miss_texts)
        print(f"  [{desc}] {n_miss} cache misses / {len(texts)} texts → "
              f"concurrent API ({self._max_workers} workers)", flush=True)

        done = 0
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            # Submit all misses. as_completed for progress; map results back by index.
            future_to_idx = {
                pool.submit(self._call_with_retry, t, mode): i
                for i, t in zip(miss_idx, miss_texts)
            }
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    records = fut.result()
                except Exception as e:
                    warnings.warn(f"keyword extraction failed for text idx {i}: {e}")
                    records = []
                t = texts[i]
                self._cache[_hash_key(t, self._provider.model, mode)] = records
                self._dirty = True
                results[i] = {r["keyword"]: r for r in records}
                done += 1
                if done % 100 == 0 or done == n_miss:
                    print(f"    [{desc}] {done}/{n_miss}", flush=True)
                    self._dump()  # periodic flush (crash-safe progress)

        self._dump()
        return results  # type: ignore[return-value]

    # ── internal ─────────────────────────────────────────────────────────

    def _call_with_retry(self, text: str, mode: str = "doc") -> list[dict]:
        """Call DeepSeek with logprobs, retry on transient errors with backoff."""
        prompt = _build_prompt(text, mode)
        last_err = None
        for attempt in range(self._max_retries):
            # NOTE: DeepSeekProvider.complete() does not support logprobs/temperature
            # (hardcoded stream, single-user-message). So we call the client directly
            # here, bypassing the provider, to get logprobs. This is deliberate —
            # we don't want to modify the shared provider for one experiment.
            try:
                records = self._call_raw(prompt)
                if records:  # got at least one keyword → success
                    return records
                last_err = "empty keyword parse"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
            # exponential backoff: 1s, 2s, 4s
            if attempt < self._max_retries - 1:
                time.sleep(2 ** attempt)
        warnings.warn(f"keyword extraction failed after {self._max_retries} retries: {last_err}")
        return []

    def _call_raw(self, prompt: str) -> list[dict]:
        """Direct OpenAI client call with logprobs enabled. Bypasses DeepSeekProvider
        because that provider doesn't expose logprobs (see _call_with_retry note)."""
        from jgraphrag.llm import _client  # reuse the module-level singleton client

        client = _client()
        resp = client.chat.completions.create(
            model=self._provider.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
            logprobs=True,
            top_logprobs=_LOGPROB_TOP,
            stream=False,
        )
        choice = resp.choices[0]
        content = choice.message.content or ""
        lp_items = choice.logprobs.content if choice.logprobs else None
        return _parse_keywords(content, lp_items)

    # ── cache I/O (mirrors embed_cache.py) ───────────────────────────────

    def _load(self) -> None:
        if not self._cache_file.exists():
            return
        try:
            self._cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._cache = {}  # corrupt → start fresh

    def _dump(self) -> None:
        if not self._cache:
            return
        tmp = self._cache_file.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._cache_file)  # atomic on POSIX
        self._dirty = False

    def flush(self) -> None:
        if self._dirty:
            self._dump()


def _hash_key(text: str, model: str, mode: str = "doc") -> str:
    return hashlib.sha256(
        f"{text}|{model}|{PROMPT_VERSION}|{mode}".encode("utf-8")
    ).hexdigest()


# ── smoke test ────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import os

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: set -a; . .env; set +a  first", file=sys.stderr)
        raise SystemExit(1)

    ext = CachedKeywordExtractor()
    print(f"cache: {ext.size} entries @ {ext._cache_file}")

    sample = (
        "export async function authenticateUser(email, password) {\n"
        "  const user = await db.findUserByEmail(email);\n"
        "  const valid = await bcrypt.compare(password, user.passwordHash);\n"
        '  if (!valid) throw new AuthError("invalid credentials");\n'
        '  const token = jwt.sign({ userId: user.id }, SECRET, { expiresIn: "7d" });\n'
        "  return { ...user, token };\n"
        "}"
    )

    print("\n=== extract doc mode (first call — hits API) ===")
    kws = ext.extract_keywords(sample, mode="doc")
    print(f"  {len(kws)} keywords, cache now {ext.size}")
    print(f"  {'keyword':<28s} {'w_llm':>6s} {'w_logprob':>10s}")
    for kw, rec in sorted(kws.items(), key=lambda x: -x[1]["weight_logprob"]):
        print(f"  {kw:<28s} {rec['weight_llm']:>6.3f} {rec['weight_logprob']:>10.4f}")

    print("\n=== second call (should be cache hit) ===")
    kws2 = ext.extract_keywords(sample, mode="doc")
    assert kws2 == kws, "cache inconsistency!"
    print(f"  cache hit verified, size still {ext.size}")

    print("\nOK")
