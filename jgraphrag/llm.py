"""DeepSeekProvider — real concern-inference LLM for M2.7.

Implements the same surface as the Rust `LlmProvider` trait (complete a prompt →
text). Uses the OpenAI SDK pointed at DeepSeek's OpenAI-compatible endpoint.
DeepSeek provides NO embedding model, so this provider is purely the LLM role
(concern inference + NL→DAG); embeddings are entirely bge-m3's job (embed.py).

Adopts the pi-ai 3 disciplines (see spec/providers.rs):
- stream-first: stream() is the primitive; complete() collects it.
- errors-as-values: an in-flight failure returns a Message with stop_reason
  "error" rather than raising (so the caller's retry logic owns backoff).
- pure retry classifier is in Rust (is_retryable_llm_error); this provider
  just surfaces the error message faithfully.

NOTE: requires `pip install openai` and DEEPSEEK_API_KEY in the environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

from .config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, require_deepseek_key

# Lazy import.
_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("openai not installed. Run: pip install openai") from e
        _CLIENT = OpenAI(api_key=require_deepseek_key(), base_url=DEEPSEEK_BASE_URL)
    return _CLIENT


@dataclass
class Message:
    """Mirrors spec::providers::Message (errors-as-values)."""

    content: str
    stop_reason: str  # "stop" | "length" | "tool_use" | "error" | "aborted"
    usage_input: int = 0
    usage_output: int = 0
    error_message: Optional[str] = None
    response_model: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.stop_reason in ("error", "aborted")


class DeepSeekProvider:
    """DeepSeek LLM provider (OpenAI-compatible). Single-shot completions for M2."""

    def __init__(self, model: str = DEEPSEEK_MODEL) -> None:
        self.model = model

    def stream(self, prompt: str, max_tokens: int = 512) -> Iterator:
        """Stream completion events. Yields ('start',) | ('delta', chunk, partial)
        | ('done'|'error', Message). Errors arrive as ('error', Message), not raised."""
        try:
            client = _client()
        except Exception as e:  # pre-flight (auth/config) failure
            yield ("error", Message("", "error", error_message=str(e)))
            return

        yield ("start",)
        partial_parts: list[str] = []
        try:
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                stream=True,
                messages=[{"role": "user", "content": prompt}],
            )
            for chunk in resp:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = delta.content or ""
                if piece:
                    partial_parts.append(piece)
                    yield ("delta", piece, "".join(partial_parts))
        except Exception as e:
            # In-flight failure → errors-as-values (discipline 2).
            yield (
                "error",
                Message(
                    "".join(partial_parts),
                    "error",
                    error_message=f"{type(e).__name__}: {e}",
                ),
            )
            return

        # Map OpenAI finish_reason → our StopReason.
        finish = "stop"  # default for streams that don't surface a finish_reason
        yield (
            "done",
            Message(
                "".join(partial_parts),
                stop_reason=finish,
                response_model=self.model,
            ),
        )

    def complete(self, prompt: str, max_tokens: int = 512) -> Message:
        """Collect a stream into a single Message (discipline 1: complete=stream().result())."""
        msg: Optional[Message] = None
        for ev in self.stream(prompt, max_tokens):
            if ev[0] in ("done", "error"):
                msg = ev[1]
        if msg is None:
            msg = Message("", "error", error_message="stream ended without terminal event")
        return msg


# Standalone smoke test: python -m jgraphrag.llm
if __name__ == "__main__":  # pragma: no cover
    p = DeepSeekProvider()
    m = p.complete("Reply with the single word OK.", max_tokens=8)
    print(f"stop={m.stop_reason} content={m.content!r} err={m.error_message}")
    if m.is_error:
        raise SystemExit(f"FAILED: {m.error_message}")
    print("OK")
