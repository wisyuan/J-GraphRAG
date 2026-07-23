"""Tests for the pure-CPU functions of jgraphrag.extract.relations."""
from __future__ import annotations

from jgraphrag.extract.relations import build_relation_prompt, find_pair_context


class _ChatTemplateTokenizer:
    """Mock tokenizer with apply_chat_template (records the call)."""

    RESULT = "<|im_start|>user ... <|im_end|>"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.RESULT


class _BrokenTemplateTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        raise RuntimeError("template missing")


class TestBuildRelationPrompt:
    A, B = "insulin", "glucose"
    DOC = "Insulin regulates blood glucose levels. " * 30

    def test_uses_chat_template_when_available(self):
        tok = _ChatTemplateTokenizer()
        out = build_relation_prompt(self.DOC, self.A, self.B, tok)
        assert out == _ChatTemplateTokenizer.RESULT
        assert len(tok.calls) == 1
        messages, kwargs = tok.calls[0]
        assert messages[0]["role"] == "user"
        assert self.A in messages[0]["content"]
        assert self.B in messages[0]["content"]
        # assistant prefill anchors the readout position
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == (
            f"The relationship between {self.A} and {self.B} is")
        assert kwargs["tokenize"] is False
        assert kwargs["continue_final_message"] is True
        assert kwargs["add_generation_prompt"] is False

    def test_falls_back_when_template_raises(self):
        out = build_relation_prompt(self.DOC, self.A, self.B,
                                    _BrokenTemplateTokenizer())
        self._assert_fallback(out)

    def test_falls_back_without_template_method(self):
        out = build_relation_prompt(self.DOC, self.A, self.B, object())
        self._assert_fallback(out)

    def _assert_fallback(self, out: str) -> None:
        prefill = f"The relationship between {self.A} and {self.B} is"
        assert out.endswith(prefill)
        assert self.A in out and self.B in out
        # document is truncated into the user message
        assert self.DOC[:100].strip()[:50] in out

    def test_doc_truncated_to_600_chars(self):
        tok = _ChatTemplateTokenizer()
        build_relation_prompt(self.DOC, self.A, self.B, tok)
        user_content = tok.calls[0][0][0]["content"]
        assert len(self.DOC) > 600
        assert self.DOC[:600] in user_content
        assert self.DOC[:601] not in user_content


class TestFindPairContext:
    TEXTS = {
        "c0": "a much longer chunk containing only alpha " * 5,
        "c1": "alpha beta together",
        "c2": "beta only, medium length text",
    }

    def test_shared_chunk_preferred(self):
        chunks = {"a": ["c0", "c1"], "b": ["c1", "c2"]}
        text, strategy = find_pair_context("a", "b", chunks, self.TEXTS)
        assert strategy == "shared"
        assert text == self.TEXTS["c1"]

    def test_shortest_shared_chunk_wins(self):
        texts = dict(self.TEXTS, c3="tiny a b")
        chunks = {"a": ["c1", "c3"], "b": ["c1", "c3"]}
        text, strategy = find_pair_context("a", "b", chunks, texts)
        assert strategy == "shared"
        assert text == "tiny a b"

    def test_concat_when_no_shared_chunk(self):
        chunks = {"a": ["c0"], "b": ["c2"]}
        text, strategy = find_pair_context("a", "b", chunks, self.TEXTS)
        assert strategy == "concat"
        assert text == f"{self.TEXTS['c0']}\n{self.TEXTS['c2']}"

    def test_concat_uses_shortest_of_each(self):
        chunks = {"a": ["c0", "c1"], "b": ["c2"]}
        text, strategy = find_pair_context("a", "b", chunks, self.TEXTS)
        assert strategy == "concat"
        assert text == f"{self.TEXTS['c1']}\n{self.TEXTS['c2']}"

    def test_missing_concept(self):
        text, strategy = find_pair_context("a", "ghost", {"a": ["c0"]},
                                           self.TEXTS)
        assert text is None and strategy == "missing"

    def test_chunk_without_text_is_missing(self):
        chunks = {"a": ["c9"], "b": ["c2"]}  # c9 has no text
        text, strategy = find_pair_context("a", "b", chunks, self.TEXTS)
        assert text is None and strategy == "missing"
