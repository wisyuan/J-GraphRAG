"""Tests for jgraphrag.extract.entities — text-side rule-based detection."""
from __future__ import annotations

from jgraphrag.extract.entities import (
    TERM_MIN_FREQ,
    _tokenize_sentences,
    _word_stats,
    detect_entities,
    extract_cap_spans,
    extract_term_spans,
)


def _spans_from_text(text: str) -> list[str]:
    sentences = _tokenize_sentences(text)
    n_lower, n_cap_mid = _word_stats(sentences)
    return extract_cap_spans(sentences, n_lower, n_cap_mid)


class TestExtractCapSpans:
    def test_multiword_span_with_connector(self):
        spans = _spans_from_text(
            "Arthur gave Excalibur to Princess Frederica of Hanover.")
        assert "Princess Frederica of Hanover" in spans
        assert "Arthur" in spans
        assert "Excalibur" in spans

    def test_sentence_initial_false_positive_stripped(self):
        # "the" is frequent lowercase and never capitalized mid-sentence →
        # sentence-initial "The" is stripped from the run; "King" stays.
        text = ("The King arrived. the queen left early. "
                "the court was silent. King Charles spoke.")
        spans = _spans_from_text(text)
        assert "King" in spans
        assert "King Charles" in spans
        assert not any(s.startswith("The") for s in spans)

    def test_possessive_stripped(self):
        spans = _spans_from_text("Sterne's house was sold.")
        assert "Sterne" in spans
        assert not any("'" in s for s in spans)

    def test_dangling_trailing_connector_dropped(self):
        spans = _spans_from_text("He met Baron von.")
        assert "Baron" in spans
        assert "Baron von" not in spans

    def test_blocklisted_single_word_rejected(self):
        assert "However" not in _spans_from_text("However she left.")

    def test_roman_numeral_rejected(self):
        assert "XIV" not in _spans_from_text("XIV ruled the land.")

    def test_contraction_not_cap_word(self):
        spans = _spans_from_text("Don't touch Excalibur.")
        assert not any(s.startswith("Don") for s in spans)
        assert "Excalibur" in spans

    def test_span_capped_at_four_content_words(self):
        spans = _spans_from_text("Alpha Bravo Charlie Delta Echo left.")
        assert "Alpha Bravo Charlie Delta" in spans
        assert "Alpha Bravo Charlie Delta Echo" not in spans


class TestExtractTermSpans:
    def test_frequency_counting(self):
        sentences = _tokenize_sentences(
            "insulin therapy lowers glucose. insulin therapy helps.")
        cand, _cap = extract_term_spans(sentences)
        assert cand["insulin therapy"] == 2
        assert cand["insulin therapy"] >= TERM_MIN_FREQ

    def test_tail_verb_blocked(self):
        sentences = _tokenize_sentences(
            "insulin therapy works. insulin therapy works.")
        cand, _cap = extract_term_spans(sentences)
        assert "therapy works" not in cand

    def test_stopword_grams_skipped(self):
        sentences = _tokenize_sentences(
            "the insulin therapy and the insulin therapy.")
        cand, _cap = extract_term_spans(sentences)
        assert not any("the" in g.split() for g in cand)

    def test_capitalized_gram_rescued(self):
        # heading-style capitalization marks the n-gram even at freq 1
        sentences = _tokenize_sentences("Immune suppression matters here.")
        cand, cap_grams = extract_term_spans(sentences)
        assert "immune suppression" in cap_grams
        assert cand["immune suppression"] == 1


class TestDetectEntities:
    NOVEL_CORPUS = {
        "c0": "Arthur met Dozmare. Arthur left quietly.",
        "c1": "Dozmare smiled at Arthur.",
    }

    def test_end_to_end_capitalized(self):
        out = detect_entities("novel", self.NOVEL_CORPUS, set())
        assert out["domain"] == "novel"
        assert out["n_chunks"] == 2
        assert "dozmare" in out["entity_chunks"]
        assert set(out["entity_chunks"]["dozmare"]) == {"c0", "c1"}
        assert "arthur" in out["entity_chunks"]
        assert out["entity_frequency"]["dozmare"] >= 2
        # display surface preserves original capitalization
        assert out["entity_display"]["dozmare"] == "Dozmare"
        assert out["entity_type"]["dozmare"] == "capitalized"

    def test_dedup_against_concept_vocab(self):
        out = detect_entities("novel", self.NOVEL_CORPUS, {"Arthur"})
        assert "arthur" not in out["entity_chunks"]
        assert "dozmare" in out["entity_chunks"]

    def test_dedup_uses_stem_normalization(self):
        # entity "arthur" vs concept "arthurs" — same _stem group → deduped
        out = detect_entities("novel", self.NOVEL_CORPUS, {"arthurs"})
        assert "arthur" not in out["entity_chunks"]

    def test_medical_term_spans(self):
        corpus = {
            "c0": "insulin therapy lowers glucose. insulin therapy helps.",
            "c1": "metformin is common. insulin therapy is discussed.",
        }
        out = detect_entities("medical", corpus, set())
        assert "insulin therapy" in out["entity_chunks"]
        assert out["entity_type"]["insulin therapy"] == "term"
        # appears in both chunks
        assert set(out["entity_chunks"]["insulin therapy"]) == {"c0", "c1"}

    def test_medical_term_freq1_dropped(self):
        corpus = {"c0": "basal carcinoma noted once here."}
        out = detect_entities("medical", corpus, set())
        assert "basal carcinoma" not in out["entity_chunks"]

    def test_output_structure(self):
        out = detect_entities("novel", self.NOVEL_CORPUS, set())
        for key in ("entity_chunks", "entity_frequency", "entity_display",
                    "entity_type", "qc"):
            assert key in out
        # frequencies only for entities with chunks
        assert set(out["entity_frequency"]) <= set(out["entity_chunks"])
