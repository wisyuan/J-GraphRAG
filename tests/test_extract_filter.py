"""Tests for jgraphrag.extract.filter — validated Phase 25 filter rules."""
from __future__ import annotations

from jgraphrag.extract.filter import (
    PREFILL_WORDS,
    ROLE_STOP,
    STOP_WORDS_EXTENDED,
    _stem,
    classify_concept_pos,
    complete_prefix,
    concept_ok,
    is_ascii_english,
    role_ok,
)


class TestConceptOk:
    def test_rejects_non_ascii(self):
        keep, reason = concept_ok("männer")
        assert not keep and reason == "non_ascii"
        keep, reason = concept_ok("novità")
        assert not keep and reason == "non_ascii"

    def test_rejects_prefill_words(self):
        keep, reason = concept_ok("discussed")
        assert not keep and reason == "prefill_stopword"
        assert "discussed" in PREFILL_WORDS

    def test_rejects_extended_stopwords(self):
        keep, reason = concept_ok("methodology")
        assert not keep and reason == "prefill_stopword"
        assert "methodology" in STOP_WORDS_EXTENDED

    def test_rejects_verb_gerund(self):
        keep, reason = concept_ok("running")
        assert not keep and reason == "verb_form"

    def test_rejects_verb_past(self):
        keep, reason = concept_ok("treated")
        assert not keep and reason == "verb_form"

    def test_accepts_domain_concepts(self):
        for word in ("insulin", "hypertension", "statin", "cancer"):
            keep, reason = concept_ok(word)
            assert keep, f"{word} dropped: {reason}"
            assert reason == ""

    def test_case_insensitive(self):
        keep, _ = concept_ok("DISCUSSED")
        assert not keep
        keep, _ = concept_ok("Insulin")
        assert keep


class TestRoleOk:
    def test_rejects_role_stop(self):
        assert not role_ok("type")
        assert "type" in ROLE_STOP
        assert not role_ok("aspects")

    def test_rejects_role_stop_via_stem(self):
        # not itself in ROLE_STOP, but its stem is
        assert "roless" not in ROLE_STOP
        assert _stem("roless") in ROLE_STOP
        assert not role_ok("roless")

    def test_rejects_concept_failures(self):
        assert not role_ok("discussed")   # prefill word
        assert not role_ok("treated")     # verb form
        assert not role_ok("männer")      # non-ascii

    def test_accepts_role_words(self):
        assert role_ok("hormone")
        assert role_ok("therapy")
        assert role_ok("biomarker")


class TestStem:
    def test_ies_to_y(self):
        assert _stem("studies") == "study"

    def test_es_stripped(self):
        assert _stem("doses") == "dos"

    def test_s_stripped(self):
        assert _stem("cats") == "cat"

    def test_short_words_unchanged(self):
        assert _stem("is") == "is"
        assert _stem("insulin") == "insulin"

    def test_lowercases(self):
        assert _stem("CATS") == "cat"


class TestClassifyConceptPos:
    def test_noun_suffix(self):
        assert classify_concept_pos("treatment") == "NN"
        assert classify_concept_pos("hypertension") == "NN"

    def test_plural(self):
        assert classify_concept_pos("statins") == "NNS"

    def test_plural_exceptions_are_not_nns(self):
        # ss/us/is/os endings are singular
        assert classify_concept_pos("glucose") != "NNS"
        assert classify_concept_pos("analysis") != "NNS"

    def test_gerund(self):
        assert classify_concept_pos("running") == "VBG"

    def test_past_tense(self):
        assert classify_concept_pos("treated") == "VBD"

    def test_eed_not_vbd(self):
        assert classify_concept_pos("need") != "VBD"

    def test_adjective(self):
        assert classify_concept_pos("careful") == "JJ"
        assert classify_concept_pos("logical") == "JJ"

    def test_short(self):
        assert classify_concept_pos("ab") == "SHORT"

    def test_unknown_root_noun(self):
        assert classify_concept_pos("insulin") == "UNK"
        assert classify_concept_pos("cancer") == "UNK"


class TestIsAsciiEnglish:
    def test_ascii(self):
        assert is_ascii_english("insulin")

    def test_non_ascii(self):
        assert not is_ascii_english("männer")
        assert not is_ascii_english("ユーザー")


class TestCompletePrefix:
    FREQ = {"statins": 50, "static": 3}

    def test_corpus_frequency_wins(self):
        # "statins"@50 beats "static"@3 (BM25-style domain-term ranking)
        assert complete_prefix("Stat", self.FREQ, set()) == "statins"

    def test_single_corpus_candidate(self):
        # "statin" is a prefix of "statins" but not of "static"
        assert complete_prefix("Statin", self.FREQ, set()) == "statins"

    def test_frequency_tie_broken_by_shorter(self):
        freq = {"statin": 5, "statins": 5}
        assert complete_prefix("stat", freq, set()) == "statin"

    def test_too_short_prefix_returns_none(self):
        assert complete_prefix("st", self.FREQ, set()) is None

    def test_no_candidates_returns_none(self):
        assert complete_prefix("zzz", self.FREQ, set()) is None

    def test_wordnet_fallback_single(self):
        assert complete_prefix("stat", {}, {"statins"}) == "statins"

    def test_wordnet_fallback_prefers_shorter(self):
        assert complete_prefix("stat", {}, {"statin", "statins"}) == "statin"

    def test_corpus_beats_wordnet(self):
        out = complete_prefix("stat", {"statins": 2}, {"static"})
        assert out == "statins"

    def test_prefix_matching_is_case_insensitive(self):
        assert complete_prefix("STAT", self.FREQ, set()) == "statins"
