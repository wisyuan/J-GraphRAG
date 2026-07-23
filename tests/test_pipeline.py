"""Tests for jgraphrag.pipeline.build_index — full chain with mock providers."""
from __future__ import annotations

import numpy as np
import pytest

from jgraphrag.index import NS_CHUNKS, NS_ENTITIES, NS_RELATIONS
from jgraphrag.pipeline import build_index
from jgraphrag.providers.base import ChunkExtraction


class MockLens:
    """Scripted LensProvider: fixed extractions + fixed relation readout."""

    def __init__(self, by_text: dict[str, ChunkExtraction],
                 rel: tuple[str, float] = ("treats", 0.9)) -> None:
        self._by_text = by_text
        self._rel = rel
        self.concept_calls: list[list[str]] = []
        self.relation_calls: list[tuple[str, str]] = []

    def extract_concepts(self, chunks: list[str]) -> list[ChunkExtraction]:
        self.concept_calls.append(list(chunks))
        return [self._by_text[t] for t in chunks]

    def extract_relation(self, context, concept_a, concept_b):
        self.relation_calls.append((concept_a, concept_b))
        return self._rel


def _v(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float64)


@pytest.fixture
def chunks() -> dict[str, str]:
    return {"c0": "insulin lowers glucose", "c1": "metformin and insulin"}


@pytest.fixture
def lens(chunks) -> MockLens:
    # "insulin"/"glucose" appear in both chunks (DF=2 → kept); "sulfonylurea"
    # only in c0 (DF=1 → dropped); "discussed" is a prefill word (dropped).
    return MockLens({
        chunks["c0"]: ChunkExtraction(
            concepts=["insulin", "glucose", "sulfonylurea"],
            roles={"insulin": ["hormone"], "glucose": ["fuel"]},
            ws_vecs={"insulin": _v(1, 0, 0), "glucose": _v(0, 1, 0),
                     "sulfonylurea": _v(0, 0, 1)},
        ),
        chunks["c1"]: ChunkExtraction(
            concepts=["insulin", "glucose", "discussed"],
            roles={"insulin": ["hormone"]},
            ws_vecs={"insulin": _v(1, 0, 0), "glucose": _v(0, 1, 0)},
        ),
    })


class TestBuildIndex:
    def test_full_chain(self, chunks, lens, token_embed):
        index = build_index(chunks, lens=lens, embed=token_embed)
        names = {e["name"] for e in index.entities}
        assert {"insulin", "glucose"} <= names
        # DF<2 and prefill-stopword concepts filtered out
        assert "sulfonylurea" not in names
        assert "discussed" not in names
        # relation readout over the one surviving co-occurring pair
        assert lens.relation_calls == [("glucose", "insulin")]
        assert len(index.relations) == 1
        rel = index.relations[0]
        assert rel["relation"] == "treats"
        assert rel["prob"] == pytest.approx(0.9)
        assert rel["text"] == "glucose treats insulin"
        # vector stores populated (entities, relations, chunks)
        assert index.vectors.count(NS_ENTITIES) == len(index.entities)
        assert index.vectors.count(NS_RELATIONS) == 1
        assert index.vectors.count(NS_CHUNKS) == 2

    def test_accepts_plain_list_of_chunks(self, lens, token_embed):
        index = build_index(["insulin lowers glucose",
                             "metformin and insulin"],
                            lens=lens, embed=token_embed)
        assert set(index.chunks) == {"0", "1"}
        assert index.vectors.count(NS_CHUNKS) == 2

    def test_injected_stores_are_used(self, chunks, lens, token_embed):
        from jgraphrag.stores.local import JsonGraphStore, NumpyVectorStore
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        index = build_index(chunks, lens=lens, embed=token_embed,
                            stores=(graph, vectors))
        assert index.graph is graph
        assert index.vectors is vectors
        assert vectors.count(NS_ENTITIES) > 0

    def test_retrieve_after_build(self, chunks, lens, token_embed):
        index = build_index(chunks, lens=lens, embed=token_embed)
        out = index.retrieve("insulin", top_k=2, embed=token_embed)
        assert len(out) == 2
        assert set(out) == {"c0", "c1"}
        # both chunks contain the insulin entity; graph channel leads
        assert out[0] == "c0"

    def test_save_load_after_build(self, chunks, lens, token_embed, tmp_path):
        from jgraphrag.index import GraphIndex
        index = build_index(chunks, lens=lens, embed=token_embed)
        index.save(tmp_path)
        loaded = GraphIndex.load(tmp_path)
        assert loaded.entities == index.entities
        assert loaded.relations == index.relations
        out = loaded.retrieve("insulin", top_k=2, embed=token_embed)
        assert set(out) == {"c0", "c1"}

    def test_no_relation_when_pairs_filtered(self, chunks, token_embed):
        # each chunk keeps only one concept → no co-occurring pairs
        lens = MockLens({
            chunks["c0"]: ChunkExtraction(concepts=["insulin"]),
            chunks["c1"]: ChunkExtraction(concepts=["insulin"]),
        })
        index = build_index(chunks, lens=lens, embed=token_embed)
        assert lens.relation_calls == []
        assert index.relations == []
