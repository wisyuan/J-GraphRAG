"""Tests for jgraphrag.index — merge_entities + GraphIndex assembly."""
from __future__ import annotations

import numpy as np
import pytest

from jgraphrag.index import (
    MERGE_WS_COS,
    NS_CHUNKS,
    NS_ENTITIES,
    NS_RELATIONS,
    GraphIndex,
    merge_entities,
)
from jgraphrag.providers.base import ChunkExtraction
from jgraphrag.stores.local import JsonGraphStore, NumpyVectorStore


def _v(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=np.float64)


class TestMergeEntities:
    def test_stem_group_merged(self):
        concepts = ["insulin", "insulins", "glucose"]
        # orthogonal ws vectors — only the _stem group may merge
        ws = np.eye(3)
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {
            frozenset({0, 1}), frozenset({2})}

    def test_ws_cosine_above_threshold_merged(self):
        concepts = ["aspirin", "ibuprofen", "glucose"]
        ws = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {
            frozenset({0, 1}), frozenset({2})}

    def test_ws_cosine_below_threshold_not_merged(self):
        concepts = ["a", "b"]
        angle = np.arccos(0.90)  # cos 0.90 < MERGE_WS_COS 0.95
        ws = np.array([[1.0, 0.0],
                       [np.cos(angle), np.sin(angle)]])
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {
            frozenset({0}), frozenset({1})}

    def test_threshold_anchor_is_095(self):
        assert MERGE_WS_COS == 0.95

    def test_at_threshold_merges(self):
        # cos = 19/20 == 0.95 exactly (sqrt(19² + 39) = 20) → >= threshold
        concepts = ["a", "b"]
        ws = np.array([[1.0, 0.0], [19.0, np.sqrt(39.0)]])
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {frozenset({0, 1})}

    def test_zero_vector_merges_nothing(self):
        concepts = ["a", "b"]
        ws = np.array([[0.0, 0.0], [1.0, 0.0]])
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {
            frozenset({0}), frozenset({1})}

    def test_transitive_merge(self):
        # stem-merge (0,1) + ws-merge (1,2) → one group
        concepts = ["statin", "statins", "lipitor"]
        ws = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])
        groups = merge_entities(concepts, ws)
        assert {frozenset(g) for g in groups} == {frozenset({0, 1, 2})}

    def test_groups_sorted_by_min_index(self):
        concepts = ["abc", "def", "defs"]  # _stem("defs") == "def"
        ws = np.eye(3)
        groups = merge_entities(concepts, ws)
        assert groups == [[0], [1, 2]]


# ── GraphIndex assembly ──────────────────────────────────────────────────


def _extractions() -> tuple[list[ChunkExtraction], list[str]]:
    c0 = ChunkExtraction(
        concepts=["Insulin", "glucose"],
        roles={"Insulin": ["hormone"], "glucose": ["sugar"]},
        ws_vecs={"Insulin": _v(1, 0, 0), "glucose": _v(0, 1, 0)},
    )
    c1 = ChunkExtraction(
        concepts=["insulin", "metformin"],
        roles={"insulin": ["hormone", "treatment"]},
        ws_vecs={"insulin": _v(1, 0, 0), "metformin": _v(0, 0, 1)},
    )
    return [c0, c1], ["c0", "c1"]


class TestFromExtractions:
    def _build(self, relation_edges=None, chunks=None):
        exts, chunk_ids = _extractions()
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        index = GraphIndex.from_extractions(
            exts, chunk_ids, relation_edges or [], graph, vectors,
            chunks=chunks)
        return index, graph, vectors

    def test_entities_aggregated_across_chunks(self):
        index, _g, _vstore = self._build()
        by_name = {e["name"]: e for e in index.entities}
        assert set(by_name) == {"glucose", "insulin", "metformin"}
        # case-insensitive aggregation: "Insulin" (c0) + "insulin" (c1)
        assert by_name["insulin"]["chunks"] == ["c0", "c1"]
        assert by_name["glucose"]["chunks"] == ["c0"]
        # roles union across chunks
        assert by_name["insulin"]["roles"] == ["hormone", "treatment"]
        assert by_name["metformin"]["roles"] == []

    def test_entity_text_format(self):
        index, _g, _v = self._build()
        by_name = {e["name"]: e for e in index.entities}
        assert by_name["insulin"]["text"] == "insulin: hormone, treatment"
        assert by_name["glucose"]["text"] == "glucose: sugar"
        assert by_name["metformin"]["text"] == "metformin"

    def test_canon_of_maps_original_concepts(self):
        index, _g, _v = self._build()
        assert set(index.canon_of) == {"glucose", "insulin", "metformin"}
        names = [e["name"] for e in index.entities]
        for concept, idx in index.canon_of.items():
            assert 0 <= idx < len(names)

    def test_stem_merged_concepts_become_one_entity(self):
        exts = [
            ChunkExtraction(concepts=["statin"], ws_vecs={"statin": _v(0, 1)}),
            ChunkExtraction(concepts=["statins"],
                            ws_vecs={"statins": _v(0, 1)}),
        ]
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        index = GraphIndex.from_extractions(exts, ["c0", "c1"], [], graph,
                                            vectors)
        assert len(index.entities) == 1
        ent = index.entities[0]
        assert sorted(ent["members"]) == ["statin", "statins"]
        assert ent["chunks"] == ["c0", "c1"]

    def test_relation_edges(self):
        edges = [
            {"concept_a": "Insulin", "concept_b": "glucose",
             "relation": "regulates", "prob": 0.8},
            # self-loop after canonicalization → skipped
            {"concept_a": "insulin", "concept_b": "INSULIN",
             "relation": "x", "prob": 1.0},
            # unknown concept → skipped
            {"concept_a": "ghost", "concept_b": "glucose",
             "relation": "x", "prob": 1.0},
        ]
        index, graph, _v = self._build(edges)
        assert len(index.relations) == 1
        rel = index.relations[0]
        names = [e["name"] for e in index.entities]
        assert names[rel["a"]] == "insulin"
        assert names[rel["b"]] == "glucose"
        assert rel["relation"] == "regulates"
        assert rel["prob"] == pytest.approx(0.8)
        assert rel["text"] == "insulin regulates glucose"
        assert rel["completed"] is False
        # graph store carries the same edge
        a, b = str(rel["a"]), str(rel["b"])
        nbs = graph.neighbors(a)
        assert len(nbs) == 1
        nb, eid, w = nbs[0]
        assert nb == b and w == pytest.approx(0.8)
        assert graph.get_edge(eid)["relation"] == "regulates"

    def test_graph_nodes_created(self):
        index, graph, _v = self._build()
        for i, ent in enumerate(index.entities):
            node = graph.get_node(str(i))
            assert node is not None
            assert node["name"] == ent["name"]
            assert node["text"] == ent["text"]


class TestBuildEmbeddings:
    def test_entity_relation_chunk_namespaces(self, onehot_embed):
        exts, chunk_ids = _extractions()
        edges = [{"concept_a": "insulin", "concept_b": "glucose",
                  "relation": "regulates", "prob": 0.8}]
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        index = GraphIndex.from_extractions(
            exts, chunk_ids, edges, graph, vectors,
            chunks={"c0": "insulin glucose text", "c1": "insulin metformin"})
        index.build_embeddings(onehot_embed)
        index.build_chunk_embeddings(onehot_embed)
        assert vectors.count(NS_ENTITIES) == 3
        assert vectors.count(NS_RELATIONS) == 1
        assert vectors.count(NS_CHUNKS) == 2
        payload = vectors.get_payload(NS_ENTITIES, "0")
        assert payload["name"] == index.entities[0]["name"]
        rel_payload = vectors.get_payload(NS_RELATIONS, "0")
        assert rel_payload["prob"] == pytest.approx(0.8)
        # stored rows are L2-normalized: exact direction match scores 1.0
        qv = np.asarray(onehot_embed.embed(
            [index.entities[0]["text"]])[0])
        _id, score = vectors.query(NS_ENTITIES, qv, 1)[0]
        assert _id == "0"
        assert score == pytest.approx(1.0, abs=1e-5)


class TestAugmentWithEntities:
    def _base_index(self):
        exts, chunk_ids = _extractions()
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        return GraphIndex.from_extractions(exts, chunk_ids, [], graph,
                                           vectors), graph

    def _cache(self):
        return {
            "entity_chunks": {
                "excalibur": ["c0", "c1"],
                "dozmare": ["c0", "c1"],
                "lancelot": ["c0"],          # only one co-occurrence
            },
            "entity_display": {"excalibur": "Excalibur", "dozmare": "Dozmare",
                               "lancelot": "Lancelot"},
            "entity_frequency": {"excalibur": 10, "dozmare": 9, "lancelot": 8},
        }

    def test_new_entity_nodes(self):
        index, graph = self._base_index()
        stats = index.augment_with_entities(self._cache())
        assert stats["n_base_entities"] == 3
        assert stats["n_new_entities"] == 3
        assert len(index.entities) == 6
        names = {e["name"] for e in index.entities}
        assert {"Excalibur", "Dozmare", "Lancelot"} <= names
        # graph nodes added for the new entities
        for i in range(3, 6):
            assert graph.get_node(str(i)) is not None

    def test_cooccurrence_edges_min2(self):
        index, _g = self._base_index()
        stats = index.augment_with_entities(self._cache())
        # excalibur × dozmare co-occur in c0 AND c1 → edge (MIN_COOCC=2)
        assert stats["n_ee_edges"] == 1
        ee = [r for r in index.relations
              if index.entities[r["a"]]["name"] == "Excalibur"
              and index.entities[r["b"]]["name"] == "Dozmare"]
        assert len(ee) == 1
        assert ee[0]["relation"] == "related_to"
        # prob normalized by max count: 2/2 = 1.0
        assert ee[0]["prob"] == pytest.approx(1.0)
        assert ee[0]["completed"] is True
        # lancelot co-occurs only once → no edge
        lanc = [r for r in index.relations
                if "Lancelot" in (index.entities[r["a"]]["name"],
                                  index.entities[r["b"]]["name"])
                and r["relation"] == "related_to"]
        assert lanc == []

    def test_entity_concept_edges(self):
        index, _g = self._base_index()
        stats = index.augment_with_entities(self._cache())
        # insulin is in c0+c1; excalibur & dozmare co-occur with it twice
        assert stats["n_ec_edges"] == 2
        ec_pairs = set()
        for r in index.relations:
            na, nb = (index.entities[r["a"]]["name"],
                      index.entities[r["b"]]["name"])
            if "insulin" in (na, nb) and r["relation"] == "related_to":
                other = nb if na == "insulin" else na
                ec_pairs.add(other)
        assert ec_pairs == {"Excalibur", "Dozmare"}

    def test_edge_text_uses_display_names(self):
        index, _g = self._base_index()
        index.augment_with_entities(self._cache())
        texts = {r["text"] for r in index.relations}
        assert "Excalibur related to Dozmare" in texts


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, onehot_embed):
        exts, chunk_ids = _extractions()
        edges = [{"concept_a": "insulin", "concept_b": "glucose",
                  "relation": "regulates", "prob": 0.8}]
        graph, vectors = JsonGraphStore(), NumpyVectorStore()
        index = GraphIndex.from_extractions(
            exts, chunk_ids, edges, graph, vectors,
            chunks={"c0": "insulin glucose text", "c1": "insulin metformin"})
        index.build_embeddings(onehot_embed)
        index.build_chunk_embeddings(onehot_embed)
        index.save(tmp_path)

        loaded = GraphIndex.load(tmp_path)
        assert loaded.entities == index.entities
        assert loaded.relations == index.relations
        assert loaded.canon_of == index.canon_of
        assert loaded.chunks == index.chunks
        # stores survived
        assert loaded.vectors.count(NS_ENTITIES) == 3
        assert loaded.vectors.count(NS_CHUNKS) == 2
        a, b = str(index.relations[0]["a"]), str(index.relations[0]["b"])
        assert loaded.graph.neighbors(a)[0][0] == b

    def test_load_missing_meta_raises(self, tmp_path):
        NumpyVectorStore().save(tmp_path)
        JsonGraphStore().save(tmp_path)
        with pytest.raises(FileNotFoundError):
            GraphIndex.load(tmp_path)
