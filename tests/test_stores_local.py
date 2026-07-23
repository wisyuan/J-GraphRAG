"""Tests for jgraphrag.stores.local — NumpyVectorStore + JsonGraphStore."""
from __future__ import annotations

import numpy as np
import pytest

from jgraphrag.stores.local import JsonGraphStore, NumpyVectorStore


# ── NumpyVectorStore ─────────────────────────────────────────────────────


class TestNumpyVectorStore:
    def test_upsert_and_query_order(self):
        vs = NumpyVectorStore()
        vs.upsert("t", ["a", "b"],
                  np.array([[1.0, 0.0], [0.6, 0.8]]),
                  payloads=[{"x": 1}, {"x": 2}])
        hits = vs.query("t", np.array([1.0, 0.0]), 2)
        assert [i for i, _ in hits] == ["a", "b"]
        # cosine scores, descending
        assert hits[0][1] == pytest.approx(1.0)
        assert hits[1][1] == pytest.approx(0.6)
        assert hits[0][1] > hits[1][1]

    def test_query_normalizes_stored_rows(self):
        vs = NumpyVectorStore()
        # unnormalized stored vec in the same direction as the query
        vs.upsert("t", ["big"], np.array([[10.0, 0.0]]))
        _id, score = vs.query("t", np.array([1.0, 0.0]), 1)[0]
        assert _id == "big"
        assert score == pytest.approx(1.0)

    def test_upsert_overwrites_by_id(self):
        vs = NumpyVectorStore()
        vs.upsert("t", ["a", "b"], np.array([[1.0, 0.0], [0.6, 0.8]]))
        vs.upsert("t", ["a"], np.array([[0.0, 1.0]]),
                  payloads=[{"x": 99}])
        assert vs.count("t") == 2  # overwrite, not append
        assert vs.query("t", np.array([1.0, 0.0]), 2)[0][0] == "b"
        assert vs.query("t", np.array([0.0, 1.0]), 2)[0][0] == "a"
        assert vs.get_payload("t", "a") == {"x": 99}

    def test_upsert_shape_mismatch_raises(self):
        vs = NumpyVectorStore()
        with pytest.raises(ValueError):
            vs.upsert("t", ["a", "b"], np.array([[1.0, 0.0]]))
        with pytest.raises(ValueError):
            vs.upsert("t", ["a"], np.array([1.0, 0.0]))  # 1-D

    def test_top_k_clamped_to_table_size(self):
        vs = NumpyVectorStore()
        vs.upsert("t", ["a"], np.array([[1.0, 0.0]]))
        assert len(vs.query("t", np.array([1.0, 0.0]), 10)) == 1

    def test_unknown_namespace_is_empty(self):
        vs = NumpyVectorStore()
        assert vs.query("nope", np.array([1.0]), 5) == []
        assert vs.count("nope") == 0
        assert vs.get_payload("nope", "a") is None

    def test_get_payload_missing_id(self):
        vs = NumpyVectorStore()
        vs.upsert("t", ["a"], np.array([[1.0]]))
        assert vs.get_payload("t", "a") == {}
        assert vs.get_payload("t", "missing") is None

    def test_namespaces_are_independent(self):
        vs = NumpyVectorStore()
        vs.upsert("n1", ["a"], np.array([[1.0]]))
        vs.upsert("n2", ["b"], np.array([[1.0]]))
        assert vs.count("n1") == 1 and vs.count("n2") == 1
        assert vs.query("n1", np.array([1.0]), 5)[0][0] == "a"
        assert sorted(vs.namespaces()) == ["n1", "n2"]

    def test_save_load_roundtrip(self, tmp_path):
        vs = NumpyVectorStore()
        vs.upsert("ents", ["a", "b"],
                  np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                  payloads=[{"name": "a"}, {"name": "b"}])
        vs.upsert("rels", ["r0"], np.array([[0.5, 0.5]]),
                  payloads=[{"prob": 0.7}])
        vs.save(tmp_path)

        vs2 = NumpyVectorStore.load(tmp_path)
        assert sorted(vs2.namespaces()) == ["ents", "rels"]
        assert vs2.count("ents") == 2
        assert vs2.get_payload("ents", "b") == {"name": "b"}
        assert vs2.get_payload("rels", "r0") == {"prob": 0.7}
        hits = vs2.query("ents", np.array([0.0, 3.0]), 2)
        assert hits[0][0] == "b"
        assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
        # store stays writable after load
        vs2.upsert("ents", ["c"], np.array([[1.0, 1.0]]))
        assert vs2.count("ents") == 3

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NumpyVectorStore.load(tmp_path / "empty")


# ── JsonGraphStore ────────────────────────────────────────────────────────


class TestJsonGraphStore:
    def test_add_node_and_attrs(self):
        gs = JsonGraphStore()
        gs.add_node("n1", kind="entity", name="alpha")
        assert gs.get_node("n1") == {"kind": "entity", "name": "alpha"}
        # re-adding merges attrs
        gs.add_node("n1", weight=3)
        assert gs.get_node("n1")["kind"] == "entity"
        assert gs.get_node("n1")["weight"] == 3
        assert gs.get_node("missing") is None

    def test_undirected_edge_adjacency(self):
        gs = JsonGraphStore()
        eid = gs.add_edge("a", "b", weight=0.7, relation="related_to")
        assert gs.neighbors("a") == [("b", eid, 0.7)]
        assert gs.neighbors("b") == [("a", eid, 0.7)]  # reverse direction
        assert gs.neighbors("isolated") == []

    def test_edge_attrs_and_ids(self):
        gs = JsonGraphStore()
        e0 = gs.add_edge("a", "b", weight=1.0, relation="treats",
                         text="a treats b")
        e1 = gs.add_edge("b", "c")  # default weight
        assert e0 != e1  # unique edge ids
        edge = gs.get_edge(e0)
        assert edge["a"] == "a" and edge["b"] == "b"
        assert edge["weight"] == 1.0
        assert edge["relation"] == "treats"
        assert edge["text"] == "a treats b"
        assert gs.get_edge("missing") is None
        # default weight
        assert gs.get_edge(e1)["weight"] == 1.0

    def test_iter_nodes_and_edges(self):
        gs = JsonGraphStore()
        gs.add_node("a", kind="entity")
        gs.add_node("b")
        eid = gs.add_edge("a", "b")
        assert dict(gs.iter_nodes()) == {"a": {"kind": "entity"}, "b": {}}
        edges = dict(gs.iter_edges())
        assert list(edges) == [eid]
        assert edges[eid]["a"] == "a"

    def test_multiple_edges_between_same_nodes(self):
        gs = JsonGraphStore()
        e0 = gs.add_edge("a", "b", weight=0.5)
        e1 = gs.add_edge("a", "b", weight=0.9)
        nbs = gs.neighbors("a")
        assert sorted(nbs) == [("b", e0, 0.5), ("b", e1, 0.9)]

    def test_save_load_roundtrip(self, tmp_path):
        gs = JsonGraphStore()
        gs.add_node("n1", kind="entity")
        eid = gs.add_edge("n1", "n2", weight=0.7, relation="related_to")
        gs.save(tmp_path)

        gs2 = JsonGraphStore.load(tmp_path)
        assert gs2.get_node("n1") == {"kind": "entity"}
        assert gs2.neighbors("n1") == [("n2", eid, 0.7)]
        assert gs2.neighbors("n2") == [("n1", eid, 0.7)]
        assert gs2.get_edge(eid)["relation"] == "related_to"
        # edge counter survives the round trip — no id collision
        eid2 = gs2.add_edge("n2", "n3")
        assert eid2 != eid
        assert gs2.get_edge(eid2) is not None

    def test_load_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            JsonGraphStore.load(tmp_path / "empty")
