"""Tests for jgraphrag.retrieve — dual-level retrieval + merge helpers."""
from __future__ import annotations

import numpy as np
import pytest

from jgraphrag.index import GraphIndex
from jgraphrag.retrieve import (
    NEIGHBOR_DECAY,
    interleave_rankings,
    lightrag_retrieve,
    merge_topk,
)
from jgraphrag.stores.local import JsonGraphStore, NumpyVectorStore


def _toy_index(embed) -> GraphIndex:
    """alpha-[c0], beta-[c0,c1], gamma-[c1]; edge alpha—beta (prob 1.0)."""
    graph, vectors = JsonGraphStore(), NumpyVectorStore()
    idx = GraphIndex(graph, vectors,
                     {"c0": "alpha beta", "c1": "beta gamma",
                      "c2": "unrelated"})
    idx.entities = [
        {"name": "alpha", "members": ["alpha"], "chunks": ["c0"],
         "roles": [], "text": "alpha"},
        {"name": "beta", "members": ["beta"], "chunks": ["c0", "c1"],
         "roles": [], "text": "beta"},
        {"name": "gamma", "members": ["gamma"], "chunks": ["c1"],
         "roles": [], "text": "gamma"},
    ]
    for i in range(3):
        graph.add_node(str(i), name=idx.entities[i]["name"])
    idx.relations = [{"a": 0, "b": 1, "relation": "related_to", "prob": 1.0,
                      "text": "alpha related to beta", "completed": False}]
    graph.add_edge("0", "1", weight=1.0, relation="related_to")
    idx.build_embeddings(embed)
    idx.build_chunk_embeddings(embed)
    return idx


class TestLightragRetrieve:
    def test_direct_hit_and_one_hop_decay(self, onehot_embed):
        idx = _toy_index(onehot_embed)
        qv = np.asarray(onehot_embed.embed(["alpha"])[0])
        scores, debug = lightrag_retrieve(idx, qv)
        # c0 holds alpha (direct hit, sim 1.0)
        assert scores["c0"] == pytest.approx(1.0)
        # c1 scored only via beta, alpha's one-hop neighbor, decayed 0.5
        assert scores["c1"] == pytest.approx(NEIGHBOR_DECAY)
        # gamma is not connected to alpha → c2 never scored
        assert "c2" not in scores

    def test_decay_anchor_is_half(self):
        assert NEIGHBOR_DECAY == 0.5

    def test_max_aggregation_not_sum(self, onehot_embed):
        # c0 contains BOTH alpha (weight 1.0) and beta (decayed 0.5):
        # MAX → 1.0; SUM would give 1.5.
        idx = _toy_index(onehot_embed)
        qv = np.asarray(onehot_embed.embed(["alpha"])[0])
        scores, _ = lightrag_retrieve(idx, qv)
        assert scores["c0"] == pytest.approx(1.0)

    def test_relation_channel_weights_endpoints_by_sim_x_prob(self,
                                                              onehot_embed):
        idx = _toy_index(onehot_embed)
        idx.relations[0]["prob"] = 0.4
        # rebuild relation embeddings so payload prob matches, then query
        # with the relation text's vector (orthogonal to all entity vecs)
        idx.build_embeddings(onehot_embed)
        qv = np.asarray(
            onehot_embed.embed(["alpha related to beta"])[0])
        scores, debug = lightrag_retrieve(idx, qv)
        # endpoints get w = sim(1.0) x prob(0.4) = 0.4
        assert scores["c0"] == pytest.approx(0.4)
        assert scores["c1"] == pytest.approx(0.4)
        assert debug["hit_relations"][0]["prob"] == pytest.approx(0.4)

    def test_zero_similarity_hits_ignored(self, onehot_embed):
        idx = _toy_index(onehot_embed)
        # a vector orthogonal to everything stored
        qv = np.zeros(onehot_embed.dim)
        qv[-1] = 1.0
        scores, debug = lightrag_retrieve(idx, qv)
        assert scores == {}
        assert debug["n_entities_weighted"] == 0

    def test_debug_payload(self, onehot_embed):
        idx = _toy_index(onehot_embed)
        qv = np.asarray(onehot_embed.embed(["alpha"])[0])
        _scores, debug = lightrag_retrieve(idx, qv)
        assert debug["hit_entities"][0][0] == "alpha"
        assert debug["hit_entities"][0][1] == pytest.approx(1.0)
        assert debug["n_entities_weighted"] >= 2


class TestMergeTopk:
    def test_graph_ranked_first_then_backfill(self):
        scores = {"c0": 1.0, "c1": 0.5}
        out = merge_topk(scores, ["c1", "c2", "c3"], top_k=3)
        assert out == ["c0", "c1", "c2"]  # c1 not duplicated

    def test_ties_broken_by_bge_sim(self):
        scores = {"c0": 0.5, "c1": 0.5}
        out = merge_topk(scores, [], bge_sim={"c1": 0.9, "c0": 0.1},
                         top_k=2)
        assert out == ["c1", "c0"]

    def test_top_k_limits_output(self):
        scores = {"c0": 1.0, "c1": 0.9, "c2": 0.8}
        assert len(merge_topk(scores, ["c3", "c4"], top_k=2)) == 2

    def test_backfill_only_when_room(self):
        scores = {"c0": 1.0}
        out = merge_topk(scores, ["c9"], top_k=2)
        assert out == ["c0", "c9"]


class TestInterleaveRankings:
    def test_round_robin(self):
        out = interleave_rankings(["g1", "g2", "g3"], ["n1", "n2"], top_k=4)
        assert out == ["g1", "n1", "g2", "n2"]

    def test_duplicates_across_pools_skipped(self):
        out = interleave_rankings(["g1", "x"], ["x", "n1"], top_k=3)
        assert out == ["g1", "x", "n1"]

    def test_tiny_corpus_exhaustion_guard(self):
        # both pools shorter than top_k combined → must terminate
        out = interleave_rankings(["g1"], ["n1"], top_k=10)
        assert out == ["g1", "n1"]

    def test_empty_pools(self):
        assert interleave_rankings([], [], top_k=5) == []

    def test_graph_pool_leads(self):
        out = interleave_rankings(["g1"], ["n1", "n2"], top_k=3)
        assert out == ["g1", "n1", "n2"]


class TestGraphIndexRetrieve:
    def test_end_to_end_with_injected_embed(self, onehot_embed):
        idx = _toy_index(onehot_embed)
        out = idx.retrieve("alpha", top_k=2, embed=onehot_embed)
        assert len(out) == 2
        assert out[0] == "c0"  # graph channel: alpha's chunk leads
        assert set(out) <= {"c0", "c1", "c2"}

    def test_top_k_respected(self, onehot_embed):
        idx = _toy_index(onehot_embed)
        out = idx.retrieve("alpha", top_k=1, embed=onehot_embed)
        assert out == ["c0"]
