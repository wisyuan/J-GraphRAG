"""Local store implementations — the default persistence backends.

Adopted from the validated experiment internals:
- ``NumpyVectorStore``: in-memory namespaced vector table (dict of
  namespace -> (ids, vecs matrix, payloads)) with cosine-matmul top-k query,
  mirroring the ``ent_emb @ qv`` / ``np.argsort(-sims)`` retrieval core of
  ``experiments/phase50_lightrag_j.py`` (LightRagIndex.build_embeddings /
  lightrag_retrieve, :228/:293). Persists to a directory as npz + JSON.
- ``JsonGraphStore``: adjacency-dict property graph mirroring the ``adj``
  structure of ``LightRagIndex`` (:155/:223, entity -> sorted neighbor list),
  extended to edge ids per the GraphStore protocol. Persists as JSON.

Both are import-safe: no disk or model access at import time.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# File names used inside a save directory (shared by save/load).
_VECS_NPZ = "vectors.npz"
_VECS_META = "vector_payloads.json"
_GRAPH_JSON = "graph.json"


class NumpyVectorStore:
    """In-memory namespaced dense-vector store with top-k cosine query.

    Layout: ``_ns[namespace] = {"ids": [...], "vecs": np.ndarray [N, dim],
    "payloads": {id: dict}, "index": {id: row}}``. ``upsert`` overwrites by
    id; ``query`` L2-normalizes stored rows and the query vector, then takes
    matmul top-k (ties broken by insertion order, stable sort).
    """

    def __init__(self) -> None:
        self._ns: dict[str, dict[str, Any]] = {}

    def _table(self, namespace: str) -> dict[str, Any]:
        if namespace not in self._ns:
            self._ns[namespace] = {
                "ids": [],
                "vecs": np.zeros((0, 0), dtype=np.float32),
                "payloads": {},
                "index": {},
            }
        return self._ns[namespace]

    def upsert(
        self,
        namespace: str,
        ids: list[str],
        vecs: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(ids):
            raise ValueError(
                f"vecs shape {vecs.shape} does not match {len(ids)} ids")
        t = self._table(namespace)
        for i, vid in enumerate(ids):
            payload = payloads[i] if payloads is not None else {}
            if vid in t["index"]:
                row = t["index"][vid]
                t["vecs"][row] = vecs[i]
                t["payloads"][vid] = payload
            else:
                t["index"][vid] = len(t["ids"])
                t["ids"].append(vid)
                t["payloads"][vid] = payload
                if t["vecs"].size == 0:
                    t["vecs"] = np.zeros((0, vecs.shape[1]), dtype=np.float32)
                t["vecs"] = np.vstack([t["vecs"], vecs[i : i + 1]])

    def query(self, namespace: str, vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if namespace not in self._ns or not self._ns[namespace]["ids"]:
            return []
        t = self._ns[namespace]
        mat = t["vecs"].astype(np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.where(norms > 0, norms, 1.0)
        qv = np.asarray(vec, dtype=np.float64).ravel()
        qv = qv / (np.linalg.norm(qv) + 1e-12)
        sims = mat @ qv
        k = min(top_k, len(t["ids"]))
        top = np.argsort(-sims, kind="stable")[:k]
        return [(t["ids"][int(i)], float(sims[i])) for i in top]

    def get_payload(self, namespace: str, id: str) -> dict[str, Any] | None:
        t = self._ns.get(namespace)
        if t is None:
            return None
        return t["payloads"].get(id)

    def count(self, namespace: str) -> int:
        t = self._ns.get(namespace)
        return len(t["ids"]) if t is not None else 0

    def namespaces(self) -> list[str]:
        return list(self._ns.keys())

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, dir: str | Path) -> None:
        """Write all namespaces to ``dir`` (npz for ids/vecs, JSON for payloads)."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, Any] = {"namespaces": [], "payloads": {}}
        for i, (ns, t) in enumerate(sorted(self._ns.items())):
            arrays[f"ids__{i}"] = np.array(t["ids"], dtype="<U64")
            arrays[f"vecs__{i}"] = t["vecs"].astype(np.float32)
            meta["namespaces"].append(ns)
            meta["payloads"][ns] = t["payloads"]
        np.savez(dir / _VECS_NPZ, **arrays)
        (dir / _VECS_META).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dir: str | Path) -> "NumpyVectorStore":
        dir = Path(dir)
        store = cls()
        npz_path = dir / _VECS_NPZ
        meta_path = dir / _VECS_META
        if not npz_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"NumpyVectorStore files not found in {dir} "
                f"(expected {_VECS_NPZ} + {_VECS_META})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        data = np.load(npz_path, allow_pickle=False)
        for i, ns in enumerate(meta["namespaces"]):
            ids = [str(x) for x in data[f"ids__{i}"].tolist()]
            vecs = data[f"vecs__{i}"].astype(np.float32)
            payloads = meta["payloads"].get(ns, {})
            store._ns[ns] = {
                "ids": ids,
                "vecs": vecs,
                "payloads": payloads,
                "index": {vid: row for row, vid in enumerate(ids)},
            }
        return store


class JsonGraphStore:
    """Adjacency-dict property graph (undirected edges) with JSON persistence.

    ``_adj[node] = [(neighbor, edge_id)]`` mirrors the validated
    ``LightRagIndex.adj`` shape (entity -> neighbors); edge attributes
    (weight, relation, text) live in a parallel edge table so ``neighbors``
    can return ``(neighbor_id, edge_id, weight)`` per the GraphStore protocol.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, dict[str, Any]] = {}
        self._adj: dict[str, list[tuple[str, str]]] = {}
        self._edge_seq = 0

    def add_node(self, node_id: str, **attrs: Any) -> None:
        if node_id in self._nodes:
            self._nodes[node_id].update(attrs)
        else:
            self._nodes[node_id] = dict(attrs)
            self._adj.setdefault(node_id, [])

    def add_edge(self, a: str, b: str, weight: float = 1.0, **attrs: Any) -> str:
        edge_id = f"e{self._edge_seq}"
        self._edge_seq += 1
        self._edges[edge_id] = {
            "a": a, "b": b, "weight": float(weight), **attrs,
        }
        self._adj.setdefault(a, []).append((b, edge_id))
        self._adj.setdefault(b, []).append((a, edge_id))
        return edge_id

    def neighbors(self, node_id: str) -> list[tuple[str, str, float]]:
        out = []
        for nb, edge_id in self._adj.get(node_id, []):
            out.append((nb, edge_id, float(self._edges[edge_id]["weight"])))
        return out

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: str) -> dict[str, Any] | None:
        return self._edges.get(edge_id)

    def iter_nodes(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(self._nodes.items())

    def iter_edges(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return iter(self._edges.items())

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, dir: str | Path) -> None:
        """Write nodes/edges/adjacency to ``dir/graph.json``."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "nodes": self._nodes,
            "edges": self._edges,
            "adj": {k: [list(pair) for pair in v] for k, v in self._adj.items()},
            "edge_seq": self._edge_seq,
        }
        (dir / _GRAPH_JSON).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dir: str | Path) -> "JsonGraphStore":
        dir = Path(dir)
        path = dir / _GRAPH_JSON
        if not path.exists():
            raise FileNotFoundError(f"JsonGraphStore file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store._nodes = {str(k): v for k, v in payload["nodes"].items()}
        store._edges = {str(k): v for k, v in payload["edges"].items()}
        store._adj = {
            str(k): [(str(nb), str(eid)) for nb, eid in v]
            for k, v in payload["adj"].items()
        }
        store._edge_seq = int(payload.get("edge_seq", len(store._edges)))
        return store


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    vs = NumpyVectorStore()
    vs.upsert("t", ["a", "b"], np.array([[1.0, 0.0], [0.6, 0.8]]),
              payloads=[{"x": 1}, {"x": 2}])
    assert [i for i, _ in vs.query("t", np.array([1.0, 0.0]), 2)] == ["a", "b"]
    vs.upsert("t", ["a"], np.array([[0.0, 1.0]]))  # overwrite a -> b now wins
    assert vs.query("t", np.array([1.0, 0.0]), 2)[0][0] == "b"
    assert vs.get_payload("t", "b") == {"x": 2}
    assert vs.count("t") == 2

    gs = JsonGraphStore()
    gs.add_node("n1", kind="entity")
    eid = gs.add_edge("n1", "n2", weight=0.7, relation="related_to")
    assert gs.neighbors("n1") == [("n2", eid, 0.7)]
    assert gs.neighbors("n2") == [("n1", eid, 0.7)]
    assert gs.get_edge(eid)["relation"] == "related_to"

    with tempfile.TemporaryDirectory() as d:
        vs.save(d)
        gs.save(d)
        vs2 = NumpyVectorStore.load(d)
        gs2 = JsonGraphStore.load(d)
        assert vs2.query("t", np.array([0.0, 1.0]), 1)[0][0] == "a"
        assert gs2.neighbors("n1") == [("n2", eid, 0.7)]
        eid2 = gs2.add_edge("n2", "n3")
        assert eid2 != eid  # edge counter survives a save/load round trip
    print("OK")
