"""Graph index assembly — entity merge + graph/vector store population.

Adopted from the validated experiment code (algorithm, thresholds and
aggregation kept exactly as validated):

- ``_stem`` / ``merge_entities``: ``experiments/phase50_lightrag_j.py:113``
  (union-find over same-_stem groups + ws cosine >= 0.95, pure CPU), with
  ``_stem`` from ``experiments/phase39_two_pass_cache.py:78``.
- ``GraphIndex.from_extractions``: the ``LightRagIndex`` constructor of
  ``experiments/phase50_lightrag_j.py:155`` (entity records with
  name/members/chunks/roles and text "name: role1, role2"; relation edges
  with sim x prob semantics; adjacency), re-targeted onto the
  GraphStore/VectorStore protocols.
- ``GraphIndex.build_embeddings``: ``LightRagIndex.build_embeddings``
  (:228), with the raw ``embed_fn`` swapped for an injected EmbedProvider.
- ``GraphIndex.augment_with_entities``: ``augment_index`` of
  ``experiments/phase53_textside_entities.py:556`` (text-side entity nodes +
  co-occurrence edges: top-15 entities per chunk, MIN_COOCC=2, caps of 20000,
  relation="related_to", prob = normalized co-occurrence count). The GPU
  variant ``jlens_readout_top_pairs`` is intentionally NOT adopted.

Import-safe: no disk or model access at import time.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .providers.base import ChunkExtraction

if TYPE_CHECKING:  # protocol typing only — no runtime dependency
    from .providers.base import EmbedProvider
    from .stores.base import GraphStore, VectorStore

# ── Validated hyperparameters (do not change without re-validation) ──────
MERGE_WS_COS = 0.95   # D(.) dedupe: ws cosine merge threshold (phase50)
MIN_COOCC = 2         # min co-occurrence count for cooc edges (phase53)
MAX_EE_EDGES = 20000  # entity-entity co-occurrence edge cap (phase53)
MAX_EC_EDGES = 20000  # entity-concept co-occurrence edge cap (phase53)

# VectorStore namespaces used by the index/retrieval pipeline.
NS_ENTITIES = "entities"
NS_RELATIONS = "relations"
NS_CHUNKS = "chunks"

_META_JSON = "meta.json"


def _stem(word: str) -> str:
    """Crude singular form for dedupe keys only (not for output)."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s"):
        return w[:-1]
    return w


def _norm_concept(s: str) -> str:
    return " ".join(_stem(w) for w in s.lower().split())


def merge_entities(
    concepts: list[str], ws_vec: np.ndarray, threshold: float = MERGE_WS_COS,
) -> list[list[int]]:
    """Union-find merge: same _stem group, or ws cosine >= threshold.

    Returns member-index groups (verbatim port of phase50_lightrag_j.py:113).
    """
    n = len(concepts)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_stem: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(concepts):
        by_stem[_stem(c)].append(i)
    for idxs in by_stem.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    ws = ws_vec.astype(np.float64)
    ws = ws / np.where(np.linalg.norm(ws, axis=1, keepdims=True) > 0,
                       np.linalg.norm(ws, axis=1, keepdims=True), 1.0)
    sim = ws @ ws.T
    ii, jj = np.where(np.triu(sim >= threshold, k=1))
    for a, b in zip(ii.tolist(), jj.tolist()):
        union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: min(g))


class GraphIndex:
    """Entity/relation graph index over a GraphStore + VectorStore pair.

    Holds the in-memory entity/relation records (the validated retrieval
    reads them directly) while the stores carry the same data for
    persistence and protocol-level access:

    - ``entities``: [{name, members, chunks, roles, text}] — text is
      "name: role1, role2" (the terminal-eval format).
    - ``relations``: [{a, b, relation, prob, text, completed}] — a/b are
      merged-entity indices; graph node ids are ``str(index)``.
    """

    def __init__(
        self,
        graph: "GraphStore",
        vectors: "VectorStore",
        chunks: dict[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.vectors = vectors
        self.chunks: dict[str, str] = dict(chunks or {})
        self.entities: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self.canon_of: dict[str, int] = {}

    # ── assembly (phase50 LightRagIndex.__init__) ────────────────────────

    @classmethod
    def from_extractions(
        cls,
        extractions: list[ChunkExtraction],
        chunk_ids: list[str],
        relation_edges: list[dict[str, Any]],
        graph: "GraphStore",
        vectors: "VectorStore",
        chunks: dict[str, str] | None = None,
    ) -> "GraphIndex":
        """Build the index from per-chunk lens extractions + relation edges.

        ``relation_edges``: [{concept_a, concept_b, relation, prob}] as
        produced by LensProvider.extract_relation over concept pairs.
        """
        # Aggregate per-concept stats across chunks (cache["concept_chunks"],
        # cache["chunks"][cid]["roles"], vecs["ws_vec"]/["count"] of phase50).
        concept_chunks: dict[str, list[str]] = defaultdict(list)
        concept_roles: dict[str, set[str]] = defaultdict(set)
        ws_sum: dict[str, np.ndarray] = {}
        ws_n: dict[str, int] = defaultdict(int)
        ws_dim = 1
        for cid, ext in zip(chunk_ids, extractions):
            seen_in_chunk: set[str] = set()
            for concept in ext.concepts:
                key = concept.lower()
                if key not in seen_in_chunk:
                    concept_chunks[key].append(cid)
                    seen_in_chunk.add(key)
            for raw_concept, role_list in (ext.roles or {}).items():
                concept_roles[raw_concept.lower()].update(role_list)
            for concept, vec in (ext.ws_vecs or {}).items():
                key = concept.lower()
                v = np.asarray(vec, dtype=np.float64)
                ws_dim = max(ws_dim, v.shape[0])
                ws_sum[key] = ws_sum.get(key, np.zeros_like(v)) + v
                ws_n[key] += 1

        concepts = sorted(concept_chunks)
        npz_index = {c: i for i, c in enumerate(concepts)}
        ws_vec = np.zeros((len(concepts), ws_dim), dtype=np.float64)
        count = np.zeros(len(concepts), dtype=np.int64)
        for c, i in npz_index.items():
            count[i] = len(concept_chunks[c])
            if c in ws_sum:
                ws_vec[i, : ws_sum[c].shape[0]] = ws_sum[c] / max(ws_n[c], 1)

        groups = merge_entities(concepts, ws_vec)
        index = cls(graph, vectors, chunks)
        entities: list[dict[str, Any]] = []
        canon_of: dict[str, int] = {}   # original concept -> merged entity idx
        for g in groups:
            members = [concepts[i] for i in g]
            rep = max(g, key=lambda i: (int(count[i]), -i))
            ent_chunks: set[str] = set()
            roles: set[str] = set()
            for m in members:
                ent_chunks.update(concept_chunks.get(m, []))
                roles.update(concept_roles.get(m, set()))
            entities.append({
                "name": concepts[rep], "members": members,
                "chunks": sorted(ent_chunks), "roles": sorted(roles),
                "text": "",
            })
            for m in members:
                canon_of[m] = len(entities) - 1
        for ent in entities:
            ent["text"] = (f"{ent['name']}: {', '.join(ent['roles'])}"
                           if ent["roles"] else ent["name"])

        rel_list: list[dict[str, Any]] = []
        for e in relation_edges:
            a = canon_of.get(str(e["concept_a"]).lower())
            b = canon_of.get(str(e["concept_b"]).lower())
            if a is None or b is None or a == b:
                continue
            rel = {
                "a": a, "b": b, "relation": str(e["relation"]),
                "prob": float(e["prob"]),
                "text": f"{entities[a]['name']} {e['relation']} "
                        f"{entities[b]['name']}",
                "completed": False,
            }
            rel_list.append(rel)
            graph.add_edge(str(a), str(b), weight=rel["prob"],
                           relation=rel["relation"], text=rel["text"])

        for i, ent in enumerate(entities):
            graph.add_node(str(i), name=ent["name"], text=ent["text"])

        index.entities = entities
        index.relations = rel_list
        index.canon_of = canon_of
        return index

    # ── embedding (phase50 LightRagIndex.build_embeddings) ───────────────

    @staticmethod
    def _l2norm_rows(x: np.ndarray) -> np.ndarray:
        return x / np.where(
            np.linalg.norm(x, axis=1, keepdims=True) > 0,
            np.linalg.norm(x, axis=1, keepdims=True), 1.0)

    def build_embeddings(self, embed: "EmbedProvider") -> None:
        """Embed entity/relation texts into the VectorStore (L2-normalized)."""
        if self.entities:
            ent = np.asarray(embed.embed([e["text"] for e in self.entities]),
                             dtype=np.float64)
            ent = self._l2norm_rows(ent)
            self.vectors.upsert(
                NS_ENTITIES, [str(i) for i in range(len(self.entities))],
                ent.astype(np.float32),
                payloads=[{"name": e["name"], "text": e["text"]}
                          for e in self.entities])
        if self.relations:
            rel = np.asarray(embed.embed([r["text"] for r in self.relations]),
                             dtype=np.float64)
            rel = self._l2norm_rows(rel)
            self.vectors.upsert(
                NS_RELATIONS, [str(i) for i in range(len(self.relations))],
                rel.astype(np.float32),
                payloads=[{"a": r["a"], "b": r["b"], "prob": r["prob"],
                           "text": r["text"]} for r in self.relations])

    def build_chunk_embeddings(self, embed: "EmbedProvider") -> None:
        """Embed raw chunk texts into the VectorStore (naive retrieval channel)."""
        if not self.chunks:
            return
        chunk_ids = sorted(self.chunks)
        mat = np.asarray(embed.embed([self.chunks[cid] for cid in chunk_ids]),
                         dtype=np.float64)
        mat = self._l2norm_rows(mat)
        self.vectors.upsert(
            NS_CHUNKS, chunk_ids, mat.astype(np.float32),
            payloads=[{"text": self.chunks[cid]} for cid in chunk_ids])

    # ── text-side augmentation (phase53 augment_index, co-occurrence path) ─

    def augment_with_entities(self, entity_cache: dict[str, Any]) -> dict[str, int]:
        """Attach text-side entity nodes + co-occurrence edges (in place).

        ``entity_cache``: {entity_chunks, entity_display, entity_frequency}
        as produced by text-side entity detection. Co-occurrence edges:
        per-chunk top-15 entities by global frequency, MIN_COOCC=2, caps of
        MAX_EE_EDGES / MAX_EC_EDGES, relation="related_to", prob = count
        normalized by the max count. Returns stats.
        """
        base_n = len(self.entities)
        ent_chunks = entity_cache["entity_chunks"]
        display = entity_cache.get("entity_display", {})
        ent_freq = entity_cache.get("entity_frequency", {})
        new_idx: dict[str, int] = {}

        for name in sorted(ent_chunks):
            new_idx[name] = len(self.entities)
            self.entities.append({
                "name": display.get(name, name), "members": [name],
                "chunks": sorted(ent_chunks[name]), "roles": [],
                "text": display.get(name, name),
            })
            self.graph.add_node(str(new_idx[name]),
                                name=display.get(name, name),
                                text=display.get(name, name))

        def _add_relation(a: int, b: int, prob: float) -> None:
            rel = {
                "a": a, "b": b, "relation": "related_to",
                "prob": prob,
                "text": f"{self.entities[a]['name']} related to "
                        f"{self.entities[b]['name']}",
                "completed": True,
            }
            self.relations.append(rel)
            self.graph.add_edge(str(a), str(b), weight=prob,
                                relation="related_to", text=rel["text"])

        # Entity-entity edges: per-chunk pairs (top-15 by global frequency
        # per chunk to control the combinatorial blow-up).
        cooc: Counter = Counter()
        cid_ents: dict[str, list[str]] = defaultdict(list)
        for name, cids in ent_chunks.items():
            for cid in cids:
                cid_ents[cid].append(name)
        for cid, names in cid_ents.items():
            names = sorted(names, key=lambda n: -ent_freq.get(n, 0))[:15]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    cooc[(names[i], names[j])] += 1
        pairs = [(c, a, b) for (a, b), c in cooc.items() if c >= MIN_COOCC]
        pairs.sort(key=lambda x: -x[0])
        pairs = pairs[:MAX_EE_EDGES]
        c_max = pairs[0][0] if pairs else 1
        n_ee = 0
        for c, a, b in pairs:
            _add_relation(new_idx[a], new_idx[b], c / c_max)
            n_ee += 1

        # Entity-concept edges: expand per chunk (avoids entity x concept
        # full-pair enumeration).
        chunk_concepts: dict[str, list[int]] = defaultdict(list)
        for ci in range(base_n):
            for cid in self.entities[ci]["chunks"]:
                chunk_concepts[cid].append(ci)
        ec_cooc: Counter = Counter()
        for cid, names in cid_ents.items():
            cis = chunk_concepts.get(cid)
            if not cis:
                continue
            for name in names:
                for ci in cis:
                    ec_cooc[(name, ci)] += 1
        ec_items = [(k, v) for k, v in ec_cooc.items() if v >= MIN_COOCC]
        ec_pairs = sorted(ec_items, key=lambda x: -x[1])[:MAX_EC_EDGES]
        ec_max = ec_pairs[0][1] if ec_pairs else 1
        for (name, ci), ov in ec_pairs:
            _add_relation(new_idx[name], ci, ov / ec_max)

        return {"n_base_entities": base_n, "n_new_entities": len(new_idx),
                "n_ee_edges": n_ee, "n_ec_edges": len(ec_pairs)}

    # ── retrieval entry point (delegates to jgraphrag.retrieve) ──────────

    def retrieve(
        self, query: str, top_k: int = 10, embed: "EmbedProvider | None" = None,
    ) -> list[str]:
        """Dual-level hybrid retrieval → ranked chunk ids (len <= top_k)."""
        from .retrieve import TOP_K, interleave_rankings, lightrag_retrieve

        if embed is None:
            from .providers.bge_m3 import CachedBgeM3Provider
            embed = CachedBgeM3Provider()

        qv = np.asarray(embed.embed([query])[0], dtype=np.float64)
        qv = qv / (np.linalg.norm(qv) + 1e-12)

        # Naive channel: full-corpus bge sims (backfill + tie-break).
        n_chunks = self.vectors.count(NS_CHUNKS)
        chunk_hits = self.vectors.query(NS_CHUNKS, qv, n_chunks) if n_chunks else []
        bge_sim = {cid: s for cid, s in chunk_hits}
        b0_ids = [cid for cid, _s in chunk_hits[:top_k]]

        scores, _debug = lightrag_retrieve(self, qv)
        graph_ranked = [cid for cid, _s in sorted(
            scores.items(), key=lambda x: (x[1], bge_sim.get(x[0], 0.0)),
            reverse=True)]
        return interleave_rankings(graph_ranked, b0_ids, top_k=top_k or TOP_K)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, dir: str | Path) -> None:
        """Persist both stores + entity/relation/chunk records to ``dir``."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        self.vectors.save(dir)
        self.graph.save(dir)
        meta = {
            "entities": self.entities,
            "relations": self.relations,
            "canon_of": self.canon_of,
            "chunks": self.chunks,
        }
        (dir / _META_JSON).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(
        cls,
        dir: str | Path,
        graph: "GraphStore | None" = None,
        vectors: "VectorStore | None" = None,
    ) -> "GraphIndex":
        """Load an index saved by ``save``. Defaults to the local stores."""
        dir = Path(dir)
        if vectors is None:
            from .stores.local import NumpyVectorStore
            vectors = NumpyVectorStore.load(dir)
        if graph is None:
            from .stores.local import JsonGraphStore
            graph = JsonGraphStore.load(dir)
        meta_path = dir / _META_JSON
        if not meta_path.exists():
            raise FileNotFoundError(f"GraphIndex meta not found: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        index = cls(graph, vectors, meta.get("chunks"))
        index.entities = meta["entities"]
        index.relations = meta["relations"]
        index.canon_of = meta.get("canon_of", {})
        return index
