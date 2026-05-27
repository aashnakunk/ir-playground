"""A tiny, educational HNSW implementation.

Why from-scratch when SPEC says 'use libraries'? Because hnswlib's Python
binding doesn't expose the per-layer neighbor graph, and the whole point of
the HNSW page is to *show* that graph. So we keep this implementation small
(~120 LOC), deterministic, and easy to instrument.

Reference: Malkov & Yashunin, 'Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs' (2016).
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from app.corpus import load_corpus
from app.embeddings import corpus_embeddings, embed_query


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = a / (np.linalg.norm(a) + 1e-12)
    nb = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - na @ nb)


@dataclass
class HNSW:
    M: int = 8
    ef_construction: int = 16
    seed: int = 7
    dist: Callable[[np.ndarray, np.ndarray], float] = cos_dist

    vectors: list[np.ndarray] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)       # external id per internal index
    levels: list[int] = field(default_factory=list)       # max layer per node
    neighbors: list[list[list[int]]] = field(default_factory=list)  # neighbors[node][layer] = [node...]
    entry: int = -1
    top_layer: int = -1

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.mL = 1.0 / math.log(max(self.M, 2))

    def _random_level(self) -> int:
        return int(-math.log(self.rng.random() + 1e-12) * self.mL)

    def _search_layer(self, q: np.ndarray, entry_points: list[int], ef: int, layer: int, visited_log: list | None = None) -> list[tuple[float, int]]:
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []
        results: list[tuple[float, int]] = []
        for ep in entry_points:
            d = self.dist(q, self.vectors[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))
            if visited_log is not None:
                visited_log.append({"layer": layer, "node": ep, "dist": d})

        while candidates:
            d, c = heapq.heappop(candidates)
            worst = -results[0][0] if results else float("inf")
            if d > worst and len(results) >= ef:
                break
            for nb in self.neighbors[c][layer]:
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self.dist(q, self.vectors[nb])
                if visited_log is not None:
                    visited_log.append({"layer": layer, "node": nb, "dist": dn})
                if len(results) < ef or dn < (-results[0][0]):
                    heapq.heappush(candidates, (dn, nb))
                    heapq.heappush(results, (-dn, nb))
                    if len(results) > ef:
                        heapq.heappop(results)
        return [(-d, n) for d, n in sorted(results, reverse=True)]

    def _select_neighbors(self, candidates: list[tuple[float, int]], M: int) -> list[int]:
        return [n for _, n in heapq.nsmallest(M, candidates)]

    def insert(self, vec: np.ndarray, label: int) -> dict:
        node = len(self.vectors)
        lvl = self._random_level()
        self.vectors.append(vec)
        self.labels.append(label)
        self.levels.append(lvl)
        self.neighbors.append([[] for _ in range(lvl + 1)])
        trace: dict = {"node": node, "label": label, "level": lvl, "links": []}

        if self.entry < 0:
            self.entry = node
            self.top_layer = lvl
            return trace

        cur = [self.entry]
        for layer in range(self.top_layer, lvl, -1):
            cur = [self._search_layer(vec, cur, 1, layer)[0][1]]

        for layer in range(min(lvl, self.top_layer), -1, -1):
            nearest = self._search_layer(vec, cur, self.ef_construction, layer)
            chosen = self._select_neighbors(nearest, self.M)
            self.neighbors[node][layer] = chosen
            trace["links"].append({"layer": layer, "neighbors": chosen})
            for nb in chosen:
                self.neighbors[nb][layer].append(node)
                # prune neighbors if degree > M
                if len(self.neighbors[nb][layer]) > self.M:
                    nb_vec = self.vectors[nb]
                    candidates = [(self.dist(nb_vec, self.vectors[x]), x) for x in self.neighbors[nb][layer]]
                    self.neighbors[nb][layer] = self._select_neighbors(candidates, self.M)
            cur = [n for _, n in nearest]

        if lvl > self.top_layer:
            self.entry = node
            self.top_layer = lvl
        return trace

    def search(self, q: np.ndarray, top_k: int = 5, ef: int = 16) -> tuple[list[tuple[float, int]], list[dict]]:
        if self.entry < 0:
            return [], []
        visited_log: list[dict] = []
        cur = [self.entry]
        for layer in range(self.top_layer, 0, -1):
            res = self._search_layer(q, cur, 1, layer, visited_log)
            cur = [res[0][1]]
        res = self._search_layer(q, cur, max(ef, top_k), 0, visited_log)
        res = sorted(res)[:top_k]
        return res, visited_log


_built: dict[tuple, dict] = {}


def build(M: int = 8, ef_construction: int = 16, seed: int = 7) -> dict:
    """Build an HNSW over the jazz corpus embeddings. Returns the graph for the viz."""
    key = (M, ef_construction, seed)
    if key in _built:
        return _built[key]
    docs, mat = corpus_embeddings()
    index = HNSW(M=M, ef_construction=ef_construction, seed=seed)
    inserts = []
    for i, doc in enumerate(docs):
        trace = index.insert(mat[i], doc["id"])
        inserts.append(trace)

    # Snapshot graph
    nodes = []
    edges_by_layer: dict[int, list[dict]] = {}
    for node_idx, (label, lvl, nbrs) in enumerate(zip(index.labels, index.levels, index.neighbors)):
        doc = next(d for d in docs if d["id"] == label)
        nodes.append({"node": node_idx, "id": label, "title": doc["title"], "level": lvl})
        for layer, ns in enumerate(nbrs):
            for n2 in ns:
                if node_idx < n2:
                    edges_by_layer.setdefault(layer, []).append({"a": node_idx, "b": n2})

    snapshot = {
        "params": {"M": M, "ef_construction": ef_construction, "seed": seed, "top_layer": index.top_layer, "entry": index.entry, "N": len(docs)},
        "nodes": nodes,
        "edges_by_layer": [{"layer": L, "edges": edges_by_layer.get(L, [])} for L in range(index.top_layer + 1)],
        "inserts": inserts,
    }
    _built[key] = {"index": index, "snapshot": snapshot}
    return _built[key]


def graph(M: int = 8, ef_construction: int = 16, seed: int = 7) -> dict:
    return build(M, ef_construction, seed)["snapshot"]


def search(query: str, M: int = 8, ef_construction: int = 16, ef_search: int = 16, top_k: int = 5, seed: int = 7) -> dict:
    state = build(M, ef_construction, seed)
    index: HNSW = state["index"]
    qv = embed_query(query)
    res, visited = index.search(qv, top_k=top_k, ef=ef_search)
    docs = load_corpus()
    results = []
    for rank, (d, node_idx) in enumerate(res, 1):
        doc = next(x for x in docs if x["id"] == index.labels[node_idx])
        results.append({"rank": rank, "node": node_idx, "id": doc["id"], "title": doc["title"], "distance": d})
    return {
        "query": query,
        "params": {"M": M, "ef_construction": ef_construction, "ef_search": ef_search, "top_k": top_k},
        "results": results,
        "visited": visited,  # ordered list of {layer, node, dist} - the traversal path
        "entry": index.entry,
        "top_layer": index.top_layer,
    }
