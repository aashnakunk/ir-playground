"""Semantic search via OpenAI embeddings + cosine, plus a cached 2D PCA projection."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.decomposition import PCA

from app.embeddings import corpus_embeddings, embed_query, cosine


@lru_cache(maxsize=1)
def _projection() -> tuple[np.ndarray, PCA]:
    _, mat = corpus_embeddings()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(mat)
    return coords, pca


def search(query: str, top_k: int = 5) -> dict:
    docs, mat = corpus_embeddings()
    qv = embed_query(query)
    sims = cosine(qv, mat)
    ranked = np.argsort(-sims)[:top_k]

    coords, pca = _projection()
    q_xy = pca.transform(qv.reshape(1, -1))[0]

    results = []
    for rank, idx in enumerate(ranked, 1):
        results.append({
            "rank": rank,
            "id": docs[idx]["id"],
            "title": docs[idx]["title"],
            "text": docs[idx]["text"],
            "score": float(sims[idx]),
            "xy": [float(coords[idx, 0]), float(coords[idx, 1])],
        })

    return {
        "query": query,
        "results": results,
        "query_xy": [float(q_xy[0]), float(q_xy[1])],
        "all_docs": [
            {"id": d["id"], "title": d["title"], "xy": [float(coords[i, 0]), float(coords[i, 1])], "sim": float(sims[i])}
            for i, d in enumerate(docs)
        ],
        "params": {"model": "text-embedding-3-small", "dim": int(mat.shape[1]), "N": len(docs)},
    }
