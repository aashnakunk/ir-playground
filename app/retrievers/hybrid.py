"""Hybrid retrieval: BM25 + semantic, fused with Reciprocal Rank Fusion (RRF).

RRF score(d) = sum over retrievers r of 1 / (k + rank_r(d)).
We use k=60 (the value from the original Cormack et al. paper).
"""
from __future__ import annotations

from app.retrievers import bm25, semantic

RRF_K = 60


def search(query: str, top_k: int = 5, candidate_k: int = 10) -> dict:
    bm25_res = bm25.search(query, top_k=candidate_k)
    sem_res = semantic.search(query, top_k=candidate_k)

    bm25_rank = {r["id"]: r["rank"] for r in bm25_res["results"]}
    sem_rank = {r["id"]: r["rank"] for r in sem_res["results"]}
    by_id = {r["id"]: r for r in bm25_res["results"]}
    for r in sem_res["results"]:
        by_id.setdefault(r["id"], r)

    ids = set(bm25_rank) | set(sem_rank)
    fused = []
    for doc_id in ids:
        br = bm25_rank.get(doc_id)
        sr = sem_rank.get(doc_id)
        rrf = 0.0
        if br is not None:
            rrf += 1.0 / (RRF_K + br)
        if sr is not None:
            rrf += 1.0 / (RRF_K + sr)
        fused.append({
            "id": doc_id,
            "title": by_id[doc_id]["title"],
            "text": by_id[doc_id]["text"],
            "bm25_rank": br,
            "semantic_rank": sr,
            "rrf_score": rrf,
        })
    fused.sort(key=lambda x: -x["rrf_score"])
    for i, r in enumerate(fused[:top_k], 1):
        r["rank"] = i

    return {
        "query": query,
        "rrf_k": RRF_K,
        "bm25": [{"rank": r["rank"], "id": r["id"], "title": r["title"], "score": r["score"]} for r in bm25_res["results"]],
        "semantic": [{"rank": r["rank"], "id": r["id"], "title": r["title"], "score": r["score"]} for r in sem_res["results"]],
        "fused": fused[:top_k],
    }
