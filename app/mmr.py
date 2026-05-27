"""Maximal Marginal Relevance re-ranking.

Given top-N semantic candidates, iteratively pick the one maximizing
    lambda * sim(query, d) - (1 - lambda) * max sim(d, d_already_picked).

lambda=1 -> pure relevance (same as semantic order).
lambda=0 -> pure diversity (least similar to what we've already picked).
"""
from __future__ import annotations

import numpy as np

from app.embeddings import corpus_embeddings, embed_query, cosine


def search(query: str, top_k: int = 5, candidate_k: int = 15, lam: float = 0.5) -> dict:
    docs, mat = corpus_embeddings()
    qv = embed_query(query)
    sims = cosine(qv, mat)
    # candidate pool: top-N by relevance
    cand_idx = np.argsort(-sims)[:candidate_k].tolist()

    # normalize doc vectors once
    norm_mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    rel = {i: float(sims[i]) for i in cand_idx}

    selected: list[int] = []
    steps: list[dict] = []
    remaining = list(cand_idx)

    while remaining and len(selected) < top_k:
        scored = []
        for i in remaining:
            if not selected:
                div_pen = 0.0
                worst_neighbor = None
            else:
                sims_to_sel = norm_mat[selected] @ norm_mat[i]
                worst = int(np.argmax(sims_to_sel))
                div_pen = float(sims_to_sel[worst])
                worst_neighbor = selected[worst]
            mmr = lam * rel[i] - (1 - lam) * div_pen
            scored.append({
                "doc_idx": i, "id": int(docs[i]["id"]), "title": docs[i]["title"],
                "relevance": rel[i], "redundancy": div_pen,
                "worst_neighbor": (None if worst_neighbor is None else int(docs[worst_neighbor]["id"])),
                "mmr": mmr,
            })
        scored.sort(key=lambda x: -x["mmr"])
        winner = scored[0]
        steps.append({
            "step": len(selected) + 1,
            "picked": winner,
            "considered": scored[:6],
            "selected_so_far": [int(docs[s]["id"]) for s in selected],
        })
        selected.append(winner["doc_idx"])
        remaining.remove(winner["doc_idx"])

    # baseline pure-semantic order over same candidate pool for comparison
    semantic_order = [{"rank": r+1, "id": int(docs[i]["id"]), "title": docs[i]["title"], "score": rel[i]}
                      for r, i in enumerate(cand_idx[:top_k])]
    mmr_order = [{"rank": r+1, "id": int(docs[i]["id"]), "title": docs[i]["title"], "score": rel[i]}
                 for r, i in enumerate(selected)]
    return {
        "query": query,
        "lambda": lam,
        "candidate_pool_size": len(cand_idx),
        "top_k": top_k,
        "semantic_order": semantic_order,
        "mmr_order": mmr_order,
        "steps": steps,
    }
