"""LLM-based re-ranker (poor man's cross-encoder).

A true cross-encoder is a small BERT trained to score (query, doc) pairs. Here
we use the same chat model as a relevance judge — slower and pricier but works
without an extra model dependency, and demonstrates the *concept*: a heavier
model re-orders the cheap retriever's top-N.
"""
from __future__ import annotations

import json
import re
import time

from app import chat
from app.retrievers import semantic


SCORING_SYSTEM = (
    "You are a strict relevance judge. For each candidate document, score how well it answers the user's query "
    "on a scale of 0 (irrelevant) to 10 (perfect direct answer). Respond ONLY with a JSON array like "
    "[{\"id\": 7, \"score\": 8}, {\"id\": 3, \"score\": 2}]. No prose."
)


def _score_batch(query: str, candidates: list[dict]) -> dict[int, int]:
    listing = "\n".join(f"[{c['id']}] {c['title']}\n{c['text']}" for c in candidates)
    user = f"QUERY: {query}\n\nCANDIDATES:\n{listing}"
    resp = chat.client().chat.completions.create(
        model=chat.model(),
        max_completion_tokens=400,
        messages=[
            {"role": "system", "content": SCORING_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or "[]"
    # tolerate fenced code blocks
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        parsed = json.loads(text)
        return {int(item["id"]): int(item["score"]) for item in parsed}
    except Exception:
        return {}


def run(query: str, top_k: int = 5, candidate_k: int = 15) -> dict:
    t0 = time.time()
    base = semantic.search(query, top_k=candidate_k)["results"]
    t_retrieve = time.time() - t0

    t1 = time.time()
    scores = _score_batch(query, base)
    t_rerank = time.time() - t1

    before = [{"rank": r["rank"], "id": r["id"], "title": r["title"], "score": r["score"]} for r in base[:top_k]]
    enriched = []
    for r in base:
        s = scores.get(r["id"])
        enriched.append({
            "id": r["id"], "title": r["title"],
            "semantic_rank": r["rank"], "semantic_score": r["score"],
            "llm_score": s if s is not None else -1,
        })
    enriched.sort(key=lambda x: (-x["llm_score"], x["semantic_rank"]))
    for i, e in enumerate(enriched, 1):
        e["rerank_rank"] = i
    return {
        "query": query,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "before": before,
        "after": enriched[:top_k],
        "all_scored": enriched,
        "timings_ms": {
            "retrieve": int(t_retrieve * 1000),
            "llm_rerank": int(t_rerank * 1000),
        },
    }
