"""BM25 retriever with per-term score breakdowns for visualization.

We wrap rank_bm25.BM25Okapi and additionally compute, for each (query term, doc)
pair, the contribution that term makes to the doc's total score. The page uses
this to draw a stacked bar chart so the user can see *why* a doc scored what it
did.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.corpus import load_corpus

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=1)
def _index():
    docs = load_corpus()
    tokenized = [tokenize(d["title"] + " " + d["text"]) for d in docs]
    bm25 = BM25Okapi(tokenized)
    return docs, tokenized, bm25


def _term_score(bm25: BM25Okapi, term: str, doc_idx: int, doc_tokens: list[str]) -> float:
    """Reproduce the per-term BM25 contribution that rank_bm25 sums internally."""
    if term not in bm25.idf:
        return 0.0
    idf = bm25.idf[term]
    tf = doc_tokens.count(term)
    if tf == 0:
        return 0.0
    k1, b = bm25.k1, bm25.b
    dl = bm25.doc_len[doc_idx]
    avgdl = bm25.avgdl
    numerator = tf * (k1 + 1)
    denominator = tf + k1 * (1 - b + b * dl / avgdl)
    return idf * numerator / denominator


def search(query: str, top_k: int = 5) -> dict:
    docs, tokenized, bm25 = _index()
    q_terms = tokenize(query)
    scores = bm25.get_scores(q_terms)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for rank, idx in enumerate(ranked, 1):
        term_scores = {t: _term_score(bm25, t, idx, tokenized[idx]) for t in q_terms}
        results.append({
            "rank": rank,
            "id": docs[idx]["id"],
            "title": docs[idx]["title"],
            "text": docs[idx]["text"],
            "score": float(scores[idx]),
            "term_scores": {t: float(s) for t, s in term_scores.items()},
            "doc_len": int(bm25.doc_len[idx]),
        })

    return {
        "query": query,
        "query_terms": q_terms,
        "results": results,
        "params": {"k1": bm25.k1, "b": bm25.b, "avgdl": float(bm25.avgdl), "N": len(docs)},
        "idf": {t: float(bm25.idf.get(t, 0.0)) for t in q_terms},
    }
