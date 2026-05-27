"""TF-IDF retriever exposing per-term contributions, for visualization."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.corpus import load_corpus

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=1)
def _index():
    docs = load_corpus()
    texts = [d["title"] + " " + d["text"] for d in docs]
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9']+")
    mat = vec.fit_transform(texts)
    return docs, vec, mat


def search(query: str, top_k: int = 5) -> dict:
    docs, vec, mat = _index()
    q_vec = vec.transform([query])
    # Cosine sim with TF-IDF vectors (sklearn already L2-normalizes by default? no — TfidfVectorizer norm='l2')
    sims = (mat @ q_vec.T).toarray().ravel()
    ranked = np.argsort(-sims)[:top_k]

    vocab = vec.vocabulary_
    idf = vec.idf_
    q_terms_present = [t for t in tokenize(query) if t in vocab]
    q_terms_missing = [t for t in tokenize(query) if t not in vocab]

    results = []
    for rank, idx in enumerate(ranked, 1):
        # Per-term TF-IDF in the doc (raw, pre-norm) — sklearn stores sublinear off by default
        row = mat[idx]
        row_dict = {vec.get_feature_names_out()[j]: float(row[0, j]) for j in row.indices}
        term_scores = {t: row_dict.get(t, 0.0) for t in q_terms_present}
        results.append({
            "rank": rank,
            "id": docs[idx]["id"],
            "title": docs[idx]["title"],
            "text": docs[idx]["text"],
            "score": float(sims[idx]),
            "term_scores": term_scores,
        })

    return {
        "query": query,
        "query_terms": q_terms_present,
        "missing_terms": q_terms_missing,
        "results": results,
        "idf": {t: float(idf[vocab[t]]) for t in q_terms_present},
        "params": {"N": len(docs), "vocab_size": len(vocab)},
    }


def vocab_stats(top_n_terms: int = 60) -> dict:
    """For the heatmap: return the top-N highest-IDF terms and the small TF-IDF matrix slice."""
    docs, vec, mat = _index()
    idf = vec.idf_
    feature_names = vec.get_feature_names_out()
    order = np.argsort(-idf)
    chosen = order[:top_n_terms]
    sub = mat[:, chosen].toarray()
    return {
        "doc_ids": [d["id"] for d in docs],
        "doc_titles": [d["title"] for d in docs],
        "terms": [feature_names[i] for i in chosen],
        "matrix": sub.tolist(),
    }
