"""HyDE: Hypothetical Document Embeddings.

For terse queries, plain semantic retrieval can underperform because the query
embedding and the doc embedding live in different parts of the space (one is a
question, the other is an answer). HyDE asks the LLM to write a *hypothetical*
answer first, then embeds that answer and searches with it.

Paper: Gao et al., 'Precise Zero-Shot Dense Retrieval without Relevance Labels' (2022).
"""
from __future__ import annotations

import time

import numpy as np

from app import chat
from app.embeddings import corpus_embeddings, embed_query, cosine
from app.theme import active


def _hypothetical(question: str) -> str:
    t = active()
    prompt = (
        f"You are an expert {t.tutor_role}. Write a short, factual paragraph (3-4 sentences) "
        f"that would plausibly answer this question about {t.name}. Be specific; invent details if needed. "
        f"Do NOT hedge, do NOT say 'I don't know'.\n\nQUESTION: {question}\n\nHYPOTHETICAL ANSWER:"
    )
    resp = chat.client().chat.completions.create(
        model=chat.model(),
        max_completion_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _rank(qv: np.ndarray, top_k: int) -> list[dict]:
    docs, mat = corpus_embeddings()
    sims = cosine(qv, mat)
    order = np.argsort(-sims)[:top_k]
    return [{"rank": r+1, "id": int(docs[i]["id"]), "title": docs[i]["title"], "score": float(sims[i])}
            for r, i in enumerate(order)]


def run(question: str, top_k: int = 5) -> dict:
    t0 = time.time()
    plain_qv = embed_query(question)
    plain = _rank(plain_qv, top_k)
    t_plain = time.time() - t0

    t1 = time.time()
    hypo = _hypothetical(question)
    t_llm = time.time() - t1

    t2 = time.time()
    hyde_qv = embed_query(hypo)
    hyde = _rank(hyde_qv, top_k)
    t_hyde = time.time() - t2

    return {
        "question": question,
        "hypothetical_doc": hypo,
        "plain_semantic": plain,
        "hyde": hyde,
        "timings_ms": {
            "plain_embed_and_search": int(t_plain * 1000),
            "llm_generate_hypothetical": int(t_llm * 1000),
            "hyde_embed_and_search": int(t_hyde * 1000),
        },
    }
