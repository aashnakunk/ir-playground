"""OpenAI embeddings with a tiny on-disk cache.

We embed the jazz corpus once with `text-embedding-3-small` (1536-dim), store
the vectors in data/embeddings.json, and reuse them on every request. Queries
are embedded live (and not cached — they vary).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI

from app.corpus import load_corpus

EMB_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings.json"
MODEL = "text-embedding-3-small"

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client().embeddings.create(model=MODEL, input=texts)
    return [d.embedding for d in resp.data]


@lru_cache(maxsize=1)
def corpus_embeddings() -> tuple[list[dict], np.ndarray]:
    """Returns (docs, matrix [N, D]) — embeds & caches on first call."""
    docs = load_corpus()
    if EMB_PATH.exists():
        cached = json.loads(EMB_PATH.read_text())
        if cached.get("model") == MODEL and cached.get("ids") == [d["id"] for d in docs]:
            mat = np.array(cached["vectors"], dtype=np.float32)
            return docs, mat
    texts = [d["title"] + ". " + d["text"] for d in docs]
    vecs = _embed_batch(texts)
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    EMB_PATH.write_text(json.dumps({
        "model": MODEL,
        "ids": [d["id"] for d in docs],
        "vectors": vecs,
    }))
    return docs, np.array(vecs, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return np.array(_embed_batch([text])[0], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: [D] or [N,D], b: [N,D] -> similarities"""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    if a.ndim == 1:
        return b @ a
    return b @ a.T
