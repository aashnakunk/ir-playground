"""CAG: stuff the entire corpus into the system prompt and generate.

OpenAI prompt caching kicks in automatically for stable prefixes >= 1024 tokens.
After the first call, subsequent calls with the same system prompt see lower
TTFT and cached input tokens are billed at ~50% rate. We surface
`cached_tokens` from the API response so the user can see when caching hits.
"""
from __future__ import annotations

import time
from functools import lru_cache

from app import chat
from app.corpus import load_corpus


SYSTEM_PREFIX = """You are a helpful jazz tutor. The user will ask questions about jazz. Answer using ONLY the corpus of documents below. If the answer isn't in the corpus, say so. Cite docs by their [id] tag like [10].

JAZZ CORPUS:
"""


@lru_cache(maxsize=1)
def system_prompt() -> str:
    docs = load_corpus()
    body = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for d in docs)
    return SYSTEM_PREFIX + body


def answer(question: str) -> dict:
    sp = system_prompt()
    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    resp = chat.client().chat.completions.create(
        model=chat.model(),
        max_completion_tokens=600,
        messages=messages,
    )
    elapsed = time.time() - t0
    cached = 0
    try:
        cached = resp.usage.prompt_tokens_details.cached_tokens or 0  # type: ignore[attr-defined]
    except Exception:
        pass
    return {
        "question": question,
        "system_prompt_preview": sp[:600] + ("..." if len(sp) > 600 else ""),
        "system_prompt_chars": len(sp),
        "answer": resp.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "cached_prompt_tokens": cached,
        },
        "elapsed_ms": int(elapsed * 1000),
        "model": chat.model(),
    }
