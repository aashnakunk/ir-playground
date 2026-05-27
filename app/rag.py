"""RAG: retrieve top-k semantic, inject into prompt, generate."""
from __future__ import annotations

import time

from app import chat
from app.retrievers import semantic
from app.theme import active


def _system() -> str:
    return f"You are a helpful {active().tutor_role}. Answer the user's question using ONLY the context passages provided below. If the answer is not in the context, say so. Cite passages by their [id] tag like [10]."


def _format_context(results: list[dict]) -> str:
    return "\n\n".join(f"[{r['id']}] {r['title']}\n{r['text']}" for r in results)


def answer(question: str, top_k: int = 4) -> dict:
    retrieved = semantic.search(question, top_k=top_k)["results"]
    context = _format_context(retrieved)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    system = _system()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    t0 = time.time()
    resp = chat.client().chat.completions.create(
        model=chat.model(),
        max_completion_tokens=600,
        messages=messages,
    )
    elapsed = time.time() - t0
    return {
        "question": question,
        "retrieved": [{"id": r["id"], "title": r["title"], "text": r["text"], "score": r["score"], "rank": r["rank"]} for r in retrieved],
        "system_prompt": system,
        "user_prompt": user,
        "answer": resp.choices[0].message.content or "",
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "elapsed_ms": int(elapsed * 1000),
        "model": chat.model(),
    }
