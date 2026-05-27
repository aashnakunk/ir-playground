"""View-aware OpenAI chat agent.

The frontend POSTs the user's message plus a `view_state` JSON blob describing
what is currently on screen (page, query, results, params, selection). We inject
that into the system prompt so the model's answers reference exactly what the
user is looking at.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def model() -> str:
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


SYSTEM_TEMPLATE = """You are a friendly tutor embedded in 'jazzbot', a learning playground for information retrieval and RAG techniques. The user is exploring one technique at a time and can see a live visualization of it on screen.

Your job: explain what they are seeing in plain language, ground every claim in the specific numbers/docs visible on their screen (provided below as VIEW_STATE), and prefer short answers with concrete references like 'doc #7 scored 4.2 because the term "modal" had IDF 2.1'.

Never invent results, scores, or docs that aren't in VIEW_STATE. If something they ask about isn't represented, say so and suggest what they could try on screen to surface it.

VIEW_STATE (JSON):
{view_state}
"""


def answer(message: str, view_state: dict[str, Any], history: list[dict] | None = None) -> str:
    history = history or []
    system = SYSTEM_TEMPLATE.format(view_state=json.dumps(view_state, indent=2))
    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": message},
    ]
    resp = client().chat.completions.create(
        model=model(),
        max_completion_tokens=1024,
        messages=messages,
    )
    return resp.choices[0].message.content or ""
