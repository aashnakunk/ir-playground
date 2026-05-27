"""Per-corpus theme configuration.

Selected by the CORPUS env var (default "jazz"). Adds the human-readable name,
the role used in tutor/RAG/CAG system prompts, and a handful of default queries
shown in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Theme:
    corpus: str          # filename stem under data/ (e.g. "jazz" -> data/jazz.json)
    name: str            # human-readable label
    tutor_role: str      # short phrase: "jazz tutor", "cuisine tutor"
    default_queries: dict[str, str]  # per page


THEMES: dict[str, Theme] = {
    "jazz": Theme(
        corpus="jazz",
        name="jazz",
        tutor_role="jazz tutor",
        default_queries={
            "bm25": "modal jazz pioneers",
            "tfidf": "bebop Charlie Parker",
            "semantic": "who turned jazz into electric music",
            "hybrid": "electric jazz fusion",
            "hnsw": "modal jazz pioneers",
            "rag": "what made bebop different from swing?",
            "cag": "what made bebop different from swing?",
            "boolean": "(modal OR cool) AND NOT bebop",
            "mmr": "miles davis",
            "hyde": "who pioneered electric jazz?",
            "rerank": "spiritual jazz saxophone",
        },
    ),
    "cuisine": Theme(
        corpus="cuisine",
        name="cuisine",
        tutor_role="culinary tutor",
        default_queries={
            "bm25": "fermented korean vegetables",
            "tfidf": "wood fired pizza oven",
            "semantic": "spicy numbing chinese food",
            "hybrid": "raw fish citrus",
            "hnsw": "fermented korean vegetables",
            "rag": "what is the difference between curry traditions?",
            "cag": "what is the difference between curry traditions?",
            "boolean": "(pasta OR pizza) AND italian",
            "mmr": "fermented foods",
            "hyde": "what makes mexican mole complex?",
            "rerank": "raw fish preparations",
        },
    ),
}


def active() -> Theme:
    key = os.environ.get("CORPUS", "jazz").lower()
    return THEMES.get(key, THEMES["jazz"])
