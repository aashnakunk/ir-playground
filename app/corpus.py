import json
from functools import lru_cache
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "corpus.json"


@lru_cache(maxsize=1)
def load_corpus() -> list[dict]:
    with CORPUS_PATH.open() as f:
        return json.load(f)


def doc_by_id(doc_id: int) -> dict | None:
    for d in load_corpus():
        if d["id"] == doc_id:
            return d
    return None
