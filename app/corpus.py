import json
from functools import lru_cache
from pathlib import Path

from app.theme import active

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@lru_cache(maxsize=4)
def load_corpus() -> list[dict]:
    path = DATA_DIR / f"{active().corpus}.json"
    with path.open() as f:
        return json.load(f)


def doc_by_id(doc_id: int) -> dict | None:
    for d in load_corpus():
        if d["id"] == doc_id:
            return d
    return None
