"""Boolean search with AND/OR/NOT/parentheses, backed by an inverted index.

Parser is shunting-yard. The index itself is a dict from term to sorted list of
(doc_id, positions). The page surfaces both the matches and the raw index.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.corpus import load_corpus

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_OPS = {"AND": 2, "OR": 1, "NOT": 3}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@lru_cache(maxsize=4)
def inverted_index() -> dict:
    docs = load_corpus()
    index: dict[str, dict[int, list[int]]] = {}
    for d in docs:
        for pos, tok in enumerate(tokenize(d["title"] + " " + d["text"])):
            index.setdefault(tok, {}).setdefault(d["id"], []).append(pos)
    return index


def index_summary(min_df: int = 1, top_n: int = 200) -> dict:
    idx = inverted_index()
    docs = load_corpus()
    all_ids = {d["id"] for d in docs}
    entries = []
    for term, posting in idx.items():
        df = len(posting)
        if df < min_df:
            continue
        entries.append({"term": term, "df": df, "doc_ids": sorted(posting.keys())})
    entries.sort(key=lambda e: (-e["df"], e["term"]))
    return {
        "N": len(docs),
        "vocab_size": len(idx),
        "shown": min(top_n, len(entries)),
        "entries": entries[:top_n],
        "doc_titles": {d["id"]: d["title"] for d in docs},
    }


def _tokenize_query(q: str) -> list[str]:
    """Split keeping operators uppercase and parens as their own tokens."""
    out = []
    for raw in re.findall(r"\(|\)|[A-Za-z0-9']+", q):
        u = raw.upper()
        if u in _OPS or raw in "()":
            out.append(u if u in _OPS else raw)
        else:
            out.append(raw.lower())
    return out


def _to_rpn(tokens: list[str]) -> list[str]:
    """Shunting-yard. NOT is unary right-associative."""
    out, stack = [], []
    for tok in tokens:
        if tok in _OPS:
            while stack and stack[-1] in _OPS and (
                (_OPS[stack[-1]] > _OPS[tok]) or
                (_OPS[stack[-1]] == _OPS[tok] and tok != "NOT")
            ):
                out.append(stack.pop())
            stack.append(tok)
        elif tok == "(":
            stack.append(tok)
        elif tok == ")":
            while stack and stack[-1] != "(":
                out.append(stack.pop())
            if stack and stack[-1] == "(":
                stack.pop()
        else:
            out.append(tok)
    while stack:
        out.append(stack.pop())
    return out


def _eval(rpn: list[str], all_ids: set[int]) -> tuple[set[int], list[dict]]:
    """Evaluate RPN, return (matched ids, step trace for the viz)."""
    idx = inverted_index()
    stack: list[set[int]] = []
    trace: list[dict] = []
    for tok in rpn:
        if tok in _OPS:
            if tok == "NOT":
                a = stack.pop() if stack else set()
                r = all_ids - a
                trace.append({"op": "NOT", "input_size": len(a), "result_size": len(r)})
                stack.append(r)
            else:
                b = stack.pop() if stack else set()
                a = stack.pop() if stack else set()
                if tok == "AND":
                    r = a & b
                else:
                    r = a | b
                trace.append({"op": tok, "left_size": len(a), "right_size": len(b), "result_size": len(r)})
                stack.append(r)
        else:
            posting = set(idx.get(tok, {}).keys())
            trace.append({"op": "TERM", "term": tok, "result_size": len(posting), "doc_ids": sorted(posting)[:20]})
            stack.append(posting)
    return (stack[-1] if stack else set()), trace


def search(query: str) -> dict:
    docs = load_corpus()
    all_ids = {d["id"] for d in docs}
    by_id = {d["id"]: d for d in docs}
    tokens = _tokenize_query(query)
    rpn = _to_rpn(tokens)
    matched, trace = _eval(rpn, all_ids)
    results = [{"id": i, "title": by_id[i]["title"], "text": by_id[i]["text"]} for i in sorted(matched)]
    return {
        "query": query,
        "tokens": tokens,
        "rpn": rpn,
        "trace": trace,
        "matched_count": len(matched),
        "results": results[:30],
        "N": len(docs),
    }
