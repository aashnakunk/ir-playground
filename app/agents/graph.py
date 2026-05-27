"""BFS and DFS on a small hand-crafted graph.

The graph models a tiny 'Tree of Thoughts' style decision space: each node is a
labelled reasoning step, edges connect candidate next steps. Same graph is used
for both algos so the user can compare frontier behavior side-by-side.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


NODES: list[dict] = [
    {"id": "A", "label": "start: question",        "x": 100, "y": 250},
    {"id": "B", "label": "break it down",          "x": 230, "y": 130},
    {"id": "C", "label": "look up facts",          "x": 230, "y": 250},
    {"id": "D", "label": "ask the user",           "x": 230, "y": 370},
    {"id": "E", "label": "sub-claim 1",            "x": 380, "y": 70},
    {"id": "F", "label": "sub-claim 2",            "x": 380, "y": 190},
    {"id": "G", "label": "doc lookup",             "x": 380, "y": 280},
    {"id": "H", "label": "clarifying q.",          "x": 380, "y": 370},
    {"id": "I", "label": "synthesize",             "x": 540, "y": 130},
    {"id": "J", "label": "final answer (goal)",    "x": 700, "y": 200},
]

EDGES: list[tuple[str, str]] = [
    ("A", "B"), ("A", "C"), ("A", "D"),
    ("B", "E"), ("B", "F"),
    ("C", "F"), ("C", "G"),
    ("D", "H"),
    ("E", "I"), ("F", "I"), ("G", "I"), ("H", "I"),
    ("I", "J"),
]


def _neighbors() -> dict[str, list[str]]:
    n: dict[str, list[str]] = {nd["id"]: [] for nd in NODES}
    for a, b in EDGES:
        n[a].append(b)
        n[b].append(a)
    # deterministic neighbor order for stable trace
    return {k: sorted(v) for k, v in n.items()}


def graph_def() -> dict:
    return {"nodes": NODES, "edges": [{"a": a, "b": b} for a, b in EDGES]}


def _trace(algo: str, start: str, goal: str) -> list[dict]:
    nbrs = _neighbors()
    visited: set[str] = set()
    parents: dict[str, str | None] = {start: None}
    events: list[dict] = []

    if algo == "bfs":
        frontier: deque[str] = deque([start])
        events.append({"event": "enqueue", "node": start, "frontier": list(frontier), "visited": [], "reason": "start node enqueued"})
        while frontier:
            cur = frontier.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            events.append({"event": "visit", "node": cur, "frontier": list(frontier), "visited": list(visited), "reason": f"dequeued (FIFO) {cur}"})
            if cur == goal:
                events.append({"event": "found", "node": cur, "frontier": list(frontier), "visited": list(visited), "reason": f"goal {goal} reached", "path": _reconstruct(parents, goal)})
                return events
            for nb in nbrs[cur]:
                if nb not in visited and nb not in frontier:
                    parents[nb] = cur
                    frontier.append(nb)
                    events.append({"event": "enqueue", "node": nb, "frontier": list(frontier), "visited": list(visited), "reason": f"add neighbor of {cur} to back of queue"})
    else:  # dfs
        stack: list[str] = [start]
        events.append({"event": "push", "node": start, "frontier": list(stack), "visited": [], "reason": "start node pushed"})
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            events.append({"event": "visit", "node": cur, "frontier": list(stack), "visited": list(visited), "reason": f"popped (LIFO) {cur}"})
            if cur == goal:
                events.append({"event": "found", "node": cur, "frontier": list(stack), "visited": list(visited), "reason": f"goal {goal} reached", "path": _reconstruct(parents, goal)})
                return events
            for nb in reversed(nbrs[cur]):  # reversed so first neighbor explored first
                if nb not in visited:
                    parents.setdefault(nb, cur)
                    stack.append(nb)
                    events.append({"event": "push", "node": nb, "frontier": list(stack), "visited": list(visited), "reason": f"push neighbor of {cur} onto stack"})
    events.append({"event": "exhausted", "node": None, "frontier": [], "visited": list(visited), "reason": "frontier empty; goal not reachable"})
    return events


def _reconstruct(parents: dict[str, str | None], goal: str) -> list[str]:
    path = []
    cur: str | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents.get(cur)
    return list(reversed(path))


def search(algo: str, start: str = "A", goal: str = "J") -> dict:
    if algo not in ("bfs", "dfs"):
        raise ValueError("algo must be 'bfs' or 'dfs'")
    return {
        "algo": algo, "start": start, "goal": goal,
        "graph": graph_def(),
        "events": _trace(algo, start, goal),
    }
