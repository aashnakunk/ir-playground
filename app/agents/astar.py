"""A* on a small grid with optional walls.

Heuristic = Manhattan distance. Step trace records: cell visited, its g/h/f,
the open-set state (sorted by f), the closed set, and the parent chain.
"""
from __future__ import annotations

import heapq

DEFAULT_W, DEFAULT_H = 14, 10

# Default walls — a U-shape so A* has to go around
DEFAULT_WALLS: list[tuple[int, int]] = [
    (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 7), (6, 7), (7, 7), (8, 7),
    (8, 6), (8, 5), (8, 4), (8, 3),
    (10, 1), (10, 2), (10, 3), (10, 4),
]


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def search(start: tuple[int, int] = (1, 1), goal: tuple[int, int] = (12, 8),
           walls: list[tuple[int, int]] | None = None,
           w: int = DEFAULT_W, h: int = DEFAULT_H, heuristic_weight: float = 1.0) -> dict:
    if walls is None:
        walls = DEFAULT_WALLS
    wall_set = {tuple(c) for c in walls}

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    g: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    h_start = manhattan(start, goal)
    heapq.heappush(open_heap, (h_start * heuristic_weight, counter, start))
    closed: set[tuple[int, int]] = set()
    events: list[dict] = []

    def snapshot_open():
        items = sorted(open_heap)[:6]
        return [{"cell": list(c), "f": round(f, 2), "g": round(g.get(c, 0), 2), "h": round(manhattan(c, goal), 2)} for f, _, c in items]

    events.append({"event": "init", "current": list(start), "f": h_start, "g": 0, "h": h_start,
                   "reason": "open set = {start}; g(start)=0, f=h(start)",
                   "open": snapshot_open(), "closed": []})

    while open_heap:
        f, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        events.append({
            "event": "visit", "current": list(current),
            "f": round(f, 2), "g": round(g[current], 2), "h": manhattan(current, goal),
            "reason": f"pop cell with smallest f = {round(f,2)}",
            "open": snapshot_open(), "closed": [list(c) for c in closed],
        })

        if current == goal:
            path = []
            cur: tuple[int, int] | None = goal
            while cur is not None:
                path.append(list(cur))
                cur = came_from.get(cur)
            path.reverse()
            events.append({"event": "found", "current": list(goal), "path": path,
                           "reason": f"goal reached. path length = {len(path)-1}",
                           "open": snapshot_open(), "closed": [list(c) for c in closed]})
            return {"start": list(start), "goal": list(goal), "w": w, "h": h, "walls": [list(c) for c in walls],
                    "events": events, "heuristic_weight": heuristic_weight}

        x, y = current
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (x + dx, y + dy)
            if not (0 <= nb[0] < w and 0 <= nb[1] < h):
                continue
            if nb in wall_set or nb in closed:
                continue
            tentative_g = g[current] + 1
            if tentative_g < g.get(nb, float("inf")):
                came_from[nb] = current
                g[nb] = tentative_g
                hn = manhattan(nb, goal)
                fn = tentative_g + heuristic_weight * hn
                counter += 1
                heapq.heappush(open_heap, (fn, counter, nb))
                events.append({
                    "event": "consider", "current": list(nb),
                    "f": round(fn, 2), "g": round(tentative_g, 2), "h": hn,
                    "reason": f"neighbor of {list(current)}: g={tentative_g} + {heuristic_weight}*h={hn} -> f={round(fn,2)}",
                    "open": snapshot_open(), "closed": [list(c) for c in closed],
                })

    events.append({"event": "exhausted", "reason": "open set empty; no path."})
    return {"start": list(start), "goal": list(goal), "w": w, "h": h, "walls": [list(c) for c in walls],
            "events": events, "heuristic_weight": heuristic_weight}
