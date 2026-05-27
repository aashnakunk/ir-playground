# A* (A-Star)

BFS with a compass. At every step, A* picks the unexplored cell with the
smallest **f = g + h**, where g is the cost so far and h is an *estimated*
remaining cost to the goal.

## Intuition

GPS routing. Not "what's the closest road I haven't tried" but "what's the
closest road that's *also* pointing me toward my destination." The estimate
(h) keeps the search from wandering.

## How it works

```
open_set = priority_queue([start], key=f)
g[start] = 0
while open_set:
    current = pop_min_f(open_set)
    if current == goal: return reconstruct_path()
    for neighbor in neighbors(current):
        tentative_g = g[current] + cost(current, neighbor)
        if tentative_g < g.get(neighbor, ∞):
            g[neighbor] = tentative_g
            f[neighbor] = tentative_g + h(neighbor)
            push(open_set, neighbor, key=f[neighbor])
```

- **g** = actual cost from start to here.
- **h** = heuristic estimate of cost from here to goal (Manhattan distance for
  grids, straight-line for maps).
- **f = g + h** = best-case total cost through this cell.

## The heuristic matters

- **h = 0** → A* degenerates to Dijkstra (still optimal, but explores everything).
- **h = true cost** → A* explores only the path. Magic but unrealistic.
- **h is admissible** (never overestimates) → A* is optimal.
- **h > true cost** ("greedy A*") → fast but path may be suboptimal.

The demo's heuristic-weight slider lets you walk this trade-off live.

## When to use

- Pathfinding in games, robotics, maps.
- Any planning problem where you have a reasonable "distance to goal"
  estimate.

## When not to use

- You have no good heuristic → just use BFS or Dijkstra.
- The search space is so large that even A* can't fit. Then [[MCTS]] or
  [[Beam Search]] (which sacrifice optimality for tractability).

## Where it shows up in modern AI

- **LLM agents** that search over plans where g = tokens-used so far and h =
  an LLM-estimated remaining-cost (or value).
- Robotics, game AI, route planning — the classic uses are still alive.
- Pieces of A* show up in compiler optimization and constraint solvers.

## Related

- [[BFS-DFS]] — predecessors
- [[Beam Search]] — width-limited search without admissibility guarantees
- [[MCTS]] — alternative for huge search spaces
