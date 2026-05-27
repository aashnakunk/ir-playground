# BFS / DFS

Two ways to explore any graph or tree. Same algorithm skeleton; the only
difference is whether you use a queue (BFS) or a stack (DFS) for the frontier.

## Intuition

Looking for your keys:

- **BFS** — check every spot 1 foot from where you started, then every spot 2
  feet, then 3. Wide net, level-by-level. Guarantees shortest path.
- **DFS** — pick a corner. Search it top to bottom. Then the next corner. Goes
  deep first; backs up only when stuck.

## How they work

Both share this loop:

```python
frontier = [start]
visited = set()
while frontier:
    node = frontier.pop()  # BFS: pop front (FIFO) | DFS: pop end (LIFO)
    if node in visited: continue
    visited.add(node)
    if node == goal: return path
    for neighbor in neighbors(node):
        if neighbor not in visited:
            frontier.append(neighbor)
```

BFS uses a **queue** (FIFO). DFS uses a **stack** (LIFO). That's the only
algorithmic difference.

## When to use

- **BFS** when you need the shortest path in an unweighted graph, or you want
  to explore "nearby" options first.
- **DFS** when you want to enumerate a deep solution space (puzzle solvers,
  game trees, file-system walks), or when memory matters — DFS only keeps the
  current path in memory; BFS can blow up at the wide frontier.

## Where they show up in modern AI

- **Tree of Thoughts** (Yao et al. 2023) — explore candidate next reasoning
  steps as a tree. BFS-style ToT keeps multiple thoughts alive per level;
  DFS-style commits to a branch and backtracks if it fails.
- **ReAct** agents — depth-first chains of (thought → action → observation).
- Anywhere an agent has a discrete set of "next moves" to consider, these are
  the two starting points before you reach for [[A-Star]] or [[MCTS]].

## Limits

- Neither uses information about how close a node is to the goal. If you
  *have* such an estimate, use [[A-Star]] — it explores way fewer nodes.
- Both can explode on dense graphs without good pruning.

## Related

- [[A-Star]] — BFS with a heuristic
- [[Beam Search]] — width-limited BFS, used by LLMs
- [[MCTS]] — selective tree exploration with rollouts
