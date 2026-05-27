"""Monte Carlo Tree Search on a small abstract decision tree.

The 'game': from each leaf you get a hidden reward in [0, 1]. The agent has to
figure out which root action is best by running rollouts. This is a stand-in
for the kind of tree LATS / AlphaCode-style agents explore.

Tree shape: branching factor 3, depth 3 (so 27 leaves). Rewards baked in with
a seed so the trace is reproducible.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

BRANCH = 3
DEPTH = 3
C_UCB = math.sqrt(2)


@dataclass
class Node:
    id: int
    parent: int | None
    depth: int
    action: int | None  # index taken from parent
    visits: int = 0
    value_sum: float = 0.0
    children: list[int] = field(default_factory=list)

    @property
    def avg_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def _build_full_tree(seed: int) -> tuple[list[Node], dict[int, float]]:
    """Build the full implicit tree of decisions and assign hidden leaf rewards."""
    rng = random.Random(seed)
    nodes: list[Node] = [Node(id=0, parent=None, depth=0, action=None)]
    rewards: dict[int, float] = {}

    def build(parent_id: int):
        p = nodes[parent_id]
        if p.depth == DEPTH:
            # leaf reward biased so some paths are clearly better than others
            base = 0.3 + 0.7 * rng.random()
            rewards[p.id] = round(base, 3)
            return
        for a in range(BRANCH):
            n = Node(id=len(nodes), parent=parent_id, depth=p.depth + 1, action=a)
            p.children.append(n.id)
            nodes.append(n)
            build(n.id)

    build(0)
    return nodes, rewards


def _ucb(child: Node, parent_visits: int) -> float:
    if child.visits == 0:
        return float("inf")
    return child.avg_value + C_UCB * math.sqrt(math.log(parent_visits) / child.visits)


def _select(nodes: list[Node], visited_id: int, log: list[dict]) -> int:
    """Descend by UCB until we reach a node that has unvisited children or is a leaf."""
    cur = nodes[visited_id]
    while cur.children:
        unvisited = [c for c in cur.children if nodes[c].visits == 0]
        if unvisited:
            chosen = unvisited[0]
            log.append({"action": "select", "from": cur.id, "to": chosen, "reason": "child has 0 visits (UCB=inf)"})
            return chosen
        scored = [(c, _ucb(nodes[c], cur.visits)) for c in cur.children]
        scored.sort(key=lambda x: -x[1])
        chosen = scored[0][0]
        log.append({"action": "descend", "from": cur.id, "to": chosen,
                    "ucb_scores": {c: round(s, 3) for c, s in scored},
                    "reason": f"pick child with highest UCB ({round(scored[0][1],3)})"})
        cur = nodes[chosen]
    return cur.id


def _rollout(nodes: list[Node], rewards: dict[int, float], start_id: int, rng: random.Random, log: list[dict]) -> float:
    cur = nodes[start_id]
    path = [cur.id]
    while cur.children:
        c = rng.choice(cur.children)
        cur = nodes[c]
        path.append(cur.id)
    reward = rewards[cur.id]
    log.append({"action": "rollout", "from": start_id, "path": path, "leaf": cur.id, "reward": reward,
                "reason": f"random play to leaf {cur.id}, reward={reward}"})
    return reward


def _backprop(nodes: list[Node], leaf_id: int, reward: float, log: list[dict]):
    chain = []
    cur: int | None = leaf_id
    while cur is not None:
        n = nodes[cur]
        n.visits += 1
        n.value_sum += reward
        chain.append(cur)
        cur = n.parent
    log.append({"action": "backprop", "chain": chain, "reward": reward,
                "reason": f"update visits/value along path back to root"})


def _snapshot(nodes: list[Node]) -> list[dict]:
    return [{"id": n.id, "parent": n.parent, "depth": n.depth, "action": n.action,
             "visits": n.visits, "value_sum": round(n.value_sum, 3), "avg": round(n.avg_value, 3)} for n in nodes]


def run(iterations: int = 20, seed: int = 7) -> dict:
    nodes, rewards = _build_full_tree(seed)
    rng = random.Random(seed + 1)
    iterations_log: list[dict] = []

    for it in range(iterations):
        step_log: list[dict] = []
        selected = _select(nodes, 0, step_log)
        # in this fixed-tree variant we don't need explicit expansion;
        # the tree is already realized so 'expansion' is the first time we visit a node.
        if nodes[selected].visits == 0 and not nodes[selected].children:
            step_log.append({"action": "expand", "node": selected, "reason": "first time at this leaf"})
        reward = _rollout(nodes, rewards, selected, rng, step_log)
        _backprop(nodes, selected, reward, step_log)
        iterations_log.append({"iteration": it + 1, "log": step_log, "snapshot": _snapshot(nodes)})

    # best root action by average value at depth-1 children
    root = nodes[0]
    root_children_stats = [{"id": c, "action": nodes[c].action, "visits": nodes[c].visits, "avg": round(nodes[c].avg_value, 3)} for c in root.children]
    root_children_stats.sort(key=lambda x: -x["avg"])
    return {
        "iterations": iterations,
        "seed": seed,
        "branching": BRANCH,
        "depth": DEPTH,
        "log_per_iter": iterations_log,
        "root_action_summary": root_children_stats,
        "rewards": rewards,
    }
