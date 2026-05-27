# MCTS — Monte Carlo Tree Search

Build a search tree by running random rollouts and updating which paths look
promising. Powers AlphaGo / AlphaZero and modern LLM-agent planners.

## Intuition

Deciding which restaurant to try. You can't know how good each is without
going. So you randomly try meals, remember which restaurants kept being good,
and over time bias your future visits toward those. **UCB** is the rule:
"prefer good restaurants, but also occasionally check ones you haven't visited
much yet."

## How it works (four phases per iteration)

1. **Select.** Starting at the root, descend through the tree by **UCB1**:
   $$\text{UCB}(child) = \bar{v}_{child} + c \sqrt{\frac{\ln N_{parent}}{N_{child}}}$$
   First term rewards children that have paid off (exploitation); second term
   rewards children visited little (exploration). `c` is the trade-off knob,
   conventionally √2.
2. **Expand.** When you reach a leaf you haven't seen, add a new child.
3. **Simulate (rollout).** From the new node, play randomly (or with a cheap
   policy) until you reach a terminal state. Get a reward.
4. **Backpropagate.** Walk back up the path, incrementing visit counts and
   accumulating the reward at every node along the way.

Repeat for N iterations. The root's most-visited child is your best action.

## When to use

- Large search spaces with a clear reward signal (games, simulations).
- Planning problems where exact lookahead is intractable but rollouts are
  cheap.
- LLM agents that can self-evaluate intermediate plans (give a value).

## When not to use

- No reward signal — MCTS needs something to "win."
- Tiny search spaces — just enumerate or use [[A-Star]].
- Hard time budgets where you can't afford enough rollouts to converge.

## Where it shows up in modern AI

- **AlphaGo / AlphaZero / MuZero** (DeepMind) — MCTS guided by a neural net
  that gives both move priors and value estimates.
- **LATS** (Language Agent Tree Search, Zhou et al. 2023) — MCTS over LLM
  reasoning trees, with the LLM acting as both policy and value head.
- **AlphaCode** — MCTS over code generations.
- Modern reasoning agents broadly use MCTS variants.

## Trade-offs vs alternatives

- **Beam Search:** good when you have token probabilities. MCTS is better when
  you have a sparse end-state reward instead.
- **A\*:** good when you have a heuristic and a clear cost function. MCTS
  works when you don't.
- **BFS/DFS:** plain enumeration. MCTS is selective enumeration biased toward
  what's working.

## Related

- [[BFS-DFS]]
- [[A-Star]]
- [[Beam Search]]
