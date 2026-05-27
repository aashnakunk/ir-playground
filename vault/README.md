# jazzbot vault

Obsidian vault documenting the jazzbot / ir-playground project — an interactive
playground for learning information retrieval, RAG, and agent-search algorithms.

## How to open

In Obsidian: **Open another vault** → **Open folder as vault** → point at this
`vault/` directory.

## Start here

- [[Project Overview]] — what this is, what it does, why it exists
- [[Architecture]] — stack, repo layout, how the pieces fit
- [[Map of Content]] — the index of every note

## Suggested reading order

If you've never thought about IR / RAG before, walk these in order:

1. [[Boolean]] → [[TF-IDF]] → [[BM25]] (keyword search, oldest to current)
2. [[Embeddings]] → [[Cosine Similarity]] → [[Semantic Search]] (search by meaning)
3. [[Hybrid Search]] → [[HNSW]] (combining + making it fast)
4. [[RAG]] → [[CAG]] (using retrieval with LLMs)
5. [[MMR]] → [[Rerank]] → [[HyDE]] (production RAG tricks)

Then, the agent-search algorithms:

6. [[BFS-DFS]] → [[A-Star]] (graph traversal basics)
7. [[Beam Search]] → [[MCTS]] (the search algorithms inside modern LLM agents)

## Conventions in this vault

- **[[Wiki-style links]]** point to other notes. Click them.
- Algorithm notes share a structure: **intuition**, **how it works**, **formula**,
  **when to use**, **when not to use**, **where it shows up**, **related**.
- "Where it shows up" calls out real production systems and recent papers, so
  the notes stay grounded.
- Notes are short on purpose. The interactive demo in the running app is where
  intuition really lands — the vault is for the conceptual scaffold.
