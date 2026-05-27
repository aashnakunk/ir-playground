# Map of Content

The index of every note in this vault.

## Project

- [[Project Overview]] — what this is, what it does
- [[Architecture]] — stack, repo layout, the view-state contract
- [[Deployment]] — Railway, env vars, embedding cache
- [[Backlog]] — open items for the next iteration

## Concepts (cross-cutting)

- [[Embeddings]] — what they are, how OpenAI makes them, how we use them
- [[Cosine Similarity]] — the distance metric for vectors
- [[Inverted Index]] — the data structure under every keyword search
- [[IDF]] — inverse document frequency, the "rare = useful" idea

## Algorithms — Foundational

- [[Boolean]] · AND/OR/NOT over an [[Inverted Index]]
- [[TF-IDF]] · count rare words, the original ranking scheme

## Algorithms — Core retrieval

- [[BM25]] · the keyword workhorse used everywhere
- [[Semantic Search]] · search by meaning via [[Embeddings]]
- [[Hybrid Search]] · BM25 + semantic, fused with RRF
- [[HNSW]] · the layered graph that makes semantic search fast

## Algorithms — RAG pipelines

- [[RAG]] · retrieve, paste into prompt, generate
- [[CAG]] · skip the lookup, cache the whole corpus
- [[MMR]] · diversity re-ranking
- [[Rerank]] · 2-stage retrieval with a smarter scorer
- [[HyDE]] · LLM-augmented query expansion

## Algorithms — Agent search

- [[BFS-DFS]] · the two ways to explore any graph
- [[A-Star]] · BFS with a heuristic
- [[Beam Search]] · how LLMs pick their next token
- [[MCTS]] · Monte Carlo Tree Search, behind AlphaGo and LATS
