# Project Overview

## What is jazzbot?

A browser-based, single-page-per-topic playground for understanding how modern
search and AI-agent systems actually work. You type a query, watch the
algorithm run with live visuals, and ask an embedded LLM tutor questions about
exactly what's on your screen.

The local version uses a small hand-written **jazz** corpus. The deployed
version uses a **world cuisines** corpus. Same code, different `CORPUS` env var.

## What it covers

| Category | Algorithms |
|---|---|
| Foundational | [[Boolean]], [[TF-IDF]] |
| Core retrieval | [[BM25]], [[Semantic Search]], [[Hybrid Search]], [[HNSW]] |
| RAG pipelines | [[RAG]], [[CAG]], [[MMR]], [[Rerank]], [[HyDE]] |
| Agent search | [[BFS-DFS]], [[A-Star]], [[Beam Search]], [[MCTS]] |

15 algorithms, 15 pages, one app.

## Why it exists

Most IR and RAG tutorials are either pure prose ("BM25 normalizes by document
length") or pure code (an arxiv link to a 40-line numpy implementation). The
missing piece is **seeing the algorithm think**: which docs it ranked and why,
which nodes it visited and in what order, what the candidate heap looked like
at step 7.

That's what this site is. Every page has:

- An interactive visual specific to that algorithm (heatmap / graph / map /
  tree / token expansion).
- A "try this" guide telling you what to do step-by-step.
- A "Think of it like" plain-English analogy.
- A sidebar tutor (OpenAI) that receives the page's current state as JSON and
  can answer grounded questions like *"why did doc #10 win this query?"* with
  reference to actual scores on screen.
- Spark buttons (✦ analogy / ELI5 / real example / why this matters) for
  instant beginner explanations.

## Audience

- People learning retrieval / RAG / agent systems for the first time.
- People who roughly know the theory but want to see it on real-ish data.
- People building a RAG system who want a mental model of which knob to turn.

## Related

- [[Architecture]] — how the code is laid out
- [[Deployment]] — running it locally or on Railway
- [[Map of Content]]
