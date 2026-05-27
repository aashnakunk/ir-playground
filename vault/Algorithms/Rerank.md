# Rerank — Two-Stage Retrieval

Retrieve cheap, then re-order with a smarter (slower) model. The 2-stage
trick behind almost every production [[RAG]] system.

## Intuition

Hiring funnel. A 5-minute resume scan narrows 1,000 applicants to 15. Then a
real interviewer (much smarter, much more expensive) picks the top 5. Same
pattern: fast & dumb first, slow & sharp second.

## How it works

1. **Stage 1 (retrieve, cheap):** [[BM25]] or [[Semantic Search]] gives you the
   top-15 or top-50 candidates from millions of docs. Fast index lookup.
2. **Stage 2 (rerank, expensive):** Score each (query, candidate) pair with a
   **cross-encoder** — a small BERT-style model fine-tuned to predict
   relevance. Sort by that score. Take the top 5.

A true cross-encoder reads the query AND the candidate together (unlike
embedding-based retrieval where each is encoded independently). It catches
subtle relevance that single-vector cosine misses.

In this project's demo we use an LLM (`gpt-4o-mini`) as the reranker — slower
and pricier than a real cross-encoder, but demonstrates the pattern without an
extra dependency.

## When to use

- Production RAG. Almost always worth it. Pulls noticeable precision wins.
- High-stakes retrieval (legal, medical) where the cost of getting the top-5
  wrong is high.

## When not to use

- You have so few docs that the cheap retriever already has trivial precision.
- Strict latency budgets where the extra cross-encoder call doesn't fit.

## Where it shows up

- Cohere Rerank, Voyage Rerank, BGE Reranker — all production cross-encoders.
- Every serious RAG system that cares about answer quality.

## Cost

A 6-layer cross-encoder on 50 candidates is ~50ms on a GPU, or 200ms on CPU —
cheap. Using a full LLM as the reranker (as here) costs more, ~$0.0007 per
50-candidate call with `gpt-4o-mini`.

## Related

- [[Semantic Search]] (typical stage-1 retriever)
- [[BM25]] (alternative stage-1 retriever)
- [[Hybrid Search]] (often the stage-1 retriever)
- [[RAG]]
