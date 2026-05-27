# Hybrid Search

Combine [[BM25]] (keyword) and [[Semantic Search]] (meaning) into a single
ranked list. Usually beats either alone.

## Intuition

Two friends recommend restaurants: one knows the menu by exact name, one
"gets your vibe." You're better off combining their picks than picking one
friend.

## How it works

Run [[BM25]] and [[Semantic Search]] separately for the same query; get two
ranked lists. Fuse them.

The cleanest fusion is **Reciprocal Rank Fusion (RRF)** (Cormack et al. 2009):

$$\text{RRF}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}_r(d)}$$

where `k = 60` is conventional. A doc that ranks #1 in BM25 contributes
`1/(60+1) ≈ 0.0164`; rank #5 contributes `1/(60+5) ≈ 0.0154`. Sum across
retrievers. Sort descending.

Why this works: it only uses *ranks*, not raw scores. BM25 scores and cosine
similarities live on different scales and can't be added directly. Ranks
sidestep that.

## When to use

- Production RAG, almost always. It's the default in serious systems for a reason.
- Any query mix that includes proper nouns + conceptual stuff.

## When not to use

- You're certain the user query is purely conceptual or purely keyword. (Rare.)

## Where it shows up

- Elasticsearch's `rank_features` and hybrid query DSL.
- Weaviate, Pinecone, Qdrant — all expose hybrid + RRF.
- Most production RAG systems.

## Related

- [[BM25]]
- [[Semantic Search]]
- [[Rerank]] (often layered after hybrid for the final top-k)
