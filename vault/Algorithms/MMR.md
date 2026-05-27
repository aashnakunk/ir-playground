# MMR — Maximal Marginal Relevance

Diversity re-ranking. Stops your top-k from being five near-duplicates.

## Intuition

Building a playlist. Even if your top 5 most-liked songs are all by one
artist, you'd want some variety. MMR is the rule "great track, but how
different is it from what I've already picked?"

## How it works

After initial retrieval, pick the top-k one at a time. At each step, score
remaining candidates by:

$$\text{MMR}(d) = \lambda \cdot \text{sim}(q, d) - (1 - \lambda) \cdot \max_{d' \in \text{picked}} \text{sim}(d, d')$$

- First term: how relevant the doc is to the query.
- Second term: a penalty for how similar it is to any doc you've already picked.
- **λ = 1** → pure relevance (same as plain ranking).
- **λ = 0** → pure diversity (whatever's least like what's picked).
- **λ ≈ 0.5–0.7** → typically the useful sweet spot.

## When to use

- Search results UIs where users want to see different angles.
- [[RAG]] chunk selection: stop feeding the LLM three paraphrases of the same
  paragraph. Better coverage of the source material per prompt token.

## When not to use

- You want the absolute best match and don't care if it's redundant
  (single-answer QA).
- Tiny candidate pools where you'd lose obvious top picks.

## Where it shows up

- LangChain has a `mmr_search_type` option for vector stores.
- Recommendation systems re-rank with MMR-like diversity all the time.
- Original paper: Carbonell & Goldstein (1998).

## Related

- [[Semantic Search]]
- [[RAG]]
- [[Rerank]]
