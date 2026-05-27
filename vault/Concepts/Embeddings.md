# Embeddings

A dense vector representation of text (or images, or audio, or anything). For
text: a model maps a string to a fixed-length list of floats — typically 384,
768, or 1536 dimensions — where **similar meanings produce similar vectors**.

## Intuition

Think of every possible sentence as a point in a 1536-dimensional space.
"electric music in the 60s" and "jazz fusion" land near each other; "jazz
fusion" and "tax accounting" land far apart. The model was trained on enough
text to learn this geometry. You never see or interpret individual dimensions —
they're just a substrate for measuring distance.

## What we use

OpenAI `text-embedding-3-small` — 1536 dimensions, $0.02 per million tokens.
Cached to disk per corpus at `data/embeddings_<corpus>.json` so we only pay
once per corpus.

## How they're used in this project

- [[Semantic Search]] — embed query and docs, rank by [[Cosine Similarity]]
- [[HNSW]] — graph built over embedding distances
- [[Hybrid Search]] — embeddings on one side, [[BM25]] on the other
- [[HyDE]] — embed an LLM-generated hypothetical answer instead of the question
- [[Rerank]] — initial retrieval is embedding-based

## What you can't do with them

- **Interpret a dimension.** No, dim 472 doesn't mean "jazziness." Embeddings
  are a black box; only distances are meaningful.
- **Trivially mix models.** Embeddings from one model are not compatible with
  another. Don't store and re-rank with a different embedder.
- **Search by exact word.** Two docs sharing a rare proper noun may still embed
  far apart if the surrounding context differs. That's why [[Hybrid Search]]
  exists.

## Related

- [[Cosine Similarity]]
- [[Semantic Search]]
- [[HNSW]]
