# Semantic Search

Search by meaning, not by exact words. Embed the query and every doc as
high-dimensional vectors; rank by [[Cosine Similarity]].

## Intuition

Shazam for ideas. You hum the *meaning* of what you want; it finds docs that
hum a similar tune — even if the words are totally different. Query "electric
music in the 60s" finds the doc about jazz fusion without either phrase
appearing in it.

## How it works

1. Embed each doc with a model like OpenAI `text-embedding-3-small` → vector
   of 1536 floats. Cache it.
2. At query time, embed the query → another 1536-float vector.
3. Compute [[Cosine Similarity]] between query and every doc.
4. Return the top-k.

At scale, step 3 becomes the bottleneck — that's where [[HNSW]] comes in.

## When to use

- Natural-language questions ("how do I cancel my subscription?").
- Cross-lingual or paraphrase-heavy queries.
- Anywhere the user might not know the exact terminology in the docs.

## When not to use

- Exact-match needs (proper nouns, IDs, code). Use [[BM25]] or [[Hybrid Search]].
- Tiny corpora where vibes don't matter and a regex would do.

## Where it shows up

- Every modern RAG system's retrieval step.
- Semantic deduplication, clustering, recommendation.

## Limits

- "Meaning" is the model's interpretation of meaning. Two unrelated topics
  with similar phrasing can rank close. (See [[Hybrid Search]] and [[Rerank]].)
- Embedding models are not interchangeable — switch models and re-embed.

## Related

- [[Embeddings]]
- [[Cosine Similarity]]
- [[HNSW]]
- [[Hybrid Search]]
