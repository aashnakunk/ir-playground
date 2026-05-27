# HNSW — Hierarchical Navigable Small World

The graph data structure that makes [[Semantic Search]] fast when the corpus
grows from "30 docs" to "30 million." An **approximate** nearest-neighbor
index over [[Embeddings]].

## Intuition

A subway map for finding the nearest coffee shop. You don't walk every street.
You take the long-distance trains (top layers) to roughly the right
neighborhood, then switch to local lines (lower layers) for fine-grained
search.

## How it works

**Build:**
- Each node (doc embedding) is assigned a max layer with exponentially
  decreasing probability — most nodes only exist at layer 0; a few span up.
- At each layer, every node is linked to its M nearest neighbors.

**Search:**
1. Start at the entry point in the top layer.
2. Greedily walk to the nearest neighbor of the query (by [[Cosine Similarity]]
   distance) until no neighbor is closer.
3. Drop down one layer, use the current node as the new starting point.
4. Repeat until layer 0.
5. At layer 0, do an expanded search keeping a candidate heap of size
   `efSearch`. Return the top-k.

The result is **approximate** — you might miss the true nearest neighbor
occasionally, but you check ~`log(N)` nodes instead of N.

## Knobs

- **M** — max neighbors per node per layer. Higher = denser graph, better
  recall, more memory.
- **efConstruction** — candidate list size during insert. Higher = better
  graph quality, slower build.
- **efSearch** — candidate list size during query. Higher = better recall,
  slower query.

## When to use

- Any [[Semantic Search]] at scale beyond ~10k docs.
- Standard in vector DBs (Pinecone, Qdrant, Weaviate, Milvus, pgvector).

## When not to use

- Small corpora (~thousands of docs) — brute-force cosine is faster and exact.
- You need *exact* nearest neighbors (rare).

## Where it shows up

- `hnswlib` library (the canonical C++ implementation).
- Default index in basically every production vector DB.

## Trade-offs vs alternatives

- **IVF (Inverted File Index):** partitions the space into clusters; queries
  visit a few. Lower memory, slower than HNSW at high recall.
- **Product Quantization (PQ):** compresses vectors. Usually combined with
  IVF as IVF-PQ for billion-scale.
- HNSW dominates for "low millions, need fast queries" — most apps.

## Related

- [[Semantic Search]]
- [[Embeddings]]
- [[Cosine Similarity]]
