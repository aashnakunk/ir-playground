# Inverted Index

The data structure that makes keyword search fast. A dict from **term →
list of doc-ids that contain it** (often with positions).

## Example

For two docs:

```
[1] "the cat sat"
[2] "a cat ran"
```

The inverted index is:

```
the → [1]
cat → [1, 2]
sat → [1]
a   → [2]
ran → [2]
```

Now "find docs containing cat AND ran" is just `intersect([1,2], [2]) = [2]`.
No scanning every doc.

## Why this matters

- [[Boolean]] search uses it directly.
- [[TF-IDF]] and [[BM25]] use it for the lookup, then layer scoring on top.
- Elasticsearch, Solr, Lucene, every full-text search engine — all inverted
  indexes under the hood, with extra tricks (compression, skip lists, sharded
  postings).

## Postings list

Each term's list of doc-ids (sometimes with positions and term frequencies) is
called its **postings list**. Compressed and stored on disk in production.

## Limits

The inverted index doesn't help with [[Semantic Search]] — that's why
embeddings + [[HNSW]] exist as a parallel system.

## Related

- [[Boolean]]
- [[BM25]]
- [[TF-IDF]]
- [[IDF]]
