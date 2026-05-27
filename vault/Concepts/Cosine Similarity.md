# Cosine Similarity

A measure of how aligned two vectors are, ignoring their magnitudes. Output
ranges from -1 (opposite) through 0 (orthogonal) to 1 (identical direction).
For [[Embeddings]] from a sane model, scores are typically in `[0, 1]`.

## Formula

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

In code:

```python
def cosine(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

If you L2-normalize all vectors first (set magnitude to 1), cosine similarity
is just the dot product — which is why production vector DBs pre-normalize.

## Why cosine, not Euclidean?

Text [[Embeddings]] tend to have varying magnitudes that don't correlate with
meaning. Two docs about jazz might have very different magnitudes but point in
the same direction. Cosine ignores magnitude; Euclidean doesn't.

## Cosine distance

`1 - cosine_similarity`. Used in [[HNSW]] because the algorithm wants a
*distance* (smaller = closer) rather than a similarity.

## Related

- [[Embeddings]]
- [[Semantic Search]]
- [[HNSW]]
