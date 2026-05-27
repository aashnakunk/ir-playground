# TF-IDF

The original text ranking scheme. Score each doc by the sum, over query
terms, of (term frequency in this doc) × (term's rarity across the corpus).

## Intuition

Searching your group chat for "pizza." Messages with lots of "pizza" matter.
But if your group always talks about pizza, the word stops being a signal —
it has to be rare somewhere. TF-IDF rewards words that are **frequent here,
rare overall**.

## How it works

For each (doc, term) pair, the TF-IDF weight is:

$$w(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

where TF = count of term in doc (often normalized) and [[IDF]] = `log(N / df(t))`.

To rank, treat the query as a vector of term weights and the doc as a vector
of term weights; compute [[Cosine Similarity]]. That's how scikit-learn's
`TfidfVectorizer` does it.

## When to use

- Baseline retrieval when you have no time / no infra for embeddings.
- Tasks dominated by exact word match (legal, code search).

## When not to use

- General modern retrieval — [[BM25]] is strictly a better keyword scorer.
- Anything requiring semantic match. Use [[Semantic Search]] or [[Hybrid Search]].

## Where it shows up

- Still the default for fast in-memory keyword search in small apps.
- The conceptual ancestor of [[BM25]].

## Related

- [[BM25]] (TF-IDF with diminishing returns and length normalization)
- [[IDF]]
- [[Inverted Index]]
