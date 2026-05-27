# IDF — Inverse Document Frequency

A measure of how *rare* a word is across a corpus. Rare words are more
discriminating than common ones, so they're worth more in a match.

## Formula

$$\text{IDF}(t) = \log\!\left(\frac{N}{\text{df}(t)}\right)$$

Where `N` is total docs and `df(t)` is the number of docs containing term `t`.
A word in every doc → `log(1)` = 0. A word in one of a million docs →
`log(1,000,000)` ≈ 14.

[[BM25]] uses a smoothed variant:
$\log\!\left(\frac{N - \text{df} + 0.5}{\text{df} + 0.5}\right)$
which can go negative for very common terms.

## Intuition

"The" appears in every doc → useless as a search signal → IDF ≈ 0.
"Tandoor" appears in 2 of 30 docs → great signal → IDF is high.

## Used in

- [[TF-IDF]] — score = TF × IDF
- [[BM25]] — same idea with saturation and length normalization

## Common quirk

If you search for a term that's in more than half of all docs, BM25's IDF
formula goes negative. `rank_bm25` (and most libraries) floor it at 0 with an
epsilon, which is why you'll sometimes see a query term with IDF = 0.000.

## Related

- [[TF-IDF]]
- [[BM25]]
- [[Inverted Index]]
