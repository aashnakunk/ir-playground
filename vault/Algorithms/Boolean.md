# Boolean

Pre-1970s keyword search. A query is a boolean expression over terms with
AND, OR, NOT, and parentheses. Docs either match or don't — there's no scoring.

## Intuition

Spotify-style filters: `rock AND NOT live`, `jazz OR latin`. The user
specifies hard rules; the engine returns the set of docs that pass.

## How it works

1. Build an [[Inverted Index]] (term → set of doc-ids).
2. Parse the query into an expression tree (shunting-yard → reverse Polish).
3. Evaluate the expression as set operations:
   - `a AND b` = intersection of postings
   - `a OR b` = union
   - `NOT a` = `all_docs - a`
4. Return the resulting set.

## When to use

- You need hard rules, not "best match" ranking. (Tax docs filtered by year,
  log lines filtered by severity, products filtered by attributes.)
- Power-user search interfaces where the user knows the corpus.

## When not to use

- General-user search. People don't think in boolean.
- Anything where partial matches are OK. Boolean has no scoring — a doc with
  4 of 5 terms is no better than one with 0.

## Where it shows up

- Email rules, log filters, Lucene's underlying query layer (even when wrapped
  in [[BM25]] scoring).
- Database `WHERE` clauses on text columns.

## Related

- [[Inverted Index]]
- [[TF-IDF]] (the next step: add scoring instead of binary match)
