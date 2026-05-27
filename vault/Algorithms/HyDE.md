# HyDE — Hypothetical Document Embeddings

Ask the LLM to write a hypothetical answer first, embed *that*, then retrieve.
Often beats embedding a short question directly.

## Intuition

Googling. If you search "why does X happen" you get junk. If you search the
answer you'd expect ("X happens because…") you get great pages. HyDE asks the
LLM to write that fake answer for you, then uses it as a richer query.

## How it works

1. User asks a terse question.
2. Prompt an LLM: *"Write a short factual passage that would plausibly answer
   this. Invent details if needed. Don't hedge."*
3. Embed the generated passage (not the question).
4. Retrieve nearest docs to that embedding.

The hypothetical answer can be wrong — that's fine. We're not using it as
truth. We're using it because it looks like a doc, which makes it embed near
real docs.

## Why it works

Question [[Embeddings]] and answer [[Embeddings]] live in slightly different
regions of vector space (questions cluster with questions). A 5-word query
doesn't look much like a 100-word factual passage. The hypothetical answer
does. So [[Cosine Similarity]] does a better job finding the real passage.

Original paper: Gao et al., *"Precise Zero-Shot Dense Retrieval without
Relevance Labels"* (2022).

## When to use

- Terse user queries where plain [[Semantic Search]] underperforms.
- Zero-shot / few-shot retrieval where you can't fine-tune the embedder.
- Q&A systems with conversational user input.

## When not to use

- Well-phrased, query-like inputs (already similar to docs).
- Latency-critical paths — adds one LLM call before retrieval.

## Where it shows up

- LangChain's `HypotheticalDocumentEmbedder`.
- Mentioned in many "advanced RAG" tutorials.
- One of several "query expansion" techniques (alongside multi-query, step-back
  prompting, etc.).

## Related

- [[Semantic Search]]
- [[Embeddings]]
- [[RAG]]
- [[Rerank]]
