# RAG — Retrieval-Augmented Generation

"Look it up first, then answer." How chatbots ground their replies in real
docs instead of guessing from training data.

## Intuition

Open-book exam. The chatbot first looks up the relevant chapter ([[Semantic Search]]
/ [[Hybrid Search]]), then writes its answer using what it just read. The
model itself doesn't change — only its inputs do.

## How it works

1. Embed the user's question (or use [[Hybrid Search]]).
2. Retrieve top-k relevant chunks from the corpus.
3. Build a prompt: a system message telling the model to answer using only the
   provided context, plus the user's question and the retrieved chunks pasted
   in.
4. Send to the LLM. Return its answer.

That's it. RAG is a pattern, not a model.

## Variants worth knowing

- **Naive RAG** — what's above.
- **+ Reranking** — retrieve top-20, rerank with a heavier model ([[Rerank]]),
  feed top-5 to the LLM.
- **+ Query rewriting** — let the LLM rephrase the query first.
- **+ [[HyDE]]** — let the LLM dream up a hypothetical answer, embed *that*
  for retrieval.
- **+ Diversity** — apply [[MMR]] to retrieved chunks before feeding them in.

## When to use

- Private data the model doesn't know.
- Up-to-date facts (the model's training is frozen).
- Anywhere you need source citations.

## When not to use

- Tasks that need to span the *entire* corpus and retrieval loses the thread.
  Consider [[CAG]].
- Pure reasoning problems with no need for external facts.

## Where it shows up

- Every "chat with your docs" product.
- Most enterprise LLM systems.
- Copilot, Perplexity, custom GPTs with retrieval.

## Related

- [[CAG]] (the alternative: no retrieval, full corpus in prompt)
- [[Semantic Search]]
- [[Hybrid Search]]
- [[Rerank]]
- [[HyDE]]
- [[MMR]]
