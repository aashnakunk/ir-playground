# CAG — Cache-Augmented Generation

Skip retrieval. Hand the LLM the entire corpus in the system prompt and let
prompt caching make it cheap.

## Intuition

Memorize the textbook before the exam vs. flipping to a chapter mid-answer
(that's [[RAG]]). CAG is slower per question the first time, but **prompt
caching** makes the second question onwards basically free.

## How it works

1. Concatenate the entire corpus into the system prompt (once).
2. Send `system + user` to the LLM normally.
3. OpenAI (and Anthropic) automatically cache stable prompt prefixes ≥ ~1024
   tokens. Subsequent calls with the same prefix hit the cache: input tokens
   are billed at ~50% (OpenAI) or 10% (Anthropic) and time-to-first-token
   drops dramatically.

The response includes `prompt_tokens_details.cached_tokens` so you can see
caching at work.

## When to use

- Small-to-medium corpus that fits in the model's context window (Claude
  Sonnet: 200k tokens; GPT-4o: 128k tokens).
- Repeat queries against the same corpus by the same user / session.
- Cases where retrieval keeps missing important context.

## When not to use

- Huge corpora (millions of docs). Use [[RAG]].
- One-off queries where the cache won't hit again.
- Latency-critical first calls (caching only helps from the 2nd hit).

## Trade-offs vs RAG

| | RAG | CAG |
|---|---|---|
| Prompt size | small (top-k chunks) | huge (full corpus) |
| First call cost | low | high |
| Repeat call cost | low | very low (cache hit) |
| Recall | depends on retriever quality | perfect — model sees everything |
| Best for | large corpora | small corpora, repeat traffic |

## Where it shows up

- Anthropic's prompt caching docs explicitly call out the pattern.
- Tool documentation systems that load the whole API reference in context.
- Personal-assistant apps that load a user's knowledge base each session.

## Related

- [[RAG]]
