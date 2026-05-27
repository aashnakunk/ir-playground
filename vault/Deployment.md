# Deployment

## Local

```bash
cp .env.example .env
# paste OPENAI_API_KEY into .env
./run.sh
```

Open <http://localhost:8000>. First request to `/api/semantic/search` will
embed the 30-doc corpus once (~$0.0001) and cache to
`data/embeddings_<corpus>.json`. Subsequent runs are instant.

## Railway

1. Create a new Railway project from the GitHub repo (`aashnakunk/ir-playground`).
2. Add env vars in Railway dashboard:
   - `OPENAI_API_KEY` = your OpenAI key
   - `OPENAI_MODEL` = `gpt-4o-mini` (default)
   - `CORPUS` = `cuisine` (or `jazz`)
3. Set start command:
   ```
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
4. Ship the cached embeddings file with the repo so the first deployed request
   doesn't pay for embedding (already committed under
   `data/embeddings_jazz.json`; cuisine cache will be generated on first
   request and persist within the running container — for cold-start speed,
   embed cuisine locally once and commit that JSON too).

## Vercel (not recommended)

Possible via `@vercel/python` but:

- FastAPI's static-file mount doesn't survive Vercel's serverless model; you'd
  need to move HTML/JS to `/public` and write a `vercel.json` rewriting
  `/api/*` to a handler.
- numpy + scikit-learn + openai together push the 250 MB function size limit.

Use Railway or Render instead.

## Cost notes

Per query, approximate:

| Page | Cost |
|---|---|
| BM25, TF-IDF, Boolean, HNSW build, MMR | $0 (no API calls) |
| Semantic, Hybrid | ~$0.00002 (one embedding) |
| RAG | ~$0.0002 (embed + small chat completion) |
| CAG | ~$0.0006 first call, ~$0.0003 with cache hits |
| HyDE | ~$0.0003 (LLM hypothetical + embedding) |
| Rerank | ~$0.0007 (embed + scoring 15 candidates) |
| Chat tutor | ~$0.0001 per message |

So $5 of OpenAI credit gets you thousands of interactions.

## Related

- [[Architecture]]
- [[Backlog]]
