# jazzbot

Interactive playground for learning TF-IDF, BM25, semantic search, hybrid
search, HNSW, RAG, and CAG — themed around a small jazz corpus. Click around,
watch the algorithms work, and ask the sidebar Claude agent questions about
what you're seeing.

See [SPEC.md](SPEC.md) for the full design.

## Quick start

```bash
cp .env.example .env
# put your OPENAI_API_KEY into .env
./run.sh
```

Then open http://localhost:8000.

## Status

This is built one vertical slice at a time. Currently shipped:

- [x] Repo scaffold
- [x] Jazz corpus
- [x] FastAPI backend skeleton
- [x] OpenAI chat agent (view-aware)
- [x] BM25 page
- [ ] TF-IDF, semantic, HNSW, hybrid, RAG, CAG, compare pages
