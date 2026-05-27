# ir-playground

An interactive, browser-based playground for learning information retrieval and
RAG techniques. Each page implements one technique end-to-end against a small
corpus and shows the live internals — scores, embeddings, graph edges, prompts.
Every page has a sidebar tutor (OpenAI) that receives the page's current state
and can answer grounded questions like *"why did doc #10 win?"*

**Topics:** TF-IDF · BM25 · Semantic search · Hybrid (RRF) · HNSW · RAG · CAG.

## Quick start

```bash
cp .env.example .env
# put your OPENAI_API_KEY into .env
./run.sh
```

Open http://localhost:8000.

## Themes

The corpus is selectable via the `CORPUS` env var:

| `CORPUS=` | Theme | File |
|-----------|-------|------|
| `jazz` (default) | jazz history, subgenres, artists | `data/jazz.json` |
| `cuisine` | world cuisines, techniques, dishes | `data/cuisine.json` |

Embeddings are cached per-corpus to `data/embeddings_<corpus>.json`. To add
another theme: drop a `data/<name>.json` (list of `{id,title,text}`), add an
entry to `THEMES` in `app/theme.py`, set `CORPUS=<name>`.

## Stack

- **Backend:** Python · FastAPI
- **Frontend:** vanilla HTML/JS + D3.js (no build step, one page per topic)
- **Algorithms:** library implementations (`rank_bm25`, scikit-learn,
  OpenAI embeddings) — focus is visualization, not reimplementation. The
  exception is HNSW, which is a small from-scratch implementation in
  `app/retrievers/hnsw.py` so the graph can be drawn.
- **LLM:** OpenAI via the official `openai` SDK (default model `gpt-4o-mini`)

## Layout

```
app/
  main.py              # FastAPI routes
  theme.py             # corpus selection + per-theme defaults
  corpus.py            # loads data/<corpus>.json
  embeddings.py        # OpenAI text-embedding-3-small + on-disk cache
  chat.py              # view-aware tutor
  rag.py / cag.py      # the two generation pipelines
  retrievers/          # bm25, tfidf, semantic, hybrid, hnsw
static/
  index.html
  common.css
  theme.js             # fetches /api/theme and primes page defaults
  chat-sidebar.js      # the always-on sidebar tutor
  explain.js           # click-to-explain tooltips
  pages/               # bm25, tfidf, semantic, hybrid, hnsw, rag, cag
data/
  jazz.json
  cuisine.json
  embeddings_<corpus>.json   # cached, generated on first run
```

See [SPEC.md](SPEC.md) for the original design notes.

## Deploying

Tested locally; no platform-specific config shipped. For a quick public deploy:

- **Render** / **Railway** / **Fly.io** — straightforward for FastAPI:
  point at this repo, set start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`,
  set `OPENAI_API_KEY` and `CORPUS` env vars.
- **Vercel** — possible via the `@vercel/python` runtime but you'll bump into
  the 250 MB function-size limit because of numpy + scikit-learn. Workable but
  not the easiest path.

Either way, ship the cached `data/embeddings_<corpus>.json` so the first
request doesn't have to pay for embedding the corpus.
