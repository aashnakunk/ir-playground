# Architecture

## Stack

- **Backend:** Python · FastAPI · uvicorn
- **Frontend:** Vanilla HTML / JS + D3.js. No build step. One self-contained
  HTML file per algorithm page.
- **Algorithms:** library implementations where possible
  (`rank_bm25`, scikit-learn `TfidfVectorizer`, OpenAI embeddings). Exceptions
  written from scratch where the library hides the internals we want to
  visualize:
  - **HNSW** — `app/retrievers/hnsw.py` (need per-node graph access for the viz)
  - **Boolean** — `app/retrievers/boolean.py` (shunting-yard parser + inverted index)
  - **MCTS / Beam / A* / BFS-DFS** — pedagogical, no real lib needed
- **LLM:** OpenAI Chat Completions (default `gpt-4o-mini`)
- **Embeddings:** OpenAI `text-embedding-3-small`, cached to disk per corpus

## Repo layout

```
ir-playground/
  SPEC.md / README.md / BACKLOG.md
  requirements.txt / .env.example / run.sh
  data/
    jazz.json                  # default corpus
    cuisine.json               # deployed corpus
    embeddings_<corpus>.json   # cached on first run
  app/
    main.py        # FastAPI routes
    theme.py       # corpus + per-page defaults selector
    corpus.py      # loads data/<corpus>.json
    embeddings.py  # OpenAI embeddings + disk cache
    chat.py        # view-aware tutor
    rag.py / cag.py / mmr.py / hyde.py / rerank.py
    retrievers/    # bm25, tfidf, semantic, hybrid, hnsw, boolean
    agents/        # graph (BFS/DFS), astar, beam, mcts
  static/
    index.html
    common.css
    theme.js          # /api/theme primes default queries
    chat-sidebar.js   # the always-on tutor with spark buttons
    explain.js        # click-to-explain tooltips
    pages/
      bm25.html, tfidf.html, semantic.html, hybrid.html,
      hnsw.html, mmr.html, rerank.html, hyde.html, rag.html, cag.html,
      boolean.html, bfs-dfs.html, astar.html, beam.html, mcts.html
  vault/          # this Obsidian vault
```

## The view-aware tutor

The thing that makes the chat tutor feel smart: every page sets a
`window.JazzbotView` object with a `state()` function that returns a JSON
snapshot of what's currently on screen — current query, results, scores,
selected doc, parameters. The chat widget POSTs that snapshot to `/api/chat`
alongside the user message. The backend injects it into the system prompt so
the model's answers can reference specific numbers visible to the user.

This contract — `window.JazzbotView.state()` — must be implemented by every
new page.

## Theme system

`CORPUS=jazz` (default) or `CORPUS=cuisine` env var selects which
`data/<corpus>.json` is loaded. Embeddings are cached per-corpus. The
`/api/theme` endpoint exposes the active theme name + per-page default queries;
`static/theme.js` reads it on each page to populate the input and retitle the
tab.

Adding a new theme: drop a JSON corpus, add an entry to `THEMES` in
`app/theme.py`, set `CORPUS=<name>`.

## Related

- [[Project Overview]]
- [[Deployment]]
