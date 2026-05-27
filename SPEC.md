# jazzbot — IR & RAG learning playground

An interactive playground for learning how classic information retrieval and
modern RAG techniques actually work, themed around a small hand-written jazz
corpus. You run it locally, open a page per topic, poke at the visuals, and
talk to an embedded Claude agent that knows what you're currently looking at.

## Goals

- **See it happen.** Every algorithm has a live, interactive visual.
- **Click-to-explain.** Hover/click any UI element for a plain-English explainer.
- **Talk to it.** A sidebar Claude agent receives the page's current state
  (corpus, query, scores, selected doc, params) and answers questions grounded
  in what you're seeing.
- **Side-by-side comparisons** to make the differences obvious.

## Stack

- **Backend:** Python + FastAPI
- **Frontend:** Vanilla HTML/JS + D3.js (one self-contained page per topic, no build step)
- **Algorithms:** library implementations (not from scratch) — focus is visualization
  - `rank_bm25` (BM25)
  - `scikit-learn` `TfidfVectorizer` (TF-IDF)
  - `sentence-transformers` `all-MiniLM-L6-v2` (embeddings)
  - `hnswlib` (HNSW)
- **LLM:** OpenAI via official `openai` SDK (default model `gpt-4o-mini`)
- **API key:** `OPENAI_API_KEY` from server-side `.env` (gitignored)

## Topics

| Page | URL | Visual |
|------|-----|--------|
| TF-IDF | `/pages/tfidf.html` | Term×doc matrix heatmap, score breakdown per doc |
| BM25 | `/pages/bm25.html` | Per-term score bars, doc-length normalization knobs |
| Semantic search | `/pages/semantic.html` | 2D UMAP/PCA projection of docs, query vector overlaid |
| Hybrid search | `/pages/hybrid.html` | BM25 list + semantic list + RRF-fused list, side-by-side |
| HNSW | `/pages/hnsw.html` | Layered graph build, search traversal, param sliders, 2D projection |
| RAG | `/pages/rag.html` | Retrieve top-k → see chunks injected into prompt → model answer |
| CAG | `/pages/cag.html` | Full corpus in system prompt (OpenAI prompt caching when available) → model answer, shows latency + token cost |
| Compare retrievers | `/pages/compare.html` | Same query, 4 retrievers, side-by-side rankings |
| RAG vs CAG | `/pages/rag-vs-cag.html` | Same question, both pipelines, latency + cost + answer diff |

## Corpus

~30 hand-written jazz documents covering eras (early, swing, bebop, cool,
hard bop, modal, free, fusion, smooth, contemporary), major artists, and
adjacent concepts (improvisation, standards, jam sessions). Deliberately
written so subgenre overlap is visible to the retrievers.

Stored as a JSON list of `{id, title, text}` in `data/corpus.json`.

## Chat agent

Every page mounts a sidebar (`static/chat-sidebar.js`) that POSTs to `/api/chat`
with the user's message plus a `view_state` blob the page owns and updates.
The backend uses the OpenAI Chat Completions API.

`view_state` is small JSON that varies by page, e.g. for BM25:

```json
{
  "page": "bm25",
  "query": "modal jazz pioneers",
  "top_results": [{"id": 7, "title": "...", "score": 4.2, "term_scores": {...}}, ...],
  "selected_doc_id": 7,
  "params": {"k1": 1.5, "b": 0.75}
}
```

The backend formats this into a system prompt: "The user is looking at the
BM25 page. Here is what's currently on their screen: <view_state>. Explain
clearly, refer to specific scores/docs the user can see."

## Repo layout

```
jazzbot/
  SPEC.md
  README.md
  requirements.txt
  .env.example
  .gitignore
  run.sh
  data/
    corpus.json
  app/
    main.py              # FastAPI app + route registration
    corpus.py            # loads corpus.json
    chat.py              # Claude client + view-aware prompt
    retrievers/
      bm25.py
      tfidf.py
      semantic.py
      hnsw.py
      hybrid.py
    rag.py
    cag.py
  static/
    index.html           # landing page with topic links
    common.css
    chat-sidebar.js
    explain.js           # click-to-explain tooltip system
    pages/
      bm25.html
      tfidf.html
      ...
```

## Build order

1. Scaffold + corpus + FastAPI skeleton + chat agent + BM25 page (first vertical slice — this PR)
2. TF-IDF page
3. Semantic search page (embedding model download lives here)
4. HNSW page (biggest viz lift)
5. Hybrid page
6. RAG page
7. CAG page
8. Two comparison pages
