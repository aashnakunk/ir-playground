# Backlog

Open items for the next iteration. Picked up from the chat session ending 2026-05-27.

## High priority — UX / navigation overhaul

- [ ] **Reshape the site into two distinct surfaces:**
  - **Home page** = wiki-style intro. Lead with what IR/RAG/agent-search *are*, with a clear conceptual map, key terms defined inline, plain English. No card grid of algorithms here.
  - **`/explore` page** = tiled "Explore Algos" page. Grid of algorithm cards (icon + name + one-liner + category badge). Filter / sort by category. This is where the current per-section path lists belong.
- [ ] **Replace the top-bar nav** ("tfidf bm25 semantic …"). It's too dense and reads as jargon. Possible replacements: a compact "Explore algos" link, a "Home" link, a corpus picker, a "Help" link. Per-page back-button to /explore.
- [ ] **Audit every page for jargon.** Plain English first, vocabulary second. Concrete miss to fix: BM25 page subtitle says *"upgrade to TF-IDF"* but in the new Core section BM25 is listed before TF-IDF (TF-IDF is now under Foundational). Either reorder, or rewrite copy so each page stands alone without referencing siblings by position.

## Copy / content cleanup

- [ ] Tighten the landing hero to fewer words (current still reads long).
- [ ] Make sure each page's intro stands alone — no "as we saw on the previous page" framing.
- [ ] Standardize the subtitle voice. Pick one of: plain analogy / what it does / when to use it. Right now it's mixed.
- [ ] Re-check "Think of it like" lines for any that feel forced; cut or rewrite.
- [ ] Spark buttons (✦ analogy / ELI5 / etc.) → consider hover descriptions so users know what each does before clicking.

## Information architecture

- [ ] Define each algorithm's metadata once (name, category, one-liner, analogy, status: "production"/"agent"/"foundational", related-algos) in a single config so both the wiki home and the explore page can render from one source of truth.
- [ ] Add inter-algorithm links: at the bottom of each page, "related: …" pointing to siblings + foundational ancestors.
- [ ] Add a glossary page or hover-tooltip system for recurring terms (embedding, cosine, top-k, logprob, UCB, heuristic, posting, …).

## Visualization polish

- [ ] HNSW: the corpus is too small (30 nodes) — every search visits almost all of them, which makes "approximate" search look like brute force. Either ship a bigger corpus by default or add a "synthetic corpus" mode with hundreds of points.
- [ ] HNSW step mode: speed slider hard-coded to setInterval; consider keyboard shortcuts (←/→/space).
- [ ] A*: let user click cells to add/remove walls + reposition start/goal interactively.
- [ ] Beam search: synthetic token tree is fine pedagogically, but flag it explicitly as not real LLM output. Maybe a stretch goal: real beam search via OpenAI logprobs.
- [ ] MCTS: tree gets crowded at 27 leaves — add zoom/pan, or shrink the tree to branching 2 depth 3.

## Feature ideas (raised but not yet built)

- [ ] Compare-mode landing widget: pick 2-3 algorithms, run the same query side-by-side, see ranking differences.
- [ ] A "what's wrong with each algo" callout per page — common failure modes & when NOT to use it.
- [ ] Per-page "next steps" — what to read / try next.
- [ ] Persist API key + corpus selection in localStorage so users don't re-set on reload (currently env-driven only).

## Deploy

- [ ] Railway deploy (start command, env vars, embedding cache pre-shipped). See README "Deploying" section.
- [ ] Decide whether the deployed instance uses `cuisine` or a third theme.

## Notes for future me

- The corpus theme system is at `app/theme.py`. New theme = drop `data/<name>.json` + add an entry. Embeddings auto-cache per-corpus.
- All algorithm-specific code lives under `app/retrievers/` (IR) or `app/agents/` (graph search) — keep that split.
- The view-aware tutor reads `window.JazzbotView.state()` from each page. New pages MUST expose this.
- `static/theme.js` rewrites page titles and primes default queries; new pages should add a `default_queries` entry in `theme.py`.
- The brand "jazzbot" still appears in some page titles via `theme.js` replacement. Public deployment uses `<theme> IR playground` automatically.
