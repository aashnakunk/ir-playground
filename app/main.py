from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from app import cag, chat, hyde, mmr, rag, rerank
from app.agents import astar as agent_astar
from app.agents import beam as agent_beam
from app.agents import graph as agent_graph
from app.agents import mcts as agent_mcts
from app.corpus import load_corpus
from app.theme import active as active_theme
from app.retrievers import bm25 as bm25_retriever
from app.retrievers import boolean as boolean_retriever
from app.retrievers import hnsw as hnsw_retriever
from app.retrievers import hybrid as hybrid_retriever
from app.retrievers import semantic as semantic_retriever
from app.retrievers import tfidf as tfidf_retriever

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="jazzbot")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/theme")
def get_theme():
    t = active_theme()
    return {"corpus": t.corpus, "name": t.name, "tutor_role": t.tutor_role, "default_queries": t.default_queries}


@app.get("/api/corpus")
def get_corpus():
    return load_corpus()


@app.get("/api/bm25/search")
def bm25_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    return bm25_retriever.search(q, top_k=top_k)


@app.get("/api/tfidf/search")
def tfidf_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    return tfidf_retriever.search(q, top_k=top_k)


@app.get("/api/tfidf/matrix")
def tfidf_matrix(top_n_terms: int = 50):
    return tfidf_retriever.vocab_stats(top_n_terms=top_n_terms)


@app.get("/api/semantic/search")
def semantic_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    try:
        return semantic_retriever.search(q, top_k=top_k)
    except Exception as e:
        raise HTTPException(500, f"semantic search failed: {e}")


@app.get("/api/hybrid/search")
def hybrid_search(q: str, top_k: int = 5):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    try:
        return hybrid_retriever.search(q, top_k=top_k)
    except Exception as e:
        raise HTTPException(500, f"hybrid search failed: {e}")


@app.get("/api/hnsw/graph")
def hnsw_graph(M: int = 8, ef_construction: int = 16, seed: int = 7):
    try:
        return hnsw_retriever.graph(M=M, ef_construction=ef_construction, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"hnsw graph failed: {e}")


@app.get("/api/hnsw/search")
def hnsw_search(q: str, M: int = 8, ef_construction: int = 16, ef_search: int = 16, top_k: int = 5, seed: int = 7):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    try:
        return hnsw_retriever.search(q, M=M, ef_construction=ef_construction, ef_search=ef_search, top_k=top_k, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"hnsw search failed: {e}")


@app.get("/api/boolean/search")
def boolean_search(q: str):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    try:
        return boolean_retriever.search(q)
    except Exception as e:
        raise HTTPException(500, f"boolean search failed: {e}")


@app.get("/api/boolean/index")
def boolean_index(min_df: int = 1, top_n: int = 200):
    return boolean_retriever.index_summary(min_df=min_df, top_n=top_n)


@app.get("/api/mmr/search")
def mmr_search(q: str, top_k: int = 5, candidate_k: int = 15, lam: float = 0.5):
    if not q.strip():
        raise HTTPException(400, "query is empty")
    try:
        return mmr.search(q, top_k=top_k, candidate_k=candidate_k, lam=lam)
    except Exception as e:
        raise HTTPException(500, f"mmr failed: {e}")


class HydeRequest(BaseModel):
    question: str
    top_k: int = 5


@app.post("/api/hyde/run")
def hyde_run(req: HydeRequest):
    try:
        return hyde.run(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"hyde failed: {e}")


class RerankRequest(BaseModel):
    query: str
    top_k: int = 5
    candidate_k: int = 15


@app.post("/api/rerank/run")
def rerank_run(req: RerankRequest):
    try:
        return rerank.run(req.query, top_k=req.top_k, candidate_k=req.candidate_k)
    except Exception as e:
        raise HTTPException(500, f"rerank failed: {e}")


class AskRequest(BaseModel):
    question: str
    top_k: int = 4


@app.post("/api/rag/ask")
def rag_ask(req: AskRequest):
    try:
        return rag.answer(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(500, f"rag failed: {e}")


@app.post("/api/cag/ask")
def cag_ask(req: AskRequest):
    try:
        return cag.answer(req.question)
    except Exception as e:
        raise HTTPException(500, f"cag failed: {e}")


class ChatRequest(BaseModel):
    message: str
    view_state: dict
    history: list[dict] = []


@app.post("/api/chat")
def post_chat(req: ChatRequest):
    try:
        reply = chat.answer(req.message, req.view_state, req.history)
    except Exception as e:
        raise HTTPException(500, f"chat failed: {e}")
    return {"reply": reply}


@app.get("/api/graph/search")
def graph_search(algo: str = "bfs", start: str = "A", goal: str = "J"):
    try:
        return agent_graph.search(algo, start=start, goal=goal)
    except Exception as e:
        raise HTTPException(500, f"graph search failed: {e}")


@app.get("/api/astar/search")
def astar_search(start_x: int = 1, start_y: int = 1, goal_x: int = 12, goal_y: int = 8, w: float = 1.0):
    try:
        return agent_astar.search(start=(start_x, start_y), goal=(goal_x, goal_y), heuristic_weight=w)
    except Exception as e:
        raise HTTPException(500, f"astar failed: {e}")


@app.get("/api/beam/search")
def beam_search(beam_width: int = 3, max_steps: int = 5):
    try:
        return agent_beam.search(beam_width=beam_width, max_steps=max_steps)
    except Exception as e:
        raise HTTPException(500, f"beam failed: {e}")


@app.get("/api/mcts/run")
def mcts_run(iterations: int = 20, seed: int = 7):
    try:
        return agent_mcts.run(iterations=iterations, seed=seed)
    except Exception as e:
        raise HTTPException(500, f"mcts failed: {e}")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/pages", StaticFiles(directory=STATIC_DIR / "pages", html=True), name="pages")
