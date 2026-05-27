from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from app import cag, chat, rag
from app.corpus import load_corpus
from app.theme import active as active_theme
from app.retrievers import bm25 as bm25_retriever
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/pages", StaticFiles(directory=STATIC_DIR / "pages", html=True), name="pages")
