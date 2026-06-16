"""
FastAPI Routes
===============
Exposes /ingest and /query endpoints backed by the LangGraph pipeline.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from langsmith import traceable

from app.graph.rag_graph import ingest_repo, query_repo
from app.memory.session import create_session, save_message, get_full_history, list_sessions
from app.memory.db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(
    title       = "GitHub Codebase RAG",
    description = "Ask questions about any GitHub repository",
    version     = "0.1.0",
    lifespan    = lifespan,
)

# Allow Streamlit frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request/Response models ─────────────────────────────────

class IngestRequest(BaseModel):
    url: str = Field(..., description="GitHub repository URL")


class IngestResponse(BaseModel):
    repo            : str
    files_scanned   : int
    chunks_indexed  : int
    vector_db_stats : dict


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about the codebase")


class QueryResponse(BaseModel):
    answer        : str
    sources       : list[str]
    chunks_used   : int
    intent        : str
    primary_files : list[str]


class ChatRequest(BaseModel):
    question   : str
    session_id : str | None = None   # None → creates a new session


class ChatResponse(BaseModel):
    session_id    : str
    answer        : str
    sources       : list[str]
    chunks_used   : int
    intent        : str
    primary_files : list[str]





# ── Routes ───────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
@traceable(run_type="chain")
def ingest(req: IngestRequest):
    logger.info(f"Ingest request: {req.url}")
    try:
        result = ingest_repo(req.url)
        return result
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@traceable(run_type="chain")
async def query(req: QueryRequest):
    logger.info(f"Query: {req.question}")
    try:
        # Pass a dummy session ID since /query is stateless
        result = await query_repo(req.question, "00000000-0000-0000-0000-000000000000")
        return result
    except RuntimeError as e:
        # BM25 index not built yet
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
@traceable(run_type="chain")
async def chat(req: ChatRequest):
    session_id = req.session_id or await create_session()

    await save_message(session_id, role="user", content=req.question)

    try:
        result = await query_repo(req.question, session_id)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    await save_message(
        session_id, role="assistant", content=result["answer"],
        intent=result["intent"], sources=result["sources"],
    )

    return {**result, "session_id": session_id}


@app.get("/sessions")
async def sessions():
    return await list_sessions()


@app.get("/sessions/{session_id}/history")
async def session_history(session_id: str):
    return await get_full_history(session_id)