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
from app.memory.session import (
    create_session,
    save_message,
    get_full_history,
    list_sessions,
    get_session_info,
    bind_session_repo,
    extract_repo_name,
)
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
    session_id      : str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question about the codebase")
    repo_name: str | None = Field(default=None, description="Optional target repo scope")
    strict_mode: bool = Field(default=True, description="When False, skips CRAG correction + Self-RAG critique loops for speed")


class QueryResponse(BaseModel):
    answer        : str
    sources       : list[str]
    chunks_used   : int
    intent        : str
    primary_files : list[str]


class ChatRequest(BaseModel):
    question   : str
    session_id : str | None = None   # None → creates a new session
    repo_name  : str | None = None   # Explicit target repo override
    strict_mode: bool = True


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
async def ingest(req: IngestRequest):
    logger.info(f"Ingest request: {req.url}")
    try:
        result = ingest_repo(req.url)
        # Create a new session bound to this repo URL
        session_id = await create_session(repo_url=req.url)
        return {**result, "session_id": session_id}
    except Exception as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
@traceable(run_type="chain")
async def query(req: QueryRequest):
    logger.info(f"Query: {req.question} | repo_name={req.repo_name} | strict_mode={req.strict_mode}")
    try:
        # Pass a dummy session ID since /query is stateless
        result = await query_repo(
            req.question,
            "00000000-0000-0000-0000-000000000000",
            repo_name=req.repo_name,
            strict_mode=req.strict_mode,
        )
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
    session_id = req.session_id
    repo_name = req.repo_name

    if session_id:
        info = await get_session_info(session_id)
        if info and not repo_name:
            repo_name = info.get("repo_name")
    else:
        # Create a new session
        session_id = await create_session()

    await save_message(session_id, role="user", content=req.question)

    try:
        result = await query_repo(
            req.question,
            session_id,
            repo_name=repo_name,
            strict_mode=req.strict_mode,
        )
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


@app.post("/reset")
async def reset():
    from reset import reset_all
    try:
        await reset_all()
        return {"status": "success", "message": "All database sessions, vector store embeddings, and repository indices cleared."}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))