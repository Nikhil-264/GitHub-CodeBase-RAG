"""
RAG Graph (LangGraph)
======================
Wires all agents together into a stateful graph:

    Question
       │
       ▼
   Intent Agent
       │
       ▼
   Retrieval Agent
       │
       ▼
    Reranker
       │
       ▼
   Analysis Agent
       │
       ▼
   Answer Agent
       │
       ▼
   Answer + Sources
"""

import os
from typing import TypedDict, Optional
from loguru import logger
from dotenv import load_dotenv
from langsmith import traceable

from langgraph.graph import StateGraph, END

from app.ingestion.clone_repo import clone_repo
from app.ingestion.scan_repo import scan_repo
from app.ingestion.chunker import chunk_files
from app.retrieval.vector_store import index_chunks, collection_stats
from app.retrieval.bm25 import BM25Retriever
from app.reranker.reranker import rerank
from app.agents.retrieval_agent import retrieve
from app.agents.intent_agent import classify_intent
from app.agents.analysis_agent import analyse
from app.agents.answer_agent import answer

load_dotenv()

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "repos/bm25_index.pkl")

# ════════════════════════════════════════════════════════════
# Graph State
# ════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    question         : str
    session_id       : str                    # NEW
    chat_history      : Optional[str]          # NEW — formatted past turns
    intent           : Optional[str]
    intent_meta      : Optional[dict]
    retrieved_chunks : Optional[list]
    reranked_chunks  : Optional[list]
    analysis_brief   : Optional[dict]
    final_answer     : Optional[dict]

# ════════════════════════════════════════════════════════════
# BM25 retriever cache (loaded once per process)
# ════════════════════════════════════════════════════════════

_bm25_retriever: BM25Retriever | None = None

def _get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        try:
            _bm25_retriever = BM25Retriever.load(BM25_INDEX_PATH)
        except FileNotFoundError:
            raise RuntimeError(
                "BM25 index not found. Run ingest_repo() first to build it."
            )
    return _bm25_retriever

# ════════════════════════════════════════════════════════════
# Graph nodes
# ════════════════════════════════════════════════════════════

@traceable(run_type="chain")
def node_intent(state: RAGState) -> RAGState:
    result = classify_intent(state["question"])
    return {**state, "intent": result["intent"], "intent_meta": result}


@traceable(run_type="retriever")
def node_retrieve(state: RAGState) -> RAGState:
    bm25 = _get_bm25_retriever()
    chunks = retrieve(
        question       = state["question"],
        intent         = state["intent"] or "explain",
        bm25_retriever = bm25,
    )
    return {**state, "retrieved_chunks": chunks}


@traceable(run_type="tool")
def node_rerank(state: RAGState) -> RAGState:
    top_chunks = rerank(state["question"], state["retrieved_chunks"] or [], top_k=5)
    return {**state, "reranked_chunks": top_chunks}


@traceable(run_type="chain")
def node_analyse(state: RAGState) -> RAGState:
    brief = analyse(state["question"], state["reranked_chunks"] or [])
    return {**state, "analysis_brief": brief}


@traceable(run_type="chain")
def node_answer(state: RAGState) -> RAGState:
    result = answer(
        state["analysis_brief"] or {},
        intent       = state["intent"] or "explain",
        chat_history = state.get("chat_history", "") or "",
    )
    return {**state, "final_answer": result}

# ════════════════════════════════════════════════════════════
# Build the graph
# ════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(RAGState)  # type: ignore

    graph.add_node("intent",   node_intent)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank",   node_rerank)
    graph.add_node("analyse",  node_analyse)
    graph.add_node("answer",   node_answer)

    graph.set_entry_point("intent")
    graph.add_edge("intent",   "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank",   "analyse")
    graph.add_edge("analyse",  "answer")
    graph.add_edge("answer",   END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("LangGraph RAG pipeline compiled")
    return _compiled_graph

# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

@traceable(run_type="chain")
def ingest_repo(url: str) -> dict:
    """
    Full ingestion pipeline:
    clone → scan → chunk → embed/store → build BM25 index
    """
    repo_path = clone_repo(url)
    files     = scan_repo(repo_path)

    repo_name = url.rstrip("/").split("/")[-1]
    for f in files:
        f["repo"] = repo_name

    chunks = chunk_files(files)
    if not chunks:
        raise ValueError("No chunks were produced — check repo contents and supported extensions.")

    indexed = index_chunks(chunks)

    # Build + persist BM25 index
    bm25 = BM25Retriever(chunks)
    bm25.save(BM25_INDEX_PATH)

    global _bm25_retriever
    _bm25_retriever = bm25   # refresh in-memory cache

    return {
        "repo"           : repo_name,
        "files_scanned"  : len(files),
        "chunks_indexed" : indexed,
        "vector_db_stats": collection_stats(),
    }


@traceable(run_type="chain")
async def query_repo(question: str, session_id: str) -> dict:
    """
    Run the full LangGraph pipeline for a single question,
    with conversation history injected.
    """
    from app.memory.session import get_history, format_history_for_prompt

    history       = await get_history(session_id)
    history_text  = format_history_for_prompt(history)

    graph = _get_graph()

    initial_state: RAGState = {
        "question"         : question,
        "session_id"       : session_id,
        "chat_history"     : history_text,
        "intent"           : None,
        "intent_meta"      : None,
        "retrieved_chunks" : None,
        "reranked_chunks"  : None,
        "analysis_brief"   : None,
        "final_answer"     : None,
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_answer"]

if __name__ == "__main__":
    import asyncio
    
    async def main():
        # quick smoke test (requires repo already ingested)
        result = await query_repo(
            question   = "How does authentication work?",
            session_id = "00000000-0000-0000-0000-000000000000"
        )
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")

    asyncio.run(main())