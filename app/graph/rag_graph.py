"""
RAG Graph (LangGraph)
======================
Wires all agents together into a stateful graph with CRAG and Self-RAG upgrades:

    Question
       │
       ▼
   Intent Agent
       │
       ├──► Needs Retrieval? (No) ──► Direct Answer ──► END
       │
       └──► (Yes) ──► Retrieval Agent
                          │
                          ▼
                       Reranker
                          │
                          ▼
                    Grade Chunks (CRAG)
                          │
                          ▼
                       Correct (CRAG Decision)
                          │
                          ├──► Irrelevant (Attempts < 1) ──► Rewrite & Loop to Retrieval
                          │
                          └──► Proceed ──► Analysis Agent
                                               │
                                               ▼
                                          Answer Agent ◄──────┐ (Self-RAG Retry)
                                               │              │
                                               ▼              │
                                         Self-Critique ───────┘
                                               │
                                               ▼
                                              END
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

# Import new agents/functions
from app.agents.grader_agent import check_need_retrieval, grade_chunk_relevance
from app.agents.critique_agent import check_grounding, check_utility

load_dotenv()

BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "repos/bm25_index.pkl")

# ════════════════════════════════════════════════════════════
# Graph State
# ════════════════════════════════════════════════════════════

class RAGState(TypedDict):
    question            : str
    original_question   : Optional[str]          # Preserves original query during rewrite
    session_id          : str
    repo_name           : Optional[str]          # Target repository name for scoped retrieval
    chat_history        : Optional[str]          # Formatted past turns
    intent              : Optional[str]
    intent_meta         : Optional[dict]
    retrieved_chunks    : Optional[list]
    reranked_chunks     : Optional[list]
    analysis_brief      : Optional[dict]
    final_answer        : Optional[dict]
    # NEW fields for CRAG and Self-RAG
    correction_attempts : int
    critique_attempts   : int
    needs_retrieval     : bool
    chunk_grades        : Optional[list[str]]
    stricter_prompt     : bool
    needs_regeneration  : bool
    strict_mode         : bool          # NEW — when False, skips CRAG correction + Self-RAG critique loops for speed

# ════════════════════════════════════════════════════════════
# BM25 retriever cache (loaded once per repo)
# ════════════════════════════════════════════════════════════

_bm25_retrievers: dict[str, BM25Retriever] = {}
_fallback_bm25_retriever: BM25Retriever | None = None

def _get_bm25_retriever(repo_name: str | None = None) -> BM25Retriever:
    global _bm25_retrievers, _fallback_bm25_retriever

    if repo_name:
        if repo_name in _bm25_retrievers:
            return _bm25_retrievers[repo_name]
        
        repo_bm25_path = f"repos/bm25_{repo_name}.pkl"
        if os.path.exists(repo_bm25_path):
            try:
                retriever = BM25Retriever.load(repo_bm25_path)
                _bm25_retrievers[repo_name] = retriever
                return retriever
            except Exception as e:
                logger.warning(f"Could not load BM25 index for {repo_name}: {e}")

    # Fallback to general index or default
    if _fallback_bm25_retriever is None:
        try:
            _fallback_bm25_retriever = BM25Retriever.load(BM25_INDEX_PATH)
        except FileNotFoundError:
            raise RuntimeError(
                f"BM25 index not found for repo '{repo_name}'. Run ingest_repo() first to build it."
            )
    return _fallback_bm25_retriever

# ════════════════════════════════════════════════════════════
# Graph nodes
# ════════════════════════════════════════════════════════════

@traceable(run_type="chain")
def node_intent(state: RAGState) -> RAGState:
    result = classify_intent(state["question"])
    needs_ret = check_need_retrieval(state["question"])
    return {
        **state,
        "intent": result["intent"],
        "intent_meta": result,
        "needs_retrieval": needs_ret,
        "original_question": state.get("original_question") or state["question"]
    }


@traceable(run_type="chain")
def node_direct_answer(state: RAGState) -> RAGState:
    from app.agents.answer_agent import _get_llm
    llm = _get_llm()
    prompt = f"""You are an expert programming assistant. Answer the user's question directly.
No codebase context is needed to answer this question.

Conversation History:
{state.get("chat_history", "") or "No previous history."}

Question: {state["question"]}

Answer:"""
    response = llm.invoke(prompt).strip()
    result = {
        "answer": response,
        "sources": [],
        "chunks_used": 0,
        "intent": state["intent"] or "explain",
        "primary_files": [],
    }
    return {**state, "final_answer": result}


@traceable(run_type="retriever")
def node_retrieve(state: RAGState) -> RAGState:
    repo_name = state.get("repo_name")
    filter_meta = {"repo": repo_name} if repo_name else None

    bm25 = _get_bm25_retriever(repo_name)
    chunks = retrieve(
        question       = state["question"],
        intent         = state["intent"] or "explain",
        bm25_retriever = bm25,
        filter_meta    = filter_meta,
    )
    return {**state, "retrieved_chunks": chunks}


@traceable(run_type="tool")
def node_rerank(state: RAGState) -> RAGState:
    top_chunks = rerank(state["question"], state["retrieved_chunks"] or [], top_k=5)
    return {**state, "reranked_chunks": top_chunks}


@traceable(run_type="chain")
def node_grade_chunks(state: RAGState) -> RAGState:
    chunks = state.get("reranked_chunks") or []
    
    # If not in strict mode, skip grading chunks
    if not state.get("strict_mode", True):
        logger.info("CRAG: strict_mode is False, skipping chunk grading.")
        # Mark all chunks as relevant to bypass correction
        return {**state, "chunk_grades": ["relevant"] * len(chunks)}

    question = state.get("original_question") or state["question"]
    
    logger.info(f"CRAG: Grading {len(chunks)} chunks...")
    grades = []
    for c in chunks:
        grade = grade_chunk_relevance(question, c["text"])
        grades.append(grade)
    
    logger.info(f"CRAG: Chunk grades: {grades}")
    return {**state, "chunk_grades": grades}


@traceable(run_type="chain")
def node_correct(state: RAGState) -> RAGState:
    chunks = state.get("reranked_chunks") or []
    grades = state.get("chunk_grades") or []
    
    # Filter only relevant or ambiguous chunks
    filtered_chunks = []
    for chunk, grade in zip(chunks, grades):
        if grade in ("relevant", "ambiguous"):
            filtered_chunks.append(chunk)
            
    # Check if we have any relevant chunks
    has_relevant = len(filtered_chunks) > 0
    
    if not has_relevant and state["correction_attempts"] < 1:
        # We need to retry! Rewrite query.
        from app.agents.answer_agent import _get_llm
        llm = _get_llm()
        question = state.get("original_question") or state["question"]
        prompt = f"""You are an expert query optimizer. The user is asking a question about a codebase, but initial retrieval failed to find relevant files.
Rewrite the question to focus on code keywords, API names, functions, or specific codebase details that will improve search retrieval.
Do not include any greeting or explanation. Only return the rewritten query.

Original Question: {question}

Rewritten Query:"""
        rewritten_query = llm.invoke(prompt).strip()
        logger.info(f"CRAG: No relevant chunks found. Rewriting query from '{question}' to '{rewritten_query}' and retrying.")
        
        return {
            **state,
            "question": rewritten_query,
            "correction_attempts": state["correction_attempts"] + 1,
            "retrieved_chunks": None,
            "reranked_chunks": None,
            "chunk_grades": None
        }
    else:
        # Proceed with either filtered chunks or empty chunks
        orig_q = state.get("original_question") or state["question"]
        logger.info(f"CRAG: Proceeding with {len(filtered_chunks)} chunks. Restoring query to original: '{orig_q}'")
        return {
            **state,
            "question": orig_q,
            "reranked_chunks": filtered_chunks
        }


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
        stricter     = state.get("stricter_prompt", False),
    )
    return {**state, "final_answer": result}


@traceable(run_type="chain")
def node_self_critique(state: RAGState) -> RAGState:
    # If not in strict mode, skip self critique
    if not state.get("strict_mode", True):
        logger.info("Self-RAG: strict_mode is False, skipping self-critique.")
        return {
            **state,
            "needs_regeneration": False
        }

    answer_text = state["final_answer"]["answer"] if state.get("final_answer") else ""
    chunks = state.get("reranked_chunks") or []
    question = state.get("original_question") or state["question"]
    
    # 1. Grounding check
    grounded = check_grounding(answer_text, chunks)
    # 2. Utility check
    utility = check_utility(answer_text, question)
    
    logger.info(f"Self-RAG: Grounded={grounded}, Utility={utility}")
    
    if (not grounded or not utility) and state["critique_attempts"] < 1:
        if not grounded:
            logger.warning("Self-RAG: Hallucination detected! Retrying generation with stricter prompt.")
        else:
            logger.warning("Self-RAG: Answer did not address the question. Retrying generation with stricter prompt.")
        return {
            **state,
            "stricter_prompt": True,
            "needs_regeneration": True,
            "critique_attempts": state["critique_attempts"] + 1
        }
    
    return {
        **state,
        "needs_regeneration": False
    }

# ════════════════════════════════════════════════════════════
# Conditional Router Functions
# ════════════════════════════════════════════════════════════

def route_after_intent(state: RAGState) -> str:
    if not state.get("needs_retrieval", True):
        logger.info("Self-RAG Gate: Skipping retrieval, routing to direct answer.")
        return "direct_answer"
    return "retrieve"


def route_after_correct(state: RAGState) -> str:
    # If chunk_grades is cleared, it means we are retrying retrieval
    if state.get("chunk_grades") is None and state["correction_attempts"] > 0:
        return "retrieve"
    return "analyse"


def route_after_critique(state: RAGState) -> str:
    if state.get("needs_regeneration"):
        return "answer"
    return "end"

# ════════════════════════════════════════════════════════════
# Build the graph
# ════════════════════════════════════════════════════════════

def build_graph():
    graph = StateGraph(RAGState)  # type: ignore

    graph.add_node("intent",        node_intent)
    graph.add_node("direct_answer", node_direct_answer)
    graph.add_node("retrieve",      node_retrieve)
    graph.add_node("rerank",        node_rerank)
    graph.add_node("grade_chunks",  node_grade_chunks)
    graph.add_node("correct",       node_correct)
    graph.add_node("analyse",       node_analyse)
    graph.add_node("answer",        node_answer)
    graph.add_node("self_critique", node_self_critique)

    graph.set_entry_point("intent")
    
    # Retrieve-or-not gate
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "retrieve": "retrieve",
            "direct_answer": "direct_answer"
        }
    )
    
    # Direct answer flow terminates directly
    graph.add_edge("direct_answer", END)

    # Retrieval flow
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank",   "grade_chunks")
    graph.add_edge("grade_chunks", "correct")
    
    # CRAG correction loop
    graph.add_conditional_edges(
        "correct",
        route_after_correct,
        {
            "retrieve": "retrieve",
            "analyse": "analyse"
        }
    )
    
    graph.add_edge("analyse",  "answer")
    graph.add_edge("answer",   "self_critique")
    
    # Self-RAG critique loop
    graph.add_conditional_edges(
        "self_critique",
        route_after_critique,
        {
            "answer": "answer",
            "end": END
        }
    )

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
        logger.info("LangGraph RAG pipeline compiled with CRAG & Self-RAG")
    return _compiled_graph

# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

@traceable(run_type="chain")
def ingest_repo(url: str) -> dict:
    """
    Full ingestion pipeline:
    clone → scan → chunk → embed/store → build per-repo BM25 index
    """
    repo_path = clone_repo(url)
    files     = scan_repo(repo_path)

    repo_name = url.rstrip("/").split("/")[-1].removesuffix(".git")
    for f in files:
        f["repo"] = repo_name

    chunks = chunk_files(files)
    if not chunks:
        raise ValueError("No chunks were produced — check repo contents and supported extensions.")

    indexed = index_chunks(chunks)

    # Build + persist per-repo BM25 index
    bm25 = BM25Retriever(chunks)
    repo_bm25_path = f"repos/bm25_{repo_name}.pkl"
    bm25.save(repo_bm25_path)
    bm25.save(BM25_INDEX_PATH)

    global _bm25_retrievers, _fallback_bm25_retriever
    _bm25_retrievers[repo_name] = bm25   # refresh in-memory cache
    _fallback_bm25_retriever = bm25

    return {
        "repo"           : repo_name,
        "files_scanned"  : len(files),
        "chunks_indexed" : indexed,
        "vector_db_stats": collection_stats(),
    }


@traceable(run_type="chain")
async def query_repo(
    question: str,
    session_id: str,
    repo_name: str | None = None,
    strict_mode: bool = True,
) -> dict:
    """
    Run the full LangGraph pipeline for a single question,
    with conversation history and repository scope injected.
    """
    from app.memory.session import get_history, format_history_for_prompt

    history       = await get_history(session_id)
    history_text  = format_history_for_prompt(history)

    graph = _get_graph()

    initial_state: RAGState = {
        "question"         : question,
        "original_question": question,
        "session_id"       : session_id,
        "repo_name"        : repo_name,
        "chat_history"     : history_text,
        "intent"           : None,
        "intent_meta"      : None,
        "retrieved_chunks" : None,
        "reranked_chunks"  : None,
        "analysis_brief"   : None,
        "final_answer"     : None,
        "correction_attempts": 0,
        "critique_attempts"  : 0,
        "needs_retrieval"    : True,
        "chunk_grades"       : None,
        "stricter_prompt"    : False,
        "needs_regeneration" : False,
        "strict_mode"        : strict_mode,
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