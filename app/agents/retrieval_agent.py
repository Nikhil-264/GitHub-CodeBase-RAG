"""
Retrieval Agent
===============
Uses the classified intent to decide HOW to retrieve chunks.

Different intents trigger different retrieval strategies:

    code_search   →  hybrid search, filter by language if possible
    explain       →  hybrid search, broader n_results
    trace_flow    →  hybrid search, favour cross-file results
    architecture  →  vector only, high n_results for broad coverage
    debug         →  hybrid search, prioritise exact keyword matches
"""

import os
from loguru import logger
from dotenv import load_dotenv

from app.retrieval.vector_store import vector_search
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid import hybrid_search

load_dotenv()

# How many chunks to fetch before reranking
RETRIEVAL_N = int(os.getenv("RETRIEVAL_N", "30"))


# ════════════════════════════════════════════════════════════
# Strategy map
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="retriever")
def retrieve(
    question       : str,
    intent         : str,
    bm25_retriever : BM25Retriever,
    filter_meta    : dict | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks based on the classified intent.

    Args:
        question       : original user question
        intent         : output of classify_intent()["intent"]
        bm25_retriever : BM25Retriever instance built at index time
        filter_meta    : optional ChromaDB metadata filter
                         e.g. {"language": "py"} or {"repo": "langgraph"}

    Returns:
        list of chunk dicts ready for reranking
    """
    logger.info(f"Retrieval strategy for intent: [{intent}]")

    if intent == "code_search":
        return _strategy_code_search(question, bm25_retriever, filter_meta)

    elif intent == "explain":
        return _strategy_explain(question, bm25_retriever, filter_meta)

    elif intent == "trace_flow":
        return _strategy_trace_flow(question, bm25_retriever, filter_meta)

    elif intent == "architecture":
        return _strategy_architecture(question, filter_meta)

    elif intent == "debug":
        return _strategy_debug(question, bm25_retriever, filter_meta)

    else:
        # unknown intent — safe default
        logger.warning(f"Unknown intent '{intent}' — using hybrid search")
        return hybrid_search(question, bm25_retriever, n_results=RETRIEVAL_N)


# ════════════════════════════════════════════════════════════
# Retrieval strategies
# ════════════════════════════════════════════════════════════

def _strategy_code_search(
    question       : str,
    bm25_retriever : BM25Retriever,
    filter_meta    : dict | None,
) -> list[dict]:
    """
    Hybrid search with higher BM25 weight.
    Code search benefits from exact identifier matching.
    """
    return hybrid_search(
        question       = question,
        bm25_retriever = bm25_retriever,
        n_results      = RETRIEVAL_N,
        vector_weight  = 0.4,
        bm25_weight    = 0.6,     # favour exact keyword matching
    )


def _strategy_explain(
    question       : str,
    bm25_retriever : BM25Retriever,
    filter_meta    : dict | None,
) -> list[dict]:
    """
    Balanced hybrid search with broader coverage.
    Explanations need semantic context, not just exact matches.
    """
    return hybrid_search(
        question       = question,
        bm25_retriever = bm25_retriever,
        n_results      = RETRIEVAL_N,
        vector_weight  = 0.6,
        bm25_weight    = 0.4,
    )


def _strategy_trace_flow(
    question       : str,
    bm25_retriever : BM25Retriever,
    filter_meta    : dict | None,
) -> list[dict]:
    """
    Hybrid search then deduplicate by file so we get
    a broader cross-file view of the flow.
    """
    chunks = hybrid_search(
        question       = question,
        bm25_retriever = bm25_retriever,
        n_results      = RETRIEVAL_N,
        vector_weight  = 0.5,
        bm25_weight    = 0.5,
    )

    # Keep at most 2 chunks per file so the LLM sees more files
    return _cap_per_file(chunks, max_per_file=2)


def _strategy_architecture(
    question    : str,
    filter_meta : dict | None,
) -> list[dict]:
    """
    Vector-only with high n_results for broad semantic coverage.
    Architecture questions need wide context across many files.
    BM25 keyword matching is less useful here.
    """
    return vector_search(
        question    = question,
        n_results   = RETRIEVAL_N,
        filter_meta = filter_meta,
    )


def _strategy_debug(
    question       : str,
    bm25_retriever : BM25Retriever,
    filter_meta    : dict | None,
) -> list[dict]:
    """
    Hybrid search with heavy BM25 weight.
    Debug questions often contain specific error messages,
    variable names, or identifiers — exact matching matters most.
    """
    return hybrid_search(
        question       = question,
        bm25_retriever = bm25_retriever,
        n_results      = RETRIEVAL_N,
        vector_weight  = 0.3,
        bm25_weight    = 0.7,     # heavily favour exact matches
    )


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _cap_per_file(chunks: list[dict], max_per_file: int = 2) -> list[dict]:
    """Limit chunks per file to encourage cross-file diversity."""
    file_counts : dict[str, int] = {}
    result      : list[dict]     = []

    for chunk in chunks:
        fp = chunk["metadata"]["file_path"]
        file_counts[fp] = file_counts.get(fp, 0) + 1
        if file_counts[fp] <= max_per_file:
            result.append(chunk)

    logger.debug(f"Cap per file ({max_per_file}): {len(chunks)} → {len(result)} chunks")
    return result


if __name__ == "__main__":
    print("Retrieval agent loaded. BM25Retriever must be passed in at runtime.")