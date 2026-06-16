"""
Hybrid Retrieval
================
Combines vector search + BM25 using Reciprocal Rank Fusion (RRF).

RRF formula:
    score(d) = Σ  1 / (k + rank(d))
              results

Much better than simple score averaging because it handles
different score scales across the two systems.
"""

from loguru import logger
from app.retrieval.vector_store import vector_search
from app.retrieval.bm25 import BM25Retriever


# ── RRF constant (60 is standard) ────────────────────────────
RRF_K = 60


from langsmith import traceable

@traceable(run_type="retriever")
def hybrid_search(
    question      : str,
    bm25_retriever: BM25Retriever,
    n_results     : int  = 20,
    vector_weight : float = 0.6,
    bm25_weight   : float = 0.4,
    filter_meta   : dict | None = None,
) -> list[dict]:
    """
    Run vector + BM25 search and fuse results with RRF.

    Returns a deduplicated, RRF-ranked list of chunks,
    each with a `rrf_score` and `sources` field showing
    which systems retrieved it.
    """
    # ── Fetch from both systems ──────────────────────────────
    vector_results = vector_search(question, n_results=n_results, filter_meta=filter_meta)
    bm25_results   = bm25_retriever.query(question, n_results=n_results, filter_meta=filter_meta)

    logger.debug(f"Vector: {len(vector_results)} | BM25: {len(bm25_results)}")

    # ── Build RRF score map ──────────────────────────────────
    # key = (file_path, start_line)  →  unique chunk identity
    rrf_scores : dict[str, float] = {}
    chunk_map  : dict[str, dict]  = {}

    def _chunk_key(chunk: dict) -> str:
        m = chunk["metadata"]
        return f"{m['file_path']}::{m['start_line']}"

    # Score from vector results
    for rank, chunk in enumerate(vector_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + vector_weight * (1 / (RRF_K + rank + 1))
        if key not in chunk_map:
            chunk_map[key] = {**chunk, "sources": ["vector"]}
        else:
            if "vector" not in chunk_map[key]["sources"]:
                chunk_map[key]["sources"].append("vector")

    # Score from BM25 results
    for rank, chunk in enumerate(bm25_results):
        key = _chunk_key(chunk)
        rrf_scores[key] = rrf_scores.get(key, 0) + bm25_weight * (1 / (RRF_K + rank + 1))
        if key not in chunk_map:
            chunk_map[key] = {**chunk, "sources": ["bm25"]}
        else:
            if "bm25" not in chunk_map[key]["sources"]:
                chunk_map[key]["sources"].append("bm25")

    # ── Sort by RRF score ────────────────────────────────────
    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in ranked_keys:
        chunk = chunk_map[key]
        chunk["rrf_score"] = round(rrf_scores[key], 6)
        results.append(chunk)

    # ── Log fusion summary ───────────────────────────────────
    both   = sum(1 for c in results if len(c["sources"]) == 2)
    logger.info(
        f"Hybrid search: {len(results)} unique chunks "
        f"({both} found by both systems)"
    )

    return results[:n_results]


def simple_merge(
    question      : str,
    bm25_retriever: BM25Retriever,
    n_results     : int = 20,
    filter_meta   : dict | None = None,
) -> list[dict]:
    """
    Simpler alternative to RRF — just deduplicate and combine.
    Vector results come first (higher priority), BM25 fills the rest.
    Use this if RRF feels like overkill for small repos.
    """
    vector_results = vector_search(question, n_results=n_results, filter_meta=filter_meta)
    bm25_results   = bm25_retriever.query(question, n_results=n_results, filter_meta=filter_meta)

    seen   : set[str]  = set()
    merged : list[dict] = []

    for chunk in vector_results + bm25_results:
        m   = chunk["metadata"]
        key = f"{m['file_path']}::{m['start_line']}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    return merged[:n_results]
