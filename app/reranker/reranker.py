"""
Reranker
========
Takes the top-N chunks from hybrid retrieval and reranks them
using a cross-encoder model (much more accurate than bi-encoder similarity).

Pipeline:
    Hybrid Search (top 30)
        │
        ▼
    Cross-Encoder reranker
        │
        ▼
    Top K chunks (default 5)
        │
        ▼
    LLM

Two backends supported:
    Backend A → sentence-transformers CrossEncoder  (local, recommended)
    Backend B → Ollama LLM-based reranker           (fallback, no extra model needed)
"""

import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "cross_encoder")   # or "ollama"
TOP_K_DEFAULT   = int(os.getenv("RERANKER_TOP_K", "10"))


# ════════════════════════════════════════════════════════════
# Backend A — CrossEncoder (sentence-transformers)
# ════════════════════════════════════════════════════════════

_cross_encoder = None


def _load_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.success(f"CrossEncoder loaded: {RERANKER_MODEL}")
        return _cross_encoder
    except Exception as e:
        logger.warning(f"CrossEncoder unavailable: {e}")
        return None


def _rerank_cross_encoder(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    model = _load_cross_encoder()
    if model is None:
        return []

    pairs = [(question, str(c.get("text", ""))) for c in chunks]
    scores = model.predict(pairs)  # type: ignore

    ranked = sorted(
        zip(chunks, scores),
        key     = lambda x: float(x[1]),
        reverse = True,
    )

    results = []
    for chunk, score in ranked[:top_k]:
        results.append({
            **chunk,
            "rerank_score"   : round(float(score), 4),
            "rerank_backend" : "cross_encoder",
        })

    return results


# ════════════════════════════════════════════════════════════
# Backend B — Ollama LLM-based reranker (fallback)
# ════════════════════════════════════════════════════════════

def _rerank_ollama(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Ask the LLM to score each chunk's relevance to the question.
    Slower than CrossEncoder but needs no extra model download.
    Used only when CrossEncoder is unavailable.
    """
    import os
    from langchain_ollama import OllamaLLM

    llm = OllamaLLM(
        model    = os.getenv("LLM_MODEL", "qwen3"),
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature = 0,
    )

    scored = []
    for i, chunk in enumerate(chunks):
        prompt = f"""Rate how relevant this code chunk is for answering the question.
Respond with ONLY a number between 0 and 10. Nothing else.

Question: {question}

Code chunk:
{chunk['text'][:800]}

Relevance score (0-10):"""

        try:
            response = llm.invoke(prompt).strip()
            # extract first number found in response
            import re
            match = re.search(r"\d+(\.\d+)?", response)
            score = float(match.group()) if match else 0.0
            score = max(0.0, min(10.0, score))   # clamp 0–10
        except Exception as e:
            logger.warning(f"Ollama reranker failed on chunk {i}: {e}")
            score = 0.0

        scored.append((chunk, score))
        logger.debug(f"Chunk {i+1}/{len(chunks)} scored: {score}")

    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return [
        {**chunk, "rerank_score": round(score, 4), "rerank_backend": "ollama"}
        for chunk, score in ranked[:top_k]
    ]


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="tool")
def rerank(
    question : str,
    chunks   : list[dict],
    top_k    : int = TOP_K_DEFAULT,
) -> list[dict]:
    """
    Rerank chunks against the question.
    Tries CrossEncoder first, falls back to Ollama, then falls back
    to returning the first top_k chunks unchanged.

    Args:
        question : the user's original question
        chunks   : output of hybrid_search (typically 20–30 chunks)
        top_k    : how many chunks to return after reranking

    Returns:
        top_k chunks sorted by relevance, each with rerank_score added
    """
    if not chunks:
        return []

    if len(chunks) <= top_k:
        logger.debug("Fewer chunks than top_k — skipping reranker.")
        return chunks

    logger.info(f"Reranking {len(chunks)} chunks → top {top_k}  [{RERANKER_BACKEND}]")

    # ── Try CrossEncoder ─────────────────────────────────────
    if RERANKER_BACKEND == "cross_encoder":
        results = _rerank_cross_encoder(question, chunks, top_k)
        if results:
            _log_summary(results)
            return results
        logger.warning("CrossEncoder failed — trying Ollama reranker")

    # ── Try Ollama ───────────────────────────────────────────
    if RERANKER_BACKEND in ("cross_encoder", "ollama"):
        results = _rerank_ollama(question, chunks, top_k)
        if results:
            _log_summary(results)
            return results
        logger.warning("Ollama reranker failed — returning top_k chunks unranked")

    # ── Last resort: slice ───────────────────────────────────
    logger.warning("All reranker backends failed — returning first top_k chunks")
    return chunks[:top_k]


def _log_summary(results: list[dict]) -> None:
    logger.info(f"Reranker kept {len(results)} chunks:")
    for i, c in enumerate(results, 1):
        m = c["metadata"]
        logger.info(
            f"  [{i}] score={c.get('rerank_score', '?'):<6}  "
            f"{m['file_path']}:{m['start_line']}–{m['end_line']}  "
            f"({m['chunk_type']})"
        )


# ════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    dummy_chunks = [
        {
            "text": "def generate_jwt(user_id): payload = {'sub': user_id} return jwt.encode(payload, SECRET_KEY)",
            "metadata": {"file_path": "auth/jwt.py", "start_line": 10, "end_line": 15, "chunk_type": "function_definition"},
            "rrf_score": 0.02,
            "sources": ["vector", "bm25"],
        },
        {
            "text": "def login(username, password): user = db.get_user(username) if not verify(password, user.hash): raise AuthError",
            "metadata": {"file_path": "auth/login.py", "start_line": 5, "end_line": 12, "chunk_type": "function_definition"},
            "rrf_score": 0.018,
            "sources": ["vector"],
        },
        {
            "text": "DATABASES = {'default': {'ENGINE': 'django.db.backends.postgresql', 'NAME': 'mydb'}}",
            "metadata": {"file_path": "settings.py", "start_line": 80, "end_line": 90, "chunk_type": "sliding_window"},
            "rrf_score": 0.01,
            "sources": ["bm25"],
        },
    ]

    question = "Where is JWT generated?"
    top = rerank(question, dummy_chunks, top_k=2)

    print(f"\nTop {len(top)} chunks for: '{question}'\n")
    for i, c in enumerate(top, 1):
        print(f"  [{i}] {c['metadata']['file_path']} | score={c.get('rerank_score', 'N/A')}")