"""
Embedder
========
Embeds documents/queries via either Google's Gemini embeddings API or a
local Ollama instance, switched by EMBED_BACKEND — same pattern as
app/llm_provider.py for the chat LLM and app/reranker/reranker.py's
RERANKER_BACKEND.
Provides both single-query and batch-document embedding.
"""

import os
import time
from loguru import logger
from dotenv import load_dotenv
from typing import List

load_dotenv()

# ── Config ───────────────────────────────────────────────────
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "gemini")   # "gemini" or "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_DEFAULT_EMBED_MODELS = {
    "gemini": "text-embedding-004",
    "ollama": "nomic-embed-text",
}
# Same rule as LLM_MODEL in llm_provider.py: an explicit EMBED_MODEL always
# wins; otherwise pick the default that actually matches the active backend.
EMBED_MODEL = os.getenv("EMBED_MODEL") or _DEFAULT_EMBED_MODELS.get(EMBED_BACKEND, _DEFAULT_EMBED_MODELS["gemini"])

EMBED_MAX_RETRIES = int(os.getenv("EMBED_MAX_RETRIES", "6"))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _call_with_retry(fn, *args, **kwargs):
    """
    Call an embedding function, retrying with backoff on Gemini free-tier
    rate limits (429 RESOURCE_EXHAUSTED). The free tier resets per-minute,
    so backoff grows toward ~60s rather than the sub-second delay Gemini
    reports in the error (which refers to its own internal retry, not ours).
    """
    for attempt in range(EMBED_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limit_error(e) or attempt == EMBED_MAX_RETRIES - 1:
                raise
            wait_sec = min(60, 5 * (2 ** attempt))
            logger.warning(
                f"Gemini embedding rate limit hit (attempt {attempt + 1}/{EMBED_MAX_RETRIES}). "
                f"Retrying in {wait_sec}s..."
            )
            time.sleep(wait_sec)

# ── Singleton ────────────────────────────────────────────────
_embedder = None

def get_embedder():
    """
    Return a cached embedder instance per EMBED_BACKEND.
    Initialized once, reuse everywhere.
    """
    global _embedder
    if _embedder is None:
        if EMBED_BACKEND == "ollama":
            from langchain_ollama import OllamaEmbeddings
            logger.info(f"Loading Ollama Embeddings: {EMBED_MODEL} @ {OLLAMA_BASE_URL}")
            _embedder = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            logger.info(f"Loading Google Embeddings: {EMBED_MODEL}")
            _embedder = GoogleGenerativeAIEmbeddings(
                model=EMBED_MODEL,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )

    return _embedder

def embed_query(question : str) -> list[float]:
    """ Embed a single search query"""
    return _call_with_retry(get_embedder().embed_query, question)

def embed_documents(texts : list[str]) -> list[list[float]]:
    """
    Embed a list of documents in batches.
    Batching prevents API timeouts on large inputs, and keeps each batch as
    one Gemini API request — fewer, larger batches use the free-tier's
    per-minute request quota more efficiently than many small ones.
    """
    embedder = get_embedder()

    batch_size = 100
    all_embeddings : list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug(f"Embedding Batch {i // batch_size + 1} ({len(batch)} docs)")
        all_embeddings.extend(_call_with_retry(embedder.embed_documents, batch))

    return all_embeddings


if __name__ == '__main__':
    vec = embed_query("how does authentication work?")
    print(f"Query embedding dim : {len(vec)}")
    print(f"First 5 values : {vec[:5]}...")
