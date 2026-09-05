"""
Embedder
========
Wraps Google's Gemini embeddings API.
Provides both single-query and batch-document embedding.
"""

import os
import time
from loguru import logger
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

load_dotenv()

# ── Config ───────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")
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
_embedder: GoogleGenerativeAIEmbeddings | None = None

def get_embedder() -> GoogleGenerativeAIEmbeddings:
    """
    Return a cached embedder instance.
    Initialized once, reuse everywhere
    """
    global _embedder
    if _embedder is None:
        api_key = os.getenv("GEMINI_API_KEY")
        logger.info(f"Loading Google Embeddings: {EMBED_MODEL}")

        _embedder = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=api_key,
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
