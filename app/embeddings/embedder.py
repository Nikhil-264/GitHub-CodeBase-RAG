"""
Embedder
========
Wraps Google's Gemini embeddings API.
Provides both single-query and batch-document embedding.
"""

import os
from loguru import logger
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List

load_dotenv()

# ── Config ───────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

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
    return get_embedder().embed_query(question)

def embed_documents(texts : list[str]) -> list[list[float]]:
    """
    Embed a list of documents in batches.
    Batching prevents API timeouts on large inputs.
    """
    embedder = get_embedder()

    batch_size = 32
    all_embeddings : list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        logger.debug(f"Embedding Batch {i // batch_size + 1} ({len(batch)} docs)")
        all_embeddings.extend(embedder.embed_documents(batch))

    return all_embeddings


if __name__ == '__main__':
    vec = embed_query("how does authentication work?")
    print(f"Query embedding dim : {len(vec)}")
    print(f"First 5 values : {vec[:5]}...")
