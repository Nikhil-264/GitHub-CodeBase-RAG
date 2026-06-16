"""
Embedder
========
Wraps Ollama's nomic-embed-text model.
Provides both single-query and batch-document embedding.
"""

import os
from loguru import logger
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from typing import List, Tuple
import httpx
import numpy as np

load_dotenv()

# ── Config ───────────────────────────────────────────────────
EMBED_MODEL  = os.getenv("EMBED_MODEL",      "nomic-embed-text")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")

# ── Singleton ────────────────────────────────────────────────
_embedder: OllamaEmbeddings | None = None

def get_embedder() -> OllamaEmbeddings:
    """
    Return a cached embedder instance.
    Initialized once, resuse everywhere
    """

    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedder : {EMBED_MODEL} at {OLLAMA_URL}")

        _embedder = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=OLLAMA_URL,
        )
    
    return _embedder

def embed_query(question : str) -> list[float]:
    """ Embed a single search query"""
    return get_embedder().embed_query(question)

def embed_documents(texts : list[str]) -> list[list[float]]:
    """
    Embed a list of documents in batches.
    Ollama handles one at a time internally - we batch here to avoid sending thousands of individual requests.
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
