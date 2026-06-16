from __future__ import annotations
"""
Vector Store
=============
Persists chunks + embeddings in ChromaDB.
Handles upsert (skips already-indexed chunks via content hash).
"""

import os
import hashlib
from loguru import logger
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings

from app.embeddings.embedder import embed_documents, embed_query

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = 'codebase'

# ── Client singleton ─────────────────────────────────────────
_client: chromadb.PersistentClient | None = None  # type: ignore


def get_client() -> chromadb.PersistentClient:  # type: ignore
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection(name : str = COLLECTION_NAME) -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine"
        }
    )


from langsmith import traceable

@traceable(run_type="tool")
def index_chunks(chunks : list[dict], collection_name : str = COLLECTION_NAME) -> int:
    """
    Embed and upsert chunks into ChromaDB.
    Uses MD5 of chunk text as the ID — duplicate chunks are silently skipped.
    Returns number of chunks upserted.
    """
    
    if not chunks:
        logger.warning("Index_chunks called with empty list.")
        return 0

    collection = get_collection(collection_name)

    texts : list[str] = []
    metadatas : list[dict] = []
    ids : list[str] = []
    seen_ids = set()

    for chunk in chunks:
        chunk_id = hashlib.md5(chunk['text'].encode()).hexdigest()
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        
        texts.append(chunk['text'])
        metadatas.append(_sanitize_metadata(chunk['metadata']))
        ids.append(chunk_id)

    logger.info(f"Generating embeddings for {len(texts)} unique chunks")
    embeddings = embed_documents(texts)
    collection.upsert(
        documents = texts,
        embeddings = embeddings,  # type: ignore
        metadatas = metadatas,    # type: ignore
        ids = ids,
    )

    logger.success(f"Indexed {len(texts)} chunks into collections '{collection_name}'")
    return len(texts)        


def _sanitize_metadata(meta: dict) -> dict:
    """
    ChromaDB only accepts str / int / float / bool in metadata.
    Convert anything else to string.
    """
    clean = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


@traceable(run_type="retriever")
def vector_search(
    question : str,
    n_results : int = 20,
    collection_name : str = COLLECTION_NAME,
    filter_meta : dict | None = None,
) -> list[dict]:
    """
    semantic vector search. 
    Returns chunks sorted by cosine similarity(highest first).

    Optional `filter_meta` example:
        {"language": "py"}   →  only Python files
        {"repo": "langgraph"} →  only from a specific repo
    """

    collection = get_collection(collection_name)
    q_embedding = embed_query(question)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n_results,
        where=filter_meta,
        include=['documents', 'metadatas', 'distances']
    )

    chunks = []
    
    # Safely retrieve values to avoid None-type subscriptable warning in Pyrefly
    documents = results.get('documents')
    metadatas = results.get('metadatas')
    distances = results.get('distances')

    if documents and metadatas and distances:
        for doc, meta, dist in zip(
            documents[0],
            metadatas[0],
            distances[0],
        ):
            if doc is None or meta is None:
                continue
            chunks.append({
                "text": doc,
                "metadata": meta,
                "score" : round(1 - dist, 4),
                "source" : "vector",
            })
    
    logger.info(f"Retrieved {len(chunks)} chunks via semantic search for {question[:60]}")
    return chunks

def collection_stats(collection_name : str = COLLECTION_NAME) -> dict:
    """Return basic stats about the collection"""
    col = get_collection(collection_name)
    return {
        'collection' : collection_name,
        'count' : col.count(),
        'path' : CHROMA_PATH,
    }

def delete_collection(collection_name : str = COLLECTION_NAME) -> None:
    """Wipe a collection entirely (useful for reindexing from scratch)"""
    get_client().delete_collection(collection_name)
    logger.warning(f"Deleted collection '{collection_name}'")

if __name__ == '__main__':
    stats = collection_stats()
    print(f"Collection stats : {stats}")
    