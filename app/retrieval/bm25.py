"""
BM25 Retriever
==============
Lexical keyword search over all indexed chunks.
Complements vector search for exact identifiers,
function names, and file paths.
"""

import pickle
from pathlib import Path
from loguru import logger
from rank_bm25 import BM25Okapi  # type: ignore

# Tokenizer

def tokenize(text : str) -> list[str]:
    """ Simple tokenize that handles both prose and code well
    Splits on whitespace + common code delimiters
    Lowercases for case sensitive matching
    """
    import re
    tokens = re.split(r"[\s\(\)\[\]{}<>,\.;:=\+\-\*/\\\"\'`#@!&\|]+", text.lower())
    return [t for t in tokens if len(t) > 1]  

# ── BM25 Retriever class ─────────────────────────────────────

class BM25Retriever:
    """
    Build a BM25 index from a list of chunk dicts.

    Usage:
        retriever = BM25Retriever(chunks)
        results   = retriever.query("JWT token generation", n_results=10)
    """
    def __init__(self, chunks : list[dict]):
        if not chunks:
            raise ValueError("cannot build bm25 index from empty chunk list")
            
        self.chunks = chunks
        tokenized = [tokenize(c['text']) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(chunks)} chunks")

    def query(self, question : str, n_results:int = 20, filter_meta: dict | None = None)-> list[dict]:
        """
        Search the BM25 index.
        Returns top-n chunks with bm25_score attached.
        Optional filter_meta applies simple metadata matching (e.g. {"language": "py"}).
        """
        tokens = tokenize(question)
        if not tokens:
            return []

        scores      = self.bm25.get_scores(tokens)
        sorted_indices = sorted(
            range(len(scores)),
            key     = lambda i: scores[i],
            reverse = True,
        )

        results = []
        for i in sorted_indices:
            if scores[i] <= 0:
                continue                        # skip zero-score results
                
            chunk = self.chunks[i]
            
            # Apply metadata filtering if specified
            if filter_meta:
                meta = chunk.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filter_meta.items()):
                    continue

            results.append({
                **chunk,
                "bm25_score" : round(float(scores[i]), 4),
                "source"     : "bm25",
            })
            
            if len(results) >= n_results:
                break

        logger.debug(f"BM25 search: {len(results)} results for '{question[:60]}'")
        return results

    # ── Persistence ──────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the BM25 index to disk for reuse."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        logger.info(f"BM25 index saved to '{path}'")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """Load a previously saved BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance        = cls.__new__(cls)
        instance.chunks = data["chunks"]
        instance.bm25   = data["bm25"]
        logger.info(f"BM25 index loaded from '{path}' ({len(instance.chunks)} chunks)")
        return instance