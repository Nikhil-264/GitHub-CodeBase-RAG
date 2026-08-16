"""
Evaluation Metrics
==================
Implements metrics for retrieval (Precision@k, Recall@k, MRR) and
generation (grounding/faithfulness and utility) by reusing
inference-time grading calls.
"""

from app.agents.critique_agent import check_grounding, check_utility

def is_match(expected: str, retrieved: str) -> bool:
    """Helper to check if expected file path matches retrieved file path."""
    expected_clean = expected.replace("\\", "/").lower().strip("/")
    retrieved_clean = retrieved.replace("\\", "/").lower().strip("/")
    return expected_clean in retrieved_clean or retrieved_clean in expected_clean

def precision_at_k(retrieved_files: list[str], expected_files: list[str], k: int) -> float:
    """Computes Precision@k."""
    if k <= 0 or not retrieved_files:
        return 0.0
    retrieved_k = retrieved_files[:k]
    relevant_found = 0
    # Keep track of expected matched to avoid double counting if needed
    matched_retrieved = set()
    for r in retrieved_k:
        if any(is_match(e, r) for e in expected_files):
            relevant_found += 1
    return relevant_found / k

def recall_at_k(retrieved_files: list[str], expected_files: list[str], k: int) -> float:
    """Computes Recall@k."""
    if not expected_files:
        return 1.0
    retrieved_k = retrieved_files[:k]
    relevant_found = 0
    for e in expected_files:
        if any(is_match(e, r) for r in retrieved_k):
            relevant_found += 1
    return relevant_found / len(expected_files)

def reciprocal_rank(retrieved_files: list[str], expected_files: list[str]) -> float:
    """Computes Reciprocal Rank for a single query."""
    for index, r in enumerate(retrieved_files):
        if any(is_match(e, r) for e in expected_files):
            return 1.0 / (index + 1)
    return 0.0

def compute_generation_metrics(answer_text: str, chunks: list[dict], question: str) -> dict:
    """
    Computes faithfulness and relevancy (utility) scores for the generation.
    Returns:
        {
            "faithfulness": 1.0 if grounded else 0.0,
            "utility": 1.0 if relevant else 0.0
        }
    """
    grounded = check_grounding(answer_text, chunks)
    useful = check_utility(answer_text, question)
    
    return {
        "faithfulness": 1.0 if grounded else 0.0,
        "utility": 1.0 if useful else 0.0
    }
