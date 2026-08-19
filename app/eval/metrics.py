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

def context_precision(retrieved_files: list[str], expected_files: list[str]) -> float:
    """
    Computes RAGAS-style Context Precision:
    Ratio of relevant items in the retrieved context weighted by rank.
    """
    if not retrieved_files:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    for i, file_path in enumerate(retrieved_files, 1):
        if any(is_match(e, file_path) for e in expected_files):
            hits += 1
            sum_precision += hits / i
            
    return sum_precision / len(retrieved_files) if hits > 0 else 0.0

def context_recall(retrieved_files: list[str], expected_files: list[str]) -> float:
    """
    Computes RAGAS-style Context Recall:
    Ratio of expected ground-truth files successfully retrieved in context.
    """
    if not expected_files:
        return 1.0
    if not retrieved_files:
        return 0.0
        
    found_count = sum(
        1 for e in expected_files 
        if any(is_match(e, r) for r in retrieved_files)
    )
    return found_count / len(expected_files)

def accuracy_score(y_true: list, y_pred: list) -> float:
    """Computes exact classification accuracy between two lists."""
    if not y_true or len(y_true) != len(y_pred):
        return 0.0
    matches = sum(1 for true_val, pred_val in zip(y_true, y_pred) if true_val == pred_val)
    return matches / len(y_true)

# ════════════════════════════════════════════════════════════
# DeepEval Metric Factory (Powered by GoogleGeminiJudge)
# ════════════════════════════════════════════════════════════

from app.eval.deepeval_judge import GoogleGeminiJudge

_judge_instance = None

def get_judge() -> GoogleGeminiJudge:
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = GoogleGeminiJudge()
    return _judge_instance

def get_faithfulness_metric(threshold: float = 0.7):
    """DeepEval Faithfulness metric (Groundedness check)."""
    from deepeval.metrics import FaithfulnessMetric
    return FaithfulnessMetric(threshold=threshold, model=get_judge())

def get_answer_relevancy_metric(threshold: float = 0.7):
    """DeepEval Answer Relevancy metric (Utility check)."""
    from deepeval.metrics import AnswerRelevancyMetric
    return AnswerRelevancyMetric(threshold=threshold, model=get_judge())

def get_contextual_relevance_metric(threshold: float = 0.7):
    """DeepEval Contextual Relevance metric."""
    from deepeval.metrics import ContextualRelevancyMetric
    return ContextualRelevancyMetric(threshold=threshold, model=get_judge())

def get_contextual_precision_metric(threshold: float = 0.7):
    """DeepEval Contextual Precision metric."""
    from deepeval.metrics import ContextualPrecisionMetric
    return ContextualPrecisionMetric(threshold=threshold, model=get_judge())

def get_contextual_recall_metric(threshold: float = 0.7):
    """DeepEval Contextual Recall metric."""
    from deepeval.metrics import ContextualRecallMetric
    return ContextualRecallMetric(threshold=threshold, model=get_judge())

def get_code_correctness_geval(threshold: float = 0.7):
    """Custom DeepEval GEval metric for Code Correctness and Citation Accuracy."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams
    return GEval(
        name="Code Correctness & Citation",
        criteria="Evaluate if the response accurately explains the code, uses correct function/class names, and cites source files accurately.",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVED_CONTEXT],
        model=get_judge(),
        threshold=threshold
    )


