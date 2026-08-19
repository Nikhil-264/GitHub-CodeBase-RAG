"""
Pytest DeepEval Continuous Regression Test Suite (Stage 4)
================================--------------------------
Runs automated regression tests using DeepEval and Gemini LLM-as-a-Judge.
Can be executed directly via:
    pytest app/eval/test_rag_eval.py
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from app.eval.golden_dataset import GOLDEN_DATASET
from app.eval.metrics import (
    get_faithfulness_metric,
    get_answer_relevancy_metric,
    get_contextual_relevance_metric,
)
from app.graph.rag_graph import _get_graph, RAGState

# Run regression tests on sample golden questions
SAMPLE_TEST_ITEMS = GOLDEN_DATASET[:3]

@pytest.mark.parametrize("item", SAMPLE_TEST_ITEMS)
def test_rag_triad_pipeline(item):
    """Evaluates RAG Triad (Faithfulness, Answer Relevancy, Context Relevance) via DeepEval."""
    question = item["question"]
    graph = _get_graph()
    
    initial_state: RAGState = {
        "question": question,
        "original_question": question,
        "session_id": f"pytest-eval",
        "chat_history": "",
        "intent": None,
        "intent_meta": None,
        "retrieved_chunks": None,
        "reranked_chunks": None,
        "analysis_brief": None,
        "final_answer": None,
        "correction_attempts": 0,
        "critique_attempts": 0,
        "needs_retrieval": True,
        "chunk_grades": None,
        "stricter_prompt": False,
        "needs_regeneration": False,
        "strict_mode": True,
    }
    
    final_state = graph.invoke(initial_state)
    reranked_chunks = final_state.get("reranked_chunks") or []
    retrieved_texts = [c.get("text", "") for c in reranked_chunks]
    
    final_answer = final_state.get("final_answer") or {}
    answer_text = final_answer.get("answer", "")
    
    test_case = LLMTestCase(
        input=question,
        actual_output=answer_text,
        retrieved_context=retrieved_texts if retrieved_texts else ["No context retrieved."],
        expected_output="Detailed technical explanation based on codebase."
    )
    
    faithfulness_metric = get_faithfulness_metric(threshold=0.6)
    answer_relevancy_metric = get_answer_relevancy_metric(threshold=0.6)
    context_relevance_metric = get_contextual_relevance_metric(threshold=0.6)
    
    assert_test(
        test_case,
        [faithfulness_metric, answer_relevancy_metric, context_relevance_metric]
    )
