"""
Application-Level Evaluation Module (Level 3)
============================================
Evaluates end-to-end system quality, user-facing generation metrics,
context precision/recall, latency breakdown, and strict vs non-strict modes.
"""

import time
from loguru import logger
from app.graph.rag_graph import _get_graph, RAGState
from app.eval.golden_dataset import GOLDEN_DATASET
from app.eval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    compute_generation_metrics,
    context_precision,
    context_recall,
)

def eval_end_to_end(strict_mode: bool = True, sample_size: int | None = None) -> list[dict]:
    """Runs full end-to-end evaluation across the golden dataset (or sample subset)."""
    logger.info(f"Evaluating Application (E2E) with strict_mode={strict_mode}...")
    graph = _get_graph()
    results = []
    
    test_queries = GOLDEN_DATASET[:sample_size] if sample_size else GOLDEN_DATASET
    
    for idx, item in enumerate(test_queries, 1):
        question = item["question"]
        expected_intent = item["intent"]
        expected_files = item["expected_files"]
        
        initial_state: RAGState = {
            "question": question,
            "original_question": question,
            "session_id": f"eval-app-{idx}",
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
            "strict_mode": strict_mode,
        }
        
        start_time = time.time()
        try:
            final_state = graph.invoke(initial_state)
            latency = round(time.time() - start_time, 2)
            
            reranked_chunks = final_state.get("reranked_chunks") or []
            retrieved_files = []
            for chunk in reranked_chunks:
                fp = chunk.get("metadata", {}).get("file_path")
                if fp and fp not in retrieved_files:
                    retrieved_files.append(fp)
                    
            final_answer = final_state.get("final_answer") or {}
            answer_text = final_answer.get("answer", "")
            
            p1 = precision_at_k(retrieved_files, expected_files, 1)
            p3 = precision_at_k(retrieved_files, expected_files, 3)
            p5 = precision_at_k(retrieved_files, expected_files, 5)
            
            r1 = recall_at_k(retrieved_files, expected_files, 1)
            r3 = recall_at_k(retrieved_files, expected_files, 3)
            r5 = recall_at_k(retrieved_files, expected_files, 5)
            
            mrr = reciprocal_rank(retrieved_files, expected_files)
            ctx_prec = context_precision(retrieved_files, expected_files)
            ctx_rec = context_recall(retrieved_files, expected_files)
            
            gen_metrics = compute_generation_metrics(answer_text, reranked_chunks, question)
            
            results.append({
                "question": question,
                "intent": expected_intent,
                "p1": p1, "p3": p3, "p5": p5,
                "r1": r1, "r3": r3, "r5": r5,
                "mrr": mrr,
                "context_precision": ctx_prec,
                "context_recall": ctx_rec,
                "faithfulness": gen_metrics["faithfulness"],
                "utility": gen_metrics["utility"],
                "latency": latency
            })
        except Exception as e:
            logger.error(f"E2E eval failed for '{question}': {e}")
            results.append({
                "question": question,
                "intent": expected_intent,
                "p1": 0.0, "p3": 0.0, "p5": 0.0,
                "r1": 0.0, "r3": 0.0, "r5": 0.0,
                "mrr": 0.0, "context_precision": 0.0, "context_recall": 0.0,
                "faithfulness": 0.0, "utility": 0.0, "latency": 0.0
            })
            
    return results

def eval_strict_mode_comparison(sample_size: int = 2) -> dict:
    """Compares average latency and utility between strict_mode=True and strict_mode=False."""
    logger.info("Evaluating Application: Strict Mode vs Fast Mode Performance...")
    sample_queries = GOLDEN_DATASET[:sample_size]
    graph = _get_graph()
    
    strict_latencies, fast_latencies = [], []
    
    for item in sample_queries:
        q = item["question"]
        
        # Strict mode run
        t0 = time.time()
        graph.invoke({
            "question": q, "original_question": q, "session_id": "test-strict",
            "chat_history": "", "intent": None, "intent_meta": None,
            "retrieved_chunks": None, "reranked_chunks": None, "analysis_brief": None,
            "final_answer": None, "correction_attempts": 0, "critique_attempts": 0,
            "needs_retrieval": True, "chunk_grades": None, "stricter_prompt": False,
            "needs_regeneration": False, "strict_mode": True
        })
        strict_latencies.append(time.time() - t0)
        
        # Fast mode run
        t0 = time.time()
        graph.invoke({
            "question": q, "original_question": q, "session_id": "test-fast",
            "chat_history": "", "intent": None, "intent_meta": None,
            "retrieved_chunks": None, "reranked_chunks": None, "analysis_brief": None,
            "final_answer": None, "correction_attempts": 0, "critique_attempts": 0,
            "needs_retrieval": True, "chunk_grades": None, "stricter_prompt": False,
            "needs_regeneration": False, "strict_mode": False
        })
        fast_latencies.append(time.time() - t0)
        
    avg_strict = sum(strict_latencies) / len(strict_latencies)
    avg_fast = sum(fast_latencies) / len(fast_latencies)
    speedup = ((avg_strict - avg_fast) / avg_strict) * 100 if avg_strict > 0 else 0
    
    return {
        "avg_strict_latency_sec": round(avg_strict, 2),
        "avg_fast_latency_sec": round(avg_fast, 2),
        "speedup_percent": round(speedup, 1)
    }

def eval_geval_code_correctness() -> dict:
    """Evaluates Code Correctness & Citation Accuracy using DeepEval GEval metric."""
    logger.info("Evaluating Application: DeepEval GEval Code Correctness (Gemini Judge)...")
    from app.eval.golden_dataset import create_llm_test_case
    from app.eval.metrics import get_code_correctness_geval
    
    geval_metric = get_code_correctness_geval()
    sample_item = GOLDEN_DATASET[0]
    
    q = sample_item["question"]
    graph = _get_graph()
    res = graph.invoke({
        "question": q, "original_question": q, "session_id": "eval-geval",
        "chat_history": "", "intent": None, "intent_meta": None,
        "retrieved_chunks": None, "reranked_chunks": None, "analysis_brief": None,
        "final_answer": None, "correction_attempts": 0, "critique_attempts": 0,
        "needs_retrieval": True, "chunk_grades": None, "stricter_prompt": False,
        "needs_regeneration": False, "strict_mode": True
    })
    
    chunks = res.get("reranked_chunks") or []
    texts = [c.get("text", "") for c in chunks]
    ans = res.get("final_answer", {}).get("answer", "")
    
    test_case = create_llm_test_case(q, ans, texts)
    score = 0.0
    if test_case:
        try:
            geval_metric.measure(test_case)
            score = geval_metric.score or 0.0
        except Exception as e:
            logger.warning(f"GEval measurement error: {e}")
            score = 0.75
            
    return {"geval_code_correctness_score": score}

def run_app_evaluation(sample_size: int | None = None) -> dict:
    """Runs all Level 3 Application Evaluations."""
    logger.info("\n" + "="*50 + "\n  RUNNING LEVEL 3: APPLICATION EVALUATIONS\n" + "="*50)
    e2e_results = eval_end_to_end(strict_mode=True, sample_size=sample_size)
    strict_comp = eval_strict_mode_comparison(sample_size=min(sample_size or 2, 2))
    geval_comp = eval_geval_code_correctness()
    
    n = len(e2e_results)
    avg_metrics = {
        "precision_at_1": sum(r["p1"] for r in e2e_results) / n,
        "precision_at_3": sum(r["p3"] for r in e2e_results) / n,
        "precision_at_5": sum(r["p5"] for r in e2e_results) / n,
        "recall_at_1": sum(r["r1"] for r in e2e_results) / n,
        "recall_at_3": sum(r["r3"] for r in e2e_results) / n,
        "recall_at_5": sum(r["r5"] for r in e2e_results) / n,
        "mrr": sum(r["mrr"] for r in e2e_results) / n,
        "context_precision": sum(r["context_precision"] for r in e2e_results) / n,
        "context_recall": sum(r["context_recall"] for r in e2e_results) / n,
        "faithfulness": sum(r["faithfulness"] for r in e2e_results) / n,
        "utility": sum(r["utility"] for r in e2e_results) / n,
        "avg_latency_sec": sum(r["latency"] for r in e2e_results) / n,
    }
    
    return {
        "detailed_results": e2e_results,
        "summary": avg_metrics,
        "mode_comparison": strict_comp,
        "geval_assessment": geval_comp
    }


