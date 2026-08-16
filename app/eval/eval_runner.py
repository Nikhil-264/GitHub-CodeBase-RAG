"""
Evaluation Runner
=================
Runs the golden dataset questions through the LangGraph RAG pipeline,
computes metrics, logs results, and outputs a summary report.

Usage:
    python -m app.eval.eval_runner
"""

import asyncio
import csv
import os
import time
from loguru import logger

from app.graph.rag_graph import _get_graph, RAGState
from app.eval.golden_dataset import GOLDEN_DATASET
from app.eval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    compute_generation_metrics
)

# Optional dependency — falls back to plain print() formatting if not installed
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

RESULTS_CSV_PATH = "repos/eval_results.csv"

async def run_evaluation():
    logger.info("Starting codebase RAG evaluation runner...")
    
    # Check that BM25 index and vector DB are available
    from app.graph.rag_graph import _get_bm25_retriever
    try:
        _get_bm25_retriever()
    except Exception as e:
        logger.error(f"Cannot run evaluation: {e}. Please ingest a repository first (e.g. run test_pipeline.py).")
        return

    graph = _get_graph()
    results = []
    
    # Ensure repos/ directory exists for CSV logging
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    
    headers = [
        "Question", "Intent", "P@1", "P@3", "P@5", 
        "R@1", "R@3", "R@5", "MRR", "Faithfulness", "Utility", "Latency (s)"
    ]

    for idx, item in enumerate(GOLDEN_DATASET, 1):
        question = item["question"]
        expected_intent = item["intent"]
        expected_files = item["expected_files"]
        
        logger.info(f"[{idx}/{len(GOLDEN_DATASET)}] Running: '{question}' (Expected Intent: {expected_intent})")
        
        # Build initial state
        initial_state: RAGState = {
            "question": question,
            "original_question": question,
            "session_id": f"eval-session-{idx}",
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
        }
        
        start_time = time.time()
        try:
            # Invoke graph synchronously/async-safely
            final_state = graph.invoke(initial_state)
            latency = round(time.time() - start_time, 2)
            
            # Extract retrieved file paths from reranked chunks
            reranked_chunks = final_state.get("reranked_chunks") or []
            retrieved_files = []
            for chunk in reranked_chunks:
                fp = chunk.get("metadata", {}).get("file_path")
                if fp and fp not in retrieved_files:
                    retrieved_files.append(fp)
            
            final_answer = final_state.get("final_answer") or {}
            answer_text = final_answer.get("answer", "")
            
            # Calculate retrieval metrics
            p1 = precision_at_k(retrieved_files, expected_files, k=1)
            p3 = precision_at_k(retrieved_files, expected_files, k=3)
            p5 = precision_at_k(retrieved_files, expected_files, k=5)
            
            r1 = recall_at_k(retrieved_files, expected_files, k=1)
            r3 = recall_at_k(retrieved_files, expected_files, k=3)
            r5 = recall_at_k(retrieved_files, expected_files, k=5)
            
            mrr = reciprocal_rank(retrieved_files, expected_files)
            
            # Calculate generation metrics
            gen_metrics = compute_generation_metrics(answer_text, reranked_chunks, question)
            faithfulness = gen_metrics["faithfulness"]
            utility = gen_metrics["utility"]
            
            result_row = {
                "question": question,
                "intent": expected_intent,
                "p1": p1,
                "p3": p3,
                "p5": p5,
                "r1": r1,
                "r3": r3,
                "r5": r5,
                "mrr": mrr,
                "faithfulness": faithfulness,
                "utility": utility,
                "latency": latency
            }
            results.append(result_row)
            logger.success(f"Finished. MRR={mrr}, Faithfulness={faithfulness}, Utility={utility}, Latency={latency}s")
            
        except Exception as e:
            logger.error(f"Failed to evaluate question: {question}. Error: {e}")
            results.append({
                "question": question,
                "intent": expected_intent,
                "p1": 0.0, "p3": 0.0, "p5": 0.0,
                "r1": 0.0, "r3": 0.0, "r5": 0.0,
                "mrr": 0.0, "faithfulness": 0.0, "utility": 0.0,
                "latency": 0.0
            })

    # Save detailed results to CSV
    with open(RESULTS_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Compute Aggregate Metrics
    n = len(results)
    avg_p1 = sum(r["p1"] for r in results) / n
    avg_p3 = sum(r["p3"] for r in results) / n
    avg_p5 = sum(r["p5"] for r in results) / n
    avg_r1 = sum(r["r1"] for r in results) / n
    avg_r3 = sum(r["r3"] for r in results) / n
    avg_r5 = sum(r["r5"] for r in results) / n
    avg_mrr = sum(r["mrr"] for r in results) / n
    avg_faithfulness = sum(r["faithfulness"] for r in results) / n
    avg_utility = sum(r["utility"] for r in results) / n
    avg_latency = sum(r["latency"] for r in results) / n

    summary_data = [
        ["Precision@1", f"{avg_p1:.2%}"],
        ["Precision@3", f"{avg_p3:.2%}"],
        ["Precision@5", f"{avg_p5:.2%}"],
        ["Recall@1", f"{avg_r1:.2%}"],
        ["Recall@3", f"{avg_r3:.2%}"],
        ["Recall@5", f"{avg_r5:.2%}"],
        ["MRR", f"{avg_mrr:.4f}"],
        ["Faithfulness (Groundedness)", f"{avg_faithfulness:.2%}"],
        ["Utility (Relevance)", f"{avg_utility:.2%}"],
        ["Average Latency", f"{avg_latency:.2f}s"]
    ]

    print("\n" + "="*80)
    print("                      AGGREGATE EVALUATION REPORT")
    print("="*80)
    if tabulate:
        print(tabulate(summary_data, headers=["Metric", "Average Score"], tablefmt="grid"))
    else:
        for row in summary_data:
            print(f"{row[0]:<30}: {row[1]}")
    print("="*80)
    print(f"Detailed logs saved to {RESULTS_CSV_PATH}\n")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
