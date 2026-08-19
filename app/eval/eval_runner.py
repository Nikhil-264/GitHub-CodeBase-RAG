"""
3-Step Unified Evaluation Suite Runner
======================================
Runs component-level, pipeline-level, and application-level evaluations.

Usage:
    python -m app.eval.eval_runner --level all
    python -m app.eval.eval_runner --level component
    python -m app.eval.eval_runner --level pipeline
    python -m app.eval.eval_runner --level app
"""

import argparse
import asyncio
import csv
import os
from loguru import logger

from app.eval.component_eval import run_component_evaluation
from app.eval.pipeline_eval import run_pipeline_evaluation
from app.eval.app_eval import run_app_evaluation
from app.graph.rag_graph import _get_bm25_retriever

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

RESULTS_CSV_PATH = "repos/eval_results.csv"

def print_table(title: str, data: list[list[str]], headers: list[str]):
    print("\n" + "=" * 80)
    print(f" {title.center(78)} ")
    print("=" * 80)
    if tabulate:
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        for row in data:
            print(f"{row[0]:<35}: {row[1]}")
    print("=" * 80)

async def run_evaluation(level: str = "all", sample: int | None = None):
    logger.info(f"Starting Codebase RAG Evaluation Suite (Level: {level.upper()}, Sample Limit: {sample or 'ALL'})...")
    
    # Check that BM25 index and vector store exist
    try:
        bm25_retriever = _get_bm25_retriever()
    except Exception as e:
        logger.error(f"Cannot run evaluation: {e}. Please ingest a repository first (e.g. run test_pipeline.py).")
        return

    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # LEVEL 1: Component Evaluation
    # ─────────────────────────────────────────────────────────
    if level in ("all", "component"):
        comp_res = run_component_evaluation(bm25_retriever, sample_size=sample)
        
        comp_table = [
            ["Intent Classifier (Rules Accuracy)", f"{comp_res['intent_classifier']['rule_accuracy']:.2%}"],
            ["Intent Classifier (LLM Accuracy)", f"{comp_res['intent_classifier']['llm_accuracy']:.2%}"],
            ["Vector Search Alone (MRR)", f"{comp_res['isolated_retrieval']['vector']['mrr']:.4f}"],
            ["BM25 Search Alone (MRR)", f"{comp_res['isolated_retrieval']['bm25']['mrr']:.4f}"],
            ["Reranker Impact (P@1 Before)", f"{comp_res['reranker_impact']['before_rerank']['p1']:.2%}"],
            ["Reranker Impact (P@1 After)", f"{comp_res['reranker_impact']['after_rerank']['p1']:.2%}"],
            ["Reranker P@1 Gain Delta", f"{comp_res['reranker_impact']['p1_gain']:+.2%}"],
            ["Document Grader Accuracy (CRAG)", f"{comp_res['document_grader']['grader_accuracy']:.2%}"],
        ]
        print_table("LEVEL 1: COMPONENT-LEVEL EVALUATION REPORT", comp_table, ["Metric / Component", "Score"])

    # ─────────────────────────────────────────────────────────
    # LEVEL 2: Pipeline Evaluation
    # ─────────────────────────────────────────────────────────
    if level in ("all", "pipeline"):
        pipe_res = run_pipeline_evaluation(bm25_retriever, sample_size=sample)
        triad = pipe_res.get("rag_triad_deepeval", {})
        
        pipe_table = [
            ["Hybrid Search + Reranker Precision@1", f"{pipe_res['hybrid_retrieval_pipeline']['p1']:.2%}"],
            ["Hybrid Search + Reranker Recall@5", f"{pipe_res['hybrid_retrieval_pipeline']['r5']:.2%}"],
            ["Hybrid Search + Reranker MRR", f"{pipe_res['hybrid_retrieval_pipeline']['mrr']:.4f}"],
            ["Retrieve Gate Accuracy (Self-RAG)", f"{pipe_res['retrieve_gate']['retrieve_gate_accuracy']:.2%}"],
            ["Self-RAG Critique Check Accuracy", f"{pipe_res['self_rag_critique']['grounding_check_accuracy']:.2%}"],
            ["DeepEval Faithfulness Metric (Gemini Judge)", f"{triad.get('deepeval_faithfulness', 0.0):.2%}"],
            ["DeepEval Answer Relevancy Metric (Gemini Judge)", f"{triad.get('deepeval_answer_relevancy', 0.0):.2%}"],
            ["DeepEval Context Relevance Metric (Gemini Judge)", f"{triad.get('deepeval_context_relevance', 0.0):.2%}"],
        ]
        print_table("LEVEL 2: PIPELINE-LEVEL (RAG TRIAD) REPORT", pipe_table, ["Pipeline Component / Metric", "Score"])

    # ─────────────────────────────────────────────────────────
    # LEVEL 3: Application Evaluation (E2E)
    # ─────────────────────────────────────────────────────────
    if level in ("all", "app"):
        app_res = run_app_evaluation(sample_size=sample)
        summary = app_res["summary"]
        mode_comp = app_res["mode_comparison"]
        geval = app_res.get("geval_assessment", {})
        
        app_table = [
            ["Precision@1", f"{summary['precision_at_1']:.2%}"],
            ["Precision@3", f"{summary['precision_at_3']:.2%}"],
            ["Precision@5", f"{summary['precision_at_5']:.2%}"],
            ["Recall@1", f"{summary['recall_at_1']:.2%}"],
            ["Recall@3", f"{summary['recall_at_3']:.2%}"],
            ["Recall@5", f"{summary['recall_at_5']:.2%}"],
            ["MRR", f"{summary['mrr']:.4f}"],
            ["Context Precision (RAGAS)", f"{summary['context_precision']:.2%}"],
            ["Context Recall (RAGAS)", f"{summary['context_recall']:.2%}"],
            ["Faithfulness (Groundedness)", f"{summary['faithfulness']:.2%}"],
            ["Utility (Relevance)", f"{summary['utility']:.2%}"],
            ["DeepEval GEval Code Correctness Score", f"{geval.get('geval_code_correctness_score', 0.0):.2%}"],
            ["Average E2E Latency", f"{summary['avg_latency_sec']:.2f}s"],
            ["Fast Mode Avg Latency (strict_mode=False)", f"{mode_comp['avg_fast_latency_sec']}s"],
            ["Strict Mode Avg Latency (strict_mode=True)", f"{mode_comp['avg_strict_latency_sec']}s"],
            ["Fast Mode Speedup Gain", f"{mode_comp['speedup_percent']:.1f}%"],
        ]
        print_table("LEVEL 3: APPLICATION-LEVEL (END-TO-END) REPORT", app_table, ["Application Metric", "Score"])

        # Save detailed per-query results to CSV
        with open(RESULTS_CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=app_res["detailed_results"][0].keys())
            writer.writeheader()
            writer.writerows(app_res["detailed_results"])
        logger.info(f"Detailed E2E results saved to {RESULTS_CSV_PATH}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GitHub Codebase RAG 3-Step Evaluation Suite")
    parser.add_argument(
        "--level",
        type=str,
        choices=["all", "component", "pipeline", "app"],
        default="all",
        help="Evaluation level to run (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit number of queries evaluated per stage to respect API rate limits (e.g. --sample 2)"
    )
    args = parser.parse_args()
    asyncio.run(run_evaluation(level=args.level, sample=args.sample))

