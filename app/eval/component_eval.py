"""
Component-Level Evaluation Module (Level 1)
===========================================
Evaluates individual RAG components in isolation:
- Intent Classifier (Rules vs LLM)
- Isolated Dense (ChromaDB) vs Sparse (BM25) Retrieval
- Reranker Precision Gain
- Document Grader Accuracy
"""

import time
from loguru import logger
from app.eval.golden_dataset import GOLDEN_DATASET, GRADER_TEST_SET
from app.eval.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    accuracy_score,
)
from app.agents.intent_agent import _classify_rules, _classify_llm
from app.agents.grader_agent import grade_chunk_relevance
from app.retrieval.vector_store import get_collection
from app.retrieval.bm25 import BM25Retriever
from app.embeddings.embedder import embed_query
from app.reranker.reranker import rerank

def eval_intent_classifier(sample_size: int | None = None) -> dict:
    """Evaluates rule-based and LLM-based intent classifier accuracy."""
    logger.info("Evaluating Component: Intent Classifier...")
    test_queries = GOLDEN_DATASET[:sample_size] if sample_size else GOLDEN_DATASET
    true_intents = [item["intent"] for item in test_queries]
    
    rule_preds = [_classify_rules(item["question"]) for item in test_queries]
    llm_preds = [_classify_llm(item["question"]) for item in test_queries]
    
    rule_acc = accuracy_score(true_intents, rule_preds)
    llm_acc = accuracy_score(true_intents, llm_preds)
    
    logger.info(f"Intent Classifier Rule Accuracy: {rule_acc:.2%}, LLM Accuracy: {llm_acc:.2%}")
    return {
        "rule_accuracy": rule_acc,
        "llm_accuracy": llm_acc,
    }

def eval_isolated_retrieval(bm25_retriever: BM25Retriever, sample_size: int | None = None) -> dict:
    """Compares vector search (ChromaDB) vs BM25 sparse search in isolation."""
    logger.info("Evaluating Component: Vector Search vs BM25 Retrieval...")
    collection = get_collection()
    test_queries = GOLDEN_DATASET[:sample_size] if sample_size else GOLDEN_DATASET
    
    vec_p1, vec_p3, vec_p5 = [], [], []
    vec_r1, vec_r3, vec_r5 = [], [], []
    vec_mrr = []
    
    bm25_p1, bm25_p3, bm25_p5 = [], [], []
    bm25_r1, bm25_r3, bm25_r5 = [], [], []
    bm25_mrr = []
    
    for item in test_queries:
        question = item["question"]
        expected_files = item["expected_files"]
        
        # 1. Vector Search Alone
        q_emb = embed_query(question)
        v_results = collection.query(query_embeddings=[q_emb], n_results=5)
        v_files = []
        if v_results and v_results["metadatas"]:
            for meta in v_results["metadatas"][0]:
                fp = meta.get("file_path")
                if fp and fp not in v_files:
                    v_files.append(fp)
                    
        vec_p1.append(precision_at_k(v_files, expected_files, 1))
        vec_p3.append(precision_at_k(v_files, expected_files, 3))
        vec_p5.append(precision_at_k(v_files, expected_files, 5))
        vec_r1.append(recall_at_k(v_files, expected_files, 1))
        vec_r3.append(recall_at_k(v_files, expected_files, 3))
        vec_r5.append(recall_at_k(v_files, expected_files, 5))
        vec_mrr.append(reciprocal_rank(v_files, expected_files))
        
        # 2. BM25 Search Alone
        bm25_chunks = bm25_retriever.query(question, n_results=5)
        bm25_files = []
        for chunk in bm25_chunks:
            fp = chunk.get("metadata", {}).get("file_path")
            if fp and fp not in bm25_files:
                bm25_files.append(fp)
                
        bm25_p1.append(precision_at_k(bm25_files, expected_files, 1))
        bm25_p3.append(precision_at_k(bm25_files, expected_files, 3))
        bm25_p5.append(precision_at_k(bm25_files, expected_files, 5))
        bm25_r1.append(recall_at_k(bm25_files, expected_files, 1))
        bm25_r3.append(recall_at_k(bm25_files, expected_files, 3))
        bm25_r5.append(recall_at_k(bm25_files, expected_files, 5))
        bm25_mrr.append(reciprocal_rank(bm25_files, expected_files))
        
    n = len(test_queries)
    return {
        "vector": {
            "p1": sum(vec_p1)/n, "p3": sum(vec_p3)/n, "p5": sum(vec_p5)/n,
            "r1": sum(vec_r1)/n, "r3": sum(vec_r3)/n, "r5": sum(vec_r5)/n,
            "mrr": sum(vec_mrr)/n
        },
        "bm25": {
            "p1": sum(bm25_p1)/n, "p3": sum(bm25_p3)/n, "p5": sum(bm25_p5)/n,
            "r1": sum(bm25_r1)/n, "r3": sum(bm25_r3)/n, "r5": sum(bm25_r5)/n,
            "mrr": sum(bm25_mrr)/n
        }
    }

def eval_reranker_impact(bm25_retriever: BM25Retriever, sample_size: int | None = None) -> dict:
    """Evaluates retrieval precision gain before vs after Cross-Encoder reranking."""
    logger.info("Evaluating Component: Reranker Impact...")
    from app.agents.retrieval_agent import retrieve
    test_queries = GOLDEN_DATASET[:sample_size] if sample_size else GOLDEN_DATASET
    
    before_p1, after_p1 = [], []
    before_mrr, after_mrr = [], []
    
    for item in test_queries:
        question = item["question"]
        expected_files = item["expected_files"]
        intent = item["intent"]
        
        # Hybrid retrieval before rerank
        raw_chunks = retrieve(question, intent, bm25_retriever)
        raw_files = [c["metadata"]["file_path"] for c in raw_chunks if "metadata" in c and "file_path" in c["metadata"]]
        
        # Top chunks after rerank
        reranked_chunks = rerank(question, raw_chunks, top_k=5)
        reranked_files = [c["metadata"]["file_path"] for c in reranked_chunks if "metadata" in c and "file_path" in c["metadata"]]
        
        before_p1.append(precision_at_k(raw_files, expected_files, 1))
        after_p1.append(precision_at_k(reranked_files, expected_files, 1))
        
        before_mrr.append(reciprocal_rank(raw_files, expected_files))
        after_mrr.append(reciprocal_rank(reranked_files, expected_files))
        
    n = len(test_queries)
    return {
        "before_rerank": {"p1": sum(before_p1)/n, "mrr": sum(before_mrr)/n},
        "after_rerank": {"p1": sum(after_p1)/n, "mrr": sum(after_mrr)/n},
        "p1_gain": (sum(after_p1)/n) - (sum(before_p1)/n)
    }

def eval_chunk_grader() -> dict:
    """Evaluates Document Grader accuracy against labeled snippet test set."""
    logger.info("Evaluating Component: Document Grader (CRAG)...")
    y_true = [item["expected_grade"] for item in GRADER_TEST_SET]
    y_pred = []
    
    for item in GRADER_TEST_SET:
        grade = grade_chunk_relevance(item["question"], item["chunk"])
        y_pred.append(grade)
        
    acc = accuracy_score(y_true, y_pred)
    logger.info(f"Document Grader Accuracy: {acc:.2%}")
    return {"grader_accuracy": acc}

def run_component_evaluation(bm25_retriever: BM25Retriever, sample_size: int | None = None) -> dict:
    """Runs all Level 1 Component Evaluations."""
    logger.info("\n" + "="*50 + "\n  RUNNING LEVEL 1: COMPONENT EVALUATIONS\n" + "="*50)
    intent_res = eval_intent_classifier(sample_size=sample_size)
    isolated_res = eval_isolated_retrieval(bm25_retriever, sample_size=sample_size)
    rerank_res = eval_reranker_impact(bm25_retriever, sample_size=sample_size)
    grader_res = eval_chunk_grader()
    
    return {
        "intent_classifier": intent_res,
        "isolated_retrieval": isolated_res,
        "reranker_impact": rerank_res,
        "document_grader": grader_res
    }

