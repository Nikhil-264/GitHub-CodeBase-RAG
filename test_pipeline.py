from pathlib import Path
from loguru import logger
import os

from app.ingestion.clone_repo import clone_repo
from app.ingestion.scan_repo import scan_repo
from app.ingestion.chunker import chunk_file
from app.retrieval.vector_store import index_chunks, collection_stats, delete_collection
from app.retrieval.bm25 import BM25Retriever
from app.agents.intent_agent import classify_intent
from app.agents.retrieval_agent import retrieve
from app.reranker.reranker import rerank
from app.agents.analysis_agent import analyse
from app.agents.answer_agent import answer
from langsmith import traceable

@traceable(run_type="chain")
def run_test_pipeline(repo_url: str, questions: list[str] | str):
    if isinstance(questions, str):
        questions = [questions]

    logger.info("=== STEP 1: Ingestion & Indexing ===")
    
    # 1. Clone repo
    repo_path = clone_repo(repo_url)
    logger.info(f"Repo path: {repo_path}")
    
    # 2. Scan repo
    scanned_files = scan_repo(repo_path)
    logger.info(f"Scanned {len(scanned_files)} files")
    
    # 3. Chunk files
    all_chunks = []
    for f in scanned_files:
        try:
            chunks = chunk_file(f)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk {f['relative_path']}: {e}")
            
    logger.info(f"Created a total of {len(all_chunks)} chunks from code files.")
    
    if not all_chunks:
        logger.error("No chunks created. Aborting test.")
        return
        
    # 4. Index in ChromaDB
    # Wipe database first for clean test
    logger.info("Wiping existing collection for clean run...")
    try:
        delete_collection()
    except Exception as e:
        logger.warning(f"Could not delete collection (might not exist yet): {e}")
        
    indexed_count = index_chunks(all_chunks)
    logger.info(f"Successfully indexed {indexed_count} chunks in ChromaDB.")
    logger.info(f"Chroma Collection Stats: {collection_stats()}")
    
    # 5. Build and Save BM25
    bm25_path = "repos/bm25_index.pkl"
    logger.info("Building BM25 index...")
    bm25_retriever = BM25Retriever(all_chunks)
    bm25_retriever.save(bm25_path)
    logger.info(f"BM25 index saved to {bm25_path}")
    
    logger.info("Loading BM25 index from disk...")
    loaded_bm25 = BM25Retriever.load(bm25_path)
    
    logger.info(f"=== STEP 2 & 3: Running Pipeline for {len(questions)} Questions ===")
    
    for idx, question in enumerate(questions, 1):
        logger.info(f"\n--- Running Pipeline for Question {idx}/{len(questions)}: '{question}' ---")
        
        # 6. Intent classification
        intent_result = classify_intent(question)
        intent = intent_result["intent"]
        
        # 7. Retrieval Agent
        retrieved_chunks = retrieve(question, intent, loaded_bm25)
        logger.info(f"Retrieval Agent fetched {len(retrieved_chunks)} candidate chunks.")
        
        # 8. Rerank
        reranked_chunks = rerank(question, retrieved_chunks, top_k=10)
        logger.info(f"Reranker returned top {len(reranked_chunks)} chunks.")
        
        # 9. Analysis Agent
        brief = analyse(question, reranked_chunks)
        logger.info(f"Analysis Agent compiled context brief. Rank files: {brief['primary_files']}")
        
        # 10. Answer Agent
        logger.info("Generating final answer using Ollama LLM...")
        response_result = answer(brief, intent)
        
        print("\n" + "=" * 60)
        print(f"QUESTION {idx}: {question}")
        print(f"CLASSIFIED INTENT: {intent.upper()} ({intent_result['description']})")
        print(f"SOURCES CITED: {', '.join(response_result['sources'])}")
        print("=" * 60)
        print(response_result["answer"])
        print("=" * 60)

if __name__ == "__main__":
    # Ensure environment variables are loaded
    from dotenv import load_dotenv
    load_dotenv()

    test_repo = "https://github.com/Nikhil-264/flashcard-study-agent.git"
    
    # Predefined batch of test queries covering different parts of the flashcard-study-agent repo
    test_queries = [
        "What model is used for flashcard generation and what is the expected JSON response schema?"
    ]
    
    run_test_pipeline(test_repo, test_queries)

