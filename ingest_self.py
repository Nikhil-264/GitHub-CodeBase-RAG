"""
Script to ingest this repository into the database for evaluation.
Wipes the database collection first for a clean run.
"""
from app.graph.rag_graph import ingest_repo
from app.retrieval.vector_store import delete_collection
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    print("Wiping existing ChromaDB collection for a clean run...")
    try:
        delete_collection()
    except Exception as e:
        print(f"Could not delete collection (might not exist yet): {e}")

    print("Indexing GitHub-CodeBase-RAG repository...")
    try:
        result = ingest_repo("https://github.com/Nikhil-264/GitHub-CodeBase-RAG.git")
        print("Ingestion complete!")
        print(result)
    except Exception as e:
        print(f"Failed to ingest: {e}")
