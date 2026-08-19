"""
Golden Evaluation Dataset
=========================
Contains 20-30 curated queries about this codebase, including expected intents,
expected source files, and expected answers/descriptions.
"""

GOLDEN_DATASET = [
    # ── 1. Code Search ────────────────────────────────────────────────────────
    {
        "question": "Where is the StateGraph compiled?",
        "intent": "code_search",
        "expected_files": ["app/graph/rag_graph.py"]
    },
    {
        "question": "Find where index_chunks is defined.",
        "intent": "code_search",
        "expected_files": ["app/retrieval/vector_store.py"]
    },
    {
        "question": "Where do we initialize the database and run migrations?",
        "intent": "code_search",
        "expected_files": ["app/memory/db.py"]
    },
    {
        "question": "Which file defines the intent agent?",
        "intent": "code_search",
        "expected_files": ["app/agents/intent_agent.py"]
    },
    {
        "question": "Where is the retrieve function implemented?",
        "intent": "code_search",
        "expected_files": ["app/agents/retrieval_agent.py"]
    },
    {
        "question": "Where are the FastAPI endpoints like /chat and /query?",
        "intent": "code_search",
        "expected_files": ["app/api/routes.py"]
    },
    # ── 2. Explain ────────────────────────────────────────────────────────────
    {
        "question": "How does the intent agent classify user questions?",
        "intent": "explain",
        "expected_files": ["app/agents/intent_agent.py"]
    },
    {
        "question": "Explain how chunks are stored and embedded in ChromaDB.",
        "intent": "explain",
        "expected_files": ["app/retrieval/vector_store.py"]
    },
    {
        "question": "How is the conversation history loaded and formatted?",
        "intent": "explain",
        "expected_files": ["app/memory/session.py"]
    },
    {
        "question": "Explain the hybrid search strategy combining BM25 and Vector search.",
        "intent": "explain",
        "expected_files": ["app/retrieval/hybrid.py"]
    },
    {
        "question": "How does the analysis agent rank relevant files?",
        "intent": "explain",
        "expected_files": ["app/agents/analysis_agent.py"]
    },
    # ── 3. Trace Flow ─────────────────────────────────────────────────────────
    {
        "question": "Trace the path of a query from routes.py to rag_graph.py.",
        "intent": "trace_flow",
        "expected_files": ["app/api/routes.py", "app/graph/rag_graph.py"]
    },
    {
        "question": "What happens step-by-step during repository ingestion?",
        "intent": "trace_flow",
        "expected_files": ["app/graph/rag_graph.py", "app/ingestion/clone_repo.py"]
    },
    {
        "question": "Trace how the rerank node is executed in the graph.",
        "intent": "trace_flow",
        "expected_files": ["app/graph/rag_graph.py", "app/reranker/reranker.py"]
    },
    {
        "question": "Walk me through the flow of the retrieve node.",
        "intent": "trace_flow",
        "expected_files": ["app/graph/rag_graph.py", "app/agents/retrieval_agent.py"]
    },
    # ── 4. Architecture ───────────────────────────────────────────────────────
    {
        "question": "Describe the overall package structure and design of this project.",
        "intent": "architecture",
        "expected_files": ["app/graph/rag_graph.py"]
    },
    {
        "question": "What are the main components of the app/ folder and their roles?",
        "intent": "architecture",
        "expected_files": ["README.md"]
    },
    {
        "question": "How is database schema and models set up?",
        "intent": "architecture",
        "expected_files": ["app/memory/models.py"]
    },
    {
        "question": "How does the streamlit frontend talk to the backend API?",
        "intent": "architecture",
        "expected_files": ["frontend/streamlit_app.py", "app/api/routes.py"]
    },
    # ── 5. Debug ──────────────────────────────────────────────────────────────
    {
        "question": "Why would BM25 index fail to load and throw FileNotFoundError?",
        "intent": "debug",
        "expected_files": ["app/graph/rag_graph.py", "app/retrieval/bm25.py"]
    },
    {
        "question": "What happens if a scanned file fails to chunk?",
        "intent": "debug",
        "expected_files": ["app/ingestion/chunker.py"]
    },
    {
        "question": "Why would routes.py return a 400 Bad Request if the repo is not ingested?",
        "intent": "debug",
        "expected_files": ["app/api/routes.py"]
    },
    {
        "question": "How does the system handle database connection failures?",
        "intent": "debug",
        "expected_files": ["app/memory/db.py"]
    },
    {
        "question": "What happens if the LLM call fails during answer generation?",
        "intent": "debug",
        "expected_files": ["app/agents/answer_agent.py"]
    }
]

# ── 6. Component Test Set: Document Grader ─────────────────────────────────
GRADER_TEST_SET = [
    {
        "question": "Where is the StateGraph compiled?",
        "chunk": "def _get_graph():\n    global _compiled_graph\n    if _compiled_graph is None:\n        _compiled_graph = build_graph()\n    return _compiled_graph",
        "expected_grade": "relevant"
    },
    {
        "question": "Where is the StateGraph compiled?",
        "chunk": "class ChatMessage(Base):\n    __tablename__ = 'chat_messages'\n    id = Column(String, primary_key=True)",
        "expected_grade": "irrelevant"
    },
    {
        "question": "How is database schema and models set up?",
        "chunk": "class ChatSession(Base):\n    __tablename__ = 'chat_sessions'\n    session_id = Column(String, primary_key=True)",
        "expected_grade": "relevant"
    },
    {
        "question": "How is database schema and models set up?",
        "chunk": "def clone_repo(repo_url: str) -> str:\n    logger.info(f'Cloning {repo_url}')",
        "expected_grade": "irrelevant"
    }
]

# ── 7. Component Test Set: Retrieve Gate ─────────────────────────────────────
RETRIEVE_GATE_TEST_SET = [
    {
        "question": "Hello, how are you today?",
        "expected_needs_retrieval": False
    },
    {
        "question": "Write a python function to compute fibonacci numbers.",
        "expected_needs_retrieval": False
    },
    {
        "question": "Where is the StateGraph compiled in this project?",
        "expected_needs_retrieval": True
    },
    {
        "question": "How does the intent agent classify user questions in app/agents/intent_agent.py?",
        "expected_needs_retrieval": True
    }
]

def create_llm_test_case(
    question: str,
    actual_output: str,
    retrieved_context: list[str],
    expected_output: str | None = None
):
    """Constructs a DeepEval LLMTestCase for evaluation metrics."""
    try:
        from deepeval.test_case import LLMTestCase
        return LLMTestCase(
            input=question,
            actual_output=actual_output,
            retrieved_context=retrieved_context if retrieved_context else ["No context retrieved."],
            expected_output=expected_output or "Target answer based on repository source files."
        )
    except Exception as e:
        return None

