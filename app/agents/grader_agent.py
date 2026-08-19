"""
Grader Agent
============
Grades the relevance of retrieved code chunks and decides whether
a user question requires querying the codebase at all (retrieve gate).
"""

import os
import re
from loguru import logger
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

load_dotenv()

LLM_MODEL  = os.getenv("LLM_MODEL", "gemini-1.5-flash")

_llm: ChatGoogleGenerativeAI | None = None

def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        logger.info(f"Grader LLM loaded: {LLM_MODEL}")
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,  # deterministic grading
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    return _llm

# Keywords that explicitly indicate codebase-specific queries about this RAG system
_CODEBASE_KEYWORDS = [
    "stategraph", "state_graph", "ragstate", "rag_state",
    "index_chunks", "vector_store", "chromadb", "chroma",
    "migration", "alembic", "db.py", "session.py", "models.py",
    "intent_agent", "intent agent", "retrieval_agent", "retrieval agent",
    "routes.py", "fastapi", "lifespan", "/chat", "/query",
    "analysis_agent", "analysis agent", "answer_agent", "answer agent",
    "critique_agent", "critique agent", "reranker", "bge-reranker",
    "bm25", "bm25_index", "ingest", "ingestion", "clone_repo", "scan_repo",
    "chunk_file", "chunk_files", "chunker", "eval_runner"
]

_RETRIEVE_GATE_PROMPT = """You are a gatekeeper deciding if a user's question requires retrieving code from a codebase.
Questions that require retrieval include:
- Finding where functions/classes/variables are defined, used, or imported.
- Conceptual explanations of how features inside this specific codebase work (including BM25, ChromaDB, intent classifier, agents, and graphs).
- Tracing execution flow of features in the codebase.
- Debugging errors or understanding bugs in the codebase.
- Architectural design or file layout of the codebase.

Questions that do NOT require retrieval include:
- General programming questions (e.g., 'how do I write a binary search in Python?', 'explain the difference between REST and GraphQL').
- General greetings or conversational messages (e.g., 'hello', 'who are you?').
- General knowledge or off-topic questions.

Respond with exactly 'yes' or 'no'. No explanation or punctuation.

Question: {question}

Retrieve:"""

_CHUNK_GRADER_PROMPT = """You are a relevance classifier assessing whether a retrieved code chunk is relevant to a user's question.
If the chunk contains code, comments, or documentation relevant to answering the user's question, classify it as 'relevant'.
If the chunk is partially relevant, or it is unclear, classify it as 'ambiguous'.
If the chunk is completely unrelated to the question, classify it as 'irrelevant'.

Provide a single word response: 'relevant', 'ambiguous', or 'irrelevant'. No explanation or punctuation.

Question: {question}
Code Chunk:
{chunk}

Relevance (relevant/ambiguous/irrelevant):"""

@traceable(run_type="chain")
def check_need_retrieval(question: str) -> bool:
    """
    Decides whether a user question requires querying the codebase.
    Returns True if retrieval is needed, False otherwise.
    """
    q_lower = question.lower()
    # Rule-based keyword override
    for keyword in _CODEBASE_KEYWORDS:
        if keyword in q_lower:
            logger.info(f"Retrieve Gate Override: Found codebase keyword '{keyword}' in question. Forcing retrieval.")
            return True

    try:
        llm = _get_llm()
        prompt = _RETRIEVE_GATE_PROMPT.format(question=question)
        res = llm.invoke(prompt)
        content_str = res.content if hasattr(res, "content") else str(res)
        response = str(content_str).strip().lower()
        logger.info(f"Retrieve Gate Response: '{response}' for question: '{question[:60]}'")
        
        words = set(re.findall(r'\b\w+\b', response))
        if "no" in words:
            return False
        return "yes" in words or True
    except Exception as e:
        logger.error(f"Retrieve gate check failed: {e}. Defaulting to True.")
        return True

@traceable(run_type="chain")
def grade_chunk_relevance(question: str, chunk_text: str) -> str:
    """
    Grades a single chunk's relevance to the question.
    Returns: 'relevant', 'ambiguous', or 'irrelevant'.
    """
    try:
        llm = _get_llm()
        prompt = _CHUNK_GRADER_PROMPT.format(question=question, chunk=chunk_text)
        res = llm.invoke(prompt)
        content_str = res.content if hasattr(res, "content") else str(res)
        response = str(content_str).strip().lower()
        
        # Parse output using strict word boundaries to handle conversational/verbose responses
        words = set(re.findall(r'\b\w+\b', response))
        if "irrelevant" in words:
            return "irrelevant"
        elif "relevant" in words:
            return "relevant"
        elif "ambiguous" in words:
            return "ambiguous"
        
        logger.warning(f"Unrecognized grader response: '{response}'. Defaulting to 'ambiguous'.")
        return "ambiguous"
    except Exception as e:
        logger.error(f"Chunk grading failed: {e}. Defaulting to 'relevant' to be safe.")
        return "relevant"
