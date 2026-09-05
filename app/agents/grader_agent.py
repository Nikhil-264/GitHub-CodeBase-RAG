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
from langsmith import traceable
from app.llm_provider import get_chat_llm

load_dotenv()

_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm(temperature=0)   # deterministic grading
    return _llm

# Keywords that explicitly indicate codebase-specific queries about this RAG system.
# NOTE: these only ever match questions about *this tool's own* source — they
# give zero help once a *different* repository is ingested and active. That
# case is handled by `_PROJECT_META_PATTERNS` + the has_repo-aware prompt below.
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

# Generic "tell me about the active repo" phrasing — repo-agnostic, so it
# works regardless of which project is ingested (unlike _CODEBASE_KEYWORDS).
_PROJECT_META_PATTERNS = [
    r"\bthis (project|repo|repository|codebase|application|app)\b",
    r"\bthe (project|repo|repository|codebase)\b.*\b(about|do|for|structure|architecture)\b",
    r"^what (is|does) this\b",
    r"^(explain|describe|summarise|summarize) (this|the) (project|repo|repository|codebase)",
]
_PROJECT_META_RE = re.compile("|".join(_PROJECT_META_PATTERNS), re.IGNORECASE)

_RETRIEVE_GATE_PROMPT = """You are a gatekeeper deciding if a user's question requires retrieving code from a codebase.
{repo_context}
Questions that require retrieval include:
- Finding where functions/classes/variables are defined, used, or imported.
- Conceptual explanations of how features inside the active codebase work.
- Tracing execution flow of features in the codebase.
- Debugging errors or understanding bugs in the codebase.
- Architectural design or file layout of the codebase, INCLUDING generic-sounding
  questions like "what is this project about" or "what does this repo do" — when a
  codebase is loaded, those are asking about it, not asking for general knowledge.

Questions that do NOT require retrieval include:
- General programming questions with no reference to "this"/"the" project (e.g.,
  'how do I write a binary search in Python?', 'explain the difference between REST and GraphQL').
- General greetings or conversational messages (e.g., 'hello', 'who are you?').
- General knowledge or off-topic questions unrelated to any codebase.

Examples:
{examples}

Respond with exactly 'yes' or 'no'. No explanation or punctuation.

Question: {question}

Retrieve:"""

_EXAMPLES_WITH_REPO = """Question: What is this project about?
Retrieve: yes

Question: How is the animation implemented?
Retrieve: yes

Question: Tell me how the login flow works
Retrieve: yes

Question: hello, how are you today?
Retrieve: no

Question: Write a python function to compute fibonacci numbers.
Retrieve: no

Question: What's the difference between REST and GraphQL?
Retrieve: no"""

_EXAMPLES_NO_REPO = """Question: What is this project about?
Retrieve: no

Question: hello, how are you today?
Retrieve: no

Question: Write a python function to compute fibonacci numbers.
Retrieve: no

Question: What's the difference between REST and GraphQL?
Retrieve: no"""

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
def check_need_retrieval(question: str, has_repo: bool = False) -> bool:
    """
    Decides whether a user question requires querying the codebase.
    Returns True if retrieval is needed, False otherwise.

    `has_repo` should reflect whether a repository is actively bound to the
    session (RAGState.repo_name). Without it, the LLM gate has no way to know
    a codebase is even loaded, so repo-agnostic meta-questions like "what is
    this project about" can read as generic/off-topic small talk and get
    wrongly routed to the no-context direct-answer path.
    """
    q_lower = question.lower()
    # Rule-based keyword override (matches on this tool's own internals only)
    for keyword in _CODEBASE_KEYWORDS:
        if keyword in q_lower:
            logger.info(f"Retrieve Gate Override: Found codebase keyword '{keyword}' in question. Forcing retrieval.")
            return True

    # Repo-agnostic override: "what is this project about" etc., while a repo is active
    if has_repo and _PROJECT_META_RE.search(question):
        logger.info("Retrieve Gate Override: Project-meta question with an active repo. Forcing retrieval.")
        return True

    try:
        llm = _get_llm()
        if has_repo:
            repo_context = "A codebase is currently loaded and active for this conversation."
            examples = _EXAMPLES_WITH_REPO
        else:
            repo_context = "No codebase is currently loaded for this conversation."
            examples = _EXAMPLES_NO_REPO
        prompt = _RETRIEVE_GATE_PROMPT.format(question=question, repo_context=repo_context, examples=examples)
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
