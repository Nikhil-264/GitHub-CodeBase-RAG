"""
Grader Agent
============
Grades the relevance of retrieved code chunks and decides whether
a user question requires querying the codebase at all (retrieve gate).
"""

import os
from loguru import logger
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langsmith import traceable

load_dotenv()

LLM_MODEL  = os.getenv("LLM_MODEL", "qwen3")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_llm: OllamaLLM | None = None

def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        logger.info(f"Grader LLM loaded: {LLM_MODEL} @ {OLLAMA_URL}")
        _llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,  # deterministic grading
        )
    return _llm

_RETRIEVE_GATE_PROMPT = """You are a gatekeeper deciding if a user's question requires retrieving code from a codebase.
Questions that require retrieval include:
- Finding where functions/classes/variables are defined, used, or imported.
- Conceptual explanations of how features inside this specific codebase work.
- Tracing execution flow of features in the codebase.
- Debugging errors or understanding bugs in the codebase.
- Architectural design or file layout of the codebase.

Questions that do NOT require retrieval include:
- General programming questions (e.g., 'how do I write a binary search in Python?', 'explain the difference between REST and GraphQL').
- General greetings or conversational messages (e.g., 'hello', 'who are you?').
- General knowledge or off-topic questions.

Respond with exactly 'yes' if retrieval is needed, or 'no' if it is not. No other characters or punctuation.

Question: {question}

Retrieve:"""

_CHUNK_GRADER_PROMPT = """You are a grader assessing relevance of a retrieved code chunk to a user question.
If the chunk contains code, comments, or documentation relevant to answering the user's question, grade it as 'relevant'.
If the chunk is partially relevant, or it is unclear, grade it as 'ambiguous'.
If the chunk is completely unrelated to the question, grade it as 'irrelevant'.

Provide a single word response: 'relevant', 'ambiguous', or 'irrelevant'. No explanation or punctuation.

Question: {question}
Code Chunk:
{chunk}

Grade:"""

@traceable(run_type="chain")
def check_need_retrieval(question: str) -> bool:
    """
    Decides whether a user question requires querying the codebase.
    Returns True if retrieval is needed, False otherwise.
    """
    try:
        llm = _get_llm()
        prompt = _RETRIEVE_GATE_PROMPT.format(question=question)
        response = llm.invoke(prompt).strip().lower()
        logger.info(f"Retrieve Gate Response: '{response}' for question: '{question[:60]}'")
        return "yes" in response
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
        response = llm.invoke(prompt).strip().lower()
        
        # Parse output
        if "relevant" in response:
            return "relevant"
        elif "ambiguous" in response:
            return "ambiguous"
        elif "irrelevant" in response:
            return "irrelevant"
        
        logger.warning(f"Unrecognized grader response: '{response}'. Defaulting to 'ambiguous'.")
        return "ambiguous"
    except Exception as e:
        logger.error(f"Chunk grading failed: {e}. Defaulting to 'relevant' to be safe.")
        return "relevant"
