"""
Critique Agent
==============
Self-RAG reflection module.
Grades generated answers for grounding (hallucination) and utility (relevance to question).
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
        logger.info(f"Critique LLM loaded: {LLM_MODEL}")
        _llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0,  # deterministic critique
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    return _llm

_GROUNDING_PROMPT = """You are a grader assessing if a response is grounded in / supported by a set of retrieved code chunks.
Your task is to check for hallucinations or unsupported statements.
Every claim or fact in the answer must be derived directly from the code context.
If the answer makes assumptions or claims not supported by the context, grade it as 'no'.
If the answer is fully supported by the context, grade it as 'yes'.

Provide exactly 'yes' or 'no'. No explanation or punctuation.

Code Context:
{context}

Answer:
{answer}

Grounded:"""

_UTILITY_PROMPT = """You are a grader assessing whether an answer actually addresses the user's question.
If the answer answers the user's question directly and is useful, grade it as 'yes'.
If the answer is off-topic, incomplete, or does not address the core question, grade it as 'no'.

Provide exactly 'yes' or 'no'. No explanation or punctuation.

Question: {question}

Answer:
{answer}

Useful:"""

@traceable(run_type="chain")
def check_grounding(answer_text: str, chunks: list[dict]) -> bool:
    """
    Grades if the answer is grounded in (supported by) the chunks context.
    Returns True if grounded, False if there are hallucinations.
    """
    if not chunks:
        # If no chunks were retrieved, it can only be grounded if the answer says it couldn't find relevant code.
        # But we let the utility checker handle off-topic answers.
        return True

    context_parts = []
    for chunk in chunks:
        m = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
        file_path = m.get("file_path", "unknown") if isinstance(m, dict) else "unknown"
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        context_parts.append(f"File: {file_path}\n{text}")
    context_text = "\n\n".join(context_parts)

    try:
        llm = _get_llm()
        prompt = _GROUNDING_PROMPT.format(context=context_text, answer=answer_text)
        res = llm.invoke(prompt)
        content_str = res.content if hasattr(res, "content") else str(res)
        response = str(content_str).strip().lower()
        logger.info(f"Grounding Critique Response: '{response}'")
        
        # Clean response and extract words
        cleaned = response.strip().lower()
        if cleaned in ["yes", "no"]:
            return cleaned == "yes"
        
        words = set(re.findall(r'\b\w+\b', cleaned))
        if "no" in words:
            return False
        return "yes" in words
    except Exception as e:
        logger.error(f"Grounding critique check failed: {e}. Defaulting to True.")
        return True

@traceable(run_type="chain")
def check_utility(answer_text: str, question: str) -> bool:
    """
    Grades if the answer actually addresses the user's question.
    Returns True if useful, False otherwise.
    """
    try:
        llm = _get_llm()
        prompt = _UTILITY_PROMPT.format(question=question, answer=answer_text)
        res = llm.invoke(prompt)
        content_str = res.content if hasattr(res, "content") else str(res)
        response = str(content_str).strip().lower()
        logger.info(f"Utility Critique Response: '{response}'")
        
        # Clean response and extract words
        cleaned = response.strip().lower()
        if cleaned in ["yes", "no"]:
            return cleaned == "yes"
        
        words = set(re.findall(r'\b\w+\b', cleaned))
        if "no" in words:
            return False
        return "yes" in words
    except Exception as e:
        logger.error(f"Utility critique check failed: {e}. Defaulting to True.")
        return True
