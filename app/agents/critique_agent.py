"""
Critique Agent
==============
Self-RAG reflection module.
Grades generated answers for grounding (hallucination) and utility (relevance to question).
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
        logger.info(f"Critique LLM loaded: {LLM_MODEL} @ {OLLAMA_URL}")
        _llm = OllamaLLM(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,  # deterministic critique
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
        m = chunk["metadata"]
        context_parts.append(f"File: {m['file_path']}\n{chunk['text']}")
    context_text = "\n\n".join(context_parts)

    try:
        llm = _get_llm()
        prompt = _GROUNDING_PROMPT.format(context=context_text, answer=answer_text)
        response = llm.invoke(prompt).strip().lower()
        logger.info(f"Grounding Critique Response: '{response}'")
        return "yes" in response
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
        response = llm.invoke(prompt).strip().lower()
        logger.info(f"Utility Critique Response: '{response}'")
        return "yes" in response
    except Exception as e:
        logger.error(f"Utility critique check failed: {e}. Defaulting to True.")
        return True
