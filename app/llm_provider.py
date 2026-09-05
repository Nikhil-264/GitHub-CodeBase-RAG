"""
LLM Provider
============
Single factory for the chat LLM used by every agent (answer, grader,
critique, intent), switching between Google Gemini and a local Ollama
instance based on env config — so the backend choice lives in one place
instead of being duplicated (and drifting) across four agent modules.
"""

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")   # "gemini" or "ollama"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "ollama": "qwen2.5:3b",
}

# LLM_MODEL is shared across backends: if the user sets it explicitly, honor
# it regardless of backend; otherwise pick the right default for whichever
# backend is active (an unset LLM_MODEL should not send "gemini-2.5-flash"
# to Ollama or vice versa).
LLM_MODEL = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(LLM_BACKEND, _DEFAULT_MODELS["gemini"])


def get_chat_llm(temperature: float = 0.0):
    """
    Return a chat LLM instance per LLM_BACKEND:
        "gemini" (default) -> ChatGoogleGenerativeAI (cloud, needs GEMINI_API_KEY)
        "ollama"           -> ChatOllama (local, private, no API quota)
    """
    if LLM_BACKEND == "ollama":
        from langchain_ollama import ChatOllama
        logger.debug(f"Chat LLM: Ollama '{LLM_MODEL}' @ {OLLAMA_BASE_URL}")
        return ChatOllama(
            model       = LLM_MODEL,
            base_url    = OLLAMA_BASE_URL,
            temperature = temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    logger.debug(f"Chat LLM: Gemini '{LLM_MODEL}'")
    return ChatGoogleGenerativeAI(
        model          = LLM_MODEL,
        temperature    = temperature,
        google_api_key = os.getenv("GEMINI_API_KEY"),
    )
