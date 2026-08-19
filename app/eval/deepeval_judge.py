"""
DeepEval Custom Gemini Judge Wrapper
====================================
Subclasses DeepEvalBaseLLM so that Google Gemini (gemini-2.5-flash) acts
as the LLM-as-a-Judge for all DeepEval metrics without requiring OpenAI keys.
"""

import os
from dotenv import load_dotenv
from loguru import logger
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class GoogleGeminiJudge(DeepEvalBaseLLM):
    """Custom DeepEval LLM Judge backed by Google's Gemma/Gemini models."""
    
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.api_key = os.getenv("GEMINI_API_KEY")
        self._llm = None
        super().__init__(model_name=self.model_name)

    def load_model(self):
        """Loads and returns the ChatGoogleGenerativeAI model instance."""
        if self._llm is None:
            logger.info(f"Initializing DeepEval Gemini Judge with model: {self.model_name}")
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.0,
                google_api_key=self.api_key
            )
        return self._llm

    def generate(self, prompt: str) -> str:
        """Synchronous generation for DeepEval evaluation calls with rate limit retry backoff."""
        import time
        model = self.load_model()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = model.invoke(prompt)
                return str(response.content).strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resourceexhausted" in err_str or "quota" in err_str:
                    wait_sec = (attempt + 1) * 7  # Wait 7s, 14s, 21s to respect 10 RPM limit
                    logger.warning(f"Gemini Rate Limit (429) hit. Retrying in {wait_sec}s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                else:
                    raise e
        raise RuntimeError("Gemini API rate limit retries exhausted.")

    async def a_generate(self, prompt: str) -> str:
        """Asynchronous generation for DeepEval evaluation calls with rate limit retry backoff."""
        import asyncio
        model = self.load_model()
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await model.ainvoke(prompt)
                return str(response.content).strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resourceexhausted" in err_str or "quota" in err_str:
                    wait_sec = (attempt + 1) * 7
                    logger.warning(f"Gemini Rate Limit (429) hit. Retrying in {wait_sec}s (Attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(wait_sec)
                else:
                    raise e
        raise RuntimeError("Gemini API rate limit retries exhausted.")

    def get_model_name(self) -> str:
        return self.model_name
