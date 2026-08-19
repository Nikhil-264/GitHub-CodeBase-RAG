"""
Answer Agent
============
Takes the analysis brief + reranked chunks and generates
a final answer using the LLM.

Prompt structure:
    [System instructions]
    [Context summary from Analysis Agent]
    [Code chunks with file labels]
    [User question]
    → Answer
"""

import os
from loguru import logger
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

LLM_MODEL  = os.getenv("LLM_MODEL",       "gemini-1.5-flash")


# ════════════════════════════════════════════════════════════
# Prompt templates
# ════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an expert code analyst helping developers understand codebases.

Rules:
- Answer using ONLY the provided code context. Do not guess or hallucinate.
- Always cite the source file for every piece of information (e.g. `auth/jwt.py`).
- If the answer spans multiple files, explain each file's role clearly.
- If the context does not contain enough information, say so explicitly.
- Format code references using backticks.
- Be concise but complete."""


_INTENT_INSTRUCTIONS = {
    "code_search" : "Point to the exact file and line range where it is defined or used.",
    "explain"     : "Give a clear conceptual explanation, then back it up with the relevant code.",
    "trace_flow"  : "Walk through the execution step by step, file by file, in order.",
    "architecture": "Give a high-level overview of how the files and modules are organised.",
    "debug"       : "Identify what could cause the issue and where in the code it originates.",
}


# ════════════════════════════════════════════════════════════
# LLM singleton
# ════════════════════════════════════════════════════════════

_llm: ChatGoogleGenerativeAI | None = None


def _get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        logger.info(f"Loading LLM: {LLM_MODEL}")
        _llm = ChatGoogleGenerativeAI(
            model       = LLM_MODEL,
            temperature = 0.1,    # slight creativity for explanations
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    return _llm


# ════════════════════════════════════════════════════════════
# Prompt builder
# ════════════════════════════════════════════════════════════

def _build_prompt(brief: dict, intent: str, chat_history: str = "", stricter: bool = False) -> str:
    question        = brief["question"]
    chunks          = brief["chunks"]
    context_summary = brief["context_summary"]

    intent_instruction = _INTENT_INSTRUCTIONS.get(intent, "Answer clearly and completely.")

    context_parts = []
    for chunk in chunks:
        m = chunk["metadata"]
        header = f"### File: {m['file_path']} | Lines {m.get('start_line','?')}–{m.get('end_line','?')}"
        context_parts.append(f"{header}\n```{m.get('language','')}\n{chunk['text']}\n```")
    code_context = "\n\n".join(context_parts)

    system_prompt = _SYSTEM_PROMPT
    if stricter:
        system_prompt += "\n\nCRITICAL: Answer using ONLY facts directly shown in the code context. Do not make assumptions, extrapolate, or bring in outside knowledge. If the code does not explicitly show it, state that it is not in the context."

    prompt = f"""{system_prompt}

─── Conversation So Far ───────────────────────────────────
{chat_history}

─── Context Summary ───────────────────────────────────────
{context_summary}

─── Code Context ──────────────────────────────────────────
{code_context}

─── Current Question ──────────────────────────────────────
{question}

─── Instructions ──────────────────────────────────────────
{intent_instruction}
If the current question refers to something discussed earlier (e.g. "what about X", "and that file"), use the conversation above to resolve it.

Answer:"""
    return prompt


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="chain")
def answer(brief: dict, intent: str = "explain", chat_history: str = "", stricter: bool = False) -> dict:
    """
    Generate a final answer from the analysis brief.

    Args:
        brief    : output of analysis_agent.analyse()
        intent   : classified intent string
        stricter : whether to enforce strict factual correctness

    Returns:
    {
        "answer"       : str,          # the LLM's response
        "sources"      : list[str],    # unique file paths cited
        "chunks_used"  : int,
        "intent"       : str,
        "primary_files": list[str],
    }
    """
    if not brief["chunks"]:
        return {
            "answer"        : "I could not find relevant code for your question. Try rephrasing or indexing the repository first.",
            "sources"       : [],
            "chunks_used"   : 0,
            "intent"        : intent,
            "primary_files" : [],
        }

    prompt = _build_prompt(brief, intent, chat_history, stricter=stricter)

    logger.info(f"Generating answer | intent={intent} | chunks={len(brief['chunks'])} | stricter={stricter}")
    logger.debug(f"Prompt length: {len(prompt)} chars")

    try:
        llm      = _get_llm()
        response = llm.invoke(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "answer"        : f"LLM error: {e}",
            "sources"       : [],
            "chunks_used"   : 0,
            "intent"        : intent,
            "primary_files" : brief["primary_files"],
        }

    sources = sorted({
        c["metadata"]["file_path"]
        for c in brief["chunks"]
    })

    result = {
        "answer"        : response.strip(),
        "sources"       : sources,
        "chunks_used"   : len(brief["chunks"]),
        "intent"        : intent,
        "primary_files" : brief["primary_files"],
    }

    logger.success(f"Answer generated | {len(sources)} source files cited")
    return result


if __name__ == "__main__":
    # Minimal smoke test
    dummy_brief = {
        "question"        : "Where is JWT generated?",
        "chunks"          : [
            {
                "text"     : "def generate_token(user_id):\n    return jwt.encode({'sub': user_id}, SECRET)",
                "metadata" : {
                    "file_path"  : "auth/jwt.py",
                    "chunk_name" : "generate_token",
                    "chunk_type" : "function_definition",
                    "start_line" : 1,
                    "end_line"   : 3,
                    "language"   : "py",
                },
            }
        ],
        "file_map"        : {"auth/jwt.py": [{"chunk_name": "generate_token"}]},
        "cross_refs"      : [],
        "primary_files"   : ["auth/jwt.py"],
        "languages"       : ["py"],
        "context_summary" : "Most relevant file: auth/jwt.py",
    }

    result = answer(dummy_brief, intent="code_search")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")