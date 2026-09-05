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
from app.llm_provider import get_chat_llm

load_dotenv()


# ════════════════════════════════════════════════════════════
# Prompt templates
# ════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """You are an expert code analyst helping developers understand codebases.

Your job is to explain what the code actually DOES when it runs — the logic,
control flow, and behavior — not to summarize what symbols exist in it. Do not
stop at "this file includes X and defines constant Y"; explain what happens
when the code executes: what triggers it, what it computes or transforms,
what it calls, and what the end result is.

Rules:
- Answer using ONLY the provided code context. Do not guess or hallucinate.
- For every function, class, or block of logic you describe, cite it
  precisely as `file_path:start_line-end_line` (e.g. `server.c:42-58`)
  right after the claim — a bare file name is not a citation.
- Trace behavior across functions/files when the logic spans more than one
  chunk (e.g. "the server accepts a connection in `handle_client`
  (server.c:80-120), which spawns a thread running `worker_loop`
  (server.c:130-170) to read incoming requests...").
- Do not just restate declarations, imports, or constant definitions as if
  they were the answer — explain what they're used FOR in the actual logic.
  A list of `#include`s or `#define`s is never a complete answer on its own.
- If the context does not contain enough information to explain the actual
  behavior, say so explicitly and name what's missing (e.g. "the function
  that handles X wasn't retrieved").
- Format all code identifiers and citations using backticks.
- Be concise but complete — prioritize depth on the real logic over breadth
  across unrelated surface details.

Example of the citation style required — note the file:line right after each
function, not gathered at the end or left as a bare filename:

  The server accepts a connection and hands it off to a worker thread in
  `main` (`server.c:75-86`). That thread runs `handle_client`
  (`server.c:42-68`), which reads the request and dispatches on its command:
  a `LIST` request walks the configured directory and writes back the
  filenames, while a `GET <name>` request opens and streams that file back
  to the client."""


_INTENT_INSTRUCTIONS = {
    "code_search" : "Point to the exact file and line range where it is defined or used, and briefly explain what happens there.",
    "explain"     : "Explain what this code actually does when it runs — the real behavior and logic — not a structural summary of what it contains. Walk through the key functions' logic in the order they'd execute, citing file:line for each.",
    "trace_flow"  : "Walk through the execution step by step, file by file, in the order things actually happen at runtime — what calls what, what data flows where — citing file:line at each step.",
    "architecture": "Explain what the major components actually do and how they interact at runtime (e.g. what the client sends, what the server does with it, how they communicate) — not just a list of files and their imports.",
    "debug"       : "Identify what could cause the issue and explain the exact code path that leads there, citing file:line for each step.",
}


# ════════════════════════════════════════════════════════════
# LLM singleton
# ════════════════════════════════════════════════════════════

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_chat_llm(temperature=0.1)   # slight creativity for explanations
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
        answer_text = (response.content if hasattr(response, "content") else str(response)).strip()
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
        "answer"        : answer_text,
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