"""
Intent Agent
============
Classifies the user's question into a retrieval intent.
This tells the rest of the pipeline HOW to search.

Intents:
    code_search   →  "Where is X defined / used?"
    explain       →  "How does X work? / Explain X"
    trace_flow    →  "Walk me through the flow of X"
    architecture  →  "How is the project structured?"
    debug         →  "Why does X fail? / What's wrong with X?"

Two modes:
    Mode A → rule-based  (fast, no LLM call, works offline)
    Mode B → LLM-based   (smarter, handles ambiguous questions)
"""

import os
import re
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

INTENT_MODE = os.getenv("INTENT_MODE", "rules")   # "rules" or "llm"


# ════════════════════════════════════════════════════════════
# Intent definitions
# ════════════════════════════════════════════════════════════

INTENTS = {
    "code_search" : {
        "description" : "Locate where something is defined, called, or used",
        "keywords"    : [
            "where", "which file", "find", "locate", "defined",
            "called", "imported", "used", "show me", "what file",
        ],
    },
    "explain" : {
        "description" : "Explain how something works conceptually",
        "keywords"    : [
            "how does", "explain", "what is", "what does",
            "describe", "tell me about", "what are",
        ],
    },
    "trace_flow" : {
        "description" : "Trace the execution flow or sequence of a feature",
        "keywords"    : [
            "flow", "trace", "sequence", "step by step", "pipeline",
            "walk me through", "what happens when", "process", "lifecycle",
        ],
    },
    "architecture" : {
        "description" : "Understand overall project or module structure",
        "keywords"    : [
            "architecture", "structure", "overview", "design",
            "how is", "organised", "organized", "layout", "modules",
            "folders", "project",
        ],
    },
    "debug" : {
        "description" : "Understand why something fails or behaves unexpectedly",
        "keywords"    : [
            "why", "error", "bug", "fail", "broken", "issue",
            "wrong", "problem", "fix", "exception", "crash",
        ],
    },
}

# Default when nothing matches
DEFAULT_INTENT = "explain"


# ════════════════════════════════════════════════════════════
# Mode A — Rule-based classification
# ════════════════════════════════════════════════════════════

def _classify_rules(question: str) -> str:
    q = question.lower()

    scores: dict[str, int] = {intent: 0 for intent in INTENTS}

    for intent, config in INTENTS.items():
        for keyword in config["keywords"]:
            if keyword in q:
                scores[intent] += 1

    best_intent = max(scores, key=lambda k: scores[k])

    # If nothing matched return default
    if scores[best_intent] == 0:
        logger.debug(f"No intent matched — defaulting to '{DEFAULT_INTENT}'")
        return DEFAULT_INTENT

    logger.debug(f"Rule-based intent scores: {scores} → '{best_intent}'")
    return best_intent


# ════════════════════════════════════════════════════════════
# Mode B — LLM-based classification
# ════════════════════════════════════════════════════════════

_INTENT_PROMPT = """You are a code assistant that classifies user questions.

Classify the question into exactly ONE of these intents:

- code_search   : finding where something is defined, used, or imported
- explain       : explaining how something works conceptually
- trace_flow    : tracing execution flow or sequence of a feature
- architecture  : understanding project structure or design
- debug         : understanding why something fails or behaves unexpectedly

Rules:
- Respond with ONLY the intent label. No explanation, no punctuation.
- If unsure, choose the closest match.

Question: {question}

Intent:"""


def _classify_llm(question: str) -> str:
    try:
        from app.llm_provider import get_chat_llm
        llm = get_chat_llm(temperature=0)
        prompt   = _INTENT_PROMPT.format(question=question)
        res = llm.invoke(prompt)
        content_str = res.content if hasattr(res, "content") else str(res)
        response = str(content_str).strip().lower()

        # extract valid intent from response
        for intent in INTENTS:
            if intent in response:
                logger.debug(f"LLM intent: '{intent}' for question: '{question[:60]}'")
                return intent

        logger.warning(f"LLM returned unrecognised intent '{response}' — falling back to rules")
        return _classify_rules(question)

    except Exception as e:
        logger.warning(f"LLM intent classification failed: {e} — falling back to rules")
        return _classify_rules(question)


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="chain")
def classify_intent(question: str) -> dict:
    """
    Classify the user's question into a retrieval intent.

    Returns:
    {
        "intent"      : "code_search",
        "description" : "Locate where something is defined, called, or used",
        "mode"        : "rules",
        "question"    : "Where is JWT generated?"
    }
    """
    if INTENT_MODE == "llm":
        intent = _classify_llm(question)
    else:
        intent = _classify_rules(question)

    result = {
        "intent"      : intent,
        "description" : INTENTS[intent]["description"],
        "mode"        : INTENT_MODE,
        "question"    : question,
    }

    logger.info(f"Intent → [{intent}] for: '{question[:80]}'")
    return result


# ════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    questions = [
        "Where is JWT generated?",
        "How does authentication work?",
        "Walk me through the login flow.",
        "Which files handle database operations?",
        "Why does the token expire immediately?",
        "How is the project structured?",
    ]

    print("\nIntent Classification Test\n" + "─" * 40)
    for q in questions:
        result = classify_intent(q)
        print(f"  Q: {q}")
        print(f"  → {result['intent']}  ({result['description']})\n")