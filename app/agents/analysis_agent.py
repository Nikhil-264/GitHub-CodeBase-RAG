"""
Analysis Agent
==============
Sits between retrieval and answer generation.
Analyses the retrieved chunks before passing them to the LLM.

Responsibilities:
    1. Detect which files are most relevant
    2. Build a mini file map from the chunks
    3. Detect cross-file relationships (imports, calls)
    4. Summarise the chunk context into a structured brief
       that makes the Answer Agent's job easier
"""

import re
from loguru import logger


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="chain")
def analyse(question: str, chunks: list[dict]) -> dict:
    """
    Analyse retrieved chunks and return a structured brief.

    Returns:
    {
        "question"        : str,
        "chunks"          : list[dict],      # same chunks, unchanged
        "file_map"        : dict,            # file → list of chunk summaries
        "cross_refs"      : list[str],       # detected import / call relationships
        "primary_files"   : list[str],       # top files by chunk count
        "languages"       : list[str],       # unique languages in chunks
        "context_summary" : str,             # one-paragraph brief for the LLM
    }
    """
    if not chunks:
        return _empty_brief(question)

    file_map      = _build_file_map(chunks)
    cross_refs    = _detect_cross_refs(chunks)
    primary_files = _rank_files(file_map)
    languages     = _unique_languages(chunks)
    summary       = _build_context_summary(question, file_map, cross_refs, primary_files)

    result = {
        "question"        : question,
        "chunks"          : chunks,
        "file_map"        : file_map,
        "cross_refs"      : cross_refs,
        "primary_files"   : primary_files,
        "languages"       : languages,
        "context_summary" : summary,
    }

    logger.info(
        f"Analysis: {len(chunks)} chunks | "
        f"{len(file_map)} files | "
        f"{len(cross_refs)} cross-refs"
    )
    return result


# ════════════════════════════════════════════════════════════
# File map
# ════════════════════════════════════════════════════════════

def _build_file_map(chunks: list[dict]) -> dict:
    """
    Group chunks by file path.

    Returns:
    {
        "auth/jwt.py": [
            {"chunk_name": "generate_jwt", "lines": "10-25", "type": "function_definition"},
            ...
        ],
        ...
    }
    """
    file_map: dict[str, list[dict]] = {}

    for chunk in chunks:
        m  = chunk["metadata"]
        fp = m["file_path"]

        if fp not in file_map:
            file_map[fp] = []

        file_map[fp].append({
            "chunk_name" : m.get("chunk_name", "unknown"),
            "lines"      : f"{m.get('start_line', '?')}–{m.get('end_line', '?')}",
            "type"       : m.get("chunk_type", "unknown"),
            "tier"       : m.get("chunking_tier", "?"),
        })

    return file_map


# ════════════════════════════════════════════════════════════
# Cross-reference detection
# ════════════════════════════════════════════════════════════

# Patterns to detect cross-file relationships in code (imports, includes, stylesheets, links)
_IMPORT_PATTERNS = [
    r"^import\s+([\w\.]+)",                          # Python/Swift: import x.y
    r"^from\s+([\w\.]+)\s+import",                   # Python: from x import y
    r"require\(['\"](.+?)['\"]\)",                   # JS/TS: require('...')
    r"import\s+.*\s+from\s+['\"](.+?)['\"]",        # JS/TS: import x from '...'
    r"^import\s+\"([\w\.\/]+)\"",                    # Go: import "pkg"
    r"use\s+([\w:]+);",                              # Rust: use crate::x
    r"^import\s+([\w\.\*]+);",                       # Java: import x.y.z;
    r'^#include\s+["<]([\w\.\/\\_-]+)[">]',          # C/C++: #include "header.h"
    r"^using\s+([\w\.]+);",                          # C#: using System.Text;
    r"^\s*(?:require|load)\s+['\"](.+?)['\"]",       # Ruby: require 'json'
    r"\b(?:include|require)(?:_once)?\s*\(?['\"](.+?)['\"]",  # PHP: require_once 'db.php'
    r"@import\s+['\"](.+?)['\"]",                    # CSS/SCSS/Sass: @import 'styles.css'
    r"@use\s+['\"](.+?)['\"]",                       # SCSS/Sass: @use 'variables'
    r'<link\s+[^>]*href=["\'](.+?)["\']',            # HTML stylesheet links
    r'<script\s+[^>]*src=["\'](.+?)["\']',          # HTML script sources
]

_COMBINED_PATTERN = re.compile(
    "|".join(_IMPORT_PATTERNS),
    re.MULTILINE,
)


def _detect_cross_refs(chunks: list[dict]) -> list[str]:
    """
    Scan chunk text for import/require statements.
    Returns unique list of detected relationships as strings.
    """
    refs: set[str] = set()

    for chunk in chunks:
        fp   = chunk["metadata"]["file_path"]
        text = chunk["text"]

        for match in _COMBINED_PATTERN.finditer(text):
            # grab first non-None group
            target = next((g for g in match.groups() if g), None)
            if target:
                refs.add(f"{fp}  →  {target}")

    return sorted(refs)


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def _rank_files(file_map: dict) -> list[str]:
    """Return files sorted by number of relevant chunks (most first)."""
    return sorted(file_map, key=lambda f: len(file_map[f]), reverse=True)


def _unique_languages(chunks: list[dict]) -> list[str]:
    return sorted({c["metadata"].get("language", "unknown") for c in chunks})


def _build_context_summary(
    question      : str,
    file_map      : dict,
    cross_refs    : list[str],
    primary_files : list[str],
) -> str:
    """
    Build a short structured summary injected before the code context.
    Helps the LLM orient itself before reading raw code chunks.
    """
    file_lines = "\n".join(
        f"  • {fp} ({len(chunks)} relevant chunk{'s' if len(chunks)>1 else ''})"
        for fp, chunks in file_map.items()
    )

    ref_lines = (
        "\n".join(f"  • {r}" for r in cross_refs[:10])
        if cross_refs
        else "  • None detected"
    )

    summary = f"""The following files are most relevant to the question:
{file_lines}

Detected cross-file relationships:
{ref_lines}

Primary file to focus on: {primary_files[0] if primary_files else 'unknown'}
"""
    return summary


def _empty_brief(question: str) -> dict:
    return {
        "question"        : question,
        "chunks"          : [],
        "file_map"        : {},
        "cross_refs"      : [],
        "primary_files"   : [],
        "languages"       : [],
        "context_summary" : "No relevant code chunks were found for this question.",
    }


if __name__ == "__main__":
    dummy = [
        {
            "text"     : "from auth.jwt import generate_token\ndef login(user, pwd): token = generate_token(user.id)",
            "metadata" : {
                "file_path"     : "auth/login.py",
                "chunk_name"    : "login",
                "chunk_type"    : "function_definition",
                "start_line"    : 5,
                "end_line"      : 10,
                "language"      : "py",
                "chunking_tier" : "tier1",
            },
        },
        {
            "text"     : "import jwt\ndef generate_token(user_id): return jwt.encode({'sub': user_id}, SECRET)",
            "metadata" : {
                "file_path"     : "auth/jwt.py",
                "chunk_name"    : "generate_token",
                "chunk_type"    : "function_definition",
                "start_line"    : 1,
                "end_line"      : 8,
                "language"      : "py",
                "chunking_tier" : "tier1",
            },
        },
    ]

    brief = analyse("How does login work?", dummy)
    print(f"\nPrimary files : {brief['primary_files']}")
    print(f"Cross-refs    : {brief['cross_refs']}")
    print(f"\nContext summary:\n{brief['context_summary']}")