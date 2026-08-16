"""
Multi-tier fallback chunker
===========================
Tier 1  →  tree-sitter              (AST-aware, individual language packages)
Tier 2  →  regex patterns           (detects functions/classes by syntax)
Tier 3  →  line sliding window      (always works, any language)
"""

import re
from pathlib import Path
from typing import Any
from loguru import logger


# ── Tier 1: tree-sitter (individual packages, works on Python 3.14) ──
try:
    import tree_sitter  # type: ignore
    TREE_SITTER_AVAILABLE = True
    logger.info("tree-sitter available — Tier 1 active")
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.info("tree-sitter not available — starting from Tier 2")



# Map file extension → individual pip package + language() function
# Each package exposes a `language()` function directly
_TS_LANG_MAP: dict[str, str] = {
    "py":    "tree_sitter_python",
    "js":    "tree_sitter_javascript",
    "ts":    "tree_sitter_typescript",
    "tsx":   "tree_sitter_typescript",   # tsx is inside tree-sitter-typescript
    "jsx":   "tree_sitter_javascript",
    "go":    "tree_sitter_go",
    "rs":    "tree_sitter_rust",
    "java":  "tree_sitter_java",
    "cpp":   "tree_sitter_cpp",
    "c":     "tree_sitter_c",
    "h":     "tree_sitter_cpp",
    "cs":    "tree_sitter_c_sharp",
    "rb":    "tree_sitter_ruby",
    "php":   "tree_sitter_php",
    "swift": "tree_sitter_swift",
    "html":  "tree_sitter_html",
    "css":   "tree_sitter_css",
    "scss":  "tree_sitter_css",          # fallback to CSS parser
    "sass":  "tree_sitter_css",          # fallback to CSS parser
    "json":  "tree_sitter_json",
    "yaml":  "tree_sitter_yaml",
    "yml":   "tree_sitter_yaml",
    "md":    "tree_sitter_markdown",
}

_AST_NODE_TYPES = {
    # General Programming Languages
    "function_definition",
    "class_definition",
    "function_declaration",
    "method_declaration",
    "arrow_function",
    "impl_item",
    "function_item",

    # HTML / Markup
    "element",
    "script_element",
    "style_element",

    # CSS / Stylesheets
    "rule_set",
    "media_statement",

    # JSON / YAML
    "pair",
    "object",
    "block_mapping_pair",

    # Markdown
    "section",
}


# ── Tier 2: regex patterns per language ─────────────────────
REGEX_PATTERNS: dict[str, list[str]] = {
    "py": [
        r"^(async\s+)?def\s+\w+",
        r"^class\s+\w+",
    ],
    "js": [
        r"^(export\s+)?(async\s+)?function\s+\w+",
        r"^(export\s+)?class\s+\w+",
        r"^(const|let|var)\s+\w+\s*=\s*(async\s+)?\(",
    ],
    "ts": [
        r"^(export\s+)?(async\s+)?function\s+\w+",
        r"^(export\s+)?(abstract\s+)?class\s+\w+",
        r"^(export\s+)?(const|let)\s+\w+\s*=\s*(async\s+)?\(",
        r"^(export\s+)?interface\s+\w+",
        r"^(export\s+)?type\s+\w+\s*=",
    ],
    "tsx": [
        r"^(export\s+)?(default\s+)?(async\s+)?function\s+\w+",
        r"^(export\s+)?(const|let)\s+\w+\s*=\s*\(",
        r"^(export\s+)?(abstract\s+)?class\s+\w+",
    ],
    "jsx": [
        r"^(export\s+)?(default\s+)?(async\s+)?function\s+\w+",
        r"^(export\s+)?(const|let)\s+\w+\s*=\s*\(",
    ],
    "java": [
        r"^\s*(public|private|protected|static|\s)+[\w<>\[\]]+\s+\w+\s*\(",
        r"^\s*(public|private|protected)?\s*(abstract\s+)?class\s+\w+",
        r"^\s*(public)?\s*interface\s+\w+",
    ],
    "go": [
        r"^func\s+(\(\w+\s+\*?\w+\)\s+)?\w+\s*\(",
        r"^type\s+\w+\s+struct",
        r"^type\s+\w+\s+interface",
    ],
    "rs": [
        r"^(pub\s+)?(async\s+)?fn\s+\w+",
        r"^(pub\s+)?struct\s+\w+",
        r"^(pub\s+)?enum\s+\w+",
        r"^(pub\s+)?impl(\s+\w+)?\s+\w+",
        r"^(pub\s+)?trait\s+\w+",
    ],
    "cpp": [
        r"^\w[\w\s\*&:<>]+\s+\w+\s*\([^;]*$",
        r"^class\s+\w+",
        r"^struct\s+\w+",
        r"^namespace\s+\w+",
    ],
    "c": [
        r"^\w[\w\s\*]+\s+\w+\s*\([^;]*$",
        r"^struct\s+\w+",
        r"^typedef\s+",
    ],
    "h": [
        r"^\w[\w\s\*&:<>]+\s+\w+\s*\([^;]*$",
        r"^class\s+\w+",
        r"^struct\s+\w+",
        r"^namespace\s+\w+",
        r"^#ifndef\s+",
        r"^#define\s+",
    ],
    "cs": [
        r"^\s*(public|private|protected|internal|static|\s)+(async\s+)?\w+\s+\w+\s*\(",
        r"^\s*(public|private|protected)?\s*(abstract\s+)?class\s+\w+",
        r"^\s*(public)?\s*interface\s+\w+",
    ],
    "rb": [
        r"^\s*def\s+\w+",
        r"^\s*class\s+\w+",
        r"^\s*module\s+\w+",
    ],
    "php": [
        r"^\s*(public|private|protected|static|\s)*function\s+\w+",
        r"^\s*(abstract\s+)?class\s+\w+",
        r"^\s*interface\s+\w+",
    ],
    "swift": [
        r"^\s*(public|private|internal|open|fileprivate)?\s*(func)\s+\w+",
        r"^\s*(public|private|internal|open)?\s*(class|struct|enum|protocol)\s+\w+",
    ],
    "html": [
        r"^\s*<[a-zA-Z0-9_-]+[^>]*>",
        r"^\s*<!DOCTYPE",
    ],
    "css": [
        r"^\s*[\.#a-zA-Z0-9_-]+(\s*,\s*[\.#a-zA-Z0-9_-]+)*\s*\{",
        r"^\s*@media",
    ],
    "scss": [
        r"^\s*[\.#a-zA-Z0-9_-]+(\s*,\s*[\.#a-zA-Z0-9_-]+)*\s*\{",
        r"^\s*@media",
        r"^\s*\$[a-zA-Z0-9_-]+\s*:",
    ],
    "sass": [
        r"^\s*[\.#a-zA-Z0-9_-]+(\s*,\s*[\.#a-zA-Z0-9_-]+)*\s*\{",
        r"^\s*@media",
    ],
    "json": [
        r"^\s*\"[a-zA-Z0-9_-]+\"\s*:",
        r"^\s*[\{\[]",
    ],
    "yaml": [
        r"^[a-zA-Z0-9_-]+\s*:",
        r"^-\s+\w+",
    ],
    "yml": [
        r"^[a-zA-Z0-9_-]+\s*:",
        r"^-\s+\w+",
    ],
    "md": [
        r"^#+\s+\S+",
        r"^-\s+",
    ],
}

CHUNK_LINES   = 100
CHUNK_OVERLAP = 20


# ════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════

from langsmith import traceable

@traceable(run_type="tool")
def chunk_file(file_meta: dict) -> list[dict]:
    path     = file_meta["path"]
    language = file_meta["language"]

    try:
        code = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Cannot read '{path}': {e}")
        return []

    if not code.strip():
        return []

    # ── Tier 1: tree-sitter ──────────────────────────────────
    if TREE_SITTER_AVAILABLE and language in _TS_LANG_MAP:
        chunks = _tier1_treesitter(code, language, file_meta)
        if chunks:
            logger.debug(f"[Tier 1 - AST]   {file_meta['relative_path']} → {len(chunks)} chunks")
            return chunks
        logger.debug(f"Tier 1 gave no chunks for '{path}', trying Tier 2")

    # ── Tier 2: regex ────────────────────────────────────────
    if language in REGEX_PATTERNS:
        chunks = _tier2_regex(code, language, file_meta)
        if chunks:
            logger.debug(f"[Tier 2 - Regex] {file_meta['relative_path']} → {len(chunks)} chunks")
            return chunks
        logger.debug(f"Tier 2 gave no chunks for '{path}', falling to Tier 3")

    # ── Tier 3: sliding window ───────────────────────────────
    chunks = _tier3_sliding_window(code, file_meta)
    logger.debug(f"[Tier 3 - Lines] {file_meta['relative_path']} → {len(chunks)} chunks")
    return chunks


@traceable(run_type="tool")
def chunk_files(file_list: list[dict]) -> list[dict]:
    all_chunks  = []
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}

    for file_meta in file_list:
        chunks = chunk_file(file_meta)
        all_chunks.extend(chunks)
        if chunks:
            tier_counts[chunks[0]["metadata"]["chunking_tier"]] += 1

    logger.info(
        f"Chunking done: {len(all_chunks)} chunks | "
        f"Tier1={tier_counts['tier1']} Tier2={tier_counts['tier2']} Tier3={tier_counts['tier3']} files"
    )
    return all_chunks


# ════════════════════════════════════════════════════════════
# Tier 1 — tree-sitter (individual packages)
# ════════════════════════════════════════════════════════════

# Cache loaded languages so we don't re-import on every file
_lang_cache: dict[str, Any] = {}


def _load_ts_language(language: str):
    """
    Dynamically import the individual tree-sitter language package.

    tree-sitter-typescript exposes two grammars:
        tree_sitter_typescript.language_typescript()
        tree_sitter_typescript.language_tsx()
    All others expose:
        tree_sitter_<lang>.language()
    """
    if language in _lang_cache:
        return _lang_cache[language]

    module_name = _TS_LANG_MAP[language]
    try:
        mod = __import__(module_name, fromlist=["language"])

        if language == "tsx":
            lang = mod.language_tsx()
        elif language == "ts":
            lang = mod.language_typescript()
        else:
            lang = mod.language()

        from tree_sitter import Language  # type: ignore
        ts_lang = Language(lang)  # type: ignore
        _lang_cache[language] = ts_lang
        return ts_lang

    except Exception as e:
        logger.debug(f"Could not load tree-sitter language '{language}': {e}")
        _lang_cache[language] = None
        return None


def _tier1_treesitter(code: str, language: str, file_meta: dict) -> list[dict]:
    ts_lang = _load_ts_language(language)
    if ts_lang is None:
        return []

    # Encode code to UTF-8 bytes to safely use tree-sitter's byte offsets for slicing
    code_bytes = code.encode("utf-8")

    try:
        from tree_sitter import Parser
        parser = Parser(ts_lang)
        tree   = parser.parse(code_bytes)
    except Exception as e:
        logger.debug(f"tree-sitter parse error: {e}")
        return []

    chunks = []
    for node in tree.root_node.children:
        if node.type not in _AST_NODE_TYPES:
            continue
        # Slice using bytes and decode
        chunk_bytes = code_bytes[node.start_byte:node.end_byte]
        text = chunk_bytes.decode("utf-8", errors="ignore")
        name = _get_node_name(node, code_bytes)
        chunks.append(_make_chunk(
            text=text, file_meta=file_meta,
            chunk_type=node.type, chunk_name=name,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            tier="tier1",
        ))
    return chunks


def _get_node_name(node, code_bytes: bytes) -> str:
    # Try finding typical identifier nodes
    for child in node.children:
        if child.type in ("identifier", "tag_name", "property_name", "key"):
            return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore")
            
    # Language-specific type extraction
    if node.type == "pair" and node.children:
        # JSON/YAML key-value pair
        key_node = node.children[0]
        return code_bytes[key_node.start_byte:key_node.end_byte].decode("utf-8", errors="ignore").strip("\"'")
        
    if node.type == "rule_set" and node.children:
        # CSS selector name
        selector_node = node.children[0]
        return code_bytes[selector_node.start_byte:selector_node.end_byte].decode("utf-8", errors="ignore").strip()
        
    if node.type == "section" and node.children:
        # Markdown heading name
        for child in node.children:
            if child.type == "atx_heading":
                return code_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
                
    return "unknown"


# ════════════════════════════════════════════════════════════
# Tier 2 — Regex pattern splitting
# ════════════════════════════════════════════════════════════

def _tier2_regex(code: str, language: str, file_meta: dict) -> list[dict]:
    patterns = REGEX_PATTERNS[language]
    combined = re.compile("|".join(patterns), re.MULTILINE)
    lines    = code.splitlines()

    split_points = [0]
    for i, line in enumerate(lines):
        if i > 0 and combined.match(line):
            split_points.append(i)
    split_points.append(len(lines))

    chunks = []
    for idx in range(len(split_points) - 1):
        start = split_points[idx]
        end   = split_points[idx + 1]
        text  = "\n".join(lines[start:end]).strip()
        if not text:
            continue
        name = _extract_name_from_line(lines[start].strip())
        chunks.append(_make_chunk(
            text=text, file_meta=file_meta,
            chunk_type="regex_block", chunk_name=name,
            start_line=start + 1, end_line=end,
            tier="tier2",
        ))
    return chunks


def _extract_name_from_line(line: str) -> str:
    match = re.search(r"\b(\w+)\s*[\(\{:]", line)
    return match.group(1) if match else line[:40].strip()


# ════════════════════════════════════════════════════════════
# Tier 3 — Sliding window
# ════════════════════════════════════════════════════════════

def _tier3_sliding_window(code: str, file_meta: dict) -> list[dict]:
    lines  = code.splitlines()
    chunks = []
    start  = 0
    while start < len(lines):
        end  = min(start + CHUNK_LINES, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(_make_chunk(
                text=text, file_meta=file_meta,
                chunk_type="sliding_window",
                chunk_name=f"lines_{start + 1}_{end}",
                start_line=start + 1, end_line=end,
                tier="tier3",
            ))
        start += CHUNK_LINES - CHUNK_OVERLAP
    return chunks


# ════════════════════════════════════════════════════════════
# Shared chunk builder
# ════════════════════════════════════════════════════════════

def _make_chunk(
    text: str, file_meta: dict, chunk_type: str,
    chunk_name: str, start_line: int, end_line: int, tier: str,
) -> dict:
    return {
        "text": text,
        "metadata": {
            "repo":          file_meta.get("repo", ""),
            "file_path":     file_meta["relative_path"],
            "language":      file_meta["language"],
            "chunk_type":    chunk_type,
            "chunk_name":    chunk_name,
            "start_line":    start_line,
            "end_line":      end_line,
            "size_kb":       file_meta["size_kb"],
            "chunking_tier": tier,
        },
    }


# ════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else __file__
    p = Path(test_path)

    meta = {
        "path":          str(p),
        "relative_path": p.name,
        "language":      p.suffix.lstrip("."),
        "size_kb":       round(p.stat().st_size / 1024, 2),
        "repo":          "test",
    }

    chunks = chunk_file(meta)
    print(f"\n{len(chunks)} chunks from '{p.name}':\n")
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        print(f"  [{i}] [{m['chunking_tier']}] {m['chunk_type']} '{m['chunk_name']}'"
              f" — lines {m['start_line']}–{m['end_line']}")
