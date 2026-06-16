import os
import stat
from pathlib import Path
from loguru import logger

import sys
from dotenv import load_dotenv
load_dotenv()

SUPPORTED_EXTENSIONS = {
    '.py',
    '.js',
    '.ts',
    '.jsx',
    '.tsx',
    '.html',
    '.css',
    '.scss',
    '.sass',
    '.json',
    '.yaml',
    '.yml',
    '.md',
    '.txt',
    ".java", # Java
    ".cpp",  # C++
    ".c",    # C
    ".h",    # C/C++ headers
    ".go",   # Go
    ".rs",   # Rust
    ".rb",   # Ruby
    ".cs",   # C#
    ".php",  # PHP
    ".swift" # Swift
}

EXCLUDED_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    "coverage",
    ".next",
    "out",
    "target",   # Rust build
    "vendor",   # Go vendor
}

EXCLUDED_FILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "pnpm-lock.json",
    "poetry.lock",
    "cargo.lock",
    "uv.lock",
    "composer.lock",
    "mix.lock",
}

MAX_FILE_SIZE_KB = 200

from langsmith import traceable

@traceable(run_type="tool")
def scan_repo(repo_path : str)->list[dict]:
    """
    Recursively scan a cloned repo and return metadata
    for every supported code file.

    Each item in the returned list looks like:
    {
        "path":          "/abs/path/to/file.py",
        "relative_path": "src/auth/login.py",
        "language":      "py",
        "size_kb":       4.2,
    }
    """
    repo_root = Path(repo_path)
    files = []

    # Check if the path exists
    if not repo_root.exists():
        logger.warning(f"Repository path does not exist: {repo_path}")
        return files

    # Using os.walk allows us to prune EXCLUDED_DIRS in-place, preventing
    # the scanner from traversing large directories like node_modules or .git.
    for root, dirs, filenames in os.walk(repo_root):
        # Case-insensitive check for excluded directories
        dirs[:] = [d for d in dirs if d.lower() not in EXCLUDED_DIRS]

        for filename in filenames:
            # Case-insensitive check for excluded files
            if filename.lower() in EXCLUDED_FILES:
                continue

            path = Path(root) / filename
            
            # Case-insensitive suffix check
            suffix_lower = path.suffix.lower()
            if suffix_lower not in SUPPORTED_EXTENSIONS:
                continue

            try:
                size_kb = path.stat().st_size / 1024
            except OSError as e:
                logger.warning(f"Could not access file size for {path}: {e}")
                continue

            if size_kb > MAX_FILE_SIZE_KB:
                logger.debug(f"Skipping {path} : size {size_kb:.2f}kb > {MAX_FILE_SIZE_KB}kb")
                continue

            files.append({
                "path": str(path.resolve()),
                "relative_path": str(path.relative_to(repo_root)),
                "language": suffix_lower.lstrip('.'), # Remove leading dot and make lowercase
                "size_kb": round(size_kb, 2),
            })

    logger.info(f"Scan complete : {len(files)} code files found in '{repo_path}'")
    return files


def print_summary(files: list[dict]) -> None:
    """Print a breakdown of files by language."""
    from collections import Counter
    counts = Counter(f["language"] for f in files)
    print("\nFiles by language:")
    for lang, count in counts.most_common():
        print(f"  .{lang:<10} {count} files")
    print(f"\n  Total: {len(files)} files")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo = sys.argv[1]
    else:
        repos_base = Path(os.getenv("REPOS_PATH", "./repos"))
        repo = "./repos/langgraph"  # default fallback
        if repos_base.exists() and repos_base.is_dir():
            # Filter out .git folder and find directories
            subdirs = [d for d in repos_base.iterdir() if d.is_dir() and d.name != ".git"]
            if subdirs:
                # Sort by modification time to find the newest cloned/modified repo
                subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                repo = str(subdirs[0])
                logger.info(f"No repository path provided. Auto-detected latest: '{repo}'")
            else:
                logger.warning(f"No repositories found in base path '{repos_base}'. Using fallback: '{repo}'")
        else:
            logger.warning(f"Base repository directory '{repos_base}' does not exist. Using fallback: '{repo}'")

    files = scan_repo(repo)
    print_summary(files)