import os
import shutil
from pathlib import Path

import git  # type: ignore
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

REPOS_PATH = os.getenv("REPOS_PATH", "./repos")

from langsmith import traceable

@traceable(run_type="tool")
def clone_repo(url : str) -> str:
    """
    shallow-clone a github repository locally.
    Return the path to the cloned repo
    Skips cloning if repo already exists on disk
    """

    repo_name = url.rstrip('/').split('/')[-1].replace('.git', '')
    target_path = Path(REPOS_PATH) / repo_name
    
    if target_path.exists():
        logger.info(f"Repo already exists at '{target_path}', skipping clone.")
        return str(target_path)

    Path(REPOS_PATH).mkdir(parents=True, exist_ok=True)

    logger.info(f"Cloning '{url}' into '{target_path}'...")
    git.Repo.clone_from(
        url,
        str(target_path),
        depth=1,
        no_single_branch=False,
    )

    logger.success(f"Clones Successfully : {repo_name}")
    return str(target_path)

def delete_repo(repo_path : str)->None:
    """
    Delete a cloned repo from disk.
    Useful for cleanup after indexing.
    """
    path = Path(repo_path)
    if path.exists():
        def remove_readonly(func, path, excinfo):
            import stat
            os.chmod(path, stat.S_IWRITE)
            func(path)

        try:
            shutil.rmtree(path, onexc=remove_readonly)
        except TypeError:
            shutil.rmtree(path, onerror=remove_readonly)  # type: ignore
        logger.info(f"Deleted repo at: {repo_path}")
    else:
        logger.warning(f"Repo path not found, nothing to delete: {repo_path}")

if __name__ == '__main__':
    path = clone_repo("https://github.com/Nikhil-264/ai-astrologer.git")
    print(f"Repo cloned at: {path}")
    