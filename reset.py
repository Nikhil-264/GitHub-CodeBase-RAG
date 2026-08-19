"""
Reset Script
============
Clears all stored chat history in Postgres, wipes ChromaDB vector embeddings,
and deletes all cached BM25 indices and cloned repos in `repos/`.
"""

import os
import shutil
import asyncio
from pathlib import Path
from loguru import logger
from sqlalchemy import delete

from app.memory.db import AsyncSessionLocal, init_db
from app.memory.models import ChatMessage, ChatSession
from app.retrieval.vector_store import delete_collection


async def clear_database():
    logger.info("Clearing Postgres chat history...")
    await init_db()
    async with AsyncSessionLocal() as db:
        await db.execute(delete(ChatMessage))
        await db.execute(delete(ChatSession))
        await db.commit()
    logger.success("Postgres database cleared.")


def clear_vector_store():
    logger.info("Clearing ChromaDB collection...")
    try:
        delete_collection("codebase")
        logger.success("ChromaDB vector collection deleted.")
    except Exception as e:
        logger.warning(f"Could not delete ChromaDB collection: {e}")

    chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
    if os.path.exists(chroma_path):
        try:
            shutil.rmtree(chroma_path)
            logger.success(f"Removed ChromaDB directory '{chroma_path}'.")
        except Exception as e:
            logger.warning(f"Could not remove '{chroma_path}' directory: {e}")


def clear_repos_directory():
    logger.info("Clearing repos directory...")
    repos_dir = Path("repos")
    if repos_dir.exists():
        for item in repos_dir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception as e:
                logger.warning(f"Could not remove {item}: {e}")
        logger.success("Repos directory cleared.")


async def reset_all():
    logger.info("Starting complete reset...")
    await clear_database()
    clear_vector_store()
    clear_repos_directory()
    logger.success("✨ Complete reset finished! System is fresh.")


if __name__ == "__main__":
    asyncio.run(reset_all())
