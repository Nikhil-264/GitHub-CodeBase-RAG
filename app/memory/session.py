"""
Chat memory operations — create sessions, save turns, load history.
"""

import json
import uuid
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.models import ChatSession, ChatMessage
from app.memory.db import AsyncSessionLocal

MAX_HISTORY_TURNS = 6   # how many past turns to inject into the LLM prompt


async def create_session(repo_url: str | None = None) -> str:
    async with AsyncSessionLocal() as db:
        session = ChatSession(repo_url=repo_url)
        db.add(session)
        await db.commit()
        logger.info(f"Created chat session: {session.id}")
        return str(session.id)


async def save_message(
    session_id : str,
    role       : str,
    content    : str,
    intent     : str | None = None,
    sources    : list[str] | None = None,
) -> None:
    async with AsyncSessionLocal() as db:
        msg = ChatMessage(
            session_id = uuid.UUID(session_id),
            role       = role,
            content    = content,
            intent     = intent,
            sources    = json.dumps(sources) if sources else None,
        )
        db.add(msg)
        await db.commit()


async def get_history(session_id: str, limit: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Return the last `limit` messages for a session, oldest first.
    Used to inject conversational context into prompts.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        messages = list(reversed(result.scalars().all()))

        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]


async def get_full_history(session_id: str) -> list[dict]:
    """Return the entire conversation for displaying in the chat UI."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == uuid.UUID(session_id))
            .order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()

        return [
            {
                "role"    : m.role,
                "content" : m.content,
                "intent"  : m.intent,
                "sources" : json.loads(m.sources) if m.sources else [],
            }
            for m in messages
        ]


def extract_repo_name(repo_url: str | None) -> str | None:
    if not repo_url:
        return None
    return repo_url.rstrip("/").split("/")[-1].removesuffix(".git")


async def get_session_info(session_id: str) -> dict | None:
    """Fetch session details including bound repo_url and repo_name."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        return None

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_uuid)
        )
        session = result.scalar_one_or_none()
        if not session:
            return None
        return {
            "id": str(session.id),
            "repo_url": session.repo_url,
            "repo_name": extract_repo_name(session.repo_url),
            "created_at": session.created_at.isoformat(),
        }


async def bind_session_repo(session_id: str, repo_url: str) -> None:
    """Bind or update the repo_url for an existing session."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        return

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_uuid)
        )
        session = result.scalar_one_or_none()
        if session:
            session.repo_url = repo_url
            await db.commit()
            logger.info(f"Bound session {session_id} to repo: {repo_url}")


async def list_sessions() -> list[dict]:
    """Return all chat sessions for a sidebar session picker."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatSession).order_by(ChatSession.created_at.desc())
        )
        sessions = result.scalars().all()
        return [
            {
                "id": str(s.id),
                "repo_url": s.repo_url,
                "repo_name": extract_repo_name(s.repo_url),
                "created_at": s.created_at.isoformat(),
            }
            for s in sessions
        ]


def format_history_for_prompt(history: list[dict]) -> str:
    """Turn history list into a readable block for the LLM prompt."""
    if not history:
        return "No previous conversation."

    lines = []
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)