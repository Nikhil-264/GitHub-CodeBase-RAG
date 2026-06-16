"""
Database engine + session factory for Postgres chat memory.
"""

import os
from loguru import logger
from dotenv import load_dotenv
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://rag_user:rag_password@localhost:5432/rag_chat",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_      = AsyncSession,
    expire_on_commit = False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create tables if they don't exist. Call once on app startup."""
    from app.memory.models import ChatSession, ChatMessage  # noqa: F401
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.success("Postgres tables ready")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session