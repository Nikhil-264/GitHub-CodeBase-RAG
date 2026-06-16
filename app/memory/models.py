"""
SQLAlchemy models for chat sessions + messages.
"""

import uuid
from datetime import datetime
from sqlalchemy import String, Text, ForeignKey, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.memory.db import Base


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id          : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_url    : Mapped[str]       = mapped_column(String(500), nullable=True)
    created_at  : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates = "session",
        cascade        = "all, delete-orphan",
        order_by       = "ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id          : Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  : Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id"))
    role        : Mapped[str]       = mapped_column(String(20))     # "user" or "assistant"
    content     : Mapped[str]       = mapped_column(Text)
    intent      : Mapped[str]       = mapped_column(String(50), nullable=True)
    sources     : Mapped[str]       = mapped_column(Text, nullable=True)   # JSON-encoded list
    created_at  : Mapped[datetime]  = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ChatSession"] = relationship(back_populates="messages")