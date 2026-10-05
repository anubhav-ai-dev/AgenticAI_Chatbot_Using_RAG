"""
SQLAlchemy ORM models — ChatSession, Message, Document.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, JSON, String, Text
)
from sqlalchemy.orm import relationship

from backend.database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class ChatSession(Base):
    __tablename__ = "sessions"

    id         = Column(String, primary_key=True, default=_uuid)
    user_id    = Column(String, nullable=False, default="default", index=True)
    title      = Column(String, nullable=False, default="New Chat")
    model      = Column(String, nullable=False, default="claude-sonnet-4-5")
    mode       = Column(String, nullable=False, default="general")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages   = relationship(
        "Message", back_populates="session",
        cascade="all, delete-orphan", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"

    id         = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role       = Column(String, nullable=False)          # user | assistant | system
    content    = Column(Text, nullable=False)
    citations  = Column(JSON, nullable=True)             # [{id,source,page,snippet}]
    created_at = Column(DateTime, default=datetime.utcnow)

    session    = relationship("ChatSession", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id          = Column(String, primary_key=True, default=_uuid)
    user_id     = Column(String, nullable=False, default="default", index=True)
    filename    = Column(String, nullable=False)
    file_type   = Column(String, nullable=False)          # pdf | docx | csv | txt | url
    chunk_count = Column(Integer, default=0)
    indexed_at  = Column(DateTime, default=datetime.utcnow)
