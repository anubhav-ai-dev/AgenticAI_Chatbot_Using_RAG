"""
Chat sessions, messages and feedback persistence.

Every method opens its own short-lived DB unit of work and returns plain
pydantic schemas, so callers (routes, chat pipeline, MCP) never hold ORM state.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import joinedload

from backend.core.config import MODES, Settings
from backend.core.database import Database
from backend.core.exceptions import SessionNotFound
from backend.models.api import (
    CitationSchema,
    MessageSchema,
    SessionCreate,
    SessionSchema,
    SessionSummary,
    SessionUpdate,
)
from backend.models.db import ChatSession, Message
from backend.utils.timeutil import utcnow

DEFAULT_TITLES = ("New Chat", "Untitled")


class SessionService:
    def __init__(self, settings: Settings, db: Database) -> None:
        self._s = settings
        self._db = db

    # ── sessions ─────────────────────────────────────────────────────────────

    def list_sessions(self, user_id: str) -> list[SessionSummary]:
        with self._db.session() as s:
            rows = (s.query(ChatSession).filter(ChatSession.user_id == user_id)
                    .order_by(ChatSession.updated_at.desc()).all())
            return [SessionSummary.model_validate(r) for r in rows]

    def create_session(self, payload: SessionCreate) -> SessionSchema:
        mode = payload.mode if payload.mode in MODES else "general"
        with self._db.session() as s:
            row = ChatSession(user_id=payload.user_id, title=payload.title,
                              model=payload.model or self._s.llm_default_model, mode=mode)
            s.add(row)
            s.flush()
            s.refresh(row)
            return SessionSchema.model_validate(row)

    def get_session(self, session_id: str) -> SessionSchema:
        with self._db.session() as s:
            row = (s.query(ChatSession).options(joinedload(ChatSession.messages))
                   .filter(ChatSession.id == session_id).first())
            if row is None:
                raise SessionNotFound(session_id)
            return SessionSchema.model_validate(row)

    def session_meta(self, session_id: str) -> Optional[dict[str, str]]:
        with self._db.session() as s:
            row = s.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None:
                return None
            return {"id": row.id, "user_id": row.user_id, "model": row.model, "mode": row.mode}

    def update_session(self, session_id: str, payload: SessionUpdate) -> SessionSchema:
        with self._db.session() as s:
            row = s.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None:
                raise SessionNotFound(session_id)
            if payload.title is not None:
                row.title = payload.title
            if payload.model is not None:
                row.model = payload.model
            if payload.mode is not None and payload.mode in MODES:
                row.mode = payload.mode
            s.flush()
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        with self._db.session() as s:
            row = s.query(ChatSession).filter(ChatSession.id == session_id).first()
            if row is None:
                raise SessionNotFound(session_id)
            s.delete(row)

    # ── messages ─────────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str,
                    citations: Optional[list[dict[str, Any] | CitationSchema]] = None,
                    meta: Optional[dict[str, Any]] = None) -> MessageSchema:
        cits = None
        if citations:
            cits = [c.model_dump() if isinstance(c, CitationSchema) else dict(c) for c in citations]
        with self._db.session() as s:
            session = s.query(ChatSession).filter(ChatSession.id == session_id).first()
            if session is None:
                raise SessionNotFound(session_id)
            msg = Message(session_id=session_id, role=role, content=content,
                          citations=cits, meta=meta)
            s.add(msg)
            session.updated_at = utcnow()
            if role == "user" and session.title in DEFAULT_TITLES:
                words = content.strip().split()
                session.title = " ".join(words[:6]) + ("..." if len(words) > 6 else "")
            s.flush()
            s.refresh(msg)
            return MessageSchema.model_validate(msg)

    def history(self, session_id: str, limit: int, exclude_last_user: bool = False) -> list[dict[str, str]]:
        """Most recent `limit` user/assistant messages, oldest first."""
        with self._db.session() as s:
            rows = (s.query(Message)
                    .filter(Message.session_id == session_id, Message.role.in_(("user", "assistant")))
                    .order_by(Message.created_at.desc(), Message.id.desc())
                    .limit(limit + (1 if exclude_last_user else 0)).all())
        rows.reverse()
        if exclude_last_user and rows and rows[-1].role == "user":
            rows = rows[:-1]
        return [{"role": r.role, "content": r.content} for r in rows][-limit:] if limit else []

    def pop_last_exchange(self, session_id: str) -> Optional[str]:
        """For regenerate: delete the trailing assistant reply; return the last user text."""
        with self._db.session() as s:
            rows = (s.query(Message).filter(Message.session_id == session_id)
                    .order_by(Message.created_at.desc(), Message.id.desc()).limit(2).all())
            if rows and rows[0].role == "assistant":
                s.delete(rows[0])
                rows = rows[1:]
            if rows and rows[0].role == "user":
                return rows[0].content
            return None

    def set_feedback(self, message_id: str, value: int) -> MessageSchema:
        with self._db.session() as s:
            msg = s.query(Message).filter(Message.id == message_id).first()
            if msg is None or msg.role != "assistant":
                raise SessionNotFound(message_id)
            msg.feedback = value or None
            s.flush()
            s.refresh(msg)
            return MessageSchema.model_validate(msg)
