"""
SQLAlchemy engine/session management for chat sessions, messages and documents.

`Base` is the declarative base for ORM models; `Database` owns the engine and is
created once by the ServiceContainer (no module-level engine).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        is_sqlite = url.startswith("sqlite")
        self.engine: Engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if is_sqlite else {},
            pool_pre_ping=not is_sqlite,
            echo=False,
        )
        if is_sqlite:
            event.listen(self.engine, "connect", self._sqlite_pragmas)
        # expire_on_commit=False keeps ORM objects readable after the session closes
        self._factory = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False
        )

    @staticmethod
    def _sqlite_pragmas(dbapi_conn, _record) -> None:
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Short-lived unit of work: commits on success, rolls back on error."""
        db = self._factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def init(self) -> None:
        """Create tables and add any columns introduced after the DB was created."""
        from backend.models import db as _models  # noqa: F401 — registers ORM tables

        Base.metadata.create_all(bind=self.engine)
        self._ensure_columns()

    def _ensure_columns(self) -> None:
        """Idempotent additive migration (nullable columns only)."""
        insp = inspect(self.engine)
        with self.engine.begin() as conn:
            for table in Base.metadata.sorted_tables:
                if not insp.has_table(table.name):
                    continue
                existing = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name in existing or not col.nullable:
                        continue
                    ddl_type = col.type.compile(dialect=self.engine.dialect)
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}'))
                    logger.info("Migrated: added %s.%s", table.name, col.name)

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def dispose(self) -> None:
        self.engine.dispose()
