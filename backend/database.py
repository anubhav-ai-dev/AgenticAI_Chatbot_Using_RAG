"""
SQLAlchemy database engine, session factory, and Base.
All ORM models import Base from here.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.config import DATABASE_URL


class Base(DeclarativeBase):
    pass


# connect_args only needed for SQLite (allows multi-thread access)
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency that yields a DB session and closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once at startup."""
    from backend.models import db as _  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
