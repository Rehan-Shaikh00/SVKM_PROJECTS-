"""Database engine, session management and declarative base.

SQLite by default (zero-setup for dev/pilot) and Postgres in production —
switch with DATABASE_URL. When the URL points at Postgres and pgvector is
installed, the vector store uses the same database (see app/kb/vectorstore.py).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from .config import REPO_ROOT, settings


class Base(DeclarativeBase):
    pass


def _abs_sqlite(url: str) -> str:
    """Pin a relative SQLite path to the repo root.

    `DATABASE_URL=sqlite:///data/runtime/nims_voice.db` is relative, and the app is
    documented to run from `backend/` (`make run` does `cd backend && uvicorn`).
    Left alone the engine would look for `backend/data/runtime/…` while
    `config.get_settings()` creates `data/runtime/…` at the repo root — so a fresh
    clone, or any boot after `make reset-data`, dies with "unable to open database
    file". Resolving here makes the working directory irrelevant.
    """
    prefix, _, path = url.partition("///")
    if not path or path.startswith(":memory:") or Path(path).is_absolute():
        return url
    return f"{prefix}///{REPO_ROOT / path}"


def _to_async_url(url: str) -> str:
    if url.startswith("sqlite+aiosqlite"):
        return _abs_sqlite(url)
    if url.startswith("sqlite://"):
        return _abs_sqlite(url.replace("sqlite://", "sqlite+aiosqlite://", 1))
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)
    return url


DATABASE_URL = _to_async_url(settings.database_url)

_engine_kwargs: dict[str, object] = {"echo": False, "future": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
else:
    _engine_kwargs.update(pool_size=10, max_overflow=20, pool_pre_ping=True)

engine: AsyncEngine = create_async_engine(DATABASE_URL, **_engine_kwargs)

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Create tables if they do not exist."""
    from . import models  # noqa: F401  (register mappers)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db() -> None:
    await engine.dispose()


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional context manager: commit on success, roll back on error."""
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency."""
    async with SessionLocal() as session:
        yield session


def is_postgres() -> bool:
    return DATABASE_URL.startswith("postgresql")
