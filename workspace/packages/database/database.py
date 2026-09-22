"""Database connection and session management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from packages.config.settings import get_settings


class Database:
    """Database connection manager."""

    def __init__(self):
        self.settings = get_settings()
        self.engine = None
        self.session_factory = None

    def initialize(self):
        """Initialize database engine and session factory."""
        db_url = str(self.settings.database.url)

        # Convert postgresql:// to postgresql+asyncpg://
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

        self.engine = create_async_engine(
            db_url,
            echo=self.settings.database.echo,
            pool_size=self.settings.database.pool_size,
            max_overflow=self.settings.database.max_overflow,
            pool_pre_ping=True,
            poolclass=NullPool if self.settings.is_development() else None,
        )

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def close(self):
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session context manager."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized. Call initialize() first.")

        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


# Global database instance
db = Database()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI to get database session."""
    async with db.session() as session:
        yield session
