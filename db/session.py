"""
Async Database Session Module for MLOps Agent.

Provides async PostgreSQL connection using asyncpg with SQLAlchemy async support.
Supports both SQLite (development) and PostgreSQL (production) via DATABASE_URL.

Components:
- AsyncDatabaseConfig: Configuration from environment
- get_async_engine: Create async SQLAlchemy engine
- get_async_session: Async context manager for database sessions
- get_async_db: FastAPI dependency for async database sessions
- init_async_db: Initialize database schema asynchronously
"""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import SQLModel


class AsyncDatabaseConfig:
    """
    Async database configuration loaded from environment variables.

    Attributes:
        database_url: Database connection URL (default: SQLite file)
        echo: Whether to echo SQL statements (default: False)
        pool_size: Connection pool size for PostgreSQL (default: 5)
        max_overflow: Max overflow connections (default: 10)
    """

    def __init__(self):
        self.database_url = self._get_async_url()
        self.echo = os.environ.get("DATABASE_ECHO", "false").lower() == "true"
        self.pool_size = int(os.environ.get("DATABASE_POOL_SIZE", "5"))
        self.max_overflow = int(os.environ.get("DATABASE_MAX_OVERFLOW", "10"))

    def _get_async_url(self) -> str:
        """
        Get async-compatible database URL.

        Converts standard URLs to async driver URLs:
        - postgresql:// -> postgresql+asyncpg://
        - sqlite:// -> sqlite+aiosqlite://
        """
        url = os.environ.get("DATABASE_URL", "sqlite:///./mlops_agent.db")

        # Convert to async drivers
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            # Heroku-style URLs
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        elif url.startswith("sqlite://"):
            return url.replace("sqlite://", "sqlite+aiosqlite://", 1)

        return url

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return "sqlite" in self.database_url

    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return "postgresql" in self.database_url


# Global configuration instance
_async_config: AsyncDatabaseConfig | None = None

# Global async engine instance
_async_engine: AsyncEngine | None = None

# Global async session factory
_AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def get_async_config() -> AsyncDatabaseConfig:
    """
    Get async database configuration singleton.

    Returns:
        AsyncDatabaseConfig instance
    """
    global _async_config
    if _async_config is None:
        _async_config = AsyncDatabaseConfig()
    return _async_config


def get_async_engine() -> AsyncEngine:
    """
    Get or create async SQLAlchemy engine singleton.

    Creates appropriate engine based on database type:
    - SQLite: Uses aiosqlite driver
    - PostgreSQL: Uses asyncpg with connection pooling

    Returns:
        Async SQLAlchemy Engine instance
    """
    global _async_engine
    if _async_engine is None:
        config = get_async_config()

        if config.is_sqlite:
            # SQLite-specific settings for async
            _async_engine = create_async_engine(
                config.database_url,
                echo=config.echo,
                connect_args={"check_same_thread": False},
            )
        else:
            # PostgreSQL with asyncpg and connection pooling
            _async_engine = create_async_engine(
                config.database_url,
                echo=config.echo,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_pre_ping=True,
            )

    return _async_engine


def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create async session factory singleton.

    Returns:
        Async SQLAlchemy sessionmaker instance
    """
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        _AsyncSessionLocal = async_sessionmaker(
            bind=get_async_engine(),
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _AsyncSessionLocal


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Yields an async database session and handles commit/rollback automatically.

    Usage:
        async with get_async_session() as session:
            result = await session.execute(select(User))
            user = result.scalars().first()

    Yields:
        Async database session
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for async database sessions.

    Usage in FastAPI endpoints:
        @app.get("/users/{user_id}")
        async def get_user(user_id: int, db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(User).where(User.id == user_id))
            return result.scalars().first()

    Yields:
        Async database session
    """
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_async_db() -> None:
    """
    Initialize database schema asynchronously.

    Creates all tables defined in SQLModel models.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def reset_async_db() -> None:
    """
    Reset database by dropping and recreating all tables asynchronously.

    WARNING: This deletes all data. Use only for testing.
    """
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)


async def close_async_db() -> None:
    """
    Close async database connections and reset singletons.

    Call this during application shutdown or testing cleanup.
    """
    global _async_engine, _AsyncSessionLocal, _async_config

    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None

    _AsyncSessionLocal = None
    _async_config = None


__all__ = [
    # Configuration
    "AsyncDatabaseConfig",
    "get_async_config",
    # Engine and sessions
    "get_async_engine",
    "get_async_session",
    "get_async_session_factory",
    "get_async_db",
    # Initialization
    "init_async_db",
    "reset_async_db",
    "close_async_db",
]
