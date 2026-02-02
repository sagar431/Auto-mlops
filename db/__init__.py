"""
Database Module for MLOps Agent.

Provides database connection, session management, and initialization.
Supports SQLite (development) and PostgreSQL (production) via DATABASE_URL.

Components:
- DatabaseConfig: Configuration from environment
- get_engine: Create SQLAlchemy engine
- get_session: Get database session (context manager)
- get_db: FastAPI dependency for database sessions
- init_db: Initialize database schema
"""

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlmodel import SQLModel

from db.models import AgentSession, ExperimentState, Step
from security.models import APIKey, User


class DatabaseConfig:
    """
    Database configuration loaded from environment variables.

    Attributes:
        database_url: Database connection URL (default: SQLite file)
        echo: Whether to echo SQL statements (default: False)
        pool_size: Connection pool size for PostgreSQL (default: 5)
        max_overflow: Max overflow connections (default: 10)
    """

    def __init__(self):
        self.database_url = os.environ.get("DATABASE_URL", "sqlite:///./mlops_agent.db")
        self.echo = os.environ.get("DATABASE_ECHO", "false").lower() == "true"
        self.pool_size = int(os.environ.get("DATABASE_POOL_SIZE", "5"))
        self.max_overflow = int(os.environ.get("DATABASE_MAX_OVERFLOW", "10"))

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_url.startswith("sqlite")

    @property
    def is_postgresql(self) -> bool:
        """Check if using PostgreSQL database."""
        return self.database_url.startswith("postgresql")


# Global configuration instance
_config: DatabaseConfig | None = None

# Global engine instance
_engine: Engine | None = None

# Global session factory
_SessionLocal: sessionmaker | None = None


def get_config() -> DatabaseConfig:
    """
    Get database configuration singleton.

    Returns:
        DatabaseConfig instance
    """
    global _config
    if _config is None:
        _config = DatabaseConfig()
    return _config


def get_engine() -> Engine:
    """
    Get or create SQLAlchemy engine singleton.

    Creates appropriate engine based on database type:
    - SQLite: Uses check_same_thread=False for FastAPI compatibility
    - PostgreSQL: Uses connection pooling

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine
    if _engine is None:
        config = get_config()

        if config.is_sqlite:
            # SQLite-specific settings
            _engine = create_engine(
                config.database_url,
                echo=config.echo,
                connect_args={"check_same_thread": False},
            )

            # Enable foreign keys for SQLite
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        else:
            # PostgreSQL or other databases with connection pooling
            _engine = create_engine(
                config.database_url,
                echo=config.echo,
                pool_size=config.pool_size,
                max_overflow=config.max_overflow,
                pool_pre_ping=True,
            )

    return _engine


def get_session_factory() -> sessionmaker:
    """
    Get or create session factory singleton.

    Returns:
        SQLAlchemy sessionmaker instance
    """
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            class_=Session,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.

    Yields a database session and handles commit/rollback automatically.

    Usage:
        with get_session() as session:
            user = session.get(User, user_id)
            session.add(new_item)

    Yields:
        Database session
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI endpoints:
        @app.get("/users/{user_id}")
        def get_user(user_id: int, db: Session = Depends(get_db)):
            return db.get(User, user_id)

    Yields:
        Database session
    """
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Initialize database schema.

    Creates all tables defined in SQLModel models.
    Safe to call multiple times (uses CREATE IF NOT EXISTS).
    """
    engine = get_engine()
    SQLModel.metadata.create_all(engine)


def reset_db() -> None:
    """
    Reset database by dropping and recreating all tables.

    WARNING: This deletes all data. Use only for testing.
    """
    engine = get_engine()
    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def close_db() -> None:
    """
    Close database connections and reset singletons.

    Call this during application shutdown or testing cleanup.
    """
    global _engine, _SessionLocal, _config

    if _engine is not None:
        _engine.dispose()
        _engine = None

    _SessionLocal = None
    _config = None


__all__ = [
    # Configuration
    "DatabaseConfig",
    "get_config",
    # Engine and sessions
    "get_engine",
    "get_session",
    "get_session_factory",
    "get_db",
    # Initialization
    "init_db",
    "reset_db",
    "close_db",
    # Re-export models for convenience
    "User",
    "APIKey",
    # Agent models
    "AgentSession",
    "Step",
    "ExperimentState",
]
