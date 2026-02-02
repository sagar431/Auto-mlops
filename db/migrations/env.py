"""
Alembic Environment Configuration for MLOps Agent.

Supports both synchronous and asynchronous database migrations.
Reads DATABASE_URL from environment or falls back to SQLite.
"""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel, create_engine

# Import all models to ensure they are registered with SQLModel.metadata
from db.models import AgentSession, ExperimentState, Step  # noqa: F401
from security.models import APIKey, User  # noqa: F401

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Model metadata for autogenerate support
target_metadata = SQLModel.metadata


def get_database_url() -> str:
    """
    Get database URL from environment or alembic.ini.

    Priority:
    1. DATABASE_URL environment variable
    2. sqlalchemy.url from alembic.ini
    """
    url = os.environ.get("DATABASE_URL")
    if url:
        return url
    return config.get_main_option("sqlalchemy.url", "sqlite:///./mlops_agent.db")


def get_async_database_url() -> str:
    """
    Convert database URL to async driver format.

    Converts:
    - postgresql:// -> postgresql+asyncpg://
    - sqlite:// -> sqlite+aiosqlite://
    """
    url = get_database_url()
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://", 1)
    return url


def is_async_database() -> bool:
    """Check if using an async-compatible database driver."""
    url = get_database_url()
    return "asyncpg" in url or "aiosqlite" in url or url.startswith("postgresql://")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Creates an Engine and associates a connection with the context.
    Uses synchronous engine for simplicity in migrations.
    """
    url = get_database_url()

    # Use sync URL for migrations (convert async URLs back to sync)
    sync_url = url.replace("postgresql+asyncpg://", "postgresql://")
    sync_url = sync_url.replace("sqlite+aiosqlite://", "sqlite://")

    # Create engine with appropriate settings
    connect_args = {}
    if sync_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    connectable = create_engine(
        sync_url,
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=sync_url.startswith("sqlite"),  # Required for SQLite ALTER
        )

        with context.begin_transaction():
            context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with an existing connection."""
    url = get_database_url()
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=url.startswith("sqlite"),  # Required for SQLite ALTER
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Run migrations asynchronously.

    Creates an async Engine and associates a connection with the context.
    """
    url = get_async_database_url()

    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = url

    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    # Use synchronous migrations for reliability
    # Async migrations can be enabled by uncommenting below
    run_migrations_online()
    # asyncio.run(run_async_migrations())
