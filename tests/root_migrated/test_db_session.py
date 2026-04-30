#!/usr/bin/env python3
"""
Tests for the async database session module.

Run with: pytest tests/root_migrated/test_db_session.py -v
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from db.session import (
    AsyncDatabaseConfig,
    close_async_db,
    get_async_config,
    get_async_db,
    get_async_engine,
    get_async_session,
    get_async_session_factory,
    init_async_db,
    reset_async_db,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_async_mlops.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


# ============================================================================
# AsyncDatabaseConfig Tests
# ============================================================================


class TestAsyncDatabaseConfig:
    """Tests for AsyncDatabaseConfig class."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_default_config(self, monkeypatch):
        """Test default configuration values."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_ECHO", raising=False)
        monkeypatch.delenv("DATABASE_POOL_SIZE", raising=False)
        monkeypatch.delenv("DATABASE_MAX_OVERFLOW", raising=False)
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert "sqlite+aiosqlite" in config.database_url
        assert config.echo is False
        assert config.pool_size == 5
        assert config.max_overflow == 10

    @pytest.mark.asyncio
    async def test_sqlite_url_conversion(self, monkeypatch):
        """Test SQLite URL is converted to async driver."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.database_url == "sqlite+aiosqlite:///./test.db"

    @pytest.mark.asyncio
    async def test_postgresql_url_conversion(self, monkeypatch):
        """Test PostgreSQL URL is converted to asyncpg."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    @pytest.mark.asyncio
    async def test_postgres_url_conversion(self, monkeypatch):
        """Test Heroku-style postgres:// URL is converted to asyncpg."""
        monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost/db")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.database_url == "postgresql+asyncpg://user:pass@localhost/db"

    @pytest.mark.asyncio
    async def test_echo_enabled(self, monkeypatch):
        """Test enabling SQL echo."""
        monkeypatch.setenv("DATABASE_ECHO", "true")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.echo is True

    @pytest.mark.asyncio
    async def test_custom_pool_settings(self, monkeypatch):
        """Test custom pool settings."""
        monkeypatch.setenv("DATABASE_POOL_SIZE", "10")
        monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "20")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.pool_size == 10
        assert config.max_overflow == 20

    @pytest.mark.asyncio
    async def test_is_sqlite(self, monkeypatch):
        """Test SQLite detection."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.is_sqlite is True
        assert config.is_postgresql is False

    @pytest.mark.asyncio
    async def test_is_postgresql(self, monkeypatch):
        """Test PostgreSQL detection."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        await close_async_db()

        config = AsyncDatabaseConfig()
        assert config.is_postgresql is True
        assert config.is_sqlite is False


# ============================================================================
# Async Engine Tests
# ============================================================================


class TestAsyncEngine:
    """Tests for async database engine creation."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_get_async_engine_returns_engine(self, temp_db_url, monkeypatch):
        """Test that get_async_engine returns an AsyncEngine instance."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        engine = get_async_engine()
        assert isinstance(engine, AsyncEngine)

    @pytest.mark.asyncio
    async def test_get_async_engine_singleton(self, temp_db_url, monkeypatch):
        """Test that get_async_engine returns the same instance."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        engine1 = get_async_engine()
        engine2 = get_async_engine()
        assert engine1 is engine2

    @pytest.mark.asyncio
    async def test_async_engine_connects(self, temp_db_url, monkeypatch):
        """Test that async engine can connect to database."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            assert result.scalar() == 1


# ============================================================================
# Async Session Tests
# ============================================================================


class TestAsyncSessions:
    """Tests for async database session management."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_get_async_session_factory(self, temp_db_url, monkeypatch):
        """Test async session factory creation."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        factory = get_async_session_factory()
        assert factory is not None

    @pytest.mark.asyncio
    async def test_get_async_session_factory_singleton(self, temp_db_url, monkeypatch):
        """Test async session factory singleton."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        factory1 = get_async_session_factory()
        factory2 = get_async_session_factory()
        assert factory1 is factory2

    @pytest.mark.asyncio
    async def test_get_async_session_context_manager(self, temp_db_url, monkeypatch):
        """Test get_async_session as async context manager."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()
        await init_async_db()

        async with get_async_session() as session:
            assert isinstance(session, AsyncSession)
            # Verify session is usable
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_get_async_session_auto_commit(self, temp_db_url, monkeypatch):
        """Test that async session auto-commits on success."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()
        await init_async_db()

        # Create a table
        async with get_async_session() as session:
            await session.execute(
                text("CREATE TABLE IF NOT EXISTS test_commit (id INTEGER PRIMARY KEY, name TEXT)")
            )

        # Insert data
        async with get_async_session() as session:
            await session.execute(text("INSERT INTO test_commit (id, name) VALUES (1, 'test')"))

        # Verify data was persisted
        async with get_async_session() as session:
            result = await session.execute(text("SELECT name FROM test_commit WHERE id = 1"))
            assert result.scalar() == "test"

    @pytest.mark.asyncio
    async def test_get_async_session_rollback_on_error(self, temp_db_url, monkeypatch):
        """Test that async session rolls back on error."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()
        await init_async_db()

        # Create table
        async with get_async_session() as session:
            await session.execute(
                text("CREATE TABLE IF NOT EXISTS test_rollback (id INTEGER PRIMARY KEY, name TEXT)")
            )

        try:
            async with get_async_session() as session:
                await session.execute(
                    text("INSERT INTO test_rollback (id, name) VALUES (1, 'test')")
                )
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify data was not persisted
        async with get_async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM test_rollback"))
            assert result.scalar() == 0

    @pytest.mark.asyncio
    async def test_get_async_db_generator(self, temp_db_url, monkeypatch):
        """Test get_async_db as FastAPI dependency."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()
        await init_async_db()

        gen = get_async_db()
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)

        # Verify session is usable
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1

        # Clean up
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass


# ============================================================================
# Async Database Initialization Tests
# ============================================================================


class TestInitAsyncDb:
    """Tests for async database initialization."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_init_async_db_creates_tables(self, temp_db_url, monkeypatch):
        """Test that init_async_db creates tables."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        await init_async_db()

        # Verify tables exist by querying
        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = {row[0] for row in result}
            assert "users" in tables
            assert "api_keys" in tables

    @pytest.mark.asyncio
    async def test_init_async_db_idempotent(self, temp_db_url, monkeypatch):
        """Test that init_async_db can be called multiple times."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        await init_async_db()
        await init_async_db()  # Should not raise

        engine = get_async_engine()
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            )
            assert result.fetchone() is not None

    @pytest.mark.asyncio
    async def test_reset_async_db_clears_data(self, temp_db_url, monkeypatch):
        """Test that reset_async_db clears model data."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()
        await init_async_db()

        # Insert data into the users table (managed by SQLModel)
        async with get_async_session() as session:
            await session.execute(
                text(
                    "INSERT INTO users (username, email, hashed_password, is_active, is_admin, "
                    "created_at, updated_at) "
                    "VALUES ('testuser', 'test@example.com', 'hash', 1, 0, "
                    "'2024-01-01 00:00:00', '2024-01-01 00:00:00')"
                )
            )

        # Verify data exists
        async with get_async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            assert result.scalar() == 1

        # Reset database
        await reset_async_db()

        # Data should be cleared (table recreated)
        async with get_async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM users"))
            assert result.scalar() == 0


# ============================================================================
# Close/Cleanup Tests
# ============================================================================


class TestCloseAsyncDb:
    """Tests for async database cleanup."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_close_async_db_resets_singletons(self, temp_db_url, monkeypatch):
        """Test that close_async_db resets all singletons."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        # Create engine and session factory
        engine1 = get_async_engine()
        factory1 = get_async_session_factory()

        # Close
        await close_async_db()

        # Get new instances
        engine2 = get_async_engine()
        factory2 = get_async_session_factory()

        # Should be different instances
        assert engine1 is not engine2
        assert factory1 is not factory2


# ============================================================================
# Config Singleton Tests
# ============================================================================


class TestAsyncConfigSingleton:
    """Tests for async config singleton."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up database state before and after each test."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_config_singleton(self, temp_db_url, monkeypatch):
        """Test get_async_config returns singleton."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        await close_async_db()

        config1 = get_async_config()
        config2 = get_async_config()
        assert config1 is config2


# ============================================================================
# Export Tests
# ============================================================================


class TestAsyncExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all __all__ exports are available."""
        from db import session
        from db.session import __all__

        for name in __all__:
            assert hasattr(session, name), f"Missing export: {name}"

    def test_async_exports_from_db_module(self):
        """Test that async exports are available from db module."""
        from db import (
            AsyncDatabaseConfig,
            close_async_db,
            get_async_config,
            get_async_db,
            get_async_engine,
            get_async_session,
            get_async_session_factory,
            init_async_db,
            reset_async_db,
        )

        assert AsyncDatabaseConfig is not None
        assert get_async_config is not None
        assert get_async_engine is not None
        assert get_async_session is not None
        assert get_async_session_factory is not None
        assert get_async_db is not None
        assert init_async_db is not None
        assert reset_async_db is not None
        assert close_async_db is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
