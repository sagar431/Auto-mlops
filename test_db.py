#!/usr/bin/env python3
"""
Tests for the database module.

Run with: pytest test_db.py -v
"""


import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from db import (
    APIKey,
    DatabaseConfig,
    User,
    close_db,
    get_config,
    get_db,
    get_engine,
    get_session,
    get_session_factory,
    init_db,
    reset_db,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_db_state():
    """Clean up database state before and after each test."""
    close_db()
    yield
    close_db()


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_mlops.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary database."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    close_db()  # Reset any cached config
    init_db()
    yield
    close_db()


# ============================================================================
# DatabaseConfig Tests
# ============================================================================


class TestDatabaseConfig:
    """Tests for DatabaseConfig class."""

    def test_default_config(self, monkeypatch):
        """Test default configuration values."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DATABASE_ECHO", raising=False)
        monkeypatch.delenv("DATABASE_POOL_SIZE", raising=False)
        monkeypatch.delenv("DATABASE_MAX_OVERFLOW", raising=False)
        close_db()

        config = DatabaseConfig()
        assert config.database_url == "sqlite:///./mlops_agent.db"
        assert config.echo is False
        assert config.pool_size == 5
        assert config.max_overflow == 10

    def test_custom_database_url(self, temp_db_url, monkeypatch):
        """Test custom database URL from environment."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        config = DatabaseConfig()
        assert config.database_url == temp_db_url

    def test_echo_enabled(self, monkeypatch):
        """Test enabling SQL echo."""
        monkeypatch.setenv("DATABASE_ECHO", "true")
        close_db()

        config = DatabaseConfig()
        assert config.echo is True

    def test_echo_disabled(self, monkeypatch):
        """Test disabling SQL echo."""
        monkeypatch.setenv("DATABASE_ECHO", "false")
        close_db()

        config = DatabaseConfig()
        assert config.echo is False

    def test_custom_pool_settings(self, monkeypatch):
        """Test custom pool settings."""
        monkeypatch.setenv("DATABASE_POOL_SIZE", "10")
        monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "20")
        close_db()

        config = DatabaseConfig()
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_is_sqlite(self, monkeypatch):
        """Test SQLite detection."""
        monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
        close_db()

        config = DatabaseConfig()
        assert config.is_sqlite is True
        assert config.is_postgresql is False

    def test_is_postgresql(self, monkeypatch):
        """Test PostgreSQL detection."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        close_db()

        config = DatabaseConfig()
        assert config.is_postgresql is True
        assert config.is_sqlite is False


# ============================================================================
# Engine Tests
# ============================================================================


class TestEngine:
    """Tests for database engine creation."""

    def test_get_engine_returns_engine(self, temp_db_url, monkeypatch):
        """Test that get_engine returns an Engine instance."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        engine = get_engine()
        assert isinstance(engine, Engine)

    def test_get_engine_singleton(self, temp_db_url, monkeypatch):
        """Test that get_engine returns the same instance."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        engine1 = get_engine()
        engine2 = get_engine()
        assert engine1 is engine2

    def test_engine_connects(self, temp_db_url, monkeypatch):
        """Test that engine can connect to database."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1


# ============================================================================
# Session Tests
# ============================================================================


class TestSessions:
    """Tests for database session management."""

    def test_get_session_factory(self, temp_db_url, monkeypatch):
        """Test session factory creation."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        factory = get_session_factory()
        assert factory is not None

    def test_get_session_factory_singleton(self, temp_db_url, monkeypatch):
        """Test session factory singleton."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        factory1 = get_session_factory()
        factory2 = get_session_factory()
        assert factory1 is factory2

    def test_get_session_context_manager(self, configured_db):
        """Test get_session as context manager."""
        with get_session() as session:
            assert isinstance(session, Session)
            # Verify session is usable
            result = session.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_get_session_auto_commit(self, configured_db):
        """Test that session auto-commits on success."""
        # Create a user
        with get_session() as session:
            user = User(
                username="testuser",
                email="test@example.com",
                hashed_password=User.hash_password("password"),
            )
            session.add(user)

        # Verify user was persisted
        with get_session() as session:
            users = session.query(User).filter(User.username == "testuser").all()
            assert len(users) == 1

    def test_get_session_rollback_on_error(self, configured_db):
        """Test that session rolls back on error."""
        try:
            with get_session() as session:
                user = User(
                    username="testuser",
                    email="test@example.com",
                    hashed_password="hashed",
                )
                session.add(user)
                session.flush()  # Force the insert
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify user was not persisted
        with get_session() as session:
            users = session.query(User).all()
            assert len(users) == 0

    def test_get_db_generator(self, configured_db):
        """Test get_db as FastAPI dependency."""
        gen = get_db()
        session = next(gen)
        assert isinstance(session, Session)

        # Verify session is usable
        result = session.execute(text("SELECT 1"))
        assert result.scalar() == 1

        # Clean up
        try:
            next(gen)
        except StopIteration:
            pass


# ============================================================================
# Database Initialization Tests
# ============================================================================


class TestInitDb:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self, temp_db_url, monkeypatch):
        """Test that init_db creates tables."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        init_db()

        # Verify tables exist
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = {row[0] for row in result}
            assert "users" in tables
            assert "api_keys" in tables

    def test_init_db_idempotent(self, temp_db_url, monkeypatch):
        """Test that init_db can be called multiple times."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        init_db()
        init_db()  # Should not raise

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
            )
            assert result.fetchone() is not None

    def test_reset_db_clears_data(self, configured_db):
        """Test that reset_db clears all data."""
        # Add some data
        with get_session() as session:
            user = User(
                username="testuser",
                email="test@example.com",
                hashed_password="hashed",
            )
            session.add(user)

        # Verify data exists
        with get_session() as session:
            assert session.query(User).count() == 1

        # Reset database
        reset_db()

        # Verify data is cleared
        with get_session() as session:
            assert session.query(User).count() == 0


# ============================================================================
# Close/Cleanup Tests
# ============================================================================


class TestCloseDb:
    """Tests for database cleanup."""

    def test_close_db_resets_singletons(self, temp_db_url, monkeypatch):
        """Test that close_db resets all singletons."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        # Create engine and session factory
        engine1 = get_engine()
        factory1 = get_session_factory()

        # Close
        close_db()

        # Get new instances
        engine2 = get_engine()
        factory2 = get_session_factory()

        # Should be different instances
        assert engine1 is not engine2
        assert factory1 is not factory2


# ============================================================================
# Integration Tests
# ============================================================================


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_create_and_query_user(self, configured_db):
        """Test creating and querying a user."""
        # Create user
        with get_session() as session:
            user = User(
                username="integrationuser",
                email="integration@example.com",
                hashed_password=User.hash_password("password123"),
                is_admin=True,
            )
            session.add(user)

        # Query user
        with get_session() as session:
            user = session.query(User).filter(User.username == "integrationuser").first()
            assert user is not None
            assert user.email == "integration@example.com"
            assert user.is_admin is True
            assert user.verify_password("password123")

    def test_create_user_with_api_key(self, configured_db):
        """Test creating a user with an API key."""
        # Create user
        with get_session() as session:
            user = User(
                username="keyuser",
                email="keyuser@example.com",
                hashed_password=User.hash_password("password"),
            )
            session.add(user)
            session.flush()  # Get user ID

            # Create API key for user
            api_key = APIKey(
                key_hash=APIKey.hash_key("test_api_key"),
                name="Test Key",
                user_id=user.id,
            )
            session.add(api_key)

        # Query user and API key
        with get_session() as session:
            user = session.query(User).filter(User.username == "keyuser").first()
            assert user is not None

            api_key = session.query(APIKey).filter(APIKey.user_id == user.id).first()
            assert api_key is not None
            assert api_key.name == "Test Key"

    def test_foreign_key_constraint(self, configured_db):
        """Test that foreign key constraints are enforced."""
        # Try to create API key with non-existent user
        with pytest.raises(Exception):  # IntegrityError
            with get_session() as session:
                api_key = APIKey(
                    key_hash=APIKey.hash_key("orphan_key"),
                    name="Orphan Key",
                    user_id=99999,  # Non-existent user
                )
                session.add(api_key)
                session.flush()

    def test_unique_constraints(self, configured_db):
        """Test unique constraints on username and email."""
        # Create first user
        with get_session() as session:
            user1 = User(
                username="unique",
                email="unique@example.com",
                hashed_password="hashed",
            )
            session.add(user1)

        # Try to create user with same username
        with pytest.raises(Exception):  # IntegrityError
            with get_session() as session:
                user2 = User(
                    username="unique",  # Duplicate
                    email="different@example.com",
                    hashed_password="hashed",
                )
                session.add(user2)
                session.flush()

    def test_config_singleton(self, temp_db_url, monkeypatch):
        """Test get_config returns singleton."""
        monkeypatch.setenv("DATABASE_URL", temp_db_url)
        close_db()

        config1 = get_config()
        config2 = get_config()
        assert config1 is config2


# ============================================================================
# Export Tests
# ============================================================================


class TestExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all __all__ exports are available."""
        import db
        from db import __all__

        for name in __all__:
            assert hasattr(db, name), f"Missing export: {name}"

    def test_model_reexports(self):
        """Test that models are re-exported."""
        from db import APIKey, User

        assert User is not None
        assert APIKey is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
