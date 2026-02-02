#!/usr/bin/env python3
"""
Tests for Alembic database migrations.

Run with: pytest test_db_migrations.py -v
"""

import os
import subprocess

import pytest
from sqlalchemy import inspect, text
from sqlmodel import create_engine

from db import close_db

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
    return str(tmp_path / "test_migrations.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def migration_env(temp_db_url, monkeypatch):
    """Set up environment for running migrations."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    yield temp_db_url


def run_alembic_command(cmd, env=None):
    """Run an alembic command and return the result."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    result = subprocess.run(
        ["alembic"] + cmd,
        cwd="/home/ubuntu/Auto-mlops",
        capture_output=True,
        text=True,
        env=full_env,
    )
    return result


# ============================================================================
# Migration Upgrade Tests
# ============================================================================


class TestMigrationUpgrade:
    """Tests for migration upgrade operations."""

    def test_upgrade_to_head(self, migration_env):
        """Test upgrading to head creates all tables."""
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0, f"Alembic upgrade failed: {result.stderr}"

        # Verify tables exist
        engine = create_engine(migration_env, connect_args={"check_same_thread": False})
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "alembic_version" in tables
        assert "agent_sessions" in tables
        assert "users" in tables
        assert "api_keys" in tables
        assert "experiment_states" in tables
        assert "steps" in tables

        engine.dispose()

    def test_upgrade_creates_indexes(self, migration_env):
        """Test that upgrade creates necessary indexes."""
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        engine = create_engine(migration_env, connect_args={"check_same_thread": False})
        inspector = inspect(engine)

        # Check agent_sessions indexes
        agent_sessions_indexes = inspector.get_indexes("agent_sessions")
        index_names = {idx["name"] for idx in agent_sessions_indexes}
        assert "ix_agent_sessions_session_id" in index_names

        # Check users indexes
        users_indexes = inspector.get_indexes("users")
        user_index_names = {idx["name"] for idx in users_indexes}
        assert "ix_users_username" in user_index_names
        assert "ix_users_email" in user_index_names

        # Check api_keys indexes
        api_keys_indexes = inspector.get_indexes("api_keys")
        api_key_index_names = {idx["name"] for idx in api_keys_indexes}
        assert "ix_api_keys_key_hash" in api_key_index_names
        assert "ix_api_keys_user_id" in api_key_index_names

        engine.dispose()

    def test_upgrade_creates_foreign_keys(self, migration_env):
        """Test that upgrade creates foreign key constraints."""
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        engine = create_engine(migration_env, connect_args={"check_same_thread": False})
        inspector = inspect(engine)

        # Check api_keys foreign key to users
        api_keys_fks = inspector.get_foreign_keys("api_keys")
        assert len(api_keys_fks) >= 1
        fk_tables = {fk["referred_table"] for fk in api_keys_fks}
        assert "users" in fk_tables

        # Check steps foreign key to agent_sessions
        steps_fks = inspector.get_foreign_keys("steps")
        assert len(steps_fks) >= 1
        steps_fk_tables = {fk["referred_table"] for fk in steps_fks}
        assert "agent_sessions" in steps_fk_tables

        # Check experiment_states foreign key to agent_sessions
        exp_fks = inspector.get_foreign_keys("experiment_states")
        assert len(exp_fks) >= 1
        exp_fk_tables = {fk["referred_table"] for fk in exp_fks}
        assert "agent_sessions" in exp_fk_tables

        engine.dispose()


# ============================================================================
# Migration Downgrade Tests
# ============================================================================


class TestMigrationDowngrade:
    """Tests for migration downgrade operations."""

    def test_downgrade_removes_tables(self, migration_env):
        """Test that downgrade removes tables."""
        # First upgrade
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Then downgrade
        result = run_alembic_command(["downgrade", "base"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Verify tables are removed
        engine = create_engine(migration_env, connect_args={"check_same_thread": False})
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        assert "agent_sessions" not in tables
        assert "users" not in tables
        assert "api_keys" not in tables
        assert "experiment_states" not in tables
        assert "steps" not in tables

        engine.dispose()

    def test_downgrade_one_step(self, migration_env):
        """Test downgrading one migration step."""
        # Upgrade
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Downgrade one step
        result = run_alembic_command(["downgrade", "-1"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0


# ============================================================================
# Migration Current/History Tests
# ============================================================================


class TestMigrationInfo:
    """Tests for migration information commands."""

    def test_current_shows_version(self, migration_env):
        """Test that current command works after upgrade."""
        # Upgrade first
        run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})

        # Check current
        result = run_alembic_command(["current"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0
        assert "cf06fa948e58" in result.stdout  # Initial migration revision

    def test_history_shows_migrations(self):
        """Test that history command shows migrations."""
        result = run_alembic_command(["history"])
        assert result.returncode == 0
        assert "initial_schema" in result.stdout


# ============================================================================
# Alembic Configuration Tests
# ============================================================================


class TestAlembicConfig:
    """Tests for Alembic configuration."""

    def test_alembic_ini_exists(self):
        """Test that alembic.ini exists."""
        assert os.path.exists("/home/ubuntu/Auto-mlops/alembic.ini")

    def test_migrations_dir_exists(self):
        """Test that migrations directory structure exists."""
        assert os.path.isdir("/home/ubuntu/Auto-mlops/db/migrations")
        assert os.path.isdir("/home/ubuntu/Auto-mlops/db/migrations/versions")
        assert os.path.exists("/home/ubuntu/Auto-mlops/db/migrations/env.py")
        assert os.path.exists("/home/ubuntu/Auto-mlops/db/migrations/script.py.mako")

    def test_initial_migration_exists(self):
        """Test that initial migration file exists."""
        versions_dir = "/home/ubuntu/Auto-mlops/db/migrations/versions"
        migration_files = [f for f in os.listdir(versions_dir) if f.endswith(".py")]
        assert len(migration_files) >= 1

        # Check that the file contains expected content
        for f in migration_files:
            with open(os.path.join(versions_dir, f)) as fp:
                content = fp.read()
                if "initial_schema" in content:
                    assert "agent_sessions" in content
                    assert "users" in content
                    assert "api_keys" in content
                    assert "experiment_states" in content
                    assert "steps" in content
                    break
        else:
            pytest.fail("Initial migration not found")


# ============================================================================
# Schema Integrity Tests
# ============================================================================


class TestSchemaIntegrity:
    """Tests for schema integrity after migrations."""

    def test_can_insert_and_query_data(self, migration_env):
        """Test that migrated schema supports data operations."""
        run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})

        engine = create_engine(migration_env, connect_args={"check_same_thread": False})

        with engine.connect() as conn:
            # Insert a user
            conn.execute(text("""
                    INSERT INTO users (username, email, hashed_password, is_active, is_admin,
                                       created_at, updated_at)
                    VALUES ('testuser', 'test@example.com', 'hash123', 1, 0,
                            datetime('now'), datetime('now'))
                """))
            conn.commit()

            # Query the user
            result = conn.execute(
                text("SELECT username, email FROM users WHERE username = 'testuser'")
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "testuser"
            assert row[1] == "test@example.com"

        engine.dispose()

    def test_can_insert_agent_session_with_relations(self, migration_env):
        """Test inserting agent session with steps and experiment state."""
        run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})

        engine = create_engine(migration_env, connect_args={"check_same_thread": False})

        with engine.connect() as conn:
            # Insert an agent session
            conn.execute(text("""
                    INSERT INTO agent_sessions
                    (session_id, original_query, profile, status, created_at, updated_at)
                    VALUES ('sess-123', 'Test query', 'default', 'active',
                            datetime('now'), datetime('now'))
                """))

            # Get the session ID
            result = conn.execute(
                text("SELECT id FROM agent_sessions WHERE session_id = 'sess-123'")
            )
            session_id = result.fetchone()[0]

            # Insert a step
            conn.execute(
                text("""
                    INSERT INTO steps
                    (agent_session_id, step_index, description, step_type, status, created_at)
                    VALUES (:session_id, '0', 'Test step', 'ROOT', 'pending', datetime('now'))
                """),
                {"session_id": session_id},
            )

            # Insert experiment state
            conn.execute(
                text("""
                    INSERT INTO experiment_states
                    (agent_session_id, target_accuracy, best_accuracy, improvement_attempt,
                     max_improvement_attempts, stage, created_at, updated_at)
                    VALUES (:session_id, 0.85, 0.0, 0, 3, 'setup',
                            datetime('now'), datetime('now'))
                """),
                {"session_id": session_id},
            )
            conn.commit()

            # Verify the relations
            result = conn.execute(
                text("SELECT COUNT(*) FROM steps WHERE agent_session_id = :session_id"),
                {"session_id": session_id},
            )
            assert result.fetchone()[0] == 1

            result = conn.execute(
                text("SELECT COUNT(*) FROM experiment_states WHERE agent_session_id = :session_id"),
                {"session_id": session_id},
            )
            assert result.fetchone()[0] == 1

        engine.dispose()

    def test_foreign_key_enforcement(self, migration_env):
        """Test that foreign key constraints are enforced."""
        run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})

        engine = create_engine(migration_env, connect_args={"check_same_thread": False})

        # Enable foreign keys for SQLite
        with engine.connect() as conn:
            conn.execute(text("PRAGMA foreign_keys = ON"))
            conn.commit()

            # Try to insert a step with non-existent agent_session_id
            with pytest.raises(Exception):  # IntegrityError
                conn.execute(text("""
                        INSERT INTO steps
                        (agent_session_id, step_index, description, step_type, status, created_at)
                        VALUES (99999, '0', 'Invalid step', 'ROOT', 'pending', datetime('now'))
                    """))
                conn.commit()

        engine.dispose()


# ============================================================================
# Environment Variable Tests
# ============================================================================


class TestEnvironmentConfig:
    """Tests for environment-based configuration."""

    def test_uses_database_url_env_var(self, tmp_path):
        """Test that migrations use DATABASE_URL environment variable."""
        db_path = str(tmp_path / "env_test.db")
        db_url = f"sqlite:///{db_path}"

        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": db_url})
        assert result.returncode == 0

        # Verify database was created at specified path
        assert os.path.exists(db_path)


# ============================================================================
# Idempotency Tests
# ============================================================================


class TestMigrationIdempotency:
    """Tests for migration idempotency."""

    def test_upgrade_twice_is_safe(self, migration_env):
        """Test that running upgrade twice doesn't cause errors."""
        result1 = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result1.returncode == 0

        result2 = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result2.returncode == 0

    def test_upgrade_downgrade_upgrade(self, migration_env):
        """Test upgrade -> downgrade -> upgrade cycle."""
        # Upgrade
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Downgrade
        result = run_alembic_command(["downgrade", "base"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Upgrade again
        result = run_alembic_command(["upgrade", "head"], env={"DATABASE_URL": migration_env})
        assert result.returncode == 0

        # Verify tables exist
        engine = create_engine(migration_env, connect_args={"check_same_thread": False})
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "agent_sessions" in tables
        assert "users" in tables
        engine.dispose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
