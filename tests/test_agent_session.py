#!/usr/bin/env python3
"""
Tests for agent/agentSession.py - AgentSession and SessionManager with database persistence.

Run with: pytest tests/test_agent_session.py -v
"""

import asyncio

import pytest

from agent.agentSession import AgentSession, SessionManager
from db import close_async_db, close_db, init_async_db, init_db

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clean_db_state():
    """Clean up database state before and after each test."""
    close_db()
    close_async_db_sync()
    yield
    close_db()
    close_async_db_sync()


def close_async_db_sync():
    """Helper to close async db synchronously."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(close_async_db())


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_agent_session.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary database (sync)."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    close_db()
    init_db()
    yield
    close_db()


@pytest.fixture
def configured_async_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary database (async)."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)

    async def setup():
        await close_async_db()
        await init_async_db()

    asyncio.run(setup())
    yield

    async def teardown():
        await close_async_db()

    asyncio.run(teardown())


# ============================================================================
# AgentSession Tests
# ============================================================================


class TestAgentSession:
    """Tests for AgentSession class."""

    def test_create_session(self):
        """Test creating an agent session in memory."""
        session = AgentSession(
            session_id="test-123",
            original_query="Set up MLOps pipeline",
            project_path="/path/to/project",
            profile="default",
        )

        assert session.session_id == "test-123"
        assert session.original_query == "Set up MLOps pipeline"
        assert session.project_path == "/path/to/project"
        assert session.profile == "default"
        assert session.status == "active"
        assert session.db_id is None  # Not yet persisted

    def test_add_message(self):
        """Test adding messages to session."""
        session = AgentSession(
            session_id="test-msg",
            original_query="Test query",
        )

        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!", metadata={"tool": "greeting"})

        assert len(session.messages) == 2
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello"
        assert session.messages[1]["metadata"]["tool"] == "greeting"

    def test_add_tool_call(self):
        """Test recording tool calls."""
        session = AgentSession(
            session_id="test-tool",
            original_query="Test query",
        )

        session.add_tool_call(
            tool_name="create_hydra_config",
            args={"project_path": "/path"},
            result={"success": True, "config_path": "/path/config.yaml"},
        )

        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "tool"
        assert session.messages[0]["metadata"]["tool"] == "create_hydra_config"
        assert session.messages[0]["metadata"]["success"] is True

    def test_create_snapshot(self):
        """Test creating experiment snapshots."""
        session = AgentSession(
            session_id="test-snapshot",
            original_query="Test query",
        )

        session.add_message("user", "Test message")
        snapshot_id = session.create_snapshot(
            context_snapshot={"accuracy": 0.85, "stage": "training"},
            label="checkpoint-1",
        )

        assert snapshot_id == 0
        assert len(session.snapshots) == 1
        assert session.snapshots[0]["label"] == "checkpoint-1"
        assert session.snapshots[0]["context"]["accuracy"] == 0.85
        assert session.snapshots[0]["message_count"] == 1

    def test_get_latest_snapshot(self):
        """Test getting latest snapshot."""
        session = AgentSession(
            session_id="test-latest",
            original_query="Test query",
        )

        assert session.get_latest_snapshot() is None

        session.create_snapshot({"stage": "setup"}, label="first")
        session.create_snapshot({"stage": "training"}, label="second")

        latest = session.get_latest_snapshot()
        assert latest["label"] == "second"
        assert latest["id"] == 1

    def test_get_conversation_for_llm(self):
        """Test formatting conversation for LLM."""
        session = AgentSession(
            session_id="test-llm",
            original_query="Test query",
        )

        session.add_message("user", "Hello")
        session.add_tool_call("test_tool", {}, {"success": True, "data": "result"})
        session.add_message("assistant", "Done!")

        formatted = session.get_conversation_for_llm()

        assert len(formatted) == 3
        assert formatted[0]["role"] == "user"
        assert formatted[1]["role"] == "assistant"  # Tool call formatted as assistant
        assert "[Tool: test_tool]" in formatted[1]["content"]
        assert formatted[2]["role"] == "assistant"

    def test_mark_completed_success(self):
        """Test marking session as completed successfully."""
        session = AgentSession(
            session_id="test-complete",
            original_query="Test query",
        )

        session.mark_completed(success=True)

        assert session.status == "completed"
        assert session.completed_at is not None

    def test_mark_completed_failure(self):
        """Test marking session as failed."""
        session = AgentSession(
            session_id="test-failed",
            original_query="Test query",
        )

        session.mark_completed(success=False)

        assert session.status == "failed"
        assert session.completed_at is not None

    def test_to_dict(self):
        """Test serializing session to dict."""
        session = AgentSession(
            session_id="test-dict",
            original_query="Test query",
            project_path="/path",
            profile="test",
        )
        session.add_message("user", "Hello")

        data = session.to_dict()

        assert data["session_id"] == "test-dict"
        assert data["original_query"] == "Test query"
        assert data["project_path"] == "/path"
        assert data["profile"] == "test"
        assert data["status"] == "active"
        assert data["message_count"] == 1
        assert data["snapshot_count"] == 0

    def test_to_json_alias(self):
        """Test that to_json is an alias for to_dict."""
        session = AgentSession(
            session_id="test-json",
            original_query="Test query",
        )

        assert session.to_json() == session.to_dict()

    def test_get_experiment_history(self):
        """Test extracting experiment history from messages."""
        session = AgentSession(
            session_id="test-history",
            original_query="Test query",
        )

        # Add various messages
        session.add_message("user", "Start training")
        session.add_tool_call("log_mlflow_metrics", {"accuracy": 0.8}, {"success": True})
        session.add_tool_call("random_tool", {}, {"success": True})
        session.add_tool_call("check_accuracy_threshold", {}, {"success": True, "met": False})
        session.add_tool_call("suggest_improvements", {}, {"success": True})

        history = session.get_experiment_history()

        assert len(history) == 3
        assert history[0]["tool"] == "log_mlflow_metrics"
        assert history[1]["tool"] == "check_accuracy_threshold"
        assert history[2]["tool"] == "suggest_improvements"


class TestAgentSessionDatabasePersistence:
    """Tests for AgentSession database persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, configured_async_db):
        """Test saving and loading session from database."""
        # Create and save session
        session = AgentSession(
            session_id="test-save-load",
            original_query="Test MLOps pipeline",
            project_path="/test/path",
            profile="test-profile",
        )
        session.add_message("user", "Hello")
        session.create_snapshot({"stage": "setup"}, label="initial")

        await session.save()

        assert session.db_id is not None

        # Load session
        loaded = await AgentSession.load("test-save-load")

        assert loaded is not None
        assert loaded.session_id == "test-save-load"
        assert loaded.original_query == "Test MLOps pipeline"
        assert loaded.project_path == "/test/path"
        assert loaded.profile == "test-profile"
        assert loaded.status == "active"
        assert loaded.db_id == session.db_id

    @pytest.mark.asyncio
    async def test_save_updates_existing(self, configured_async_db):
        """Test that save updates existing session."""
        session = AgentSession(
            session_id="test-update",
            original_query="Initial query",
        )
        await session.save()
        original_db_id = session.db_id

        # Update and save again
        session.add_message("user", "New message")
        session.mark_completed(success=True)
        await session.save()

        # Should have same db_id
        assert session.db_id == original_db_id

        # Verify update persisted
        loaded = await AgentSession.load("test-update")
        assert loaded.status == "completed"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, configured_async_db):
        """Test loading non-existent session returns None."""
        loaded = await AgentSession.load("nonexistent-session")
        assert loaded is None


# ============================================================================
# SessionManager Tests
# ============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self, configured_async_db):
        """Test creating session through manager."""
        manager = SessionManager()

        session = await manager.create_session(
            session_id="mgr-test-1",
            query="Test query",
            project_path="/test/path",
            profile="test",
        )

        assert session.session_id == "mgr-test-1"
        assert session.db_id is not None
        assert "mgr-test-1" in manager._cache

    @pytest.mark.asyncio
    async def test_get_session_from_cache(self, configured_async_db):
        """Test getting session from cache."""
        manager = SessionManager()

        created = await manager.create_session(
            session_id="mgr-cache-test",
            query="Test query",
        )

        # Should get from cache
        retrieved = await manager.get_session("mgr-cache-test")
        assert retrieved is created

    @pytest.mark.asyncio
    async def test_get_session_from_database(self, configured_async_db):
        """Test getting session from database when not in cache."""
        manager1 = SessionManager()
        await manager1.create_session(
            session_id="mgr-db-test",
            query="Test query",
        )

        # Create new manager (empty cache)
        manager2 = SessionManager()
        assert "mgr-db-test" not in manager2._cache

        # Should load from database
        session = await manager2.get_session("mgr-db-test")
        assert session is not None
        assert session.session_id == "mgr-db-test"
        assert "mgr-db-test" in manager2._cache

    @pytest.mark.asyncio
    async def test_get_session_nonexistent(self, configured_async_db):
        """Test getting non-existent session returns None."""
        manager = SessionManager()
        session = await manager.get_session("nonexistent")
        assert session is None

    @pytest.mark.asyncio
    async def test_list_sessions(self, configured_async_db):
        """Test listing sessions."""
        manager = SessionManager()

        await manager.create_session("list-1", "Query 1")
        await manager.create_session("list-2", "Query 2")
        await manager.create_session("list-3", "Query 3")

        sessions = await manager.list_sessions()

        assert len(sessions) >= 3
        session_ids = [s["session_id"] for s in sessions]
        assert "list-1" in session_ids
        assert "list-2" in session_ids
        assert "list-3" in session_ids

    @pytest.mark.asyncio
    async def test_list_sessions_with_status_filter(self, configured_async_db):
        """Test listing sessions filtered by status."""
        manager = SessionManager()

        await manager.create_session("filter-active", "Query")
        session2 = await manager.create_session("filter-complete", "Query")
        session2.mark_completed(success=True)
        await session2.save()

        # Filter by active
        active_sessions = await manager.list_sessions(status="active")
        active_ids = [s["session_id"] for s in active_sessions]
        assert "filter-active" in active_ids

        # Filter by completed
        completed_sessions = await manager.list_sessions(status="completed")
        completed_ids = [s["session_id"] for s in completed_sessions]
        assert "filter-complete" in completed_ids

    @pytest.mark.asyncio
    async def test_update_session_status(self, configured_async_db):
        """Test updating session status."""
        manager = SessionManager()

        await manager.create_session("status-test", "Query")

        await manager.update_session_status("status-test", "completed", success=True)

        # Verify cache updated
        cached = manager._cache.get("status-test")
        assert cached.status == "completed"

        # Verify database updated
        loaded = await AgentSession.load("status-test")
        assert loaded.status == "completed"

    @pytest.mark.asyncio
    async def test_delete_session(self, configured_async_db):
        """Test deleting a session."""
        manager = SessionManager()

        await manager.create_session("delete-test", "Query")
        assert "delete-test" in manager._cache

        deleted = await manager.delete_session("delete-test")

        assert deleted is True
        assert "delete-test" not in manager._cache

        # Verify removed from database
        loaded = await AgentSession.load("delete-test")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self, configured_async_db):
        """Test deleting non-existent session."""
        manager = SessionManager()
        deleted = await manager.delete_session("nonexistent")
        assert deleted is False

    def test_clear_cache(self, configured_db):
        """Test clearing session cache."""
        manager = SessionManager()
        manager._cache["test-1"] = AgentSession("test-1", "Query 1")
        manager._cache["test-2"] = AgentSession("test-2", "Query 2")

        manager.clear_cache()

        assert len(manager._cache) == 0

    def test_get_session_sync(self, configured_db):
        """Test synchronous session retrieval from cache."""
        manager = SessionManager()
        session = AgentSession("sync-test", "Query")
        manager._cache["sync-test"] = session

        retrieved = manager.get_session_sync("sync-test")
        assert retrieved is session

        # Non-cached returns None
        assert manager.get_session_sync("not-cached") is None


# ============================================================================
# Backwards Compatibility Tests
# ============================================================================


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with synchronous code."""

    def test_session_can_be_created_without_db(self):
        """Test that sessions can be created without database setup."""
        session = AgentSession(
            session_id="no-db-test",
            original_query="Query without database",
        )

        assert session.session_id == "no-db-test"
        assert session.status == "active"
        assert session.db_id is None

    def test_session_manager_sync_methods(self):
        """Test SessionManager synchronous methods work without database."""
        manager = SessionManager()

        # Get from empty cache
        session = manager.get_session_sync("nonexistent")
        assert session is None

        # Create in cache manually
        manual_session = AgentSession("manual", "Query")
        manager._cache["manual"] = manual_session

        retrieved = manager.get_session_sync("manual")
        assert retrieved is manual_session
