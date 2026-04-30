#!/usr/bin/env python3
"""
Tests for the database-backed SessionManager in api_server.py.

Run with: pytest tests/root_migrated/test_session_manager.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

from api_server import SessionManager

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def async_engine():
    """Create an async engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine):
    """Create an async session for testing."""
    async_session_maker = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def session_manager():
    """Create a fresh SessionManager instance."""
    return SessionManager()


# ============================================================================
# SessionManager Tests
# ============================================================================


class TestSessionManager:
    """Tests for SessionManager class."""

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, async_session):
        """Test creating a new session."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path="/test/path",
            threshold=0.9,
        )

        assert session_id is not None
        assert len(session_id) == 36  # UUID format

        # Verify session was created in database
        session = await session_manager.get_session(async_session, session_id)
        assert session is not None
        assert session["query"] == "Test query"
        assert session["project_path"] == "/test/path"
        assert session["accuracy_threshold"] == 0.9
        assert session["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_session_without_project_path(self, session_manager, async_session):
        """Test creating a session without project path."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Another query",
            project_path=None,
            threshold=0.85,
        )

        session = await session_manager.get_session(async_session, session_id)
        assert session is not None
        assert session["project_path"] is None

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, session_manager, async_session):
        """Test getting a non-existent session."""
        session = await session_manager.get_session(async_session, "non-existent-id")
        assert session is None

    @pytest.mark.asyncio
    async def test_update_session_status(self, session_manager, async_session):
        """Test updating session status."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path=None,
            threshold=0.85,
        )

        await session_manager.update_session(async_session, session_id, {"status": "running"})

        session = await session_manager.get_session(async_session, session_id)
        assert session["status"] == "running"

    @pytest.mark.asyncio
    async def test_update_session_multiple_fields(self, session_manager, async_session):
        """Test updating multiple session fields."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path=None,
            threshold=0.85,
        )

        await session_manager.update_session(
            async_session,
            session_id,
            {
                "status": "success",
                "current_phase": "completed",
                "steps_completed": 5,
                "steps_total": 5,
                "accuracy": 0.92,
            },
        )

        session = await session_manager.get_session(async_session, session_id)
        assert session["status"] == "success"
        assert session["current_phase"] == "completed"
        assert session["steps_completed"] == 5
        assert session["steps_total"] == 5
        assert session["accuracy"] == 0.92

    @pytest.mark.asyncio
    async def test_update_session_errors(self, session_manager, async_session):
        """Test updating session errors list."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path=None,
            threshold=0.85,
        )

        await session_manager.update_session(
            async_session,
            session_id,
            {"errors": ["Error 1", "Error 2"]},
        )

        session = await session_manager.get_session(async_session, session_id)
        assert session["errors"] == ["Error 1", "Error 2"]

    @pytest.mark.asyncio
    async def test_update_nonexistent_session(self, session_manager, async_session):
        """Test updating a non-existent session (should not raise)."""
        # Should not raise an error
        await session_manager.update_session(
            async_session, "non-existent-id", {"status": "running"}
        )

    @pytest.mark.asyncio
    async def test_broadcast_event_stores_in_db(self, session_manager, async_session):
        """Test that broadcast_event stores events in the database."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path=None,
            threshold=0.85,
        )

        await session_manager.broadcast_event(
            async_session,
            session_id,
            "test_event",
            {"key": "value"},
        )

        session = await session_manager.get_session(async_session, session_id)
        assert len(session["events"]) == 1
        assert session["events"][0]["type"] == "test_event"
        assert session["events"][0]["data"] == {"key": "value"}
        assert "timestamp" in session["events"][0]

    @pytest.mark.asyncio
    async def test_broadcast_multiple_events(self, session_manager, async_session):
        """Test broadcasting multiple events."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path=None,
            threshold=0.85,
        )

        await session_manager.broadcast_event(async_session, session_id, "event1", {"data": 1})
        await session_manager.broadcast_event(async_session, session_id, "event2", {"data": 2})

        session = await session_manager.get_session(async_session, session_id)
        assert len(session["events"]) == 2

    def test_add_websocket(self, session_manager):
        """Test adding a websocket to a session."""
        mock_ws = MagicMock()
        session_manager.add_websocket("session-1", mock_ws)

        assert "session-1" in session_manager.websockets
        assert mock_ws in session_manager.websockets["session-1"]

    def test_add_websocket_creates_list(self, session_manager):
        """Test that add_websocket creates a list if not present."""
        mock_ws = MagicMock()
        session_manager.add_websocket("new-session", mock_ws)

        assert "new-session" in session_manager.websockets
        assert len(session_manager.websockets["new-session"]) == 1

    def test_remove_websocket(self, session_manager):
        """Test removing a websocket from a session."""
        mock_ws = MagicMock()
        session_manager.websockets["session-1"] = [mock_ws]

        session_manager.remove_websocket("session-1", mock_ws)

        assert mock_ws not in session_manager.websockets["session-1"]

    def test_remove_websocket_not_in_list(self, session_manager):
        """Test removing a websocket that's not in the list."""
        mock_ws1 = MagicMock()
        mock_ws2 = MagicMock()
        session_manager.websockets["session-1"] = [mock_ws1]

        # Should not raise
        session_manager.remove_websocket("session-1", mock_ws2)
        assert mock_ws1 in session_manager.websockets["session-1"]

    def test_remove_websocket_nonexistent_session(self, session_manager):
        """Test removing a websocket from a non-existent session."""
        mock_ws = MagicMock()
        # Should not raise
        session_manager.remove_websocket("non-existent", mock_ws)


class TestSessionManagerSessionToDict:
    """Tests for _session_to_dict helper method."""

    @pytest.mark.asyncio
    async def test_session_to_dict_conversion(self, session_manager, async_session):
        """Test that session is correctly converted to dict format."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test query",
            project_path="/test/path",
            threshold=0.9,
        )

        session = await session_manager.get_session(async_session, session_id)

        # Verify all expected fields are present
        expected_fields = [
            "session_id",
            "status",
            "query",
            "project_path",
            "accuracy_threshold",
            "current_phase",
            "steps_completed",
            "steps_total",
            "accuracy",
            "started_at",
            "completed_at",
            "result",
            "errors",
            "events",
        ]
        for field in expected_fields:
            assert field in session, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_session_to_dict_default_values(self, session_manager, async_session):
        """Test default values in session dict."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test",
            project_path=None,
            threshold=0.85,
        )

        session = await session_manager.get_session(async_session, session_id)

        assert session["status"] == "pending"
        assert session["current_phase"] == "initializing"
        assert session["steps_completed"] == 0
        assert session["steps_total"] == 0
        assert session["accuracy"] is None
        assert session["completed_at"] is None
        assert session["result"] is None
        assert session["errors"] == []
        assert session["events"] == []


class TestSessionManagerWebSocketBroadcast:
    """Tests for WebSocket broadcasting functionality."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_websockets(self, session_manager, async_session):
        """Test that broadcast_event sends to all connected websockets."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test",
            project_path=None,
            threshold=0.85,
        )

        # Add mock websockets
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        session_manager.websockets[session_id] = [mock_ws1, mock_ws2]

        await session_manager.broadcast_event(
            async_session, session_id, "test_event", {"key": "value"}
        )

        # Verify both websockets received the event
        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()

        # Verify event format
        call_args = mock_ws1.send_json.call_args[0][0]
        assert call_args["type"] == "test_event"
        assert call_args["data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_websockets(self, session_manager, async_session):
        """Test that disconnected websockets are removed during broadcast."""
        session_id = await session_manager.create_session(
            db=async_session,
            query="Test",
            project_path=None,
            threshold=0.85,
        )

        # Add mock websockets - one that works and one that raises
        working_ws = AsyncMock()
        failing_ws = AsyncMock()
        failing_ws.send_json.side_effect = Exception("Connection closed")

        session_manager.websockets[session_id] = [working_ws, failing_ws]

        await session_manager.broadcast_event(async_session, session_id, "test_event", {})

        # Failing websocket should be removed
        assert working_ws in session_manager.websockets[session_id]
        assert failing_ws not in session_manager.websockets[session_id]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
