"""
Agent Session for MLOps Agent - Session management with experiment snapshots.

This module provides in-memory session management with database persistence support.
Sessions track conversation history, experiment snapshots, and tool call history.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Optional

from db import AsyncSessionRepository, get_async_session


class AgentSession:
    """
    Manages a single agent session with experiment tracking.
    Handles persistence and recovery of session state via database.
    """

    def __init__(
        self,
        session_id: str,
        original_query: str,
        project_path: str | None = None,
        profile: str = "default",
        db_id: int | None = None,
    ):
        self.session_id = session_id
        self.original_query = original_query
        self.project_path = project_path
        self.profile = profile

        # Database primary key (set after save)
        self._db_id = db_id

        # Session metadata
        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at
        self.completed_at: str | None = None
        self.status = "active"  # active, completed, failed, paused

        # Conversation history for LLM context
        self.messages: list[dict[str, Any]] = []

        # Experiment snapshots (checkpoints)
        self.snapshots: list[dict[str, Any]] = []

    @property
    def db_id(self) -> int | None:
        """Get database primary key."""
        return self._db_id

    def add_message(self, role: str, content: str, metadata: dict | None = None):
        """Add a message to conversation history."""
        message = {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
        if metadata:
            message["metadata"] = metadata

        self.messages.append(message)
        self.updated_at = datetime.utcnow().isoformat()

    def add_tool_call(self, tool_name: str, args: dict, result: dict):
        """Record a tool call in session history."""
        self.add_message(
            role="tool",
            content=json.dumps(result, default=str)[:1000],
            metadata={"tool": tool_name, "args": args, "success": result.get("success", False)},
        )

    def create_snapshot(self, context_snapshot: dict, label: str = "auto"):
        """Create an experiment snapshot/checkpoint."""
        snapshot = {
            "id": len(self.snapshots),
            "label": label,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context_snapshot,
            "message_count": len(self.messages),
        }
        self.snapshots.append(snapshot)
        return snapshot["id"]

    def get_latest_snapshot(self) -> dict | None:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_conversation_for_llm(self, max_messages: int = 20) -> list[dict]:
        """Get recent conversation history formatted for LLM."""
        recent = (
            self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        )

        formatted = []
        for msg in recent:
            if msg["role"] == "tool":
                # Format tool calls as assistant messages
                formatted.append(
                    {
                        "role": "assistant",
                        "content": f"[Tool: {msg.get('metadata', {}).get('tool', 'unknown')}]\n{msg['content']}",
                    }
                )
            else:
                formatted.append({"role": msg["role"], "content": msg["content"]})

        return formatted

    def mark_completed(self, success: bool = True):
        """Mark session as completed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.utcnow().isoformat()
        self.updated_at = self.completed_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "project_path": self.project_path,
            "profile": self.profile,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "message_count": len(self.messages),
            "snapshot_count": len(self.snapshots),
            "messages": self.messages[-10:],  # Last 10 messages
            "latest_snapshot": self.get_latest_snapshot(),
        }

    # Alias for backwards compatibility
    def to_json(self) -> dict[str, Any]:
        """Serialize session to JSON-compatible dict (alias for to_dict)."""
        return self.to_dict()

    async def save(self):
        """Save session to database asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)

            if self._db_id is None:
                # Create new session in database
                db_session = await repo.create_session(
                    session_id=self.session_id,
                    original_query=self.original_query,
                    project_path=self.project_path,
                    profile=self.profile,
                )
                self._db_id = db_session.id
            else:
                # Update existing session
                db_session = await repo.get_session_by_id(self.session_id)
                if db_session:
                    db_session.status = self.status
                    db_session.updated_at = datetime.utcnow()
                    if self.completed_at:
                        db_session.completed_at = datetime.fromisoformat(self.completed_at)
                    # Store messages and snapshots in events field
                    db_session.events = {
                        "messages": self.messages,
                        "snapshots": self.snapshots,
                    }
                    await db.flush()

    def save_sync(self):
        """
        Synchronous save for backwards compatibility.

        Note: This creates a new event loop if needed. For async contexts,
        prefer using `await save()` directly.
        """
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, schedule the coroutine with error handling
            task = loop.create_task(self.save())
            task.add_done_callback(AgentSession._handle_save_result)
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(self.save())

    @staticmethod
    def _handle_save_result(task: asyncio.Task):
        """Log errors from background save tasks instead of silently dropping them."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            import logging

            logging.getLogger("agent.agentSession").warning(
                "Background session save failed: %s", exc
            )

    @classmethod
    async def load(cls, session_id: str) -> Optional["AgentSession"]:
        """Load session from database asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            db_session = await repo.get_session_by_id(session_id)

            if not db_session:
                return None

            session = cls(
                session_id=db_session.session_id,
                original_query=db_session.original_query,
                project_path=db_session.project_path,
                profile=db_session.profile,
                db_id=db_session.id,
            )
            session.status = db_session.status
            session.created_at = db_session.created_at.isoformat()
            session.updated_at = db_session.updated_at.isoformat()
            if db_session.completed_at:
                session.completed_at = db_session.completed_at.isoformat()

            # Restore messages and snapshots from events
            events = db_session.events or {}
            session.messages = events.get("messages", [])
            session.snapshots = events.get("snapshots", [])

            return session

    @classmethod
    def load_sync(cls, session_id: str) -> Optional["AgentSession"]:
        """
        Synchronous load for backwards compatibility.

        Note: This creates a new event loop if needed. For async contexts,
        prefer using `await load()` directly.
        """
        try:
            asyncio.get_running_loop()
            # Can't run async in sync context with running loop
            # Return None and let caller use async version
            return None
        except RuntimeError:
            # No running loop, run synchronously
            return asyncio.run(cls.load(session_id))

    def get_experiment_history(self) -> list[dict]:
        """Extract experiment-related events from session."""
        history = []

        for msg in self.messages:
            metadata = msg.get("metadata", {})
            if metadata.get("tool") in [
                "log_mlflow_metrics",
                "check_accuracy_threshold",
                "suggest_improvements",
                "update_hydra_config",
            ]:
                history.append(
                    {
                        "timestamp": msg["timestamp"],
                        "tool": metadata["tool"],
                        "args": metadata.get("args", {}),
                        "success": metadata.get("success", False),
                    }
                )

        return history


class SessionManager:
    """
    Manages multiple agent sessions with database persistence.

    Uses AsyncSessionRepository for database operations with in-memory caching
    for frequently accessed sessions.
    """

    def __init__(self):
        self._cache: dict[str, AgentSession] = {}

    async def create_session(
        self,
        session_id: str,
        query: str,
        project_path: str | None = None,
        profile: str = "default",
    ) -> AgentSession:
        """Create a new session and persist to database."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)

            # Create in database
            db_session = await repo.create_session(
                session_id=session_id,
                original_query=query,
                project_path=project_path,
                profile=profile,
            )

            # Create in-memory session
            session = AgentSession(
                session_id=session_id,
                original_query=query,
                project_path=project_path,
                profile=profile,
                db_id=db_session.id,
            )

            # Cache it
            self._cache[session_id] = session
            return session

    def create_session_sync(
        self,
        session_id: str,
        query: str,
        project_path: str | None = None,
        profile: str = "default",
    ) -> AgentSession:
        """
        Create a new session synchronously.

        Creates an in-memory session and schedules database persistence.
        For fully synchronous operation in non-async contexts.
        """
        session = AgentSession(
            session_id=session_id,
            original_query=query,
            project_path=project_path,
            profile=profile,
        )
        self._cache[session_id] = session

        # Schedule async save
        session.save_sync()
        return session

    async def get_session(self, session_id: str) -> AgentSession | None:
        """Get session by ID, loading from database if not cached."""
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]

        # Try loading from database
        session = await AgentSession.load(session_id)
        if session:
            self._cache[session_id] = session
        return session

    def get_session_sync(self, session_id: str) -> AgentSession | None:
        """
        Get session by ID synchronously.

        Checks cache first, returns None if not cached and can't load synchronously.
        """
        return self._cache.get(session_id)

    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List all sessions from database, optionally filtered by status."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            db_sessions = await repo.list_sessions(status=status, limit=limit, offset=offset)

            return [
                {
                    "session_id": s.session_id,
                    "query": s.original_query[:50] if s.original_query else "",
                    "status": s.status,
                    "created_at": s.created_at.isoformat(),
                }
                for s in db_sessions
            ]

    async def update_session_status(
        self,
        session_id: str,
        status: str,
        success: bool = True,
    ) -> AgentSession | None:
        """Update session status in database."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)

            if status in ("completed", "failed"):
                await repo.mark_session_completed(session_id, success=success)
            else:
                await repo.update_session_status(session_id, status)

            # Update cache if present
            if session_id in self._cache:
                self._cache[session_id].status = status
                if status in ("completed", "failed"):
                    self._cache[session_id].mark_completed(success)

            return self._cache.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from database and cache."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            deleted = await repo.delete_session(session_id)

            # Remove from cache
            self._cache.pop(session_id, None)
            return deleted

    def clear_cache(self):
        """Clear the in-memory session cache."""
        self._cache.clear()
