#!/usr/bin/env python3
"""
Tests for the database repositories module.

Run with: pytest test_db_repositories.py -v
"""

import pytest

from db import (
    close_db,
    get_session,
    init_db,
)
from db.repositories import AsyncSessionRepository, SessionRepository
from db.session import close_async_db, get_async_session, init_async_db

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
    return str(tmp_path / "test_repos.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary database."""
    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    close_db()
    init_db()
    yield
    close_db()


# ============================================================================
# SessionRepository - AgentSession Tests
# ============================================================================


class TestSessionRepositoryAgentSession:
    """Tests for SessionRepository AgentSession CRUD operations."""

    def test_create_session(self, configured_db):
        """Test creating an agent session through repository."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="repo-test-session",
                original_query="Set up MLOps pipeline",
                project_path="/path/to/project",
                profile="default",
            )
            assert session.id is not None
            assert session.session_id == "repo-test-session"
            assert session.status == "active"
            assert session.original_query == "Set up MLOps pipeline"

    def test_get_session_by_id(self, configured_db):
        """Test getting session by session_id."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="get-test-session",
                original_query="Test query",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            found = repo.get_session_by_id("get-test-session")
            assert found is not None
            assert found.original_query == "Test query"

    def test_get_session_by_id_not_found(self, configured_db):
        """Test getting non-existent session returns None."""
        with get_session() as db:
            repo = SessionRepository(db)
            found = repo.get_session_by_id("nonexistent-session")
            assert found is None

    def test_get_session_by_pk(self, configured_db):
        """Test getting session by primary key."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="pk-test-session",
                original_query="Test",
            )
            pk = session.id

        with get_session() as db:
            repo = SessionRepository(db)
            found = repo.get_session_by_pk(pk)
            assert found is not None
            assert found.session_id == "pk-test-session"

    def test_get_session_with_relations(self, configured_db):
        """Test getting session with eager-loaded relations."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="relations-test-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Init step",
                step_type="CODE",
            )
            repo.create_experiment_state(
                agent_session_id=session.id,
                experiment_name="test-exp",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            found = repo.get_session_with_relations("relations-test-session")
            assert found is not None
            assert len(found.steps) == 1
            assert found.experiment_state is not None
            assert found.experiment_state.experiment_name == "test-exp"

    def test_list_sessions(self, configured_db):
        """Test listing sessions."""
        with get_session() as db:
            repo = SessionRepository(db)
            for i in range(5):
                repo.create_session(
                    session_id=f"list-session-{i}",
                    original_query=f"Query {i}",
                )

        with get_session() as db:
            repo = SessionRepository(db)
            sessions = repo.list_sessions()
            assert len(sessions) == 5

    def test_list_sessions_with_status_filter(self, configured_db):
        """Test listing sessions with status filter."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="active-session",
                original_query="Query",
            )
            repo.create_session(
                session_id="completed-session",
                original_query="Query",
            )
            repo.mark_session_completed("completed-session", success=True)

        with get_session() as db:
            repo = SessionRepository(db)
            active = repo.list_sessions(status="active")
            completed = repo.list_sessions(status="completed")
            assert len(active) == 1
            assert len(completed) == 1
            assert active[0].session_id == "active-session"
            assert completed[0].session_id == "completed-session"

    def test_list_sessions_with_pagination(self, configured_db):
        """Test listing sessions with limit and offset."""
        with get_session() as db:
            repo = SessionRepository(db)
            for i in range(10):
                repo.create_session(
                    session_id=f"page-session-{i}",
                    original_query=f"Query {i}",
                )

        with get_session() as db:
            repo = SessionRepository(db)
            page1 = repo.list_sessions(limit=3, offset=0)
            page2 = repo.list_sessions(limit=3, offset=3)
            assert len(page1) == 3
            assert len(page2) == 3
            assert page1[0].session_id != page2[0].session_id

    def test_update_session_status(self, configured_db):
        """Test updating session status."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="status-test-session",
                original_query="Test",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            updated = repo.update_session_status("status-test-session", "paused")
            assert updated is not None
            assert updated.status == "paused"

    def test_mark_session_completed_success(self, configured_db):
        """Test marking session as completed successfully."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="complete-success-session",
                original_query="Test",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            updated = repo.mark_session_completed("complete-success-session", success=True)
            assert updated is not None
            assert updated.status == "completed"
            assert updated.completed_at is not None

    def test_mark_session_completed_failure(self, configured_db):
        """Test marking session as failed."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="complete-fail-session",
                original_query="Test",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            updated = repo.mark_session_completed("complete-fail-session", success=False)
            assert updated is not None
            assert updated.status == "failed"
            assert updated.completed_at is not None

    def test_delete_session(self, configured_db):
        """Test deleting a session."""
        with get_session() as db:
            repo = SessionRepository(db)
            repo.create_session(
                session_id="delete-test-session",
                original_query="Test",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            deleted = repo.delete_session("delete-test-session")
            assert deleted is True

        with get_session() as db:
            repo = SessionRepository(db)
            found = repo.get_session_by_id("delete-test-session")
            assert found is None

    def test_delete_session_not_found(self, configured_db):
        """Test deleting non-existent session returns False."""
        with get_session() as db:
            repo = SessionRepository(db)
            deleted = repo.delete_session("nonexistent-session")
            assert deleted is False


# ============================================================================
# SessionRepository - Step Tests
# ============================================================================


class TestSessionRepositoryStep:
    """Tests for SessionRepository Step CRUD operations."""

    def test_create_step(self, configured_db):
        """Test creating a step through repository."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-create-session",
                original_query="Test",
            )
            step = repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Initialize MLflow",
                step_type="CODE",
                tool="init_mlflow_experiment",
                args={"experiment_name": "test"},
            )
            assert step.id is not None
            assert step.status == "pending"
            assert step.tool == "init_mlflow_experiment"

    def test_create_step_with_from_step(self, configured_db):
        """Test creating a step with parent reference."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-parent-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="First step",
                step_type="CODE",
            )
            step2 = repo.create_step(
                agent_session_id=session.id,
                step_index="1",
                description="Second step",
                step_type="CODE",
                from_step="0",
            )
            assert step2.from_step == "0"

    def test_get_step(self, configured_db):
        """Test getting a step by session and index."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-get-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="test-idx",
                description="Test step",
                step_type="CODE",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("step-get-session")
            step = repo.get_step(session.id, "test-idx")
            assert step is not None
            assert step.description == "Test step"

    def test_get_steps_for_session(self, configured_db):
        """Test getting all steps for a session."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="steps-list-session",
                original_query="Test",
            )
            for i in range(3):
                repo.create_step(
                    agent_session_id=session.id,
                    step_index=str(i),
                    description=f"Step {i}",
                    step_type="CODE",
                )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("steps-list-session")
            steps = repo.get_steps_for_session(session.id)
            assert len(steps) == 3

    def test_update_step_status(self, configured_db):
        """Test updating step status."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-status-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Test",
                step_type="CODE",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("step-status-session")
            updated = repo.update_step_status(
                session.id,
                "0",
                "completed",
                result={"success": True},
            )
            assert updated is not None
            assert updated.status == "completed"
            assert updated.result == {"success": True}
            assert updated.completed_at is not None

    def test_mark_step_completed(self, configured_db):
        """Test marking step as completed."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-complete-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Test",
                step_type="CODE",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("step-complete-session")
            result = {"accuracy": 0.95}
            updated = repo.mark_step_completed(session.id, "0", result)
            assert updated is not None
            assert updated.status == "completed"
            assert updated.result["accuracy"] == 0.95

    def test_mark_step_failed(self, configured_db):
        """Test marking step as failed."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="step-fail-session",
                original_query="Test",
            )
            repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Test",
                step_type="CODE",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("step-fail-session")
            updated = repo.mark_step_failed(session.id, "0", "Connection timeout")
            assert updated is not None
            assert updated.status == "failed"
            assert updated.error == "Connection timeout"


# ============================================================================
# SessionRepository - ExperimentState Tests
# ============================================================================


class TestSessionRepositoryExperimentState:
    """Tests for SessionRepository ExperimentState CRUD operations."""

    def test_create_experiment_state(self, configured_db):
        """Test creating experiment state through repository."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-create-session",
                original_query="Train model",
            )
            exp_state = repo.create_experiment_state(
                agent_session_id=session.id,
                experiment_name="cat-dog-classifier",
                target_accuracy=0.90,
                max_improvement_attempts=5,
            )
            assert exp_state.id is not None
            assert exp_state.experiment_name == "cat-dog-classifier"
            assert exp_state.target_accuracy == 0.90
            assert exp_state.max_improvement_attempts == 5

    def test_get_experiment_state(self, configured_db):
        """Test getting experiment state for session."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-get-session",
                original_query="Test",
            )
            repo.create_experiment_state(
                agent_session_id=session.id,
                experiment_name="test-exp",
            )

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-get-session")
            state = repo.get_experiment_state(session.id)
            assert state is not None
            assert state.experiment_name == "test-exp"

    def test_update_experiment_metrics(self, configured_db):
        """Test updating experiment metrics."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-metrics-session",
                original_query="Test",
            )
            repo.create_experiment_state(agent_session_id=session.id)

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-metrics-session")
            updated = repo.update_experiment_metrics(session.id, 0.85, loss=0.12)
            assert updated is not None
            assert updated.current_accuracy == 0.85
            assert updated.current_loss == 0.12
            assert updated.best_accuracy == 0.85

    def test_update_experiment_config(self, configured_db):
        """Test updating experiment configuration."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-config-session",
                original_query="Test",
            )
            repo.create_experiment_state(agent_session_id=session.id)

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-config-session")
            config = {"learning_rate": 0.001, "batch_size": 32}
            updated = repo.update_experiment_config(session.id, config)
            assert updated is not None
            assert updated.current_config["learning_rate"] == 0.001

    def test_update_experiment_stage(self, configured_db):
        """Test updating experiment pipeline stage."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-stage-session",
                original_query="Test",
            )
            repo.create_experiment_state(agent_session_id=session.id)

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-stage-session")
            updated = repo.update_experiment_stage(session.id, "training")
            assert updated is not None
            assert updated.stage == "training"

    def test_record_improvement_attempt(self, configured_db):
        """Test recording an improvement attempt."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-improve-session",
                original_query="Test",
            )
            repo.create_experiment_state(agent_session_id=session.id)
            session_obj = repo.get_session_by_id("exp-improve-session")
            repo.update_experiment_metrics(session_obj.id, 0.75)

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-improve-session")
            config_changes = {"learning_rate": 0.0001}
            updated = repo.record_improvement_attempt(session.id, config_changes, 0.82)
            assert updated is not None
            assert updated.improvement_attempt == 1
            assert len(updated.improvement_history) == 1
            assert updated.current_accuracy == 0.82

    def test_add_artifact(self, configured_db):
        """Test adding an artifact to experiment state."""
        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.create_session(
                session_id="exp-artifact-session",
                original_query="Test",
            )
            repo.create_experiment_state(agent_session_id=session.id)

        with get_session() as db:
            repo = SessionRepository(db)
            session = repo.get_session_by_id("exp-artifact-session")
            updated = repo.add_artifact(session.id, "model.pkl")
            assert updated is not None
            assert "model.pkl" in updated.artifacts_created


# ============================================================================
# AsyncSessionRepository Tests
# ============================================================================


@pytest.fixture
def async_configured_db(temp_db_url, monkeypatch):
    """Configure and initialize a temporary async database."""
    import asyncio

    monkeypatch.setenv("DATABASE_URL", temp_db_url)
    asyncio.get_event_loop().run_until_complete(close_async_db())
    asyncio.get_event_loop().run_until_complete(init_async_db())
    yield
    asyncio.get_event_loop().run_until_complete(close_async_db())


class TestAsyncSessionRepositoryAgentSession:
    """Tests for AsyncSessionRepository AgentSession CRUD operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up async database state."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_create_session(self, async_configured_db):
        """Test creating an agent session through async repository."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-test-session",
                original_query="Set up MLOps pipeline",
                project_path="/path/to/project",
            )
            assert session.id is not None
            assert session.session_id == "async-test-session"
            assert session.status == "active"

    @pytest.mark.asyncio
    async def test_get_session_by_id(self, async_configured_db):
        """Test getting session by session_id."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            await repo.create_session(
                session_id="async-get-session",
                original_query="Test query",
            )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            found = await repo.get_session_by_id("async-get-session")
            assert found is not None
            assert found.original_query == "Test query"

    @pytest.mark.asyncio
    async def test_list_sessions(self, async_configured_db):
        """Test listing sessions asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            for i in range(3):
                await repo.create_session(
                    session_id=f"async-list-session-{i}",
                    original_query=f"Query {i}",
                )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            sessions = await repo.list_sessions()
            assert len(sessions) == 3

    @pytest.mark.asyncio
    async def test_mark_session_completed(self, async_configured_db):
        """Test marking session as completed asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            await repo.create_session(
                session_id="async-complete-session",
                original_query="Test",
            )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            updated = await repo.mark_session_completed("async-complete-session", success=True)
            assert updated is not None
            assert updated.status == "completed"

    @pytest.mark.asyncio
    async def test_delete_session(self, async_configured_db):
        """Test deleting a session asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            await repo.create_session(
                session_id="async-delete-session",
                original_query="Test",
            )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            deleted = await repo.delete_session("async-delete-session")
            assert deleted is True

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            found = await repo.get_session_by_id("async-delete-session")
            assert found is None


class TestAsyncSessionRepositoryStep:
    """Tests for AsyncSessionRepository Step CRUD operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up async database state."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_create_step(self, async_configured_db):
        """Test creating a step through async repository."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-step-session",
                original_query="Test",
            )
            step = await repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Initialize",
                step_type="CODE",
                tool="init_mlflow",
            )
            assert step.id is not None
            assert step.tool == "init_mlflow"

    @pytest.mark.asyncio
    async def test_get_steps_for_session(self, async_configured_db):
        """Test getting all steps for a session asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-steps-session",
                original_query="Test",
            )
            for i in range(3):
                await repo.create_step(
                    agent_session_id=session.id,
                    step_index=str(i),
                    description=f"Step {i}",
                    step_type="CODE",
                )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.get_session_by_id("async-steps-session")
            steps = await repo.get_steps_for_session(session.id)
            assert len(steps) == 3

    @pytest.mark.asyncio
    async def test_mark_step_completed(self, async_configured_db):
        """Test marking step as completed asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-step-complete",
                original_query="Test",
            )
            await repo.create_step(
                agent_session_id=session.id,
                step_index="0",
                description="Test",
                step_type="CODE",
            )

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.get_session_by_id("async-step-complete")
            updated = await repo.mark_step_completed(session.id, "0", {"success": True})
            assert updated is not None
            assert updated.status == "completed"


class TestAsyncSessionRepositoryExperimentState:
    """Tests for AsyncSessionRepository ExperimentState CRUD operations."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up async database state."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(close_async_db())
        yield
        asyncio.get_event_loop().run_until_complete(close_async_db())

    @pytest.mark.asyncio
    async def test_create_experiment_state(self, async_configured_db):
        """Test creating experiment state through async repository."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-exp-session",
                original_query="Train",
            )
            exp_state = await repo.create_experiment_state(
                agent_session_id=session.id,
                experiment_name="async-exp",
                target_accuracy=0.90,
            )
            assert exp_state.id is not None
            assert exp_state.target_accuracy == 0.90

    @pytest.mark.asyncio
    async def test_update_experiment_metrics(self, async_configured_db):
        """Test updating experiment metrics asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-exp-metrics",
                original_query="Test",
            )
            await repo.create_experiment_state(agent_session_id=session.id)

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.get_session_by_id("async-exp-metrics")
            updated = await repo.update_experiment_metrics(session.id, 0.88, loss=0.10)
            assert updated is not None
            assert updated.current_accuracy == 0.88

    @pytest.mark.asyncio
    async def test_add_artifact(self, async_configured_db):
        """Test adding artifact asynchronously."""
        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.create_session(
                session_id="async-exp-artifact",
                original_query="Test",
            )
            await repo.create_experiment_state(agent_session_id=session.id)

        async with get_async_session() as db:
            repo = AsyncSessionRepository(db)
            session = await repo.get_session_by_id("async-exp-artifact")
            updated = await repo.add_artifact(session.id, "model.pkl")
            assert updated is not None
            assert "model.pkl" in updated.artifacts_created


# ============================================================================
# Export Tests
# ============================================================================


class TestRepositoryExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all exports are available."""
        from db.repositories import __all__

        assert "SessionRepository" in __all__
        assert "AsyncSessionRepository" in __all__

    def test_exports_from_db_module(self):
        """Test that repository exports are available from db module."""
        from db import AsyncSessionRepository, SessionRepository

        assert SessionRepository is not None
        assert AsyncSessionRepository is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
