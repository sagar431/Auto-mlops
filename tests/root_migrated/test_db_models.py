#!/usr/bin/env python3
"""
Tests for the database models module.

Run with: pytest tests/root_migrated/test_db_models.py -v
"""

import pytest
from sqlalchemy import text

from db import close_db, get_session, init_db
from db.models import AgentSession, ExperimentState, Step

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
    return str(tmp_path / "test_models.db")


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
# AgentSession Model Tests
# ============================================================================


class TestAgentSessionModel:
    """Tests for AgentSession model."""

    def test_create_agent_session(self, configured_db):
        """Test creating an agent session."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="test-session-123",
                original_query="Set up MLOps pipeline",
                project_path="/path/to/project",
                profile="default",
            )
            db.add(agent_session)
            db.flush()
            assert agent_session.id is not None
            assert agent_session.status == "active"
            assert agent_session.created_at is not None

    def test_agent_session_mark_completed(self, configured_db):
        """Test marking agent session as completed."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="test-session-complete",
                original_query="Train model",
            )
            db.add(agent_session)
            db.flush()

            agent_session.mark_completed(success=True)
            assert agent_session.status == "completed"
            assert agent_session.completed_at is not None

    def test_agent_session_mark_failed(self, configured_db):
        """Test marking agent session as failed."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="test-session-failed",
                original_query="Train model",
            )
            db.add(agent_session)
            db.flush()

            agent_session.mark_completed(success=False)
            assert agent_session.status == "failed"
            assert agent_session.completed_at is not None

    def test_agent_session_update_timestamp(self, configured_db):
        """Test updating agent session timestamp."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="test-session-ts",
                original_query="Test query",
            )
            db.add(agent_session)
            db.flush()

            agent_session.update_timestamp()
            # In practice there may be no time difference in fast tests
            assert agent_session.updated_at is not None

    def test_agent_session_unique_session_id(self, configured_db):
        """Test session_id uniqueness constraint."""
        with get_session() as db:
            agent_session1 = AgentSession(
                session_id="unique-session",
                original_query="Query 1",
            )
            db.add(agent_session1)

        with pytest.raises(Exception):  # IntegrityError
            with get_session() as db:
                agent_session2 = AgentSession(
                    session_id="unique-session",  # Duplicate
                    original_query="Query 2",
                )
                db.add(agent_session2)
                db.flush()

    def test_query_agent_session_by_session_id(self, configured_db):
        """Test querying agent session by session_id."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="query-test-session",
                original_query="Test query",
                project_path="/test/path",
            )
            db.add(agent_session)

        with get_session() as db:
            found = (
                db.query(AgentSession)
                .filter(AgentSession.session_id == "query-test-session")
                .first()
            )
            assert found is not None
            assert found.project_path == "/test/path"


# ============================================================================
# Step Model Tests
# ============================================================================


class TestStepModel:
    """Tests for Step model."""

    def test_create_step(self, configured_db):
        """Test creating a step."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="step-test-session",
                original_query="Test",
            )
            db.add(agent_session)
            db.flush()

            step = Step(
                agent_session_id=agent_session.id,
                step_index="0",
                description="Initialize MLflow",
                step_type="CODE",
                tool="init_mlflow_experiment",
                args={"experiment_name": "test"},
            )
            db.add(step)
            db.flush()

            assert step.id is not None
            assert step.status == "pending"
            assert step.args == {"experiment_name": "test"}

    def test_step_mark_completed(self, configured_db):
        """Test marking step as completed."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="step-complete-session",
                original_query="Test",
            )
            db.add(agent_session)
            db.flush()

            step = Step(
                agent_session_id=agent_session.id,
                step_index="1",
                description="Run training",
                step_type="CODE",
            )
            db.add(step)
            db.flush()

            result = {"success": True, "accuracy": 0.95}
            step.mark_completed(result)

            assert step.status == "completed"
            assert step.completed_at is not None
            assert step.result == result

    def test_step_mark_failed(self, configured_db):
        """Test marking step as failed."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="step-failed-session",
                original_query="Test",
            )
            db.add(agent_session)
            db.flush()

            step = Step(
                agent_session_id=agent_session.id,
                step_index="2",
                description="Deploy model",
                step_type="CODE",
            )
            db.add(step)
            db.flush()

            step.mark_failed("Deployment failed: connection timeout")

            assert step.status == "failed"
            assert step.error == "Deployment failed: connection timeout"
            assert step.completed_at is not None

    def test_step_foreign_key_constraint(self, configured_db):
        """Test step foreign key constraint to agent session."""
        with pytest.raises(Exception):  # IntegrityError
            with get_session() as db:
                step = Step(
                    agent_session_id=99999,  # Non-existent session
                    step_index="0",
                    description="Orphan step",
                    step_type="CODE",
                )
                db.add(step)
                db.flush()

    def test_step_with_json_result(self, configured_db):
        """Test step with complex JSON result."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="json-result-session",
                original_query="Test",
            )
            db.add(agent_session)
            db.flush()

            complex_result = {
                "success": True,
                "metrics": {
                    "accuracy": 0.92,
                    "loss": 0.08,
                    "f1_score": 0.91,
                },
                "artifacts": ["model.pkl", "config.yaml"],
            }

            step = Step(
                agent_session_id=agent_session.id,
                step_index="3",
                description="Training complete",
                step_type="CODE",
                result=complex_result,
            )
            db.add(step)

        with get_session() as db:
            found = db.query(Step).filter(Step.step_index == "3").first()
            assert found.result["metrics"]["accuracy"] == 0.92
            assert len(found.result["artifacts"]) == 2


# ============================================================================
# ExperimentState Model Tests
# ============================================================================


class TestExperimentStateModel:
    """Tests for ExperimentState model."""

    def test_create_experiment_state(self, configured_db):
        """Test creating an experiment state."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-state-session",
                original_query="Train classifier",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                experiment_name="cat-dog-classifier",
                target_accuracy=0.90,
                stage="training",
            )
            db.add(exp_state)
            db.flush()

            assert exp_state.id is not None
            assert exp_state.target_accuracy == 0.90
            assert exp_state.best_accuracy == 0.0

    def test_experiment_state_update_metrics(self, configured_db):
        """Test updating experiment metrics."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-metrics-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                target_accuracy=0.85,
            )
            db.add(exp_state)
            db.flush()

            exp_state.update_metrics(0.80, loss=0.15)
            assert exp_state.current_accuracy == 0.80
            assert exp_state.current_loss == 0.15
            assert exp_state.best_accuracy == 0.80

            # Update with better accuracy
            exp_state.update_metrics(0.85, loss=0.10)
            assert exp_state.current_accuracy == 0.85
            assert exp_state.best_accuracy == 0.85

    def test_experiment_state_threshold_met(self, configured_db):
        """Test threshold_met check."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-threshold-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                target_accuracy=0.85,
            )
            db.add(exp_state)
            db.flush()

            assert exp_state.threshold_met() is False

            exp_state.update_metrics(0.80)
            assert exp_state.threshold_met() is False

            exp_state.update_metrics(0.85)
            assert exp_state.threshold_met() is True

            exp_state.update_metrics(0.90)
            assert exp_state.threshold_met() is True

    def test_experiment_state_can_improve(self, configured_db):
        """Test can_improve check."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-improve-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                max_improvement_attempts=3,
            )
            db.add(exp_state)
            db.flush()

            assert exp_state.can_improve() is True

            exp_state.improvement_attempt = 2
            assert exp_state.can_improve() is True

            exp_state.improvement_attempt = 3
            assert exp_state.can_improve() is False

    def test_experiment_state_record_improvement(self, configured_db):
        """Test recording improvement attempts."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-record-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                current_accuracy=0.75,
            )
            db.add(exp_state)
            db.flush()

            config_changes = {"learning_rate": 0.001, "batch_size": 64}
            exp_state.record_improvement_attempt(config_changes, 0.80)

            assert exp_state.improvement_attempt == 1
            assert len(exp_state.improvement_history) == 1
            assert exp_state.improvement_history[0]["accuracy_before"] == 0.75
            assert exp_state.improvement_history[0]["accuracy_after"] == 0.80
            assert exp_state.current_accuracy == 0.80

    def test_experiment_state_get_accuracy_gap(self, configured_db):
        """Test getting accuracy gap."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-gap-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                target_accuracy=0.90,
            )
            db.add(exp_state)
            db.flush()

            # No current accuracy
            assert exp_state.get_accuracy_gap() == 0.90

            exp_state.update_metrics(0.75)
            gap = exp_state.get_accuracy_gap()
            assert abs(gap - 0.15) < 0.001

    def test_experiment_state_to_dict(self, configured_db):
        """Test to_dict conversion."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-dict-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                experiment_name="test-exp",
                run_id="run-123",
                current_accuracy=0.85,
                stage="evaluation",
                artifacts_created=["model.pkl"],
            )
            db.add(exp_state)
            db.flush()

            state_dict = exp_state.to_dict()
            assert state_dict["experiment_name"] == "test-exp"
            assert state_dict["run_id"] == "run-123"
            assert state_dict["current_accuracy"] == 0.85
            assert state_dict["stage"] == "evaluation"
            assert state_dict["artifacts_created"] == ["model.pkl"]

    def test_experiment_state_unique_per_session(self, configured_db):
        """Test that each session can only have one experiment state."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-unique-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state1 = ExperimentState(
                agent_session_id=agent_session.id,
                experiment_name="exp1",
            )
            db.add(exp_state1)

        with pytest.raises(Exception):  # IntegrityError
            with get_session() as db:
                sess = (
                    db.query(AgentSession)
                    .filter(AgentSession.session_id == "exp-unique-session")
                    .first()
                )
                exp_state2 = ExperimentState(
                    agent_session_id=sess.id,  # Same session
                    experiment_name="exp2",
                )
                db.add(exp_state2)
                db.flush()

    def test_experiment_state_json_fields(self, configured_db):
        """Test JSON fields storage and retrieval."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="exp-json-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                current_config={
                    "model": {"type": "resnet", "layers": 50},
                    "training": {"epochs": 100, "batch_size": 32},
                },
                improvement_history=[
                    {"attempt": 1, "accuracy": 0.75},
                    {"attempt": 2, "accuracy": 0.82},
                ],
                artifacts_created=["model_v1.pkl", "model_v2.pkl"],
            )
            db.add(exp_state)

        with get_session() as db:
            found = (
                db.query(ExperimentState)
                .join(AgentSession)
                .filter(AgentSession.session_id == "exp-json-session")
                .first()
            )
            assert found.current_config["model"]["type"] == "resnet"
            assert found.current_config["training"]["epochs"] == 100
            assert len(found.improvement_history) == 2
            assert found.improvement_history[1]["accuracy"] == 0.82
            assert "model_v2.pkl" in found.artifacts_created


# ============================================================================
# Relationship Tests
# ============================================================================


class TestModelRelationships:
    """Tests for model relationships."""

    def test_agent_session_steps_relationship(self, configured_db):
        """Test AgentSession to Steps relationship."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="rel-steps-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            step1 = Step(
                agent_session_id=agent_session.id,
                step_index="0",
                description="Init",
                step_type="CODE",
            )
            step2 = Step(
                agent_session_id=agent_session.id,
                step_index="1",
                description="Train",
                step_type="CODE",
            )
            db.add(step1)
            db.add(step2)

        with get_session() as db:
            agent_session = (
                db.query(AgentSession)
                .filter(AgentSession.session_id == "rel-steps-session")
                .first()
            )
            assert len(agent_session.steps) == 2
            assert agent_session.steps[0].step_index in ["0", "1"]

    def test_agent_session_experiment_state_relationship(self, configured_db):
        """Test AgentSession to ExperimentState relationship."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="rel-exp-session",
                original_query="Train",
            )
            db.add(agent_session)
            db.flush()

            exp_state = ExperimentState(
                agent_session_id=agent_session.id,
                experiment_name="rel-test",
            )
            db.add(exp_state)

        with get_session() as db:
            agent_session = (
                db.query(AgentSession).filter(AgentSession.session_id == "rel-exp-session").first()
            )
            assert agent_session.experiment_state is not None
            assert agent_session.experiment_state.experiment_name == "rel-test"

    def test_step_agent_session_relationship(self, configured_db):
        """Test Step to AgentSession back-reference."""
        with get_session() as db:
            agent_session = AgentSession(
                session_id="rel-back-session",
                original_query="Query",
            )
            db.add(agent_session)
            db.flush()

            step = Step(
                agent_session_id=agent_session.id,
                step_index="0",
                description="Test",
                step_type="CODE",
            )
            db.add(step)

        with get_session() as db:
            step = db.query(Step).filter(Step.step_index == "0").first()
            assert step.agent_session is not None
            assert step.agent_session.session_id == "rel-back-session"


# ============================================================================
# Table Creation Tests
# ============================================================================


class TestTableCreation:
    """Tests for table creation."""

    def test_tables_created(self, configured_db):
        """Test that all tables are created."""
        from db import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = {row[0] for row in result}
            assert "agent_sessions" in tables
            assert "steps" in tables
            assert "experiment_states" in tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
