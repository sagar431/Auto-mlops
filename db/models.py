"""
Database Models for MLOps Agent.

SQLModel tables for AgentSession, Step, and ExperimentState entities.
These models persist agent execution state to the database.
"""

from datetime import datetime
from typing import Any, Optional

from sqlmodel import JSON, Column, Field, Relationship, SQLModel


class AgentSession(SQLModel, table=True):
    """
    AgentSession model for tracking agent execution sessions.

    Stores session metadata, query, and status information.
    """

    __tablename__ = "agent_sessions"

    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(index=True, unique=True, max_length=100)
    original_query: str = Field(max_length=2000)
    project_path: str | None = Field(default=None, max_length=500)
    profile: str = Field(default="default", max_length=50)
    status: str = Field(default="active", max_length=20)  # active, completed, failed, paused
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)

    # Relationships - use Optional with string refs for forward-declared classes
    steps: list["Step"] = Relationship(back_populates="agent_session")
    experiment_state: Optional["ExperimentState"] = Relationship(  # noqa: UP007
        back_populates="agent_session",
        sa_relationship_kwargs={"uselist": False},
    )

    def mark_completed(self, success: bool = True) -> None:
        """Mark session as completed or failed."""
        self.status = "completed" if success else "failed"
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()


class Step(SQLModel, table=True):
    """
    Step model for tracking individual execution steps within a session.

    Represents a node in the execution graph with status and results.
    """

    __tablename__ = "steps"

    id: int | None = Field(default=None, primary_key=True)
    agent_session_id: int = Field(foreign_key="agent_sessions.id", index=True)
    step_index: str = Field(max_length=50)  # Node ID - supports labels like "0", "0A", "ROOT"
    description: str = Field(max_length=1000)
    step_type: str = Field(max_length=50)  # ROOT, CODE, PERCEPTION, IMPROVE, CONCLUDE
    tool: str | None = Field(default=None, max_length=100)
    args: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    status: str = Field(default="pending", max_length=20)  # pending, completed, failed, skipped
    result: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    error: str | None = Field(default=None, max_length=2000)
    from_step: str | None = Field(default=None, max_length=50)  # Parent node for lineage
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = Field(default=None)

    # Relationship to session - AgentSession is defined above
    agent_session: AgentSession | None = Relationship(back_populates="steps")

    def mark_completed(self, result: dict[str, Any] | None = None) -> None:
        """Mark step as completed with optional result."""
        self.status = "completed"
        self.completed_at = datetime.utcnow()
        if result is not None:
            self.result = result

    def mark_failed(self, error: str) -> None:
        """Mark step as failed with error message."""
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.utcnow()


class ExperimentState(SQLModel, table=True):
    """
    ExperimentState model for tracking ML experiment state within a session.

    Stores experiment metrics, configuration, and improvement history.
    """

    __tablename__ = "experiment_states"

    id: int | None = Field(default=None, primary_key=True)
    agent_session_id: int = Field(foreign_key="agent_sessions.id", unique=True, index=True)

    # Experiment identification
    experiment_name: str | None = Field(default=None, max_length=200)
    run_id: str | None = Field(default=None, max_length=100)
    run_name: str | None = Field(default=None, max_length=200)

    # Metrics tracking
    current_accuracy: float | None = Field(default=None)
    current_loss: float | None = Field(default=None)
    target_accuracy: float = Field(default=0.85)
    best_accuracy: float = Field(default=0.0)

    # Training configuration
    current_config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Improvement loop
    improvement_attempt: int = Field(default=0)
    max_improvement_attempts: int = Field(default=3)
    improvement_history: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))

    # Pipeline stage
    stage: str = Field(
        default="setup", max_length=50
    )  # setup, config, data, training, evaluation, improvement, deploy

    # Artifacts
    artifacts_created: list[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship to session - AgentSession is defined above
    agent_session: AgentSession | None = Relationship(back_populates="experiment_state")

    def update_metrics(self, accuracy: float, loss: float | None = None) -> None:
        """Update current metrics and track best accuracy."""
        self.current_accuracy = accuracy
        if loss is not None:
            self.current_loss = loss
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        self.updated_at = datetime.utcnow()

    def record_improvement_attempt(
        self, config_changes: dict[str, Any], result_accuracy: float
    ) -> None:
        """Record an improvement attempt for history."""
        self.improvement_attempt += 1
        self.improvement_history.append(
            {
                "attempt": self.improvement_attempt,
                "config_changes": config_changes,
                "accuracy_before": self.current_accuracy,
                "accuracy_after": result_accuracy,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.update_metrics(result_accuracy)

    def threshold_met(self) -> bool:
        """Check if accuracy threshold is met."""
        return self.current_accuracy is not None and self.current_accuracy >= self.target_accuracy

    def can_improve(self) -> bool:
        """Check if more improvement attempts are allowed."""
        return self.improvement_attempt < self.max_improvement_attempts

    def get_accuracy_gap(self) -> float:
        """Get gap between current and target accuracy."""
        if self.current_accuracy is None:
            return self.target_accuracy
        return self.target_accuracy - self.current_accuracy

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "current_accuracy": self.current_accuracy,
            "current_loss": self.current_loss,
            "target_accuracy": self.target_accuracy,
            "best_accuracy": self.best_accuracy,
            "current_config": self.current_config,
            "improvement_attempt": self.improvement_attempt,
            "max_improvement_attempts": self.max_improvement_attempts,
            "improvement_history": self.improvement_history,
            "stage": self.stage,
            "artifacts_created": self.artifacts_created,
        }
