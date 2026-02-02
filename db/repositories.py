"""
Repository classes for MLOps Agent database operations.

Provides CRUD operations for AgentSession, Step, and ExperimentState models.
Supports both synchronous and asynchronous database access patterns.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, selectinload

from db.models import AgentSession, ExperimentState, Step


class SessionRepository:
    """
    Repository for AgentSession CRUD operations.

    Provides methods for creating, reading, updating, and deleting agent sessions,
    along with their related steps and experiment states.
    """

    def __init__(self, db: Session):
        """
        Initialize repository with a database session.

        Args:
            db: SQLAlchemy Session instance
        """
        self.db = db

    # --- AgentSession CRUD ---

    def create_session(
        self,
        session_id: str,
        original_query: str,
        project_path: str | None = None,
        profile: str = "default",
    ) -> AgentSession:
        """
        Create a new agent session.

        Args:
            session_id: Unique session identifier
            original_query: The user's original query
            project_path: Optional project path
            profile: Configuration profile name

        Returns:
            Created AgentSession instance
        """
        agent_session = AgentSession(
            session_id=session_id,
            original_query=original_query,
            project_path=project_path,
            profile=profile,
            status="active",
        )
        self.db.add(agent_session)
        self.db.flush()
        return agent_session

    def get_session_by_id(self, session_id: str) -> AgentSession | None:
        """
        Get an agent session by its session_id.

        Args:
            session_id: The session identifier

        Returns:
            AgentSession if found, None otherwise
        """
        stmt = select(AgentSession).where(AgentSession.session_id == session_id)
        return self.db.execute(stmt).scalars().first()

    def get_session_by_pk(self, pk: int) -> AgentSession | None:
        """
        Get an agent session by its primary key.

        Args:
            pk: Primary key (id)

        Returns:
            AgentSession if found, None otherwise
        """
        return self.db.get(AgentSession, pk)

    def get_session_with_relations(self, session_id: str) -> AgentSession | None:
        """
        Get an agent session with its steps and experiment state eagerly loaded.

        Args:
            session_id: The session identifier

        Returns:
            AgentSession with relations if found, None otherwise
        """
        stmt = (
            select(AgentSession)
            .where(AgentSession.session_id == session_id)
            .options(
                selectinload(AgentSession.steps),
                selectinload(AgentSession.experiment_state),
            )
        )
        return self.db.execute(stmt).scalars().first()

    def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentSession]:
        """
        List agent sessions with optional filtering.

        Args:
            status: Optional status filter (active, completed, failed, paused)
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of AgentSession instances
        """
        stmt = select(AgentSession).order_by(AgentSession.created_at.desc())
        if status:
            stmt = stmt.where(AgentSession.status == status)
        stmt = stmt.limit(limit).offset(offset)
        return list(self.db.execute(stmt).scalars().all())

    def update_session_status(
        self, session_id: str, status: str, completed_at: datetime | None = None
    ) -> AgentSession | None:
        """
        Update an agent session's status.

        Args:
            session_id: The session identifier
            status: New status value
            completed_at: Optional completion timestamp

        Returns:
            Updated AgentSession if found, None otherwise
        """
        session = self.get_session_by_id(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.utcnow()
            if completed_at:
                session.completed_at = completed_at
            self.db.flush()
        return session

    def mark_session_completed(self, session_id: str, success: bool = True) -> AgentSession | None:
        """
        Mark a session as completed or failed.

        Args:
            session_id: The session identifier
            success: Whether the session completed successfully

        Returns:
            Updated AgentSession if found, None otherwise
        """
        session = self.get_session_by_id(session_id)
        if session:
            session.mark_completed(success)
            self.db.flush()
        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete an agent session and its related records.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        session = self.get_session_by_id(session_id)
        if session:
            self.db.delete(session)
            self.db.flush()
            return True
        return False

    # --- Step CRUD ---

    def create_step(
        self,
        agent_session_id: int,
        step_index: str,
        description: str,
        step_type: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        from_step: str | None = None,
    ) -> Step:
        """
        Create a new execution step.

        Args:
            agent_session_id: Foreign key to agent session
            step_index: Step identifier (e.g., "0", "1A", "ROOT")
            description: Human-readable step description
            step_type: Type of step (ROOT, CODE, PERCEPTION, etc.)
            tool: Optional tool name
            args: Optional tool arguments
            from_step: Optional parent step index

        Returns:
            Created Step instance
        """
        step = Step(
            agent_session_id=agent_session_id,
            step_index=step_index,
            description=description,
            step_type=step_type,
            tool=tool,
            args=args,
            from_step=from_step,
            status="pending",
        )
        self.db.add(step)
        self.db.flush()
        return step

    def get_step(self, agent_session_id: int, step_index: str) -> Step | None:
        """
        Get a step by session ID and step index.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index

        Returns:
            Step if found, None otherwise
        """
        stmt = select(Step).where(
            Step.agent_session_id == agent_session_id,
            Step.step_index == step_index,
        )
        return self.db.execute(stmt).scalars().first()

    def get_steps_for_session(self, agent_session_id: int) -> list[Step]:
        """
        Get all steps for a session.

        Args:
            agent_session_id: The agent session's primary key

        Returns:
            List of Step instances
        """
        stmt = (
            select(Step).where(Step.agent_session_id == agent_session_id).order_by(Step.created_at)
        )
        return list(self.db.execute(stmt).scalars().all())

    def update_step_status(
        self,
        agent_session_id: int,
        step_index: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Step | None:
        """
        Update a step's status and optionally its result or error.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            status: New status value
            result: Optional result dictionary
            error: Optional error message

        Returns:
            Updated Step if found, None otherwise
        """
        step = self.get_step(agent_session_id, step_index)
        if step:
            step.status = status
            if result is not None:
                step.result = result
            if error is not None:
                step.error = error
            if status in ("completed", "failed"):
                step.completed_at = datetime.utcnow()
            self.db.flush()
        return step

    def mark_step_completed(
        self, agent_session_id: int, step_index: str, result: dict[str, Any] | None = None
    ) -> Step | None:
        """
        Mark a step as completed.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            result: Optional result dictionary

        Returns:
            Updated Step if found, None otherwise
        """
        step = self.get_step(agent_session_id, step_index)
        if step:
            step.mark_completed(result)
            self.db.flush()
        return step

    def mark_step_failed(self, agent_session_id: int, step_index: str, error: str) -> Step | None:
        """
        Mark a step as failed.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            error: Error message

        Returns:
            Updated Step if found, None otherwise
        """
        step = self.get_step(agent_session_id, step_index)
        if step:
            step.mark_failed(error)
            self.db.flush()
        return step

    # --- ExperimentState CRUD ---

    def create_experiment_state(
        self,
        agent_session_id: int,
        experiment_name: str | None = None,
        target_accuracy: float = 0.85,
        max_improvement_attempts: int = 3,
    ) -> ExperimentState:
        """
        Create experiment state for a session.

        Args:
            agent_session_id: Foreign key to agent session
            experiment_name: Optional experiment name
            target_accuracy: Target accuracy threshold
            max_improvement_attempts: Maximum improvement attempts allowed

        Returns:
            Created ExperimentState instance
        """
        experiment_state = ExperimentState(
            agent_session_id=agent_session_id,
            experiment_name=experiment_name,
            target_accuracy=target_accuracy,
            max_improvement_attempts=max_improvement_attempts,
        )
        self.db.add(experiment_state)
        self.db.flush()
        return experiment_state

    def get_experiment_state(self, agent_session_id: int) -> ExperimentState | None:
        """
        Get experiment state for a session.

        Args:
            agent_session_id: The agent session's primary key

        Returns:
            ExperimentState if found, None otherwise
        """
        stmt = select(ExperimentState).where(ExperimentState.agent_session_id == agent_session_id)
        return self.db.execute(stmt).scalars().first()

    def update_experiment_metrics(
        self,
        agent_session_id: int,
        accuracy: float,
        loss: float | None = None,
    ) -> ExperimentState | None:
        """
        Update experiment metrics.

        Args:
            agent_session_id: The agent session's primary key
            accuracy: Current accuracy value
            loss: Optional loss value

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = self.get_experiment_state(agent_session_id)
        if state:
            state.update_metrics(accuracy, loss)
            self.db.flush()
        return state

    def update_experiment_config(
        self, agent_session_id: int, config: dict[str, Any]
    ) -> ExperimentState | None:
        """
        Update experiment configuration.

        Args:
            agent_session_id: The agent session's primary key
            config: New configuration dictionary

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = self.get_experiment_state(agent_session_id)
        if state:
            state.current_config = config
            state.updated_at = datetime.utcnow()
            self.db.flush()
        return state

    def update_experiment_stage(self, agent_session_id: int, stage: str) -> ExperimentState | None:
        """
        Update experiment pipeline stage.

        Args:
            agent_session_id: The agent session's primary key
            stage: New stage value

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = self.get_experiment_state(agent_session_id)
        if state:
            state.stage = stage
            state.updated_at = datetime.utcnow()
            self.db.flush()
        return state

    def record_improvement_attempt(
        self,
        agent_session_id: int,
        config_changes: dict[str, Any],
        result_accuracy: float,
    ) -> ExperimentState | None:
        """
        Record an improvement attempt.

        Args:
            agent_session_id: The agent session's primary key
            config_changes: Configuration changes made
            result_accuracy: Resulting accuracy

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = self.get_experiment_state(agent_session_id)
        if state:
            state.record_improvement_attempt(config_changes, result_accuracy)
            self.db.flush()
        return state

    def add_artifact(self, agent_session_id: int, artifact_path: str) -> ExperimentState | None:
        """
        Add an artifact to experiment state.

        Args:
            agent_session_id: The agent session's primary key
            artifact_path: Path to the artifact

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = self.get_experiment_state(agent_session_id)
        if state:
            state.artifacts_created.append(artifact_path)
            state.updated_at = datetime.utcnow()
            self.db.flush()
        return state


class AsyncSessionRepository:
    """
    Async repository for AgentSession CRUD operations.

    Provides async methods for creating, reading, updating, and deleting agent sessions,
    along with their related steps and experiment states.
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize repository with an async database session.

        Args:
            db: SQLAlchemy AsyncSession instance
        """
        self.db = db

    # --- AgentSession CRUD ---

    async def create_session(
        self,
        session_id: str,
        original_query: str,
        project_path: str | None = None,
        profile: str = "default",
    ) -> AgentSession:
        """
        Create a new agent session.

        Args:
            session_id: Unique session identifier
            original_query: The user's original query
            project_path: Optional project path
            profile: Configuration profile name

        Returns:
            Created AgentSession instance
        """
        agent_session = AgentSession(
            session_id=session_id,
            original_query=original_query,
            project_path=project_path,
            profile=profile,
            status="active",
        )
        self.db.add(agent_session)
        await self.db.flush()
        return agent_session

    async def get_session_by_id(self, session_id: str) -> AgentSession | None:
        """
        Get an agent session by its session_id.

        Args:
            session_id: The session identifier

        Returns:
            AgentSession if found, None otherwise
        """
        stmt = select(AgentSession).where(AgentSession.session_id == session_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_session_by_pk(self, pk: int) -> AgentSession | None:
        """
        Get an agent session by its primary key.

        Args:
            pk: Primary key (id)

        Returns:
            AgentSession if found, None otherwise
        """
        return await self.db.get(AgentSession, pk)

    async def get_session_with_relations(self, session_id: str) -> AgentSession | None:
        """
        Get an agent session with its steps and experiment state eagerly loaded.

        Args:
            session_id: The session identifier

        Returns:
            AgentSession with relations if found, None otherwise
        """
        stmt = (
            select(AgentSession)
            .where(AgentSession.session_id == session_id)
            .options(
                selectinload(AgentSession.steps),
                selectinload(AgentSession.experiment_state),
            )
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def list_sessions(
        self,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentSession]:
        """
        List agent sessions with optional filtering.

        Args:
            status: Optional status filter (active, completed, failed, paused)
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            List of AgentSession instances
        """
        stmt = select(AgentSession).order_by(AgentSession.created_at.desc())
        if status:
            stmt = stmt.where(AgentSession.status == status)
        stmt = stmt.limit(limit).offset(offset)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_session_status(
        self, session_id: str, status: str, completed_at: datetime | None = None
    ) -> AgentSession | None:
        """
        Update an agent session's status.

        Args:
            session_id: The session identifier
            status: New status value
            completed_at: Optional completion timestamp

        Returns:
            Updated AgentSession if found, None otherwise
        """
        session = await self.get_session_by_id(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.utcnow()
            if completed_at:
                session.completed_at = completed_at
            await self.db.flush()
        return session

    async def mark_session_completed(
        self, session_id: str, success: bool = True
    ) -> AgentSession | None:
        """
        Mark a session as completed or failed.

        Args:
            session_id: The session identifier
            success: Whether the session completed successfully

        Returns:
            Updated AgentSession if found, None otherwise
        """
        session = await self.get_session_by_id(session_id)
        if session:
            session.mark_completed(success)
            await self.db.flush()
        return session

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete an agent session and its related records.

        Args:
            session_id: The session identifier

        Returns:
            True if deleted, False if not found
        """
        session = await self.get_session_by_id(session_id)
        if session:
            await self.db.delete(session)
            await self.db.flush()
            return True
        return False

    # --- Step CRUD ---

    async def create_step(
        self,
        agent_session_id: int,
        step_index: str,
        description: str,
        step_type: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        from_step: str | None = None,
    ) -> Step:
        """
        Create a new execution step.

        Args:
            agent_session_id: Foreign key to agent session
            step_index: Step identifier (e.g., "0", "1A", "ROOT")
            description: Human-readable step description
            step_type: Type of step (ROOT, CODE, PERCEPTION, etc.)
            tool: Optional tool name
            args: Optional tool arguments
            from_step: Optional parent step index

        Returns:
            Created Step instance
        """
        step = Step(
            agent_session_id=agent_session_id,
            step_index=step_index,
            description=description,
            step_type=step_type,
            tool=tool,
            args=args,
            from_step=from_step,
            status="pending",
        )
        self.db.add(step)
        await self.db.flush()
        return step

    async def get_step(self, agent_session_id: int, step_index: str) -> Step | None:
        """
        Get a step by session ID and step index.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index

        Returns:
            Step if found, None otherwise
        """
        stmt = select(Step).where(
            Step.agent_session_id == agent_session_id,
            Step.step_index == step_index,
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_steps_for_session(self, agent_session_id: int) -> list[Step]:
        """
        Get all steps for a session.

        Args:
            agent_session_id: The agent session's primary key

        Returns:
            List of Step instances
        """
        stmt = (
            select(Step).where(Step.agent_session_id == agent_session_id).order_by(Step.created_at)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_step_status(
        self,
        agent_session_id: int,
        step_index: str,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> Step | None:
        """
        Update a step's status and optionally its result or error.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            status: New status value
            result: Optional result dictionary
            error: Optional error message

        Returns:
            Updated Step if found, None otherwise
        """
        step = await self.get_step(agent_session_id, step_index)
        if step:
            step.status = status
            if result is not None:
                step.result = result
            if error is not None:
                step.error = error
            if status in ("completed", "failed"):
                step.completed_at = datetime.utcnow()
            await self.db.flush()
        return step

    async def mark_step_completed(
        self, agent_session_id: int, step_index: str, result: dict[str, Any] | None = None
    ) -> Step | None:
        """
        Mark a step as completed.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            result: Optional result dictionary

        Returns:
            Updated Step if found, None otherwise
        """
        step = await self.get_step(agent_session_id, step_index)
        if step:
            step.mark_completed(result)
            await self.db.flush()
        return step

    async def mark_step_failed(
        self, agent_session_id: int, step_index: str, error: str
    ) -> Step | None:
        """
        Mark a step as failed.

        Args:
            agent_session_id: The agent session's primary key
            step_index: The step's index
            error: Error message

        Returns:
            Updated Step if found, None otherwise
        """
        step = await self.get_step(agent_session_id, step_index)
        if step:
            step.mark_failed(error)
            await self.db.flush()
        return step

    # --- ExperimentState CRUD ---

    async def create_experiment_state(
        self,
        agent_session_id: int,
        experiment_name: str | None = None,
        target_accuracy: float = 0.85,
        max_improvement_attempts: int = 3,
    ) -> ExperimentState:
        """
        Create experiment state for a session.

        Args:
            agent_session_id: Foreign key to agent session
            experiment_name: Optional experiment name
            target_accuracy: Target accuracy threshold
            max_improvement_attempts: Maximum improvement attempts allowed

        Returns:
            Created ExperimentState instance
        """
        experiment_state = ExperimentState(
            agent_session_id=agent_session_id,
            experiment_name=experiment_name,
            target_accuracy=target_accuracy,
            max_improvement_attempts=max_improvement_attempts,
        )
        self.db.add(experiment_state)
        await self.db.flush()
        return experiment_state

    async def get_experiment_state(self, agent_session_id: int) -> ExperimentState | None:
        """
        Get experiment state for a session.

        Args:
            agent_session_id: The agent session's primary key

        Returns:
            ExperimentState if found, None otherwise
        """
        stmt = select(ExperimentState).where(ExperimentState.agent_session_id == agent_session_id)
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def update_experiment_metrics(
        self,
        agent_session_id: int,
        accuracy: float,
        loss: float | None = None,
    ) -> ExperimentState | None:
        """
        Update experiment metrics.

        Args:
            agent_session_id: The agent session's primary key
            accuracy: Current accuracy value
            loss: Optional loss value

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = await self.get_experiment_state(agent_session_id)
        if state:
            state.update_metrics(accuracy, loss)
            await self.db.flush()
        return state

    async def update_experiment_config(
        self, agent_session_id: int, config: dict[str, Any]
    ) -> ExperimentState | None:
        """
        Update experiment configuration.

        Args:
            agent_session_id: The agent session's primary key
            config: New configuration dictionary

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = await self.get_experiment_state(agent_session_id)
        if state:
            state.current_config = config
            state.updated_at = datetime.utcnow()
            await self.db.flush()
        return state

    async def update_experiment_stage(
        self, agent_session_id: int, stage: str
    ) -> ExperimentState | None:
        """
        Update experiment pipeline stage.

        Args:
            agent_session_id: The agent session's primary key
            stage: New stage value

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = await self.get_experiment_state(agent_session_id)
        if state:
            state.stage = stage
            state.updated_at = datetime.utcnow()
            await self.db.flush()
        return state

    async def record_improvement_attempt(
        self,
        agent_session_id: int,
        config_changes: dict[str, Any],
        result_accuracy: float,
    ) -> ExperimentState | None:
        """
        Record an improvement attempt.

        Args:
            agent_session_id: The agent session's primary key
            config_changes: Configuration changes made
            result_accuracy: Resulting accuracy

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = await self.get_experiment_state(agent_session_id)
        if state:
            state.record_improvement_attempt(config_changes, result_accuracy)
            await self.db.flush()
        return state

    async def add_artifact(
        self, agent_session_id: int, artifact_path: str
    ) -> ExperimentState | None:
        """
        Add an artifact to experiment state.

        Args:
            agent_session_id: The agent session's primary key
            artifact_path: Path to the artifact

        Returns:
            Updated ExperimentState if found, None otherwise
        """
        state = await self.get_experiment_state(agent_session_id)
        if state:
            state.artifacts_created.append(artifact_path)
            state.updated_at = datetime.utcnow()
            await self.db.flush()
        return state


__all__ = [
    "SessionRepository",
    "AsyncSessionRepository",
]
