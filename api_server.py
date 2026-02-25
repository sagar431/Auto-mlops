#!/usr/bin/env python3
"""
MLOps Agent API Server - REST API for ML Pipeline Automation

Usage:
    python api_server.py
    uvicorn api_server:app --reload --port 8000

Endpoints:
    POST /run           - Run agent with query
    GET  /status/{id}   - Get session status
    GET  /sessions      - List past sessions
    GET  /health        - Health check
    WS   /ws/{id}       - WebSocket for real-time events
"""

import asyncio
import hashlib
import hmac
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from agent.agent_loop import AgentLoop
from db import (
    AgentSession as DBAgentSession,
)
from db import (
    close_async_db,
    get_async_db,
    init_async_db,
)
from db.repositories import AsyncSessionRepository
from memory.memory_search import MemorySearch
from metrics.collector import metrics_collector
from metrics.models import (
    AgentMetrics,
    LogsResponse,
    MetricsSummary,
    PipelineMetrics,
    SystemMetrics,
)
from security.api_keys import api_key_manager
from security.middleware import AuthorizationError, CurrentUser, get_current_user


def get_cors_origins() -> list[str]:
    """
    Get CORS origins from environment variable.

    Returns a list of allowed origins parsed from the CORS_ORIGINS environment variable.
    The variable should contain comma-separated origins (e.g., "http://localhost:3000,https://example.com").
    If not set, defaults to ["*"] for backwards compatibility.
    """
    cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
    if not cors_origins_env:
        return ["*"]
    return [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]


def get_rate_limit() -> str:
    """
    Get rate limit from environment variable.

    Returns rate limit string for slowapi.
    Default: 100 requests per minute.
    Format: "{count}/{period}" where period is one of: second, minute, hour, day
    """
    return os.environ.get("RATE_LIMIT", "100/minute")


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# Data Models
# ============================================================================


class RunRequest(BaseModel):
    """Request model for running the agent."""

    query: str = Field(..., description="Natural language query", min_length=1)
    project_path: str | None = Field(None, description="Path to ML project")
    accuracy_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Target accuracy")
    auto_approve: bool = Field(
        default=False, description="Auto-approve human-in-loop deployment gates"
    )


class RunResponse(BaseModel):
    """Response model for run request."""

    session_id: str
    status: str
    message: str


class ApprovalRequest(BaseModel):
    """Request model for approval decisions."""

    session_id: str = Field(..., description="Session ID awaiting approval")
    approval_id: str = Field(..., description="Approval request ID")
    approved: bool = Field(..., description="Approval decision")
    reason: str | None = Field(default=None, description="Optional approval reason")


class SessionStatus(BaseModel):
    """Status of an agent session."""

    session_id: str
    status: str  # pending, running, success, failed
    query: str
    project_path: str | None
    current_phase: str
    steps_completed: int
    steps_total: int
    accuracy: float | None
    target_accuracy: float
    started_at: str
    completed_at: str | None
    result: str | None
    errors: list[str]


class SessionSummary(BaseModel):
    """Summary of a past session."""

    session_id: str
    query: str
    status: str
    goal_achieved: bool
    accuracy: float | None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    timestamp: str


# ============================================================================
# Admin Data Models
# ============================================================================


class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""

    username: str = Field(..., description="Username", min_length=1, max_length=50)
    email: str = Field(..., description="Email address", min_length=5, max_length=255)
    password: str = Field(..., description="Password", min_length=8)
    is_admin: bool = Field(default=False, description="Whether user is an admin")


class UserResponse(BaseModel):
    """Response model for user data."""

    id: str
    username: str
    email: str
    is_active: bool
    is_admin: bool
    created_at: str


class CreateAPIKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    name: str = Field(..., description="Name for the API key", min_length=1, max_length=100)
    user_id: str | None = Field(default=None, description="User ID to associate with the key")
    expires_in_days: int | None = Field(default=None, ge=1, description="Days until expiration")
    scopes: list[str] | None = Field(default=None, description="Allowed scopes")


class CreateAPIKeyResponse(BaseModel):
    """Response model for created API key."""

    raw_key: str = Field(..., description="The raw API key (only shown once)")
    key_id: str
    name: str
    user_id: str | None
    created_at: str
    expires_at: str | None


class APIKeyResponse(BaseModel):
    """Response model for API key info."""

    key_id: str
    name: str
    key_prefix: str
    user_id: str | None
    is_active: bool
    created_at: str
    expires_at: str | None
    last_used_at: str | None


# ============================================================================
# Session Manager
# ============================================================================


class SessionManager:
    """Manages active agent sessions using database for persistence."""

    def __init__(self):
        # WebSocket connections are kept in memory (runtime state only)
        self.websockets: dict[str, list[WebSocket]] = {}
        self.session_configs: dict[str, dict[str, Any]] = {}

    def _session_to_dict(self, session: DBAgentSession) -> dict[str, Any]:
        """Convert database session model to API dict format."""
        return {
            "session_id": session.session_id,
            "status": session.status,
            "query": session.original_query,
            "project_path": session.project_path,
            "accuracy_threshold": session.accuracy_threshold,
            "current_phase": session.current_phase,
            "steps_completed": session.steps_completed,
            "steps_total": session.steps_total,
            "accuracy": session.accuracy,
            "started_at": session.created_at.isoformat(),
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "result": session.result,
            "errors": session.errors or [],
            "events": session.events or [],
        }

    async def create_session(
        self,
        db: AsyncSession,
        query: str,
        project_path: str | None,
        threshold: float,
        auto_approve: bool = False,
    ) -> str:
        """Create a new session in the database."""
        session_id = str(uuid.uuid4())

        db_session = DBAgentSession(
            session_id=session_id,
            original_query=query,
            project_path=project_path,
            accuracy_threshold=threshold,
            status="pending",
            current_phase="initializing",
            steps_completed=0,
            steps_total=0,
            accuracy=None,
            result=None,
            errors=[],
            events=[],
        )
        db.add(db_session)
        await db.commit()

        self.websockets[session_id] = []
        self.session_configs[session_id] = {"auto_approve": auto_approve}
        return session_id

    async def get_session(self, db: AsyncSession, session_id: str) -> dict | None:
        """Get session by ID from database."""
        repo = AsyncSessionRepository(db)
        session = await repo.get_session_by_id(session_id)
        if session:
            return self._session_to_dict(session)
        return None

    async def update_session(self, db: AsyncSession, session_id: str, updates: dict):
        """Update session data in the database."""
        repo = AsyncSessionRepository(db)
        session = await repo.get_session_by_id(session_id)
        if session:
            # Map API field names to model field names
            field_mapping = {
                "query": "original_query",
            }
            # JSON fields that need to be flagged as modified
            json_fields = {"errors", "events"}

            for key, value in updates.items():
                db_key = field_mapping.get(key, key)
                if hasattr(session, db_key):
                    setattr(session, db_key, value)
                    if db_key in json_fields:
                        flag_modified(session, db_key)
            session.updated_at = datetime.utcnow()
            await db.commit()

    async def broadcast_event(self, db: AsyncSession, session_id: str, event_type: str, data: dict):
        """Broadcast event to all connected websockets and store in database."""
        event = {"type": event_type, "data": data, "timestamp": datetime.utcnow().isoformat()}

        # Store event in database
        repo = AsyncSessionRepository(db)
        session = await repo.get_session_by_id(session_id)
        if session:
            events = list(session.events or [])
            events.append(event)
            session.events = events
            flag_modified(session, "events")
            session.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(session)

        # Send to websockets (in-memory)
        if session_id not in self.websockets:
            return

        disconnected = []
        for ws in self.websockets[session_id]:
            try:
                await ws.send_json(event)
            except Exception:
                disconnected.append(ws)

        # Remove disconnected websockets
        for ws in disconnected:
            self.websockets[session_id].remove(ws)

    def add_websocket(self, session_id: str, websocket: WebSocket):
        """Add websocket to session."""
        if session_id not in self.websockets:
            self.websockets[session_id] = []
        self.websockets[session_id].append(websocket)

    def remove_websocket(self, session_id: str, websocket: WebSocket):
        """Remove websocket from session."""
        if session_id in self.websockets and websocket in self.websockets[session_id]:
            self.websockets[session_id].remove(websocket)


# Global session manager
session_manager = SessionManager()


# ============================================================================
# User Store (In-Memory)
# ============================================================================


class UserStore:
    """Simple in-memory user store for admin operations."""

    def __init__(self):
        self._users: dict[str, dict] = {}
        self._next_id: int = 1

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        is_admin: bool = False,
    ) -> dict:
        """Create a new user."""
        # Check for duplicate username or email
        for user in self._users.values():
            if user["username"] == username:
                raise ValueError(f"Username already exists: {username}")
            if user["email"] == email:
                raise ValueError(f"Email already exists: {email}")

        user_id = str(self._next_id)
        self._next_id += 1

        # Use salted hash for password security
        salt = os.urandom(32)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=100_000)
        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_salt": salt.hex(),
            "hashed_password": hashed.hex(),
            "is_active": True,
            "is_admin": is_admin,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._users[user_id] = user
        return user

    def get_user(self, user_id: str) -> dict | None:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> dict | None:
        """Get user by username."""
        for user in self._users.values():
            if user["username"] == username:
                return user
        return None

    def list_users(self) -> list[dict]:
        """List all users."""
        return list(self._users.values())


# Global user store
user_store = UserStore()


# ============================================================================
# Admin Authorization Dependency
# ============================================================================


async def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """
    Dependency that requires admin privileges.

    Checks if the current user has admin role or is authenticated as an admin user.
    """
    if not current_user.is_authenticated:
        raise AuthorizationError("Authentication required for admin access")

    # Check for admin role in JWT claims
    if "admin" in current_user.roles:
        return current_user

    # Check if user_id corresponds to an admin user in the store
    if current_user.user_id and not current_user.user_id.startswith("apikey:"):
        user = user_store.get_user(current_user.user_id)
        if user and user.get("is_admin"):
            return current_user

    raise AuthorizationError("Admin privileges required")


# ============================================================================
# Agent Runner
# ============================================================================


async def run_agent_session(session_id: str):
    """Run agent in background and emit events."""
    from db.session import get_async_session

    # Get initial session data
    async with get_async_session() as db:
        session = await session_manager.get_session(db, session_id)
        if not session:
            return
        # Store a copy of session data for the agent run
        session_data = dict(session)
    auto_approve = session_manager.session_configs.get(session_id, {}).get("auto_approve", False)

    async def event_handler(event_type: str, data: dict):
        """Handle agent events and broadcast to websockets."""
        async with get_async_session() as db:
            # Update session based on event
            if event_type == "status":
                await session_manager.update_session(
                    db, session_id, {"status": data.get("status", "running")}
                )

            elif event_type == "phase":
                await session_manager.update_session(
                    db, session_id, {"current_phase": data.get("phase", "")}
                )

            elif event_type == "plan":
                await session_manager.update_session(
                    db, session_id, {"steps_total": data.get("total_steps", 0)}
                )

            elif event_type == "step_complete":
                current = await session_manager.get_session(db, session_id)
                if current:
                    await session_manager.update_session(
                        db, session_id, {"steps_completed": current.get("steps_completed", 0) + 1}
                    )

            elif event_type == "step_failed":
                current = await session_manager.get_session(db, session_id)
                if current:
                    errors = current.get("errors", [])
                    errors.append(data.get("error", "Unknown error"))
                    await session_manager.update_session(db, session_id, {"errors": errors})

            elif event_type == "approval_required":
                await session_manager.update_session(
                    db, session_id, {"status": "paused", "current_phase": "approval"}
                )

            elif event_type == "approval_granted":
                await session_manager.update_session(
                    db, session_id, {"status": "running", "current_phase": "deployment"}
                )

            elif event_type == "approval_denied":
                await session_manager.update_session(
                    db, session_id, {"status": "failed", "current_phase": "approval"}
                )
            elif event_type == "approval_timeout":
                await session_manager.update_session(
                    db, session_id, {"status": "paused", "current_phase": "approval"}
                )

            elif event_type == "improvement_complete":
                await session_manager.update_session(
                    db, session_id, {"accuracy": data.get("new_accuracy")}
                )

            # Broadcast to websockets
            await session_manager.broadcast_event(db, session_id, event_type, data)

    # Update status to running
    async with get_async_session() as db:
        await session_manager.update_session(db, session_id, {"status": "running"})

    try:
        agent = AgentLoop(on_event=event_handler, auto_approve=auto_approve)
        result = await agent.run(
            query=session_data["query"],
            project_path=session_data["project_path"],
            accuracy_threshold=session_data["accuracy_threshold"],
        )

        # Update session with result
        async with get_async_session() as db:
            await session_manager.update_session(
                db,
                session_id,
                {
                    "status": agent.status,
                    "result": result,
                    "completed_at": datetime.utcnow(),
                },
            )

            # Broadcast completion
            await session_manager.broadcast_event(
                db, session_id, "complete", {"status": agent.status, "result": result}
            )

    except Exception as e:
        async with get_async_session() as db:
            current = await session_manager.get_session(db, session_id)
            errors = (current.get("errors", []) if current else []) + [str(e)]
            await session_manager.update_session(
                db,
                session_id,
                {
                    "status": "failed",
                    "errors": errors,
                    "completed_at": datetime.utcnow(),
                },
            )

            await session_manager.broadcast_event(db, session_id, "error", {"error": str(e)})


# ============================================================================
# FastAPI App
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    print("🚀 MLOps Agent API Server starting...")
    # Initialize async database
    await init_async_db()
    print("📦 Database initialized")
    yield
    print("👋 MLOps Agent API Server shutting down...")
    # Close database connections
    await close_async_db()


app = FastAPI(
    title="MLOps Agent API",
    description="AI-powered ML Pipeline Automation API",
    version="1.0.0",
    lifespan=lifespan,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - origins configurable via CORS_ORIGINS env var
# Note: allow_credentials=True is invalid with allow_origins=["*"] per CORS spec.
# Only enable credentials when specific origins are configured.
_cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
@limiter.limit(get_rate_limit)
async def health_check(request: Request):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy", version="1.0.0", timestamp=datetime.utcnow().isoformat()
    )


@app.post("/run", response_model=RunResponse, tags=["Agent"])
@limiter.limit(get_rate_limit)
async def run_agent(
    request: Request,
    run_request: RunRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """
    Start a new agent session.

    The agent will run in the background. Use /status/{session_id} or
    WebSocket /ws/{session_id} to monitor progress.
    """
    # Validate project path if provided
    if run_request.project_path:
        path = Path(run_request.project_path)
        if not path.exists():
            raise HTTPException(
                status_code=400, detail=f"Project path does not exist: {run_request.project_path}"
            )

    # Create session in database
    session_id = await session_manager.create_session(
        db=db,
        query=run_request.query,
        project_path=run_request.project_path,
        threshold=run_request.accuracy_threshold,
        auto_approve=run_request.auto_approve,
    )

    # Run agent in background
    background_tasks.add_task(run_agent_session, session_id)

    return RunResponse(
        session_id=session_id,
        status="started",
        message=f"Agent started. Monitor progress at /status/{session_id} or connect to /ws/{session_id}",
    )


@app.post("/approve", tags=["Agent"])
@limiter.limit(get_rate_limit)
async def approve_action(
    request: Request,
    approval_request: ApprovalRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: CurrentUser = Depends(get_current_user),
):
    """Record an approval decision for a running session."""
    session = await session_manager.get_session(db, approval_request.session_id)
    if not session:
        raise HTTPException(
            status_code=404, detail=f"Session not found: {approval_request.session_id}"
        )

    await session_manager.broadcast_event(
        db,
        approval_request.session_id,
        "approval_decision",
        {
            "approval_id": approval_request.approval_id,
            "approved": approval_request.approved,
            "reason": approval_request.reason,
            "user": current_user.user_id if current_user else None,
        },
    )

    return {"status": "recorded", "approved": approval_request.approved}


@app.get("/status/{session_id}", response_model=SessionStatus, tags=["Agent"])
@limiter.limit(get_rate_limit)
async def get_session_status(
    request: Request,
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get status of an agent session."""
    session = await session_manager.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    return SessionStatus(
        session_id=session["session_id"],
        status=session["status"],
        query=session["query"],
        project_path=session["project_path"],
        current_phase=session["current_phase"],
        steps_completed=session["steps_completed"],
        steps_total=session["steps_total"],
        accuracy=session["accuracy"],
        target_accuracy=session["accuracy_threshold"],
        started_at=session["started_at"],
        completed_at=session["completed_at"],
        result=session["result"],
        errors=session["errors"],
    )


@app.get("/sessions", response_model=list[SessionSummary], tags=["History"])
@limiter.limit(get_rate_limit)
async def list_sessions(
    request: Request,
    limit: int = 20,
    current_user: CurrentUser = Depends(get_current_user),
):
    """List past sessions from memory."""
    ms = MemorySearch()
    sessions = []

    for entry in ms.index_data[-limit:]:
        exp_state = entry.get("experiment_state", {})
        sessions.append(
            SessionSummary(
                session_id=entry["session_id"],
                query=entry["original_query"],
                status=entry.get("status", "unknown"),
                goal_achieved=entry.get("goal_achieved", False),
                accuracy=exp_state.get("best_accuracy"),
                timestamp=entry.get("timestamp", ""),
            )
        )

    return sessions


@app.get("/sessions/{session_id}", tags=["History"])
@limiter.limit(get_rate_limit)
async def get_session_details(
    request: Request,
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Get detailed information about a past session."""
    # Check active sessions first (from database)
    session = await session_manager.get_session(db, session_id)
    if session:
        return session

    # Search in memory (legacy file-based storage)
    ms = MemorySearch()
    for entry in ms.index_data:
        if entry["session_id"] == session_id:
            return entry

    raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for real-time session events.

    Connect to receive events as they happen:
    - status: Agent status changes
    - phase: Phase transitions (perception, decision, etc.)
    - plan: Execution plan generated
    - step_start/step_complete/step_failed: Step progress
    - improvement_start/improvement_complete: Improvement loop
    - complete: Session finished
    - error: Error occurred
    """
    from db.session import get_async_session

    # Get session from database
    async with get_async_session() as db:
        session = await session_manager.get_session(db, session_id)

    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    session_manager.add_websocket(session_id, websocket)

    # Send current state
    await websocket.send_json(
        {
            "type": "connected",
            "data": {
                "session_id": session_id,
                "status": session["status"],
                "current_phase": session["current_phase"],
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

    # Send past events
    for event in session.get("events", []):
        await websocket.send_json(event)

    try:
        while True:
            # Keep connection alive, wait for client messages
            data = await websocket.receive_text()
            # Handle ping/pong
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        session_manager.remove_websocket(session_id, websocket)


@app.get("/tools", tags=["Info"])
@limiter.limit(get_rate_limit)
async def list_available_tools(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all available MLOps tools."""
    from action.execute_step import AVAILABLE_TOOLS

    return {
        "tools": AVAILABLE_TOOLS,
        "count": len(AVAILABLE_TOOLS),
        "categories": {
            "hydra": [
                "analyze_project_config",
                "create_hydra_config",
                "update_hydra_config",
                "validate_hydra_config",
            ],
            "mlflow": [
                "init_mlflow_experiment",
                "start_mlflow_run",
                "log_mlflow_params",
                "log_mlflow_metrics",
                "log_mlflow_artifact",
                "register_mlflow_model",
                "get_best_mlflow_run",
                "end_mlflow_run",
            ],
            "dvc": [
                "init_dvc_repo",
                "configure_dvc_remote",
                "add_data_to_dvc",
                "create_dvc_pipeline",
                "dvc_push",
                "dvc_pull",
                "dvc_reproduce",
            ],
            "docker": [
                "create_ml_dockerfile",
                "build_ml_docker_image",
                "run_training_container",
                "push_docker_image",
            ],
            "github": ["create_github_workflow", "add_workflow_step"],
            "training": [
                "analyze_training_results",
                "suggest_improvements",
                "check_accuracy_threshold",
            ],
        },
    }


# ============================================================================
# Metrics Endpoints
# ============================================================================


@app.get("/metrics", response_model=MetricsSummary, tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def get_metrics(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get complete metrics summary for the dashboard.

    Returns system metrics, agent performance, pipeline stats,
    and time series data for charts.
    """
    return metrics_collector.get_metrics_summary()


@app.get("/metrics/system", response_model=SystemMetrics, tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def get_system_metrics(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get system resource metrics (CPU, memory, disk)."""
    return metrics_collector.get_system_metrics()


@app.get("/metrics/agent", response_model=AgentMetrics, tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def get_agent_metrics(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get agent performance metrics (sessions, success rate, execution time)."""
    return metrics_collector.get_agent_metrics()


@app.get("/metrics/pipeline", response_model=PipelineMetrics, tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def get_pipeline_metrics(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Get pipeline and tool usage metrics."""
    return metrics_collector.get_pipeline_metrics()


@app.get("/metrics/demo", tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def generate_demo_metrics(
    request: Request,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Generate demo data for frontend testing."""
    metrics_collector.generate_demo_data()
    return {"status": "ok", "message": "Demo data generated"}


@app.get("/metrics/prometheus", tags=["Metrics"])
@limiter.limit(get_rate_limit)
async def get_prometheus_metrics(request: Request):
    """
    Get metrics in Prometheus text format.

    This endpoint is designed to be scraped by Prometheus.
    Returns metrics in the standard Prometheus exposition format.

    Example scrape config for prometheus.yml:
        scrape_configs:
          - job_name: 'mlops-agent'
            static_configs:
              - targets: ['localhost:8000']
            metrics_path: '/metrics/prometheus'
    """
    from fastapi.responses import Response

    from observability.metrics import get_metrics_endpoint

    return Response(
        content=get_metrics_endpoint(),
        media_type="text/plain; charset=utf-8",
    )


# ============================================================================
# Logs Endpoints
# ============================================================================


@app.get("/logs", response_model=LogsResponse, tags=["Logs"])
@limiter.limit(get_rate_limit)
async def get_logs(
    request: Request,
    page: int = 1,
    page_size: int = 50,
    level: str = None,
    source: str = None,
    session_id: str = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get execution logs with pagination and filtering.

    Parameters:
    - page: Page number (1-indexed)
    - page_size: Number of logs per page (max 100)
    - level: Filter by log level (info, warning, error, debug)
    - source: Filter by source component
    - session_id: Filter by session ID
    """
    page_size = min(page_size, 100)  # Cap at 100
    return metrics_collector.get_logs(
        page=page, page_size=page_size, level=level, source=source, session_id=session_id
    )


@app.post("/logs", tags=["Logs"])
@limiter.limit(get_rate_limit)
async def create_log(
    request: Request,
    level: str,
    source: str,
    message: str,
    session_id: str = None,
    current_user: CurrentUser = Depends(get_current_user),
):
    """Create a new log entry (for internal use)."""
    metrics_collector.log(level, source, message, session_id)
    return {"status": "ok"}


# ============================================================================
# Admin Endpoints
# ============================================================================


@app.post("/admin/users", response_model=UserResponse, tags=["Admin"])
@limiter.limit(get_rate_limit)
async def create_user(
    request: Request,
    user_request: CreateUserRequest,
    current_user: CurrentUser = Depends(require_admin),
):
    """
    Create a new user (admin only).

    Creates a new user account with the specified credentials.
    """
    try:
        user = user_store.create_user(
            username=user_request.username,
            email=user_request.email,
            password=user_request.password,
            is_admin=user_request.is_admin,
        )
        return UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            is_active=user["is_active"],
            is_admin=user["is_admin"],
            created_at=user["created_at"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/keys", response_model=CreateAPIKeyResponse, tags=["Admin"])
@limiter.limit(get_rate_limit)
async def create_api_key(
    request: Request,
    key_request: CreateAPIKeyRequest,
    current_user: CurrentUser = Depends(require_admin),
):
    """
    Create a new API key (admin only).

    Generates a new API key with the specified configuration.
    The raw key is only returned once and should be stored securely.
    """
    result = api_key_manager.generate(
        name=key_request.name,
        user_id=key_request.user_id,
        expires_in_days=key_request.expires_in_days,
        scopes=key_request.scopes,
    )

    return CreateAPIKeyResponse(
        raw_key=result.raw_key,
        key_id=result.key_info.key_id,
        name=result.key_info.name,
        user_id=result.key_info.user_id,
        created_at=result.key_info.created_at.isoformat(),
        expires_at=result.key_info.expires_at.isoformat() if result.key_info.expires_at else None,
    )


@app.get("/admin/users", response_model=list[UserResponse], tags=["Admin"])
@limiter.limit(get_rate_limit)
async def list_users(
    request: Request,
    current_user: CurrentUser = Depends(require_admin),
):
    """
    List all users (admin only).

    Returns a list of all registered users.
    """
    users = user_store.list_users()
    return [
        UserResponse(
            id=user["id"],
            username=user["username"],
            email=user["email"],
            is_active=user["is_active"],
            is_admin=user["is_admin"],
            created_at=user["created_at"],
        )
        for user in users
    ]


@app.get("/admin/keys", response_model=list[APIKeyResponse], tags=["Admin"])
@limiter.limit(get_rate_limit)
async def list_api_keys(
    request: Request,
    user_id: str | None = None,
    include_revoked: bool = False,
    current_user: CurrentUser = Depends(require_admin),
):
    """
    List all API keys (admin only).

    Returns a list of all API keys, optionally filtered by user.
    """
    keys = api_key_manager.list_keys(user_id=user_id, include_revoked=include_revoked)
    return [
        APIKeyResponse(
            key_id=key.key_id,
            name=key.name,
            key_prefix=key.key_prefix,
            user_id=key.user_id,
            is_active=key.is_active,
            created_at=key.created_at.isoformat(),
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
        )
        for key in keys
    ]


@app.delete("/admin/keys/{key_id}", tags=["Admin"])
@limiter.limit(get_rate_limit)
async def revoke_api_key(
    request: Request,
    key_id: str,
    current_user: CurrentUser = Depends(require_admin),
):
    """
    Revoke an API key (admin only).

    Revokes the specified API key, making it invalid for authentication.
    """
    # First check if the key exists
    key_info = api_key_manager.get_key_info(key_id)
    if key_info is None:
        raise HTTPException(status_code=404, detail=f"API key not found: {key_id}")

    if not key_info.is_active:
        raise HTTPException(status_code=400, detail="API key is already revoked")

    success = api_key_manager.revoke(key_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

    return {"status": "ok", "message": f"API key {key_id} revoked successfully"}


# ============================================================================
# Real-time Metrics WebSocket
# ============================================================================


@app.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """
    WebSocket for real-time metrics updates.

    Sends metrics updates every 5 seconds:
    - system: CPU, memory, disk usage
    - agent: Session counts, success rate
    - pipeline: Pipeline stats

    Also broadcasts log entries as they occur.
    """
    await websocket.accept()

    try:
        while True:
            # Send current metrics
            metrics = metrics_collector.get_metrics_summary()
            await websocket.send_json(
                {
                    "type": "metrics_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": metrics.model_dump(),
                }
            )

            # Wait 5 seconds before next update
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")


if __name__ == "__main__":
    main()
