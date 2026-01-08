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
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent.agent_loop import AgentLoop
from memory.memory_search import MemorySearch


# ============================================================================
# Data Models
# ============================================================================

class RunRequest(BaseModel):
    """Request model for running the agent."""
    query: str = Field(..., description="Natural language query", min_length=1)
    project_path: Optional[str] = Field(None, description="Path to ML project")
    accuracy_threshold: float = Field(0.85, ge=0.0, le=1.0, description="Target accuracy")


class RunResponse(BaseModel):
    """Response model for run request."""
    session_id: str
    status: str
    message: str


class SessionStatus(BaseModel):
    """Status of an agent session."""
    session_id: str
    status: str  # pending, running, success, failed
    query: str
    project_path: Optional[str]
    current_phase: str
    steps_completed: int
    steps_total: int
    accuracy: Optional[float]
    target_accuracy: float
    started_at: str
    completed_at: Optional[str]
    result: Optional[str]
    errors: List[str]


class SessionSummary(BaseModel):
    """Summary of a past session."""
    session_id: str
    query: str
    status: str
    goal_achieved: bool
    accuracy: Optional[float]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str


# ============================================================================
# Session Manager
# ============================================================================

class SessionManager:
    """Manages active agent sessions."""

    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.websockets: Dict[str, List[WebSocket]] = {}

    def create_session(self, query: str, project_path: Optional[str], threshold: float) -> str:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "status": "pending",
            "query": query,
            "project_path": project_path,
            "accuracy_threshold": threshold,
            "current_phase": "initializing",
            "steps_completed": 0,
            "steps_total": 0,
            "accuracy": None,
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "result": None,
            "errors": [],
            "events": []
        }
        self.websockets[session_id] = []
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def update_session(self, session_id: str, updates: Dict):
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)

    async def broadcast_event(self, session_id: str, event_type: str, data: Dict):
        """Broadcast event to all connected websockets."""
        if session_id not in self.websockets:
            return

        event = {"type": event_type, "data": data, "timestamp": datetime.utcnow().isoformat()}

        # Store event in session
        if session_id in self.sessions:
            self.sessions[session_id]["events"].append(event)

        # Send to websockets
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
        if session_id in self.websockets:
            self.websockets[session_id].append(websocket)

    def remove_websocket(self, session_id: str, websocket: WebSocket):
        """Remove websocket from session."""
        if session_id in self.websockets and websocket in self.websockets[session_id]:
            self.websockets[session_id].remove(websocket)


# Global session manager
session_manager = SessionManager()


# ============================================================================
# Agent Runner
# ============================================================================

async def run_agent_session(session_id: str):
    """Run agent in background and emit events."""
    session = session_manager.get_session(session_id)
    if not session:
        return

    async def event_handler(event_type: str, data: Dict):
        """Handle agent events and broadcast to websockets."""
        # Update session based on event
        if event_type == "status":
            session_manager.update_session(session_id, {"status": data.get("status", "running")})

        elif event_type == "phase":
            session_manager.update_session(session_id, {"current_phase": data.get("phase", "")})

        elif event_type == "plan":
            session_manager.update_session(session_id, {"steps_total": data.get("total_steps", 0)})

        elif event_type == "step_complete":
            current = session_manager.get_session(session_id)
            if current:
                session_manager.update_session(session_id, {
                    "steps_completed": current.get("steps_completed", 0) + 1
                })

        elif event_type == "step_failed":
            current = session_manager.get_session(session_id)
            if current:
                errors = current.get("errors", [])
                errors.append(data.get("error", "Unknown error"))
                session_manager.update_session(session_id, {"errors": errors})

        elif event_type == "improvement_complete":
            session_manager.update_session(session_id, {"accuracy": data.get("new_accuracy")})

        # Broadcast to websockets
        await session_manager.broadcast_event(session_id, event_type, data)

    # Update status to running
    session_manager.update_session(session_id, {"status": "running"})

    try:
        agent = AgentLoop(on_event=event_handler)
        result = await agent.run(
            query=session["query"],
            project_path=session["project_path"],
            accuracy_threshold=session["accuracy_threshold"]
        )

        # Update session with result
        session_manager.update_session(session_id, {
            "status": agent.status,
            "result": result,
            "completed_at": datetime.utcnow().isoformat()
        })

        # Broadcast completion
        await session_manager.broadcast_event(session_id, "complete", {
            "status": agent.status,
            "result": result
        })

    except Exception as e:
        session_manager.update_session(session_id, {
            "status": "failed",
            "errors": session.get("errors", []) + [str(e)],
            "completed_at": datetime.utcnow().isoformat()
        })

        await session_manager.broadcast_event(session_id, "error", {"error": str(e)})


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    print("🚀 MLOps Agent API Server starting...")
    yield
    print("👋 MLOps Agent API Server shutting down...")


app = FastAPI(
    title="MLOps Agent API",
    description="AI-powered ML Pipeline Automation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/run", response_model=RunResponse, tags=["Agent"])
async def run_agent(request: RunRequest, background_tasks: BackgroundTasks):
    """
    Start a new agent session.

    The agent will run in the background. Use /status/{session_id} or
    WebSocket /ws/{session_id} to monitor progress.
    """
    # Validate project path if provided
    if request.project_path:
        path = Path(request.project_path)
        if not path.exists():
            raise HTTPException(status_code=400, detail=f"Project path does not exist: {request.project_path}")

    # Create session
    session_id = session_manager.create_session(
        query=request.query,
        project_path=request.project_path,
        threshold=request.accuracy_threshold
    )

    # Run agent in background
    background_tasks.add_task(run_agent_session, session_id)

    return RunResponse(
        session_id=session_id,
        status="started",
        message=f"Agent started. Monitor progress at /status/{session_id} or connect to /ws/{session_id}"
    )


@app.get("/status/{session_id}", response_model=SessionStatus, tags=["Agent"])
async def get_session_status(session_id: str):
    """Get status of an agent session."""
    session = session_manager.get_session(session_id)
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
        errors=session["errors"]
    )


@app.get("/sessions", response_model=List[SessionSummary], tags=["History"])
async def list_sessions(limit: int = 20):
    """List past sessions from memory."""
    ms = MemorySearch()
    sessions = []

    for entry in ms.index_data[-limit:]:
        exp_state = entry.get("experiment_state", {})
        sessions.append(SessionSummary(
            session_id=entry["session_id"],
            query=entry["original_query"],
            status=entry.get("status", "unknown"),
            goal_achieved=entry.get("goal_achieved", False),
            accuracy=exp_state.get("best_accuracy"),
            timestamp=entry.get("timestamp", "")
        ))

    return sessions


@app.get("/sessions/{session_id}", tags=["History"])
async def get_session_details(session_id: str):
    """Get detailed information about a past session."""
    # Check active sessions first
    session = session_manager.get_session(session_id)
    if session:
        return session

    # Search in memory
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
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await websocket.accept()
    session_manager.add_websocket(session_id, websocket)

    # Send current state
    await websocket.send_json({
        "type": "connected",
        "data": {
            "session_id": session_id,
            "status": session["status"],
            "current_phase": session["current_phase"]
        },
        "timestamp": datetime.utcnow().isoformat()
    })

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
async def list_available_tools():
    """List all available MLOps tools."""
    from action.execute_step import AVAILABLE_TOOLS
    return {
        "tools": AVAILABLE_TOOLS,
        "count": len(AVAILABLE_TOOLS),
        "categories": {
            "hydra": ["analyze_project_config", "create_hydra_config", "update_hydra_config", "validate_hydra_config"],
            "mlflow": ["init_mlflow_experiment", "start_mlflow_run", "log_mlflow_params", "log_mlflow_metrics",
                      "log_mlflow_artifact", "register_mlflow_model", "get_best_mlflow_run", "end_mlflow_run"],
            "dvc": ["init_dvc_repo", "configure_dvc_remote", "add_data_to_dvc", "create_dvc_pipeline",
                   "dvc_push", "dvc_pull", "dvc_reproduce"],
            "docker": ["create_ml_dockerfile", "build_ml_docker_image", "run_training_container", "push_docker_image"],
            "github": ["create_github_workflow", "add_workflow_step"],
            "training": ["analyze_training_results", "suggest_improvements", "check_accuracy_threshold"]
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
