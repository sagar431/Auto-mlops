"""
Synchronous MLOps Client for the Auto-MLOps API.
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx


@dataclass
class SessionResult:
    """Result of an agent session."""

    session_id: str
    status: str
    query: str
    result: str | None
    steps_completed: int
    steps_total: int
    started_at: str
    completed_at: str | None
    error: str | None

    @property
    def success(self) -> bool:
        return self.status == "completed"


@dataclass
class ToolInfo:
    """Information about an MCP tool."""

    name: str
    description: str
    category: str


class MLOpsClient:
    """
    Synchronous client for the Auto-MLOps API.

    Usage:
        client = MLOpsClient(api_key="your-api-key")

        # Run a query and wait for completion
        result = client.run("Set up MLOps pipeline", project_path="/path/to/project")
        print(result.status)

        # Run without waiting
        session_id = client.start("Set up MLOps pipeline")
        while True:
            status = client.status(session_id)
            if status.status in ("completed", "failed"):
                break
            time.sleep(1)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the client.

        Args:
            api_key: API key for authentication. Defaults to MLOPS_API_KEY env var.
            base_url: Base URL of the API. Defaults to MLOPS_API_URL env var or localhost.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("MLOPS_API_KEY")
        self.base_url = base_url or os.environ.get("MLOPS_API_URL", "http://localhost:8000")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    def _request(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an HTTP request."""
        url = f"{self.base_url}{path}"
        response = self._client.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=json,
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        """Check API health."""
        return self._request("GET", "/health")

    def tools(self) -> list[ToolInfo]:
        """List available MCP tools."""
        data = self._request("GET", "/tools")
        return [
            ToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                category=t.get("category", "general"),
            )
            for t in data.get("tools", [])
        ]

    def start(
        self,
        query: str,
        project_path: str | None = None,
        accuracy_threshold: float = 0.85,
    ) -> str:
        """
        Start an agent session without waiting for completion.

        Args:
            query: Natural language query for the agent.
            project_path: Path to the ML project directory.
            accuracy_threshold: Target accuracy threshold.

        Returns:
            Session ID.
        """
        data = self._request(
            "POST",
            "/run",
            json={
                "query": query,
                "project_path": project_path,
                "accuracy_threshold": accuracy_threshold,
            },
        )
        return data["session_id"]

    def status(self, session_id: str) -> SessionResult:
        """
        Get the status of a session.

        Args:
            session_id: The session ID to check.

        Returns:
            SessionResult with current status.
        """
        data = self._request("GET", f"/status/{session_id}")
        return SessionResult(
            session_id=data["session_id"],
            status=data["status"],
            query=data.get("query", ""),
            result=data.get("result"),
            steps_completed=data.get("steps_completed", 0),
            steps_total=data.get("steps_total", 0),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )

    def run(
        self,
        query: str,
        project_path: str | None = None,
        accuracy_threshold: float = 0.85,
        poll_interval: float = 1.0,
        on_status: Callable[[SessionResult], None] | None = None,
    ) -> SessionResult:
        """
        Run a query and wait for completion.

        Args:
            query: Natural language query for the agent.
            project_path: Path to the ML project directory.
            accuracy_threshold: Target accuracy threshold.
            poll_interval: How often to poll for status (seconds).
            on_status: Optional callback for status updates.

        Returns:
            Final SessionResult.
        """
        session_id = self.start(query, project_path, accuracy_threshold)

        while True:
            result = self.status(session_id)

            if on_status:
                on_status(result)

            if result.status in ("completed", "failed"):
                return result

            time.sleep(poll_interval)

    def sessions(self, limit: int = 10) -> list[dict]:
        """
        List past sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session summaries.
        """
        data = self._request("GET", "/sessions", params={"limit": limit})
        return data.get("sessions", [])

    def metrics(self) -> dict:
        """Get system and agent metrics."""
        return self._request("GET", "/metrics")

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
