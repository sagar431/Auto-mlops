"""
Asynchronous MLOps Client for the Auto-MLOps API.
"""

import asyncio
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

import httpx

try:
    import websockets
except ImportError:
    websockets = None


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
class AgentEvent:
    """Event from the agent WebSocket stream."""

    event_type: str
    data: dict[str, Any]
    timestamp: str | None = None


class AsyncMLOpsClient:
    """
    Asynchronous client for the Auto-MLOps API with WebSocket support.

    Usage:
        async with AsyncMLOpsClient(api_key="your-key") as client:
            # Run with real-time events
            async for event in client.run_stream("Set up MLOps pipeline"):
                print(f"{event.event_type}: {event.data}")

            # Or run and wait
            result = await client.run("Set up MLOps pipeline")
            print(result.status)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the async client.

        Args:
            api_key: API key for authentication. Defaults to MLOPS_API_KEY env var.
            base_url: Base URL of the API. Defaults to MLOPS_API_URL env var or localhost.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("MLOPS_API_KEY")
        self.base_url = base_url or os.environ.get("MLOPS_API_URL", "http://localhost:8000")
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        json: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        """Make an async HTTP request."""
        client = await self._get_client()
        url = f"{self.base_url}{path}"
        response = await client.request(
            method=method,
            url=url,
            headers=self._headers(),
            json=json,
            params=params,
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> dict:
        """Check API health."""
        return await self._request("GET", "/health")

    async def tools(self) -> list[dict]:
        """List available MCP tools."""
        data = await self._request("GET", "/tools")
        return data.get("tools", [])

    async def start(
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
        data = await self._request(
            "POST",
            "/run",
            json={
                "query": query,
                "project_path": project_path,
                "accuracy_threshold": accuracy_threshold,
            },
        )
        return data["session_id"]

    async def status(self, session_id: str) -> SessionResult:
        """
        Get the status of a session.

        Args:
            session_id: The session ID to check.

        Returns:
            SessionResult with current status.
        """
        data = await self._request("GET", f"/status/{session_id}")
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

    async def run(
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
        session_id = await self.start(query, project_path, accuracy_threshold)

        while True:
            result = await self.status(session_id)

            if on_status:
                on_status(result)

            if result.status in ("completed", "failed"):
                return result

            await asyncio.sleep(poll_interval)

    async def run_stream(
        self,
        query: str,
        project_path: str | None = None,
        accuracy_threshold: float = 0.85,
    ) -> AsyncIterator[AgentEvent]:
        """
        Run a query and stream events via WebSocket.

        Args:
            query: Natural language query for the agent.
            project_path: Path to the ML project directory.
            accuracy_threshold: Target accuracy threshold.

        Yields:
            AgentEvent objects as they occur.

        Raises:
            ImportError: If websockets library is not installed.
        """
        if websockets is None:
            raise ImportError(
                "websockets library required for streaming. Install with: pip install websockets"
            )

        # Start the session
        session_id = await self.start(query, project_path, accuracy_threshold)

        # Connect to WebSocket
        ws_url = f"{self.ws_url}/ws/{session_id}"
        async with websockets.connect(ws_url) as ws:
            async for message in ws:
                try:
                    data = json.loads(message)
                    event = AgentEvent(
                        event_type=data.get("type", "unknown"),
                        data=data.get("data", {}),
                        timestamp=data.get("timestamp"),
                    )
                    yield event

                    # Check for completion
                    if event.event_type == "complete":
                        break

                except json.JSONDecodeError:
                    continue

    async def sessions(self, limit: int = 10) -> list[dict]:
        """
        List past sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session summaries.
        """
        data = await self._request("GET", "/sessions", params={"limit": limit})
        return data.get("sessions", [])

    async def metrics(self) -> dict:
        """Get system and agent metrics."""
        return await self._request("GET", "/metrics")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
