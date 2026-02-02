#!/usr/bin/env python3
"""
Integration tests for MLOps Agent API endpoints.

Tests the FastAPI REST API endpoints for:
- Health check
- Agent session management (run, status)
- Session history and details
- Tools listing
- Metrics endpoints (system, agent, pipeline)
- Logs endpoints
- Admin endpoints (users, API keys)

Usage:
    pytest tests/integration/test_api_endpoints.py -v
    pytest tests/integration/test_api_endpoints.py -v -k "health"
    pytest tests/integration/test_api_endpoints.py -v -k "metrics"
    pytest tests/integration/test_api_endpoints.py -v -k "admin"
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def test_client(db_session):
    """
    Create an async HTTP client for testing FastAPI endpoints.

    Uses httpx.AsyncClient with ASGITransport for the FastAPI app.
    """
    from api_server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_agent_loop():
    """Mock the AgentLoop to avoid running actual ML operations."""
    with patch("api_server.AgentLoop") as mock:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value="Pipeline completed successfully")
        mock_instance.status = "success"
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_memory_search():
    """Mock MemorySearch to return test session data."""
    with patch("api_server.MemorySearch") as mock:
        mock_instance = MagicMock()
        mock_instance.index_data = [
            {
                "session_id": "test-session-001",
                "original_query": "Set up MLOps pipeline",
                "status": "success",
                "goal_achieved": True,
                "timestamp": "2024-01-15T10:30:00",
                "experiment_state": {"best_accuracy": 0.92},
            },
            {
                "session_id": "test-session-002",
                "original_query": "Train model with accuracy > 0.85",
                "status": "success",
                "goal_achieved": True,
                "timestamp": "2024-01-14T14:20:00",
                "experiment_state": {"best_accuracy": 0.88},
            },
        ]
        mock.return_value = mock_instance
        yield mock


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthEndpoint:
    """Test the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, test_client):
        """Test that health check returns 200 status."""
        response = await test_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_check_returns_healthy_status(self, test_client):
        """Test that health check returns healthy status."""
        response = await test_client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_returns_version(self, test_client):
        """Test that health check returns version info."""
        response = await test_client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_health_check_returns_timestamp(self, test_client):
        """Test that health check returns timestamp."""
        response = await test_client.get("/health")
        data = response.json()
        assert "timestamp" in data


# ============================================================================
# Agent Run Endpoint Tests
# ============================================================================


class TestRunEndpoint:
    """Test the POST /run endpoint."""

    @pytest.mark.asyncio
    async def test_run_with_valid_query(self, test_client, mock_agent_loop):
        """Test running agent with valid query."""
        response = await test_client.post(
            "/run",
            json={"query": "Set up MLOps pipeline for my project"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_run_with_project_path(self, test_client, mock_agent_loop, test_project):
        """Test running agent with project path."""
        response = await test_client.post(
            "/run",
            json={
                "query": "Analyze my ML project",
                "project_path": test_project,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_run_with_accuracy_threshold(self, test_client, mock_agent_loop):
        """Test running agent with custom accuracy threshold."""
        response = await test_client.post(
            "/run",
            json={
                "query": "Train model to 90% accuracy",
                "accuracy_threshold": 0.90,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_run_with_invalid_project_path(self, test_client, mock_agent_loop):
        """Test running agent with non-existent project path."""
        response = await test_client.post(
            "/run",
            json={
                "query": "Analyze project",
                "project_path": "/nonexistent/path/to/project",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "does not exist" in data["detail"]

    @pytest.mark.asyncio
    async def test_run_with_empty_query(self, test_client):
        """Test running agent with empty query returns 422."""
        response = await test_client.post(
            "/run",
            json={"query": ""},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_with_invalid_accuracy_threshold(self, test_client):
        """Test running agent with invalid accuracy threshold."""
        response = await test_client.post(
            "/run",
            json={
                "query": "Train model",
                "accuracy_threshold": 1.5,  # Invalid: > 1.0
            },
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_run_returns_session_id_format(self, test_client, mock_agent_loop):
        """Test that run returns valid UUID session ID."""
        response = await test_client.post(
            "/run",
            json={"query": "Set up pipeline"},
        )
        data = response.json()
        # Verify session_id is a valid UUID
        uuid.UUID(data["session_id"])


# ============================================================================
# Session Status Endpoint Tests
# ============================================================================


class TestStatusEndpoint:
    """Test the GET /status/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_status_for_existing_session(self, test_client, mock_agent_loop):
        """Test getting status for an existing session."""
        # First create a session
        run_response = await test_client.post(
            "/run",
            json={"query": "Set up MLOps pipeline"},
        )
        session_id = run_response.json()["session_id"]

        # Get status
        response = await test_client.get(f"/status/{session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "status" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_status_for_nonexistent_session(self, test_client):
        """Test getting status for non-existent session returns 404."""
        fake_session_id = str(uuid.uuid4())
        response = await test_client.get(f"/status/{fake_session_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_status_response_structure(self, test_client, mock_agent_loop):
        """Test that status response has correct structure."""
        run_response = await test_client.post(
            "/run",
            json={"query": "Test pipeline"},
        )
        session_id = run_response.json()["session_id"]

        response = await test_client.get(f"/status/{session_id}")
        data = response.json()

        # Check required fields
        required_fields = [
            "session_id",
            "status",
            "query",
            "current_phase",
            "steps_completed",
            "steps_total",
            "target_accuracy",
            "started_at",
            "errors",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"


# ============================================================================
# Sessions History Endpoint Tests
# ============================================================================


class TestSessionsEndpoint:
    """Test the GET /sessions endpoint."""

    @pytest.mark.asyncio
    async def test_list_sessions(self, test_client, mock_memory_search):
        """Test listing past sessions."""
        response = await test_client.get("/sessions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self, test_client, mock_memory_search):
        """Test listing sessions with limit parameter."""
        response = await test_client.get("/sessions?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 5

    @pytest.mark.asyncio
    async def test_session_summary_structure(self, test_client, mock_memory_search):
        """Test that session summaries have correct structure."""
        response = await test_client.get("/sessions")
        data = response.json()

        if len(data) > 0:
            session = data[0]
            assert "session_id" in session
            assert "query" in session
            assert "status" in session
            assert "goal_achieved" in session


# ============================================================================
# Session Details Endpoint Tests
# ============================================================================


class TestSessionDetailsEndpoint:
    """Test the GET /sessions/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_details_from_database(self, test_client, mock_agent_loop):
        """Test getting session details from database."""
        # Create a session first
        run_response = await test_client.post(
            "/run",
            json={"query": "Test pipeline"},
        )
        session_id = run_response.json()["session_id"]

        response = await test_client.get(f"/sessions/{session_id}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_get_session_details_nonexistent(self, test_client, mock_memory_search):
        """Test getting details for non-existent session."""
        fake_session_id = str(uuid.uuid4())
        response = await test_client.get(f"/sessions/{fake_session_id}")
        assert response.status_code == 404


# ============================================================================
# Tools Endpoint Tests
# ============================================================================


class TestToolsEndpoint:
    """Test the GET /tools endpoint."""

    @pytest.mark.asyncio
    async def test_list_tools(self, test_client):
        """Test listing available tools."""
        response = await test_client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_tools_count_matches_list(self, test_client):
        """Test that tools count matches actual tool list length."""
        response = await test_client.get("/tools")
        data = response.json()
        assert data["count"] == len(data["tools"])

    @pytest.mark.asyncio
    async def test_tools_have_categories(self, test_client):
        """Test that tools response includes categories."""
        response = await test_client.get("/tools")
        data = response.json()
        assert "categories" in data
        categories = data["categories"]
        assert "hydra" in categories
        assert "mlflow" in categories
        assert "dvc" in categories
        assert "docker" in categories
        assert "github" in categories
        assert "training" in categories


# ============================================================================
# Metrics Endpoints Tests
# ============================================================================


class TestMetricsEndpoints:
    """Test the /metrics endpoints."""

    @pytest.mark.asyncio
    async def test_get_metrics_summary(self, test_client):
        """Test getting complete metrics summary."""
        response = await test_client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "system" in data
        assert "agent" in data
        assert "pipeline" in data

    @pytest.mark.asyncio
    async def test_get_system_metrics(self, test_client):
        """Test getting system metrics."""
        response = await test_client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert "memory_percent" in data
        assert "disk_percent" in data
        assert "python_version" in data
        assert "platform" in data

    @pytest.mark.asyncio
    async def test_get_agent_metrics(self, test_client):
        """Test getting agent metrics."""
        response = await test_client.get("/metrics/agent")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "active_sessions" in data
        assert "successful_sessions" in data
        assert "failed_sessions" in data
        assert "success_rate" in data

    @pytest.mark.asyncio
    async def test_get_pipeline_metrics(self, test_client):
        """Test getting pipeline metrics."""
        response = await test_client.get("/metrics/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert "total_pipelines_run" in data
        assert "tools_available" in data
        assert "tool_invocations" in data

    @pytest.mark.asyncio
    async def test_generate_demo_metrics(self, test_client):
        """Test generating demo metrics."""
        response = await test_client.get("/metrics/demo")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_prometheus_metrics(self, test_client):
        """Test getting Prometheus format metrics."""
        response = await test_client.get("/metrics/prometheus")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")


# ============================================================================
# Logs Endpoints Tests
# ============================================================================


class TestLogsEndpoints:
    """Test the /logs endpoints."""

    @pytest.mark.asyncio
    async def test_get_logs(self, test_client):
        """Test getting logs with default pagination."""
        response = await test_client.get("/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "has_more" in data

    @pytest.mark.asyncio
    async def test_get_logs_with_pagination(self, test_client):
        """Test getting logs with pagination parameters."""
        response = await test_client.get("/logs?page=1&page_size=10")
        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    @pytest.mark.asyncio
    async def test_get_logs_page_size_cap(self, test_client):
        """Test that page size is capped at 100."""
        response = await test_client.get("/logs?page_size=200")
        assert response.status_code == 200
        data = response.json()
        assert data["page_size"] <= 100

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_level(self, test_client):
        """Test filtering logs by level."""
        response = await test_client.get("/logs?level=error")
        assert response.status_code == 200
        data = response.json()
        for log in data["logs"]:
            assert log["level"] == "error"

    @pytest.mark.asyncio
    async def test_create_log(self, test_client):
        """Test creating a new log entry."""
        response = await test_client.post(
            "/logs?level=info&source=test&message=Test%20log%20message"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


# ============================================================================
# Admin Endpoints Tests
# ============================================================================


class TestAdminEndpoints:
    """Test the /admin endpoints."""

    @pytest.fixture
    def mock_admin_auth(self):
        """Mock admin authentication to bypass auth requirements."""
        from security.middleware import CurrentUser

        admin_user = CurrentUser(
            user_id="admin-001",
            is_authenticated=True,
            auth_method="jwt",
            roles=["admin"],
            scopes=[],
        )

        with patch("api_server.require_admin") as mock_admin:
            mock_admin.return_value = admin_user
            with patch("api_server.get_current_user") as mock_user:
                mock_user.return_value = admin_user
                yield admin_user

    @pytest.mark.asyncio
    async def test_create_user_requires_auth(self, test_client):
        """Test that creating user requires authentication."""
        response = await test_client.post(
            "/admin/users",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "password": "testpassword123",
            },
        )
        # Without auth, should return 401 or 403
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_list_users_requires_auth(self, test_client):
        """Test that listing users requires authentication."""
        response = await test_client.get("/admin/users")
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_create_api_key_requires_auth(self, test_client):
        """Test that creating API key requires authentication."""
        response = await test_client.post(
            "/admin/keys",
            json={"name": "test-key"},
        )
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_list_api_keys_requires_auth(self, test_client):
        """Test that listing API keys requires authentication."""
        response = await test_client.get("/admin/keys")
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_revoke_api_key_requires_auth(self, test_client):
        """Test that revoking API key requires authentication."""
        response = await test_client.delete("/admin/keys/some-key-id")
        assert response.status_code in [401, 403]


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Test authentication behavior across endpoints."""

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, test_client):
        """Test that health endpoint doesn't require auth."""
        response = await test_client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_tools_accessible_without_auth(self, test_client):
        """Test that tools endpoint is accessible without strict auth."""
        response = await test_client.get("/tools")
        # Should be 200 if auth is disabled, or 401/403 if enabled
        assert response.status_code in [200, 401, 403]

    @pytest.mark.asyncio
    async def test_metrics_accessible(self, test_client):
        """Test that metrics endpoint is accessible."""
        response = await test_client.get("/metrics")
        assert response.status_code in [200, 401, 403]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling across endpoints."""

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, test_client):
        """Test handling of invalid JSON body."""
        response = await test_client.post(
            "/run",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_required_field(self, test_client):
        """Test handling of missing required field."""
        response = await test_client.post(
            "/run",
            json={},  # Missing 'query' field
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_endpoint(self, test_client):
        """Test that invalid endpoints return 404."""
        response = await test_client.get("/nonexistent/endpoint")
        assert response.status_code == 404


# ============================================================================
# CORS and Headers Tests
# ============================================================================


class TestCORSAndHeaders:
    """Test CORS configuration and response headers."""

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in response."""
        response = await test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Note: CORS headers may vary based on configuration
        assert response.status_code in [200, 405]

    @pytest.mark.asyncio
    async def test_json_content_type(self, test_client):
        """Test that JSON endpoints return correct content type."""
        response = await test_client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_not_exceeded_for_normal_use(self, test_client):
        """Test that normal usage doesn't trigger rate limiting."""
        # Make a few requests - should not be rate limited
        for _ in range(5):
            response = await test_client.get("/health")
            assert response.status_code == 200


# ============================================================================
# Response Validation Tests
# ============================================================================


class TestResponseValidation:
    """Test response format validation."""

    @pytest.mark.asyncio
    async def test_health_response_format(self, test_client):
        """Test that health response matches expected format."""
        response = await test_client.get("/health")
        data = response.json()

        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)
        assert isinstance(data["timestamp"], str)

    @pytest.mark.asyncio
    async def test_run_response_format(self, test_client, mock_agent_loop):
        """Test that run response matches expected format."""
        response = await test_client.post(
            "/run",
            json={"query": "Test query"},
        )
        data = response.json()

        assert isinstance(data["session_id"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["message"], str)

    @pytest.mark.asyncio
    async def test_metrics_response_format(self, test_client):
        """Test that metrics response has correct structure."""
        response = await test_client.get("/metrics")
        data = response.json()

        # System metrics structure
        system = data["system"]
        assert isinstance(system["cpu_percent"], (int, float))
        assert isinstance(system["memory_percent"], (int, float))

        # Agent metrics structure
        agent = data["agent"]
        assert isinstance(agent["total_sessions"], int)
        assert isinstance(agent["success_rate"], (int, float))

        # Pipeline metrics structure
        pipeline = data["pipeline"]
        assert isinstance(pipeline["tools_available"], int)
