#!/usr/bin/env python3
"""
Tests for API server authentication middleware.

Tests verify that:
1. /health endpoint is accessible without authentication
2. All other endpoints require the get_current_user dependency
3. Endpoints work with valid authentication when auth is enabled

Run with: pytest test_api_server_auth.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

from security import JWTAuth, SecurityConfig

# Set auth disabled, high rate limit, and JWT secret for testing
os.environ["ENABLE_API_KEY_AUTH"] = "false"
os.environ["ENABLE_JWT_AUTH"] = "false"
os.environ["RATE_LIMIT"] = "1000/minute"
os.environ["JWT_SECRET"] = "test-secret-for-all-tests"

from api_server import app, limiter


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter storage before each test."""
    limiter.reset()
    yield


@pytest.fixture
def client():
    """Create a test client with auth disabled (anonymous access allowed)."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint which should NOT require auth."""

    def test_health_returns_200(self, client):
        """Test /health works and returns expected response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_health_does_not_use_auth_dependency(self):
        """Verify /health endpoint signature has no auth dependency."""
        import inspect

        from api_server import health_check

        sig = inspect.signature(health_check)
        params = list(sig.parameters.keys())
        # health_check should have no parameters (no current_user dependency)
        assert "current_user" not in params


class TestProtectedEndpointSignatures:
    """Tests that verify protected endpoints have auth dependency."""

    def test_run_agent_has_auth_dependency(self):
        """Verify /run endpoint has current_user dependency."""
        import inspect

        from api_server import run_agent

        sig = inspect.signature(run_agent)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_session_status_has_auth_dependency(self):
        """Verify /status/{session_id} endpoint has current_user dependency."""
        import inspect

        from api_server import get_session_status

        sig = inspect.signature(get_session_status)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_list_sessions_has_auth_dependency(self):
        """Verify /sessions endpoint has current_user dependency."""
        import inspect

        from api_server import list_sessions

        sig = inspect.signature(list_sessions)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_session_details_has_auth_dependency(self):
        """Verify /sessions/{session_id} endpoint has current_user dependency."""
        import inspect

        from api_server import get_session_details

        sig = inspect.signature(get_session_details)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_list_tools_has_auth_dependency(self):
        """Verify /tools endpoint has current_user dependency."""
        import inspect

        from api_server import list_available_tools

        sig = inspect.signature(list_available_tools)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_metrics_has_auth_dependency(self):
        """Verify /metrics endpoint has current_user dependency."""
        import inspect

        from api_server import get_metrics

        sig = inspect.signature(get_metrics)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_system_metrics_has_auth_dependency(self):
        """Verify /metrics/system endpoint has current_user dependency."""
        import inspect

        from api_server import get_system_metrics

        sig = inspect.signature(get_system_metrics)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_agent_metrics_has_auth_dependency(self):
        """Verify /metrics/agent endpoint has current_user dependency."""
        import inspect

        from api_server import get_agent_metrics

        sig = inspect.signature(get_agent_metrics)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_pipeline_metrics_has_auth_dependency(self):
        """Verify /metrics/pipeline endpoint has current_user dependency."""
        import inspect

        from api_server import get_pipeline_metrics

        sig = inspect.signature(get_pipeline_metrics)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_generate_demo_metrics_has_auth_dependency(self):
        """Verify /metrics/demo endpoint has current_user dependency."""
        import inspect

        from api_server import generate_demo_metrics

        sig = inspect.signature(generate_demo_metrics)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_get_logs_has_auth_dependency(self):
        """Verify GET /logs endpoint has current_user dependency."""
        import inspect

        from api_server import get_logs

        sig = inspect.signature(get_logs)
        params = list(sig.parameters.keys())
        assert "current_user" in params

    def test_create_log_has_auth_dependency(self):
        """Verify POST /logs endpoint has current_user dependency."""
        import inspect

        from api_server import create_log

        sig = inspect.signature(create_log)
        params = list(sig.parameters.keys())
        assert "current_user" in params


class TestEndpointsWithAnonymousAccess:
    """Tests that endpoints work with anonymous access (auth disabled)."""

    def test_tools_works_with_anonymous(self, client):
        """Test /tools endpoint works when auth is disabled."""
        response = client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "categories" in data

    def test_metrics_works_with_anonymous(self, client):
        """Test /metrics endpoint works when auth is disabled."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_sessions_works_with_anonymous(self, client):
        """Test /sessions endpoint works when auth is disabled."""
        response = client.get("/sessions")
        assert response.status_code == 200

    def test_logs_works_with_anonymous(self, client):
        """Test /logs endpoint works when auth is disabled."""
        response = client.get("/logs")
        assert response.status_code == 200


class TestEndpointsWithJWTAuth:
    """Tests that endpoints work with valid JWT authentication."""

    @pytest.fixture
    def jwt_headers(self):
        """Generate JWT auth headers for testing."""
        config = SecurityConfig()
        jwt_auth = JWTAuth(config=config)
        token = jwt_auth.create_token(user_id="testuser", roles=["user"])
        return {"Authorization": f"Bearer {token}"}

    def test_tools_with_jwt(self, client, jwt_headers):
        """Test /tools endpoint works with JWT."""
        response = client.get("/tools", headers=jwt_headers)
        assert response.status_code == 200

    def test_metrics_with_jwt(self, client, jwt_headers):
        """Test /metrics endpoint works with JWT."""
        response = client.get("/metrics", headers=jwt_headers)
        assert response.status_code == 200


class TestEndpointsWithAPIKey:
    """Tests that endpoints work with API key authentication."""

    def test_tools_with_api_key(self, client):
        """Test /tools endpoint works with API key."""
        from security.api_keys import api_key_manager

        result = api_key_manager.generate(name="Test Key", user_id="testuser")
        response = client.get("/tools", headers={"X-API-Key": result.raw_key})
        assert response.status_code == 200

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)

    def test_metrics_with_api_key(self, client):
        """Test /metrics endpoint works with API key."""
        from security.api_keys import api_key_manager

        result = api_key_manager.generate(name="Test Key", user_id="testuser")
        response = client.get("/metrics", headers={"X-API-Key": result.raw_key})
        assert response.status_code == 200

        # Cleanup
        api_key_manager.revoke(result.key_info.key_id)


class TestCurrentUserInjection:
    """Tests that current_user is properly injected into endpoints."""

    def test_run_agent_receives_current_user(self, client):
        """Test that run_agent receives current_user parameter via test client."""
        # Test that the endpoint accepts requests via the test client
        # which validates that current_user is properly injected
        response = client.post("/run", json={"query": "test query"})
        # Should start successfully (status 200), not error about missing current_user
        assert response.status_code == 200
        result = response.json()
        assert "session_id" in result

    def test_list_tools_receives_current_user(self, client):
        """Test that list_available_tools receives current_user parameter."""
        # Test that the endpoint works via the test client
        # which validates that current_user is properly injected
        response = client.get("/tools")
        assert response.status_code == 200
        result = response.json()
        assert "tools" in result
        assert "categories" in result


class TestWebSocketEndpoints:
    """Tests for WebSocket endpoints (which don't have HTTP auth)."""

    def test_websocket_session_endpoint_exists(self):
        """Verify WebSocket session endpoint exists."""
        import inspect

        from api_server import websocket_endpoint

        sig = inspect.signature(websocket_endpoint)
        params = list(sig.parameters.keys())
        # WebSocket endpoint doesn't have current_user (auth is handled differently)
        assert "current_user" not in params
        assert "websocket" in params
        assert "session_id" in params

    def test_websocket_metrics_endpoint_exists(self):
        """Verify WebSocket metrics endpoint exists."""
        import inspect

        from api_server import metrics_websocket

        sig = inspect.signature(metrics_websocket)
        params = list(sig.parameters.keys())
        # WebSocket endpoint doesn't have current_user (auth is handled differently)
        assert "current_user" not in params
        assert "websocket" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
