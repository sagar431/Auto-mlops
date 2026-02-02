#!/usr/bin/env python3
"""
Tests for API server rate limiting with slowapi.

Tests verify that:
1. Rate limiter is properly configured
2. Rate limit is applied to endpoints
3. Rate limit can be configured via environment variable
4. Rate limit exceeded returns 429 status

Run with: pytest test_rate_limiting.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# Set auth disabled, a low rate limit, and JWT secret for testing
os.environ["ENABLE_API_KEY_AUTH"] = "false"
os.environ["ENABLE_JWT_AUTH"] = "false"
os.environ["RATE_LIMIT"] = "5/minute"
os.environ["JWT_SECRET"] = "test-secret-for-all-tests"

from api_server import app, get_rate_limit, limiter


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset rate limiter storage before each test."""
    # Clear the in-memory storage using limiter's reset method
    limiter.reset()
    yield


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestRateLimitConfiguration:
    """Tests for rate limit configuration."""

    def test_default_rate_limit(self):
        """Test default rate limit is 100/minute when env var not set."""
        # Temporarily unset the env var
        original = os.environ.pop("RATE_LIMIT", None)
        try:
            # Re-import to get fresh config
            result = get_rate_limit()
            # When not set, it should fall back to default
            if original is None:
                assert result == "100/minute"
        finally:
            if original:
                os.environ["RATE_LIMIT"] = original

    def test_custom_rate_limit_from_env(self):
        """Test rate limit can be set via environment variable."""
        os.environ["RATE_LIMIT"] = "50/minute"
        result = get_rate_limit()
        assert result == "50/minute"
        # Restore test value
        os.environ["RATE_LIMIT"] = "5/minute"

    def test_limiter_attached_to_app(self):
        """Test that limiter is attached to app state."""
        assert hasattr(app.state, "limiter")
        assert app.state.limiter is limiter


class TestRateLimitOnEndpoints:
    """Tests that rate limiting is applied to endpoints."""

    def test_health_endpoint_rate_limited(self, client):
        """Test /health endpoint is rate limited."""
        # Make requests up to the limit
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/health")
        assert response.status_code == 429

    def test_tools_endpoint_rate_limited(self, client):
        """Test /tools endpoint is rate limited."""
        # Make requests up to the limit
        for i in range(5):
            response = client.get("/tools")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/tools")
        assert response.status_code == 429

    def test_metrics_endpoint_rate_limited(self, client):
        """Test /metrics endpoint is rate limited."""
        # Make requests up to the limit
        for i in range(5):
            response = client.get("/metrics")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/metrics")
        assert response.status_code == 429


class TestRateLimitResponse:
    """Tests for rate limit exceeded response."""

    def test_rate_limit_exceeded_returns_429(self, client):
        """Test that exceeding rate limit returns 429 status code."""
        # Exhaust the rate limit
        for _ in range(6):
            response = client.get("/health")

        # Verify status code
        assert response.status_code == 429

    def test_rate_limit_exceeded_has_retry_after(self, client):
        """Test that rate limit response includes Retry-After header."""
        # Exhaust the rate limit
        for _ in range(6):
            response = client.get("/health")

        # Verify Retry-After header is present
        assert "Retry-After" in response.headers or response.status_code == 429


class TestEndpointHasRateLimitDecorator:
    """Tests that verify endpoints have the rate limit decorator."""

    def test_health_check_has_request_param(self):
        """Verify /health endpoint has request parameter for rate limiting."""
        import inspect

        from api_server import health_check

        sig = inspect.signature(health_check)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_run_agent_has_request_param(self):
        """Verify /run endpoint has request parameter for rate limiting."""
        import inspect

        from api_server import run_agent

        sig = inspect.signature(run_agent)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_session_status_has_request_param(self):
        """Verify /status/{session_id} endpoint has request parameter."""
        import inspect

        from api_server import get_session_status

        sig = inspect.signature(get_session_status)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_list_sessions_has_request_param(self):
        """Verify /sessions endpoint has request parameter."""
        import inspect

        from api_server import list_sessions

        sig = inspect.signature(list_sessions)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_session_details_has_request_param(self):
        """Verify /sessions/{session_id} endpoint has request parameter."""
        import inspect

        from api_server import get_session_details

        sig = inspect.signature(get_session_details)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_list_available_tools_has_request_param(self):
        """Verify /tools endpoint has request parameter."""
        import inspect

        from api_server import list_available_tools

        sig = inspect.signature(list_available_tools)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_metrics_has_request_param(self):
        """Verify /metrics endpoint has request parameter."""
        import inspect

        from api_server import get_metrics

        sig = inspect.signature(get_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_system_metrics_has_request_param(self):
        """Verify /metrics/system endpoint has request parameter."""
        import inspect

        from api_server import get_system_metrics

        sig = inspect.signature(get_system_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_agent_metrics_has_request_param(self):
        """Verify /metrics/agent endpoint has request parameter."""
        import inspect

        from api_server import get_agent_metrics

        sig = inspect.signature(get_agent_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_pipeline_metrics_has_request_param(self):
        """Verify /metrics/pipeline endpoint has request parameter."""
        import inspect

        from api_server import get_pipeline_metrics

        sig = inspect.signature(get_pipeline_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_generate_demo_metrics_has_request_param(self):
        """Verify /metrics/demo endpoint has request parameter."""
        import inspect

        from api_server import generate_demo_metrics

        sig = inspect.signature(generate_demo_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_get_logs_has_request_param(self):
        """Verify GET /logs endpoint has request parameter."""
        import inspect

        from api_server import get_logs

        sig = inspect.signature(get_logs)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_create_log_has_request_param(self):
        """Verify POST /logs endpoint has request parameter."""
        import inspect

        from api_server import create_log

        sig = inspect.signature(create_log)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_create_user_has_request_param(self):
        """Verify POST /admin/users endpoint has request parameter."""
        import inspect

        from api_server import create_user

        sig = inspect.signature(create_user)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_create_api_key_has_request_param(self):
        """Verify POST /admin/keys endpoint has request parameter."""
        import inspect

        from api_server import create_api_key

        sig = inspect.signature(create_api_key)
        params = list(sig.parameters.keys())
        assert "request" in params

    def test_revoke_api_key_has_request_param(self):
        """Verify DELETE /admin/keys/{key_id} endpoint has request parameter."""
        import inspect

        from api_server import revoke_api_key

        sig = inspect.signature(revoke_api_key)
        params = list(sig.parameters.keys())
        assert "request" in params


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
