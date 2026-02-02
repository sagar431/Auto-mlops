#!/usr/bin/env python3
"""
Tests for the Prometheus metrics endpoint.

Tests verify that:
1. /metrics/prometheus endpoint returns Prometheus text format
2. Endpoint includes expected metric types and names
3. Endpoint is rate limited
4. Endpoint works without authentication (designed for Prometheus scraping)

Run with: pytest test_prometheus_metrics.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

# Set auth disabled and high rate limit for testing
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
    """Create a test client."""
    return TestClient(app)


class TestPrometheusEndpoint:
    """Tests for /metrics/prometheus endpoint."""

    def test_prometheus_endpoint_returns_200(self, client):
        """Test /metrics/prometheus endpoint returns 200 status."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200

    def test_prometheus_endpoint_returns_text_plain(self, client):
        """Test /metrics/prometheus returns text/plain content type."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_prometheus_endpoint_contains_help_lines(self, client):
        """Test /metrics/prometheus output contains HELP lines."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        assert "# HELP" in content

    def test_prometheus_endpoint_contains_type_lines(self, client):
        """Test /metrics/prometheus output contains TYPE lines."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        assert "# TYPE" in content

    def test_prometheus_endpoint_contains_mlops_metrics(self, client):
        """Test /metrics/prometheus output contains MLOps-specific metrics."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        # Should contain MLOps agent metrics defined in observability/metrics.py
        assert "mlops_" in content

    def test_prometheus_endpoint_contains_sessions_metric(self, client):
        """Test /metrics/prometheus includes sessions metrics."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        assert "mlops_sessions_total" in content or "mlops_sessions_active" in content

    def test_prometheus_endpoint_contains_tool_metrics(self, client):
        """Test /metrics/prometheus includes tool invocation metrics."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        assert "mlops_tool_invocations_total" in content or "mlops_tool_duration_seconds" in content


class TestPrometheusFormatCompliance:
    """Tests for Prometheus text format compliance."""

    def test_prometheus_output_ends_with_newline(self, client):
        """Test Prometheus output ends with newline as per spec."""
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200
        content = response.text
        assert content.endswith("\n")

    def test_prometheus_help_format(self, client):
        """Test HELP lines follow Prometheus format."""
        response = client.get("/metrics/prometheus")
        content = response.text
        # HELP lines should be: # HELP metric_name description
        lines = [line for line in content.split("\n") if line.startswith("# HELP")]
        assert len(lines) > 0
        for line in lines:
            parts = line.split(" ", 3)
            assert len(parts) >= 3, f"Invalid HELP format: {line}"
            assert parts[0] == "#"
            assert parts[1] == "HELP"

    def test_prometheus_type_format(self, client):
        """Test TYPE lines follow Prometheus format."""
        response = client.get("/metrics/prometheus")
        content = response.text
        # TYPE lines should be: # TYPE metric_name type
        lines = [line for line in content.split("\n") if line.startswith("# TYPE")]
        assert len(lines) > 0
        for line in lines:
            parts = line.split()
            assert len(parts) >= 4, f"Invalid TYPE format: {line}"
            assert parts[0] == "#"
            assert parts[1] == "TYPE"
            # Type should be one of: counter, gauge, histogram, summary
            assert parts[3] in [
                "counter",
                "gauge",
                "histogram",
                "summary",
            ], f"Invalid metric type: {parts[3]}"


class TestPrometheusEndpointNoAuth:
    """Tests that Prometheus endpoint works without authentication."""

    def test_prometheus_no_auth_required(self, client):
        """Test /metrics/prometheus works without authentication headers."""
        # Don't send any auth headers
        response = client.get("/metrics/prometheus")
        assert response.status_code == 200

    def test_prometheus_no_current_user_dependency(self):
        """Verify /metrics/prometheus endpoint has no current_user dependency."""
        import inspect

        from api_server import get_prometheus_metrics

        sig = inspect.signature(get_prometheus_metrics)
        params = list(sig.parameters.keys())
        # Should only have 'request' parameter for rate limiting, not current_user
        assert "current_user" not in params

    def test_prometheus_has_request_param(self):
        """Verify /metrics/prometheus endpoint has request parameter for rate limiting."""
        import inspect

        from api_server import get_prometheus_metrics

        sig = inspect.signature(get_prometheus_metrics)
        params = list(sig.parameters.keys())
        assert "request" in params


class TestPrometheusRateLimiting:
    """Tests for rate limiting on Prometheus endpoint."""

    def test_prometheus_endpoint_is_rate_limited(self):
        """Test /metrics/prometheus endpoint has rate limit decorator."""
        # Set a low rate limit for this test
        os.environ["RATE_LIMIT"] = "5/minute"
        limiter.reset()

        client = TestClient(app)

        # Make requests up to the limit
        for i in range(5):
            response = client.get("/metrics/prometheus")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/metrics/prometheus")
        assert response.status_code == 429

        # Restore rate limit
        os.environ["RATE_LIMIT"] = "1000/minute"
