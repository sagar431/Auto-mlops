#!/usr/bin/env python3
"""
Tests for the /metrics endpoints.

Tests verify that:
1. /metrics endpoint returns complete MetricsSummary
2. /metrics/system returns SystemMetrics
3. /metrics/agent returns AgentMetrics
4. /metrics/pipeline returns PipelineMetrics
5. /metrics/demo generates demo data
6. Response models have correct structure and types

Run with: pytest tests/root_migrated/test_metrics_endpoints.py -v
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


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Test /metrics endpoint returns 200 status."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_json(self, client):
        """Test /metrics returns JSON content type."""
        response = client.get("/metrics")
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "application/json" in content_type

    def test_metrics_has_timestamp(self, client):
        """Test /metrics response has timestamp."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_metrics_has_system_section(self, client):
        """Test /metrics response has system metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        system = data["system"]
        assert "cpu_percent" in system
        assert "memory_percent" in system
        assert "disk_percent" in system

    def test_metrics_has_agent_section(self, client):
        """Test /metrics response has agent metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "agent" in data
        agent = data["agent"]
        assert "total_sessions" in agent
        assert "active_sessions" in agent
        assert "success_rate" in agent

    def test_metrics_has_pipeline_section(self, client):
        """Test /metrics response has pipeline metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "pipeline" in data
        pipeline = data["pipeline"]
        assert "total_pipelines_run" in pipeline
        assert "tools_available" in pipeline

    def test_metrics_has_accuracy_section(self, client):
        """Test /metrics response has accuracy metrics."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "accuracy" in data
        accuracy = data["accuracy"]
        assert "experiments_tracked" in accuracy
        assert "total_training_runs" in accuracy

    def test_metrics_has_history_arrays(self, client):
        """Test /metrics response has time series history arrays."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_history" in data
        assert "memory_history" in data
        assert "sessions_history" in data
        assert "accuracy_history" in data
        assert isinstance(data["cpu_history"], list)
        assert isinstance(data["memory_history"], list)


class TestSystemMetricsEndpoint:
    """Tests for /metrics/system endpoint."""

    def test_system_metrics_returns_200(self, client):
        """Test /metrics/system endpoint returns 200 status."""
        response = client.get("/metrics/system")
        assert response.status_code == 200

    def test_system_metrics_has_cpu_percent(self, client):
        """Test /metrics/system has cpu_percent field."""
        response = client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "cpu_percent" in data
        assert isinstance(data["cpu_percent"], (int, float))
        assert 0 <= data["cpu_percent"] <= 100

    def test_system_metrics_has_memory_fields(self, client):
        """Test /metrics/system has memory fields."""
        response = client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "memory_percent" in data
        assert "memory_used_gb" in data
        assert "memory_total_gb" in data
        assert isinstance(data["memory_percent"], (int, float))

    def test_system_metrics_has_disk_fields(self, client):
        """Test /metrics/system has disk fields."""
        response = client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "disk_percent" in data
        assert "disk_used_gb" in data
        assert "disk_total_gb" in data

    def test_system_metrics_has_platform_info(self, client):
        """Test /metrics/system has platform info."""
        response = client.get("/metrics/system")
        assert response.status_code == 200
        data = response.json()
        assert "python_version" in data
        assert "platform" in data
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))


class TestAgentMetricsEndpoint:
    """Tests for /metrics/agent endpoint."""

    def test_agent_metrics_returns_200(self, client):
        """Test /metrics/agent endpoint returns 200 status."""
        response = client.get("/metrics/agent")
        assert response.status_code == 200

    def test_agent_metrics_has_session_counts(self, client):
        """Test /metrics/agent has session count fields."""
        response = client.get("/metrics/agent")
        assert response.status_code == 200
        data = response.json()
        assert "total_sessions" in data
        assert "active_sessions" in data
        assert "successful_sessions" in data
        assert "failed_sessions" in data
        assert isinstance(data["total_sessions"], int)

    def test_agent_metrics_has_success_rate(self, client):
        """Test /metrics/agent has success rate field."""
        response = client.get("/metrics/agent")
        assert response.status_code == 200
        data = response.json()
        assert "success_rate" in data
        assert isinstance(data["success_rate"], (int, float))
        assert 0 <= data["success_rate"] <= 100

    def test_agent_metrics_has_execution_fields(self, client):
        """Test /metrics/agent has execution time and step fields."""
        response = client.get("/metrics/agent")
        assert response.status_code == 200
        data = response.json()
        assert "avg_execution_time_seconds" in data
        assert "total_steps_executed" in data
        assert "avg_steps_per_session" in data


class TestPipelineMetricsEndpoint:
    """Tests for /metrics/pipeline endpoint."""

    def test_pipeline_metrics_returns_200(self, client):
        """Test /metrics/pipeline endpoint returns 200 status."""
        response = client.get("/metrics/pipeline")
        assert response.status_code == 200

    def test_pipeline_metrics_has_pipeline_counts(self, client):
        """Test /metrics/pipeline has pipeline count fields."""
        response = client.get("/metrics/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert "total_pipelines_run" in data
        assert "pipelines_in_progress" in data
        assert "completed_pipelines" in data
        assert "failed_pipelines" in data

    def test_pipeline_metrics_has_tool_info(self, client):
        """Test /metrics/pipeline has tool information."""
        response = client.get("/metrics/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert "tools_available" in data
        assert "tool_invocations" in data
        assert "most_used_tools" in data
        assert isinstance(data["most_used_tools"], list)

    def test_pipeline_metrics_has_duration(self, client):
        """Test /metrics/pipeline has duration field."""
        response = client.get("/metrics/pipeline")
        assert response.status_code == 200
        data = response.json()
        assert "avg_pipeline_duration_seconds" in data


class TestDemoMetricsEndpoint:
    """Tests for /metrics/demo endpoint."""

    def test_demo_metrics_returns_200(self, client):
        """Test /metrics/demo endpoint returns 200 status."""
        response = client.get("/metrics/demo")
        assert response.status_code == 200

    def test_demo_metrics_returns_ok_status(self, client):
        """Test /metrics/demo returns ok status."""
        response = client.get("/metrics/demo")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "message" in data

    def test_demo_metrics_generates_data(self, client):
        """Test /metrics/demo actually generates demo data."""
        # First call demo endpoint
        response = client.get("/metrics/demo")
        assert response.status_code == 200

        # Then check main metrics endpoint has data
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        # After demo, should have some session history
        agent = data["agent"]
        assert agent["total_sessions"] > 0


class TestMetricsDataTypes:
    """Tests for metrics response data types."""

    def test_system_metrics_numeric_types(self, client):
        """Test system metrics have correct numeric types."""
        response = client.get("/metrics/system")
        data = response.json()

        # All percentage fields should be floats/ints between 0-100
        for field in ["cpu_percent", "memory_percent", "disk_percent"]:
            assert isinstance(data[field], (int, float))

        # GB fields should be non-negative
        for field in ["memory_used_gb", "memory_total_gb", "disk_used_gb", "disk_total_gb"]:
            assert isinstance(data[field], (int, float))
            assert data[field] >= 0

    def test_agent_metrics_numeric_types(self, client):
        """Test agent metrics have correct numeric types."""
        response = client.get("/metrics/agent")
        data = response.json()

        # Session counts should be integers
        for field in [
            "total_sessions",
            "active_sessions",
            "successful_sessions",
            "failed_sessions",
        ]:
            assert isinstance(data[field], int)
            assert data[field] >= 0

        # Rates and averages should be floats
        assert isinstance(data["success_rate"], (int, float))
        assert isinstance(data["avg_execution_time_seconds"], (int, float))

    def test_pipeline_metrics_tool_stats_structure(self, client):
        """Test most_used_tools has correct structure."""
        # Generate demo data first
        client.get("/metrics/demo")

        response = client.get("/metrics/pipeline")
        data = response.json()

        most_used = data["most_used_tools"]
        assert isinstance(most_used, list)

        if len(most_used) > 0:
            tool = most_used[0]
            assert "tool_name" in tool
            assert "invocations" in tool
            assert "success_count" in tool
            assert "failure_count" in tool
            assert "avg_duration_ms" in tool


class TestMetricsRateLimiting:
    """Tests for rate limiting on metrics endpoints."""

    def test_metrics_endpoint_is_rate_limited(self):
        """Test /metrics endpoint has rate limit."""
        # Set a low rate limit for this test
        os.environ["RATE_LIMIT"] = "5/minute"
        limiter.reset()

        client = TestClient(app)

        # Make requests up to the limit
        for i in range(5):
            response = client.get("/metrics")
            assert response.status_code == 200, f"Request {i+1} should succeed"

        # Next request should be rate limited
        response = client.get("/metrics")
        assert response.status_code == 429

        # Restore rate limit
        os.environ["RATE_LIMIT"] = "1000/minute"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
