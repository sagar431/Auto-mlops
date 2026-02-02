"""
Locust load testing for Auto-MLOps API.

Usage:
    # Run with web UI (default)
    locust -f tests/load/locustfile.py --host=http://localhost:8000

    # Headless mode with specific parameters
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           --headless -u 100 -r 10 -t 60s

    # Run specific user class
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \
           -u 50 -r 5 MLOpsAPIUser

Environment Variables:
    LOAD_TEST_API_KEY: API key for authenticated requests
    LOAD_TEST_ADMIN_KEY: Admin API key for admin endpoints
"""

import os
import random
import string
import uuid
from typing import Any

from locust import HttpUser, between, events, task
from locust.runners import MasterRunner

# Configuration
API_KEY = os.environ.get("LOAD_TEST_API_KEY", "test-api-key-load")
ADMIN_API_KEY = os.environ.get("LOAD_TEST_ADMIN_KEY", "test-admin-key-load")


def random_string(length: int = 8) -> str:
    """Generate a random string for test data."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


class MLOpsAPIUser(HttpUser):
    """
    Simulates a typical MLOps user interacting with the API.

    This user class performs common operations like:
    - Health checks
    - Starting agent sessions
    - Checking session status
    - Viewing available tools
    - Fetching logs and metrics
    """

    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    # Track session IDs created by this user for status polling
    session_ids: list[str] = []

    def on_start(self) -> None:
        """Initialize user session state."""
        self.session_ids = []
        self.headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

    @task(10)
    def health_check(self) -> None:
        """High-frequency health check - baseline endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data}")
            else:
                response.failure(f"Status {response.status_code}")

    @task(5)
    def list_tools(self) -> None:
        """Fetch available MLOps tools."""
        with self.client.get("/tools", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "tools" in data and len(data["tools"]) > 0:
                    response.success()
                else:
                    response.failure("No tools returned")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def start_session(self) -> None:
        """Start a new agent session with a simple query."""
        queries = [
            "Set up MLOps pipeline for my project",
            "Initialize DVC for data versioning",
            "Create Hydra configuration",
            "Set up MLflow tracking",
            "Create GitHub Actions workflow",
        ]
        payload = {
            "query": random.choice(queries),
            "project_path": f"/tmp/test-project-{random_string()}",
            "accuracy_threshold": 0.85,
        }

        with self.client.post(
            "/run", json=payload, headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "session_id" in data:
                    self.session_ids.append(data["session_id"])
                    # Keep only last 10 session IDs to avoid memory growth
                    if len(self.session_ids) > 10:
                        self.session_ids = self.session_ids[-10:]
                    response.success()
                else:
                    response.failure("No session_id in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")

    @task(8)
    def check_session_status(self) -> None:
        """Poll status of an existing session."""
        if not self.session_ids:
            # No sessions yet, skip this task
            return

        session_id = random.choice(self.session_ids)
        with self.client.get(
            f"/status/{session_id}",
            headers=self.headers,
            catch_response=True,
            name="/status/{session_id}",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "status" in data:
                    response.success()
                else:
                    response.failure("No status in response")
            elif response.status_code == 404:
                # Session not found - remove from our list
                if session_id in self.session_ids:
                    self.session_ids.remove(session_id)
                response.success()  # Expected behavior for old sessions
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(4)
    def list_sessions(self) -> None:
        """List past sessions with pagination."""
        limit = random.choice([5, 10, 20])
        with self.client.get(
            f"/sessions?limit={limit}",
            headers=self.headers,
            catch_response=True,
            name="/sessions",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "sessions" in data:
                    response.success()
                else:
                    response.failure("No sessions key in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def get_session_details(self) -> None:
        """Get detailed information about a specific session."""
        if not self.session_ids:
            return

        session_id = random.choice(self.session_ids)
        with self.client.get(
            f"/sessions/{session_id}",
            headers=self.headers,
            catch_response=True,
            name="/sessions/{session_id}",
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404:
                if session_id in self.session_ids:
                    self.session_ids.remove(session_id)
                response.success()
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def fetch_logs(self) -> None:
        """Fetch execution logs with various filters."""
        params: dict[str, Any] = {"page": 1, "page_size": random.choice([10, 20, 50])}

        # Randomly add filters
        if random.random() > 0.5:
            params["level"] = random.choice(["INFO", "WARNING", "ERROR"])

        with self.client.get(
            "/logs",
            params=params,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "logs" in data and "pagination" in data:
                    response.success()
                else:
                    response.failure("Invalid logs response structure")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")


class MetricsUser(HttpUser):
    """
    Simulates monitoring systems polling metrics endpoints.

    These users focus on metrics and monitoring endpoints,
    simulating Prometheus scrapers and monitoring dashboards.
    """

    wait_time = between(5, 15)

    def on_start(self) -> None:
        """Initialize headers."""
        self.headers = {"X-API-Key": API_KEY}

    @task(5)
    def get_metrics_summary(self) -> None:
        """Fetch complete metrics summary."""
        with self.client.get("/metrics", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if all(k in data for k in ["system_metrics", "agent_metrics", "timestamp"]):
                    response.success()
                else:
                    response.failure("Incomplete metrics response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def get_system_metrics(self) -> None:
        """Fetch system resource metrics."""
        with self.client.get(
            "/metrics/system", headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "cpu_percent" in data and "memory" in data:
                    response.success()
                else:
                    response.failure("Missing system metrics fields")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def get_agent_metrics(self) -> None:
        """Fetch agent performance metrics."""
        with self.client.get(
            "/metrics/agent", headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(3)
    def get_pipeline_metrics(self) -> None:
        """Fetch pipeline and tool usage metrics."""
        with self.client.get(
            "/metrics/pipeline", headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def get_prometheus_metrics(self) -> None:
        """Fetch Prometheus-formatted metrics."""
        with self.client.get(
            "/metrics/prometheus", headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                # Prometheus metrics are plain text
                if response.text and "#" in response.text:
                    response.success()
                else:
                    response.failure("Invalid Prometheus format")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            else:
                response.failure(f"Status {response.status_code}")


class AdminUser(HttpUser):
    """
    Simulates admin users performing administrative tasks.

    Lower weight as admin operations are less frequent.
    """

    wait_time = between(10, 30)
    weight = 1  # Lower weight than regular users

    def on_start(self) -> None:
        """Initialize admin headers."""
        self.headers = {
            "X-API-Key": ADMIN_API_KEY,
            "Content-Type": "application/json",
        }
        self.created_user_ids: list[str] = []
        self.created_key_ids: list[str] = []

    @task(2)
    def list_users(self) -> None:
        """List all users."""
        with self.client.get("/admin/users", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "users" in data:
                    response.success()
                else:
                    response.failure("No users key in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def create_user(self) -> None:
        """Create a new user."""
        payload = {
            "username": f"loadtest_user_{random_string()}",
            "email": f"loadtest_{random_string()}@example.com",
            "role": random.choice(["user", "admin"]),
        }

        with self.client.post(
            "/admin/users",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                data = response.json()
                if "user_id" in data or "id" in data:
                    user_id = data.get("user_id") or data.get("id")
                    self.created_user_ids.append(user_id)
                    if len(self.created_user_ids) > 5:
                        self.created_user_ids = self.created_user_ids[-5:]
                    response.success()
                else:
                    response.failure("No user_id in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            elif response.status_code == 409:
                response.success()  # User already exists is acceptable
            else:
                response.failure(f"Status {response.status_code}")

    @task(2)
    def list_api_keys(self) -> None:
        """List API keys."""
        params = {}
        if random.random() > 0.5:
            params["include_revoked"] = random.choice(["true", "false"])

        with self.client.get(
            "/admin/keys",
            params=params,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "keys" in data or "api_keys" in data:
                    response.success()
                else:
                    response.failure("No keys in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"Status {response.status_code}")

    @task(1)
    def create_api_key(self) -> None:
        """Create a new API key."""
        payload = {
            "name": f"loadtest_key_{random_string()}",
            "user_id": (
                random.choice(self.created_user_ids) if self.created_user_ids else str(uuid.uuid4())
            ),
        }

        with self.client.post(
            "/admin/keys",
            json=payload,
            headers=self.headers,
            catch_response=True,
        ) as response:
            if response.status_code in (200, 201):
                data = response.json()
                if "key_id" in data or "id" in data:
                    key_id = data.get("key_id") or data.get("id")
                    self.created_key_ids.append(key_id)
                    if len(self.created_key_ids) > 5:
                        self.created_key_ids = self.created_key_ids[-5:]
                    response.success()
                else:
                    response.failure("No key_id in response")
            elif response.status_code == 401:
                response.failure("Authentication failed")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"Status {response.status_code}")


class HighFrequencyPollingUser(HttpUser):
    """
    Simulates high-frequency polling for real-time updates.

    This represents dashboard users or CI/CD systems
    that poll frequently for status updates.
    """

    wait_time = between(0.5, 1.5)
    weight = 2

    def on_start(self) -> None:
        """Initialize user state."""
        self.headers = {"X-API-Key": API_KEY}
        self.session_ids: list[str] = []

    @task(10)
    def poll_health(self) -> None:
        """Rapid health polling."""
        self.client.get("/health")

    @task(5)
    def poll_session_status(self) -> None:
        """Rapid status polling for active sessions."""
        if not self.session_ids:
            # Try to get a session from /sessions endpoint
            response = self.client.get(
                "/sessions?limit=5", headers=self.headers, catch_response=True
            )
            if response.status_code == 200:
                data = response.json()
                sessions = data.get("sessions", [])
                self.session_ids = [
                    s.get("session_id") or s.get("id")
                    for s in sessions
                    if s.get("session_id") or s.get("id")
                ][:5]
            return

        session_id = random.choice(self.session_ids)
        with self.client.get(
            f"/status/{session_id}",
            headers=self.headers,
            catch_response=True,
            name="/status/{session_id}",
        ) as response:
            if response.status_code in (200, 404):
                response.success()
            else:
                response.failure(f"Status {response.status_code}")


# Event hooks for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs) -> None:
    """Log test start information."""
    if isinstance(environment.runner, MasterRunner):
        print("Load test starting on master node")
    print(f"Target host: {environment.host}")
    print(f"User classes: {[uc.__name__ for uc in environment.user_classes]}")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs) -> None:
    """Log test completion summary."""
    stats = environment.stats
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)
    print(f"Total requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    if stats.total.num_requests > 0:
        failure_rate = (stats.total.num_failures / stats.total.num_requests) * 100
        print(f"Failure rate: {failure_rate:.2f}%")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Median response time: {stats.total.median_response_time}ms")
    if stats.total.get_response_time_percentile(0.95):
        print(f"95th percentile: {stats.total.get_response_time_percentile(0.95)}ms")
    if stats.total.get_response_time_percentile(0.99):
        print(f"99th percentile: {stats.total.get_response_time_percentile(0.99)}ms")
    print("=" * 60)
