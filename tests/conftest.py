#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for Auto-MLOps tests.

This module provides common fixtures used across test modules including:
- Database setup/teardown
- Mock LLM responses
- Context manager instances
- Session fixtures
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

# ============================================================================
# Async Event Loop Configuration
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database file path."""
    return str(tmp_path / "test_mlops.db")


@pytest.fixture
def temp_db_url(temp_db_path):
    """Create a temporary database URL."""
    return f"sqlite:///{temp_db_path}"


# ============================================================================
# Context Manager Fixtures
# ============================================================================


@pytest.fixture
def context_manager():
    """Create a fresh ContextManager instance for testing."""
    from agent.contextManager import ContextManager

    return ContextManager(
        session_id="test-session-123",
        original_query="Test MLOps pipeline setup",
        project_path="/test/project",
    )


@pytest.fixture
def experiment_state():
    """Create a fresh ExperimentState instance for testing."""
    from agent.contextManager import ExperimentState

    return ExperimentState()


@pytest.fixture
def mlops_step_node():
    """Create a sample MLOpsStepNode for testing."""
    from agent.contextManager import MLOpsStepNode

    return MLOpsStepNode(
        index="0",
        description="Initialize MLflow experiment",
        type="CODE",
        tool="init_mlflow_experiment",
        args={"experiment_name": "test_experiment"},
    )


# ============================================================================
# Mock LLM Fixtures
# ============================================================================


@pytest.fixture
def mock_model_manager():
    """Create a mock ModelManager that returns predefined responses."""
    mock = MagicMock()
    mock.generate_json = AsyncMock()
    mock.generate_text = AsyncMock()
    return mock


@pytest.fixture
def mock_perception_response():
    """Return a typical perception module response."""
    return {
        "entities": {
            "project_path": "/test/project",
            "experiment_name": "test_experiment",
            "accuracy_threshold": 0.85,
        },
        "pipeline_stage": "setup",
        "required_tools": ["analyze_project_config", "create_hydra_config"],
        "result_requirement": "Set up MLOps infrastructure",
        "original_goal_achieved": False,
        "local_goal_achieved": False,
        "missing_requirements": [],
        "route": "decision",
        "confidence": 0.9,
        "reasoning": "Project needs initial setup",
    }


@pytest.fixture
def mock_decision_response():
    """Return a typical decision module response."""
    return {
        "strategy": "sequential",
        "reasoning": "Execute setup steps in order",
        "plan_graph": {
            "nodes": [
                {
                    "id": "0",
                    "description": "Analyze project configuration",
                    "tool": "analyze_project_config",
                    "args": {"project_path": "/test/project"},
                    "depends_on": [],
                },
                {
                    "id": "1",
                    "description": "Create Hydra configuration",
                    "tool": "create_hydra_config",
                    "args": {"project_path": "/test/project"},
                    "depends_on": ["0"],
                },
            ]
        },
        "next_step_id": "0",
        "code_variants": {},
    }


# ============================================================================
# Session Fixtures
# ============================================================================


@pytest.fixture
def mock_session():
    """Create a mock AgentSession for testing."""
    mock = MagicMock()
    mock.session_id = "test-session-123"
    mock.add_message = MagicMock()
    mock.add_tool_call = MagicMock()
    mock.create_snapshot = MagicMock(return_value=0)
    mock.messages = []
    return mock


# ============================================================================
# Tool Result Fixtures
# ============================================================================


@pytest.fixture
def successful_tool_result():
    """Return a successful tool execution result."""
    return {
        "success": True,
        "config_path": "/test/project/config/config.yaml",
        "message": "Configuration created successfully",
    }


@pytest.fixture
def failed_tool_result():
    """Return a failed tool execution result."""
    return {
        "success": False,
        "error": "Permission denied: /test/project/config",
    }


@pytest.fixture
def mlflow_init_result():
    """Return a successful MLflow initialization result."""
    return {
        "success": True,
        "experiment_id": "exp-123",
        "experiment_name": "test_experiment",
        "artifact_location": "/mlruns/exp-123",
    }


@pytest.fixture
def mlflow_run_result():
    """Return a successful MLflow run result."""
    return {
        "success": True,
        "run_id": "run-456",
        "run_name": "test_run",
        "experiment_id": "exp-123",
    }


@pytest.fixture
def training_result():
    """Return a training completion result with metrics."""
    return {
        "success": True,
        "current_value": 0.82,
        "metrics": {
            "accuracy": 0.82,
            "loss": 0.45,
            "f1_score": 0.80,
        },
    }


# ============================================================================
# Prompt Template Fixtures
# ============================================================================


@pytest.fixture
def perception_prompt_template():
    """Return a minimal perception prompt template for testing."""
    return """Analyze the MLOps context and provide JSON output.

Query: {query}
Stage: {stage}
Experiment State: {experiment_state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Available Tools: {tools}
Memory: {memory}
Previous Steps: {previous_steps}

Output JSON with: entities, pipeline_stage, required_tools, route, confidence, reasoning
"""


@pytest.fixture
def decision_prompt_template():
    """Return a minimal decision prompt template for testing."""
    return """Generate an execution plan for the MLOps task.

Query: {query}
Perception: {perception}
State: {state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Experiment State: {experiment_state}

Output JSON with: strategy, plan_graph (nodes with id, description, tool, args, depends_on), next_step_id
"""


# ============================================================================
# Test Project Fixture
# ============================================================================


@pytest.fixture
def test_project(tmp_path):
    """
    Create a temporary test project directory with basic ML project structure.

    Returns:
        Path to the temporary project directory
    """
    project_dir = tmp_path / "test_ml_project"
    project_dir.mkdir()

    # Create basic project structure
    (project_dir / "config").mkdir()
    (project_dir / "data").mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "models").mkdir()

    # Create a basic config file
    config_content = """
defaults:
  - _self_

experiment:
  name: test_experiment
  seed: 42

model:
  type: mlp
  hidden_dim: 128
  dropout: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
"""
    (project_dir / "config" / "config.yaml").write_text(config_content)

    # Create a basic training script placeholder
    train_script = """
#!/usr/bin/env python3
\"\"\"Placeholder training script for testing.\"\"\"

def train():
    print("Training model...")
    return {"accuracy": 0.85, "loss": 0.15}

if __name__ == "__main__":
    train()
"""
    (project_dir / "src" / "train.py").write_text(train_script)

    # Create requirements.txt
    (project_dir / "requirements.txt").write_text("torch>=2.0.0\nhydra-core>=1.3.0\n")

    return str(project_dir)


# ============================================================================
# Async Client Fixture (for FastAPI testing)
# ============================================================================


@pytest.fixture
async def async_client(db_session):
    """
    Create an async HTTP client for testing FastAPI endpoints.

    Uses httpx.AsyncClient with the app's TestClient transport.
    Requires db_session fixture to ensure database is initialized.

    Yields:
        httpx.AsyncClient configured to test the FastAPI app
    """
    from httpx import ASGITransport, AsyncClient

    from api_server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ============================================================================
# Database Session Fixture
# ============================================================================


@pytest.fixture
async def db_session(tmp_path, monkeypatch):
    """
    Create an async database session for testing.

    Sets up a temporary SQLite database, initializes the schema,
    and yields an async session. Cleans up after the test.

    Yields:
        AsyncSession connected to a temporary test database
    """
    from db import close_async_db, get_async_session_factory, init_async_db

    # Create temporary database
    db_path = tmp_path / "test_db.sqlite"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Reset any existing global state
    await close_async_db()

    # Initialize database
    await init_async_db()

    # Get a session
    session_factory = get_async_session_factory()
    async with session_factory() as session:
        yield session

    # Cleanup
    await close_async_db()


# ============================================================================
# Mock LLM Fixture
# ============================================================================


@pytest.fixture
def mock_llm(monkeypatch):
    """
    Create a mock LLM (ModelManager) for testing without real API calls.

    Provides configurable mock responses for generate_json and generate_text.
    The mock can be configured by setting attributes on the returned object:
        mock_llm.json_response = {"key": "value"}
        mock_llm.text_response = "response text"

    Returns:
        Mock ModelManager instance with configurable responses
    """

    class MockLLM:
        """Mock LLM class for testing."""

        def __init__(self):
            self.json_response = {}
            self.text_response = ""
            self.call_history = []
            self._generate_calls = 0

        async def generate_json(
            self, prompt, model_name=None, system_prompt=None, use_fallback=True
        ):
            """Mock generate_json that returns configured response."""
            self.call_history.append(
                {
                    "method": "generate_json",
                    "prompt": prompt,
                    "model_name": model_name,
                    "system_prompt": system_prompt,
                }
            )
            self._generate_calls += 1

            # Support callable for dynamic responses
            if callable(self.json_response):
                return self.json_response(prompt)
            return self.json_response

        async def generate_text(
            self, prompt, model_name=None, system_prompt=None, use_fallback=True
        ):
            """Mock generate_text that returns configured response."""
            self.call_history.append(
                {
                    "method": "generate_text",
                    "prompt": prompt,
                    "model_name": model_name,
                    "system_prompt": system_prompt,
                }
            )
            self._generate_calls += 1

            # Support callable for dynamic responses
            if callable(self.text_response):
                return self.text_response(prompt)
            return self.text_response

        async def generate(
            self,
            prompt,
            model_name=None,
            system_prompt=None,
            temperature=None,
            max_tokens=None,
            response_format="text",
            use_fallback=True,
        ):
            """Mock generate that returns configured response based on format."""
            self.call_history.append(
                {
                    "method": "generate",
                    "prompt": prompt,
                    "model_name": model_name,
                    "response_format": response_format,
                }
            )
            self._generate_calls += 1

            if response_format == "json":
                if callable(self.json_response):
                    return self.json_response(prompt)
                return self.json_response
            else:
                if callable(self.text_response):
                    return self.text_response(prompt)
                return self.text_response

        def list_models(self):
            """Return list of mock model names."""
            return ["mock_model", "mock_model_2"]

        def get_model_info(self, model_name):
            """Return mock model info."""
            return {
                "provider": "mock",
                "model": model_name,
                "temperature": 0.7,
                "max_tokens": 4096,
            }

        def get_call_count(self):
            """Return the number of generate calls made."""
            return self._generate_calls

        def reset(self):
            """Reset call history and counters."""
            self.call_history = []
            self._generate_calls = 0

    mock = MockLLM()

    # Patch every module that imports get_model_manager directly. Patching only
    # agent.model_manager misses already-bound references in these modules.
    def get_mock_manager():
        return mock

    monkeypatch.setattr("agent.model_manager.get_model_manager", get_mock_manager)
    monkeypatch.setattr("agent.agent_loop.get_model_manager", get_mock_manager)
    monkeypatch.setattr("perception.perception.get_model_manager", get_mock_manager)
    monkeypatch.setattr("decision.decision.get_model_manager", get_mock_manager)
    monkeypatch.setattr("summarization.summarizer.get_model_manager", get_mock_manager)

    return mock


# ============================================================================
# Additional Helper Fixtures
# ============================================================================


@pytest.fixture
def mock_tool_executor():
    """
    Create a mock tool executor for testing action execution.

    Returns:
        Mock that can be used to simulate tool execution
    """
    mock = MagicMock()
    mock.return_value = {"success": True, "result": "Mock execution completed"}
    return mock


@pytest.fixture
def sample_execution_plan():
    """
    Return a sample execution plan for testing decision/action modules.

    Returns:
        Dict containing a sample plan graph with nodes and dependencies
    """
    return {
        "strategy": "sequential",
        "reasoning": "Execute setup steps in order",
        "plan_graph": {
            "nodes": [
                {
                    "id": "0",
                    "description": "Analyze project configuration",
                    "tool": "analyze_project_config",
                    "args": {"project_path": "/test/project"},
                    "depends_on": [],
                },
                {
                    "id": "1",
                    "description": "Create Hydra configuration",
                    "tool": "create_hydra_config",
                    "args": {"project_path": "/test/project"},
                    "depends_on": ["0"],
                },
                {
                    "id": "2",
                    "description": "Initialize MLflow experiment",
                    "tool": "init_mlflow_experiment",
                    "args": {"experiment_name": "test_experiment"},
                    "depends_on": ["1"],
                },
            ]
        },
        "next_step_id": "0",
    }


@pytest.fixture
def mock_api_server_dependencies(mock_llm, db_session):
    """
    Combined fixture that sets up all dependencies needed for API server testing.

    Provides mock LLM and database session together.

    Returns:
        Dict with mock_llm and db_session
    """
    return {
        "mock_llm": mock_llm,
        "db_session": db_session,
    }
