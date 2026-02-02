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
