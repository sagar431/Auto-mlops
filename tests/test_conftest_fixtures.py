#!/usr/bin/env python3
"""
Tests to verify that conftest.py fixtures work correctly.

Run with: pytest tests/test_conftest_fixtures.py -v
"""

import os
from pathlib import Path

import pytest


class TestProjectFixture:
    """Tests for the test_project fixture."""

    def test_project_directory_exists(self, test_project):
        """Test that the project directory is created."""
        assert os.path.isdir(test_project)

    def test_project_has_structure(self, test_project):
        """Test that the project has expected subdirectories."""
        project_path = Path(test_project)
        assert (project_path / "config").is_dir()
        assert (project_path / "data").is_dir()
        assert (project_path / "src").is_dir()
        assert (project_path / "models").is_dir()

    def test_project_has_config_file(self, test_project):
        """Test that config.yaml exists."""
        config_path = Path(test_project) / "config" / "config.yaml"
        assert config_path.is_file()
        content = config_path.read_text()
        assert "experiment:" in content
        assert "model:" in content

    def test_project_has_train_script(self, test_project):
        """Test that train.py exists."""
        train_path = Path(test_project) / "src" / "train.py"
        assert train_path.is_file()


class TestMockLLMFixture:
    """Tests for the mock_llm fixture."""

    @pytest.mark.asyncio
    async def test_mock_llm_json_response(self, mock_llm):
        """Test mock_llm.generate_json returns configured response."""
        mock_llm.json_response = {"status": "ok", "data": [1, 2, 3]}

        result = await mock_llm.generate_json("test prompt")

        assert result == {"status": "ok", "data": [1, 2, 3]}
        assert mock_llm.get_call_count() == 1

    @pytest.mark.asyncio
    async def test_mock_llm_text_response(self, mock_llm):
        """Test mock_llm.generate_text returns configured response."""
        mock_llm.text_response = "Hello, this is a test response."

        result = await mock_llm.generate_text("test prompt")

        assert result == "Hello, this is a test response."
        assert mock_llm.get_call_count() == 1

    @pytest.mark.asyncio
    async def test_mock_llm_callable_response(self, mock_llm):
        """Test mock_llm supports callable responses."""

        def dynamic_response(prompt):
            return {"prompt_length": len(prompt)}

        mock_llm.json_response = dynamic_response

        result = await mock_llm.generate_json("short")
        assert result["prompt_length"] == 5

        result = await mock_llm.generate_json("a longer prompt")
        assert result["prompt_length"] == 15

    @pytest.mark.asyncio
    async def test_mock_llm_call_history(self, mock_llm):
        """Test mock_llm tracks call history."""
        mock_llm.json_response = {}

        await mock_llm.generate_json("first prompt", model_name="test_model")
        await mock_llm.generate_text("second prompt")

        assert len(mock_llm.call_history) == 2
        assert mock_llm.call_history[0]["prompt"] == "first prompt"
        assert mock_llm.call_history[0]["model_name"] == "test_model"
        assert mock_llm.call_history[1]["prompt"] == "second prompt"

    @pytest.mark.asyncio
    async def test_mock_llm_reset(self, mock_llm):
        """Test mock_llm.reset clears state."""
        mock_llm.json_response = {"test": "data"}
        await mock_llm.generate_json("prompt")

        assert mock_llm.get_call_count() == 1

        mock_llm.reset()

        assert mock_llm.get_call_count() == 0
        assert len(mock_llm.call_history) == 0

    def test_mock_llm_list_models(self, mock_llm):
        """Test mock_llm.list_models returns mock models."""
        models = mock_llm.list_models()
        assert "mock_model" in models


class TestDatabaseSessionFixture:
    """Tests for the db_session fixture."""

    @pytest.mark.asyncio
    async def test_db_session_available(self, db_session):
        """Test that db_session yields a valid session."""
        assert db_session is not None

    @pytest.mark.asyncio
    async def test_db_session_can_execute_query(self, db_session):
        """Test that db_session can execute a simple query."""
        from sqlalchemy import text

        result = await db_session.execute(text("SELECT 1"))
        row = result.scalar()
        assert row == 1

    @pytest.mark.asyncio
    async def test_db_session_can_create_agent_session(self, db_session):
        """Test that we can create an AgentSession using db_session."""
        from db.models import AgentSession

        session = AgentSession(
            session_id="test-fixture-session",
            original_query="Test query for fixture",
            project_path="/test/path",
        )
        db_session.add(session)
        await db_session.commit()

        # Verify it was created
        from sqlalchemy import select

        result = await db_session.execute(
            select(AgentSession).where(AgentSession.session_id == "test-fixture-session")
        )
        loaded = result.scalars().first()
        assert loaded is not None
        assert loaded.original_query == "Test query for fixture"


class TestAsyncClientFixture:
    """Tests for the async_client fixture."""

    @pytest.mark.asyncio
    async def test_async_client_health_endpoint(self, async_client):
        """Test async_client can call the health endpoint."""
        response = await async_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_async_client_tools_endpoint(self, async_client):
        """Test async_client can call the tools endpoint."""
        response = await async_client.get("/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "categories" in data


class TestHelperFixtures:
    """Tests for additional helper fixtures."""

    def test_sample_execution_plan_structure(self, sample_execution_plan):
        """Test sample_execution_plan has expected structure."""
        assert "strategy" in sample_execution_plan
        assert "plan_graph" in sample_execution_plan
        assert "nodes" in sample_execution_plan["plan_graph"]

        nodes = sample_execution_plan["plan_graph"]["nodes"]
        assert len(nodes) >= 3

        # Check first node structure
        node = nodes[0]
        assert "id" in node
        assert "description" in node
        assert "tool" in node
        assert "args" in node
        assert "depends_on" in node

    def test_mock_tool_executor(self, mock_tool_executor):
        """Test mock_tool_executor returns success by default."""
        result = mock_tool_executor()
        assert result["success"] is True

    def test_context_manager_fixture(self, context_manager):
        """Test context_manager fixture provides valid ContextManager."""
        assert context_manager.session_id == "test-session-123"
        assert context_manager.original_query == "Test MLOps pipeline setup"

    def test_experiment_state_fixture(self, experiment_state):
        """Test experiment_state fixture provides valid ExperimentState."""
        assert experiment_state is not None
        # ExperimentState should have default values
        assert experiment_state.improvement_attempt == 0
