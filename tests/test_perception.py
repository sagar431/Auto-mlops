#!/usr/bin/env python3
"""
Tests for perception/perception.py - MLOps context analysis and routing.

Run with: pytest tests/test_perception.py -v
"""

import pytest

from perception.perception import Perception, build_perception_input

# ============================================================================
# Perception Class Tests
# ============================================================================


class TestPerception:
    """Tests for Perception class."""

    @pytest.fixture
    def perception(self, tmp_path, perception_prompt_template):
        """Create a Perception instance with test prompt."""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(perception_prompt_template)
        return Perception(str(prompt_file))

    @pytest.fixture
    def perception_with_mock_llm(self, perception, mock_model_manager):
        """Create a Perception instance with mocked LLM."""
        perception.model_manager = mock_model_manager
        return perception

    def test_create_perception(self, perception, perception_prompt_template):
        """Test creating a Perception instance."""
        assert perception.prompt_template == perception_prompt_template
        assert perception.model_manager is not None

    def test_load_prompt_from_file(self, tmp_path):
        """Test loading prompt template from file."""
        prompt_content = "Test prompt: {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content)

        perception = Perception(str(prompt_file))
        assert perception.prompt_template == prompt_content

    def test_load_prompt_missing_file(self, tmp_path):
        """Test loading prompt with missing file falls back to default."""
        perception = Perception(str(tmp_path / "nonexistent.txt"))

        # Should use default prompt
        assert "entities" in perception.prompt_template
        assert "pipeline_stage" in perception.prompt_template

    def test_get_default_prompt(self, perception):
        """Test default prompt content."""
        default = perception._get_default_prompt()

        assert "entities" in default
        assert "pipeline_stage" in default
        assert "route" in default
        assert "decision|summarize|improve" in default

    @pytest.mark.asyncio
    async def test_run_returns_normalized_output(
        self, perception_with_mock_llm, mock_perception_response
    ):
        """Test that run returns normalized output."""
        perception_with_mock_llm.model_manager.generate_json.return_value = mock_perception_response

        perception_input = {
            "query": "Set up MLOps pipeline",
            "stage": "setup",
            "experiment_state": {},
            "completed_steps": [],
            "failed_steps": [],
            "tools": ["analyze_project_config"],
            "memory": [],
        }

        result = await perception_with_mock_llm.run(perception_input)

        assert result["route"] == "decision"
        assert result["pipeline_stage"] == "setup"
        assert result["entities"]["project_path"] == "/test/project"

    @pytest.mark.asyncio
    async def test_run_logs_to_session(
        self, perception_with_mock_llm, mock_perception_response, mock_session
    ):
        """Test that run logs to session when provided."""
        perception_with_mock_llm.model_manager.generate_json.return_value = mock_perception_response

        perception_input = {
            "query": "Set up MLOps",
            "stage": "setup",
            "experiment_state": {},
            "completed_steps": [],
            "failed_steps": [],
            "tools": [],
            "memory": [],
        }

        await perception_with_mock_llm.run(perception_input, session=mock_session)

        mock_session.add_message.assert_called_once()
        call_kwargs = mock_session.add_message.call_args
        assert call_kwargs[1]["role"] == "assistant"
        assert "perception" in call_kwargs[1]["metadata"]["module"]

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self, perception_with_mock_llm):
        """Test that run returns fallback on LLM error."""
        perception_with_mock_llm.model_manager.generate_json.side_effect = Exception("LLM Error")

        perception_input = {
            "query": "Test query",
            "stage": "training",
            "experiment_state": {},
            "completed_steps": [],
            "failed_steps": [],
            "tools": [],
            "memory": [],
        }

        result = await perception_with_mock_llm.run(perception_input)

        assert result["route"] == "decision"
        assert result["confidence"] == 0.3
        assert "Fallback" in result["reasoning"]


class TestPerceptionNormalization:
    """Tests for Perception output normalization."""

    @pytest.fixture
    def perception(self, tmp_path, perception_prompt_template):
        """Create a Perception instance."""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(perception_prompt_template)
        return Perception(str(prompt_file))

    def test_normalize_adds_missing_fields(self, perception):
        """Test that normalization adds missing required fields."""
        output = {"route": "decision"}
        normalized = perception._normalize_output(output)

        assert "entities" in normalized
        assert "pipeline_stage" in normalized
        assert "required_tools" in normalized
        assert "confidence" in normalized
        assert "reasoning" in normalized

    def test_normalize_preserves_existing_fields(self, perception):
        """Test that normalization preserves existing fields."""
        output = {
            "entities": {"project_path": "/test"},
            "route": "summarize",
            "confidence": 0.95,
        }
        normalized = perception._normalize_output(output)

        assert normalized["entities"]["project_path"] == "/test"
        assert normalized["route"] == "summarize"
        assert normalized["confidence"] == 0.95

    def test_normalize_invalid_route(self, perception):
        """Test that invalid route is corrected."""
        output = {"route": "invalid_route"}
        normalized = perception._normalize_output(output)

        assert normalized["route"] == "decision"

    def test_normalize_valid_routes(self, perception):
        """Test that valid routes are preserved."""
        for route in ["decision", "summarize", "improve"]:
            output = {"route": route}
            normalized = perception._normalize_output(output)
            assert normalized["route"] == route

    def test_normalize_invalid_stage(self, perception):
        """Test that invalid pipeline stage is corrected."""
        output = {"pipeline_stage": "invalid_stage"}
        normalized = perception._normalize_output(output)

        assert normalized["pipeline_stage"] == "setup"

    def test_normalize_valid_stages(self, perception):
        """Test that valid pipeline stages are preserved."""
        valid_stages = [
            "setup",
            "config",
            "data",
            "training",
            "evaluation",
            "improvement",
            "deploy",
        ]

        for stage in valid_stages:
            output = {"pipeline_stage": stage}
            normalized = perception._normalize_output(output)
            assert normalized["pipeline_stage"] == stage


class TestPerceptionFallback:
    """Tests for Perception fallback behavior."""

    @pytest.fixture
    def perception(self, tmp_path, perception_prompt_template):
        """Create a Perception instance."""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(perception_prompt_template)
        return Perception(str(prompt_file))

    def test_fallback_output_structure(self, perception):
        """Test fallback output has correct structure."""
        perception_input = {"stage": "training"}
        fallback = perception._get_fallback_output(perception_input)

        assert fallback["route"] == "decision"
        assert fallback["pipeline_stage"] == "training"
        assert fallback["confidence"] == 0.3
        assert "Fallback" in fallback["reasoning"]

    def test_fallback_preserves_stage(self, perception):
        """Test fallback preserves input stage."""
        perception_input = {"stage": "evaluation"}
        fallback = perception._get_fallback_output(perception_input)

        assert fallback["pipeline_stage"] == "evaluation"

    def test_fallback_defaults_to_setup(self, perception):
        """Test fallback defaults to setup stage when not provided."""
        perception_input = {}
        fallback = perception._get_fallback_output(perception_input)

        assert fallback["pipeline_stage"] == "setup"


# ============================================================================
# build_perception_input Tests
# ============================================================================


class TestBuildPerceptionInput:
    """Tests for build_perception_input function."""

    def test_build_perception_input_structure(self, context_manager):
        """Test that build_perception_input returns correct structure."""
        result = build_perception_input(
            query="Set up MLOps pipeline",
            memory=[],
            ctx=context_manager,
        )

        assert result["query"] == "Set up MLOps pipeline"
        assert result["stage"] == "setup"
        assert "experiment_state" in result
        assert "completed_steps" in result
        assert "failed_steps" in result
        assert "tools" in result
        assert "project_path" in result

    def test_build_perception_input_includes_tools(self, context_manager):
        """Test that build_perception_input includes available tools."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=context_manager,
        )

        tools = result["tools"]
        assert "analyze_project_config" in tools
        assert "create_hydra_config" in tools
        assert "init_mlflow_experiment" in tools
        assert "init_dvc_repo" in tools

    def test_build_perception_input_uses_context_state(self, context_manager):
        """Test that build_perception_input uses context state."""
        context_manager.experiment_state.stage = "training"
        context_manager.project_path = "/custom/path"

        result = build_perception_input(
            query="Check accuracy",
            memory=[],
            ctx=context_manager,
        )

        assert result["stage"] == "training"
        assert result["project_path"] == "/custom/path"

    def test_build_perception_input_limits_memory(self, context_manager):
        """Test that build_perception_input limits memory items."""
        memory = [{"query": f"Query {i}"} for i in range(10)]

        result = build_perception_input(
            query="Test",
            memory=memory,
            ctx=context_manager,
        )

        assert len(result["memory"]) <= 5

    def test_build_perception_input_handles_empty_memory(self, context_manager):
        """Test that build_perception_input handles empty memory."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=context_manager,
        )

        assert result["memory"] == []

    def test_build_perception_input_includes_completed_steps(self, context_manager):
        """Test that build_perception_input includes completed steps."""
        context_manager.add_step("0", "First step", "CODE", from_node="ROOT")
        context_manager.mark_step_completed("0")

        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=context_manager,
        )

        completed_indices = [step["index"] for step in result["completed_steps"]]
        assert "0" in completed_indices

    def test_build_perception_input_includes_failed_steps(self, context_manager):
        """Test that build_perception_input includes failed steps."""
        context_manager.add_step("0", "Failed step", "CODE", from_node="ROOT")
        context_manager.mark_step_failed("0", "Error")

        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=context_manager,
        )

        failed_indices = [step["index"] for step in result["failed_steps"]]
        assert "0" in failed_indices

    def test_build_perception_input_snapshot_type(self, context_manager):
        """Test that build_perception_input includes snapshot type."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=context_manager,
            snapshot_type="step_result",
        )

        assert result["snapshot_type"] == "step_result"
