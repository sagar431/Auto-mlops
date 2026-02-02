#!/usr/bin/env python3
"""
Unit tests for perception/perception.py - MLOps context analysis and routing.

Run with: pytest tests/unit/test_perception.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Import agent.agent_loop first to resolve circular import between
# perception.perception and agent.agent_loop
from agent.agent_loop import AgentLoop  # noqa: F401
from perception.perception import Perception, build_perception_input

# ============================================================================
# Perception Initialization Tests
# ============================================================================


class TestPerceptionInit:
    """Tests for Perception class initialization."""

    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create temporary prompts directory with test prompts."""
        prompt_content = """Analyze the MLOps context and provide JSON output.

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
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(prompt_content)
        return str(prompt_file)

    def test_create_perception_with_valid_prompt(self, mock_prompts_dir):
        """Test creating Perception with valid prompt file."""
        perception = Perception(mock_prompts_dir)
        assert perception.prompt_template is not None
        assert "{query}" in perception.prompt_template
        assert perception.model_manager is not None

    def test_create_perception_with_missing_file(self, tmp_path):
        """Test creating Perception with missing prompt file uses default."""
        perception = Perception(str(tmp_path / "nonexistent.txt"))
        # Should use default prompt
        assert "entities" in perception.prompt_template
        assert "pipeline_stage" in perception.prompt_template
        assert "route" in perception.prompt_template

    def test_load_prompt_from_file(self, tmp_path):
        """Test _load_prompt successfully reads file content."""
        prompt_content = "Test prompt: {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content)

        perception = Perception(str(prompt_file))
        assert perception.prompt_template == prompt_content

    def test_load_prompt_handles_unicode(self, tmp_path):
        """Test _load_prompt handles Unicode content."""
        prompt_content = "Test prompt with Unicode: こんにちは {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        perception = Perception(str(prompt_file))
        assert "こんにちは" in perception.prompt_template

    def test_get_default_prompt_structure(self, tmp_path):
        """Test default prompt has correct structure."""
        perception = Perception(str(tmp_path / "nonexistent.txt"))
        default = perception._get_default_prompt()

        assert "entities" in default
        assert "pipeline_stage" in default
        assert "required_tools" in default
        assert "route" in default
        assert "decision|summarize|improve" in default
        assert "original_goal_achieved" in default


# ============================================================================
# Perception Output Normalization Tests
# ============================================================================


class TestPerceptionNormalization:
    """Tests for Perception._normalize_output method."""

    @pytest.fixture
    def perception(self, tmp_path):
        """Create Perception instance with minimal prompt."""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(
            "Test: {query} {stage} {experiment_state} "
            "{completed_steps} {failed_steps} {tools} "
            "{memory} {previous_steps}"
        )
        return Perception(str(prompt_file))

    def test_normalize_adds_all_missing_fields(self, perception):
        """Test normalization adds all required fields when missing."""
        output = {}
        normalized = perception._normalize_output(output)

        assert "entities" in normalized
        assert "pipeline_stage" in normalized
        assert "required_tools" in normalized
        assert "result_requirement" in normalized
        assert "original_goal_achieved" in normalized
        assert "local_goal_achieved" in normalized
        assert "missing_requirements" in normalized
        assert "route" in normalized
        assert "confidence" in normalized
        assert "reasoning" in normalized

    def test_normalize_default_values(self, perception):
        """Test normalization sets correct default values."""
        output = {}
        normalized = perception._normalize_output(output)

        assert normalized["entities"] == {}
        assert normalized["pipeline_stage"] == "setup"
        assert normalized["required_tools"] == []
        assert normalized["result_requirement"] == ""
        assert normalized["original_goal_achieved"] is False
        assert normalized["local_goal_achieved"] is False
        assert normalized["missing_requirements"] == []
        assert normalized["route"] == "decision"
        assert normalized["confidence"] == 0.5
        assert normalized["reasoning"] == ""

    def test_normalize_preserves_existing_fields(self, perception):
        """Test normalization preserves valid existing fields."""
        output = {
            "entities": {"project_path": "/test"},
            "pipeline_stage": "training",
            "route": "summarize",
            "confidence": 0.95,
            "reasoning": "Test reasoning",
        }
        normalized = perception._normalize_output(output)

        assert normalized["entities"]["project_path"] == "/test"
        assert normalized["pipeline_stage"] == "training"
        assert normalized["route"] == "summarize"
        assert normalized["confidence"] == 0.95
        assert normalized["reasoning"] == "Test reasoning"

    def test_normalize_corrects_invalid_route(self, perception):
        """Test normalization corrects invalid route value."""
        output = {"route": "invalid_route"}
        normalized = perception._normalize_output(output)
        assert normalized["route"] == "decision"

    def test_normalize_valid_routes(self, perception):
        """Test all valid routes are preserved."""
        valid_routes = ["decision", "summarize", "improve"]
        for route in valid_routes:
            output = {"route": route}
            normalized = perception._normalize_output(output)
            assert normalized["route"] == route

    def test_normalize_corrects_invalid_stage(self, perception):
        """Test normalization corrects invalid pipeline stage."""
        output = {"pipeline_stage": "invalid_stage"}
        normalized = perception._normalize_output(output)
        assert normalized["pipeline_stage"] == "setup"

    def test_normalize_valid_stages(self, perception):
        """Test all valid pipeline stages are preserved."""
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

    def test_normalize_preserves_extra_fields(self, perception):
        """Test normalization preserves additional fields not in defaults."""
        output = {
            "route": "decision",
            "custom_field": "custom_value",
            "another_field": 123,
        }
        normalized = perception._normalize_output(output)

        assert normalized["custom_field"] == "custom_value"
        assert normalized["another_field"] == 123


# ============================================================================
# Perception Fallback Tests
# ============================================================================


class TestPerceptionFallback:
    """Tests for Perception._get_fallback_output method."""

    @pytest.fixture
    def perception(self, tmp_path):
        """Create Perception instance with minimal prompt."""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(
            "Test: {query} {stage} {experiment_state} "
            "{completed_steps} {failed_steps} {tools} "
            "{memory} {previous_steps}"
        )
        return Perception(str(prompt_file))

    def test_fallback_output_structure(self, perception):
        """Test fallback output has correct structure."""
        perception_input = {"stage": "training"}
        fallback = perception._get_fallback_output(perception_input)

        assert "entities" in fallback
        assert "pipeline_stage" in fallback
        assert "required_tools" in fallback
        assert "result_requirement" in fallback
        assert "original_goal_achieved" in fallback
        assert "local_goal_achieved" in fallback
        assert "missing_requirements" in fallback
        assert "route" in fallback
        assert "confidence" in fallback
        assert "reasoning" in fallback

    def test_fallback_output_values(self, perception):
        """Test fallback output has correct values."""
        perception_input = {"stage": "training"}
        fallback = perception._get_fallback_output(perception_input)

        assert fallback["entities"] == {}
        assert fallback["pipeline_stage"] == "training"
        assert fallback["required_tools"] == []
        assert fallback["result_requirement"] == "Continue with next step"
        assert fallback["original_goal_achieved"] is False
        assert fallback["local_goal_achieved"] is False
        assert fallback["missing_requirements"] == []
        assert fallback["route"] == "decision"
        assert fallback["confidence"] == 0.3
        assert "Fallback" in fallback["reasoning"]

    def test_fallback_preserves_stage(self, perception):
        """Test fallback preserves input stage."""
        for stage in ["setup", "config", "data", "training", "evaluation", "improvement", "deploy"]:
            perception_input = {"stage": stage}
            fallback = perception._get_fallback_output(perception_input)
            assert fallback["pipeline_stage"] == stage

    def test_fallback_defaults_to_setup(self, perception):
        """Test fallback defaults to setup stage when not provided."""
        perception_input = {}
        fallback = perception._get_fallback_output(perception_input)
        assert fallback["pipeline_stage"] == "setup"

    def test_fallback_low_confidence(self, perception):
        """Test fallback has low confidence score."""
        perception_input = {"stage": "training"}
        fallback = perception._get_fallback_output(perception_input)
        assert fallback["confidence"] == 0.3


# ============================================================================
# Perception Run Method Tests
# ============================================================================


class TestPerceptionRun:
    """Tests for Perception.run async method."""

    @pytest.fixture
    def perception(self, tmp_path):
        """Create Perception instance with test prompt."""
        prompt_content = """Query: {query}
Stage: {stage}
Experiment State: {experiment_state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Tools: {tools}
Memory: {memory}
Previous Steps: {previous_steps}"""
        prompt_file = tmp_path / "perception_prompt.txt"
        prompt_file.write_text(prompt_content)
        return Perception(str(prompt_file))

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        mock = MagicMock()
        mock.generate_json = AsyncMock()
        return mock

    @pytest.fixture
    def perception_with_mock_llm(self, perception, mock_model_manager):
        """Create Perception with mocked LLM."""
        perception.model_manager = mock_model_manager
        return perception

    @pytest.fixture
    def sample_perception_input(self):
        """Create sample perception input."""
        return {
            "query": "Set up MLOps pipeline",
            "stage": "setup",
            "experiment_state": {},
            "completed_steps": [],
            "failed_steps": [],
            "tools": ["analyze_project_config", "create_hydra_config"],
            "memory": [],
        }

    @pytest.fixture
    def sample_llm_response(self):
        """Create sample LLM response."""
        return {
            "entities": {
                "project_path": "/test/project",
                "experiment_name": "test_experiment",
            },
            "pipeline_stage": "setup",
            "required_tools": ["analyze_project_config"],
            "route": "decision",
            "confidence": 0.9,
            "reasoning": "Project needs initial setup",
        }

    @pytest.mark.asyncio
    async def test_run_returns_normalized_output(
        self, perception_with_mock_llm, sample_perception_input, sample_llm_response
    ):
        """Test run returns normalized output from LLM."""
        perception_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        result = await perception_with_mock_llm.run(sample_perception_input)

        assert result["route"] == "decision"
        assert result["pipeline_stage"] == "setup"
        assert result["entities"]["project_path"] == "/test/project"
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_run_adds_missing_fields(self, perception_with_mock_llm, sample_perception_input):
        """Test run adds missing fields to LLM output."""
        # LLM returns incomplete response
        perception_with_mock_llm.model_manager.generate_json.return_value = {
            "route": "decision",
        }

        result = await perception_with_mock_llm.run(sample_perception_input)

        assert "entities" in result
        assert "pipeline_stage" in result
        assert "required_tools" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_run_logs_to_session(
        self, perception_with_mock_llm, sample_perception_input, sample_llm_response
    ):
        """Test run logs to session when provided."""
        perception_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        mock_session = MagicMock()
        mock_session.add_message = MagicMock()

        await perception_with_mock_llm.run(sample_perception_input, session=mock_session)

        mock_session.add_message.assert_called_once()
        call_kwargs = mock_session.add_message.call_args
        assert call_kwargs[1]["role"] == "assistant"
        assert "perception" in call_kwargs[1]["metadata"]["module"]

    @pytest.mark.asyncio
    async def test_run_without_session(
        self, perception_with_mock_llm, sample_perception_input, sample_llm_response
    ):
        """Test run works without session."""
        perception_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        # Should not raise
        result = await perception_with_mock_llm.run(sample_perception_input, session=None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self, perception_with_mock_llm, sample_perception_input):
        """Test run returns fallback on LLM error."""
        perception_with_mock_llm.model_manager.generate_json.side_effect = Exception(
            "LLM API Error"
        )

        result = await perception_with_mock_llm.run(sample_perception_input)

        assert result["route"] == "decision"
        assert result["confidence"] == 0.3
        assert "Fallback" in result["reasoning"]
        assert result["pipeline_stage"] == "setup"

    @pytest.mark.asyncio
    async def test_run_formats_prompt_correctly(
        self, perception_with_mock_llm, sample_llm_response
    ):
        """Test run formats prompt with all input fields."""
        perception_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        perception_input = {
            "query": "Test query",
            "stage": "training",
            "experiment_state": {"accuracy": 0.85},
            "completed_steps": [{"id": "1", "status": "completed"}],
            "failed_steps": [{"id": "2", "error": "Failed"}],
            "tools": ["tool1", "tool2"],
            "memory": [{"query": "past query"}],
        }

        await perception_with_mock_llm.run(perception_input)

        # Verify generate_json was called
        perception_with_mock_llm.model_manager.generate_json.assert_called_once()
        prompt = perception_with_mock_llm.model_manager.generate_json.call_args[0][0]

        assert "Test query" in prompt
        assert "training" in prompt

    @pytest.mark.asyncio
    async def test_run_handles_empty_input(self, perception_with_mock_llm, sample_llm_response):
        """Test run handles minimal/empty input gracefully."""
        perception_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        perception_input = {}

        # Should not raise
        result = await perception_with_mock_llm.run(perception_input)
        assert result is not None


# ============================================================================
# build_perception_input Tests
# ============================================================================


class TestBuildPerceptionInput:
    """Tests for build_perception_input function."""

    @pytest.fixture
    def mock_context_manager(self):
        """Create mock context manager."""
        from agent.contextManager import ContextManager

        ctx = ContextManager(
            session_id="test-session-123",
            original_query="Test MLOps pipeline setup",
            project_path="/test/project",
        )
        return ctx

    def test_build_perception_input_structure(self, mock_context_manager):
        """Test build_perception_input returns correct structure."""
        result = build_perception_input(
            query="Set up MLOps pipeline",
            memory=[],
            ctx=mock_context_manager,
        )

        assert "query" in result
        assert "stage" in result
        assert "experiment_state" in result
        assert "completed_steps" in result
        assert "failed_steps" in result
        assert "tools" in result
        assert "memory" in result
        assert "snapshot_type" in result
        assert "project_path" in result

    def test_build_perception_input_query(self, mock_context_manager):
        """Test build_perception_input includes query."""
        result = build_perception_input(
            query="Set up MLOps pipeline",
            memory=[],
            ctx=mock_context_manager,
        )

        assert result["query"] == "Set up MLOps pipeline"

    def test_build_perception_input_includes_all_tools(self, mock_context_manager):
        """Test build_perception_input includes all available tools."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        tools = result["tools"]
        # Hydra tools
        assert "analyze_project_config" in tools
        assert "create_hydra_config" in tools
        assert "update_hydra_config" in tools
        assert "validate_hydra_config" in tools
        # MLflow tools
        assert "init_mlflow_experiment" in tools
        assert "start_mlflow_run" in tools
        assert "log_mlflow_params" in tools
        assert "log_mlflow_metrics" in tools
        assert "log_mlflow_artifact" in tools
        assert "register_mlflow_model" in tools
        assert "get_best_mlflow_run" in tools
        assert "end_mlflow_run" in tools
        # DVC tools
        assert "init_dvc_repo" in tools
        assert "configure_dvc_remote" in tools
        assert "add_data_to_dvc" in tools
        assert "create_dvc_pipeline" in tools
        assert "dvc_push" in tools
        assert "dvc_pull" in tools
        assert "dvc_reproduce" in tools
        # Docker tools
        assert "create_ml_dockerfile" in tools
        assert "build_ml_docker_image" in tools
        assert "run_training_container" in tools
        assert "push_docker_image" in tools
        # GitHub Actions tools
        assert "create_github_workflow" in tools
        assert "add_workflow_step" in tools
        # Training tools
        assert "analyze_training_results" in tools
        assert "suggest_improvements" in tools
        assert "check_accuracy_threshold" in tools

    def test_build_perception_input_uses_context_state(self, mock_context_manager):
        """Test build_perception_input uses context state."""
        mock_context_manager.experiment_state.stage = "training"
        mock_context_manager.project_path = "/custom/path"

        result = build_perception_input(
            query="Check accuracy",
            memory=[],
            ctx=mock_context_manager,
        )

        assert result["stage"] == "training"
        assert result["project_path"] == "/custom/path"

    def test_build_perception_input_limits_memory(self, mock_context_manager):
        """Test build_perception_input limits memory items to 5."""
        memory = [{"query": f"Query {i}"} for i in range(10)]

        result = build_perception_input(
            query="Test",
            memory=memory,
            ctx=mock_context_manager,
        )

        assert len(result["memory"]) == 5

    def test_build_perception_input_handles_empty_memory(self, mock_context_manager):
        """Test build_perception_input handles empty memory list."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        assert result["memory"] == []

    def test_build_perception_input_handles_none_memory(self, mock_context_manager):
        """Test build_perception_input handles None memory."""
        result = build_perception_input(
            query="Test",
            memory=None,
            ctx=mock_context_manager,
        )

        assert result["memory"] == []

    def test_build_perception_input_includes_completed_steps(self, mock_context_manager):
        """Test build_perception_input includes completed steps."""
        mock_context_manager.add_step("0", "First step", "CODE", from_node="ROOT")
        mock_context_manager.mark_step_completed("0")

        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        completed_indices = [step["index"] for step in result["completed_steps"]]
        assert "0" in completed_indices

    def test_build_perception_input_includes_failed_steps(self, mock_context_manager):
        """Test build_perception_input includes failed steps."""
        mock_context_manager.add_step("0", "Failed step", "CODE", from_node="ROOT")
        mock_context_manager.mark_step_failed("0", "Error message")

        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        failed_indices = [step["index"] for step in result["failed_steps"]]
        assert "0" in failed_indices

    def test_build_perception_input_default_snapshot_type(self, mock_context_manager):
        """Test build_perception_input default snapshot type is 'initial'."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        assert result["snapshot_type"] == "initial"

    def test_build_perception_input_custom_snapshot_type(self, mock_context_manager):
        """Test build_perception_input custom snapshot type."""
        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
            snapshot_type="step_result",
        )

        assert result["snapshot_type"] == "step_result"

    def test_build_perception_input_experiment_state(self, mock_context_manager):
        """Test build_perception_input includes experiment state."""
        mock_context_manager.experiment_state.current_accuracy = 0.85
        mock_context_manager.experiment_state.target_accuracy = 0.90
        mock_context_manager.experiment_state.run_id = "test-run-123"

        result = build_perception_input(
            query="Test",
            memory=[],
            ctx=mock_context_manager,
        )

        assert "experiment_state" in result
        assert isinstance(result["experiment_state"], dict)
