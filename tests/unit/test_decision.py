#!/usr/bin/env python3
"""
Unit tests for decision/decision.py - MLOps execution plan generation.

Run with: pytest tests/unit/test_decision.py -v
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# Import agent.agent_loop first to resolve circular import between
# decision.decision and agent.agent_loop
from agent.agent_loop import AgentLoop  # noqa: F401
from decision.decision import Decision, build_decision_input

# ============================================================================
# Decision Initialization Tests
# ============================================================================


class TestDecisionInit:
    """Tests for Decision class initialization."""

    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create temporary prompts directory with test prompts."""
        prompt_content = """Generate an execution plan for the MLOps task.

Query: {query}
Perception: {perception}
State: {state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Experiment State: {experiment_state}

Output JSON with: strategy, plan_graph (nodes with id, description, tool, args, depends_on), next_step_id
"""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(prompt_content)
        return str(prompt_file)

    def test_create_decision_with_valid_prompt(self, mock_prompts_dir):
        """Test creating Decision with valid prompt file."""
        decision = Decision(mock_prompts_dir)
        assert decision.prompt_template is not None
        assert "{query}" in decision.prompt_template
        assert decision.model_manager is not None

    def test_create_decision_with_missing_file(self, tmp_path):
        """Test creating Decision with missing prompt file uses default."""
        decision = Decision(str(tmp_path / "nonexistent.txt"))
        # Should use default prompt
        assert "plan_graph" in decision.prompt_template
        assert "next_step_id" in decision.prompt_template
        assert "code_variants" in decision.prompt_template

    def test_create_decision_with_tools_module(self, mock_prompts_dir):
        """Test creating Decision with tools module."""
        mock_tools = MagicMock()
        decision = Decision(mock_prompts_dir, tools_module=mock_tools)
        assert decision.tools_module is mock_tools

    def test_load_prompt_from_file(self, tmp_path):
        """Test _load_prompt successfully reads file content."""
        prompt_content = "Test prompt: {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content)

        decision = Decision(str(prompt_file))
        assert decision.prompt_template == prompt_content

    def test_load_prompt_handles_unicode(self, tmp_path):
        """Test _load_prompt handles Unicode content."""
        prompt_content = "Test prompt with Unicode: こんにちは {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content, encoding="utf-8")

        decision = Decision(str(prompt_file))
        assert "こんにちは" in decision.prompt_template

    def test_get_default_prompt_structure(self, tmp_path):
        """Test default prompt has correct structure."""
        decision = Decision(str(tmp_path / "nonexistent.txt"))
        default = decision._get_default_prompt()

        assert "plan_graph" in default
        assert "next_step_id" in default
        assert "code_variants" in default
        assert "nodes" in default


# ============================================================================
# Decision Output Normalization Tests
# ============================================================================


class TestDecisionNormalization:
    """Tests for Decision._normalize_output method."""

    @pytest.fixture
    def decision(self, tmp_path):
        """Create Decision instance with minimal prompt."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(
            "Test: {query} {perception} {state} "
            "{completed_steps} {failed_steps} {experiment_state}"
        )
        return Decision(str(prompt_file))

    def test_normalize_adds_all_missing_fields(self, decision):
        """Test normalization adds all required fields when missing."""
        output = {}
        normalized = decision._normalize_output(output)

        assert "strategy" in normalized
        assert "reasoning" in normalized
        assert "plan_graph" in normalized
        assert "next_step_id" in normalized
        assert "code_variants" in normalized

    def test_normalize_default_values(self, decision):
        """Test normalization sets correct default values."""
        output = {}
        normalized = decision._normalize_output(output)

        assert normalized["strategy"] == "sequential"
        assert normalized["reasoning"] == ""
        assert normalized["plan_graph"] == {"nodes": []}
        assert normalized["next_step_id"] == "0"
        assert normalized["code_variants"] == {}

    def test_normalize_preserves_existing_fields(self, decision):
        """Test normalization preserves valid existing fields."""
        output = {
            "strategy": "parallel",
            "reasoning": "Execute steps in parallel",
            "plan_graph": {
                "nodes": [
                    {
                        "id": "1",
                        "description": "Test step",
                        "tool": "test_tool",
                        "args": {},
                        "depends_on": [],
                    }
                ]
            },
            "next_step_id": "1",
            "code_variants": {"variant1": "code"},
        }
        normalized = decision._normalize_output(output)

        assert normalized["strategy"] == "parallel"
        assert normalized["reasoning"] == "Execute steps in parallel"
        assert len(normalized["plan_graph"]["nodes"]) == 1
        assert normalized["next_step_id"] == "1"
        assert normalized["code_variants"]["variant1"] == "code"

    def test_normalize_adds_missing_nodes_key(self, decision):
        """Test normalization adds nodes key when missing from plan_graph."""
        output = {"plan_graph": {}}
        normalized = decision._normalize_output(output)
        assert "nodes" in normalized["plan_graph"]
        assert normalized["plan_graph"]["nodes"] == []

    def test_normalize_adds_missing_node_fields(self, decision):
        """Test normalization adds missing fields to nodes."""
        output = {
            "plan_graph": {
                "nodes": [
                    {
                        "id": "1",
                        "description": "Test step",
                    }
                ]
            }
        }
        normalized = decision._normalize_output(output)

        node = normalized["plan_graph"]["nodes"][0]
        assert node["id"] == "1"
        assert node["description"] == "Test step"
        assert node["tool"] is None
        assert node["args"] == {}
        assert node["depends_on"] == []

    def test_normalize_sets_next_step_id_to_first_node(self, decision):
        """Test normalization sets next_step_id to first node when empty."""
        output = {
            "plan_graph": {
                "nodes": [
                    {
                        "id": "5",
                        "description": "First step",
                        "tool": "test_tool",
                        "args": {},
                        "depends_on": [],
                    }
                ]
            },
            "next_step_id": "",
        }
        normalized = decision._normalize_output(output)
        assert normalized["next_step_id"] == "5"

    def test_normalize_preserves_next_step_id_when_set(self, decision):
        """Test normalization preserves next_step_id when already set."""
        output = {
            "plan_graph": {
                "nodes": [
                    {
                        "id": "1",
                        "description": "First step",
                        "tool": "test_tool",
                        "args": {},
                        "depends_on": [],
                    },
                    {
                        "id": "2",
                        "description": "Second step",
                        "tool": "test_tool",
                        "args": {},
                        "depends_on": ["1"],
                    },
                ]
            },
            "next_step_id": "2",
        }
        normalized = decision._normalize_output(output)
        assert normalized["next_step_id"] == "2"

    def test_normalize_preserves_extra_fields(self, decision):
        """Test normalization preserves additional fields not in defaults."""
        output = {
            "strategy": "sequential",
            "custom_field": "custom_value",
            "another_field": 123,
        }
        normalized = decision._normalize_output(output)

        assert normalized["custom_field"] == "custom_value"
        assert normalized["another_field"] == 123


# ============================================================================
# Decision Fallback Tests
# ============================================================================


class TestDecisionFallback:
    """Tests for Decision._get_fallback_output method."""

    @pytest.fixture
    def decision(self, tmp_path):
        """Create Decision instance with minimal prompt."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(
            "Test: {query} {perception} {state} "
            "{completed_steps} {failed_steps} {experiment_state}"
        )
        return Decision(str(prompt_file))

    def test_fallback_output_structure(self, decision):
        """Test fallback output has correct structure."""
        decision_input = {"perception": {"pipeline_stage": "setup"}}
        fallback = decision._get_fallback_output(decision_input)

        assert "strategy" in fallback
        assert "reasoning" in fallback
        assert "plan_graph" in fallback
        assert "next_step_id" in fallback
        assert "code_variants" in fallback

    def test_fallback_output_values(self, decision):
        """Test fallback output has correct values."""
        decision_input = {"perception": {"pipeline_stage": "setup"}}
        fallback = decision._get_fallback_output(decision_input)

        assert fallback["strategy"] == "sequential"
        assert "Fallback" in fallback["reasoning"]
        assert fallback["next_step_id"] == "0"
        assert fallback["code_variants"] == {}

    def test_fallback_setup_stage(self, decision):
        """Test fallback for setup stage generates analyze_project_config."""
        decision_input = {
            "perception": {
                "pipeline_stage": "setup",
                "entities": {"project_path": "/test/project"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert len(nodes) >= 1
        assert nodes[0]["tool"] == "analyze_project_config"
        assert nodes[0]["args"]["project_path"] == "/test/project"

    def test_fallback_config_stage(self, decision):
        """Test fallback for config stage generates create_hydra_config."""
        decision_input = {
            "perception": {
                "pipeline_stage": "config",
                "entities": {"project_path": "/test/project"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert len(nodes) >= 1
        assert nodes[0]["tool"] == "create_hydra_config"

    def test_fallback_training_stage(self, decision):
        """Test fallback for training stage generates init_mlflow_experiment."""
        decision_input = {
            "perception": {
                "pipeline_stage": "training",
                "entities": {"experiment_name": "test_exp"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert len(nodes) >= 1
        assert nodes[0]["tool"] == "init_mlflow_experiment"
        assert nodes[0]["args"]["experiment_name"] == "test_exp"

    def test_fallback_unknown_stage_defaults_to_setup(self, decision):
        """Test fallback for unknown stage defaults to setup plan."""
        decision_input = {"perception": {"pipeline_stage": "unknown_stage"}}
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert len(nodes) >= 1
        assert nodes[0]["tool"] == "analyze_project_config"

    def test_fallback_uses_default_project_path(self, decision):
        """Test fallback uses '.' when project_path not specified."""
        decision_input = {"perception": {"pipeline_stage": "setup", "entities": {}}}
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert nodes[0]["args"]["project_path"] == "."

    def test_fallback_uses_default_experiment_name(self, decision):
        """Test fallback uses default experiment name when not specified."""
        decision_input = {"perception": {"pipeline_stage": "training", "entities": {}}}
        fallback = decision._get_fallback_output(decision_input)

        nodes = fallback["plan_graph"]["nodes"]
        assert nodes[0]["args"]["experiment_name"] == "default_experiment"


# ============================================================================
# Decision Run Method Tests
# ============================================================================


class TestDecisionRun:
    """Tests for Decision.run async method."""

    @pytest.fixture
    def decision(self, tmp_path):
        """Create Decision instance with test prompt."""
        prompt_content = """Query: {query}
Perception: {perception}
State: {state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Experiment State: {experiment_state}"""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(prompt_content)
        return Decision(str(prompt_file))

    @pytest.fixture
    def mock_model_manager(self):
        """Create mock model manager."""
        mock = MagicMock()
        mock.generate_json = AsyncMock()
        return mock

    @pytest.fixture
    def decision_with_mock_llm(self, decision, mock_model_manager):
        """Create Decision with mocked LLM."""
        decision.model_manager = mock_model_manager
        return decision

    @pytest.fixture
    def sample_decision_input(self):
        """Create sample decision input."""
        return {
            "original_query": "Set up MLOps pipeline",
            "perception": {
                "pipeline_stage": "setup",
                "entities": {"project_path": "/test/project"},
            },
            "state": {"pipeline_stage": "setup", "project_path": "/test/project"},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
        }

    @pytest.fixture
    def sample_llm_response(self):
        """Create sample LLM response."""
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

    @pytest.mark.asyncio
    async def test_run_returns_normalized_output(
        self, decision_with_mock_llm, sample_decision_input, sample_llm_response
    ):
        """Test run returns normalized output from LLM."""
        decision_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        result = await decision_with_mock_llm.run(sample_decision_input)

        assert result["strategy"] == "sequential"
        assert result["reasoning"] == "Execute setup steps in order"
        assert len(result["plan_graph"]["nodes"]) == 2
        assert result["next_step_id"] == "0"

    @pytest.mark.asyncio
    async def test_run_adds_missing_fields(self, decision_with_mock_llm, sample_decision_input):
        """Test run adds missing fields to LLM output."""
        # LLM returns incomplete response
        decision_with_mock_llm.model_manager.generate_json.return_value = {
            "plan_graph": {"nodes": []},
        }

        result = await decision_with_mock_llm.run(sample_decision_input)

        assert "strategy" in result
        assert "reasoning" in result
        assert "next_step_id" in result
        assert "code_variants" in result

    @pytest.mark.asyncio
    async def test_run_logs_to_session(
        self, decision_with_mock_llm, sample_decision_input, sample_llm_response
    ):
        """Test run logs to session when provided."""
        decision_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        mock_session = MagicMock()
        mock_session.add_message = MagicMock()

        await decision_with_mock_llm.run(sample_decision_input, session=mock_session)

        mock_session.add_message.assert_called_once()
        call_kwargs = mock_session.add_message.call_args
        assert call_kwargs[1]["role"] == "assistant"
        assert "decision" in call_kwargs[1]["metadata"]["module"]

    @pytest.mark.asyncio
    async def test_run_without_session(
        self, decision_with_mock_llm, sample_decision_input, sample_llm_response
    ):
        """Test run works without session."""
        decision_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        # Should not raise
        result = await decision_with_mock_llm.run(sample_decision_input, session=None)
        assert result is not None

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self, decision_with_mock_llm, sample_decision_input):
        """Test run returns fallback on LLM error."""
        decision_with_mock_llm.model_manager.generate_json.side_effect = Exception("LLM API Error")

        result = await decision_with_mock_llm.run(sample_decision_input)

        assert result["strategy"] == "sequential"
        assert "Fallback" in result["reasoning"]
        assert len(result["plan_graph"]["nodes"]) >= 1

    @pytest.mark.asyncio
    async def test_run_formats_prompt_correctly(self, decision_with_mock_llm, sample_llm_response):
        """Test run formats prompt with all input fields."""
        decision_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        decision_input = {
            "original_query": "Test query",
            "perception": {
                "pipeline_stage": "training",
                "entities": {"project_path": "/custom/path"},
            },
            "state": {"pipeline_stage": "training", "project_path": "/custom/path"},
            "completed_steps": [{"id": "1", "status": "completed"}],
            "failed_steps": [{"id": "2", "error": "Failed"}],
            "experiment_state": {"accuracy": 0.85},
        }

        await decision_with_mock_llm.run(decision_input)

        # Verify generate_json was called
        decision_with_mock_llm.model_manager.generate_json.assert_called_once()
        prompt = decision_with_mock_llm.model_manager.generate_json.call_args[0][0]

        assert "Test query" in prompt
        assert "training" in prompt

    @pytest.mark.asyncio
    async def test_run_handles_empty_input(self, decision_with_mock_llm, sample_llm_response):
        """Test run handles minimal/empty input gracefully."""
        decision_with_mock_llm.model_manager.generate_json.return_value = sample_llm_response

        decision_input = {}

        # Should not raise
        result = await decision_with_mock_llm.run(decision_input)
        assert result is not None


# ============================================================================
# Decision Prompt Formatting Tests
# ============================================================================


class TestDecisionPromptFormatting:
    """Tests for Decision._format_prompt method."""

    @pytest.fixture
    def decision_with_template(self, tmp_path):
        """Create Decision with template variables."""
        prompt_content = """Query: {query}
Perception: {perception}
State: {state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Experiment State: {experiment_state}"""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(prompt_content)
        return Decision(str(prompt_file))

    @pytest.fixture
    def decision_with_extra_placeholder(self, tmp_path):
        """Create Decision with template that has extra placeholders not provided."""
        prompt_content = """Query: {query}
Perception: {perception}
State: {state}
Completed Steps: {completed_steps}
Failed Steps: {failed_steps}
Experiment State: {experiment_state}
Extra Field: {extra_field}"""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(prompt_content)
        return Decision(str(prompt_file))

    def test_format_prompt_with_all_fields(self, decision_with_template):
        """Test _format_prompt with all fields present."""
        decision_input = {
            "original_query": "Set up MLOps pipeline",
            "perception": {"pipeline_stage": "setup"},
            "state": {"project_path": "/test"},
            "completed_steps": [{"id": "0"}],
            "failed_steps": [],
            "experiment_state": {"accuracy": 0.85},
        }

        formatted = decision_with_template._format_prompt(decision_input)

        assert "Set up MLOps pipeline" in formatted
        assert "setup" in formatted
        assert "/test" in formatted

    def test_format_prompt_fallback_to_json(self, decision_with_extra_placeholder):
        """Test _format_prompt falls back to JSON append when template fails with KeyError."""
        decision_input = {
            "original_query": "Test query",
            "perception": {"pipeline_stage": "setup"},
            "state": {"project_path": "/test"},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
            # Note: extra_field is NOT provided, which should cause KeyError
        }

        formatted = decision_with_extra_placeholder._format_prompt(decision_input)

        # Should fall back to JSON append when KeyError occurs
        assert "Input Context" in formatted
        assert "Test query" in formatted

    def test_format_prompt_handles_missing_fields(self, decision_with_template):
        """Test _format_prompt handles missing fields gracefully."""
        decision_input = {
            "original_query": "Test query",
        }

        # Template uses .get() with defaults, so it should work
        formatted = decision_with_template._format_prompt(decision_input)

        # Should contain the query
        assert "Test query" in formatted


# ============================================================================
# build_decision_input Tests
# ============================================================================


class TestBuildDecisionInput:
    """Tests for build_decision_input function."""

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

    @pytest.fixture
    def sample_perception(self):
        """Create sample perception output."""
        return {
            "entities": {"project_path": "/test/project"},
            "pipeline_stage": "setup",
            "required_tools": ["analyze_project_config"],
            "route": "decision",
            "confidence": 0.9,
            "reasoning": "Project needs initial setup",
        }

    def test_build_decision_input_structure(self, mock_context_manager, sample_perception):
        """Test build_decision_input returns correct structure."""
        result = build_decision_input(
            ctx=mock_context_manager,
            query="Set up MLOps pipeline",
            perception=sample_perception,
        )

        assert "current_time" in result
        assert "run_id" in result
        assert "original_query" in result
        assert "perception" in result
        assert "state" in result
        assert "completed_steps" in result
        assert "failed_steps" in result
        assert "experiment_state" in result
        assert "globals_schema" in result

    def test_build_decision_input_query(self, mock_context_manager, sample_perception):
        """Test build_decision_input includes query."""
        result = build_decision_input(
            ctx=mock_context_manager,
            query="Set up MLOps pipeline",
            perception=sample_perception,
        )

        assert result["original_query"] == "Set up MLOps pipeline"

    def test_build_decision_input_perception(self, mock_context_manager, sample_perception):
        """Test build_decision_input includes perception."""
        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        assert result["perception"] == sample_perception

    def test_build_decision_input_run_id_format(self, mock_context_manager, sample_perception):
        """Test build_decision_input run_id has correct format."""
        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        assert result["run_id"].endswith("-D")
        assert mock_context_manager.session_id in result["run_id"]

    def test_build_decision_input_uses_context_state(self, mock_context_manager, sample_perception):
        """Test build_decision_input uses context state."""
        mock_context_manager.experiment_state.stage = "training"
        mock_context_manager.project_path = "/custom/path"

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Check accuracy",
            perception=sample_perception,
        )

        assert result["state"]["pipeline_stage"] == "training"
        assert result["state"]["project_path"] == "/custom/path"

    def test_build_decision_input_includes_completed_steps(
        self, mock_context_manager, sample_perception
    ):
        """Test build_decision_input includes completed steps."""
        mock_context_manager.add_step("0", "First step", "CODE", from_node="ROOT")
        mock_context_manager.mark_step_completed("0")

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        # Check that completed_steps is a list
        assert isinstance(result["completed_steps"], list)

    def test_build_decision_input_includes_failed_steps(
        self, mock_context_manager, sample_perception
    ):
        """Test build_decision_input includes failed steps."""
        mock_context_manager.add_step("0", "Failed step", "CODE", from_node="ROOT")
        mock_context_manager.mark_step_failed("0", "Error message")

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        # Check that failed_steps is a list
        assert isinstance(result["failed_steps"], list)

    def test_build_decision_input_experiment_state(self, mock_context_manager, sample_perception):
        """Test build_decision_input includes experiment state."""
        mock_context_manager.experiment_state.current_accuracy = 0.85
        mock_context_manager.experiment_state.target_accuracy = 0.90
        mock_context_manager.experiment_state.run_id = "test-run-123"

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        assert "experiment_state" in result
        assert isinstance(result["experiment_state"], dict)

    def test_build_decision_input_globals_schema(self, mock_context_manager, sample_perception):
        """Test build_decision_input includes globals schema."""
        mock_context_manager.globals["test_key"] = "test_value"

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        assert "globals_schema" in result
        assert "test_key" in result["globals_schema"]
        assert result["globals_schema"]["test_key"]["type"] == "str"

    def test_build_decision_input_globals_preview_truncation(
        self, mock_context_manager, sample_perception
    ):
        """Test build_decision_input truncates long global values."""
        # Create a very long string value
        long_value = "x" * 1000
        mock_context_manager.globals["long_key"] = long_value

        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        preview = result["globals_schema"]["long_key"]["preview"]
        # Preview should be truncated to 500 chars plus "..."
        assert len(preview) <= 503
        assert preview.endswith("...")

    def test_build_decision_input_current_time(self, mock_context_manager, sample_perception):
        """Test build_decision_input includes current time."""
        result = build_decision_input(
            ctx=mock_context_manager,
            query="Test",
            perception=sample_perception,
        )

        assert "current_time" in result
        # Should be an ISO format string
        assert "T" in result["current_time"]


# ============================================================================
# Decision Plan Graph Validation Tests
# ============================================================================


class TestDecisionPlanGraphValidation:
    """Tests for plan graph structure validation in Decision."""

    @pytest.fixture
    def decision(self, tmp_path):
        """Create Decision instance."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text("Test prompt")
        return Decision(str(prompt_file))

    def test_normalize_empty_plan_graph(self, decision):
        """Test normalization of empty plan graph (missing key)."""
        output = {}
        normalized = decision._normalize_output(output)
        assert normalized["plan_graph"] == {"nodes": []}

    def test_normalize_plan_graph_with_dependencies(self, decision):
        """Test normalization preserves node dependencies."""
        output = {
            "plan_graph": {
                "nodes": [
                    {
                        "id": "0",
                        "description": "Step 1",
                        "tool": "tool1",
                        "args": {},
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Step 2",
                        "tool": "tool2",
                        "args": {},
                        "depends_on": ["0"],
                    },
                ]
            }
        }
        normalized = decision._normalize_output(output)

        nodes = normalized["plan_graph"]["nodes"]
        assert nodes[0]["depends_on"] == []
        assert nodes[1]["depends_on"] == ["0"]

    def test_normalize_plan_graph_with_complex_args(self, decision):
        """Test normalization preserves complex node arguments."""
        output = {
            "plan_graph": {
                "nodes": [
                    {
                        "id": "0",
                        "description": "Configure training",
                        "tool": "create_hydra_config",
                        "args": {
                            "project_path": "/test",
                            "config": {
                                "model": {"type": "mlp", "hidden_dim": 128},
                                "training": {"epochs": 10, "batch_size": 32},
                            },
                        },
                        "depends_on": [],
                    }
                ]
            }
        }
        normalized = decision._normalize_output(output)

        args = normalized["plan_graph"]["nodes"][0]["args"]
        assert args["project_path"] == "/test"
        assert args["config"]["model"]["type"] == "mlp"
        assert args["config"]["training"]["epochs"] == 10

    def test_normalize_multiple_nodes_with_mixed_completeness(self, decision):
        """Test normalization handles nodes with varying completeness."""
        output = {
            "plan_graph": {
                "nodes": [
                    {"id": "0", "description": "Complete step", "tool": "tool1"},
                    {"id": "1"},  # Very incomplete
                    {
                        "id": "2",
                        "description": "Another step",
                        "tool": "tool2",
                        "args": {"key": "value"},
                    },
                ]
            }
        }
        normalized = decision._normalize_output(output)

        nodes = normalized["plan_graph"]["nodes"]
        assert len(nodes) == 3

        # Check that all nodes have required fields
        for node in nodes:
            assert "id" in node
            assert "description" in node
            assert "tool" in node
            assert "args" in node
            assert "depends_on" in node
