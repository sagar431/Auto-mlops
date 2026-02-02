#!/usr/bin/env python3
"""
Tests for decision/decision.py - MLOps execution plan generation.

Run with: pytest tests/test_decision.py -v
"""

import pytest

from decision.decision import Decision, build_decision_input

# ============================================================================
# Decision Class Tests
# ============================================================================


class TestDecision:
    """Tests for Decision class."""

    @pytest.fixture
    def decision(self, tmp_path, decision_prompt_template):
        """Create a Decision instance with test prompt."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(decision_prompt_template)
        return Decision(str(prompt_file))

    @pytest.fixture
    def decision_with_mock_llm(self, decision, mock_model_manager):
        """Create a Decision instance with mocked LLM."""
        decision.model_manager = mock_model_manager
        return decision

    def test_create_decision(self, decision, decision_prompt_template):
        """Test creating a Decision instance."""
        assert decision.prompt_template == decision_prompt_template
        assert decision.model_manager is not None

    def test_load_prompt_from_file(self, tmp_path):
        """Test loading prompt template from file."""
        prompt_content = "Generate plan for: {query}"
        prompt_file = tmp_path / "test_prompt.txt"
        prompt_file.write_text(prompt_content)

        decision = Decision(str(prompt_file))
        assert decision.prompt_template == prompt_content

    def test_load_prompt_missing_file(self, tmp_path):
        """Test loading prompt with missing file falls back to default."""
        decision = Decision(str(tmp_path / "nonexistent.txt"))

        # Should use default prompt
        assert "plan_graph" in decision.prompt_template
        assert "next_step_id" in decision.prompt_template

    def test_get_default_prompt(self, decision):
        """Test default prompt content."""
        default = decision._get_default_prompt()

        assert "plan_graph" in default
        assert "nodes" in default
        assert "next_step_id" in default

    @pytest.mark.asyncio
    async def test_run_returns_normalized_output(
        self, decision_with_mock_llm, mock_decision_response
    ):
        """Test that run returns normalized output."""
        decision_with_mock_llm.model_manager.generate_json.return_value = mock_decision_response

        decision_input = {
            "original_query": "Set up MLOps pipeline",
            "perception": {"pipeline_stage": "setup"},
            "state": {},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
        }

        result = await decision_with_mock_llm.run(decision_input)

        assert result["strategy"] == "sequential"
        assert "plan_graph" in result
        assert "nodes" in result["plan_graph"]
        assert len(result["plan_graph"]["nodes"]) == 2
        assert result["next_step_id"] == "0"

    @pytest.mark.asyncio
    async def test_run_logs_to_session(
        self, decision_with_mock_llm, mock_decision_response, mock_session
    ):
        """Test that run logs to session when provided."""
        decision_with_mock_llm.model_manager.generate_json.return_value = mock_decision_response

        decision_input = {
            "original_query": "Test",
            "perception": {},
            "state": {},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
        }

        await decision_with_mock_llm.run(decision_input, session=mock_session)

        mock_session.add_message.assert_called_once()
        call_kwargs = mock_session.add_message.call_args
        assert call_kwargs[1]["role"] == "assistant"
        assert "decision" in call_kwargs[1]["metadata"]["module"]

    @pytest.mark.asyncio
    async def test_run_handles_llm_error(self, decision_with_mock_llm):
        """Test that run returns fallback on LLM error."""
        decision_with_mock_llm.model_manager.generate_json.side_effect = Exception("LLM Error")

        decision_input = {
            "original_query": "Test query",
            "perception": {"pipeline_stage": "setup", "entities": {}},
            "state": {},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
        }

        result = await decision_with_mock_llm.run(decision_input)

        assert result["strategy"] == "sequential"
        assert "Fallback" in result["reasoning"]
        assert "plan_graph" in result


class TestDecisionNormalization:
    """Tests for Decision output normalization."""

    @pytest.fixture
    def decision(self, tmp_path, decision_prompt_template):
        """Create a Decision instance."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(decision_prompt_template)
        return Decision(str(prompt_file))

    def test_normalize_adds_missing_fields(self, decision):
        """Test that normalization adds missing required fields."""
        output = {}
        normalized = decision._normalize_output(output)

        assert "strategy" in normalized
        assert "reasoning" in normalized
        assert "plan_graph" in normalized
        assert "nodes" in normalized["plan_graph"]
        assert "next_step_id" in normalized
        assert "code_variants" in normalized

    def test_normalize_preserves_existing_fields(self, decision):
        """Test that normalization preserves existing fields."""
        output = {
            "strategy": "parallel",
            "reasoning": "Execute in parallel for speed",
            "plan_graph": {"nodes": [{"id": "0", "description": "Test", "tool": "test_tool"}]},
        }
        normalized = decision._normalize_output(output)

        assert normalized["strategy"] == "parallel"
        assert normalized["reasoning"] == "Execute in parallel for speed"

    def test_normalize_ensures_node_defaults(self, decision):
        """Test that normalization adds defaults to nodes."""
        output = {"plan_graph": {"nodes": [{"id": "0"}]}}  # Minimal node
        normalized = decision._normalize_output(output)

        node = normalized["plan_graph"]["nodes"][0]
        assert node["id"] == "0"
        assert node["description"] == ""
        assert node["tool"] is None
        assert node["args"] == {}
        assert node["depends_on"] == []

    def test_normalize_sets_next_step_id(self, decision):
        """Test that normalization sets next_step_id from first node."""
        output = {
            "plan_graph": {
                "nodes": [
                    {"id": "start"},
                    {"id": "next"},
                ]
            },
            "next_step_id": "",  # Empty
        }
        normalized = decision._normalize_output(output)

        assert normalized["next_step_id"] == "start"

    def test_normalize_handles_empty_nodes(self, decision):
        """Test that normalization handles empty nodes list."""
        output = {"plan_graph": {"nodes": []}}
        normalized = decision._normalize_output(output)

        assert normalized["plan_graph"]["nodes"] == []
        assert normalized["next_step_id"] == "0"  # Default


class TestDecisionFallback:
    """Tests for Decision fallback behavior."""

    @pytest.fixture
    def decision(self, tmp_path, decision_prompt_template):
        """Create a Decision instance."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(decision_prompt_template)
        return Decision(str(prompt_file))

    def test_fallback_output_structure(self, decision):
        """Test fallback output has correct structure."""
        decision_input = {
            "perception": {
                "pipeline_stage": "setup",
                "entities": {"project_path": "/test"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        assert fallback["strategy"] == "sequential"
        assert "Fallback" in fallback["reasoning"]
        assert "plan_graph" in fallback
        assert len(fallback["plan_graph"]["nodes"]) > 0
        assert fallback["next_step_id"] == "0"

    def test_fallback_for_setup_stage(self, decision):
        """Test fallback for setup stage."""
        decision_input = {
            "perception": {
                "pipeline_stage": "setup",
                "entities": {"project_path": "/test"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        node = fallback["plan_graph"]["nodes"][0]
        assert node["tool"] == "analyze_project_config"

    def test_fallback_for_config_stage(self, decision):
        """Test fallback for config stage."""
        decision_input = {
            "perception": {
                "pipeline_stage": "config",
                "entities": {"project_path": "/test"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        node = fallback["plan_graph"]["nodes"][0]
        assert node["tool"] == "create_hydra_config"

    def test_fallback_for_training_stage(self, decision):
        """Test fallback for training stage."""
        decision_input = {
            "perception": {
                "pipeline_stage": "training",
                "entities": {"experiment_name": "my_exp"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        node = fallback["plan_graph"]["nodes"][0]
        assert node["tool"] == "init_mlflow_experiment"

    def test_fallback_uses_project_path(self, decision):
        """Test fallback uses project_path from entities."""
        decision_input = {
            "perception": {
                "pipeline_stage": "setup",
                "entities": {"project_path": "/custom/path"},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        node = fallback["plan_graph"]["nodes"][0]
        assert node["args"]["project_path"] == "/custom/path"

    def test_fallback_defaults_project_path(self, decision):
        """Test fallback defaults project_path to '.'."""
        decision_input = {
            "perception": {
                "pipeline_stage": "setup",
                "entities": {},
            }
        }
        fallback = decision._get_fallback_output(decision_input)

        node = fallback["plan_graph"]["nodes"][0]
        assert node["args"]["project_path"] == "."


class TestDecisionFormatPrompt:
    """Tests for Decision prompt formatting."""

    @pytest.fixture
    def decision(self, tmp_path, decision_prompt_template):
        """Create a Decision instance."""
        prompt_file = tmp_path / "decision_prompt.txt"
        prompt_file.write_text(decision_prompt_template)
        return Decision(str(prompt_file))

    def test_format_prompt_with_template_vars(self, decision):
        """Test prompt formatting with template variables."""
        decision_input = {
            "original_query": "Set up MLOps",
            "perception": {"pipeline_stage": "setup"},
            "state": {"project_path": "/test"},
            "completed_steps": [],
            "failed_steps": [],
            "experiment_state": {},
        }

        formatted = decision._format_prompt(decision_input)

        assert "Set up MLOps" in formatted
        assert "setup" in formatted

    def test_format_prompt_fallback_on_error(self, tmp_path):
        """Test prompt formatting falls back on KeyError."""
        # Create decision with minimal template that will fail
        prompt_file = tmp_path / "broken_prompt.txt"
        prompt_file.write_text("Missing: {nonexistent_key}")

        decision = Decision(str(prompt_file))
        decision_input = {"original_query": "Test"}

        # Should not raise, should append as JSON
        formatted = decision._format_prompt(decision_input)
        assert "Input Context" in formatted


# ============================================================================
# build_decision_input Tests
# ============================================================================


class TestBuildDecisionInput:
    """Tests for build_decision_input function."""

    def test_build_decision_input_structure(self, context_manager, mock_perception_response):
        """Test that build_decision_input returns correct structure."""
        result = build_decision_input(
            ctx=context_manager,
            query="Set up MLOps pipeline",
            perception=mock_perception_response,
        )

        assert result["original_query"] == "Set up MLOps pipeline"
        assert result["perception"] == mock_perception_response
        assert "current_time" in result
        assert "run_id" in result
        assert "state" in result
        assert "completed_steps" in result
        assert "failed_steps" in result
        assert "experiment_state" in result
        assert "globals_schema" in result

    def test_build_decision_input_run_id_format(self, context_manager, mock_perception_response):
        """Test that build_decision_input creates proper run_id."""
        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        assert result["run_id"] == "test-session-123-D"

    def test_build_decision_input_state(self, context_manager, mock_perception_response):
        """Test that build_decision_input includes state."""
        context_manager.experiment_state.stage = "training"
        context_manager.project_path = "/custom/path"

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        assert result["state"]["pipeline_stage"] == "training"
        assert result["state"]["project_path"] == "/custom/path"

    def test_build_decision_input_completed_steps(self, context_manager, mock_perception_response):
        """Test that build_decision_input includes completed steps."""
        context_manager.add_step("0", "First step", "CODE", from_node="ROOT")
        context_manager.mark_step_completed("0")

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        completed_indices = [step["index"] for step in result["completed_steps"]]
        assert "0" in completed_indices

    def test_build_decision_input_failed_steps(self, context_manager, mock_perception_response):
        """Test that build_decision_input includes failed steps."""
        context_manager.add_step("0", "Failed step", "CODE", from_node="ROOT")
        context_manager.mark_step_failed("0", "Error")

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        failed_indices = [step["index"] for step in result["failed_steps"]]
        assert "0" in failed_indices

    def test_build_decision_input_globals_schema(self, context_manager, mock_perception_response):
        """Test that build_decision_input includes globals schema."""
        context_manager.globals["config_path"] = "/test/config.yaml"
        context_manager.globals["run_id"] = "run-123"

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        assert "config_path" in result["globals_schema"]
        assert result["globals_schema"]["config_path"]["type"] == "str"

    def test_build_decision_input_truncates_long_globals(
        self, context_manager, mock_perception_response
    ):
        """Test that build_decision_input truncates long global values."""
        context_manager.globals["long_value"] = "x" * 1000

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        preview = result["globals_schema"]["long_value"]["preview"]
        assert len(preview) <= 503  # 500 + "..."
        assert preview.endswith("...")

    def test_build_decision_input_experiment_state(self, context_manager, mock_perception_response):
        """Test that build_decision_input includes experiment state."""
        context_manager.experiment_state.experiment_name = "test_exp"
        context_manager.experiment_state.update_metrics(accuracy=0.85)

        result = build_decision_input(
            ctx=context_manager,
            query="Test",
            perception=mock_perception_response,
        )

        assert result["experiment_state"]["experiment_name"] == "test_exp"
        assert result["experiment_state"]["current_accuracy"] == 0.85
