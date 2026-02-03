"""
Unit tests for the summarizer module.

Tests summary generation, fallback behavior, and session saving.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summarization.summarizer import Summarizer


class TestSummarizer:
    """Tests for Summarizer class."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager."""
        manager = MagicMock()
        manager.generate_text = AsyncMock(return_value="# Test Summary\n\nCompleted successfully.")
        return manager

    @pytest.fixture
    def summarizer(self, tmp_path, mock_model_manager):
        """Create a Summarizer instance with mock dependencies."""
        # Create a temp prompt file
        prompt_file = tmp_path / "summarizer_prompt.txt"
        prompt_file.write_text("Generate a summary for: {context}")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_model_manager):
            s = Summarizer(str(prompt_file))
        return s

    @pytest.fixture
    def mock_context(self):
        """Create a mock context manager."""
        ctx = MagicMock()
        ctx.session_id = "test-session-001"
        ctx.original_query = "Set up MLOps pipeline"
        ctx.project_path = "/test/project"

        # Mock experiment state
        exp_state = MagicMock()
        exp_state.to_dict.return_value = {
            "stage": "completed",
            "current_accuracy": 0.95,
            "target_accuracy": 0.90,
            "improvement_attempt": 1,
            "artifacts_created": ["model.pt", "config.yaml"],
        }
        exp_state.threshold_met.return_value = True
        exp_state.artifacts_created = ["model.pt", "config.yaml"]
        ctx.experiment_state = exp_state

        # Mock graph
        mock_node = MagicMock()
        mock_node.status = "completed"
        ctx.graph = MagicMock()
        ctx.graph.nodes = {"step1": {"data": mock_node}}

        # Mock methods
        ctx.get_completed_steps.return_value = [{"id": "step1", "status": "completed"}]
        ctx.get_failed_steps.return_value = []
        ctx.globals = {"project_path": "/test/project"}
        ctx.get_context_snapshot.return_value = {"snapshot": "data"}
        ctx.attach_summary = MagicMock()

        return ctx

    @pytest.fixture
    def mock_session(self):
        """Create a mock session."""
        session = MagicMock()
        session.add_message = MagicMock()
        return session

    def test_init_with_valid_prompt(self, tmp_path):
        """Test initialization with valid prompt file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt template")

        with patch("summarization.summarizer.get_model_manager"):
            s = Summarizer(str(prompt_file))
        assert s.prompt_template == "Test prompt template"

    def test_init_with_missing_prompt(self, tmp_path):
        """Test initialization with missing prompt file uses default."""
        with patch("summarization.summarizer.get_model_manager"):
            s = Summarizer(str(tmp_path / "nonexistent.txt"))
        assert "Generate a summary" in s.prompt_template

    def test_load_prompt(self, tmp_path):
        """Test prompt loading from file."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom prompt content")

        with patch("summarization.summarizer.get_model_manager"):
            s = Summarizer(str(prompt_file))
        assert s.prompt_template == "Custom prompt content"

    def test_get_default_prompt(self, summarizer):
        """Test default prompt content."""
        prompt = summarizer._get_default_prompt()
        assert "Generate a summary" in prompt
        assert "What was accomplished" in prompt

    @pytest.mark.asyncio
    async def test_run_success(self, summarizer):
        """Test successful summarization run."""
        s_input = {
            "original_query": "Train model",
            "goal_achieved": True,
            "experiment_state": {"stage": "completed"},
        }

        result = await summarizer.run(s_input)

        assert "summary_markdown" in result
        assert result["goal_achieved"] is True
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_run_with_session(self, summarizer, mock_session):
        """Test run with session logging."""
        s_input = {
            "original_query": "Train model",
            "goal_achieved": True,
        }

        await summarizer.run(s_input, session=mock_session)

        mock_session.add_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_llm_error_returns_fallback(self, tmp_path):
        """Test fallback summary when LLM fails."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(side_effect=Exception("LLM Error"))

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            s = Summarizer(str(prompt_file))

        s_input = {
            "original_query": "Train model",
            "experiment_state": {"stage": "training"},
        }

        result = await s.run(s_input)

        assert "summary_markdown" in result
        assert result["goal_achieved"] is False
        assert result["confidence"] == 0.5

    def test_format_prompt(self, summarizer):
        """Test prompt formatting with context."""
        s_input = {"original_query": "Test query", "stage": "completed"}

        formatted = summarizer._format_prompt(s_input)

        assert summarizer.prompt_template in formatted
        assert "Test query" in formatted
        assert "Execution Context" in formatted

    def test_get_fallback_summary(self, summarizer):
        """Test fallback summary generation."""
        s_input = {
            "original_query": "Train a cat classifier",
            "experiment_state": {
                "stage": "training",
                "current_accuracy": 0.85,
                "target_accuracy": 0.90,
                "improvement_attempt": 2,
                "artifacts_created": ["model.pt", "logs/"],
            },
        }

        result = summarizer._get_fallback_summary(s_input)

        assert "MLOps Execution Summary" in result["summary_markdown"]
        assert "Train a cat classifier" in result["summary_markdown"]
        assert "model.pt" in result["summary_markdown"]
        assert result["goal_achieved"] is False
        assert result["confidence"] == 0.5

    def test_get_fallback_summary_empty_artifacts(self, summarizer):
        """Test fallback summary with no artifacts."""
        s_input = {
            "original_query": "Train model",
            "experiment_state": {"artifacts_created": []},
        }

        result = summarizer._get_fallback_summary(s_input)

        assert "None" in result["summary_markdown"]

    def test_format_artifacts_with_list(self, summarizer):
        """Test artifact formatting with list."""
        artifacts = ["model.pt", "config.yaml", "logs/training.log"]
        formatted = summarizer._format_artifacts(artifacts)

        assert "- `model.pt`" in formatted
        assert "- `config.yaml`" in formatted
        assert "- `logs/training.log`" in formatted

    def test_format_artifacts_empty(self, summarizer):
        """Test artifact formatting with empty list."""
        formatted = summarizer._format_artifacts([])
        assert "- None" in formatted

    def test_format_artifacts_limit(self, summarizer):
        """Test artifact formatting limits to 10."""
        artifacts = [f"file{i}.txt" for i in range(15)]
        formatted = summarizer._format_artifacts(artifacts)

        lines = [line for line in formatted.split("\n") if line.strip()]
        assert len(lines) == 10

    @pytest.mark.asyncio
    async def test_summarize_full_flow(self, summarizer, mock_context, mock_session):
        """Test full summarization flow."""
        perception = {
            "original_goal_achieved": True,
            "routing": "summarize",
        }

        with patch.object(summarizer, "_save_session") as mock_save:
            result = await summarizer.summarize(
                query="Set up MLOps pipeline",
                ctx=mock_context,
                perception=perception,
                session=mock_session,
            )

        assert "summary_markdown" in result
        mock_context.attach_summary.assert_called_once()
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_marks_pending_as_skipped(self, summarizer, mock_context, mock_session):
        """Test that pending steps are marked as skipped."""
        # Set up a pending step
        pending_node = MagicMock()
        pending_node.status = "pending"
        mock_context.graph.nodes = {"step1": {"data": pending_node}}

        perception = {"original_goal_achieved": False}

        with patch.object(summarizer, "_save_session"):
            await summarizer.summarize(
                query="Test",
                ctx=mock_context,
                perception=perception,
            )

        assert pending_node.status == "skipped"

    @pytest.mark.asyncio
    async def test_summarize_builds_correct_input(self, tmp_path, mock_context, mock_session):
        """Test that summarize builds correct input for run."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        captured_input = None

        async def capture_run(s_input, session=None):
            nonlocal captured_input
            captured_input = s_input
            return {"summary_markdown": "test", "goal_achieved": True}

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(return_value="test")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            s = Summarizer(str(prompt_file))
            s.run = capture_run

        perception = {"original_goal_achieved": True}

        with patch.object(s, "_save_session"):
            await s.summarize(
                query="Test query",
                ctx=mock_context,
                perception=perception,
            )

        assert captured_input is not None
        assert captured_input["original_query"] == "Test query"
        assert "experiment_state" in captured_input
        assert "completed_steps" in captured_input
        assert "perception" in captured_input

    def test_save_session(self, summarizer, mock_context, mock_session, tmp_path):
        """Test session saving to disk."""
        summary = {"summary_markdown": "Test summary", "goal_achieved": True}

        with patch.object(Path, "mkdir"):
            with patch("builtins.open", create=True) as mock_open:
                mock_file = MagicMock()
                mock_open.return_value.__enter__.return_value = mock_file

                summarizer._save_session(mock_context, mock_session, summary)

                mock_open.assert_called_once()
                mock_file.write.assert_called()

    def test_save_session_error_handling(self, summarizer, mock_context, mock_session, capsys):
        """Test session saving handles errors gracefully."""
        summary = {"summary_markdown": "Test", "goal_achieved": True}

        with patch.object(Path, "mkdir", side_effect=PermissionError("No access")):
            summarizer._save_session(mock_context, mock_session, summary)

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Could not save" in captured.out


class TestSummarizerEdgeCases:
    """Tests for edge cases and error scenarios."""

    @pytest.fixture
    def summarizer(self, tmp_path):
        """Create a basic summarizer."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test prompt")

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(return_value="Summary text")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            return Summarizer(str(prompt_file))

    def test_fallback_summary_missing_fields(self, summarizer):
        """Test fallback with minimal input."""
        s_input = {}
        result = summarizer._get_fallback_summary(s_input)

        assert "summary_markdown" in result
        assert "MLOps operation" in result["summary_markdown"]

    def test_format_prompt_with_non_serializable(self, summarizer):
        """Test prompt formatting with non-JSON-serializable content."""
        s_input = {
            "query": "test",
            "datetime_obj": datetime.now(),  # Not JSON serializable normally
        }

        # Should handle with default=str
        formatted = summarizer._format_prompt(s_input)
        assert "test" in formatted

    @pytest.mark.asyncio
    async def test_run_with_empty_response(self, tmp_path):
        """Test handling empty LLM response."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Test")

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(return_value="")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            s = Summarizer(str(prompt_file))

        result = await s.run({"query": "test"})
        assert result["summary_markdown"] == ""

    def test_format_artifacts_with_special_chars(self, summarizer):
        """Test artifact formatting with special characters."""
        artifacts = ["path/with spaces/file.txt", "file`with`backticks.txt"]
        formatted = summarizer._format_artifacts(artifacts)

        # Should be wrapped in backticks
        assert "`path/with spaces/file.txt`" in formatted


class TestSummarizerIntegration:
    """Integration-style tests for summarizer."""

    @pytest.fixture
    def full_context(self):
        """Create a full mock context."""
        ctx = MagicMock()
        ctx.session_id = "integration-test-001"
        ctx.original_query = "Train a ResNet model on CIFAR-10"
        ctx.project_path = "/projects/cifar10"

        exp_state = MagicMock()
        exp_state.to_dict.return_value = {
            "stage": "completed",
            "current_accuracy": 0.92,
            "target_accuracy": 0.85,
            "improvement_attempt": 3,
            "best_accuracy": 0.92,
            "mlflow_run_id": "run-12345",
            "experiment_name": "cifar10_resnet",
            "current_config": {
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 100,
            },
            "artifacts_created": [
                "models/resnet_final.pt",
                "configs/train.yaml",
                "logs/training.log",
            ],
            "improvement_history": [
                {"attempt": 1, "accuracy_before": 0.75, "accuracy_after": 0.85},
                {"attempt": 2, "accuracy_before": 0.85, "accuracy_after": 0.90},
                {"attempt": 3, "accuracy_before": 0.90, "accuracy_after": 0.92},
            ],
        }
        exp_state.threshold_met.return_value = True
        exp_state.artifacts_created = ["models/resnet_final.pt"]
        ctx.experiment_state = exp_state

        completed_node = MagicMock()
        completed_node.status = "completed"
        ctx.graph = MagicMock()
        ctx.graph.nodes = {
            "init": {"data": completed_node},
            "train": {"data": completed_node},
            "evaluate": {"data": completed_node},
        }

        ctx.get_completed_steps.return_value = [
            {"id": "init", "description": "Initialize MLflow"},
            {"id": "train", "description": "Train model"},
            {"id": "evaluate", "description": "Evaluate model"},
        ]
        ctx.get_failed_steps.return_value = []
        ctx.globals = {"project_path": "/projects/cifar10", "memory": "redacted"}
        ctx.get_context_snapshot.return_value = {"full": "snapshot"}
        ctx.attach_summary = MagicMock()

        return ctx

    @pytest.mark.asyncio
    async def test_full_pipeline_summary(self, tmp_path, full_context):
        """Test summarization of a complete pipeline."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("""Generate a comprehensive summary of the MLOps pipeline execution.
Include:
- What was accomplished
- Key metrics achieved
- Any issues encountered
- Recommendations

Context:
{context}""")

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(return_value="""# MLOps Pipeline Summary

## Achievements
- Successfully trained ResNet model on CIFAR-10
- Achieved 92% accuracy (target: 85%)
- Completed 3 improvement iterations

## Metrics
- Final Accuracy: 92%
- Training completed in 3 iterations

## Recommendations
- Consider deploying the model
- Set up continuous monitoring
""")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            s = Summarizer(str(prompt_file))

        perception = {
            "original_goal_achieved": True,
            "routing": "summarize",
            "stage": "completed",
        }

        with patch.object(s, "_save_session"):
            result = await s.summarize(
                query="Train a ResNet model on CIFAR-10",
                ctx=full_context,
                perception=perception,
            )

        assert "summary_markdown" in result
        assert result["goal_achieved"] is True
        full_context.attach_summary.assert_called_once_with(result)

    @pytest.mark.asyncio
    async def test_failed_pipeline_summary(self, tmp_path):
        """Test summarization of a failed pipeline."""
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Summarize: {context}")

        ctx = MagicMock()
        ctx.session_id = "failed-session"
        ctx.original_query = "Train model"
        ctx.project_path = "/test"

        exp_state = MagicMock()
        exp_state.to_dict.return_value = {
            "stage": "failed",
            "current_accuracy": 0.50,
            "target_accuracy": 0.90,
        }
        exp_state.threshold_met.return_value = False
        exp_state.artifacts_created = []
        ctx.experiment_state = exp_state

        failed_node = MagicMock()
        failed_node.status = "failed"
        ctx.graph = MagicMock()
        ctx.graph.nodes = {"train": {"data": failed_node}}

        ctx.get_completed_steps.return_value = []
        ctx.get_failed_steps.return_value = [{"id": "train", "error": "OOM"}]
        ctx.globals = {}
        ctx.get_context_snapshot.return_value = {}
        ctx.attach_summary = MagicMock()

        mock_manager = MagicMock()
        mock_manager.generate_text = AsyncMock(return_value="Pipeline failed due to OOM error.")

        with patch("summarization.summarizer.get_model_manager", return_value=mock_manager):
            s = Summarizer(str(prompt_file))

        perception = {"original_goal_achieved": False}

        with patch.object(s, "_save_session"):
            result = await s.summarize(
                query="Train model",
                ctx=ctx,
                perception=perception,
            )

        # Goal should be False since threshold not met and perception says not achieved
        assert result["goal_achieved"] is False
