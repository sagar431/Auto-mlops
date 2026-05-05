#!/usr/bin/env python3
"""
Tests for agent/agent_loop.py - MLOps agent orchestration loop.

Run with: pytest tests/unit/test_agent_loop.py -v
"""

import json
import pickle
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mcp_mlops_tools
from agent.agent_loop import (
    AgentLoop,
    Route,
    StepExecutionError,
    StepExecutionTracker,
    StepType,
    run_mlops_agent,
)
from mcp_mlops_tools import (
    _find_available_port,
    create_litserve_api,
    select_best_model_artifact,
)
from mcp_mlops_tools import (
    test_litserve_prediction_endpoint as call_litserve_prediction_endpoint,
)
from workflow.registry import (
    ApprovalRecord,
    ArtifactManifest,
    ArtifactManifestEntry,
    VerificationResult,
    WorkflowStatus,
)


def test_select_best_model_artifact_finds_training_output_pickle(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "model.pkl").write_bytes(b"pickle-model")

    result = select_best_model_artifact(str(tmp_path))

    assert result["success"] is True
    assert result["model_path"] == "outputs/model.pkl"
    assert result["model_type"] == "tabular_regressor"
    assert result["artifact_manifest"]["entries"][0]["path"] == "outputs/model.pkl"


def test_create_litserve_api_generates_tabular_server_for_pickle_artifact(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "model.pkl").write_bytes(b"pickle-model")
    (outputs_dir / "scaler.pkl").write_bytes(b"pickle-scaler")

    result = create_litserve_api(
        project_path=str(tmp_path),
        model_path="outputs/model.pkl",
        model_name="model",
    )

    server_path = tmp_path / "deployment" / "litserve" / "server.py"
    server_code = server_path.read_text()
    assert result["success"] is True
    assert "pickle.load" in server_code
    assert "torch.jit.load" not in server_code
    assert "outputs/scaler.pkl" in server_code
    assert "_litserve_server._MCP_AVAILABLE = False" in server_code
    assert "ModelAPI(max_batch_size=64, batch_timeout=0.05)" in server_code
    assert "server = ls.LitServer(\n        api,\n        accelerator=\"auto\",\n        workers_per_device=4" in server_code
    assert "array = self.scaler.transform(array.reshape(1, -1))[0]" in server_code


def test_litserve_prediction_endpoint_uses_artifact_feature_count_for_default_payload(
    tmp_path, monkeypatch
):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "scaler.pkl").write_bytes(
        pickle.dumps(SimpleNamespace(n_features_in_=8))
    )
    captured: dict[str, dict] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit):
            return b'{"predictions":[0.0]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = call_litserve_prediction_endpoint(str(tmp_path))

    assert result["prediction_passed"] is True
    assert captured["payload"] == {"input": [0.0] * 8}


def test_litserve_health_endpoint_retries_until_ready(monkeypatch):
    attempts = {"count": 0}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit):
            return b"ok"

    def fake_urlopen(url, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise mcp_mlops_tools.urllib.error.URLError("connection refused")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = mcp_mlops_tools.test_litserve_health_endpoint(
        "/home/ubuntu/Auto-mlops",
        endpoint_url="http://127.0.0.1:8001",
        timeout_seconds=3.0,
    )

    assert result["health_passed"] is True
    assert attempts["count"] == 2


def test_find_available_port_skips_bound_requested_port(monkeypatch):
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_is_port_available",
        lambda host, port: port == 8001,
    )

    selected_port = _find_available_port("127.0.0.1", 8000, attempts=2)
    assert selected_port == 8001


def _write_session_06_training_project(project_path):
    configs = project_path / "configs"
    (configs / "model").mkdir(parents=True)
    (configs / "data").mkdir()
    (project_path / "src" / "models").mkdir(parents=True)
    (project_path / "outputs").mkdir()
    (project_path / "checkpoints").mkdir()
    (project_path / "tests" / "test_train").mkdir(parents=True)
    (project_path / ".dvc").mkdir()

    (configs / "config.yaml").write_text("model:\n  lr: 1e-3\n")
    (configs / "train.yaml").write_text(
        "defaults:\n"
        "  - model: timm_classify\n"
        "  - data: cat_dog\n"
        "trainer:\n"
        "  max_epochs: 5\n"
    )
    (configs / "model" / "timm_classify.yaml").write_text(
        "_target_: src.models.timmclassifier.TimmClassifier\n"
        "model_name: resnet18\n"
    )
    (configs / "data" / "cat_dog.yaml").write_text(
        "_target_: src.data.datamodule.CatDogDataModule\n"
        "data_dir: data/catdog_test\n"
    )
    (project_path / "src" / "train.py").write_text(
        "import hydra\n\n"
        "@hydra.main(version_base='1.3', config_path='../configs', config_name='train.yaml')\n"
        "def main(cfg):\n"
        "    return None\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (project_path / "src" / "models" / "timmclassifier.py").write_text(
        "import timm\n"
        "import torch\n"
        "import lightning as L\n"
    )
    (project_path / "pyproject.toml").write_text(
        "[project]\n"
        "dependencies = [\n"
        "  'hydra-core==1.3.2',\n"
        "  'lightning==2.5.0',\n"
        "  'timm==1.0.14',\n"
        "  'torch==2.6.0',\n"
        "]\n"
    )
    (project_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (project_path / "data.dvc").write_text("outs:\n  - path: data/catdog_test\n")
    (project_path / ".dvc" / "config").write_text("[core]\n")
    (project_path / "tests" / "test_train" / "test_training.py").write_text(
        "def test_training_entrypoint_imports():\n"
        "    assert True\n"
    )


def _write_bounded_training_fixture(project_path, script_body):
    _write_session_06_training_project(project_path)
    (project_path / "src" / "train.py").write_text(script_body)


def test_detect_training_project_recognizes_session_06_shape(tmp_path):
    _write_session_06_training_project(tmp_path)

    result = mcp_mlops_tools.detect_training_project(str(tmp_path))

    assert result["success"] is True
    assert result["status"] == "supported"
    assert result["framework_family"] == "pytorch_lightning"
    assert result["model_library"] == "timm"
    assert result["config_system"] == "hydra"
    assert result["data_versioning"] == "dvc"
    assert result["training_entrypoint"] == "src/train.py"
    assert "configs/train.yaml" in result["likely_config_files"]
    assert "tests/test_train/test_training.py" in result["test_files"]
    assert result["missing_required_pieces"] == []
    assert result["next_actions"] == []
    assert result["test_command"] == "python -m pytest tests -q"
    assert {
        "training_entrypoint_detected",
        "hydra_config_detected",
        "dvc_or_data_evidence_detected",
        "pytorch_timm_signals_detected",
        "test_command_detected",
        "output_artifact_candidates_detected",
    }.issubset({item["check_name"] for item in result["verification_results"]})
    manifest_entries = result["artifact_manifest"]["entries"]
    assert {
        (entry["artifact_type"], entry["state"], entry["path"])
        for entry in manifest_entries
    }.issuperset(
        {
            ("training_entrypoint", "external", "src/train.py"),
            ("configuration", "external", "configs/train.yaml"),
            ("test_suite", "external", "tests/test_train/test_training.py"),
        }
    )


def test_detect_training_project_blocks_ambiguous_entrypoints(tmp_path):
    _write_session_06_training_project(tmp_path)
    (tmp_path / "train.py").write_text(
        "import hydra\n"
        "@hydra.main(version_base='1.3', config_path='configs', config_name='train')\n"
        "def main(cfg):\n"
        "    return None\n"
    )

    result = mcp_mlops_tools.detect_training_project(str(tmp_path))

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert set(result["training_entrypoint_candidates"]) == {"src/train.py", "train.py"}
    assert "training_entrypoint" in result["missing_required_pieces"]
    assert "multiple candidates" in " ".join(result["next_actions"])
    assert "training_entrypoint_detected" not in {
        item["check_name"] for item in result["verification_results"]
    }


def test_run_bounded_training_captures_metrics_logs_duration_and_artifacts(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "import json\n"
        "from pathlib import Path\n"
        "Path('checkpoints').mkdir(exist_ok=True)\n"
        "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
        "print(json.dumps({'metrics': {'accuracy': 0.91, 'loss': 0.12}}))\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=["trainer.fast_dev_run=true"],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert result["exit_code"] == 0
    assert result["metrics"]["accuracy"] == 0.91
    assert result["duration_seconds"] >= 0
    assert result["command"][0].endswith("python")
    assert "trainer.max_epochs=1" in result["effective_overrides"]
    assert "device=cpu" in result["effective_overrides"]
    assert "data.subset=4" in result["effective_overrides"]
    assert result["log_path"].endswith("training.log")
    assert result["config_snapshot_path"].endswith("config_snapshot.yaml")
    assert "checkpoints/model.ckpt" in result["checkpoint_artifact_paths"]

    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["bounded_training_controls_present"]["passed"] is True
    assert verification_by_name["bounded_training_command_completed"]["passed"] is True
    assert verification_by_name["training_metric_captured"]["passed"] is True
    assert verification_by_name["training_artifact_captured"]["passed"] is True
    assert verification_by_name["training_run_evidence_captured"]["passed"] is True
    assert {
        (entry["artifact_type"], entry["state"], entry["path"])
        for entry in result["artifact_manifest"]["entries"]
    }.issuperset(
        {
            ("training_log", "generated", result["log_path"]),
            ("config_snapshot", "generated", result["config_snapshot_path"]),
            ("checkpoint_or_model_artifact", "generated", "checkpoints/model.ckpt"),
        }
    )


def test_run_bounded_training_records_nonzero_exit_as_failed(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "import sys\n"
        "print('about to fail')\n"
        "print('boom', file=sys.stderr)\n"
        "raise SystemExit(3)\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=[],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["exit_code"] == 3
    assert "non-zero" in result["failure_reason"]
    assert "about to fail" in result["stdout_summary"]
    assert "boom" in result["stderr_summary"]
    assert result["next_actions"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["bounded_training_command_completed"]["passed"] is False


def test_run_bounded_training_zero_exit_without_metric_or_artifact_does_not_succeed(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "print('finished without metric or artifact')\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=[],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["exit_code"] == 0
    assert "target metric" in result["failure_reason"]
    assert "checkpoint" in result["failure_reason"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["training_metric_captured"]["passed"] is False
    assert verification_by_name["training_artifact_captured"]["passed"] is False


# ============================================================================
# Route Constants Tests
# ============================================================================


class TestRouteConstants:
    """Tests for Route constants."""

    def test_route_summarize(self):
        """Test SUMMARIZE route constant."""
        assert Route.SUMMARIZE == "summarize"

    def test_route_decision(self):
        """Test DECISION route constant."""
        assert Route.DECISION == "decision"

    def test_route_improve(self):
        """Test IMPROVE route constant."""
        assert Route.IMPROVE == "improve"

    def test_route_deploy(self):
        """Test DEPLOY route constant."""
        assert Route.DEPLOY == "deploy"


# ============================================================================
# StepType Constants Tests
# ============================================================================


class TestStepTypeConstants:
    """Tests for StepType constants."""

    def test_step_type_root(self):
        """Test ROOT step type constant."""
        assert StepType.ROOT == "ROOT"

    def test_step_type_code(self):
        """Test CODE step type constant."""
        assert StepType.CODE == "CODE"

    def test_step_type_improve(self):
        """Test IMPROVE step type constant."""
        assert StepType.IMPROVE == "IMPROVE"

    def test_step_type_deploy(self):
        """Test DEPLOY step type constant."""
        assert StepType.DEPLOY == "DEPLOY"


# ============================================================================
# StepExecutionError Tests
# ============================================================================


class TestStepExecutionError:
    """Tests for StepExecutionError exception."""

    def test_create_step_execution_error(self):
        """Test creating StepExecutionError."""
        error = StepExecutionError("step_1", "Something went wrong")
        assert error.step_id == "step_1"
        assert error.error_message == "Something went wrong"
        assert "step_1" in str(error)
        assert "Something went wrong" in str(error)

    def test_step_execution_error_inheritance(self):
        """Test StepExecutionError is an Exception."""
        error = StepExecutionError("step_1", "Error")
        assert isinstance(error, Exception)

    def test_step_execution_error_message_format(self):
        """Test error message format."""
        error = StepExecutionError("test_step", "Test error")
        expected = "Step 'test_step' failed: Test error"
        assert str(error) == expected


# ============================================================================
# StepExecutionTracker Tests
# ============================================================================


class TestStepExecutionTracker:
    """Tests for StepExecutionTracker class."""

    def test_create_tracker_with_defaults(self):
        """Test creating tracker with default values."""
        tracker = StepExecutionTracker()
        assert tracker.max_steps == 15
        assert tracker.max_retries == 3
        assert tracker.tries == 0
        assert tracker.attempts == {}

    def test_create_tracker_with_custom_values(self):
        """Test creating tracker with custom values."""
        tracker = StepExecutionTracker(max_steps=10, max_retries=5)
        assert tracker.max_steps == 10
        assert tracker.max_retries == 5

    def test_increment(self):
        """Test incrementing tries counter."""
        tracker = StepExecutionTracker()
        assert tracker.tries == 0
        tracker.increment()
        assert tracker.tries == 1
        tracker.increment()
        assert tracker.tries == 2

    def test_record_failure(self):
        """Test recording step failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 1
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 2
        tracker.record_failure("step_2")
        assert tracker.attempts["step_2"] == 1

    def test_retry_step_id_no_failures(self):
        """Test retry step ID with no failures."""
        tracker = StepExecutionTracker()
        assert tracker.retry_step_id("step_1") == "step_1"

    def test_retry_step_id_with_failures(self):
        """Test retry step ID generation with failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F1"
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F2"

    def test_should_continue_within_limits(self):
        """Test should_continue when within limits."""
        tracker = StepExecutionTracker(max_steps=3)
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True

    def test_should_continue_at_limit(self):
        """Test should_continue when at limit."""
        tracker = StepExecutionTracker(max_steps=3)
        tracker.increment()
        tracker.increment()
        tracker.increment()
        assert tracker.should_continue() is False

    def test_has_exceeded_retries_within_limit(self):
        """Test has_exceeded_retries when within limit."""
        tracker = StepExecutionTracker(max_retries=2)
        assert tracker.has_exceeded_retries("step_1") is False
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is False

    def test_has_exceeded_retries_at_limit(self):
        """Test has_exceeded_retries when at limit."""
        tracker = StepExecutionTracker(max_retries=2)
        tracker.record_failure("step_1")
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is True

    def test_is_circuit_open_initially_false(self):
        """Test circuit is initially closed."""
        tracker = StepExecutionTracker()
        assert tracker.is_circuit_open() is False

    def test_get_circuit_stats(self):
        """Test getting circuit stats."""
        tracker = StepExecutionTracker()
        stats = tracker.get_circuit_stats()
        assert "state" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "rejected_calls" in stats
        assert "consecutive_failures" in stats
        assert "consecutive_successes" in stats
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0

    def test_reset_circuit(self):
        """Test resetting circuit breaker."""
        tracker = StepExecutionTracker()
        tracker.circuit_breaker._stats.total_calls = 100
        tracker.circuit_breaker._stats.failed_calls = 50
        tracker.reset_circuit()
        stats = tracker.get_circuit_stats()
        assert stats["total_calls"] == 0
        assert stats["failed_calls"] == 0

    def test_circuit_breaker_property(self):
        """Test circuit breaker property access."""
        tracker = StepExecutionTracker()
        assert tracker.circuit_breaker is not None
        assert tracker.circuit_breaker.name == "step_execution"


# ============================================================================
# AgentLoop Initialization Tests
# ============================================================================


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create temporary prompts directory with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text(
            "Improve: target={target_accuracy} current={current_accuracy}"
        )

        return str(prompts_dir)

    def test_create_agent_loop_with_defaults(self, mock_prompts_dir):
        """Test creating AgentLoop with default values."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        assert agent.status == "idle"
        assert agent.profile == "default"
        assert agent.on_event is None
        assert agent.tools_module is None

    def test_create_agent_loop_with_custom_profile(self, mock_prompts_dir):
        """Test creating AgentLoop with custom profile."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir, profile="custom")
        assert agent.profile == "custom"

    def test_create_agent_loop_with_on_event(self, mock_prompts_dir):
        """Test creating AgentLoop with event callback."""

        async def callback(event_type, data):
            pass

        agent = AgentLoop(prompts_dir=mock_prompts_dir, on_event=callback)
        assert agent.on_event is callback

    def test_create_agent_loop_with_tools_module(self, mock_prompts_dir):
        """Test creating AgentLoop with custom tools module."""
        mock_tools = MagicMock()
        agent = AgentLoop(prompts_dir=mock_prompts_dir, tools_module=mock_tools)
        assert agent.tools_module is mock_tools

    def test_load_prompt_returns_content(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns file content."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        prompt_file = tmp_path / "test.txt"
        prompt_file.write_text("Test content")
        result = agent._load_prompt(prompt_file)
        assert result == "Test content"

    def test_load_prompt_missing_file_returns_empty(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns empty string for missing file."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        result = agent._load_prompt(tmp_path / "nonexistent.txt")
        assert result == ""


# ============================================================================
# AgentLoop Session Initialization Tests
# ============================================================================


class TestAgentLoopSessionInit:
    """Tests for AgentLoop session initialization."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_initialize_session_creates_session_id(self, agent):
        """Test that _initialize_session creates a session ID."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session_id is not None
        assert len(agent.session_id) > 0

    def test_initialize_session_sets_query(self, agent):
        """Test that _initialize_session sets the query."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.query == "Test query"

    def test_initialize_session_creates_context(self, agent):
        """Test that _initialize_session creates context manager."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.ctx is not None
        assert agent.ctx.project_path == "/test/path"

    def test_initialize_session_sets_accuracy_threshold(self, agent):
        """Test that _initialize_session sets accuracy threshold."""
        agent._initialize_session("Test query", "/test/path", 0.90)
        assert agent.ctx.experiment_state.target_accuracy == 0.90

    def test_initialize_session_creates_agent_session(self, agent):
        """Test that _initialize_session creates AgentSession."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session is not None
        assert agent.session.original_query == "Test query"

    def test_initialize_session_initializes_placeholders(self, agent):
        """Test that _initialize_session initializes placeholders."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.p_out == {}
        assert agent.code_variants == {}
        assert agent.next_step_id == "0"
        assert agent.final_output == ""


# ============================================================================
# AgentLoop Routing Tests
# ============================================================================


class TestAgentLoopRouting:
    """Tests for AgentLoop routing logic."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_should_summarize_true_when_goal_achieved(self, agent):
        """Test _should_summarize returns True when goal achieved."""
        agent.p_out = {"original_goal_achieved": True, "route": "decision"}
        assert agent._should_summarize() is True

    def test_should_summarize_true_when_route_is_summarize(self, agent):
        """Test _should_summarize returns True when route is summarize."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.SUMMARIZE}
        assert agent._should_summarize() is True

    def test_should_summarize_false_when_not_achieved(self, agent):
        """Test _should_summarize returns False when not achieved."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.DECISION}
        assert agent._should_summarize() is False

    def test_needs_improvement_when_below_threshold(self, agent):
        """Test _needs_improvement returns True when below threshold."""
        agent._initialize_session("Test", "/test", 0.90)
        agent.ctx.experiment_state.current_accuracy = 0.80
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is True

    def test_needs_improvement_false_when_at_threshold(self, agent):
        """Test _needs_improvement returns False when at threshold."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.current_accuracy = 0.85
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_improvement_false_when_no_accuracy(self, agent):
        """Test _needs_improvement returns False when no accuracy recorded."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_deployment_when_route_is_deploy(self, agent):
        """Test _needs_deployment returns True when route is deploy."""
        agent.p_out = {"route": Route.DEPLOY}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_stage_is_deploy(self, agent):
        """Test _needs_deployment returns True when stage is deploy."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "deploy"}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_deployment_target_set(self, agent):
        """Test _needs_deployment returns True when deployment target set."""
        agent.p_out = {"route": Route.DECISION, "entities": {"deployment_target": "gradio"}}
        assert agent._needs_deployment() is True

    def test_needs_deployment_false_when_not_deploying(self, agent):
        """Test _needs_deployment returns False when not deploying."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "training", "entities": {}}
        assert agent._needs_deployment() is False


# ============================================================================
# AgentLoop Event Emission Tests
# ============================================================================


class TestAgentLoopEventEmission:
    """Tests for AgentLoop event emission."""

    @pytest.fixture
    def agent_with_event_handler(self, tmp_path):
        """Create AgentLoop with event handler."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=capture_event)
        agent._events = events
        return agent

    @pytest.mark.asyncio
    async def test_emit_calls_callback(self, agent_with_event_handler):
        """Test _emit calls the event callback."""
        await agent_with_event_handler._emit("test_event", {"key": "value"})
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["type"] == "test_event"
        assert agent_with_event_handler._events[0]["data"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_with_empty_data(self, agent_with_event_handler):
        """Test _emit with empty data."""
        await agent_with_event_handler._emit("test_event")
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["data"] == {}

    @pytest.mark.asyncio
    async def test_emit_handles_callback_error(self, tmp_path):
        """Test _emit handles callback error gracefully."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        async def failing_callback(event_type, data):
            raise ValueError("Callback failed")

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=failing_callback)
        # Should not raise
        await agent._emit("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_without_callback(self, tmp_path):
        """Test _emit without callback does nothing."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        # Should not raise
        await agent._emit("test_event", {"key": "value"})


# ============================================================================
# AgentLoop Pick Next Step Tests
# ============================================================================


class TestAgentLoopPickNextStep:
    """Tests for AgentLoop _pick_next_step method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_pick_next_step_returns_first_pending(self, agent):
        """Test _pick_next_step returns first pending step."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        result = agent._pick_next_step()
        assert result == "0"

    def test_pick_next_step_skips_completed(self, agent):
        """Test _pick_next_step skips completed steps."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result == "1"

    def test_pick_next_step_returns_none_when_all_completed(self, agent):
        """Test _pick_next_step returns None when all completed."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result is None

    def test_pick_next_step_returns_none_when_no_steps(self, agent):
        """Test _pick_next_step returns None when no steps."""
        agent._initialize_session("Test", "/test", 0.85)
        result = agent._pick_next_step()
        assert result is None


# ============================================================================
# AgentLoop Run Integration Tests
# ============================================================================


class TestAgentLoopRun:
    """Integration tests for AgentLoop.run method."""

    @pytest.fixture
    def mock_agent(self, tmp_path, mock_llm):
        """Create AgentLoop with mocked dependencies."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        return agent

    @pytest.mark.asyncio
    async def test_run_sets_status_to_running(self, mock_agent, mock_llm):
        """Test run sets status to running."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path")
            # Status should be success at the end
            assert mock_agent.status in ("running", "success")

    @pytest.mark.asyncio
    async def test_run_initializes_session(self, mock_agent, mock_llm):
        """Test run initializes session."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path", 0.85)
            assert mock_agent.session is not None
            assert mock_agent.ctx is not None

    @pytest.mark.asyncio
    async def test_run_returns_summary_when_goal_achieved(self, mock_agent, mock_llm):
        """Test run returns summary when goal achieved."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Final summary"}
            result = await mock_agent.run("Test query", "/test/path")
            assert result == "Final summary"
            assert mock_agent.status == "success"

    @pytest.mark.asyncio
    async def test_run_selects_setup_workflow_before_perception_or_decision(self, mock_agent):
        """Test registry workflow selection runs before prompt-authored planning."""
        expected_step_ids = [
            step.step_id for step in mock_agent.workflow_registry.get("setup_pipeline").steps
        ]

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before selection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Set up MLOps for this project", "/test/path")

        assert mock_agent.workflow_selection.workflow_id == "setup_pipeline"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.get_pending_steps() == expected_step_ids
        for step in mock_agent.workflow_registry.get("setup_pipeline").steps:
            runtime_step = mock_agent.ctx.graph.nodes[step.step_id]["data"]
            assert runtime_step.tool in step.tool_functions
        assert "setup_pipeline" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_projects_litserve_gpu_workflow_and_blocks_before_risky_actions(
        self, mock_agent
    ):
        """Test LitServe GPU runtime uses registry steps and approval gates."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        expected_step_ids = [
            step.step_id for step in mock_agent.workflow_registry.get("deploy_litserve_gpu").steps
        ]

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before selection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        approval_events = [event for event in events if event["type"] == "approval_required"]
        assert mock_agent.workflow_selection.workflow_id == "deploy_litserve_gpu"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.get_pending_steps() == expected_step_ids
        assert approval_events
        assert approval_events[0]["data"]["workflow_id"] == "deploy_litserve_gpu"
        assert approval_events[0]["data"]["step_id"] == "detect_gpu_cuda"
        assert approval_events[0]["data"]["risk_categories"] == ["uses_gpu"]
        assert "Approval required" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_blocks_missing_entrypoint_or_config_before_training(
        self, mock_agent, tmp_path
    ):
        """Test training requests use registry detection and block unsupported project shape."""
        project_path = tmp_path / "incomplete-training-project"
        project_path.mkdir()
        (project_path / "pyproject.toml").write_text("[project]\ndependencies = ['torch']\n")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before training detection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Detect this training project",
                str(project_path),
            )

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert completed_tool_steps == ["detect_training_project"]
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "training_entrypoint" in result
        assert "hydra_config" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_succeeds_from_supported_detection(
        self, mock_agent, tmp_path
    ):
        """Test supported Phase 3 training projects complete the detection contract."""
        _write_session_06_training_project(tmp_path)

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training detection must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run("Detect this training project", str(tmp_path))

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert completed_tool_steps == ["detect_training_project"]
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in result
        assert "framework_family=pytorch_lightning" in result
        assert "entrypoint=src/train.py" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_blocks_missing_project_path(self, mock_agent):
        """Test detection workflow blocks before tools when project_path is missing."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while detection is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Detect this training project")

        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("project_path",)
        assert "project_path" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_blocks_missing_bounded_controls(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track blocks before tools when bounded controls are missing."""
        _write_session_06_training_project(tmp_path)

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while training is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Train this project", str(tmp_path))

        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert set(mock_agent.workflow_selection.missing_inputs) == {
            "timeout_seconds",
            "max_epochs",
            "device",
            "data_subset",
        }
        assert "missing_inputs" in result
        assert "timeout_seconds" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_blocks_missing_detection_evidence_before_training(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track stops after unsupported detection evidence."""
        project_path = tmp_path / "incomplete-training-project"
        project_path.mkdir()
        (project_path / "pyproject.toml").write_text("[project]\ndependencies = ['torch']\n")
        executed_step_ids = []

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            executed_step_ids.append(step_id)
            if step_id == "detect_training_project":
                result = mcp_mlops_tools.detect_training_project(str(project_path))
                return True, {
                    "success": True,
                    "result": result,
                    "step_id": step_id,
                    "tool": tool,
                }
            raise AssertionError(f"training executed without detection evidence: {step_id}")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch(
                "agent.agent_loop.execute_step",
                new_callable=AsyncMock,
                side_effect=execute_registry_step,
            ),
        ):
            result = await mock_agent.run(
                "Train this project timeout 30 max epochs 1 device cpu subset 8",
                str(project_path),
            )

        assert executed_step_ids == ["detect_training_project"]
        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "training_entrypoint_detected" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_succeeds_from_bounded_training_evidence(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track completes when detection and bounded training evidence exist."""
        _write_bounded_training_fixture(
            tmp_path,
            "# config_path='../configs', config_name='train.yaml'\n"
            "import json\n"
            "from pathlib import Path\n"
            "Path('checkpoints').mkdir(exist_ok=True)\n"
            "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
            "print(json.dumps({'metrics': {'accuracy': 0.93}}))\n",
        )

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Train this project timeout 30 max epochs 1 device cpu subset 8",
                str(tmp_path),
            )

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert completed_tool_steps == ["detect_training_project", "run_bounded_training"]
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in result
        assert "accuracy" in result
        assert "checkpoints/model.ckpt" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_succeeds_from_observed_runtime_evidence(
        self, mock_agent
    ):
        """Test LitServe GPU success is derived from observed live evidence."""
        mock_agent.auto_approve = True

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            payloads = {
                "detect_runtime_environment": {"success": True},
                "detect_gpu_cuda": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "gpu_detection_recorded",
                            "evidence_type": "observed",
                            "source_step": "detect_gpu_cuda",
                            "passed": True,
                            "evidence": "nvidia-smi observed NVIDIA A10",
                        }
                    ],
                },
                "select_best_model_artifact": {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "model_artifact",
                                "producing_step": "select_best_model_artifact",
                                "state": "selected",
                                "path": "models/model.pt",
                            }
                        ]
                    },
                },
                "generate_litserve_api": {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "serving_application",
                                "producing_step": "generate_litserve_api",
                                "state": "generated",
                                "path": "deployment/litserve/server.py",
                            }
                        ]
                    },
                },
                "configure_litserve_gpu_runtime": {"success": True},
                "create_dockerfile": {"success": True},
                "build_image_if_available": {"success": True},
                "start_litserve_server": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "server_start_command_recorded",
                            "evidence_type": "observed",
                            "source_step": "start_litserve_server",
                            "passed": True,
                            "evidence": "process 123 started: python deployment/litserve/server.py",
                        }
                    ],
                },
                "test_health_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "health_result_recorded",
                            "evidence_type": "observed",
                            "source_step": "test_health_endpoint",
                            "passed": True,
                            "evidence": "GET /health returned 200",
                        }
                    ],
                },
                "test_prediction_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "prediction_result_recorded",
                            "evidence_type": "observed",
                            "source_step": "test_prediction_endpoint",
                            "passed": True,
                            "evidence": "POST /predict returned sample prediction",
                        }
                    ],
                },
                "capture_logs_and_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "endpoint_url_recorded",
                            "evidence_type": "observed",
                            "source_step": "capture_logs_and_endpoint",
                            "passed": True,
                            "evidence": "endpoint_url=http://127.0.0.1:8000",
                        }
                    ],
                },
                "write_monitoring_and_rollback_report": {
                    "success": True,
                    "rollback_plan": {
                        "command": "kill 123",
                        "documented_target": "Stop the user-started Lambda Cloud instance manually.",
                    },
                },
            }
            return (
                True,
                {
                    "success": True,
                    "result": payloads[step_id],
                    "step_id": step_id,
                    "tool": tool,
                },
            )

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry runtime must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch(
                "agent.agent_loop.execute_step",
                new_callable=AsyncMock,
                side_effect=execute_registry_step,
            ) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert mock_agent.workflow_selection.workflow_id == "deploy_litserve_gpu"
        assert mock_agent.status == "success"
        assert "deploy_litserve_gpu final workflow status derived from SuccessContract" in result
        assert "contract_status: succeeded" in result
        assert "workflow_status: succeeded" in result
        assert "endpoint_url=http://127.0.0.1:8000" in result
        assert "Stop the user-started Lambda Cloud instance manually" in result
        assert mock_execute_step.await_count == len(
            mock_agent.workflow_registry.get("deploy_litserve_gpu").steps
        )
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_failed_gpu_check_stops_before_server_start(
        self, mock_agent
    ):
        """Test failed observed GPU evidence fails safely before deployment actions."""
        mock_agent.auto_approve = True
        executed_step_ids = []

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            executed_step_ids.append(step_id)
            if step_id == "detect_runtime_environment":
                return True, {
                    "success": True,
                    "result": {"success": True},
                    "step_id": step_id,
                    "tool": tool,
                }
            if step_id == "detect_gpu_cuda":
                return True, {
                    "success": True,
                    "result": {
                        "success": True,
                        "verification_results": [
                            {
                                "check_name": "gpu_detection_recorded",
                                "evidence_type": "observed",
                                "source_step": "detect_gpu_cuda",
                                "passed": False,
                                "evidence": "nvidia-smi unavailable and torch CUDA unavailable",
                            }
                        ],
                    },
                    "step_id": step_id,
                    "tool": tool,
                }
            raise AssertionError(f"unsafe step executed after failed GPU check: {step_id}")

        with patch(
            "agent.agent_loop.execute_step",
            new_callable=AsyncMock,
            side_effect=execute_registry_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert executed_step_ids == ["detect_runtime_environment", "detect_gpu_cuda"]
        assert mock_agent.status == "failed"
        assert "contract_status: failed" in result
        assert "gpu_detection_recorded" in result
        assert "start_litserve_server" not in executed_step_ids

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_uses_selected_pickle_artifact_for_litserve_api(
        self, mock_agent
    ):
        """Test selected sklearn artifacts are passed into LitServe generation."""
        mock_agent.auto_approve = True
        generate_args = None
        runtime_args_by_step = {}

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            nonlocal generate_args
            if step_id == "detect_runtime_environment":
                payload = {"success": True}
            elif step_id == "detect_gpu_cuda":
                payload = {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "gpu_detection_recorded",
                            "evidence_type": "observed",
                            "source_step": "detect_gpu_cuda",
                            "passed": True,
                            "evidence": "nvidia-smi observed NVIDIA A10",
                        }
                    ],
                }
            elif step_id == "select_best_model_artifact":
                payload = {
                    "success": True,
                    "model_path": "outputs/model.pkl",
                    "model_type": "tabular_regressor",
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "model_artifact",
                                "producing_step": "select_best_model_artifact",
                                "state": "selected",
                                "path": "outputs/model.pkl",
                            }
                        ]
                    },
                }
            elif step_id == "generate_litserve_api":
                generate_args = args
                payload = {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "serving_application",
                                "producing_step": "generate_litserve_api",
                                "state": "generated",
                                "path": "deployment/litserve/server.py",
                            }
                        ]
                    },
                }
            else:
                runtime_args_by_step[step_id] = args
                payload = {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": {
                                "start_litserve_server": "server_start_command_recorded",
                                "test_health_endpoint": "health_result_recorded",
                                "test_prediction_endpoint": "prediction_result_recorded",
                                "capture_logs_and_endpoint": "endpoint_url_recorded",
                            }.get(step_id, "unused"),
                            "evidence_type": "observed",
                            "source_step": step_id,
                            "passed": True,
                            "evidence": f"{step_id} evidence",
                        }
                    ],
                }
                if step_id == "start_litserve_server":
                    payload["endpoint_url"] = "http://127.0.0.1:8001"
                    payload["process_id"] = 123
                    payload["port"] = 8001
                if step_id == "write_monitoring_and_rollback_report":
                    payload = {
                        "success": True,
                        "rollback_plan": {"command": "kill 123"},
                    }
            return (
                True,
                {"success": True, "result": payload, "step_id": step_id, "tool": tool},
            )

        with patch(
            "agent.agent_loop.execute_step",
            new_callable=AsyncMock,
            side_effect=execute_registry_step,
        ):
            await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert generate_args["model_path"] == "outputs/model.pkl"
        assert generate_args["model_type"] == "tabular_regressor"
        assert runtime_args_by_step["test_health_endpoint"]["endpoint_url"] == "http://127.0.0.1:8001"
        assert (
            runtime_args_by_step["test_prediction_endpoint"]["endpoint_url"]
            == "http://127.0.0.1:8001"
        )
        assert (
            runtime_args_by_step["capture_logs_and_endpoint"]["endpoint_url"]
            == "http://127.0.0.1:8001"
        )
        assert runtime_args_by_step["write_monitoring_and_rollback_report"]["process_id"] == 123
        assert runtime_args_by_step["write_monitoring_and_rollback_report"]["port"] == 8001

    @pytest.mark.asyncio
    async def test_run_blocks_setup_workflow_missing_project_path_with_clarifying_question(
        self, mock_agent
    ):
        """Test missing setup workflow inputs block before planning or tool execution."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while workflow is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Set up MLOps for this project")

        assert mock_agent.workflow_selection.workflow_id == "setup_pipeline"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("project_path",)
        assert mock_agent.ctx.get_pending_steps() == []
        assert "missing_inputs" in result
        assert "project_path" in result
        assert "What project path should I set up MLOps for?" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_gated_step_without_approval_blocks_before_execute_step(
        self, mock_agent
    ):
        """Test setup workflow approval gates block before tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.next_step_id = workflow_step.step_id

        with patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step:
            await mock_agent._execute_steps_loop()

        approval_events = [
            event for event in events if event["type"] == "approval_required"
        ]
        assert mock_agent.status == "paused"
        assert approval_events
        assert approval_events[0]["data"]["workflow_id"] == "setup_pipeline"
        assert approval_events[0]["data"]["step_id"] == "create_or_validate_hydra_config"
        assert approval_events[0]["data"]["risk_categories"] == ["writes_project_files"]
        assert approval_events[0]["data"]["next_action"]
        assert "Approval required" in mock_agent.final_output
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_denied_approval_blocks_before_execute_step(self, mock_agent):
        """Test denied setup workflow approval prevents tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="denied",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step:
            await mock_agent._execute_steps_loop()

        denied_events = [event for event in events if event["type"] == "approval_denied"]
        assert mock_agent.status == "failed"
        assert denied_events
        assert denied_events[0]["data"]["workflow_id"] == "setup_pipeline"
        assert denied_events[0]["data"]["step_id"] == "create_or_validate_hydra_config"
        assert denied_events[0]["data"]["risk_categories"] == ["writes_project_files"]
        assert "Approval denied" in mock_agent.final_output
        assert "ops@example.com" in mock_agent.final_output
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_approved_record_allows_execute_step(self, mock_agent):
        """Test approved setup workflow approval allows tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        approval_events = [
            event
            for event in events
            if event["type"] in {"approval_required", "approval_denied"}
        ]
        assert approval_events == []
        mock_execute_step.assert_awaited_once()
        assert mock_execute_step.await_args.kwargs["step_id"] == workflow_step.step_id

    @pytest.mark.asyncio
    async def test_setup_pipeline_approved_step_skips_post_step_perception(self, mock_agent):
        """Test registry setup steps do not require LLM perception after execution."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry setup step must skip post-step perception"),
            ) as mock_perception,
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        mock_execute_step.assert_awaited_once()
        mock_perception.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_captures_structured_verification_result(self, mock_agent):
        """Test setup workflow records structured verification evidence from step results."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {
                    "success": True,
                    "step_id": workflow_step.step_id,
                    "result": {
                        "verification_results": [
                            {
                                "check_name": "hydra_config_validates",
                                "evidence_type": "observed",
                                "source_step": workflow_step.step_id,
                                "passed": True,
                                "evidence": "validated conf/config.yaml",
                            }
                        ]
                    },
                },
            )
            await mock_agent._execute_steps_loop()

        verification_results = mock_agent.ctx.globals["verification_results"]
        assert verification_results == (
            VerificationResult(
                check_name="hydra_config_validates",
                evidence_type="observed",
                source_step=workflow_step.step_id,
                passed=True,
                evidence="validated conf/config.yaml",
            ),
        )

    @pytest.mark.asyncio
    async def test_setup_pipeline_captures_artifact_manifest_entry(self, mock_agent):
        """Test setup workflow records explicit generated artifact evidence from step results."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {
                    "success": True,
                    "step_id": workflow_step.step_id,
                    "result": {
                        "artifact_manifest": {
                            "entries": [
                                {
                                    "artifact_type": "configuration",
                                    "producing_step": workflow_step.step_id,
                                    "state": "validated",
                                    "path": "conf/config.yaml",
                                }
                            ]
                        }
                    },
                },
            )
            await mock_agent._execute_steps_loop()

        assert mock_agent.ctx.globals["artifact_manifest"] == ArtifactManifest(
            entries=(
                ArtifactManifestEntry(
                    artifact_type="configuration",
                    producing_step=workflow_step.step_id,
                    state="validated",
                    path="conf/config.yaml",
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_setup_pipeline_missing_evidence_blocks_prompt_success(self, mock_agent):
        """Test setup workflow missing evidence blocks prompt-authored success."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "analyze_project_structure"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.SUMMARIZE, "original_goal_achieved": True},
            ),
            patch.object(
                mock_agent.summarizer,
                "summarize",
                new_callable=AsyncMock,
                side_effect=AssertionError("summarizer must not decide setup success"),
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.BLOCKED
        assert contract_status.failed_checks == ()
        assert contract_status.missing_evidence
        assert mock_agent.ctx.globals["workflow_status"] is WorkflowStatus.BLOCKED
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in mock_agent.final_output
        assert "missing_evidence" in mock_agent.final_output

    @pytest.mark.asyncio
    async def test_setup_pipeline_passing_contract_succeeds_from_structured_evidence(
        self, mock_agent
    ):
        """Test setup workflow succeeds only from passing contract evidence."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        template = mock_agent.workflow_registry.get("setup_pipeline")
        workflow_step = template.step_by_id("create_ci_workflow")
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.ctx.globals["verification_results"] = tuple(
            VerificationResult(
                check_name=check.name,
                evidence_type="declared",
                source_step=check.source_step,
                passed=True,
                evidence=f"{check.name} evidence",
            )
            for check in template.success_contract.checks
            if check.name != "generated_files_reported"
        )
        mock_agent.ctx.globals["artifact_manifest"] = ArtifactManifest(
            entries=(
                ArtifactManifestEntry(
                    artifact_type="configuration",
                    producing_step="create_or_validate_hydra_config",
                    state="generated",
                    path="conf/config.yaml",
                ),
                ArtifactManifestEntry(
                    artifact_type="pipeline_definition",
                    producing_step="create_dvc_yaml",
                    state="generated",
                    path="dvc.yaml",
                ),
                ArtifactManifestEntry(
                    artifact_type="container_definition",
                    producing_step="create_dockerfile",
                    state="generated",
                    path="Dockerfile",
                ),
                ArtifactManifestEntry(
                    artifact_type="automation_workflow",
                    producing_step="create_ci_workflow",
                    state="generated",
                    path=".github/workflows/mlops.yml",
                ),
            )
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
            patch.object(
                mock_agent.summarizer,
                "summarize",
                new_callable=AsyncMock,
                side_effect=AssertionError("summarizer must not decide setup success"),
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.SUCCEEDED
        assert contract_status.missing_evidence == ()
        assert contract_status.failed_checks == ()
        assert mock_agent.ctx.globals["workflow_status"] is WorkflowStatus.SUCCEEDED
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in mock_agent.final_output


# ============================================================================
# run_mlops_agent Convenience Function Tests
# ============================================================================


class TestRunMlopsAgent:
    """Tests for run_mlops_agent convenience function."""

    @pytest.mark.asyncio
    async def test_run_mlops_agent_creates_agent(self, mock_llm):
        """Test run_mlops_agent creates and runs agent."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            result = await run_mlops_agent("Test query", "/test/path", 0.85)
            assert result == "Test result"

    @pytest.mark.asyncio
    async def test_run_mlops_agent_passes_event_callback(self, mock_llm):
        """Test run_mlops_agent passes event callback."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        events = []

        async def capture_event(event_type, data):
            events.append(event_type)

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            await run_mlops_agent("Test query", on_event=capture_event)
            assert "status" in events
            assert "phase" in events


# ============================================================================
# AgentLoop Handle Failure Tests
# ============================================================================


class TestAgentLoopHandleFailure:
    """Tests for AgentLoop _handle_failure method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    @pytest.mark.asyncio
    async def test_handle_failure_sets_status_failed(self, agent, mock_llm):
        """Test _handle_failure preserves failed status while generating a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.status == "failed"
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_marks_session_completed(self, agent, mock_llm):
        """Test _handle_failure marks session as completed with failure."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_returns_summary(self, agent, mock_llm):
        """Test _handle_failure still generates a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            result = await agent._handle_failure()
            assert result == "Failure summary"
