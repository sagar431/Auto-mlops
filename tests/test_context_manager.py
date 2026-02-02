#!/usr/bin/env python3
"""
Tests for agent/contextManager.py - MLOps execution graph and experiment state management.

Run with: pytest tests/test_context_manager.py -v
"""

import pytest

from agent.contextManager import MLOpsStepNode

# ============================================================================
# MLOpsStepNode Tests
# ============================================================================


class TestMLOpsStepNode:
    """Tests for MLOpsStepNode dataclass."""

    def test_create_step_node(self):
        """Test creating a step node with required fields."""
        node = MLOpsStepNode(
            index="0",
            description="Initialize experiment",
            type="CODE",
        )

        assert node.index == "0"
        assert node.description == "Initialize experiment"
        assert node.type == "CODE"
        assert node.status == "pending"
        assert node.tool is None
        assert node.args is None

    def test_create_step_node_with_tool(self, mlops_step_node):
        """Test creating a step node with tool configuration."""
        assert mlops_step_node.tool == "init_mlflow_experiment"
        assert mlops_step_node.args == {"experiment_name": "test_experiment"}
        assert mlops_step_node.status == "pending"

    def test_step_node_has_timestamp(self):
        """Test that step node has created_at timestamp."""
        node = MLOpsStepNode(
            index="0",
            description="Test",
            type="CODE",
        )

        assert node.created_at is not None
        assert "T" in node.created_at  # ISO format

    def test_step_node_types(self):
        """Test various step node types."""
        types = ["ROOT", "CODE", "PERCEPTION", "IMPROVE", "CONCLUDE"]

        for node_type in types:
            node = MLOpsStepNode(
                index=f"node-{node_type}",
                description=f"{node_type} node",
                type=node_type,
            )
            assert node.type == node_type


# ============================================================================
# ExperimentState Tests
# ============================================================================


class TestExperimentState:
    """Tests for ExperimentState dataclass."""

    def test_create_experiment_state(self, experiment_state):
        """Test creating experiment state with defaults."""
        assert experiment_state.experiment_name is None
        assert experiment_state.run_id is None
        assert experiment_state.current_accuracy is None
        assert experiment_state.target_accuracy == 0.85
        assert experiment_state.best_accuracy == 0.0
        assert experiment_state.improvement_attempt == 0
        assert experiment_state.max_improvement_attempts == 3
        assert experiment_state.stage == "setup"

    def test_update_metrics(self, experiment_state):
        """Test updating metrics."""
        experiment_state.update_metrics(accuracy=0.75, loss=0.5)

        assert experiment_state.current_accuracy == 0.75
        assert experiment_state.current_loss == 0.5
        assert experiment_state.best_accuracy == 0.75

    def test_update_metrics_tracks_best(self, experiment_state):
        """Test that best accuracy is tracked correctly."""
        experiment_state.update_metrics(accuracy=0.70)
        experiment_state.update_metrics(accuracy=0.85)
        experiment_state.update_metrics(accuracy=0.80)

        assert experiment_state.current_accuracy == 0.80
        assert experiment_state.best_accuracy == 0.85

    def test_threshold_met(self, experiment_state):
        """Test threshold checking."""
        assert experiment_state.threshold_met() is False

        experiment_state.update_metrics(accuracy=0.80)
        assert experiment_state.threshold_met() is False

        experiment_state.update_metrics(accuracy=0.85)
        assert experiment_state.threshold_met() is True

        experiment_state.update_metrics(accuracy=0.90)
        assert experiment_state.threshold_met() is True

    def test_can_improve(self, experiment_state):
        """Test improvement attempt checking."""
        assert experiment_state.can_improve() is True

        experiment_state.improvement_attempt = 2
        assert experiment_state.can_improve() is True

        experiment_state.improvement_attempt = 3
        assert experiment_state.can_improve() is False

    def test_get_accuracy_gap(self, experiment_state):
        """Test accuracy gap calculation."""
        # No accuracy yet - gap is full target
        assert experiment_state.get_accuracy_gap() == 0.85

        experiment_state.update_metrics(accuracy=0.75)
        assert experiment_state.get_accuracy_gap() == pytest.approx(0.10)

        experiment_state.update_metrics(accuracy=0.85)
        assert experiment_state.get_accuracy_gap() == pytest.approx(0.0)

    def test_record_improvement_attempt(self, experiment_state):
        """Test recording improvement attempts."""
        experiment_state.update_metrics(accuracy=0.70)

        experiment_state.record_improvement_attempt(
            config_changes={"learning_rate": 0.01},
            result_accuracy=0.75,
        )

        assert experiment_state.improvement_attempt == 1
        assert len(experiment_state.improvement_history) == 1
        assert experiment_state.improvement_history[0]["attempt"] == 1
        assert experiment_state.improvement_history[0]["accuracy_before"] == 0.70
        assert experiment_state.improvement_history[0]["accuracy_after"] == 0.75
        assert experiment_state.current_accuracy == 0.75

    def test_to_dict(self, experiment_state):
        """Test serializing experiment state to dict."""
        experiment_state.experiment_name = "test_exp"
        experiment_state.run_id = "run-123"
        experiment_state.update_metrics(accuracy=0.80)

        data = experiment_state.to_dict()

        assert data["experiment_name"] == "test_exp"
        assert data["run_id"] == "run-123"
        assert data["current_accuracy"] == 0.80
        assert data["target_accuracy"] == 0.85
        assert data["stage"] == "setup"


# ============================================================================
# ContextManager Tests
# ============================================================================


class TestContextManager:
    """Tests for ContextManager class."""

    def test_create_context_manager(self, context_manager):
        """Test creating a context manager."""
        assert context_manager.session_id == "test-session-123"
        assert context_manager.original_query == "Test MLOps pipeline setup"
        assert context_manager.project_path == "/test/project"
        assert "ROOT" in context_manager.graph.nodes

    def test_root_node_initialized(self, context_manager):
        """Test that ROOT node is properly initialized."""
        root_node = context_manager.graph.nodes["ROOT"]["data"]

        assert root_node.index == "ROOT"
        assert root_node.type == "ROOT"
        assert root_node.status == "completed"
        assert root_node.description == "Test MLOps pipeline setup"

    def test_add_step(self, context_manager):
        """Test adding a step to the graph."""
        step_id = context_manager.add_step(
            step_id="0",
            description="Initialize MLflow",
            step_type="CODE",
            tool="init_mlflow_experiment",
            args={"experiment_name": "test"},
            from_node="ROOT",
        )

        assert step_id == "0"
        assert "0" in context_manager.graph.nodes
        assert context_manager.latest_node_id == "0"

        node = context_manager.graph.nodes["0"]["data"]
        assert node.tool == "init_mlflow_experiment"
        assert node.status == "pending"

    def test_add_step_with_dependency(self, context_manager):
        """Test adding steps with dependencies."""
        context_manager.add_step("0", "First step", "CODE", from_node="ROOT")
        context_manager.add_step("1", "Second step", "CODE", from_node="0")

        # Check edge exists
        assert context_manager.graph.has_edge("ROOT", "0")
        assert context_manager.graph.has_edge("0", "1")

    def test_is_step_completed(self, context_manager):
        """Test checking if step is completed."""
        context_manager.add_step("0", "Test step", "CODE", from_node="ROOT")

        assert context_manager.is_step_completed("0") is False
        assert context_manager.is_step_completed("ROOT") is True

    def test_update_step_result(self, context_manager, successful_tool_result):
        """Test updating step with result."""
        context_manager.add_step("0", "Create config", "CODE", from_node="ROOT")
        context_manager.update_step_result("0", successful_tool_result)

        node = context_manager.graph.nodes["0"]["data"]
        assert node.status == "completed"
        assert node.result == successful_tool_result
        assert node.completed_at is not None

    def test_mark_step_completed(self, context_manager):
        """Test marking step as completed."""
        context_manager.add_step("0", "Test step", "CODE", from_node="ROOT")
        context_manager.mark_step_completed("0")

        node = context_manager.graph.nodes["0"]["data"]
        assert node.status == "completed"

    def test_mark_step_failed(self, context_manager):
        """Test marking step as failed."""
        context_manager.add_step("0", "Test step", "CODE", from_node="ROOT")
        context_manager.mark_step_failed("0", "Connection timeout")

        node = context_manager.graph.nodes["0"]["data"]
        assert node.status == "failed"
        assert node.error == "Connection timeout"
        assert "0" in context_manager.failed_nodes

    def test_attach_perception(self, context_manager, mock_perception_response):
        """Test attaching perception to a step."""
        context_manager.add_step("0", "Perception step", "PERCEPTION", from_node="ROOT")
        context_manager.attach_perception("0", mock_perception_response)

        node = context_manager.graph.nodes["0"]["data"]
        assert node.perception == mock_perception_response

        # Check that experiment state was updated
        assert context_manager.experiment_state.target_accuracy == 0.85

    def test_conclude_step(self, context_manager):
        """Test concluding a step with final answer."""
        context_manager.add_step("0", "Final step", "CONCLUDE", from_node="ROOT")
        context_manager.conclude("0", "MLOps pipeline setup complete")

        node = context_manager.graph.nodes["0"]["data"]
        assert node.status == "completed"
        assert node.conclusion == "MLOps pipeline setup complete"

    def test_get_latest_node(self, context_manager):
        """Test getting the latest node."""
        assert context_manager.get_latest_node() is None

        context_manager.add_step("0", "First", "CODE", from_node="ROOT")
        assert context_manager.get_latest_node() == "0"

        context_manager.add_step("1", "Second", "CODE", from_node="0")
        assert context_manager.get_latest_node() == "1"

    def test_get_pending_steps(self, context_manager):
        """Test getting pending steps."""
        context_manager.add_step("0", "Step 0", "CODE", from_node="ROOT")
        context_manager.add_step("1", "Step 1", "CODE", from_node="0")
        context_manager.mark_step_completed("0")

        pending = context_manager.get_pending_steps()
        assert "1" in pending
        assert "0" not in pending

    def test_get_completed_steps(self, context_manager):
        """Test getting completed steps."""
        context_manager.add_step("0", "Step 0", "CODE", from_node="ROOT")
        context_manager.add_step("1", "Step 1", "CODE", from_node="0")
        context_manager.mark_step_completed("0")

        completed = context_manager.get_completed_steps()
        completed_indices = [step["index"] for step in completed]

        assert "0" in completed_indices
        assert "ROOT" in completed_indices
        assert "1" not in completed_indices

    def test_get_failed_steps(self, context_manager):
        """Test getting failed steps."""
        context_manager.add_step("0", "Step 0", "CODE", from_node="ROOT")
        context_manager.add_step("1", "Step 1", "CODE", from_node="0")
        context_manager.mark_step_failed("0", "Error occurred")

        failed = context_manager.get_failed_steps()
        failed_indices = [step["index"] for step in failed]

        assert "0" in failed_indices
        assert "1" not in failed_indices

    def test_set_experiment_config(self, context_manager):
        """Test setting experiment configuration."""
        config = {"learning_rate": 0.001, "batch_size": 32}
        context_manager.set_experiment_config(config)

        assert context_manager.experiment_state.current_config == config

    def test_record_improvement(self, context_manager):
        """Test recording improvement."""
        context_manager.experiment_state.update_metrics(accuracy=0.70)

        context_manager.record_improvement(
            config_changes={"learning_rate": 0.01},
            new_accuracy=0.78,
        )

        assert context_manager.experiment_state.improvement_attempt == 1
        assert context_manager.experiment_state.current_accuracy == 0.78

    def test_get_experiment_snapshot(self, context_manager):
        """Test getting experiment snapshot."""
        context_manager.experiment_state.experiment_name = "test_exp"
        context_manager.experiment_state.update_metrics(accuracy=0.80)

        snapshot = context_manager.get_experiment_snapshot()

        assert snapshot["experiment_name"] == "test_exp"
        assert snapshot["current_accuracy"] == 0.80

    def test_get_context_snapshot(self, context_manager):
        """Test getting full context snapshot."""
        context_manager.add_step("0", "Test step", "CODE", from_node="ROOT")

        snapshot = context_manager.get_context_snapshot()

        assert snapshot["session_id"] == "test-session-123"
        assert snapshot["original_query"] == "Test MLOps pipeline setup"
        assert snapshot["project_path"] == "/test/project"
        assert "graph" in snapshot
        assert "experiment_state" in snapshot
        assert "timestamp" in snapshot

    def test_attach_summary(self, context_manager):
        """Test attaching summary to session memory."""
        summary = {
            "summary_markdown": "# Summary\nPipeline setup complete.",
            "confidence": 0.95,
            "goal_achieved": True,
        }

        context_manager.attach_summary(summary)

        assert len(context_manager.session_memory) == 1
        assert "# Summary" in context_manager.session_memory[0]["summarizer_summary"]


class TestContextManagerMLOpsIntegration:
    """Integration tests for MLOps-specific context manager features."""

    def test_artifact_tracking(self, context_manager):
        """Test that artifacts are tracked from tool results."""
        context_manager.add_step("0", "Create config", "CODE", from_node="ROOT")

        result = {
            "success": True,
            "config_path": "/test/config.yaml",
        }
        context_manager.update_step_result("0", result)

        assert "/test/config.yaml" in context_manager.experiment_state.artifacts_created

    def test_mlflow_tracking(self, context_manager, mlflow_run_result):
        """Test that MLflow run info is tracked."""
        context_manager.add_step("0", "Start MLflow run", "CODE", from_node="ROOT")
        context_manager.update_step_result("0", mlflow_run_result)

        assert context_manager.experiment_state.run_id == "run-456"
        assert context_manager.experiment_state.run_name == "test_run"

    def test_metrics_from_tool_result(self, context_manager, training_result):
        """Test that metrics are extracted from training results."""
        context_manager.add_step("0", "Check threshold", "CODE", from_node="ROOT")
        context_manager.update_step_result("0", training_result)

        assert context_manager.experiment_state.current_accuracy == 0.82

    def test_pipeline_stage_update_from_perception(self, context_manager):
        """Test that pipeline stage is updated from perception."""
        perception = {
            "entities": {},
            "pipeline_stage": "training",
            "route": "decision",
        }

        context_manager.add_step("0", "Perception", "PERCEPTION", from_node="ROOT")
        context_manager.attach_perception("0", perception)

        assert context_manager.experiment_state.stage == "training"

    def test_globals_versioning(self, context_manager):
        """Test that globals are versioned on conflict."""
        context_manager.add_step("0", "Step 0", "CODE", from_node="ROOT")
        context_manager.latest_node_id = "0"

        # First update
        context_manager._update_globals({"config_path": "/path/v1"})
        assert context_manager.globals["config_path"] == "/path/v1"

        # Second update with same key
        context_manager._update_globals({"config_path": "/path/v2"})
        assert "config_path__0" in context_manager.globals
