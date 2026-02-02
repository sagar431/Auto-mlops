"""
Context Manager for MLOps Agent - Graph-based execution tracking with experiment state.
Extends base context management with MLOps-specific features.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import networkx as nx

from observability import get_logger

logger = get_logger("agent.contextManager")


@dataclass
class MLOpsStepNode:
    """Represents a node in the MLOps execution graph."""

    index: str  # Node ID - supports labels like "0", "0A", "0B", "ROOT"
    description: str
    type: str  # "ROOT", "CODE", "PERCEPTION", "IMPROVE", "CONCLUDE"
    tool: str | None = None  # MCP tool name
    args: dict[str, Any] | None = None  # Tool arguments
    status: str = "pending"  # "pending", "completed", "failed", "skipped"
    result: dict[str, Any] | None = None
    conclusion: str | None = None
    error: str | None = None
    perception: dict[str, Any] | None = None
    from_step: str | None = None  # Parent node for lineage tracking
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None


@dataclass
class ExperimentState:
    """Tracks the current ML experiment state."""

    experiment_name: str | None = None
    run_id: str | None = None
    run_name: str | None = None

    # Metrics tracking
    current_accuracy: float | None = None
    current_loss: float | None = None
    target_accuracy: float = 0.85
    best_accuracy: float = 0.0

    # Training config
    current_config: dict[str, Any] = field(default_factory=dict)

    # Improvement loop
    improvement_attempt: int = 0
    max_improvement_attempts: int = 3
    improvement_history: list[dict[str, Any]] = field(default_factory=list)

    # Pipeline stage
    stage: str = "setup"  # setup, config, data, training, evaluation, improvement, deploy

    # Artifacts
    artifacts_created: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "run_name": self.run_name,
            "current_accuracy": self.current_accuracy,
            "current_loss": self.current_loss,
            "target_accuracy": self.target_accuracy,
            "best_accuracy": self.best_accuracy,
            "current_config": self.current_config,
            "improvement_attempt": self.improvement_attempt,
            "max_improvement_attempts": self.max_improvement_attempts,
            "improvement_history": self.improvement_history,
            "stage": self.stage,
            "artifacts_created": self.artifacts_created,
        }

    def update_metrics(self, accuracy: float, loss: float = None):
        """Update current metrics and track best."""
        self.current_accuracy = accuracy
        if loss is not None:
            self.current_loss = loss
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

    def record_improvement_attempt(self, config_changes: dict, result_accuracy: float):
        """Record an improvement attempt for history."""
        self.improvement_attempt += 1
        self.improvement_history.append(
            {
                "attempt": self.improvement_attempt,
                "config_changes": config_changes,
                "accuracy_before": self.current_accuracy,
                "accuracy_after": result_accuracy,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        self.update_metrics(result_accuracy)

    def threshold_met(self) -> bool:
        """Check if accuracy threshold is met."""
        return self.current_accuracy is not None and self.current_accuracy >= self.target_accuracy

    def can_improve(self) -> bool:
        """Check if more improvement attempts are allowed."""
        return self.improvement_attempt < self.max_improvement_attempts

    def get_accuracy_gap(self) -> float:
        """Get gap between current and target accuracy."""
        if self.current_accuracy is None:
            return self.target_accuracy
        return self.target_accuracy - self.current_accuracy


class ContextManager:
    """
    Manages the execution graph and experiment state for the MLOps Agent.
    Uses NetworkX DiGraph for graph operations.
    """

    def __init__(self, session_id: str, original_query: str, project_path: str | None = None):
        self.session_id = session_id
        self.original_query = original_query
        self.project_path = project_path

        # Global state
        self.globals: dict[str, Any] = {}
        self.session_memory: list[dict] = []
        self.failed_nodes: list[str] = []

        # Execution graph
        self.graph = nx.DiGraph()
        self.latest_node_id: str | None = None
        self.executed_variants: dict[str, set[str]] = defaultdict(set)

        # MLOps-specific state
        self.experiment_state = ExperimentState()

        # Initialize with ROOT node
        root_node = MLOpsStepNode(
            index="ROOT", description=original_query, type="ROOT", status="completed"
        )
        self.graph.add_node("ROOT", data=root_node)

    # =========================================================================
    # Graph Operations
    # =========================================================================

    def add_step(
        self,
        step_id: str,
        description: str,
        step_type: str,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        from_node: str | None = None,
        edge_type: str = "normal",
    ) -> str:
        """Add a new step node to the graph."""
        step_node = MLOpsStepNode(
            index=step_id,
            description=description,
            type=step_type,
            tool=tool,
            args=args,
            from_step=from_node,
        )
        self.graph.add_node(step_id, data=step_node)
        if from_node:
            self.graph.add_edge(from_node, step_id, type=edge_type)
        self.latest_node_id = step_id
        return step_id

    def is_step_completed(self, step_id: str) -> bool:
        """Check if a step is already completed."""
        node = self.graph.nodes.get(step_id, {}).get("data")
        return node is not None and node.status == "completed"

    def update_step_result(self, step_id: str, result: dict):
        """Update a step with execution result and mark as completed."""
        node: MLOpsStepNode = self.graph.nodes[step_id]["data"]
        node.result = result
        node.status = "completed"
        node.completed_at = datetime.utcnow().isoformat()
        self._update_globals(result)
        self._process_mlops_result(result)

    def mark_step_completed(self, step_id: str):
        """Mark a step as completed."""
        if step_id in self.graph:
            node: MLOpsStepNode = self.graph.nodes[step_id]["data"]
            node.status = "completed"
            node.completed_at = datetime.utcnow().isoformat()

    def mark_step_failed(self, step_id: str, error_msg: str):
        """Mark a step as failed and record error."""
        node: MLOpsStepNode = self.graph.nodes[step_id]["data"]
        node.status = "failed"
        node.error = error_msg
        self.failed_nodes.append(step_id)
        self.session_memory.append(
            {
                "query": node.description,
                "tool": node.tool,
                "result_requirement": "Tool failed",
                "solution_summary": str(error_msg)[:300],
            }
        )

    def attach_perception(self, step_id: str, perception: dict):
        """Attach perception analysis to a step node."""
        if step_id not in self.graph.nodes:
            fallback_node = MLOpsStepNode(
                index=step_id, description="Perception-only node", type="PERCEPTION"
            )
            self.graph.add_node(step_id, data=fallback_node)

        node: MLOpsStepNode = self.graph.nodes[step_id]["data"]
        node.perception = perception

        # Update experiment state from perception
        self._update_from_perception(perception)

        if not perception.get("local_goal_achieved", True):
            self.failed_nodes.append(step_id)

    def conclude(self, step_id: str, conclusion: str):
        """Mark a step as concluded with final answer."""
        node: MLOpsStepNode = self.graph.nodes[step_id]["data"]
        node.status = "completed"
        node.conclusion = conclusion
        node.completed_at = datetime.utcnow().isoformat()

    def get_latest_node(self) -> str | None:
        """Get the ID of the most recently added node."""
        return self.latest_node_id

    # =========================================================================
    # MLOps-Specific Operations
    # =========================================================================

    def _process_mlops_result(self, result: dict):
        """Process tool results to update experiment state."""
        if not result.get("success", False):
            return

        # Track artifacts
        for artifact_key in [
            "dockerfile_path",
            "workflow_path",
            "config_path",
            "dvc_yaml_path",
            "created_files",
        ]:
            if artifact_key in result:
                artifact = result[artifact_key]
                if isinstance(artifact, list):
                    self.experiment_state.artifacts_created.extend(artifact)
                else:
                    self.experiment_state.artifacts_created.append(artifact)

        # Track experiment info
        if "experiment_id" in result:
            self.experiment_state.experiment_name = result.get("experiment_name")

        if "run_id" in result:
            self.experiment_state.run_id = result["run_id"]
            self.experiment_state.run_name = result.get("run_name")

        # Track metrics
        if "current_value" in result:  # From check_accuracy_threshold
            self.experiment_state.update_metrics(result["current_value"])

        if "best_metric" in result:  # From get_best_mlflow_run
            metrics = result["best_metric"]
            if "accuracy" in metrics:
                self.experiment_state.update_metrics(metrics["accuracy"])

    def _update_from_perception(self, perception: dict):
        """Update experiment state from perception output."""
        entities = perception.get("entities", {})

        if entities.get("accuracy_threshold"):
            self.experiment_state.target_accuracy = entities["accuracy_threshold"]

        if entities.get("current_accuracy"):
            self.experiment_state.update_metrics(entities["current_accuracy"])

        if entities.get("experiment_name"):
            self.experiment_state.experiment_name = entities["experiment_name"]

        if entities.get("project_path"):
            self.project_path = entities["project_path"]

        # Update pipeline stage
        if perception.get("pipeline_stage"):
            self.experiment_state.stage = perception["pipeline_stage"]

    def set_experiment_config(self, config: dict[str, Any]):
        """Set the current training configuration."""
        self.experiment_state.current_config = config

    def record_improvement(self, config_changes: dict, new_accuracy: float):
        """Record an improvement attempt."""
        self.experiment_state.record_improvement_attempt(config_changes, new_accuracy)

    def get_experiment_snapshot(self) -> dict[str, Any]:
        """Get current experiment state as dict."""
        return self.experiment_state.to_dict()

    # =========================================================================
    # State Management
    # =========================================================================

    def _update_globals(self, new_vars: dict[str, Any]):
        """Update global variables with new execution results."""
        for k, v in new_vars.items():
            if k in self.globals:
                versioned_key = f"{k}__{self.latest_node_id}"
                self.globals[versioned_key] = v
            else:
                self.globals[k] = v

    def get_context_snapshot(self) -> dict:
        """Get a serializable snapshot of the current context."""

        def serialize_node_data(data):
            if hasattr(data, "__dict__"):
                return {k: v for k, v in data.__dict__.items() if not k.startswith("_")}
            return data

        graph_data = nx.readwrite.json_graph.node_link_data(self.graph, edges="links")

        for node in graph_data["nodes"]:
            if "data" in node:
                node["data"] = serialize_node_data(node["data"])

        return {
            "session_id": self.session_id,
            "original_query": self.original_query,
            "project_path": self.project_path,
            "globals": self.globals,
            "memory": self.session_memory,
            "graph": graph_data,
            "experiment_state": self.experiment_state.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_pending_steps(self) -> list[str]:
        """Get all pending step IDs."""
        return [
            node_id
            for node_id in self.graph.nodes
            if self.graph.nodes[node_id]["data"].status == "pending"
        ]

    def get_completed_steps(self) -> list[dict]:
        """Get all completed step data."""
        return [
            self.graph.nodes[n]["data"].__dict__
            for n in self.graph.nodes
            if self.graph.nodes[n]["data"].status == "completed"
        ]

    def get_failed_steps(self) -> list[dict]:
        """Get all failed step data."""
        return [
            self.graph.nodes[n]["data"].__dict__ for n in self.failed_nodes if n in self.graph.nodes
        ]

    def attach_summary(self, summary: dict):
        """Attach summarizer output to session memory."""
        self.session_memory.append(
            {
                "original_query": self.original_query,
                "result_requirement": "Final summary",
                "summarizer_summary": summary.get("summary_markdown", str(summary)),
                "confidence": summary.get("confidence", 0.95),
                "original_goal_achieved": summary.get("goal_achieved", True),
                "route": "summarize",
            }
        )

    def print_graph(self, depth: int = 2):
        """Print graph structure for debugging."""
        nodes_info = []
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]["data"]
            node_info = {
                "node_id": node_id,
                "status": node.status,
                "description": node.description[:50],
            }
            if node.tool:
                node_info["tool"] = node.tool
            if node.error:
                node_info["error"] = node.error[:50]
            nodes_info.append(node_info)

        logger.debug(
            "Graph structure",
            session_id=self.session_id,
            stage=self.experiment_state.stage,
            current_accuracy=self.experiment_state.current_accuracy,
            target_accuracy=self.experiment_state.target_accuracy,
            nodes=nodes_info,
        )
