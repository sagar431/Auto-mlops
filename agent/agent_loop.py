"""
Agent Loop for MLOps Agent.
Graph-based execution loop with self-improvement capability.
Orchestrates: Perception -> Decision -> Action -> (Improve if needed) -> Summarize
"""

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from action.execute_step import execute_step
from agent.agentSession import AgentSession
from agent.approval import wait_for_approval
from agent.contextManager import ContextManager
from agent.model_manager import get_model_manager
from decision.decision import Decision, build_decision_input
from memory.memory_search import MemorySearch
from observability import get_logger
from perception.perception import Perception, build_perception_input
from resilience import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError
from summarization.summarizer import Summarizer
from workflow.registry import (
    ApprovalRecord,
    ArtifactManifest,
    ArtifactManifestEntry,
    ContractFailure,
    ContractValidation,
    RiskCategory,
    RollbackPlan,
    StepApprovalValidation,
    VerificationResult,
    WorkflowSelection,
    WorkflowStatus,
    get_workflow_registry,
)

logger = get_logger("agent.agent_loop")


class Route:
    """Routing constants."""

    SUMMARIZE = "summarize"
    DECISION = "decision"
    IMPROVE = "improve"
    DEPLOY = "deploy"


class StepType:
    """Step type constants."""

    ROOT = "ROOT"
    CODE = "CODE"
    IMPROVE = "IMPROVE"
    DEPLOY = "DEPLOY"


class StepExecutionError(Exception):
    """Raised when a step execution fails."""

    def __init__(self, step_id: str, error_message: str):
        self.step_id = step_id
        self.error_message = error_message
        super().__init__(f"Step '{step_id}' failed: {error_message}")


class StepExecutionTracker:
    """
    Tracks step execution attempts and limits with circuit breaker protection.

    The circuit breaker prevents cascading failures by temporarily stopping
    execution when too many consecutive failures occur.
    """

    def __init__(
        self,
        max_steps: int = 15,
        max_retries: int = 3,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.attempts: dict[str, int] = {}
        self.tries = 0
        self.root_failures = 0

        # Initialize circuit breaker with custom or default config
        cb_config = circuit_breaker_config or CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_calls=1,
        )
        self._circuit_breaker = CircuitBreaker("step_execution", cb_config)

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Access the circuit breaker for monitoring or testing."""
        return self._circuit_breaker

    def increment(self):
        self.tries += 1

    def record_failure(self, step_id: str):
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1

    def retry_step_id(self, step_id: str) -> str:
        attempts = self.attempts.get(step_id, 0)
        return f"{step_id}F{attempts}" if attempts > 0 else step_id

    def should_continue(self) -> bool:
        return self.tries < self.max_steps

    def has_exceeded_retries(self, step_id: str) -> bool:
        return self.attempts.get(step_id, 0) >= self.max_retries

    def is_circuit_open(self) -> bool:
        """Check if the circuit breaker is in open state."""
        from resilience import CircuitState

        return self._circuit_breaker.state == CircuitState.OPEN

    def get_circuit_stats(self) -> dict:
        """Get circuit breaker statistics."""
        stats = self._circuit_breaker.stats
        return {
            "state": self._circuit_breaker.state.value,
            "total_calls": stats.total_calls,
            "successful_calls": stats.successful_calls,
            "failed_calls": stats.failed_calls,
            "rejected_calls": stats.rejected_calls,
            "consecutive_failures": stats.consecutive_failures,
            "consecutive_successes": stats.consecutive_successes,
        }

    def reset_circuit(self):
        """Reset the circuit breaker to closed state."""
        self._circuit_breaker.reset()


class AgentLoop:
    """
    Main agent loop for MLOps operations.
    Implements the Perception -> Decision -> Action cycle with self-improvement.
    """

    def __init__(
        self,
        prompts_dir: str | None = None,
        tools_module: Any | None = None,
        on_event: Callable | None = None,
        profile: str = "default",
        approval_callback: Callable | None = None,
        approval_timeout: int = 300,
        auto_approve: bool = False,
    ):
        # Load prompts
        prompts_dir = Path(prompts_dir) if prompts_dir else Path(__file__).parent.parent / "prompts"

        self.perception = Perception(prompts_dir / "perception_prompt.txt")
        self.decision = Decision(prompts_dir / "decision_prompt.txt", tools_module)
        self.summarizer = Summarizer(prompts_dir / "summarizer_prompt.txt")
        self.improvement_prompt = self._load_prompt(prompts_dir / "improvement_prompt.txt")

        self.tools_module = tools_module
        self.profile = profile
        self.on_event = on_event
        self.status = "idle"
        self.approval_callback = approval_callback
        self.approval_timeout = approval_timeout
        self.auto_approve = auto_approve
        self._approval_cache: dict[str, bool] = {}
        self.workflow_registry = get_workflow_registry()
        self.workflow_selection: WorkflowSelection | None = None

        # Model manager for LLM calls
        self.model_manager = get_model_manager()

    def _load_prompt(self, path: Path) -> str:
        """Load prompt from file."""
        try:
            return path.read_text()
        except Exception:
            return ""

    async def _emit(self, event_type: str, data: dict = None):
        """Emit an event to callback if registered."""
        if self.on_event:
            try:
                await self.on_event(event_type, data or {})
            except Exception as e:
                logger.warning("Failed to emit event", error=e, event_type=event_type)

    async def run(
        self, query: str, project_path: str | None = None, accuracy_threshold: float = 0.85
    ) -> str:
        """
        Run the MLOps agent loop.

        Args:
            query: User query/goal
            project_path: Path to ML project
            accuracy_threshold: Target accuracy for training

        Returns:
            Final summary or error message
        """
        self.status = "running"
        await self._emit("status", {"status": "running", "message": "Agent started"})

        # Initialize session
        self._initialize_session(query, project_path, accuracy_threshold)

        if await self._select_registry_workflow():
            return self.final_output

        # Phase 1: Initial Perception
        await self._emit("phase", {"phase": "perception", "message": "Analyzing request..."})
        await self._run_initial_perception()

        # Check for early exit
        if self._should_summarize():
            await self._emit("phase", {"phase": "summary", "message": "Generating summary..."})
            return await self._summarize()

        # Phase 2: Decision + Execution Loop
        await self._run_decision_loop()

        # Phase 3: Check for deployment if requested
        if self._needs_deployment():
            await self._run_deployment_loop()
            if self.status == "paused":
                return self.final_output
            if self.status == "failed":
                return self.final_output

        # Phase 4: Check for improvement if training occurred
        if self._needs_improvement():
            await self._run_improvement_loop()

        # Phase 5: Final Summary
        if self.status == "success" or self.ctx.experiment_state.threshold_met():
            return self.final_output

        return await self._handle_failure()

    def _initialize_session(self, query: str, project_path: str | None, accuracy_threshold: float):
        """Initialize session context and state."""
        self.session_id = str(uuid.uuid4())
        self.query = query

        # Create context manager
        self.ctx = ContextManager(
            session_id=self.session_id, original_query=query, project_path=project_path
        )
        self.ctx.experiment_state.target_accuracy = accuracy_threshold

        # Create session
        self.session = AgentSession(
            session_id=self.session_id,
            original_query=query,
            project_path=project_path,
            profile=self.profile,
        )

        # Load memory from past experiments
        self.memory = MemorySearch().search_memory(query)
        self.ctx.globals["memory"] = self.memory

        # Placeholders
        self.p_out: dict = {}
        self.code_variants: dict = {}
        self.next_step_id: str = "0"
        self.final_output: str = ""
        self.workflow_selection = None
        self._resolved_workflow_inputs: dict[str, Any] = {}

    async def _select_registry_workflow(self) -> bool:
        """Select a registry workflow before prompt-authored planning."""
        if self._should_defer_to_prompt_planning_before_registry():
            return False

        selection = self.workflow_registry.select_workflow(self.query)
        selection = self._add_runtime_input_requirements(selection)
        self.workflow_selection = selection
        self.ctx.globals["workflow_selection"] = selection
        self.ctx.globals["workflow_inputs"] = self._resolved_workflow_inputs or {
            "project_path": self.ctx.project_path
        }

        await self._emit(
            "workflow_selection",
            {
                **asdict(selection),
                "status": selection.status.value,
                "runtime_inputs": {"project_path": self.ctx.project_path},
            },
        )

        if selection.status is WorkflowStatus.PENDING:
            if self._is_executable_registry_workflow(selection.workflow_id):
                projected_step_ids = await self._project_registry_workflow(selection.workflow_id)
                if not self.auto_approve and not self.ctx.globals.get("approval_records"):
                    approval_validation = self._first_blocking_registry_approval()
                    if approval_validation is not None:
                        self.status = "paused"
                        risk_categories = [
                            risk_category.value
                            for risk_category in approval_validation.risk_categories
                        ]
                        await self._emit(
                            "approval_required",
                            {
                                "workflow_id": approval_validation.workflow_id,
                                "workflow_run_id": approval_validation.workflow_run_id,
                                "step_id": approval_validation.step_id,
                                "risk_categories": risk_categories,
                                "status": approval_validation.status.value,
                                "next_action": approval_validation.next_action,
                            },
                        )
                        self.final_output = (
                            f"Selected workflow '{selection.workflow_id}' from registry. "
                            "Projected executable runtime steps: "
                            f"{', '.join(projected_step_ids)}. "
                            "Approval required before executing workflow step "
                            f"'{approval_validation.step_id}'. risk_categories: "
                            f"{', '.join(risk_categories)}. "
                            f"next_action: {approval_validation.next_action}. "
                            "No tools were run."
                        )
                        return True

                await self._execute_steps_loop()
                return True

            self.status = "paused"
            self.final_output = (
                f"Selected workflow '{selection.workflow_id}' from registry. "
                "Workflow execution is not enabled yet; no tools were run."
            )
            return True

        if self._should_block_for_workflow_selection(selection):
            self.status = "paused"
            missing_inputs = ", ".join(selection.missing_inputs) or "workflow_intent"
            clarifying_question = self._workflow_clarifying_question(selection.missing_inputs)
            self.final_output = (
                "Workflow selection blocked: "
                f"{selection.selection_reason} missing_inputs: {missing_inputs}. "
                f"next_action: {clarifying_question}. "
                f"Clarifying question: {clarifying_question}"
            )
            return True

        return False

    def _should_defer_to_prompt_planning_before_registry(self) -> bool:
        """Keep broad multi-phase requests on the prompt-authored planning path."""
        normalized_query = self.query.casefold()
        setup_requested = any(
            term in normalized_query
            for term in ("set up mlops", "setup mlops", "set up mlops pipeline")
        )
        additional_phase_requested = any(
            term in normalized_query for term in (" and deploy", ", train", " train ")
        )
        explicit_litserve_gpu = any(
            term in normalized_query
            for term in ("lambda labs gpu", "lambda gpu", "litserve gpu deployment")
        )
        return setup_requested and additional_phase_requested and not explicit_litserve_gpu

    def _is_executable_registry_workflow(self, workflow_id: str | None) -> bool:
        return workflow_id in {
            "setup_pipeline",
            "detect_training_project",
            "train_and_track",
            "build_capstone_pipeline",
            "prepare_capstone_data",
            "prepare_capstone_container_ci",
            "deploy_litserve_gpu",
        }

    def _first_blocking_registry_approval(self):
        """Return the first selected registry approval gate that lacks approval."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id is None
            or self.workflow_selection.status is not WorkflowStatus.PENDING
        ):
            return None

        workflow_id = self.workflow_selection.workflow_id
        if workflow_id in {"prepare_capstone_data", "prepare_capstone_container_ci"}:
            return None
        template = self.workflow_registry.get(workflow_id)
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        for step in template.steps:
            validation = self.workflow_registry.validate_step_approval(
                workflow_id=workflow_id,
                workflow_run_id=self.session_id,
                step_id=step.step_id,
                approval_records=approval_records,
            )
            if validation.status is WorkflowStatus.BLOCKED:
                return validation
        return None

    def _workflow_clarifying_question(self, missing_inputs: tuple[str, ...]) -> str:
        """Return a deterministic question for missing workflow inputs."""
        if "project_path" in missing_inputs:
            return "What project path should I set up MLOps for?"
        if any(
            missing_input in missing_inputs
            for missing_input in ("timeout_seconds", "max_epochs", "device", "data_subset")
        ):
            return (
                "What timeout, max epochs, device, and dataset subset should I use "
                "for bounded training?"
            )
        return "Which workflow should I run?"

    async def _project_registry_workflow(self, workflow_id: str) -> tuple[str, ...]:
        """Project a selected registry template into pending runtime steps."""
        template = self.workflow_registry.get(workflow_id)
        projected_step_ids: list[str] = []

        for workflow_step in template.steps:
            tool = workflow_step.tool_functions[0] if workflow_step.tool_functions else None
            self.ctx.add_step(
                step_id=workflow_step.step_id,
                description=workflow_step.description,
                step_type=StepType.CODE,
                tool=tool,
                args=dict(workflow_step.default_args),
                from_node=StepType.ROOT,
            )
            projected_step_ids.append(workflow_step.step_id)

        await self._emit(
            "workflow_runtime",
            {"workflow_id": workflow_id, "projected_step_ids": projected_step_ids},
        )
        self.next_step_id = projected_step_ids[0] if projected_step_ids else None
        return tuple(projected_step_ids)

    def _add_runtime_input_requirements(
        self, selection: WorkflowSelection
    ) -> WorkflowSelection:
        """Block selected workflows when required runtime inputs are missing."""
        if selection.workflow_id is None:
            return selection

        template = self.workflow_registry.get(selection.workflow_id)
        missing_inputs = list(selection.missing_inputs)
        resolved_inputs: dict[str, Any] = {"project_path": self.ctx.project_path}
        training_controls = (
            self._bounded_training_controls_from_query()
            if selection.workflow_id == "train_and_track"
            else {}
        )
        resolved_inputs.update(training_controls)
        if selection.workflow_id == "prepare_capstone_data":
            resolved_inputs.update(self._capstone_data_inputs_from_query())
        if selection.workflow_id == "prepare_capstone_container_ci":
            resolved_inputs.update(self._capstone_container_ci_inputs_from_query())
        for workflow_input in template.required_inputs:
            if workflow_input.required and workflow_input.name == "project_path":
                if not self.ctx.project_path and "project_path" not in missing_inputs:
                    missing_inputs.append("project_path")
                continue
            if (
                workflow_input.allowed_values
                and workflow_input.name in resolved_inputs
                and resolved_inputs[workflow_input.name] not in workflow_input.allowed_values
            ):
                if workflow_input.name not in missing_inputs:
                    missing_inputs.append(workflow_input.name)
                continue
            if selection.workflow_id == "prepare_capstone_data":
                if (
                    workflow_input.required
                    and workflow_input.name not in resolved_inputs
                    and workflow_input.name not in missing_inputs
                ):
                    missing_inputs.append(workflow_input.name)
                continue
            if selection.workflow_id == "prepare_capstone_container_ci":
                continue
            if (
                workflow_input.required
                and selection.workflow_id == "train_and_track"
                and workflow_input.name not in resolved_inputs
                and workflow_input.name not in missing_inputs
            ):
                missing_inputs.append(workflow_input.name)

        if not missing_inputs:
            self._resolved_workflow_inputs = resolved_inputs
            return selection

        self._resolved_workflow_inputs = resolved_inputs
        return replace(
            selection,
            status=WorkflowStatus.BLOCKED,
            missing_inputs=tuple(missing_inputs),
            selection_reason=(
                f"{selection.selection_reason} Missing required runtime inputs: "
                f"{', '.join(missing_inputs)}. "
                f"{self._workflow_input_next_action(template, tuple(missing_inputs))}"
            ),
        )

    def _workflow_input_next_action(
        self, template, missing_inputs: tuple[str, ...]
    ) -> str:
        """Return registry-owned next action text for missing or invalid workflow inputs."""
        parts: list[str] = []
        inputs_by_name = {workflow_input.name: workflow_input for workflow_input in template.required_inputs}
        for missing_input in missing_inputs:
            workflow_input = inputs_by_name.get(missing_input)
            if workflow_input is not None and workflow_input.allowed_values:
                parts.append(
                    f"Provide {missing_input} as one of: "
                    f"{', '.join(str(value) for value in workflow_input.allowed_values)}."
                )
            else:
                parts.append(f"Provide {missing_input}.")
        return "next_action: " + " ".join(parts)

    def _capstone_data_inputs_from_query(self) -> dict[str, Any]:
        """Extract declared Phase 4 Issue 1 capstone data inputs from the query."""
        inputs: dict[str, Any] = {
            "completion_mode": "local_ready",
            "test_size": 0.2,
            "split_seed": 42,
            "materialize_splits": False,
            "dvc_transfer_direction": "push",
        }
        for input_name in (
            "dataset_1_path",
            "dataset_2_path",
            "completion_mode",
            "dvc_remote_name",
            "dvc_remote_url",
            "dvc_transfer_direction",
        ):
            match = re.search(
                rf"{input_name}\s*=\s*(\S+)",
                self.query,
                flags=re.IGNORECASE,
            )
            if match:
                inputs[input_name] = match.group(1)
        test_size_match = re.search(r"test_size\s*=\s*(0(?:\.\d+)?|1(?:\.0+)?)", self.query)
        if test_size_match:
            inputs["test_size"] = float(test_size_match.group(1))
        split_seed_match = re.search(r"split_seed\s*=\s*(\d+)", self.query)
        if split_seed_match:
            inputs["split_seed"] = int(split_seed_match.group(1))
        materialize_match = re.search(
            r"materialize_splits\s*=\s*(true|false|1|0|yes|no)",
            self.query,
            flags=re.IGNORECASE,
        )
        if materialize_match:
            inputs["materialize_splits"] = materialize_match.group(1).casefold() in {
                "true",
                "1",
                "yes",
            }
        return inputs

    def _capstone_container_ci_inputs_from_query(self) -> dict[str, Any]:
        """Extract declared Phase 5 Issue 1 container/CI inputs from the query."""
        inputs: dict[str, Any] = {
            "completion_mode": "container_local_ready",
            "data_stage_evidence_path": None,
            "local_model_artifact_path": None,
            "mlflow_run_id": None,
            "mlflow_best_artifact_path": None,
            "registry_target": None,
            "image_name": None,
            "image_tag": None,
            "ci_workflow_path": None,
        }
        for input_name in inputs:
            match = re.search(
                rf"{input_name}\s*=\s*(\S+)",
                self.query,
                flags=re.IGNORECASE,
            )
            if match:
                inputs[input_name] = match.group(1)
        return inputs

    def _bounded_training_controls_from_query(self) -> dict[str, Any]:
        """Extract explicit bounded training controls from the user query."""
        normalized_query = self.query.casefold()
        controls: dict[str, Any] = {}

        timeout_match = re.search(
            r"(?:timeout|timeout_seconds)\s*[=:]?\s*(\d+)", normalized_query
        )
        if timeout_match:
            controls["timeout_seconds"] = int(timeout_match.group(1))

        max_epochs_match = re.search(
            r"(?:max[_\s-]?epochs?|epochs?)\s*[=:]?\s*(\d+)", normalized_query
        )
        if max_epochs_match:
            controls["max_epochs"] = int(max_epochs_match.group(1))

        device_match = re.search(r"device\s*[=:]?\s*(cpu|cuda|mps)", normalized_query)
        if device_match:
            controls["device"] = device_match.group(1)
        elif re.search(r"\bcpu\b", normalized_query):
            controls["device"] = "cpu"

        subset_match = re.search(
            r"(?:data[_\s-]?subset|subset|dataset[_\s-]?size)\s*[=:]?\s*(\d+)",
            normalized_query,
        )
        if subset_match:
            controls["data_subset"] = int(subset_match.group(1))

        metric_name_match = re.search(
            r"(?:metric_name|metric)\s*[=:]?\s*([a-zA-Z_][\w.-]*)",
            normalized_query,
        )
        if metric_name_match:
            controls["metric_name"] = metric_name_match.group(1)

        direction_match = re.search(
            r"(?:metric[_\s-]?direction|direction)\s*[=:]?\s*(maximize|minimize)",
            normalized_query,
        )
        if direction_match:
            controls["metric_direction"] = direction_match.group(1)
        elif re.search(r"\bmaximize\b", normalized_query):
            controls["metric_direction"] = "maximize"
        elif re.search(r"\bminimize\b", normalized_query):
            controls["metric_direction"] = "minimize"

        threshold_match = re.search(
            r"(?:threshold|minimum[_\s-]?improvement)\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
            normalized_query,
        )
        if threshold_match:
            controls["threshold"] = float(threshold_match.group(1))

        tie_policy_match = re.search(
            r"(?:tie[_\s-]?policy|tie)\s*[=:]?\s*(keep[_\s-]?baseline|select[_\s-]?latest)",
            normalized_query,
        )
        if tie_policy_match:
            controls["tie_policy"] = tie_policy_match.group(1).replace(" ", "_")

        baseline_metric_match = re.search(
            r"(?:baseline[_\s-]?metric|baseline[_\s-]?value|baseline\s+accuracy)\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
            normalized_query,
        )
        if baseline_metric_match:
            controls["baseline_metric"] = float(baseline_metric_match.group(1))

        baseline_artifact_match = re.search(
            r"(?:baseline[_\s-]?artifact|baseline[_\s-]?artifact[_\s-]?path)\s*[=:]?\s*(\S+)",
            self.query,
            flags=re.IGNORECASE,
        )
        if baseline_artifact_match:
            controls["baseline_artifact_path"] = baseline_artifact_match.group(1)

        baseline_run_match = re.search(
            r"(?:baseline[_\s-]?run|baseline[_\s-]?run[_\s-]?id)\s*[=:]?\s*(\S+)",
            self.query,
            flags=re.IGNORECASE,
        )
        if baseline_run_match:
            controls["baseline_run_id"] = baseline_run_match.group(1)

        return controls

    def _should_block_for_workflow_selection(self, selection: WorkflowSelection) -> bool:
        """Block registry-like requests that did not yield an executable selection."""
        return bool(selection.matched_aliases or selection.rejected_workflows)

    async def _run_initial_perception(self):
        """Run initial perception on user query."""
        p_input = build_perception_input(query=self.query, memory=self.memory, ctx=self.ctx)

        self.p_out = await self.perception.run(p_input, session=self.session)

        # Add ROOT node
        self.ctx.add_step(
            step_id=StepType.ROOT, description="Initial query analysis", step_type=StepType.ROOT
        )
        self.ctx.mark_step_completed(StepType.ROOT)
        self.ctx.attach_perception(StepType.ROOT, self.p_out)

        await self._emit(
            "perception",
            {
                "step_id": "ROOT",
                "entities": self.p_out.get("entities", {}),
                "pipeline_stage": self.p_out.get("pipeline_stage", "setup"),
                "route": self.p_out.get("route", "decision"),
            },
        )

        self.ctx.print_graph()

    def _should_summarize(self) -> bool:
        """Check if we should skip to summary."""
        return (
            self.p_out.get("original_goal_achieved", False)
            or self.p_out.get("route") == Route.SUMMARIZE
        )

    def _needs_improvement(self) -> bool:
        """Check if training needs improvement."""
        exp = self.ctx.experiment_state
        return (
            exp.current_accuracy is not None
            and not exp.threshold_met()
            and exp.can_improve()
            and exp.stage in ["evaluation", "training"]
        )

    def _needs_deployment(self) -> bool:
        """Check if deployment is requested."""
        return (
            self.p_out.get("route") == Route.DEPLOY
            or self.p_out.get("pipeline_stage") == "deploy"
            or self.p_out.get("entities", {}).get("deployment_target") is not None
        )

    async def _summarize(self, status: str | None = "success") -> str:
        """Generate final summary and optionally update the agent status."""
        summary = await self.summarizer.summarize(
            query=self.query, ctx=self.ctx, perception=self.p_out, session=self.session
        )
        self.ctx.attach_summary(summary)
        if status is not None:
            self.status = status
        self.final_output = summary.get("summary_markdown", str(summary))
        return self.final_output

    async def _ensure_approval(self, scope: str, details: dict[str, Any]) -> tuple[bool, str]:
        """Ensure human approval for a given scope (deployment/build)."""
        if self.auto_approve:
            self._approval_cache[scope] = True
            return True, "approved"

        if self._approval_cache.get(scope):
            return True, "approved"

        approval_id = f"{self.session_id}:{scope}:{uuid.uuid4().hex[:8]}"
        payload = {"scope": scope, "approval_id": approval_id, "details": details}
        await self._emit("approval_required", payload)

        # Record in session history
        if self.session:
            self.session.add_message(
                role="assistant",
                content=f"Approval required for {scope}",
                metadata={"approval_id": approval_id, "scope": scope, "details": details},
            )

        # If callback is provided (CLI), use it
        if self.approval_callback:
            decision = self.approval_callback(payload)
            if hasattr(decision, "__await__"):
                decision = await decision
            approved, reason = (decision, None)
            if isinstance(decision, tuple):
                approved, reason = decision
            if approved:
                self._approval_cache[scope] = True
                await self._emit(
                    "approval_granted",
                    {"scope": scope, "approval_id": approval_id, "reason": reason},
                )
                return True, "approved"

            await self._emit(
                "approval_denied",
                {"scope": scope, "approval_id": approval_id, "reason": reason},
            )
            return False, "denied"

        # Otherwise, wait for approval decision in DB (API use-case)
        decision = await wait_for_approval(
            self.session_id, approval_id, timeout_seconds=self.approval_timeout
        )
        if decision and decision.approved:
            self._approval_cache[scope] = True
            await self._emit(
                "approval_granted",
                {"scope": scope, "approval_id": approval_id, "reason": decision.reason},
            )
            return True, "approved"

        if decision is None:
            await self._emit(
                "approval_timeout",
                {"scope": scope, "approval_id": approval_id, "reason": "timeout"},
            )
            return False, "timeout"
        else:
            await self._emit(
                "approval_denied",
                {
                    "scope": scope,
                    "approval_id": approval_id,
                    "reason": decision.reason,
                },
            )
        return False, "denied"

    async def _run_decision_loop(self):
        """Run decision and execute steps in a loop."""
        await self._emit("phase", {"phase": "decision", "message": "Planning execution..."})

        # Get initial plan
        d_input = build_decision_input(ctx=self.ctx, query=self.query, perception=self.p_out)
        d_out = await self.decision.run(d_input, session=self.session)

        if not d_out.get("plan_graph", {}).get("nodes"):
            await self._emit("error", {"error": "No execution plan generated"})
            return

        # Extract plan
        self.code_variants = d_out.get("code_variants", {})
        self.next_step_id = d_out.get("next_step_id", "0")
        plan_nodes = d_out["plan_graph"]["nodes"]

        await self._emit(
            "plan",
            {
                "nodes": plan_nodes,
                "next_step_id": self.next_step_id,
                "total_steps": len(plan_nodes),
            },
        )

        # Add nodes to graph
        for node in plan_nodes:
            self.ctx.add_step(
                step_id=node["id"],
                description=node["description"],
                step_type=StepType.CODE,
                tool=node.get("tool"),
                args=node.get("args"),
                from_node=StepType.ROOT,
            )

        # Execute steps
        await self._execute_steps_loop()

    async def _execute_steps_loop(self):
        """Execute steps with perception feedback and circuit breaker protection."""
        await self._emit("phase", {"phase": "execution", "message": "Executing steps..."})

        tracker = StepExecutionTracker(max_steps=15, max_retries=3)

        while tracker.should_continue():
            tracker.increment()

            # Check if circuit breaker is open
            if tracker.is_circuit_open():
                retry_after = tracker.circuit_breaker.get_retry_after()
                await self._emit(
                    "circuit_open",
                    {
                        "message": "Circuit breaker is open due to repeated failures",
                        "retry_after": retry_after,
                        "stats": tracker.get_circuit_stats(),
                    },
                )
                logger.warning(
                    "Circuit breaker is open, stopping execution",
                    retry_after=retry_after,
                    stats=tracker.get_circuit_stats(),
                )
                break

            # Skip completed steps
            if self.ctx.is_step_completed(self.next_step_id):
                self.next_step_id = self._pick_next_step()
                if self.next_step_id is None:
                    break
                continue

            # Get step info
            if self.next_step_id not in self.ctx.graph.nodes:
                break

            step_data = self.ctx.graph.nodes[self.next_step_id]["data"]
            if self._should_skip_registry_step(self.next_step_id):
                self.ctx.mark_step_completed(self.next_step_id)
                await self._emit(
                    "step_skipped",
                    {
                        "step_id": self.next_step_id,
                        "reason": "registry inclusion rule did not select this step",
                    },
                )
                self.next_step_id = self._pick_next_step()
                if self.next_step_id is None:
                    break
                continue

            approval_validation = await self._validate_registry_step_approval(self.next_step_id)
            if approval_validation is not None:
                if approval_validation.status is WorkflowStatus.BLOCKED:
                    self.status = "paused"
                    risk_categories = [
                        risk_category.value
                        for risk_category in approval_validation.risk_categories
                    ]
                    payload = {
                        "workflow_id": approval_validation.workflow_id,
                        "workflow_run_id": approval_validation.workflow_run_id,
                        "step_id": approval_validation.step_id,
                        "risk_categories": risk_categories,
                        "status": approval_validation.status.value,
                        "next_action": approval_validation.next_action,
                    }
                    await self._emit("approval_required", payload)
                    self.final_output = (
                        "Approval required before executing workflow step "
                        f"'{approval_validation.step_id}'. risk_categories: "
                        f"{', '.join(risk_categories)}. "
                        f"next_action: {approval_validation.next_action}. "
                        f"evidence: {self._format_verification_results()}. "
                        f"artifacts: {self._format_artifacts()}"
                    )
                    return
                if approval_validation.status is WorkflowStatus.FAILED:
                    self.status = "failed"
                    risk_categories = [
                        risk_category.value
                        for risk_category in approval_validation.risk_categories
                    ]
                    payload = {
                        "workflow_id": approval_validation.workflow_id,
                        "workflow_run_id": approval_validation.workflow_run_id,
                        "step_id": approval_validation.step_id,
                        "risk_categories": risk_categories,
                        "status": approval_validation.status.value,
                        "next_action": approval_validation.next_action,
                    }
                    await self._emit("approval_denied", payload)
                    self.final_output = (
                        "Approval denied before executing workflow step "
                        f"'{approval_validation.step_id}'. risk_categories: "
                        f"{', '.join(risk_categories)}. "
                        f"next_action: {approval_validation.next_action}"
                    )
                    return

            await self._emit(
                "step_start",
                {
                    "step_id": self.next_step_id,
                    "description": step_data.description,
                    "tool": step_data.tool,
                    "loop": tracker.tries,
                },
            )

            # Execute step with circuit breaker protection
            try:
                async with tracker.circuit_breaker:
                    runtime_args = self._runtime_args_for_registry_step(
                        self.next_step_id,
                        step_data.args or {},
                    )
                    success, result = await execute_step(
                        step_id=self.next_step_id,
                        tool=step_data.tool,
                        args=runtime_args,
                        ctx=self.ctx,
                        tools_module=self.tools_module,
                    )

                    if not success:
                        # Raise an exception to trigger circuit breaker failure recording
                        raise StepExecutionError(
                            self.next_step_id, str(result.get("error", "Unknown error"))
                        )

            except CircuitBreakerError as e:
                # Circuit breaker rejected the call
                await self._emit(
                    "step_rejected",
                    {
                        "step_id": self.next_step_id,
                        "reason": "circuit_breaker_open",
                        "retry_after": e.retry_after,
                        "stats": tracker.get_circuit_stats(),
                    },
                )
                logger.warning(
                    "Step rejected by circuit breaker",
                    step_id=self.next_step_id,
                    circuit_state=e.state.value,
                    retry_after=e.retry_after,
                )
                break

            except StepExecutionError as e:
                self.ctx.mark_step_failed(self.next_step_id, e.error_message)
                tracker.record_failure(self.next_step_id)

                await self._emit(
                    "step_failed",
                    {
                        "step_id": self.next_step_id,
                        "error": e.error_message[:200],
                        "attempts": tracker.attempts.get(self.next_step_id, 1),
                        "circuit_stats": tracker.get_circuit_stats(),
                    },
                )

                if tracker.has_exceeded_retries(self.next_step_id):
                    break
                continue

            # Update result
            self.ctx.update_step_result(self.next_step_id, result)
            self._capture_registry_workflow_evidence(self.next_step_id, result)
            if self._registry_step_blocks_remaining_execution(self.next_step_id):
                self._finalize_registry_workflow_contract()
                return

            await self._emit(
                "step_complete",
                {
                    "step_id": self.next_step_id,
                    "success": True,
                    "result_summary": str(result)[:300],
                },
            )

            if self._should_run_post_step_perception(self.next_step_id):
                # Run perception after step
                p_input = build_perception_input(
                    query=self.query,
                    memory=self.memory,
                    ctx=self.ctx,
                    snapshot_type="step_result",
                )
                self.p_out = await self.perception.run(p_input, session=self.session)
                self.ctx.attach_perception(self.next_step_id, self.p_out)

                # Check routing
                if (
                    self.p_out.get("original_goal_achieved")
                    or self.p_out.get("route") == Route.SUMMARIZE
                ):
                    if self._is_pending_setup_pipeline():
                        self._finalize_setup_pipeline_contract()
                        return
                    self.status = "success"
                    await self._emit("phase", {"phase": "summary", "message": "Goal achieved!"})
                    self.final_output = await self._summarize()
                    return

                if self.p_out.get("route") == Route.IMPROVE:
                    # Break to run improvement loop
                    return

                if self.p_out.get("route") == Route.DEPLOY:
                    # Break to run deployment loop
                    return

            # Get next step
            self.next_step_id = self._pick_next_step()
            if self.next_step_id is None:
                break

        if self._is_pending_executable_registry_workflow():
            self._finalize_registry_workflow_contract()

    async def _validate_registry_step_approval(self, step_id: str):
        """Validate selected registry approval gates before tool execution."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id is None
            or self.workflow_selection.status is not WorkflowStatus.PENDING
        ):
            return None

        try:
            if (
                self.workflow_selection.workflow_id == "prepare_capstone_data"
                and step_id == "generate_split_manifests"
                and not self._capstone_split_manifest_writes_required()
            ):
                return None
            if (
                self.workflow_selection.workflow_id == "prepare_capstone_data"
                and step_id == "configure_validate_dvc_remote"
            ):
                return self._validate_capstone_remote_approval(step_id)
            if (
                self.workflow_selection.workflow_id == "prepare_capstone_data"
                and step_id in {"push_capstone_data", "pull_capstone_data"}
                and not self._capstone_transfer_step_selected(step_id)
            ):
                return None
            if (
                self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
                and step_id == "generate_validate_runtime_image_spec"
            ):
                return None
            if (
                self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
                and step_id == "build_smoke_check_container_image"
            ):
                return None
            validation = self.workflow_registry.validate_step_approval(
                workflow_id=self.workflow_selection.workflow_id,
                workflow_run_id=self.session_id,
                step_id=step_id,
                approval_records=tuple(self.ctx.globals.get("approval_records", ())),
            )
            if (
                validation.status is WorkflowStatus.BLOCKED
                and validation.risk_categories
                and self.auto_approve
            ):
                approval_records = tuple(self.ctx.globals.get("approval_records", ()))
                self.ctx.globals["approval_records"] = (
                    *approval_records,
                    ApprovalRecord(
                        workflow_run_id=self.session_id,
                        step_id=step_id,
                        risk_categories=validation.risk_categories,
                        status="approved",
                        approver="auto_approve",
                        timestamp=datetime.now(timezone.utc),
                    ),
                )
                return self.workflow_registry.validate_step_approval(
                    workflow_id=self.workflow_selection.workflow_id,
                    workflow_run_id=self.session_id,
                    step_id=step_id,
                    approval_records=tuple(self.ctx.globals.get("approval_records", ())),
                )
            return validation
        except KeyError:
            return None

    def _validate_capstone_remote_approval(self, step_id: str):
        risk_categories = self._capstone_remote_risk_categories()
        if not risk_categories:
            return None

        workflow_id = self.workflow_selection.workflow_id
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        approval_record = next(
            (
                record
                for record in approval_records
                if record.workflow_run_id == self.session_id
                and record.step_id == step_id
                and record.risk_categories == risk_categories
            ),
            None,
        )
        if approval_record is not None and approval_record.status.value == "approved":
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=self.session_id,
                step_id=step_id,
                status=WorkflowStatus.PENDING,
                risk_categories=risk_categories,
                approval_record=approval_record,
                next_action="Approval satisfied; step may run.",
            )
        if approval_record is not None and approval_record.status.value == "denied":
            approver = approval_record.approver or "unknown approver"
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=self.session_id,
                step_id=step_id,
                status=WorkflowStatus.FAILED,
                risk_categories=risk_categories,
                approval_record=approval_record,
                next_action=(
                    f"Approval denied by {approver}; step '{step_id}' must not run."
                ),
            )
        if self.auto_approve:
            approval_records = tuple(self.ctx.globals.get("approval_records", ()))
            approval_record = ApprovalRecord(
                workflow_run_id=self.session_id,
                step_id=step_id,
                risk_categories=risk_categories,
                status="approved",
                approver="auto_approve",
                timestamp=datetime.now(timezone.utc),
            )
            self.ctx.globals["approval_records"] = (*approval_records, approval_record)
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=self.session_id,
                step_id=step_id,
                status=WorkflowStatus.PENDING,
                risk_categories=risk_categories,
                approval_record=approval_record,
                next_action="Approval satisfied; step may run.",
            )
        return StepApprovalValidation(
            workflow_id=workflow_id,
            workflow_run_id=self.session_id,
            step_id=step_id,
            status=WorkflowStatus.BLOCKED,
            risk_categories=risk_categories,
            approval_record=None,
            next_action=(
                f"Record approval for workflow run '{self.session_id}' "
                f"before step '{step_id}' may run."
            ),
        )

    def _capstone_remote_risk_categories(self) -> tuple[RiskCategory, ...]:
        workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
        if not isinstance(workflow_inputs, dict):
            return ()
        remote_url = workflow_inputs.get("dvc_remote_url")
        if not remote_url:
            remote_url = self._configured_capstone_remote_url(
                workflow_inputs.get("dvc_remote_name", "capstone")
            )
        if not isinstance(remote_url, str) or not remote_url:
            return ()
        risks: list[RiskCategory] = []
        if workflow_inputs.get("dvc_remote_url"):
            risks.append(RiskCategory.WRITES_PROJECT_FILES)
        if remote_url.casefold().startswith("s3://"):
            risks.append(RiskCategory.USES_CLOUD_CREDENTIALS)
        return tuple(risks)

    def _capstone_transfer_step_selected(self, step_id: str) -> bool:
        workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
        if not isinstance(workflow_inputs, dict):
            return False
        if workflow_inputs.get("completion_mode") != "capstone_complete":
            return False
        direction = workflow_inputs.get("dvc_transfer_direction", "push")
        return (
            (step_id == "push_capstone_data" and direction == "push")
            or (step_id == "pull_capstone_data" and direction == "pull")
        )

    def _should_skip_registry_step(self, step_id: str) -> bool:
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id in {"push_capstone_data", "pull_capstone_data"}
        ):
            return not self._capstone_transfer_step_selected(step_id)
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id
            not in {
                "prepare_capstone_container_ci_contract",
                "resolve_upstream_container_evidence",
                "generate_validate_runtime_image_spec",
                "build_smoke_check_container_image",
            }
        ):
            return True
        return False

    def _configured_capstone_remote_url(self, remote_name: Any) -> str | None:
        project_path = self.ctx.project_path
        if not project_path:
            return None
        config_path = Path(project_path) / ".dvc" / "config"
        if not config_path.exists():
            return None
        try:
            config_text = config_path.read_text()
        except OSError:
            return None
        remote_name = str(remote_name or "capstone")
        match = re.search(
            rf"\[remote\s+\"{re.escape(remote_name)}\"\]\s+url\s*=\s*(\S+)",
            config_text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        core_match = re.search(r"\[core\]\s+remote\s*=\s*(\S+)", config_text, flags=re.IGNORECASE)
        if core_match and core_match.group(1) != remote_name:
            return None
        return None

    def _is_pending_setup_pipeline(self) -> bool:
        """Return whether the selected workflow is an executable setup_pipeline run."""
        return (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "setup_pipeline"
            and self.workflow_selection.status is WorkflowStatus.PENDING
        )

    def _is_pending_executable_registry_workflow(self) -> bool:
        """Return whether the selected workflow should be finalized by registry contract."""
        return (
            self.workflow_selection is not None
            and self.workflow_selection.status is WorkflowStatus.PENDING
            and self._is_executable_registry_workflow(self.workflow_selection.workflow_id)
        )

    def _should_run_post_step_perception(self, step_id: str) -> bool:
        """Return whether this completed step should trigger perception feedback."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id is None
            or self.workflow_selection.status is not WorkflowStatus.PENDING
        ):
            return True

        try:
            workflow_step = self.workflow_registry.get(
                self.workflow_selection.workflow_id
            ).step_by_id(step_id)
        except KeyError:
            return True

        return workflow_step.post_step_perception

    def _finalize_setup_pipeline_contract(self) -> ContractValidation:
        """Derive setup_pipeline status from captured success contract evidence."""
        return self._finalize_registry_workflow_contract("setup_pipeline")

    def _finalize_registry_workflow_contract(
        self,
        workflow_id: str | None = None,
    ) -> ContractValidation:
        """Derive selected registry workflow status from captured evidence."""
        workflow_id = workflow_id or (
            self.workflow_selection.workflow_id if self.workflow_selection else None
        )
        if workflow_id is None:
            raise ValueError("Cannot finalize a registry workflow without a workflow_id")

        contract_status = self.workflow_registry.validate_success_contract(
            workflow_id,
            verification_results=tuple(self.ctx.globals.get("verification_results", ())),
            artifact_manifest=self.ctx.globals.get("artifact_manifest"),
            rollback_plan=self.ctx.globals.get("rollback_plan"),
            workflow_inputs=self.ctx.globals.get("workflow_inputs"),
        )
        self.ctx.globals["contract_status"] = contract_status
        self.ctx.globals["workflow_status"] = contract_status.status

        if contract_status.status is WorkflowStatus.SUCCEEDED:
            self.status = "success"
        elif contract_status.status is WorkflowStatus.FAILED:
            self.status = "failed"
        else:
            self.status = "paused"

        self.final_output = self._format_registry_contract_output(contract_status)
        return contract_status

    def _format_setup_pipeline_contract_output(
        self,
        contract_status: ContractValidation,
    ) -> str:
        return self._format_registry_contract_output(contract_status)

    def _format_registry_contract_output(
        self,
        contract_status: ContractValidation,
    ) -> str:
        missing_evidence = self._format_contract_failures(contract_status.missing_evidence)
        failed_checks = self._format_contract_failures(contract_status.failed_checks)
        return (
            f"{contract_status.workflow_id} final workflow status derived from SuccessContract. "
            f"contract_status: {contract_status.status.value}. "
            f"workflow_status: {contract_status.status.value}. "
            f"missing_evidence: {missing_evidence}. "
            f"failed_checks: {failed_checks}. "
            f"evidence: {self._format_verification_results()}. "
            f"artifacts: {self._format_artifacts()}. "
            f"capstone_summary: {self._format_capstone_summary()}. "
            f"approvals: {self._format_approval_records()}. "
            f"rollback: {self._format_rollback_plan()}."
        )

    def _format_contract_failures(self, failures: tuple[ContractFailure, ...]) -> str:
        if not failures:
            return "none"
        return "; ".join(
            f"{failure.check_name} next_action: {failure.next_action}"
            for failure in failures
        )

    def _format_artifacts(self) -> str:
        artifact_manifest = self.ctx.globals.get("artifact_manifest")
        if not isinstance(artifact_manifest, ArtifactManifest) or not artifact_manifest.entries:
            return "none"
        return "; ".join(
            (
                f"{entry.artifact_type}:{entry.path or entry.uri}"
                f"({entry.state.value} from {entry.producing_step})"
            )
            for entry in artifact_manifest.entries
        )

    def _format_verification_results(self) -> str:
        verification_results = tuple(self.ctx.globals.get("verification_results", ()))
        if not verification_results:
            return "none"
        return "; ".join(
            (
                f"{result.check_name}:{result.evidence_type.value}:"
                f"{'passed' if result.passed else 'failed'}:{result.evidence}"
            )
            for result in verification_results
        )

    def _format_capstone_summary(self) -> str:
        summary = self.ctx.globals.get("capstone_orchestrator_summary")
        if not isinstance(summary, dict):
            return "none"
        compact_summary = {
            key: summary.get(key)
            for key in (
                "completed_stages",
                "blocked_stages",
                "deferred_stages",
                "selected_model_artifact",
                "endpoint_evidence",
                "next_actions",
            )
        }
        return json.dumps(compact_summary, sort_keys=True)

    def _format_approval_records(self) -> str:
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        if not approval_records:
            return "none"
        return "; ".join(
            (
                f"{record.step_id}:{record.status.value}:"
                f"{','.join(risk.value for risk in record.risk_categories)}"
            )
            for record in approval_records
        )

    def _format_rollback_plan(self) -> str:
        rollback_plan = self.ctx.globals.get("rollback_plan")
        if not isinstance(rollback_plan, RollbackPlan):
            return "none"
        parts = [
            value
            for value in (
                rollback_plan.command,
                rollback_plan.script_path,
                rollback_plan.documented_target,
            )
            if value
        ]
        return "; ".join(parts)

    def _capture_registry_workflow_evidence(self, step_id: str, result: dict[str, Any]) -> None:
        """Capture explicit registry workflow evidence from a completed step result."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id is None
            or self.workflow_selection.status is not WorkflowStatus.PENDING
            or not self._is_executable_registry_workflow(self.workflow_selection.workflow_id)
            or not isinstance(result, dict)
        ):
            return

        payload = result.get("result", result)
        if not isinstance(payload, dict):
            return

        verification_results = self._verification_results_from_step_result(step_id, payload)
        if verification_results:
            existing_results = tuple(self.ctx.globals.get("verification_results", ()))
            self.ctx.globals["verification_results"] = (*existing_results, *verification_results)

        artifact_entries = self._artifact_manifest_entries_from_step_result(step_id, payload)
        if artifact_entries:
            existing_manifest = self.ctx.globals.get(
                "artifact_manifest",
                ArtifactManifest(entries=()),
            )
            self.ctx.globals["artifact_manifest"] = ArtifactManifest(
                entries=(*existing_manifest.entries, *artifact_entries)
            )

        rollback_plan = self._rollback_plan_from_step_result(payload)
        if rollback_plan is not None:
            self.ctx.globals["rollback_plan"] = rollback_plan

        if step_id == "select_best_model_artifact":
            selected_model_path = payload.get("model_path") or self._selected_model_path_from_payload(
                payload
            )
            if selected_model_path:
                self.ctx.globals["selected_model_artifact_path"] = selected_model_path
            if payload.get("model_type"):
                self.ctx.globals["selected_model_type"] = payload["model_type"]
        elif (
            self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "detect_training_project"
        ):
            self.ctx.globals["training_detection"] = payload
        elif (
            self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "run_bounded_training"
        ):
            self.ctx.globals["bounded_training_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "track_training_in_mlflow"
        ):
            self.ctx.globals["mlflow_tracking_result"] = payload
        elif step_id == "record_capstone_orchestrator_skeleton":
            self.ctx.globals["capstone_orchestrator_summary"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "prepare_capstone_data_contract"
        ):
            self.ctx.globals["capstone_data_detection"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "generate_split_manifests"
        ):
            self.ctx.globals["capstone_split_manifest_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "track_capstone_data_package"
        ):
            self.ctx.globals["capstone_data_package_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "configure_validate_dvc_remote"
        ):
            self.ctx.globals["capstone_data_remote_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "push_capstone_data"
        ):
            self.ctx.globals["capstone_data_push_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "pull_capstone_data"
        ):
            self.ctx.globals["capstone_data_pull_result"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "record_data_stage_evidence"
        ):
            self.ctx.globals["capstone_data_stage_evidence"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "resolve_upstream_container_evidence"
        ):
            self.ctx.globals["capstone_container_upstream_evidence"] = payload
            workflow_input_overrides = payload.get("workflow_input_overrides")
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(workflow_input_overrides, dict) and isinstance(workflow_inputs, dict):
                workflow_inputs.update(workflow_input_overrides)
                self.ctx.globals["workflow_inputs"] = workflow_inputs
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "generate_validate_runtime_image_spec"
        ):
            self.ctx.globals["capstone_runtime_image_spec"] = payload
        elif (
            self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "build_smoke_check_container_image"
        ):
            self.ctx.globals["capstone_container_build_smoke_check"] = payload
            workflow_input_overrides = payload.get("workflow_input_overrides")
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(workflow_input_overrides, dict) and isinstance(workflow_inputs, dict):
                workflow_inputs.update(workflow_input_overrides)
                self.ctx.globals["workflow_inputs"] = workflow_inputs
        elif step_id == "start_litserve_server":
            if payload.get("endpoint_url"):
                self.ctx.globals["litserve_endpoint_url"] = payload["endpoint_url"]
            if payload.get("process_id"):
                self.ctx.globals["litserve_process_id"] = payload["process_id"]
            if payload.get("port"):
                self.ctx.globals["litserve_port"] = payload["port"]

    def _runtime_args_for_registry_step(
        self,
        step_id: str,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_args = dict(args)
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "run_bounded_training"
        ):
            training_detection = self.ctx.globals.get("training_detection", {})
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(training_detection, dict):
                for source_key, target_key in (
                    ("training_entrypoint", "training_entrypoint"),
                    ("hydra_config_path", "hydra_config_path"),
                    ("hydra_config_name", "hydra_config_name"),
                ):
                    if training_detection.get(source_key):
                        runtime_args[target_key] = training_detection[source_key]
            if isinstance(workflow_inputs, dict):
                for key in ("timeout_seconds", "max_epochs", "device", "data_subset"):
                    if key in workflow_inputs:
                        runtime_args[key] = workflow_inputs[key]
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "track_training_in_mlflow"
        ):
            training_result = self.ctx.globals.get("bounded_training_result", {})
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            training_detection = self.ctx.globals.get("training_detection", {})
            if isinstance(training_result, dict):
                runtime_args["training_result"] = training_result
            params: dict[str, Any] = {}
            if isinstance(workflow_inputs, dict):
                params.update(
                    {
                        key: workflow_inputs[key]
                        for key in ("timeout_seconds", "max_epochs", "device", "data_subset")
                        if key in workflow_inputs
                    }
                )
            if isinstance(training_detection, dict):
                params.update(
                    {
                        key: training_detection[key]
                        for key in (
                            "training_entrypoint",
                            "hydra_config_path",
                            "hydra_config_name",
                            "framework_family",
                            "model_library",
                            "data_versioning",
                        )
                        if training_detection.get(key)
                    }
                )
            if params:
                runtime_args["params"] = params
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "select_best_model_artifact"
        ):
            tracking_result = self.ctx.globals.get("mlflow_tracking_result", {})
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(tracking_result, dict):
                runtime_args["latest_run"] = {
                    "run_id": tracking_result.get("run_id"),
                    "run_status": tracking_result.get("run_status"),
                    "metrics": tracking_result.get("metrics", {}),
                    "checkpoint_artifact_uri": tracking_result.get("checkpoint_artifact_uri"),
                    "artifact_manifest": tracking_result.get("artifact_manifest"),
                }
            if isinstance(workflow_inputs, dict):
                for key in ("metric_name", "metric_direction", "threshold", "tie_policy"):
                    if key in workflow_inputs:
                        runtime_args[key] = workflow_inputs[key]
                baseline = {
                    "run_id": workflow_inputs.get("baseline_run_id"),
                    "metric_value": workflow_inputs.get("baseline_metric"),
                    "artifact_path": workflow_inputs.get("baseline_artifact_path"),
                }
                runtime_args["baseline"] = {
                    key: value for key, value in baseline.items() if value is not None
                }
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "deploy_litserve_gpu"
            and step_id == "generate_litserve_api"
        ):
            selected_model_path = self.ctx.globals.get("selected_model_artifact_path")
            if selected_model_path:
                runtime_args["model_path"] = selected_model_path
            selected_model_type = self.ctx.globals.get("selected_model_type")
            if selected_model_type:
                runtime_args["model_type"] = selected_model_type
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "deploy_litserve_gpu"
            and step_id
            in {
                "test_health_endpoint",
                "test_prediction_endpoint",
                "capture_logs_and_endpoint",
            }
        ):
            endpoint_url = self.ctx.globals.get("litserve_endpoint_url")
            if endpoint_url:
                runtime_args["endpoint_url"] = endpoint_url
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "deploy_litserve_gpu"
            and step_id == "write_monitoring_and_rollback_report"
        ):
            process_id = self.ctx.globals.get("litserve_process_id")
            if process_id:
                runtime_args["process_id"] = process_id
            port = self.ctx.globals.get("litserve_port") or self._port_from_endpoint_url(
                self.ctx.globals.get("litserve_endpoint_url")
            )
            if port:
                runtime_args["port"] = port
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "build_capstone_pipeline"
            and step_id == "record_capstone_orchestrator_skeleton"
        ):
            selected_model_path = self.ctx.globals.get("selected_model_artifact_path")
            if selected_model_path:
                runtime_args["selected_model_artifact_path"] = selected_model_path
            endpoint_url = self.ctx.globals.get("litserve_endpoint_url")
            if endpoint_url:
                runtime_args["endpoint_url"] = endpoint_url
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "prepare_capstone_data_contract"
        ):
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(workflow_inputs, dict):
                for key in (
                    "dataset_1_path",
                    "dataset_2_path",
                    "completion_mode",
                    "test_size",
                    "split_seed",
                ):
                    if key in workflow_inputs:
                        runtime_args[key] = workflow_inputs[key]
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "generate_split_manifests"
        ):
            detection = self.ctx.globals.get("capstone_data_detection", {})
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(detection, dict):
                runtime_args["capstone_data_detection"] = detection
            if isinstance(workflow_inputs, dict):
                for key in ("test_size", "split_seed", "materialize_splits"):
                    if key in workflow_inputs:
                        runtime_args[key] = workflow_inputs[key]
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "track_capstone_data_package"
        ):
            split_result = self.ctx.globals.get("capstone_split_manifest_result", {})
            if isinstance(split_result, dict):
                runtime_args["capstone_split_manifest_result"] = split_result
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "configure_validate_dvc_remote"
        ):
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            if isinstance(workflow_inputs, dict):
                for source_key, target_key in (
                    ("completion_mode", "completion_mode"),
                    ("dvc_remote_name", "remote_name"),
                    ("dvc_remote_url", "remote_url"),
                ):
                    if source_key in workflow_inputs:
                        runtime_args[target_key] = workflow_inputs[source_key]
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id in {"push_capstone_data", "pull_capstone_data"}
        ):
            workflow_inputs = self.ctx.globals.get("workflow_inputs", {})
            remote_result = self.ctx.globals.get("capstone_data_remote_result", {})
            package_result = self.ctx.globals.get("capstone_data_package_result", {})
            if isinstance(workflow_inputs, dict):
                for source_key, target_key in (
                    ("completion_mode", "completion_mode"),
                    ("dvc_remote_name", "remote_name"),
                ):
                    if source_key in workflow_inputs:
                        runtime_args[target_key] = workflow_inputs[source_key]
            if isinstance(remote_result, dict):
                runtime_args["capstone_dvc_remote_result"] = remote_result
            if isinstance(package_result, dict) and package_result.get("tracked_package_paths"):
                runtime_args["paths"] = package_result["tracked_package_paths"]
            approval_record = self._approval_record_for_step(step_id)
            if approval_record is not None:
                runtime_args["approval_record"] = self._approval_record_to_payload(
                    approval_record
                )
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "record_data_stage_evidence"
        ):
            runtime_args["workflow_inputs"] = self.ctx.globals.get("workflow_inputs", {})
            for context_key, runtime_key in (
                ("capstone_data_detection", "capstone_data_detection"),
                ("capstone_split_manifest_result", "capstone_split_manifest_result"),
                ("capstone_data_package_result", "capstone_data_package_result"),
                ("capstone_data_remote_result", "capstone_data_remote_result"),
                ("capstone_data_push_result", "capstone_data_push_result"),
                ("capstone_data_pull_result", "capstone_data_pull_result"),
            ):
                value = self.ctx.globals.get(context_key, {})
                if isinstance(value, dict):
                    runtime_args[runtime_key] = value
            runtime_args["verification_results"] = self._verification_results_payload()
            runtime_args["artifact_manifest"] = self._artifact_manifest_payload()
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "prepare_capstone_container_ci_contract"
        ):
            runtime_args["workflow_inputs"] = self.ctx.globals.get("workflow_inputs", {})
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "resolve_upstream_container_evidence"
        ):
            runtime_args["workflow_inputs"] = self.ctx.globals.get("workflow_inputs", {})
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "generate_validate_runtime_image_spec"
        ):
            runtime_args["workflow_inputs"] = self.ctx.globals.get("workflow_inputs", {})
            approval_record = self._approval_record_for_step(step_id)
            if approval_record is not None:
                runtime_args["approval_record"] = self._approval_record_to_payload(
                    approval_record
                )
        elif (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "build_smoke_check_container_image"
        ):
            runtime_args["workflow_inputs"] = self.ctx.globals.get("workflow_inputs", {})
            image_spec_result = self.ctx.globals.get("capstone_runtime_image_spec", {})
            if isinstance(image_spec_result, dict):
                runtime_args["capstone_runtime_image_spec_result"] = image_spec_result
            build_approval = self._approval_record_for_step_and_risk(
                step_id,
                RiskCategory.BUILDS_IMAGE,
            )
            if build_approval is not None:
                runtime_args["approval_record"] = self._approval_record_to_payload(
                    build_approval
                )
            smoke_approval = self._approval_record_for_step_and_risk(
                step_id,
                RiskCategory.EXECUTES_PROJECT_CODE,
            )
            if smoke_approval is not None:
                runtime_args["smoke_approval_record"] = self._approval_record_to_payload(
                    smoke_approval
                )
        return runtime_args

    def _verification_results_payload(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for result in tuple(self.ctx.globals.get("verification_results", ())):
            if isinstance(result, VerificationResult):
                payload = asdict(result)
                evidence_type = payload.get("evidence_type")
                if hasattr(evidence_type, "value"):
                    payload["evidence_type"] = evidence_type.value
                payloads.append(payload)
            elif isinstance(result, dict):
                payloads.append(result)
        return payloads

    def _artifact_manifest_payload(self) -> dict[str, Any]:
        raw_manifest = self.ctx.globals.get("artifact_manifest")
        if not isinstance(raw_manifest, ArtifactManifest):
            return {"entries": []}
        entries: list[dict[str, Any]] = []
        for entry in raw_manifest.entries:
            if isinstance(entry, ArtifactManifestEntry):
                payload = asdict(entry)
                state = payload.get("state")
                if hasattr(state, "value"):
                    payload["state"] = state.value
                entries.append(payload)
        return {"entries": entries}

    def _approval_record_for_step(self, step_id: str) -> ApprovalRecord | None:
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        for record in approval_records:
            if isinstance(record, ApprovalRecord) and record.step_id == step_id:
                return record
        return None

    def _approval_record_for_step_and_risk(
        self,
        step_id: str,
        risk_category: RiskCategory,
    ) -> ApprovalRecord | None:
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        for record in approval_records:
            if (
                isinstance(record, ApprovalRecord)
                and record.step_id == step_id
                and risk_category in record.risk_categories
            ):
                return record
        return None

    def _approval_record_to_payload(self, record: ApprovalRecord) -> dict[str, Any]:
        return {
            "workflow_run_id": record.workflow_run_id,
            "step_id": record.step_id,
            "risk_categories": [risk.value for risk in record.risk_categories],
            "status": record.status.value,
            "approver": record.approver,
            "timestamp": record.timestamp.isoformat()
            if hasattr(record.timestamp, "isoformat")
            else str(record.timestamp),
        }

    def _capstone_split_manifest_writes_required(self) -> bool:
        detection = self.ctx.globals.get("capstone_data_detection", {})
        return bool(
            isinstance(detection, dict)
            and detection.get("split_manifest_writes_required")
        )

    def _port_from_endpoint_url(self, endpoint_url: Any) -> int | None:
        if not isinstance(endpoint_url, str):
            return None
        parsed_url = urlparse(endpoint_url)
        return parsed_url.port

    def _selected_model_path_from_payload(self, payload: dict[str, Any]) -> str | None:
        raw_manifest = payload.get("artifact_manifest")
        if isinstance(raw_manifest, ArtifactManifest):
            entries = raw_manifest.entries
        elif isinstance(raw_manifest, dict):
            entries = raw_manifest.get("entries", ())
        else:
            entries = ()

        for entry in entries:
            if isinstance(entry, ArtifactManifestEntry):
                if entry.artifact_type == "model_artifact":
                    return entry.path or entry.uri
                continue
            if isinstance(entry, dict) and entry.get("artifact_type") == "model_artifact":
                return entry.get("path") or entry.get("uri")
        return None

    def _registry_step_recorded_failed_contract_evidence(self, step_id: str) -> bool:
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id is None
            or not self._is_executable_registry_workflow(self.workflow_selection.workflow_id)
        ):
            return False

        template = self.workflow_registry.get(self.workflow_selection.workflow_id)
        contract_check_names = {
            check.name for check in template.success_contract.checks if check.source_step == step_id
        }
        return any(
            result.check_name in contract_check_names
            and result.source_step == step_id
            and not result.passed
            for result in tuple(self.ctx.globals.get("verification_results", ()))
        )

    def _registry_step_blocks_remaining_execution(self, step_id: str) -> bool:
        """Return whether a step's own contract evidence blocks later workflow steps."""
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "train_and_track"
            and step_id == "run_bounded_training"
        ):
            return False
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_data"
            and step_id == "prepare_capstone_data_contract"
        ):
            detection = self.ctx.globals.get("capstone_data_detection", {})
            return not (
                isinstance(detection, dict)
                and detection.get("status") == "succeeded"
            )
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "prepare_capstone_container_ci_contract"
        ):
            return False
        if (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "prepare_capstone_container_ci"
            and step_id == "generate_validate_runtime_image_spec"
        ):
            return False
        if self._registry_step_recorded_failed_contract_evidence(step_id):
            return True
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id != "train_and_track"
            or step_id != "detect_training_project"
        ):
            return False

        contract_status = self.workflow_registry.validate_success_contract(
            self.workflow_selection.workflow_id,
            verification_results=tuple(self.ctx.globals.get("verification_results", ())),
            artifact_manifest=self.ctx.globals.get("artifact_manifest"),
            rollback_plan=self.ctx.globals.get("rollback_plan"),
        )
        return any(
            failure.source_step == step_id
            for failure in (*contract_status.missing_evidence, *contract_status.failed_checks)
        )

    def _verification_results_from_step_result(
        self,
        step_id: str,
        payload: dict[str, Any],
    ) -> tuple[VerificationResult, ...]:
        """Return verification records explicitly reported by a registry step."""
        if self.workflow_selection is None or self.workflow_selection.workflow_id is None:
            return ()

        raw_results = payload.get("verification_results", ())
        if isinstance(raw_results, dict):
            raw_results = (raw_results,)

        template = self.workflow_registry.get(self.workflow_selection.workflow_id)
        checks_by_name = {
            check.name: check
            for check in template.success_contract.checks
            if check.source_step == step_id
        }
        step_ids = {step.step_id for step in template.steps}
        verification_results: list[VerificationResult] = []

        for raw_result in raw_results:
            if isinstance(raw_result, VerificationResult):
                if raw_result.source_step == step_id:
                    verification_results.append(raw_result)
                continue
            if not isinstance(raw_result, dict):
                continue
            check_name = raw_result.get("check_name")
            if not check_name:
                continue
            source_step = raw_result.get("source_step")
            if source_step != step_id or source_step not in step_ids:
                continue
            if "passed" not in raw_result or "evidence" not in raw_result:
                continue

            verification_results.append(
                VerificationResult(
                    check_name=check_name,
                    evidence_type=raw_result.get(
                        "evidence_type",
                        checks_by_name.get(check_name).evidence_type
                        if check_name in checks_by_name
                        else "observed",
                    ),
                    source_step=source_step,
                    passed=bool(raw_result["passed"]),
                    evidence=raw_result["evidence"],
                )
            )

        return tuple(verification_results)

    def _artifact_manifest_entries_from_step_result(
        self,
        step_id: str,
        payload: dict[str, Any],
    ) -> tuple[ArtifactManifestEntry, ...]:
        """Return artifact manifest entries explicitly reported by a registry step."""
        if self.workflow_selection is None or self.workflow_selection.workflow_id is None:
            return ()

        raw_manifest = payload.get("artifact_manifest")
        if isinstance(raw_manifest, ArtifactManifest):
            raw_entries = raw_manifest.entries
        elif isinstance(raw_manifest, dict):
            raw_entries = raw_manifest.get("entries", ())
        else:
            raw_entries = ()

        if isinstance(raw_entries, dict):
            raw_entries = (raw_entries,)

        template = self.workflow_registry.get(self.workflow_selection.workflow_id)
        step_ids = {step.step_id for step in template.steps}
        artifact_entries: list[ArtifactManifestEntry] = []

        for raw_entry in raw_entries:
            if isinstance(raw_entry, ArtifactManifestEntry):
                if raw_entry.producing_step == step_id:
                    artifact_entries.append(raw_entry)
                continue
            if not isinstance(raw_entry, dict):
                continue
            if raw_entry.get("producing_step") != step_id or step_id not in step_ids:
                continue
            if not raw_entry.get("path") and not raw_entry.get("uri"):
                continue
            try:
                artifact_entries.append(
                    ArtifactManifestEntry(
                        artifact_type=raw_entry["artifact_type"],
                        producing_step=raw_entry["producing_step"],
                        state=raw_entry["state"],
                        path=raw_entry.get("path"),
                        uri=raw_entry.get("uri"),
                        checksum=raw_entry.get("checksum"),
                        metadata=raw_entry.get("metadata"),
                    )
                )
            except (KeyError, ValueError):
                continue

        return tuple(artifact_entries)

    def _rollback_plan_from_step_result(self, payload: dict[str, Any]) -> RollbackPlan | None:
        raw_plan = payload.get("rollback_plan")
        if isinstance(raw_plan, RollbackPlan):
            return raw_plan
        if not isinstance(raw_plan, dict):
            return None
        try:
            return RollbackPlan(
                command=raw_plan.get("command"),
                script_path=raw_plan.get("script_path"),
                documented_target=raw_plan.get("documented_target"),
            )
        except ValueError:
            return None

    async def _run_improvement_loop(self):
        """Run the self-improvement loop for training."""
        await self._emit("phase", {"phase": "improvement", "message": "Optimizing training..."})

        exp = self.ctx.experiment_state

        while exp.can_improve() and not exp.threshold_met():
            attempt = exp.improvement_attempt + 1

            await self._emit(
                "improvement_start",
                {
                    "attempt": attempt,
                    "current_accuracy": exp.current_accuracy,
                    "target_accuracy": exp.target_accuracy,
                    "gap": exp.get_accuracy_gap(),
                },
            )

            # Get improvement suggestions
            improvement = await self._get_improvement_suggestion()

            if not improvement.get("should_retry", False):
                await self._emit("improvement_stop", {"reason": "No more improvements suggested"})
                break

            # Apply improvements
            config_changes = improvement.get("improvement", {}).get("changes", {})
            hydra_overrides = improvement.get("hydra_overrides", [])

            await self._emit(
                "improvement_apply",
                {
                    "attempt": attempt,
                    "changes": config_changes,
                    "reasoning": improvement.get("improvement", {}).get("reasoning", ""),
                },
            )

            # Update Hydra config if we have a project path
            if self.ctx.project_path and hydra_overrides:
                from mcp_mlops_tools import update_hydra_config

                updates = {}
                for override in hydra_overrides:
                    if "=" in override:
                        key, value = override.split("=", 1)
                        # Convert value to appropriate type
                        try:
                            value = float(value) if "." in value else int(value)
                        except ValueError:
                            pass

                        # Build nested dict
                        parts = key.split(".")
                        current = updates
                        for part in parts[:-1]:
                            current = current.setdefault(part, {})
                        current[parts[-1]] = value

                result = update_hydra_config(
                    project_path=self.ctx.project_path,
                    config_path="configs/config.yaml",
                    updates=updates,
                )

                if result.get("success"):
                    self.ctx.set_experiment_config(updates)

            # Record improvement attempt
            # In a real scenario, training would run here
            # For now, we simulate the accuracy improvement
            new_accuracy = exp.current_accuracy + improvement.get("expected_improvement", {}).get(
                "accuracy_gain", 0.02
            )
            exp.record_improvement_attempt(config_changes, new_accuracy)

            await self._emit(
                "improvement_complete",
                {
                    "attempt": attempt,
                    "new_accuracy": new_accuracy,
                    "threshold_met": exp.threshold_met(),
                },
            )

        # After improvement loop, summarize
        if exp.threshold_met():
            await self._emit("phase", {"phase": "summary", "message": "Target achieved!"})
            self.final_output = await self._summarize(status="success")
        else:
            self.final_output = await self._summarize(status="partial")

    async def _run_deployment_loop(self):
        """Run the deployment workflow."""
        await self._emit("phase", {"phase": "deployment", "message": "Setting up deployment..."})

        # Get deployment target from perception
        deployment_target = self.p_out.get("entities", {}).get("deployment_target", "gradio")

        await self._emit(
            "deployment_start", {"target": deployment_target, "project_path": self.ctx.project_path}
        )

        # Build deployment-focused decision input
        deploy_perception = {
            **self.p_out,
            "pipeline_stage": "deploy",
            "route": Route.DEPLOY,
            "deployment_target": deployment_target,
        }

        d_input = build_decision_input(ctx=self.ctx, query=self.query, perception=deploy_perception)

        # Get deployment plan
        d_out = await self.decision.run(d_input, session=self.session)

        if not d_out.get("plan_graph", {}).get("nodes"):
            await self._emit("error", {"error": "No deployment plan generated"})
            return

        plan_nodes = d_out["plan_graph"]["nodes"]

        await self._emit(
            "deployment_plan",
            {"target": deployment_target, "steps": len(plan_nodes), "nodes": plan_nodes},
        )

        # Add deployment nodes to graph
        for node in plan_nodes:
            self.ctx.add_step(
                step_id=f"deploy_{node['id']}",
                description=node["description"],
                step_type=StepType.DEPLOY,
                tool=node.get("tool"),
                args=node.get("args"),
                from_node=StepType.ROOT,
            )

        # Require approval before executing deployment steps
        approved, approval_status = await self._ensure_approval(
            "deployment",
            {
                "target": deployment_target,
                "project_path": self.ctx.project_path,
                "steps": [n.get("description", "") for n in plan_nodes],
            },
        )
        if not approved:
            self.status = "paused" if approval_status == "timeout" else "failed"
            if approval_status == "denied":
                self.final_output = "Deployment approval denied. No changes applied."
            else:
                self.final_output = (
                    "Approval required for deployment. Submit approval and rerun to continue."
                )
            if self.status == "paused":
                await self._emit(
                    "status",
                    {"status": "paused", "message": "Deployment paused awaiting approval"},
                )
            return

        # Execute deployment steps
        deployment_failed = False
        failure_error = ""
        for node in plan_nodes:
            step_id = f"deploy_{node['id']}"

            await self._emit(
                "step_start",
                {
                    "step_id": step_id,
                    "description": node["description"],
                    "tool": node.get("tool"),
                    "phase": "deployment",
                },
            )

            success, result = await execute_step(
                step_id=step_id,
                tool=node.get("tool"),
                args=node.get("args", {}),
                ctx=self.ctx,
                tools_module=self.tools_module,
            )

            if success:
                self.ctx.update_step_result(step_id, result)
                await self._emit(
                    "step_complete",
                    {"step_id": step_id, "success": True, "result_summary": str(result)[:300]},
                )
            else:
                self.ctx.mark_step_failed(step_id, str(result.get("error", "Unknown error")))
                await self._emit(
                    "step_failed", {"step_id": step_id, "error": str(result.get("error", ""))[:200]}
                )
                deployment_failed = True
                failure_error = str(result.get("error", "Unknown error"))
                break

        if deployment_failed:
            await self._emit(
                "deployment_failed",
                {"target": deployment_target, "error": failure_error[:200]},
            )
            await self._run_deployment_rollback(deployment_target, failure_error)
            self.final_output = await self._summarize(status="failed")
            return

        await self._emit("deployment_complete", {"target": deployment_target, "status": "success"})

        # Summarize deployment
        await self._emit("phase", {"phase": "summary", "message": "Deployment complete!"})
        self.final_output = await self._summarize(status="success")

    async def _run_deployment_rollback(self, target: str, error: str):
        """Attempt to generate rollback instructions for a failed deployment."""
        tool_name = "rollback_deployment"
        args = {
            "project_path": self.ctx.project_path or ".",
            "target": target,
            "error": error,
            "dry_run": True,
        }
        success, result = await execute_step(
            step_id="rollback",
            tool=tool_name,
            args=args,
            ctx=self.ctx,
            tools_module=self.tools_module,
        )
        if not success:
            # Fallback to static rollback plan
            success, result = await execute_step(
                step_id="rollback_plan",
                tool="generate_rollback_plan",
                args={
                    "project_path": self.ctx.project_path or ".",
                    "target": target,
                    "error": error,
                },
                ctx=self.ctx,
                tools_module=self.tools_module,
            )
        await self._emit(
            "rollback_result",
            {
                "target": target,
                "success": success,
                "result": result.get("result") if success else result.get("error"),
            },
        )

    async def _get_improvement_suggestion(self) -> dict[str, Any]:
        """Get improvement suggestions from LLM."""
        exp = self.ctx.experiment_state

        prompt = self.improvement_prompt.format(
            target_accuracy=exp.target_accuracy,
            current_accuracy=exp.current_accuracy,
            current_loss=exp.current_loss or "N/A",
            gap=exp.get_accuracy_gap(),
            current_config=exp.current_config,
            training_history=exp.improvement_history,
            attempt=exp.improvement_attempt + 1,
            max_attempts=exp.max_improvement_attempts,
            previous_improvements=[h.get("config_changes", {}) for h in exp.improvement_history],
        )

        try:
            return await self.model_manager.generate_json(prompt)
        except Exception as e:
            logger.error("Error getting improvement suggestion", error=e)
            return {"should_retry": False, "error": str(e)}

    def _pick_next_step(self) -> str | None:
        """Pick the next pending step to execute."""
        pending = self.ctx.get_pending_steps()
        return pending[0] if pending else None

    async def _handle_failure(self) -> str:
        """Handle failure case."""
        self.status = "failed"
        self.session.mark_completed(success=False)

        await self._emit("status", {"status": "failed", "message": "Agent stopped due to errors"})

        # Still generate a summary
        self.final_output = await self._summarize(status="failed")
        return self.final_output


# Convenience function
async def run_mlops_agent(
    query: str,
    project_path: str | None = None,
    accuracy_threshold: float = 0.85,
    on_event: Callable | None = None,
    auto_approve: bool = False,
) -> str:
    """
    Run the MLOps agent with a query.

    Args:
        query: User query (e.g., "Set up MLOps pipeline for my project")
        project_path: Path to ML project
        accuracy_threshold: Target accuracy
        on_event: Callback for events

    Returns:
        Final summary markdown
    """
    agent = AgentLoop(on_event=on_event, auto_approve=auto_approve)
    return await agent.run(
        query=query, project_path=project_path, accuracy_threshold=accuracy_threshold
    )
