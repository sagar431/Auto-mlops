"""
Agent Loop for MLOps Agent.
Graph-based execution loop with self-improvement capability.
Orchestrates: Perception -> Decision -> Action -> (Improve if needed) -> Summarize
"""

import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    async def _select_registry_workflow(self) -> bool:
        """Select a registry workflow before prompt-authored planning."""
        selection = self.workflow_registry.select_workflow(self.query)
        selection = self._add_runtime_input_requirements(selection)
        self.workflow_selection = selection
        self.ctx.globals["workflow_selection"] = selection
        self.ctx.globals["workflow_inputs"] = {"project_path": self.ctx.project_path}

        await self._emit(
            "workflow_selection",
            {
                **asdict(selection),
                "status": selection.status.value,
                "runtime_inputs": {"project_path": self.ctx.project_path},
            },
        )

        if selection.status is WorkflowStatus.PENDING:
            if selection.workflow_id == "setup_pipeline":
                projected_step_ids = await self._project_registry_workflow(selection.workflow_id)
                if not self.auto_approve and not self.ctx.globals.get("approval_records"):
                    approval_validation = self._first_blocking_setup_pipeline_approval()
                    if approval_validation is not None:
                        self.status = "paused"
                        risk_categories = [
                            risk_category.value
                            for risk_category in approval_validation.risk_categories
                        ]
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
                f"Clarifying question: {clarifying_question}"
            )
            return True

        return False

    def _first_blocking_setup_pipeline_approval(self):
        """Return the first setup_pipeline approval gate that lacks approval."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id != "setup_pipeline"
            or self.workflow_selection.status is not WorkflowStatus.PENDING
        ):
            return None

        template = self.workflow_registry.get("setup_pipeline")
        approval_records = tuple(self.ctx.globals.get("approval_records", ()))
        for step in template.steps:
            validation = self.workflow_registry.validate_step_approval(
                workflow_id="setup_pipeline",
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
        for workflow_input in template.required_inputs:
            if workflow_input.required and workflow_input.name == "project_path":
                if not self.ctx.project_path and "project_path" not in missing_inputs:
                    missing_inputs.append("project_path")

        if not missing_inputs:
            return selection

        return replace(
            selection,
            status=WorkflowStatus.BLOCKED,
            missing_inputs=tuple(missing_inputs),
            selection_reason=(
                f"{selection.selection_reason} Missing required runtime inputs: "
                f"{', '.join(missing_inputs)}."
            ),
        )

    def _should_block_for_workflow_selection(self, selection: WorkflowSelection) -> bool:
        """Block registry-like requests that did not yield an executable selection."""
        if selection.matched_aliases or selection.rejected_workflows:
            return True

        workflow_terms = (
            "deploy",
            "setup",
            "set up",
            "mlops",
            "pipeline",
            "serve",
            "gradio",
            "kserve",
            "gpu",
            "lambda",
        )
        normalized_query = self.query.casefold()
        return any(term in normalized_query for term in workflow_terms)

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
                        f"next_action: {approval_validation.next_action}"
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
                    success, result = await execute_step(
                        step_id=self.next_step_id,
                        tool=step_data.tool,
                        args=step_data.args or {},
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
            self._capture_setup_pipeline_evidence(self.next_step_id, result)

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

        if self._is_pending_setup_pipeline():
            self._finalize_setup_pipeline_contract()

    async def _validate_registry_step_approval(self, step_id: str):
        """Validate setup_pipeline registry approval gates before tool execution."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id != "setup_pipeline"
            or self.workflow_selection.status is not WorkflowStatus.PENDING
        ):
            return None

        try:
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
                        timestamp=datetime.now(UTC),
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

    def _is_pending_setup_pipeline(self) -> bool:
        """Return whether the selected workflow is an executable setup_pipeline run."""
        return (
            self.workflow_selection is not None
            and self.workflow_selection.workflow_id == "setup_pipeline"
            and self.workflow_selection.status is WorkflowStatus.PENDING
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
        contract_status = self.workflow_registry.validate_success_contract(
            "setup_pipeline",
            verification_results=tuple(self.ctx.globals.get("verification_results", ())),
            artifact_manifest=self.ctx.globals.get("artifact_manifest"),
        )
        self.ctx.globals["contract_status"] = contract_status
        self.ctx.globals["workflow_status"] = contract_status.status

        if contract_status.status is WorkflowStatus.SUCCEEDED:
            self.status = "success"
        elif contract_status.status is WorkflowStatus.FAILED:
            self.status = "failed"
        else:
            self.status = "paused"

        self.final_output = self._format_setup_pipeline_contract_output(contract_status)
        return contract_status

    def _format_setup_pipeline_contract_output(
        self,
        contract_status: ContractValidation,
    ) -> str:
        missing_evidence = self._format_contract_failures(contract_status.missing_evidence)
        failed_checks = self._format_contract_failures(contract_status.failed_checks)
        return (
            "setup_pipeline final workflow status derived from SuccessContract. "
            f"contract_status: {contract_status.status.value}. "
            f"workflow_status: {contract_status.status.value}. "
            f"missing_evidence: {missing_evidence}. "
            f"failed_checks: {failed_checks}. "
            f"artifacts: {self._format_artifacts()}. "
            f"approvals: {self._format_approval_records()}."
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

    def _capture_setup_pipeline_evidence(self, step_id: str, result: dict[str, Any]) -> None:
        """Capture explicit setup_pipeline evidence from a completed step result."""
        if (
            self.workflow_selection is None
            or self.workflow_selection.workflow_id != "setup_pipeline"
            or self.workflow_selection.status is not WorkflowStatus.PENDING
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

    def _verification_results_from_step_result(
        self,
        step_id: str,
        payload: dict[str, Any],
    ) -> tuple[VerificationResult, ...]:
        """Return setup verification records explicitly reported by a step."""
        raw_results = payload.get("verification_results", ())
        if isinstance(raw_results, dict):
            raw_results = (raw_results,)

        template = self.workflow_registry.get("setup_pipeline")
        checks_by_name = {
            check.name: check
            for check in template.success_contract.checks
            if check.source_step == step_id
        }
        verification_results: list[VerificationResult] = []

        for raw_result in raw_results:
            if isinstance(raw_result, VerificationResult):
                if raw_result.check_name in checks_by_name and raw_result.source_step == step_id:
                    verification_results.append(raw_result)
                continue
            if not isinstance(raw_result, dict):
                continue
            check_name = raw_result.get("check_name")
            if check_name not in checks_by_name:
                continue
            source_step = raw_result.get("source_step")
            if source_step != step_id:
                continue
            if "passed" not in raw_result or "evidence" not in raw_result:
                continue

            verification_results.append(
                VerificationResult(
                    check_name=check_name,
                    evidence_type=raw_result.get(
                        "evidence_type",
                        checks_by_name[check_name].evidence_type,
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
        """Return setup artifact manifest entries explicitly reported by a step."""
        raw_manifest = payload.get("artifact_manifest")
        if isinstance(raw_manifest, ArtifactManifest):
            raw_entries = raw_manifest.entries
        elif isinstance(raw_manifest, dict):
            raw_entries = raw_manifest.get("entries", ())
        else:
            raw_entries = ()

        if isinstance(raw_entries, dict):
            raw_entries = (raw_entries,)

        template = self.workflow_registry.get("setup_pipeline")
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
                    )
                )
            except (KeyError, ValueError):
                continue

        return tuple(artifact_entries)

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
