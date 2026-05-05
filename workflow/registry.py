"""Code-owned workflow templates for supported Auto-MLOps workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class WorkflowStatus(str, Enum):
    """Runtime-owned workflow lifecycle states."""

    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class EvidenceType(str, Enum):
    """Supported provenance for success contract evidence."""

    DECLARED = "declared"
    OBSERVED = "observed"
    DECLARED_OR_OBSERVED = "declared_or_observed"


class ArtifactState(str, Enum):
    """Controlled lifecycle states for structured artifact manifest entries."""

    GENERATED = "generated"
    VALIDATED = "validated"
    SELECTED = "selected"
    EXTERNAL = "external"


class RiskCategory(str, Enum):
    """Controlled risk categories for approval-gated workflow steps."""

    WRITES_PROJECT_FILES = "writes_project_files"
    INSTALLS_PACKAGES = "installs_packages"
    STARTS_SERVER = "starts_server"
    BUILDS_IMAGE = "builds_image"
    PUSHES_REGISTRY = "pushes_registry"
    USES_CLOUD_CREDENTIALS = "uses_cloud_credentials"
    USES_REMOTE_SERVICE_CREDENTIALS = "uses_remote_service_credentials"
    EXECUTES_PROJECT_CODE = "executes_project_code"
    USES_GPU = "uses_gpu"
    EXPOSES_PORT = "exposes_port"


class ApprovalStatus(str, Enum):
    """Controlled states for human approval records."""

    APPROVED = "approved"
    DENIED = "denied"


@dataclass(frozen=True)
class RollbackPlan:
    """Structured rollback readiness evidence; execution is intentionally out of scope."""

    command: str | None = None
    script_path: str | None = None
    documented_target: str | None = None

    def __post_init__(self) -> None:
        if not any((self.command, self.script_path, self.documented_target)):
            raise ValueError(
                "Rollback plan requires a command, script path, or documented target"
            )


@dataclass(frozen=True)
class DeploymentCheckResult:
    """Structured observed evidence for a deployment probe."""

    passed: bool
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, dict):
            raise ValueError("Deployment check evidence must be structured data")


@dataclass(frozen=True)
class LatencySummary:
    """Structured latency evidence captured during deployment verification."""

    p50_ms: float
    p95_ms: float
    sample_count: int

    def __post_init__(self) -> None:
        if self.p50_ms < 0 or self.p95_ms < 0 or self.sample_count < 1:
            raise ValueError("Latency summary requires non-negative latency and samples")


@dataclass(frozen=True)
class GpuEvidence:
    """Structured GPU evidence captured for deployment reports."""

    available: bool
    device_name: str | None = None
    cuda_version: str | None = None
    utilization_percent: float | None = None


@dataclass(frozen=True)
class DeploymentReport:
    """Structured deployment outcome report backed by contract validation."""

    workflow_id: str
    target: str
    selected_backend: str
    endpoint_url: str
    server_start_command: str
    health_result: DeploymentCheckResult
    prediction_result: DeploymentCheckResult
    latency_summary: LatencySummary
    gpu_evidence: GpuEvidence
    artifacts: ArtifactManifest
    approvals: tuple[ApprovalRecord, ...]
    rollback_plan: RollbackPlan | None
    contract_status: ContractValidation

    def __post_init__(self) -> None:
        required_text_fields = (
            self.workflow_id,
            self.target,
            self.selected_backend,
            self.endpoint_url,
            self.server_start_command,
        )
        if any(not value for value in required_text_fields):
            raise ValueError("Deployment report requires target, backend, endpoint, and command")
        if self.contract_status.status is WorkflowStatus.SUCCEEDED and self.rollback_plan is None:
            raise ValueError("Successful deployment report requires rollback readiness")


@dataclass(frozen=True)
class WorkflowSelection:
    """A structured routing decision for a workflow template."""

    workflow_id: str | None
    status: WorkflowStatus
    confidence: float
    matched_aliases: tuple[str, ...]
    matched_branches: tuple[str, ...]
    rejected_workflows: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    selection_reason: str


@dataclass(frozen=True)
class WorkflowInput:
    """A value declared by a workflow before step arguments are expanded."""

    name: str
    description: str
    required: bool = True
    default: Any | None = None
    allowed_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class WorkflowStep:
    """A registry-owned workflow action, separate from executable tool functions."""

    step_id: str
    name: str
    description: str
    order: int
    tool_functions: tuple[str, ...] = ()
    default_args: dict[str, Any] = field(default_factory=dict)
    post_step_perception: bool = False


@dataclass(frozen=True)
class SuccessContractCheck:
    """A named check that must be satisfied before workflow success."""

    name: str
    evidence_type: EvidenceType
    source_step: str
    condition: str | None = None
    unsatisfied_status: WorkflowStatus = WorkflowStatus.FAILED

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_type", EvidenceType(self.evidence_type))
        object.__setattr__(
            self,
            "unsatisfied_status",
            WorkflowStatus(self.unsatisfied_status),
        )


@dataclass(frozen=True)
class SuccessContract:
    """The structured completion contract for a workflow template."""

    checks: tuple[SuccessContractCheck, ...]


@dataclass(frozen=True)
class ContractFailure:
    """Structured reason a success contract check did not pass."""

    check_name: str
    expected_evidence_type: EvidenceType
    source_step: str
    actual_evidence: tuple[VerificationResult | ArtifactManifestEntry | RollbackPlan, ...]
    next_action: str


@dataclass(frozen=True)
class ContractValidation:
    """Runtime-owned validation result for a workflow success contract."""

    workflow_id: str
    status: WorkflowStatus
    missing_evidence: tuple[ContractFailure, ...]
    failed_checks: tuple[ContractFailure, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", WorkflowStatus(self.status))


@dataclass(frozen=True)
class VerificationResult:
    """Evidence produced by runtime verification for a success contract check."""

    check_name: str
    evidence_type: EvidenceType
    source_step: str
    passed: bool
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_type", EvidenceType(self.evidence_type))


@dataclass(frozen=True)
class ArtifactManifestEntry:
    """A structured record for one artifact produced or selected by a workflow."""

    artifact_type: str
    producing_step: str
    state: ArtifactState
    path: str | None = None
    uri: str | None = None
    checksum: str | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.path and not self.uri:
            raise ValueError("Artifact manifest entry requires a path or uri")
        try:
            object.__setattr__(self, "state", ArtifactState(self.state))
        except ValueError as exc:
            raise ValueError(f"Unknown artifact state: {self.state}") from exc


@dataclass(frozen=True)
class ArtifactManifest:
    """Structured artifact records reported by workflow execution."""

    entries: tuple[ArtifactManifestEntry, ...]


@dataclass(frozen=True)
class ArtifactRequirement:
    """An artifact a workflow is expected to produce, validate, or report."""

    name: str
    artifact_type: str
    source_step: str
    state: ArtifactState = ArtifactState.GENERATED
    contract_check_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ArtifactState(self.state))


@dataclass(frozen=True)
class WorkflowBranch:
    """A registry-owned alternative path within a workflow template."""

    name: str
    selection_rule: str


@dataclass(frozen=True)
class ApprovalRecord:
    """Auditable approval decision for one workflow run step."""

    workflow_run_id: str
    step_id: str
    risk_categories: tuple[RiskCategory, ...]
    status: ApprovalStatus
    approver: str | None
    timestamp: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "risk_categories",
            _normalize_risk_categories(self.risk_categories),
        )
        object.__setattr__(self, "status", ApprovalStatus(self.status))


@dataclass(frozen=True)
class ApprovalGate:
    """Human approval metadata required before a risky workflow step may run."""

    step_id: str
    risk_categories: tuple[RiskCategory, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "risk_categories",
            _normalize_risk_categories(self.risk_categories),
        )


@dataclass(frozen=True)
class StepApprovalValidation:
    """Registry-owned approval validation for one workflow step."""

    workflow_id: str
    workflow_run_id: str
    step_id: str
    status: WorkflowStatus
    risk_categories: tuple[RiskCategory, ...]
    approval_record: ApprovalRecord | None
    next_action: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", WorkflowStatus(self.status))


@dataclass(frozen=True)
class WorkflowTemplate:
    """An ordered skeleton for one supported workflow."""

    workflow_id: str
    name: str
    description: str
    required_inputs: tuple[WorkflowInput, ...]
    steps: tuple[WorkflowStep, ...]
    success_contract: SuccessContract
    artifact_requirements: tuple[ArtifactRequirement, ...] = ()
    branches: tuple[WorkflowBranch, ...] = ()
    routing_aliases: tuple[str, ...] = ()
    negative_routing_rules: tuple[str, ...] = ()
    approval_gates: tuple[ApprovalGate, ...] = ()

    def step_by_id(self, step_id: str) -> WorkflowStep:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"Unknown workflow step: {step_id}")


class WorkflowRegistry:
    """Lookup API for code-owned workflow templates."""

    def __init__(self, templates: list[WorkflowTemplate] | tuple[WorkflowTemplate, ...]):
        self._templates = {template.workflow_id: template for template in templates}
        self._validate_templates()

    @property
    def workflow_ids(self) -> tuple[str, ...]:
        return tuple(self._templates)

    def get(self, workflow_id: str) -> WorkflowTemplate:
        try:
            return self._templates[workflow_id]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow template: {workflow_id}") from exc

    def select_workflow(self, request: str) -> WorkflowSelection:
        """Select a workflow from registry-owned routing aliases."""

        normalized_request = _normalize_for_routing(request)
        matches: list[tuple[str, str]] = []
        rejected_workflows: list[str] = []

        for template in self._templates.values():
            matched_aliases = tuple(
                alias
                for alias in template.routing_aliases
                if _routing_phrase_matches(normalized_request, alias)
            )
            negative_rule_matched = any(
                _routing_phrase_matches(normalized_request, negative_rule)
                for negative_rule in template.negative_routing_rules
            )
            if negative_rule_matched and (
                matched_aliases or template.workflow_id != "prepare_capstone_container_ci"
            ):
                rejected_workflows.append(template.workflow_id)
                continue

            for alias in matched_aliases:
                matches.append((template.workflow_id, alias))

        if not matches:
            return WorkflowSelection(
                workflow_id=None,
                status=WorkflowStatus.BLOCKED,
                confidence=0.0,
                matched_aliases=(),
                matched_branches=(),
                rejected_workflows=tuple(rejected_workflows),
                missing_inputs=("workflow_intent",),
                selection_reason="No registry routing alias matched the request.",
            )

        matched_workflow_ids = tuple(dict.fromkeys(workflow_id for workflow_id, _ in matches))
        if len(matched_workflow_ids) > 1:
            return WorkflowSelection(
                workflow_id=None,
                status=WorkflowStatus.BLOCKED,
                confidence=0.4,
                matched_aliases=tuple(alias for _, alias in matches),
                matched_branches=(),
                rejected_workflows=matched_workflow_ids,
                missing_inputs=("workflow_intent",),
                selection_reason="Multiple registry routing aliases matched; clarify workflow intent.",
            )

        workflow_id, matched_alias = matches[0]
        matched_branches = _matched_branches(normalized_request, self.get(workflow_id))
        branch_reason = (
            f" Matched branch evidence: {', '.join(matched_branches)}."
            if matched_branches
            else ""
        )
        return WorkflowSelection(
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            confidence=0.9,
            matched_aliases=(matched_alias,),
            matched_branches=matched_branches,
            rejected_workflows=tuple(rejected_workflows),
            missing_inputs=(),
            selection_reason=(
                f"Matched routing alias '{matched_alias}' for workflow '{workflow_id}'."
                f"{branch_reason}"
            ),
        )

    def validate_success_contract(
        self,
        workflow_id: str,
        verification_results: tuple[VerificationResult, ...],
        artifact_manifest: ArtifactManifest | None = None,
        rollback_plan: RollbackPlan | None = None,
        workflow_inputs: dict[str, Any] | None = None,
    ) -> ContractValidation:
        """Derive workflow status from required success contract verification results."""

        template = self.get(workflow_id)
        if artifact_manifest is not None:
            self.validate_artifact_manifest(workflow_id, artifact_manifest)
        missing_evidence: list[ContractFailure] = []
        failed_checks: list[ContractFailure] = []
        for check in template.success_contract.checks:
            if not _contract_check_condition_applies(check, workflow_inputs or {}):
                continue
            if rollback_plan is not None and _is_rollback_readiness_check(check):
                continue
            actual_evidence = tuple(
                result for result in verification_results if result.check_name == check.name
            )
            satisfying_evidence = tuple(
                result for result in actual_evidence if _evidence_satisfies_check(result, check)
            )
            if any(result.passed for result in satisfying_evidence):
                continue
            artifact_requirements = _artifact_requirements_for_check(template, check.name)
            if artifact_requirements:
                if all(
                    _artifact_requirement_satisfied(artifact_manifest, requirement)
                    for requirement in artifact_requirements
                ):
                    continue
                missing_artifacts = tuple(
                    requirement.name
                    for requirement in artifact_requirements
                    if not _artifact_requirement_satisfied(artifact_manifest, requirement)
                )
                missing_evidence.append(
                    ContractFailure(
                        check_name=check.name,
                        expected_evidence_type=check.evidence_type,
                        source_step=check.source_step,
                        actual_evidence=(
                            artifact_manifest.entries if artifact_manifest is not None else ()
                        ),
                        next_action=(
                            f"Record required artifact manifest entries for check '{check.name}': "
                            f"{', '.join(missing_artifacts)}."
                        ),
                    )
                )
                continue
            if satisfying_evidence:
                unsatisfied_label = (
                    "blocked"
                    if check.unsatisfied_status is WorkflowStatus.BLOCKED
                    else "failed"
                )
                failure = ContractFailure(
                    check_name=check.name,
                    expected_evidence_type=check.evidence_type,
                    source_step=check.source_step,
                    actual_evidence=satisfying_evidence,
                    next_action=(
                        f"Resolve {unsatisfied_label} verification for check '{check.name}' "
                        f"from step '{check.source_step}'."
                    ),
                )
                if check.unsatisfied_status is WorkflowStatus.BLOCKED:
                    missing_evidence.append(failure)
                else:
                    failed_checks.append(failure)
                continue
            missing_evidence.append(
                ContractFailure(
                    check_name=check.name,
                    expected_evidence_type=check.evidence_type,
                    source_step=check.source_step,
                    actual_evidence=actual_evidence,
                    next_action=(
                        f"Record {check.evidence_type.value} evidence from step '{check.source_step}' "
                        f"for check '{check.name}'."
                    ),
                )
            )
        status = WorkflowStatus.SUCCEEDED
        if failed_checks:
            status = WorkflowStatus.FAILED
        elif missing_evidence:
            status = WorkflowStatus.BLOCKED
        return ContractValidation(
            workflow_id=workflow_id,
            status=status,
            missing_evidence=tuple(missing_evidence),
            failed_checks=tuple(failed_checks),
        )

    def validate_artifact_manifest(
        self,
        workflow_id: str,
        artifact_manifest: ArtifactManifest,
    ) -> None:
        """Validate artifact manifest entries against registry-owned workflow steps."""

        template = self.get(workflow_id)
        step_ids = {step.step_id for step in template.steps}
        for entry in artifact_manifest.entries:
            if entry.producing_step not in step_ids:
                raise ValueError(
                    f"Unknown artifact producing step '{entry.producing_step}' "
                    f"for workflow '{workflow_id}'"
                )

    def validate_step_approval(
        self,
        workflow_id: str,
        workflow_run_id: str,
        step_id: str,
        approval_records: tuple[ApprovalRecord, ...],
    ) -> StepApprovalValidation:
        """Derive whether a workflow step is blocked by registry approval gates."""

        template = self.get(workflow_id)
        template.step_by_id(step_id)
        approval_gate = _approval_gate_for_step(template, step_id)
        if approval_gate is None:
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=workflow_run_id,
                step_id=step_id,
                status=WorkflowStatus.PENDING,
                risk_categories=(),
                approval_record=None,
                next_action="No approval required for this workflow step.",
            )

        approval_record = next(
            (
                record
                for record in approval_records
                if _approval_record_matches_gate(record, workflow_run_id, approval_gate)
            ),
            None,
        )
        if approval_record is not None and approval_record.status is ApprovalStatus.APPROVED:
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=workflow_run_id,
                step_id=step_id,
                status=WorkflowStatus.PENDING,
                risk_categories=approval_gate.risk_categories,
                approval_record=approval_record,
                next_action="Approval satisfied; step may run.",
            )
        if approval_record is not None and approval_record.status is ApprovalStatus.DENIED:
            approver = approval_record.approver or "unknown approver"
            return StepApprovalValidation(
                workflow_id=workflow_id,
                workflow_run_id=workflow_run_id,
                step_id=step_id,
                status=WorkflowStatus.FAILED,
                risk_categories=approval_gate.risk_categories,
                approval_record=approval_record,
                next_action=(
                    f"Approval denied by {approver}; step '{step_id}' must not run."
                ),
            )

        return StepApprovalValidation(
            workflow_id=workflow_id,
            workflow_run_id=workflow_run_id,
            step_id=step_id,
            status=WorkflowStatus.BLOCKED,
            risk_categories=approval_gate.risk_categories,
            approval_record=None,
            next_action=(
                f"Record approval for workflow run '{workflow_run_id}' "
                f"before step '{step_id}' may run."
            ),
        )

    def _validate_templates(self) -> None:
        for template in self._templates.values():
            if not template.steps:
                raise ValueError(f"Fake Template '{template.workflow_id}' has no workflow steps")
            if not template.success_contract.checks:
                raise ValueError(
                    f"Fake Template '{template.workflow_id}' has no success contract checks"
                )


def _normalize_for_routing(value: str) -> str:
    return " ".join(value.casefold().split())


def _routing_phrase_matches(normalized_request: str, phrase: str) -> bool:
    return _normalize_for_routing(phrase) in normalized_request


def _matched_branches(normalized_request: str, template: WorkflowTemplate) -> tuple[str, ...]:
    return tuple(
        branch.name
        for branch in template.branches
        if _routing_phrase_matches(normalized_request, branch.name)
    )


def _approval_gate_for_step(template: WorkflowTemplate, step_id: str) -> ApprovalGate | None:
    return next((gate for gate in template.approval_gates if gate.step_id == step_id), None)


def _normalize_risk_categories(
    risk_categories: tuple[RiskCategory | str, ...],
) -> tuple[RiskCategory, ...]:
    try:
        return tuple(RiskCategory(risk_category) for risk_category in risk_categories)
    except ValueError as exc:
        raise ValueError("Unknown risk category") from exc


def _approval_record_matches_gate(
    record: ApprovalRecord,
    workflow_run_id: str,
    approval_gate: ApprovalGate,
) -> bool:
    return (
        record.workflow_run_id == workflow_run_id
        and record.step_id == approval_gate.step_id
        and record.risk_categories == approval_gate.risk_categories
    )


def _evidence_satisfies_check(
    verification_result: VerificationResult,
    check: SuccessContractCheck,
) -> bool:
    if check.evidence_type == "declared_or_observed":
        return verification_result.evidence_type in {"declared", "observed"}
    return verification_result.evidence_type == check.evidence_type


def _artifact_requirements_for_check(
    template: WorkflowTemplate,
    check_name: str,
) -> tuple[ArtifactRequirement, ...]:
    return tuple(
        requirement
        for requirement in template.artifact_requirements
        if requirement.contract_check_name == check_name
    )


def _is_rollback_readiness_check(check: SuccessContractCheck) -> bool:
    return "rollback_plan" in check.name


def _contract_check_condition_applies(
    check: SuccessContractCheck,
    workflow_inputs: dict[str, Any],
) -> bool:
    if check.condition is None:
        return True
    condition_parts = tuple(part.strip() for part in check.condition.split(" and "))
    for condition_part in condition_parts:
        if "==" not in condition_part:
            continue
        input_name, expected_value = (part.strip() for part in condition_part.split("==", 1))
        actual_value = workflow_inputs.get(input_name)
        if expected_value == "true":
            expected: Any = True
        elif expected_value == "false":
            expected = False
        else:
            expected = expected_value
        if actual_value != expected:
            return False
    return True


def _artifact_requirement_satisfied(
    artifact_manifest: ArtifactManifest | None,
    requirement: ArtifactRequirement,
) -> bool:
    return any(
        _artifact_matches_requirement(entry, requirement)
        for entry in (() if artifact_manifest is None else artifact_manifest.entries)
    )


def _artifact_matches_requirement(
    entry: ArtifactManifestEntry,
    requirement: ArtifactRequirement,
) -> bool:
    return (
        entry.artifact_type == requirement.artifact_type
        and entry.producing_step == requirement.source_step
        and entry.state == requirement.state
    )


def get_workflow_registry() -> WorkflowRegistry:
    """Return the workflow registry."""

    return WorkflowRegistry(
        (
            _setup_pipeline_template(),
            _detect_training_project_template(),
            _train_and_track_template(),
            _build_capstone_pipeline_template(),
            _prepare_capstone_data_template(),
            _prepare_capstone_container_ci_template(),
            _deploy_litserve_preflight_template(),
            _deploy_litserve_gpu_template(),
            _deploy_gpu_inference_template(),
            _deploy_gradio_demo_template(),
            _deploy_kserve_production_template(),
        )
    )


def _prepare_capstone_container_ci_template() -> WorkflowTemplate:
    common_step = "prepare_capstone_container_ci_contract"
    return WorkflowTemplate(
        workflow_id="prepare_capstone_container_ci",
        name="Prepare Capstone Container CI",
        description=(
            "Declare the Phase 5 capstone container and CI automation workflow, its "
            "stable inputs, branch contracts, routing, deferred milestones, and "
            "approval boundaries without executing Docker, registry, CI, secret, or "
            "deployment behavior."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the project that should receive container/CI evidence.",
            ),
            WorkflowInput(
                name="completion_mode",
                description=(
                    "Whether to validate local container/CI readiness or capstone-complete "
                    "container/CI evidence."
                ),
                default="container_local_ready",
                allowed_values=("container_local_ready", "container_capstone_complete"),
            ),
            WorkflowInput(
                name="data_stage_evidence_path",
                description="Optional Phase 4 data-stage evidence artifact path.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="local_model_artifact_path",
                description="Optional local model artifact for container-local readiness.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="mlflow_run_id",
                description="Optional MLflow run id for later upstream evidence resolution.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="mlflow_best_artifact_path",
                description="Optional MLflow-linked best artifact path.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="registry_target",
                description="Optional registry target for later validation.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="image_name",
                description="Optional container image name for later build evidence.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="image_tag",
                description="Optional container image tag for later build evidence.",
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="ci_workflow_path",
                description="Optional capstone CI workflow path for later validation.",
                required=False,
                default=None,
            ),
        ),
        steps=(
            WorkflowStep(
                step_id=common_step,
                name="Prepare Capstone Container CI Contract",
                description=(
                    "Validate Phase 5 Issue 1 inputs and report structured deferred "
                    "container/CI evidence without mutating project files or invoking "
                    "Docker, registry, CI, secret, training, deployment, or Kubernetes tools."
                ),
                order=1,
                tool_functions=("prepare_capstone_container_ci_contract",),
            ),
            WorkflowStep(
                step_id="resolve_upstream_container_evidence",
                name="Resolve Upstream Container Evidence",
                description="Resolve data-stage, training, MLflow, and model artifact evidence.",
                order=2,
                tool_functions=("resolve_capstone_container_upstream_evidence",),
            ),
            WorkflowStep(
                step_id="generate_validate_runtime_image_spec",
                name="Generate Or Validate Runtime Image Spec",
                description="Generate or validate the default Capstone Runtime Image build spec.",
                order=3,
                tool_functions=("generate_validate_capstone_runtime_image_spec",),
            ),
            WorkflowStep(
                step_id="build_smoke_check_container_image",
                name="Build And Smoke Check Container Image",
                description="Build the runtime image and run bounded smoke checks when approved.",
                order=4,
                tool_functions=("build_smoke_check_capstone_container_image",),
            ),
            WorkflowStep(
                step_id="configure_validate_registry_target",
                name="Configure And Validate Registry Target",
                description="Configure or validate the selected container registry target.",
                order=5,
                tool_functions=("configure_validate_capstone_registry_target",),
            ),
            WorkflowStep(
                step_id="approval_gated_registry_login_push",
                name="Approval-Gated Registry Login Push",
                description="Login and push only after explicit approval and safe credential use.",
                order=6,
            ),
            WorkflowStep(
                step_id="record_container_ci_evidence_handoff",
                name="Record Container CI Evidence Handoff",
                description="Write durable container CI evidence for orchestrator handoff.",
                order=7,
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="upstream_evidence_resolved",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="container_build_spec_reported",
                    evidence_type="declared_or_observed",
                    source_step="generate_validate_runtime_image_spec",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="dependency_context_reported",
                    evidence_type="observed",
                    source_step="generate_validate_runtime_image_spec",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="container_ci_evidence_artifact_reported",
                    evidence_type="observed",
                    source_step="record_container_ci_evidence_handoff",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="container_artifact_manifest_reported",
                    evidence_type="declared_or_observed",
                    source_step="generate_validate_runtime_image_spec",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="capstone_ci_workflow_reported",
                    evidence_type="declared_or_observed",
                    source_step="record_container_ci_evidence_handoff",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="capstone_ci_workflow_validated",
                    evidence_type="observed",
                    source_step="record_container_ci_evidence_handoff",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="secret_safety_validated",
                    evidence_type="observed",
                    source_step="generate_validate_runtime_image_spec",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="local_model_artifact_resolved",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    condition=(
                        "completion_mode == container_local_ready and "
                        "mlflow_best_artifact_available == false"
                    ),
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="mlflow_best_artifact_resolved",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    condition=(
                        "completion_mode == container_local_ready and "
                        "local_model_artifact_available == false"
                    ),
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="docker_availability_reported",
                    evidence_type="observed",
                    source_step="build_smoke_check_container_image",
                    condition="completion_mode == container_local_ready",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="image_build_attempt_reported",
                    evidence_type="observed",
                    source_step="build_smoke_check_container_image",
                    condition="completion_mode == container_local_ready and docker_available == true",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="image_build_deferred_reported",
                    evidence_type="declared_or_observed",
                    source_step="build_smoke_check_container_image",
                    condition="completion_mode == container_local_ready and docker_available == false",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="data_stage_capstone_complete_verified",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="mlflow_best_artifact_verified",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="training_lineage_verified",
                    evidence_type="observed",
                    source_step="resolve_upstream_container_evidence",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="docker_available",
                    evidence_type="observed",
                    source_step="build_smoke_check_container_image",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="image_build_succeeded",
                    evidence_type="observed",
                    source_step="build_smoke_check_container_image",
                    condition="docker_available == true",
                    unsatisfied_status="failed",
                ),
                SuccessContractCheck(
                    name="container_smoke_check_passed",
                    evidence_type="observed",
                    source_step="build_smoke_check_container_image",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="failed",
                ),
                SuccessContractCheck(
                    name="registry_target_validated",
                    evidence_type="observed",
                    source_step="configure_validate_registry_target",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="registry_auth_capability_verified",
                    evidence_type="observed",
                    source_step="configure_validate_registry_target",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="registry_push_approved",
                    evidence_type="observed",
                    source_step="approval_gated_registry_login_push",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="registry_push_succeeded",
                    evidence_type="observed",
                    source_step="approval_gated_registry_login_push",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="pushed_image_reference_reported",
                    evidence_type="observed",
                    source_step="approval_gated_registry_login_push",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="capstone_ci_registry_usage_validated",
                    evidence_type="observed",
                    source_step="record_container_ci_evidence_handoff",
                    condition="completion_mode == container_capstone_complete",
                    unsatisfied_status="blocked",
                ),
            ),
        ),
        branches=(
            WorkflowBranch(
                name="container_local_ready",
                selection_rule=(
                    "Requires local model artifact or MLflow best artifact evidence, "
                    "Docker availability reporting, deferred or attempted image build "
                    "evidence, and common container/CI checks."
                ),
            ),
            WorkflowBranch(
                name="container_capstone_complete",
                selection_rule=(
                    "Requires capstone-complete data, MLflow best artifact, training "
                    "lineage, image build, smoke check, registry, push, and CI registry "
                    "usage evidence."
                ),
            ),
        ),
        routing_aliases=(
            "prepare capstone container CI",
            "create capstone Docker and CI evidence",
            "package capstone runtime image",
            "prepare container_ci_evidence",
        ),
        negative_routing_rules=(
            "Kubernetes",
            "KServe",
            "Helm",
            "ArgoCD",
            "GitOps",
            "EKS",
            "endpoint deployment",
            "frontend",
            "final report",
            "stress-test",
            "stress test",
            "video",
        ),
        approval_gates=(
            ApprovalGate(
                step_id="generate_validate_runtime_image_spec",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="build_smoke_check_container_image",
                risk_categories=("builds_image", "executes_project_code"),
            ),
            ApprovalGate(
                step_id="approval_gated_registry_login_push",
                risk_categories=("uses_remote_service_credentials", "pushes_registry"),
            ),
            ApprovalGate(
                step_id="record_container_ci_evidence_handoff",
                risk_categories=("writes_project_files",),
            ),
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="capstone_runtime_image_build_spec",
                artifact_type="container_build_spec",
                source_step="generate_validate_runtime_image_spec",
                state="validated",
                contract_check_name="container_artifact_manifest_reported",
            ),
        ),
    )


def _prepare_capstone_data_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="prepare_capstone_data",
        name="Prepare Capstone Data",
        description=(
            "Declare the Phase 4 capstone data automation workflow for two "
            "user-provided datasets with approval-gated data package, DVC remote, "
            "transfer, and durable evidence handoff checks."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the project that should receive capstone data evidence.",
            ),
            WorkflowInput(
                name="dataset_1_path",
                description="First user-provided local or mounted capstone dataset path.",
            ),
            WorkflowInput(
                name="dataset_2_path",
                description="Second user-provided local or mounted capstone dataset path.",
            ),
            WorkflowInput(
                name="completion_mode",
                description=(
                    "Whether to validate local data readiness or capstone-complete "
                    "S3 data evidence."
                ),
                default="local_ready",
                allowed_values=("local_ready", "capstone_complete"),
            ),
            WorkflowInput(
                name="test_size",
                description="Declared deterministic split ratio for generated manifests.",
                required=False,
                default=0.2,
            ),
            WorkflowInput(
                name="split_seed",
                description="Declared deterministic split seed for generated manifests.",
                required=False,
                default=42,
            ),
            WorkflowInput(
                name="materialize_splits",
                description=(
                    "Whether physical train/test folders are requested. Defaults to false; "
                    "folder writes require a separate future approval-gated step."
                ),
                required=False,
                default=False,
            ),
            WorkflowInput(
                name="dvc_remote_name",
                description="DVC remote name used for capstone data package validation.",
                required=False,
                default="capstone",
            ),
            WorkflowInput(
                name="dvc_remote_url",
                description=(
                    "Optional local or s3:// DVC remote URL. Configuration writes require "
                    "approval and S3 validation requires cloud credential approval."
                ),
                required=False,
                default=None,
            ),
            WorkflowInput(
                name="dvc_transfer_direction",
                description=(
                    "Approved DVC transfer direction for capstone-complete evidence. "
                    "Local-ready runs defer transfers."
                ),
                required=False,
                default="push",
                allowed_values=("push", "pull", "none"),
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="prepare_capstone_data_contract",
                name="Prepare Capstone Data Contract",
                description=(
                    "Inspect two user-provided capstone dataset paths for supported "
                    "canonical image-folder layouts without mutating data or DVC state."
                ),
                order=1,
                tool_functions=("detect_capstone_data_layouts",),
            ),
            WorkflowStep(
                step_id="generate_split_manifests",
                name="Generate Split Manifests",
                description=(
                    "Record existing split evidence or write deterministic split manifests "
                    "under data/capstone/<dataset_id>/ without mutating source datasets."
                ),
                order=2,
                tool_functions=("generate_capstone_split_manifests",),
            ),
            WorkflowStep(
                step_id="track_capstone_data_package",
                name="Track Capstone Data Package",
                description=(
                    "Validate or initialize local DVC metadata and DVC-track generated "
                    "capstone data package paths without configuring remotes or transfers."
                ),
                order=3,
                tool_functions=("track_capstone_data_package",),
            ),
            WorkflowStep(
                step_id="configure_validate_dvc_remote",
                name="Configure And Validate DVC Remote",
                description=(
                    "Configure or validate a local or S3 DVC remote for the capstone data "
                    "package without pushing or pulling data."
                ),
                order=4,
                tool_functions=("configure_validate_capstone_dvc_remote",),
            ),
            WorkflowStep(
                step_id="push_capstone_data",
                name="Push Capstone Data",
                description=(
                    "Run approval-gated dvc push for the capstone data package and "
                    "record observed S3 transfer evidence."
                ),
                order=5,
                tool_functions=("push_capstone_data",),
            ),
            WorkflowStep(
                step_id="pull_capstone_data",
                name="Pull Capstone Data",
                description=(
                    "Run approval-gated dvc pull for the capstone data package and "
                    "record observed S3 transfer evidence."
                ),
                order=6,
                tool_functions=("pull_capstone_data",),
            ),
            WorkflowStep(
                step_id="record_data_stage_evidence",
                name="Record Data Stage Evidence",
                description=(
                    "Write the durable data-stage evidence artifact for downstream "
                    "capstone orchestration handoff."
                ),
                order=7,
                tool_functions=("record_capstone_data_stage_evidence",),
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="two_dataset_paths_provided",
                    evidence_type="observed",
                    source_step="prepare_capstone_data_contract",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="two_dataset_layouts_supported",
                    evidence_type="observed",
                    source_step="prepare_capstone_data_contract",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="split_evidence_recorded",
                    evidence_type="observed",
                    source_step="generate_split_manifests",
                ),
                SuccessContractCheck(
                    name="capstone_data_package_tracked",
                    evidence_type="observed",
                    source_step="track_capstone_data_package",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="dvc_repo_validated",
                    evidence_type="observed",
                    source_step="track_capstone_data_package",
                    unsatisfied_status="blocked",
                ),
                SuccessContractCheck(
                    name="data_stage_evidence_artifact_reported",
                    evidence_type="observed",
                    source_step="record_data_stage_evidence",
                ),
                SuccessContractCheck(
                    name="dataset_lineage_artifacts_reported",
                    evidence_type="declared_or_observed",
                    source_step="generate_split_manifests",
                ),
                SuccessContractCheck(
                    name="s3_remote_validated",
                    evidence_type="observed",
                    source_step="configure_validate_dvc_remote",
                    condition="completion_mode == capstone_complete",
                ),
                SuccessContractCheck(
                    name="s3_transfer_completed",
                    evidence_type="observed",
                    source_step="push_capstone_data",
                    condition="completion_mode == capstone_complete",
                ),
            ),
        ),
        branches=(
            WorkflowBranch(
                name="local_ready",
                selection_rule=(
                    "Requires local dataset paths, split evidence, DVC tracking, "
                    "and data-stage evidence artifact checks."
                ),
            ),
            WorkflowBranch(
                name="capstone_complete",
                selection_rule=(
                    "Requires all local_ready checks plus S3 remote validation and "
                    "S3 transfer evidence."
                ),
            ),
        ),
        routing_aliases=(
            "prepare capstone data",
            "set up capstone data",
            "setup capstone data",
            "version capstone datasets",
            "prepare datasets for capstone",
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="split_manifest",
                artifact_type="split_manifest",
                source_step="generate_split_manifests",
                state="generated",
            ),
            ArtifactRequirement(
                name="capstone_data_package",
                artifact_type="capstone_data_package",
                source_step="track_capstone_data_package",
                state="generated",
            ),
            ArtifactRequirement(
                name="dvc_tracking_file",
                artifact_type="dvc_tracking_file",
                source_step="track_capstone_data_package",
                state="generated",
            ),
            ArtifactRequirement(
                name="data_stage_evidence",
                artifact_type="data_stage_evidence",
                source_step="record_data_stage_evidence",
                state="generated",
                contract_check_name="data_stage_evidence_artifact_reported",
            ),
        ),
        approval_gates=(
            ApprovalGate(
                step_id="generate_split_manifests",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="track_capstone_data_package",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="configure_validate_dvc_remote",
                risk_categories=("writes_project_files", "uses_cloud_credentials"),
            ),
            ApprovalGate(
                step_id="push_capstone_data",
                risk_categories=("uses_cloud_credentials",),
            ),
            ApprovalGate(
                step_id="pull_capstone_data",
                risk_categories=("uses_cloud_credentials", "writes_project_files"),
            ),
        ),
    )


def _build_capstone_pipeline_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="build_capstone_pipeline",
        name="Capstone Orchestrator",
        description=(
            "Record the first top-level capstone pipeline skeleton with setup, data, train, "
            "deploy, monitor, and report stages, while keeping later-phase capabilities blocked "
            "or deferred until implemented workflows can satisfy their contracts."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the project that should receive the capstone plan.",
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="record_capstone_orchestrator_skeleton",
                name="Record Capstone Orchestrator Skeleton",
                description=(
                    "Write structured stage, sub-workflow, blocked, deferred, artifact, and "
                    "next-action evidence without executing unimplemented future capabilities."
                ),
                order=1,
                tool_functions=("record_capstone_orchestrator_skeleton",),
                default_args={
                    "declared_stages": (
                        "setup",
                        "data",
                        "train",
                        "deploy",
                        "monitor",
                        "report",
                    ),
                    "implemented_subworkflows": (
                        "setup_pipeline",
                        "prepare_capstone_data",
                        "detect_training_project",
                        "train_and_track",
                        "deploy_litserve_preflight",
                        "deploy_litserve_gpu",
                    ),
                    "blocked_subworkflows": ("train_until_better",),
                },
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="capstone_stage_plan_recorded",
                    evidence_type="declared",
                    source_step="record_capstone_orchestrator_skeleton",
                ),
                SuccessContractCheck(
                    name="implemented_subworkflows_referenced",
                    evidence_type="declared",
                    source_step="record_capstone_orchestrator_skeleton",
                ),
                SuccessContractCheck(
                    name="deferred_capabilities_recorded",
                    evidence_type="declared",
                    source_step="record_capstone_orchestrator_skeleton",
                ),
                SuccessContractCheck(
                    name="capstone_orchestrator_artifact_reported",
                    evidence_type="declared",
                    source_step="record_capstone_orchestrator_skeleton",
                ),
                SuccessContractCheck(
                    name="capstone_pipeline_ready",
                    evidence_type="observed",
                    source_step="record_capstone_orchestrator_skeleton",
                ),
            ),
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="capstone_orchestrator_plan",
                artifact_type="capstone_orchestrator_plan",
                source_step="record_capstone_orchestrator_skeleton",
                state="generated",
                contract_check_name="capstone_orchestrator_artifact_reported",
            ),
        ),
        routing_aliases=(
            "Build full capstone pipeline",
            "build capstone pipeline",
            "Build the capstone pipeline",
            "capstone pipeline for this project",
            "Capstone Orchestrator",
            "create capstone orchestrator",
        ),
    )


def _train_and_track_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="train_and_track",
        name="Train And Track",
        description=(
            "Run a supported Hydra/PyTorch/TIMM training project with bounded controls "
            "and capture metrics, logs, duration, and artifacts."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the detected training project.",
            ),
            WorkflowInput(
                name="timeout_seconds",
                description="Maximum allowed wall-clock seconds for the training command.",
            ),
            WorkflowInput(
                name="max_epochs",
                description="Maximum epochs or equivalent Hydra override for the training run.",
            ),
            WorkflowInput(
                name="device",
                description="Training device such as cpu or cuda.",
            ),
            WorkflowInput(
                name="data_subset",
                description="Dataset subset or size control for bounded training.",
            ),
            WorkflowInput(
                name="metric_name",
                description="Metric name used to compare latest run against baseline.",
            ),
            WorkflowInput(
                name="metric_direction",
                description="Whether the comparison metric should be maximized or minimized.",
            ),
            WorkflowInput(
                name="threshold",
                description="Minimum required improvement threshold for selecting the latest run.",
            ),
            WorkflowInput(
                name="tie_policy",
                description="Tie policy for exact threshold comparisons.",
            ),
            WorkflowInput(
                name="baseline_metric",
                description="Baseline metric value for the currently selected artifact.",
            ),
            WorkflowInput(
                name="baseline_artifact_path",
                description="Current baseline checkpoint or model artifact path.",
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="detect_training_project",
                name="Detect Training Project",
                description=(
                    "Inspect project files for supported training evidence before any "
                    "training command is allowed."
                ),
                order=1,
                tool_functions=("detect_training_project",),
            ),
            WorkflowStep(
                step_id="run_bounded_training",
                name="Run Bounded Training",
                description=(
                    "Execute the detected training entrypoint with explicit timeout, "
                    "epoch, device, subset, and Hydra override controls."
                ),
                order=2,
                tool_functions=("run_bounded_training",),
                default_args={"target_metric": "accuracy"},
            ),
            WorkflowStep(
                step_id="track_training_in_mlflow",
                name="Track Training In MLflow",
                description=(
                    "Log bounded training params, metrics, logs, and checkpoint artifacts "
                    "to a verified local MLflow run."
                ),
                order=3,
                tool_functions=("track_training_in_mlflow",),
                default_args={"experiment_name": "mlops-training"},
            ),
            WorkflowStep(
                step_id="select_best_model_artifact",
                name="Select Best Model Artifact",
                description=(
                    "Compare the verified latest MLflow run against a baseline and select "
                    "the deployable checkpoint/model artifact deterministically."
                ),
                order=4,
                tool_functions=("select_best_model_artifact",),
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="training_project_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="training_entrypoint_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="hydra_config_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="bounded_training_controls_present",
                    evidence_type="observed",
                    source_step="run_bounded_training",
                ),
                SuccessContractCheck(
                    name="bounded_training_command_completed",
                    evidence_type="observed",
                    source_step="run_bounded_training",
                ),
                SuccessContractCheck(
                    name="training_metric_captured",
                    evidence_type="observed",
                    source_step="run_bounded_training",
                ),
                SuccessContractCheck(
                    name="training_artifact_captured",
                    evidence_type="observed",
                    source_step="run_bounded_training",
                ),
                SuccessContractCheck(
                    name="training_run_evidence_captured",
                    evidence_type="observed",
                    source_step="run_bounded_training",
                ),
                SuccessContractCheck(
                    name="mlflow_experiment_exists",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_run_exists",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_tracking_uri_recorded",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_artifact_uri_recorded",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_params_logged",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_metrics_logged",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_artifacts_logged",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_checkpoint_artifact_logged",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="mlflow_run_status_recorded",
                    evidence_type="observed",
                    source_step="track_training_in_mlflow",
                ),
                SuccessContractCheck(
                    name="model_selection_inputs_present",
                    evidence_type="observed",
                    source_step="select_best_model_artifact",
                ),
                SuccessContractCheck(
                    name="model_selection_baseline_recorded",
                    evidence_type="observed",
                    source_step="select_best_model_artifact",
                ),
                SuccessContractCheck(
                    name="model_selection_metric_compared",
                    evidence_type="observed",
                    source_step="select_best_model_artifact",
                ),
                SuccessContractCheck(
                    name="model_selection_candidate_artifact_verified",
                    evidence_type="observed",
                    source_step="select_best_model_artifact",
                ),
                SuccessContractCheck(
                    name="model_artifact_selected",
                    evidence_type="observed",
                    source_step="select_best_model_artifact",
                ),
            ),
        ),
        routing_aliases=(
            "train and track",
            "train this project",
            "train this model",
            "run training",
        ),
    )


def _detect_training_project_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="detect_training_project",
        name="Detect Training Project",
        description=(
            "Detect a supported Hydra/PyTorch/TIMM training project and entrypoint "
            "without running training or tracking."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the training project to inspect.",
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="detect_training_project",
                name="Detect Training Project",
                description=(
                    "Inspect project files for supported framework, Hydra configs, DVC signals, "
                    "tests, and a training entrypoint without running training."
                ),
                order=1,
                tool_functions=("detect_training_project",),
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="training_project_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="training_entrypoint_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="hydra_config_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="dvc_or_data_evidence_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="pytorch_timm_signals_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="test_command_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
                SuccessContractCheck(
                    name="output_artifact_candidates_detected",
                    evidence_type="observed",
                    source_step="detect_training_project",
                ),
            ),
        ),
        routing_aliases=(
            "detect this training project",
            "detect training project",
        ),
    )


def _setup_pipeline_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="setup_pipeline",
        name="Setup Pipeline",
        description="Create a reproducible MLOps foundation for an ML project.",
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the ML project that should receive MLOps setup.",
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="analyze_project_structure",
                name="Analyze project structure",
                description="Inspect the project layout and identify framework and config files.",
                order=1,
                tool_functions=("analyze_project_config",),
            ),
            WorkflowStep(
                step_id="create_or_validate_hydra_config",
                name="Create or validate Hydra config",
                description="Create missing Hydra configuration or validate the existing config.",
                order=2,
                tool_functions=("create_hydra_config", "validate_hydra_config"),
            ),
            WorkflowStep(
                step_id="initialize_dvc",
                name="Initialize DVC",
                description="Initialize data versioning metadata for the project.",
                order=3,
                tool_functions=("init_dvc_repo",),
                default_args={"no_scm": True},
            ),
            WorkflowStep(
                step_id="configure_dvc_remote",
                name="Configure DVC remote",
                description="Configure a DVC remote only when one is requested.",
                order=4,
                tool_functions=("configure_dvc_remote",),
            ),
            WorkflowStep(
                step_id="add_data_to_dvc",
                name="Add data to DVC",
                description="Track the declared data path with DVC.",
                order=5,
                tool_functions=("add_data_to_dvc",),
                default_args={"data_path": "data"},
            ),
            WorkflowStep(
                step_id="create_dvc_yaml",
                name="Create dvc.yaml",
                description="Create the pipeline definition for reproducible training.",
                order=6,
                tool_functions=("create_dvc_pipeline",),
            ),
            WorkflowStep(
                step_id="initialize_mlflow_experiment",
                name="Initialize MLflow experiment",
                description="Create or validate the MLflow experiment used by the project.",
                order=7,
                tool_functions=("init_mlflow_experiment",),
                default_args={"experiment_name": "mlops"},
            ),
            WorkflowStep(
                step_id="create_dockerfile",
                name="Create Dockerfile",
                description="Create a Dockerfile for the ML project.",
                order=8,
                tool_functions=("create_ml_dockerfile",),
            ),
            WorkflowStep(
                step_id="create_ci_workflow",
                name="Create CI workflow",
                description="Create a CI workflow for pipeline validation.",
                order=9,
                tool_functions=("create_github_workflow",),
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="hydra_config_validates",
                    evidence_type="declared_or_observed",
                    source_step="create_or_validate_hydra_config",
                ),
                SuccessContractCheck(
                    name="dvc_repo_exists",
                    evidence_type="declared_or_observed",
                    source_step="initialize_dvc",
                ),
                SuccessContractCheck(
                    name="dvc_yaml_parseable",
                    evidence_type="declared_or_observed",
                    source_step="create_dvc_yaml",
                ),
                SuccessContractCheck(
                    name="mlflow_experiment_exists",
                    evidence_type="declared_or_observed",
                    source_step="initialize_mlflow_experiment",
                ),
                SuccessContractCheck(
                    name="dockerfile_build_evidence",
                    evidence_type="declared_or_observed",
                    source_step="create_dockerfile",
                ),
                SuccessContractCheck(
                    name="generated_files_reported",
                    evidence_type="declared",
                    source_step="create_ci_workflow",
                ),
            ),
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="hydra_config",
                artifact_type="configuration",
                source_step="create_or_validate_hydra_config",
                contract_check_name="generated_files_reported",
            ),
            ArtifactRequirement(
                name="dvc_yaml",
                artifact_type="pipeline_definition",
                source_step="create_dvc_yaml",
                contract_check_name="generated_files_reported",
            ),
            ArtifactRequirement(
                name="dockerfile",
                artifact_type="container_definition",
                source_step="create_dockerfile",
                contract_check_name="generated_files_reported",
            ),
            ArtifactRequirement(
                name="ci_workflow",
                artifact_type="automation_workflow",
                source_step="create_ci_workflow",
                contract_check_name="generated_files_reported",
            ),
        ),
        approval_gates=(
            ApprovalGate(
                step_id="create_or_validate_hydra_config",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="initialize_dvc",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="configure_dvc_remote",
                risk_categories=("writes_project_files", "uses_cloud_credentials"),
            ),
            ApprovalGate(
                step_id="add_data_to_dvc",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="create_dvc_yaml",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="initialize_mlflow_experiment",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="create_dockerfile",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="create_ci_workflow",
                risk_categories=("writes_project_files",),
            ),
        ),
        routing_aliases=("Set up MLOps", "setup MLOps pipeline", "create MLOps foundation"),
    )


def _deploy_litserve_preflight_template() -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id="deploy_litserve_preflight",
        name="Deploy LitServe Preflight",
        description=(
            "Prepare local LitServe deployment artifacts and record missing live evidence "
            "without starting a server or using GPU/cloud resources."
        ),
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the ML project or model project to prepare locally.",
            ),
        ),
        steps=(
            WorkflowStep(
                step_id="select_or_create_model_artifact",
                name="Select or create model artifact",
                description="Select an existing model artifact or create a local placeholder.",
                order=1,
                tool_functions=("select_or_create_model_artifact",),
            ),
            WorkflowStep(
                step_id="generate_or_validate_litserve_app",
                name="Generate or validate LitServe app",
                description="Generate or validate the local LitServe application artifact.",
                order=2,
                tool_functions=("create_litserve_api",),
                default_args={
                    "model_path": "models/model_preflight.pt",
                    "model_name": "model",
                    "source_step": "generate_or_validate_litserve_app",
                },
            ),
            WorkflowStep(
                step_id="generate_or_validate_dockerfile",
                name="Generate or validate Dockerfile",
                description="Generate or validate the local LitServe Dockerfile artifact.",
                order=3,
                tool_functions=("generate_litserve_dockerfile",),
            ),
            WorkflowStep(
                step_id="record_launch_command",
                name="Record launch command",
                description="Record the command that would launch the local LitServe server.",
                order=4,
                tool_functions=("record_litserve_launch_command",),
            ),
            WorkflowStep(
                step_id="record_missing_live_evidence",
                name="Record missing live evidence",
                description=(
                    "Record that GPU detection, server start, /health, /predict, and endpoint "
                    "URL evidence are intentionally missing from local preflight."
                ),
                order=5,
                tool_functions=("record_litserve_missing_live_evidence",),
            ),
            WorkflowStep(
                step_id="future_server_start",
                name="Future server start approval",
                description="Approval gate declaration for future server start and port exposure.",
                order=6,
            ),
            WorkflowStep(
                step_id="future_docker_build",
                name="Future Docker build approval",
                description="Approval gate declaration for a future Docker image build.",
                order=7,
            ),
            WorkflowStep(
                step_id="future_cloud_gpu_launch",
                name="Future cloud GPU approval",
                description="Approval gate declaration for future cloud credentials or GPU usage.",
                order=8,
            ),
        ),
        success_contract=SuccessContract(
            checks=(
                SuccessContractCheck(
                    name="model_artifact_selected",
                    evidence_type="declared_or_observed",
                    source_step="select_or_create_model_artifact",
                ),
                SuccessContractCheck(
                    name="litserve_app_artifact_ready",
                    evidence_type="declared_or_observed",
                    source_step="generate_or_validate_litserve_app",
                ),
                SuccessContractCheck(
                    name="dockerfile_artifact_ready",
                    evidence_type="declared_or_observed",
                    source_step="generate_or_validate_dockerfile",
                ),
                SuccessContractCheck(
                    name="launch_command_recorded",
                    evidence_type="declared",
                    source_step="record_launch_command",
                ),
                SuccessContractCheck(
                    name="missing_live_evidence_recorded",
                    evidence_type="declared",
                    source_step="record_missing_live_evidence",
                ),
            ),
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="selected_model",
                artifact_type="model_artifact",
                source_step="select_or_create_model_artifact",
                state="selected",
                contract_check_name="model_artifact_selected",
            ),
            ArtifactRequirement(
                name="litserve_app",
                artifact_type="serving_application",
                source_step="generate_or_validate_litserve_app",
                state="generated",
                contract_check_name="litserve_app_artifact_ready",
            ),
            ArtifactRequirement(
                name="litserve_dockerfile",
                artifact_type="container_definition",
                source_step="generate_or_validate_dockerfile",
                state="generated",
                contract_check_name="dockerfile_artifact_ready",
            ),
        ),
        routing_aliases=(
            "Prepare LitServe deployment locally",
            "LitServe deployment preflight",
            "local LitServe preflight",
            "validate LitServe locally",
        ),
        negative_routing_rules=(
            "Lambda Labs GPU",
            "Lambda GPU",
            "LitServe GPU deployment",
            "start LitServe server",
        ),
        approval_gates=(
            ApprovalGate(
                step_id="generate_or_validate_litserve_app",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="generate_or_validate_dockerfile",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="future_server_start",
                risk_categories=("starts_server", "exposes_port"),
            ),
            ApprovalGate(
                step_id="future_docker_build",
                risk_categories=("builds_image",),
            ),
            ApprovalGate(
                step_id="future_cloud_gpu_launch",
                risk_categories=("uses_cloud_credentials", "uses_gpu"),
            ),
        ),
    )


def _template(
    workflow_id: str,
    name: str,
    description: str,
    step_ids: tuple[str, ...],
    contract_check_names: tuple[str, ...],
) -> WorkflowTemplate:
    return WorkflowTemplate(
        workflow_id=workflow_id,
        name=name,
        description=description,
        required_inputs=(
            WorkflowInput(
                name="project_path",
                description="Path to the ML project or model project to deploy.",
            ),
        ),
        steps=tuple(
            WorkflowStep(
                step_id=step_id,
                name=step_id.replace("_", " ").title(),
                description=step_id.replace("_", " "),
                order=index,
            )
            for index, step_id in enumerate(step_ids, start=1)
        ),
        success_contract=SuccessContract(
            checks=tuple(
                SuccessContractCheck(
                    name=check_name,
                    evidence_type="declared_or_observed",
                    source_step=step_ids[min(index, len(step_ids) - 1)],
                )
                for index, check_name in enumerate(contract_check_names)
            )
        ),
    )


def _success_contract_with_source_overrides(
    template: WorkflowTemplate,
    source_steps: dict[str, str],
    evidence_types: dict[str, EvidenceType | str] | None = None,
) -> SuccessContract:
    evidence_types = evidence_types or {}
    return SuccessContract(
        checks=tuple(
            SuccessContractCheck(
                name=check.name,
                evidence_type=evidence_types.get(check.name, check.evidence_type),
                source_step=source_steps.get(check.name, check.source_step),
            )
            for check in template.success_contract.checks
        )
    )


def _deploy_litserve_gpu_template() -> WorkflowTemplate:
    template = _template(
        workflow_id="deploy_litserve_gpu",
        name="Deploy LitServe GPU",
        description="Deploy a model to a Lambda Labs or local GPU VM using LitServe.",
        step_ids=(
            "detect_runtime_environment",
            "detect_gpu_cuda",
            "select_best_model_artifact",
            "generate_litserve_api",
            "configure_litserve_gpu_runtime",
            "create_dockerfile",
            "build_image_if_available",
            "start_litserve_server",
            "test_health_endpoint",
            "test_prediction_endpoint",
            "capture_logs_and_endpoint",
            "write_monitoring_and_rollback_report",
        ),
        contract_check_names=(
            "gpu_detection_recorded",
            "litserve_files_generated",
            "server_start_command_recorded",
            "health_result_recorded",
            "prediction_result_recorded",
            "endpoint_url_recorded",
            "rollback_plan_exists",
        ),
    )
    observed_check_source_steps = {
        "gpu_detection_recorded": "detect_gpu_cuda",
        "server_start_command_recorded": "start_litserve_server",
        "health_result_recorded": "test_health_endpoint",
        "prediction_result_recorded": "test_prediction_endpoint",
        "endpoint_url_recorded": "capture_logs_and_endpoint",
    }
    check_source_steps = {
        **observed_check_source_steps,
        "rollback_plan_exists": "write_monitoring_and_rollback_report",
    }
    return WorkflowTemplate(
        workflow_id=template.workflow_id,
        name=template.name,
        description=template.description,
        required_inputs=template.required_inputs,
        steps=(
            WorkflowStep(
                step_id="detect_runtime_environment",
                name="Detect Runtime Environment",
                description="Record local runtime context without provisioning cloud resources.",
                order=1,
                tool_functions=("detect_runtime_environment",),
            ),
            WorkflowStep(
                step_id="detect_gpu_cuda",
                name="Detect GPU CUDA",
                description="Detect GPU availability from observed nvidia-smi or PyTorch CUDA evidence.",
                order=2,
                tool_functions=("detect_gpu_cuda",),
            ),
            WorkflowStep(
                step_id="select_best_model_artifact",
                name="Select Best Model Artifact",
                description="Select an existing model artifact or validated preflight artifact.",
                order=3,
                tool_functions=("select_best_model_artifact",),
            ),
            WorkflowStep(
                step_id="generate_litserve_api",
                name="Generate LitServe API",
                description="Generate the LitServe serving application for the selected model.",
                order=4,
                tool_functions=("create_litserve_api",),
                default_args={
                    "model_path": "models/model.pt",
                    "model_name": "model",
                    "source_step": "generate_litserve_api",
                },
            ),
            WorkflowStep(
                step_id="configure_litserve_gpu_runtime",
                name="Configure LitServe GPU Runtime",
                description="Configure LitServe to use the GPU runtime.",
                order=5,
                tool_functions=("configure_litserver",),
                default_args={"accelerator": "gpu", "port": 8000},
            ),
            WorkflowStep(
                step_id="create_dockerfile",
                name="Create Dockerfile",
                description="Generate an optional GPU-capable Dockerfile without building an image.",
                order=6,
                tool_functions=("create_ml_dockerfile",),
                default_args={
                    "base_image": "python:3.11-slim",
                    "entry_point": "deployment/litserve/server.py",
                    "requirements_file": "deployment/litserve/requirements.txt",
                    "expose_port": 8000,
                },
            ),
            WorkflowStep(
                step_id="build_image_if_available",
                name="Build Image If Available",
                description="Record that Docker image build is optional and skipped by default.",
                order=7,
                tool_functions=("record_litserve_image_build_skipped",),
            ),
            WorkflowStep(
                step_id="start_litserve_server",
                name="Start LitServe Server",
                description="Start the LitServe server and record observed process evidence.",
                order=8,
                tool_functions=("start_litserve_server",),
            ),
            WorkflowStep(
                step_id="test_health_endpoint",
                name="Test Health Endpoint",
                description="Call the LitServe /health endpoint and record observed response evidence.",
                order=9,
                tool_functions=("test_litserve_health_endpoint",),
            ),
            WorkflowStep(
                step_id="test_prediction_endpoint",
                name="Test Prediction Endpoint",
                description="Call the LitServe /predict endpoint and record observed response evidence.",
                order=10,
                tool_functions=("test_litserve_prediction_endpoint",),
            ),
            WorkflowStep(
                step_id="capture_logs_and_endpoint",
                name="Capture Logs And Endpoint",
                description="Record deployed endpoint URL and server log location.",
                order=11,
                tool_functions=("capture_litserve_logs_and_endpoint",),
            ),
            WorkflowStep(
                step_id="write_monitoring_and_rollback_report",
                name="Write Monitoring And Rollback Report",
                description="Record stop command and manual Lambda Cloud stop instruction.",
                order=12,
                tool_functions=("record_litserve_gpu_rollback_readiness",),
            ),
        ),
        success_contract=SuccessContract(
            checks=tuple(
                SuccessContractCheck(
                    name=check.name,
                    evidence_type=(
                        "observed"
                        if check.name in observed_check_source_steps
                        else check.evidence_type
                    ),
                    source_step=check_source_steps.get(check.name, check.source_step),
                )
                for check in template.success_contract.checks
            )
        ),
        artifact_requirements=(
            ArtifactRequirement(
                name="selected_model",
                artifact_type="model_artifact",
                source_step="select_best_model_artifact",
                state="selected",
                contract_check_name="litserve_files_generated",
            ),
            ArtifactRequirement(
                name="litserve_api",
                artifact_type="serving_application",
                source_step="generate_litserve_api",
                state="generated",
                contract_check_name="litserve_files_generated",
            ),
        ),
        branches=template.branches,
        routing_aliases=(
            "Lambda Labs GPU",
            "Lambda GPU",
            "local GPU VM",
            "LitServe GPU deployment",
        ),
        negative_routing_rules=(
            "AWS Lambda serverless",
            "CPU Lambda function",
            "Prepare LitServe deployment locally",
            "LitServe deployment preflight",
            "local LitServe preflight",
            "validate LitServe locally",
        ),
        approval_gates=(
            ApprovalGate(
                step_id="detect_gpu_cuda",
                risk_categories=("uses_gpu",),
            ),
            ApprovalGate(
                step_id="generate_litserve_api",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="create_dockerfile",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="build_image_if_available",
                risk_categories=("builds_image",),
            ),
            ApprovalGate(
                step_id="start_litserve_server",
                risk_categories=("starts_server", "exposes_port"),
            ),
        ),
    )


def _deploy_gpu_inference_template() -> WorkflowTemplate:
    template = _template(
        workflow_id="deploy_gpu_inference",
        name="Deploy GPU Inference",
        description="Select a declared backend and verify a GPU inference deployment.",
        step_ids=(
            "detect_runtime_environment",
            "detect_gpu",
            "detect_cuda_driver_compatibility",
            "inspect_model_artifact_and_framework",
            "select_serving_backend",
            "generate_serving_app",
            "configure_gpu_runtime",
            "start_server",
            "test_health_endpoint",
            "test_prediction_endpoint",
            "collect_latency_metrics",
            "check_gpu_utilization",
            "generate_deployment_report",
            "generate_rollback_plan",
        ),
        contract_check_names=(
            "deployment_target_selected",
            "gpu_cuda_status_recorded",
            "serving_files_listed",
            "server_start_command_recorded",
            "health_check_passes",
            "prediction_test_passes",
            "gpu_utilization_evidence_captured",
            "latency_metrics_recorded",
            "endpoint_url_reported",
            "rollback_plan_exists",
        ),
    )
    return WorkflowTemplate(
        workflow_id=template.workflow_id,
        name=template.name,
        description=template.description,
        required_inputs=template.required_inputs,
        steps=template.steps,
        success_contract=_success_contract_with_source_overrides(
            template,
            {"rollback_plan_exists": "generate_rollback_plan"},
            {
                "gpu_cuda_status_recorded": "observed",
                "server_start_command_recorded": "observed",
                "health_check_passes": "observed",
                "prediction_test_passes": "observed",
                "gpu_utilization_evidence_captured": "observed",
                "latency_metrics_recorded": "observed",
            },
        ),
        artifact_requirements=template.artifact_requirements,
        branches=(
            WorkflowBranch(
                name="litserve",
                selection_rule="Image or classic PyTorch model GPU serving request.",
            ),
            WorkflowBranch(
                name="gradio",
                selection_rule="Demo or UI request for an interactive model experience.",
            ),
            WorkflowBranch(
                name="vllm",
                selection_rule="LLM serving request where vLLM is the first backend choice.",
            ),
            WorkflowBranch(
                name="kserve",
                selection_rule="Kubernetes, EKS, or canary production serving request.",
            ),
            WorkflowBranch(
                name="torchserve",
                selection_rule="Enterprise PyTorch registry, MAR, worker, or model versioning request.",
            ),
            WorkflowBranch(
                name="fastapi_lambda_cpu",
                selection_rule="AWS Lambda CPU serverless request when GPU is not required.",
            ),
        ),
        routing_aliases=(
            "deploy this classifier on GPU",
            "serve this LLM with vLLM",
            "run this model and tell me if GPU is being used",
            "optimize inference latency",
        ),
        negative_routing_rules=("Lambda Labs GPU", "Lambda Labs GPU direct LitServe request"),
        approval_gates=(
            ApprovalGate(
                step_id="generate_serving_app",
                risk_categories=("writes_project_files",),
            ),
            ApprovalGate(
                step_id="start_server",
                risk_categories=("starts_server", "exposes_port"),
            ),
        ),
    )


def _deploy_gradio_demo_template() -> WorkflowTemplate:
    template = _template(
        workflow_id="deploy_gradio_demo",
        name="Deploy Gradio Demo",
        description="Create or validate a quick Gradio demo for a trained model.",
        step_ids=(
            "select_model_artifact",
            "generate_gradio_app",
            "validate_launch_command",
            "test_sample_prediction_path",
            "document_rollback_target",
            "prepare_hugging_face_spaces_package",
        ),
        contract_check_names=(
            "app_file_exists",
            "launch_command_exists",
            "sample_prediction_path_documented",
            "rollback_plan_exists",
        ),
    )
    return WorkflowTemplate(
        workflow_id=template.workflow_id,
        name=template.name,
        description=template.description,
        required_inputs=template.required_inputs,
        steps=template.steps,
        success_contract=_success_contract_with_source_overrides(
            template,
            {"rollback_plan_exists": "document_rollback_target"},
        ),
        artifact_requirements=template.artifact_requirements,
        branches=template.branches,
        routing_aliases=("Create a Gradio demo", "Gradio UI", "demo app"),
        negative_routing_rules=("production Kubernetes deployment",),
        approval_gates=(
            ApprovalGate(
                step_id="generate_gradio_app",
                risk_categories=("writes_project_files",),
            ),
        ),
    )


def _deploy_kserve_production_template() -> WorkflowTemplate:
    template = _template(
        workflow_id="deploy_kserve_production",
        name="Deploy KServe Production",
        description="Prepare Kubernetes or EKS deployment data for KServe production serving.",
        step_ids=(
            "detect_kubernetes_context",
            "detect_eks_cluster",
            "create_or_identify_ecr_repo",
            "build_and_push_image",
            "generate_inference_service",
            "generate_kubernetes_support_manifests",
            "run_kubectl_server_dry_run",
            "prepare_canary_rollout",
            "prepare_rollback",
            "record_monitoring_endpoints",
        ),
        contract_check_names=(
            "kubernetes_manifests_validate",
            "registry_image_reference_exists",
            "canary_rollback_plan_exists",
            "dry_run_result_recorded",
        ),
    )
    return WorkflowTemplate(
        workflow_id=template.workflow_id,
        name=template.name,
        description=template.description,
        required_inputs=template.required_inputs,
        steps=template.steps,
        success_contract=_success_contract_with_source_overrides(
            template,
            {"canary_rollback_plan_exists": "prepare_rollback"},
            {
                "kubernetes_manifests_validate": "observed",
                "dry_run_result_recorded": "observed",
            },
        ),
        artifact_requirements=template.artifact_requirements,
        branches=template.branches,
        routing_aliases=(
            "KServe canary rollout",
            "Deploy to KServe with canary rollout",
            "deploy to Kubernetes",
            "deploy to EKS",
        ),
        negative_routing_rules=("quick Gradio demo", "AWS Lambda serverless"),
        approval_gates=(
            ApprovalGate(
                step_id="build_and_push_image",
                risk_categories=("builds_image", "pushes_registry"),
            ),
            ApprovalGate(
                step_id="run_kubectl_server_dry_run",
                risk_categories=("uses_cloud_credentials",),
            ),
        ),
    )
