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

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_type", EvidenceType(self.evidence_type))


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
            if any(
                _routing_phrase_matches(normalized_request, negative_rule)
                for negative_rule in template.negative_routing_rules
            ):
                rejected_workflows.append(template.workflow_id)
                continue

            for alias in template.routing_aliases:
                if _routing_phrase_matches(normalized_request, alias):
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
    ) -> ContractValidation:
        """Derive workflow status from required success contract verification results."""

        template = self.get(workflow_id)
        if artifact_manifest is not None:
            self.validate_artifact_manifest(workflow_id, artifact_manifest)
        missing_evidence: list[ContractFailure] = []
        failed_checks: list[ContractFailure] = []
        for check in template.success_contract.checks:
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
                failed_checks.append(
                    ContractFailure(
                        check_name=check.name,
                        expected_evidence_type=check.evidence_type,
                        source_step=check.source_step,
                        actual_evidence=satisfying_evidence,
                        next_action=(
                            f"Resolve failed verification for check '{check.name}' "
                            f"from step '{check.source_step}'."
                        ),
                    )
                )
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
        if missing_evidence:
            status = WorkflowStatus.BLOCKED
        elif failed_checks:
            status = WorkflowStatus.FAILED
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
    """Return the Phase 0 workflow registry."""

    return WorkflowRegistry(
        (
            _setup_pipeline_template(),
            _deploy_litserve_gpu_template(),
            _deploy_gpu_inference_template(),
            _deploy_gradio_demo_template(),
            _deploy_kserve_production_template(),
        )
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
        steps=template.steps,
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
        negative_routing_rules=("AWS Lambda serverless", "CPU Lambda function"),
        approval_gates=(
            ApprovalGate(
                step_id="generate_litserve_api",
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
