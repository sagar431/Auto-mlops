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


@dataclass(frozen=True)
class SuccessContractCheck:
    """A named check that must be satisfied before workflow success."""

    name: str
    evidence_type: str
    source_step: str


@dataclass(frozen=True)
class SuccessContract:
    """The structured completion contract for a workflow template."""

    checks: tuple[SuccessContractCheck, ...]


@dataclass(frozen=True)
class ArtifactRequirement:
    """An artifact a workflow is expected to produce, validate, or report."""

    name: str
    artifact_type: str
    source_step: str


@dataclass(frozen=True)
class WorkflowBranch:
    """A registry-owned alternative path within a workflow template."""

    name: str
    selection_rule: str


@dataclass(frozen=True)
class ApprovalGate:
    """Human approval metadata required before a risky workflow step may run."""

    step_id: str
    risk_categories: tuple[str, ...]


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
            ),
            ArtifactRequirement(
                name="dvc_yaml",
                artifact_type="pipeline_definition",
                source_step="create_dvc_yaml",
            ),
            ArtifactRequirement(
                name="dockerfile",
                artifact_type="container_definition",
                source_step="create_dockerfile",
            ),
            ArtifactRequirement(
                name="ci_workflow",
                artifact_type="automation_workflow",
                source_step="create_ci_workflow",
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
    observed_checks = {
        "gpu_detection_recorded",
        "server_start_command_recorded",
        "health_result_recorded",
        "prediction_result_recorded",
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
                    evidence_type="observed" if check.name in observed_checks else check.evidence_type,
                    source_step=check.source_step,
                )
                for check in template.success_contract.checks
            )
        ),
        artifact_requirements=template.artifact_requirements,
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
        success_contract=template.success_contract,
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
            "prepare_hugging_face_spaces_package",
        ),
        contract_check_names=(
            "app_file_exists",
            "launch_command_exists",
            "sample_prediction_path_documented",
        ),
    )
    return WorkflowTemplate(
        workflow_id=template.workflow_id,
        name=template.name,
        description=template.description,
        required_inputs=template.required_inputs,
        steps=template.steps,
        success_contract=template.success_contract,
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
        success_contract=template.success_contract,
        artifact_requirements=template.artifact_requirements,
        branches=template.branches,
        routing_aliases=("KServe canary rollout", "deploy to Kubernetes", "deploy to EKS"),
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
