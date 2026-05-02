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
class WorkflowTemplate:
    """An ordered skeleton for one supported workflow."""

    workflow_id: str
    name: str
    description: str
    required_inputs: tuple[WorkflowInput, ...]
    steps: tuple[WorkflowStep, ...]
    success_contract: SuccessContract
    artifact_requirements: tuple[ArtifactRequirement, ...] = ()

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

    def get(self, workflow_id: str) -> WorkflowTemplate:
        try:
            return self._templates[workflow_id]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow template: {workflow_id}") from exc

    def _validate_templates(self) -> None:
        for template in self._templates.values():
            if not template.steps:
                raise ValueError(f"Fake Template '{template.workflow_id}' has no workflow steps")
            if not template.success_contract.checks:
                raise ValueError(
                    f"Fake Template '{template.workflow_id}' has no success contract checks"
                )


def get_workflow_registry() -> WorkflowRegistry:
    """Return the Phase 0 workflow registry."""

    return WorkflowRegistry((_setup_pipeline_template(),))


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
