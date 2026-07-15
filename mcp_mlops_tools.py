#!/usr/bin/env python3
"""
MCP Server for MLOps Operations
Provides tools for complete ML pipeline automation:
- Hydra configuration management
- MLflow experiment tracking
- DVC data versioning
- Docker containerization
- GitHub Actions CI/CD

Author: Sagar
Version: 1.0.0
"""

import asyncio
import base64
import configparser
import hashlib
import json
import os
import pickle
import random
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

    BOTO3_AVAILABLE = True
except Exception:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    NoCredentialsError = Exception
    BOTO3_AVAILABLE = False

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from pydantic import BaseModel, Field

from mcp_servers.mlops.common.paths import ensure_directory, relative_to_project
from mcp_servers.mlops.domains import hydra as hydra_domain
from mcp_servers.mlops.schemas import hydra as hydra_schemas
from mcp_servers.mlops.server import build_tool_registry

HydraDependencies = hydra_domain.HydraDependencies
configure_hydra_dependencies = hydra_domain.configure_dependencies
analyze_project_config = hydra_domain.analyze_project_config
create_hydra_config = hydra_domain.create_hydra_config
update_hydra_config = hydra_domain.update_hydra_config
validate_hydra_config = hydra_domain.validate_hydra_config
AnalyzeProjectConfigInput = hydra_schemas.AnalyzeProjectConfigInput
CreateHydraConfigInput = hydra_schemas.CreateHydraConfigInput
UpdateHydraConfigInput = hydra_schemas.UpdateHydraConfigInput
ValidateHydraConfigInput = hydra_schemas.ValidateHydraConfigInput

# ============================================================================
# Helper Functions
# ============================================================================


def run_command(cmd: list[str], cwd: str | None = None, timeout: int = 60) -> dict[str, Any]:
    """Run a shell command and return result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_tool_installed(tool_name: str) -> bool:
    """Check if a CLI tool is installed."""
    return shutil.which(tool_name) is not None


# ============================================================================
# Pydantic Input Models
# ============================================================================

# --- MLflow Experiment Tracking Tools ---


class InitMLflowExperimentInput(BaseModel):
    """Initialize MLflow experiment."""

    experiment_name: str = Field(..., description="Name of the experiment")
    tracking_uri: str | None = Field(default=None, description="MLflow tracking URI")
    artifact_location: str | None = Field(default=None, description="Artifact storage location")
    tags: dict[str, str] | None = Field(default=None, description="Experiment tags")


class LogMLflowParamsInput(BaseModel):
    """Log parameters to MLflow."""

    run_id: str | None = Field(
        default=None, description="Run ID (uses active run if not specified)"
    )
    params: dict[str, Any] = Field(..., description="Parameters to log")


class LogMLflowMetricsInput(BaseModel):
    """Log metrics to MLflow."""

    run_id: str | None = Field(
        default=None, description="Run ID (uses active run if not specified)"
    )
    metrics: dict[str, float] = Field(..., description="Metrics to log")
    step: int | None = Field(default=None, description="Step number for the metrics")


class LogMLflowArtifactInput(BaseModel):
    """Log artifact to MLflow."""

    artifact_path: str = Field(..., description="Local path to artifact file or directory")
    artifact_dest: str | None = Field(
        default=None, description="Destination path in artifact store"
    )
    run_id: str | None = Field(
        default=None, description="Run ID (uses active run if not specified)"
    )


class RegisterMLflowModelInput(BaseModel):
    """Register model in MLflow Model Registry."""

    model_path: str = Field(..., description="Path to the model artifact")
    model_name: str = Field(..., description="Name for the registered model")
    run_id: str | None = Field(default=None, description="Run ID containing the model")
    tags: dict[str, str] | None = Field(default=None, description="Model tags")


class GetBestMLflowRunInput(BaseModel):
    """Get best run from experiment based on metric."""

    experiment_name: str = Field(..., description="Name of the experiment")
    metric_name: str = Field(default="accuracy", description="Metric to optimize")
    maximize: bool = Field(default=True, description="Whether to maximize the metric")


class StartMLflowRunInput(BaseModel):
    """Start a new MLflow run."""

    experiment_name: str = Field(..., description="Name of the experiment")
    run_name: str | None = Field(default=None, description="Name for the run")
    tags: dict[str, str] | None = Field(default=None, description="Run tags")


class EndMLflowRunInput(BaseModel):
    """End an MLflow run."""

    run_id: str | None = Field(default=None, description="Run ID to end")
    status: str = Field(default="FINISHED", description="Run status: FINISHED, FAILED, KILLED")


# --- DVC Data Versioning Tools ---


class InitDVCRepoInput(BaseModel):
    """Initialize DVC in a repository."""

    project_path: str = Field(..., description="Path to the project")
    no_scm: bool = Field(default=False, description="Initialize without Git integration")


class ConfigureDVCRemoteInput(BaseModel):
    """Configure DVC remote storage."""

    project_path: str = Field(..., description="Path to the project")
    remote_name: str = Field(default="storage", description="Name for the remote")
    remote_url: str = Field(..., description="Remote URL (s3://, gs://, azure://, etc.)")
    default: bool = Field(default=True, description="Set as default remote")


class AddDataToDVCInput(BaseModel):
    """Add data to DVC tracking."""

    project_path: str = Field(..., description="Path to the project")
    data_path: str = Field(..., description="Path to data file or directory (relative to project)")


class CreateDVCPipelineInput(BaseModel):
    """Create DVC pipeline."""

    project_path: str = Field(..., description="Path to the project")
    stages: list[dict[str, Any]] = Field(..., description="List of pipeline stages")


class DVCPushInput(BaseModel):
    """Push data to DVC remote."""

    project_path: str = Field(..., description="Path to the project")
    remote_name: str | None = Field(
        default=None, description="Remote name (uses default if not specified)"
    )


class DVCPullInput(BaseModel):
    """Pull data from DVC remote."""

    project_path: str = Field(..., description="Path to the project")
    remote_name: str | None = Field(
        default=None, description="Remote name (uses default if not specified)"
    )


class DVCReproduceInput(BaseModel):
    """Reproduce DVC pipeline."""

    project_path: str = Field(..., description="Path to the project")
    stages: list[str] | None = Field(default=None, description="Specific stages to reproduce")
    force: bool = Field(default=False, description="Force reproduction even if up-to-date")


# --- Docker Tools ---


class CreateMLDockerfileInput(BaseModel):
    """Create Dockerfile for ML project."""

    project_path: str = Field(..., description="Path to the project")
    base_image: str = Field(default="python:3.11-slim", description="Base Docker image")
    cuda_version: str | None = Field(default=None, description="CUDA version if GPU support needed")
    entry_point: str = Field(default="train.py", description="Training script entry point")
    requirements_file: str = Field(default="requirements.txt", description="Requirements file path")
    expose_port: int | None = Field(default=None, description="Port to expose")


class BuildMLDockerImageInput(BaseModel):
    """Build Docker image for ML project."""

    project_path: str = Field(..., description="Path to the project")
    image_name: str = Field(..., description="Name for the Docker image")
    tag: str = Field(default="latest", description="Image tag")
    dockerfile: str = Field(default="Dockerfile", description="Dockerfile path")


class RunTrainingContainerInput(BaseModel):
    """Run training in Docker container."""

    image_name: str = Field(..., description="Docker image name")
    tag: str = Field(default="latest", description="Image tag")
    gpu: bool = Field(default=False, description="Enable GPU support")
    volumes: dict[str, str] | None = Field(default=None, description="Volume mappings")
    env_vars: dict[str, str] | None = Field(default=None, description="Environment variables")
    command: str | None = Field(default=None, description="Override command")


class PushDockerImageInput(BaseModel):
    """Push Docker image to registry."""

    image_name: str = Field(..., description="Docker image name")
    tag: str = Field(default="latest", description="Image tag")
    registry: str | None = Field(default=None, description="Registry URL")


# --- GitHub Actions Tools ---


class CreateGitHubWorkflowInput(BaseModel):
    """Create GitHub Actions workflow for ML pipeline."""

    project_path: str = Field(..., description="Path to the project")
    workflow_name: str = Field(default="ml-pipeline", description="Workflow name")
    trigger_on: list[str] = Field(
        default=["push", "workflow_dispatch"], description="Trigger events"
    )
    python_version: str = Field(default="3.11", description="Python version")
    use_dvc: bool = Field(default=True, description="Include DVC steps")
    use_mlflow: bool = Field(default=True, description="Include MLflow tracking")
    accuracy_threshold: float | None = Field(default=None, description="Accuracy threshold for CI")


class AddWorkflowStepInput(BaseModel):
    """Add step to existing GitHub workflow."""

    project_path: str = Field(..., description="Path to the project")
    workflow_file: str = Field(
        default=".github/workflows/ml-pipeline.yml", description="Workflow file path"
    )
    job_name: str = Field(default="train", description="Job to add step to")
    step: dict[str, Any] = Field(..., description="Step configuration")


class TriggerGitHubWorkflowInput(BaseModel):
    """Trigger GitHub Actions workflow (via API)."""

    repo: str = Field(..., description="Repository in format owner/repo")
    workflow_id: str = Field(..., description="Workflow file name or ID")
    ref: str = Field(default="main", description="Branch/tag/SHA to run workflow on")
    inputs: dict[str, str] | None = Field(default=None, description="Workflow inputs")


class CheckWorkflowRunInput(BaseModel):
    """Check status of GitHub workflow run."""

    repo: str = Field(..., description="Repository in format owner/repo")
    run_id: int = Field(..., description="Workflow run ID")


# --- Training Control Tools ---


class AnalyzeTrainingResultsInput(BaseModel):
    """Analyze training results and suggest improvements."""

    project_path: str = Field(..., description="Path to the project")
    experiment_name: str = Field(..., description="MLflow experiment name")
    target_metric: str = Field(default="accuracy", description="Target metric")
    target_value: float = Field(..., description="Target value for the metric")


class SuggestImprovementsInput(BaseModel):
    """Suggest improvements based on training results."""

    current_metrics: dict[str, float] = Field(..., description="Current metrics")
    current_config: dict[str, Any] = Field(..., description="Current configuration")
    target_accuracy: float = Field(..., description="Target accuracy")
    attempt_number: int = Field(default=1, description="Current attempt number")


class CheckAccuracyThresholdInput(BaseModel):
    """Check if accuracy threshold is met."""

    experiment_name: str = Field(..., description="MLflow experiment name")
    threshold: float = Field(..., description="Accuracy threshold")
    metric_name: str = Field(default="accuracy", description="Metric name to check")


class DetectTrainingProjectInput(BaseModel):
    """Detect a supported Hydra/PyTorch/TIMM training project without running training."""

    project_path: str = Field(..., description="Path to the training project")


class RunBoundedTrainingInput(BaseModel):
    """Run a detected training entrypoint with explicit bounded controls."""

    project_path: str = Field(..., description="Path to the detected training project")
    training_entrypoint: str = Field(..., description="Detected training script path")
    hydra_config_path: str | None = Field(default=None, description="Hydra config root path")
    hydra_config_name: str | None = Field(default=None, description="Hydra config name")
    timeout_seconds: int = Field(..., description="Maximum training command duration")
    max_epochs: int = Field(..., description="Maximum epochs or equivalent control")
    device: str = Field(..., description="Training device control")
    data_subset: int = Field(..., description="Dataset subset or size control")
    hydra_overrides: list[str] | None = Field(
        default=None, description="Additional explicit Hydra overrides"
    )
    target_metric: str = Field(default="accuracy", description="Required metric for success")


class TrackTrainingInMLflowInput(BaseModel):
    """Track a bounded training result in a verified local MLflow run."""

    project_path: str = Field(..., description="Path to the training project")
    training_result: dict[str, Any] = Field(..., description="Bounded training result payload")
    experiment_name: str = Field(default="mlops-training", description="MLflow experiment name")
    tracking_uri: str | None = Field(default=None, description="Local or file MLflow tracking URI")
    run_name: str | None = Field(default=None, description="Optional MLflow run name")
    params: dict[str, Any] | None = Field(
        default=None, description="Additional bounded controls and detected params to log"
    )


class RecordCapstoneOrchestratorSkeletonInput(BaseModel):
    """Record the first Capstone Orchestrator skeleton and deferred evidence."""

    project_path: str = Field(..., description="Path to the project")
    declared_stages: list[str] | None = Field(
        default=None,
        description="Declared capstone stages owned by the registry template",
    )
    implemented_subworkflows: list[str] | None = Field(
        default=None,
        description="Implemented workflow ids the skeleton may reference",
    )
    blocked_subworkflows: list[str] | None = Field(
        default=None,
        description="Known missing workflow ids that must remain blocked",
    )
    selected_model_artifact_path: str | None = Field(
        default=None,
        description="Optional selected model artifact evidence from a prior workflow",
    )
    endpoint_url: str | None = Field(
        default=None,
        description="Optional endpoint evidence from a prior deployment workflow",
    )


class DetectCapstoneDataLayoutsInput(BaseModel):
    """Detect two user-provided canonical image-folder datasets read-only."""

    project_path: str = Field(..., description="Path to the project")
    dataset_1_path: str = Field(..., description="First user-provided dataset path")
    dataset_2_path: str = Field(..., description="Second user-provided dataset path")
    completion_mode: str = Field(default="local_ready", description="Phase 4 completion mode")
    test_size: float = Field(default=0.2, description="Declared test split ratio")
    split_seed: int = Field(default=42, description="Declared deterministic split seed")


class GenerateCapstoneSplitManifestsInput(BaseModel):
    """Generate deterministic split manifests under the project capstone data path."""

    project_path: str = Field(..., description="Path to the project")
    capstone_data_detection: dict[str, Any] = Field(
        ..., description="Output from detect_capstone_data_layouts"
    )
    test_size: float = Field(default=0.2, description="Test split ratio")
    split_seed: int = Field(default=42, description="Deterministic split seed")
    materialize_splits: bool = Field(
        default=False,
        description="Reserved for future approval-gated copied folder materialization.",
    )


class TrackCapstoneDataPackageInput(BaseModel):
    """Validate/init local DVC and track generated capstone package paths."""

    project_path: str = Field(..., description="Path to the project")
    capstone_split_manifest_result: dict[str, Any] = Field(
        ..., description="Output from generate_capstone_split_manifests"
    )
    initialize_if_missing: bool = Field(
        default=True,
        description="Initialize local DVC metadata when .dvc/config is missing.",
    )


class ConfigureValidateCapstoneDVCRemoteInput(BaseModel):
    """Configure or validate local/S3 DVC remote evidence without data transfer."""

    project_path: str = Field(..., description="Path to the project")
    completion_mode: str = Field(default="local_ready", description="Phase 4 completion mode")
    remote_name: str = Field(default="capstone", description="DVC remote name")
    remote_url: str | None = Field(
        default=None,
        description="Optional local or s3:// DVC remote URL to configure and validate.",
    )
    default: bool = Field(default=True, description="Set configured remote as DVC default")
    source_step: str = Field(
        default="configure_validate_dvc_remote",
        description="Workflow step that owns emitted remote evidence.",
    )


class PushCapstoneDataInput(BaseModel):
    """Run approval-gated DVC push for capstone data package evidence."""

    project_path: str = Field(..., description="Path to the project")
    completion_mode: str = Field(default="capstone_complete", description="Phase 4 completion mode")
    remote_name: str = Field(default="capstone", description="DVC remote name")
    capstone_dvc_remote_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from configure_validate_capstone_dvc_remote",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approved ApprovalRecord payload for push_capstone_data",
    )
    paths: list[str] | None = Field(default=None, description="Optional DVC paths to push")
    source_step: str = Field(
        default="push_capstone_data",
        description="Workflow step that owns emitted transfer evidence.",
    )


class PullCapstoneDataInput(BaseModel):
    """Run approval-gated DVC pull for capstone data package evidence."""

    project_path: str = Field(..., description="Path to the project")
    completion_mode: str = Field(default="capstone_complete", description="Phase 4 completion mode")
    remote_name: str = Field(default="capstone", description="DVC remote name")
    capstone_dvc_remote_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from configure_validate_capstone_dvc_remote",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approved ApprovalRecord payload for pull_capstone_data",
    )
    paths: list[str] | None = Field(default=None, description="Optional DVC paths to pull")
    source_step: str = Field(
        default="pull_capstone_data",
        description="Workflow step that owns emitted transfer evidence.",
    )


class RecordCapstoneDataStageEvidenceInput(BaseModel):
    """Write the durable Phase 4 data-stage evidence handoff artifact."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_data workflow inputs",
    )
    capstone_data_detection: dict[str, Any] | None = Field(
        default=None,
        description="Output from detect_capstone_data_layouts",
    )
    capstone_split_manifest_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from generate_capstone_split_manifests",
    )
    capstone_data_package_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from track_capstone_data_package",
    )
    capstone_data_remote_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from configure_validate_capstone_dvc_remote",
    )
    capstone_data_push_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from push_capstone_data",
    )
    capstone_data_pull_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from pull_capstone_data",
    )
    verification_results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Verification results captured before writing data-stage evidence",
    )
    artifact_manifest: dict[str, Any] | None = Field(
        default=None,
        description="Artifact manifest captured before writing data-stage evidence",
    )


class PrepareCapstoneContainerCIContractInput(BaseModel):
    """Validate Phase 5 Issue 1 container/CI inputs without later behavior."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )


class ResolveCapstoneContainerUpstreamEvidenceInput(BaseModel):
    """Resolve Phase 5 Issue 2 upstream evidence without mutating project state."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )


class GenerateValidateCapstoneRuntimeImageSpecInput(BaseModel):
    """Generate or validate Phase 5 Issue 3 runtime image build-spec evidence."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approval record required before writing Dockerfile or .dockerignore",
    )


class BuildSmokeCheckCapstoneContainerImageInput(BaseModel):
    """Build and smoke-check Phase 5 runtime image when Docker is available and approved."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )
    capstone_runtime_image_spec_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from generate_validate_capstone_runtime_image_spec",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approval record required before running docker build",
    )
    smoke_approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approval record required before running container smoke command",
    )
    build_timeout_seconds: int = Field(
        default=300,
        description="Bounded timeout for docker build",
    )
    smoke_timeout_seconds: int = Field(
        default=60,
        description="Bounded timeout for container smoke check",
    )


class ConfigureValidateCapstoneRegistryTargetInput(BaseModel):
    """Configure and validate Phase 5 Issue 5 registry target evidence."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )
    capstone_container_build_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from build_smoke_check_capstone_container_image",
    )


class ApprovalGatedCapstoneRegistryLoginPushInput(BaseModel):
    """Run Phase 5 Issue 6 approval-gated registry login and image push."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )
    capstone_registry_target_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from configure_validate_capstone_registry_target",
    )
    capstone_container_build_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from build_smoke_check_capstone_container_image",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approval record required before registry login and push",
    )
    push_timeout_seconds: int = Field(
        default=600,
        description="Bounded timeout for docker push",
    )


class RecordCapstoneContainerCIEvidenceInput(BaseModel):
    """Write Phase 5 Issue 7 CI workflow and durable container/CI evidence."""

    project_path: str = Field(..., description="Path to the project")
    workflow_inputs: dict[str, Any] | None = Field(
        default=None,
        description="Resolved prepare_capstone_container_ci workflow inputs",
    )
    capstone_container_upstream_evidence: dict[str, Any] | None = Field(
        default=None,
        description="Output from resolve_capstone_container_upstream_evidence",
    )
    capstone_runtime_image_spec_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from generate_validate_capstone_runtime_image_spec",
    )
    capstone_container_build_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from build_smoke_check_capstone_container_image",
    )
    capstone_registry_target_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from configure_validate_capstone_registry_target",
    )
    capstone_registry_push_result: dict[str, Any] | None = Field(
        default=None,
        description="Output from approval_gated_capstone_registry_login_push",
    )
    verification_results: list[dict[str, Any]] | None = Field(
        default=None,
        description="Verification results captured before writing container/CI evidence",
    )
    artifact_manifest: dict[str, Any] | None = Field(
        default=None,
        description="Artifact manifest captured before writing container/CI evidence",
    )
    approval_record: dict[str, Any] | None = Field(
        default=None,
        description="Approval record required before writing capstone CI workflow/evidence",
    )
    remote_ci_evidence: dict[str, Any] | None = Field(
        default=None,
        description="Optional remote GitHub Actions evidence when authenticated inspection is available",
    )


# --- Data Quality Tools ---


class ValidateDatasetInput(BaseModel):
    """Validate ML dataset for quality issues."""

    dataset_path: str = Field(..., description="Path to the dataset file or directory")
    dataset_type: str = Field(
        default="auto",
        description="Dataset type: csv, parquet, images, json, auto (auto-detect)",
    )
    checks: list[str] | None = Field(
        default=None,
        description="Specific checks to run: missing_values, duplicates, class_balance, data_types, outliers, image_validity. If None, runs all applicable checks.",
    )
    sample_size: int | None = Field(
        default=None,
        description="Number of samples to validate (for large datasets). If None, validates all.",
    )


class CreateExpectationSuiteInput(BaseModel):
    """Create a Great Expectations expectation suite for data validation."""

    project_path: str = Field(..., description="Path to the ML project")
    suite_name: str = Field(..., description="Name of the expectation suite")
    expectations: list[dict[str, Any]] = Field(
        ...,
        description="List of expectations. Each expectation is a dict with 'expectation_type' (e.g., 'expect_column_values_to_not_be_null'), optional 'column', optional 'kwargs' (expectation-specific parameters), optional 'severity' ('error'/'warning'/'info'), and optional 'description'.",
    )
    output_dir: str = Field(
        default="great_expectations/expectations",
        description="Directory to save the suite (relative to project_path)",
    )


class CheckDataQualityInput(BaseModel):
    """Check data quality using Great Expectations-based validation."""

    dataset_path: str = Field(..., description="Path to the dataset file (CSV, Parquet, or JSON)")
    expectations: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of expectations to check. Each expectation is a dict with 'expectation_type', optional 'column', optional 'kwargs'. If not provided, runs basic quality checks (nulls, duplicates, row count).",
    )
    include_statistics: bool = Field(
        default=True, description="Whether to include detailed dataset statistics in the report"
    )
    fail_on_error: bool = Field(
        default=False,
        description="If True, returns success=False when any ERROR-severity check fails",
    )


class ProfileDatasetInput(BaseModel):
    """Profile a dataset to get comprehensive statistics."""

    dataset_path: str = Field(..., description="Path to the dataset file (CSV, Parquet, or JSON)")
    dataset_name: str | None = Field(
        default=None,
        description="Name for the dataset in the report. If not provided, uses filename.",
    )
    include_column_stats: bool = Field(
        default=True, description="Whether to include detailed column-level statistics"
    )


class DetectAnomaliesInput(BaseModel):
    """Detect anomalies in a dataset using statistical methods."""

    dataset_path: str = Field(..., description="Path to the dataset file (CSV, Parquet, or JSON)")
    dataset_name: str | None = Field(
        default=None,
        description="Name for the dataset in the report. If not provided, uses filename.",
    )
    methods: list[str] | None = Field(
        default=None,
        description="List of detection methods to use: 'iqr' (interquartile range), 'zscore', 'missing' (missing value patterns), 'duplicates'. If None, uses all methods.",
    )
    outlier_threshold: float = Field(
        default=1.5, description="IQR multiplier for outlier detection (default 1.5)"
    )
    zscore_threshold: float = Field(
        default=3.0, description="Z-score threshold for outlier detection (default 3.0)"
    )


class ValidateSchemaInput(BaseModel):
    """Validate a dataset against a defined schema."""

    dataset_path: str = Field(..., description="Path to the dataset file (CSV, Parquet, or JSON)")
    schema_definition: dict[str, Any] = Field(
        ...,
        description="Schema definition with 'schema_name', 'version' (optional), 'fields' (list of field definitions with 'name', 'data_type', 'nullable', 'unique', 'min_value', 'max_value', 'allowed_values', 'pattern'), and 'strict' (bool, if true extra columns are not allowed).",
        alias="schema",
    )


class CompareDistributionsInput(BaseModel):
    """Compare distributions between a reference dataset and current dataset for drift detection."""

    reference_path: str = Field(
        ..., description="Path to the reference dataset file (e.g., training data)"
    )
    current_path: str = Field(
        ..., description="Path to the current dataset file to compare against reference"
    )
    columns: list[str] | None = Field(
        default=None,
        description="List of column names to compare. If None, compares all common numeric columns.",
    )
    significance_level: float = Field(
        default=0.05,
        description="Significance level for statistical tests (default 0.05). Lower values mean stricter detection.",
    )


# --- Monitoring Tools ---


class DetectDataDriftInput(BaseModel):
    """Detect data drift between reference and current datasets using Evidently AI."""

    reference_path: str = Field(
        ..., description="Path to the reference dataset file (e.g., training data)"
    )
    current_path: str = Field(
        ..., description="Path to the current dataset file to compare against reference"
    )
    feature_columns: list[str] | None = Field(
        default=None,
        description="List of column names to check for drift. If None, checks all columns.",
    )
    categorical_columns: list[str] | None = Field(
        default=None,
        description="Explicitly specify which columns are categorical.",
    )
    numerical_columns: list[str] | None = Field(
        default=None,
        description="Explicitly specify which columns are numerical.",
    )
    drift_threshold: float = Field(
        default=0.1,
        description="Threshold for drift detection (0-1, lower = stricter). Default 0.1.",
    )
    dataset_name: str = Field(
        default="dataset",
        description="Name for the dataset in the report.",
    )


class MonitorModelPerformanceInput(BaseModel):
    """Monitor model performance metrics and detect degradation."""

    model_name: str = Field(..., description="Name of the model to monitor")
    y_true: list[float | int] = Field(..., description="Ground truth labels/values")
    y_pred: list[float | int] = Field(..., description="Predicted labels/values")
    y_prob: list[list[float]] | None = Field(
        default=None,
        description="Prediction probabilities (for classification, shape: [n_samples, n_classes])",
    )
    task_type: str = Field(
        default="classification",
        description="Task type: 'classification' or 'regression'",
    )
    model_version: str | None = Field(default=None, description="Version of the model")
    degradation_threshold: float = Field(
        default=0.05,
        description="Threshold for degradation detection (0-1). Default 0.05 (5%)",
    )
    baseline_metrics: dict[str, float] | None = Field(
        default=None,
        description="Baseline metrics to compare against (e.g., {'accuracy': 0.95, 'f1_score': 0.92})",
    )
    metrics_to_check: list[str] | None = Field(
        default=None,
        description="List of metrics to evaluate for health status. Default: ['accuracy', 'f1_score', 'precision', 'recall']",
    )
    record_snapshot: bool = Field(
        default=True,
        description="Whether to record this evaluation as a performance snapshot",
    )
    storage_path: str | None = Field(
        default=None,
        description="Path to save/load performance history (JSON file)",
    )


class SetupAlertingInput(BaseModel):
    """Setup alerting configuration for model monitoring."""

    project_path: str = Field(..., description="Path to the project")
    alert_name: str = Field(..., description="Name for the alerting configuration")
    alert_type: str = Field(
        default="threshold",
        description="Type of alert: threshold, anomaly, drift, or composite",
    )
    metrics: list[str] = Field(
        default=["accuracy", "latency"],
        description="List of metrics to monitor for alerts",
    )
    thresholds: dict[str, float] | None = Field(
        default=None,
        description="Threshold values for each metric (e.g., {'accuracy': 0.9, 'latency': 100})",
    )
    notification_channels: list[str] = Field(
        default=["email"],
        description="Notification channels: email, slack, pagerduty, webhook",
    )
    notification_config: dict[str, Any] | None = Field(
        default=None,
        description="Configuration for notification channels (e.g., {'email': {'recipients': ['team@example.com']}, 'slack': {'webhook_url': '...'}})",
    )
    evaluation_window: str = Field(
        default="5m",
        description="Time window for metric evaluation (e.g., '5m', '1h', '1d')",
    )
    cooldown_period: str = Field(
        default="15m",
        description="Minimum time between repeated alerts (e.g., '15m', '1h')",
    )
    severity: str = Field(
        default="warning",
        description="Alert severity: info, warning, critical",
    )
    enabled: bool = Field(default=True, description="Whether the alert is enabled")


# --- Deployment Tools (Phase 4) ---


# LitServe Tools
class CreateLitserveAPIInput(BaseModel):
    """Create LitServe API for model serving."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str = Field(..., description="Path to the model file (relative to project)")
    model_name: str = Field(..., description="Name for the model/API")
    model_type: str = Field(
        default="image_classifier",
        description="Model type: image_classifier, text_classifier, object_detection",
    )
    class_labels: list[str] | None = Field(default=None, description="List of class labels")
    source_step: str = Field(
        default="generate_litserve_api",
        description="Workflow step that should own emitted evidence.",
    )


class ConfigureLitserverInput(BaseModel):
    """Configure LitServe server settings."""

    project_path: str = Field(..., description="Path to the project")
    max_batch_size: int = Field(default=64, description="Maximum batch size for inference")
    batch_timeout: float = Field(default=0.05, description="Batch timeout in seconds")
    workers_per_device: int = Field(default=4, description="Number of workers per device")
    accelerator: str = Field(default="auto", description="Accelerator: cpu, gpu, auto")
    port: int = Field(default=8000, description="Server port")


class SelectOrCreateModelArtifactInput(BaseModel):
    """Select or create a local model artifact for LitServe preflight."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str | None = Field(
        default=None,
        description="Optional model artifact path relative to the project.",
    )
    model_name: str = Field(default="model", description="Name to use for a placeholder model")
    create_placeholder: bool = Field(
        default=True,
        description="Create a local placeholder when no model artifact is found.",
    )


class GenerateLitserveDockerfileInput(BaseModel):
    """Generate or validate a local Dockerfile for LitServe preflight."""

    project_path: str = Field(..., description="Path to the project")
    server_path: str = Field(
        default="deployment/litserve/server.py",
        description="LitServe server entry point relative to the project.",
    )
    requirements_file: str = Field(
        default="deployment/litserve/requirements.txt",
        description="Requirements file relative to the project.",
    )
    port: int = Field(default=8000, description="Container port to expose")


class RecordLitserveLaunchCommandInput(BaseModel):
    """Record the local LitServe launch command without starting the server."""

    project_path: str = Field(..., description="Path to the project")
    server_path: str = Field(
        default="deployment/litserve/server.py",
        description="LitServe server entry point relative to the project.",
    )
    port: int = Field(default=8000, description="Port the command would bind")


class RecordLitserveMissingLiveEvidenceInput(BaseModel):
    """Record live deployment evidence intentionally missing from local preflight."""

    project_path: str = Field(..., description="Path to the project")


class DetectRuntimeEnvironmentInput(BaseModel):
    """Record local runtime environment for LitServe GPU deployment."""

    project_path: str = Field(..., description="Path to the project")


class DetectGpuCudaInput(BaseModel):
    """Detect GPU availability from observed runtime checks."""

    project_path: str = Field(..., description="Path to the project")


class SelectBestModelArtifactInput(BaseModel):
    """Select a model artifact for deployment or deterministic training comparison."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str | None = Field(
        default=None,
        description="Optional model artifact path relative to the project.",
    )
    latest_run: dict[str, Any] | None = Field(
        default=None,
        description="Latest run metadata with run_id, metrics, and candidate artifact evidence.",
    )
    baseline: dict[str, Any] | None = Field(
        default=None,
        description="Baseline metadata with metric value and selected artifact evidence.",
    )
    metric_name: str | None = Field(default=None, description="Metric used for comparison")
    metric_direction: str | None = Field(
        default=None,
        description="Metric direction: maximize or minimize",
    )
    threshold: float | None = Field(default=None, description="Required improvement threshold")
    tie_policy: str | None = Field(
        default=None,
        description="Tie policy: keep_baseline or select_latest",
    )


class RecordLitserveImageBuildSkippedInput(BaseModel):
    """Record that Docker image build is optional and skipped by default."""

    project_path: str = Field(..., description="Path to the project")


class StartLitserveServerInput(BaseModel):
    """Start LitServe server in the current project environment."""

    project_path: str = Field(..., description="Path to the project")
    server_path: str = Field(
        default="deployment/litserve/server.py",
        description="LitServe server entry point relative to the project.",
    )
    port: int = Field(default=8000, description="Server port")
    host: str = Field(default="127.0.0.1", description="Server host")
    log_path: str = Field(
        default="deployment/litserve/server.log",
        description="Server log path relative to the project.",
    )
    startup_wait_seconds: float = Field(default=2.0, description="Seconds to wait after start")


class TestLitserveHealthEndpointInput(BaseModel):
    """Call the LitServe health endpoint and record observed evidence."""

    project_path: str = Field(..., description="Path to the project")
    endpoint_url: str = Field(default="http://127.0.0.1:8000", description="Endpoint URL")
    timeout_seconds: float = Field(default=20.0, description="HTTP readiness timeout in seconds")


class TestLitservePredictionEndpointInput(BaseModel):
    """Call the LitServe prediction endpoint and record observed evidence."""

    project_path: str = Field(..., description="Path to the project")
    endpoint_url: str = Field(default="http://127.0.0.1:8000", description="Endpoint URL")
    sample_input: dict[str, Any] = Field(
        default_factory=lambda: {"input": [0.0]},
        description="Sample JSON payload for /predict.",
    )
    timeout_seconds: float = Field(default=20.0, description="HTTP readiness timeout in seconds")


class CaptureLitserveLogsAndEndpointInput(BaseModel):
    """Record LitServe logs and deployed endpoint URL."""

    project_path: str = Field(..., description="Path to the project")
    endpoint_url: str = Field(default="http://127.0.0.1:8000", description="Endpoint URL")
    log_path: str = Field(
        default="deployment/litserve/server.log",
        description="Server log path relative to the project.",
    )


class RecordLitserveGpuRollbackReadinessInput(BaseModel):
    """Record process cleanup and manual Lambda Cloud stop instruction."""

    project_path: str = Field(..., description="Path to the project")
    process_id: int | None = Field(default=None, description="LitServe server process ID")
    port: int = Field(default=8000, description="Server port")


# Gradio Tools
class CreateGradioInterfaceInput(BaseModel):
    """Create Gradio interface for model demo."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str = Field(..., description="Path to the model file")
    model_name: str = Field(..., description="Name for the model")
    interface_type: str = Field(
        default="image_classifier",
        description="Interface type: image_classifier, text_classifier, audio, custom",
    )
    title: str = Field(default="ML Model Demo", description="Interface title")
    description: str | None = Field(default=None, description="Interface description")
    examples: list[str] | None = Field(default=None, description="Example inputs")
    share: bool = Field(default=False, description="Create public share link")


class DeployToHuggingfaceInput(BaseModel):
    """Deploy Gradio app to Hugging Face Spaces."""

    project_path: str = Field(..., description="Path to the project")
    space_name: str = Field(..., description="Name for the HF Space")
    hf_token: str | None = Field(
        default=None, description="HF token (uses env var if not provided)"
    )
    private: bool = Field(default=False, description="Create private space")


# FastAPI + Lambda Tools
class CreateFastAPIAppInput(BaseModel):
    """Create FastAPI application for model serving."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str = Field(..., description="Path to the model file")
    model_name: str = Field(..., description="Name for the model")
    endpoint_type: str = Field(default="image", description="Endpoint type: image, text, json")
    title: str = Field(default="ML Inference API", description="API title")


class CreateLambdaDockerfileInput(BaseModel):
    """Create Dockerfile for AWS Lambda deployment."""

    project_path: str = Field(..., description="Path to the project")
    python_version: str = Field(default="3.11", description="Python version")
    model_file: str = Field(default="model.pt", description="Model file name")
    port: int = Field(default=8080, description="Container port")


class GenerateCDKStackInput(BaseModel):
    """Generate AWS CDK stack for Lambda deployment."""

    project_path: str = Field(..., description="Path to the project")
    stack_name: str = Field(..., description="CDK stack name")
    model_name: str = Field(..., description="Model name")
    memory_size: int = Field(default=1024, description="Lambda memory in MB")
    timeout: int = Field(default=30, description="Lambda timeout in seconds")
    stage: str = Field(default="prod", description="Deployment stage")


# TorchServe Tools
class CreateTorchserveHandlerInput(BaseModel):
    """Create TorchServe custom handler."""

    project_path: str = Field(..., description="Path to the project")
    model_path: str = Field(..., description="Path to the model file")
    model_name: str = Field(..., description="Name for the model")
    handler_type: str = Field(
        default="image_classifier",
        description="Handler type: image_classifier, text_classifier, object_detection",
    )


class CreateMARArchiveInput(BaseModel):
    """Create TorchServe MAR (Model Archive) file."""

    project_path: str = Field(..., description="Path to the project")
    model_name: str = Field(..., description="Model name")
    model_file: str = Field(..., description="Model file path")
    handler_file: str = Field(default="handler.py", description="Handler file path")
    version: str = Field(default="1.0", description="Model version")
    extra_files: list[str] | None = Field(default=None, description="Extra files to include")


class GenerateTorchserveConfigInput(BaseModel):
    """Generate TorchServe configuration."""

    project_path: str = Field(..., description="Path to the project")
    model_name: str = Field(..., description="Model name")
    inference_port: int = Field(default=8080, description="Inference API port")
    management_port: int = Field(default=8081, description="Management API port")
    metrics_port: int = Field(default=8082, description="Metrics port")
    workers: int = Field(default=1, description="Number of workers per model")


# KServe Tools
class CreateInferenceServiceYAMLInput(BaseModel):
    """Create KServe InferenceService YAML."""

    project_path: str = Field(..., description="Path to the project")
    service_name: str = Field(..., description="InferenceService name")
    model_name: str = Field(..., description="Model name")
    storage_uri: str = Field(..., description="Model storage URI (gs://, s3://, etc.)")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    runtime: str = Field(
        default="pytorch", description="Runtime: pytorch, tensorflow, sklearn, custom"
    )
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=5, description="Maximum replicas")


class GenerateKServeConfigInput(BaseModel):
    """Generate KServe configuration."""

    project_path: str = Field(..., description="Path to the project")
    service_name: str = Field(..., description="Service name")
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=5, description="Maximum replicas")
    target_utilization: int = Field(default=80, description="Target CPU utilization %")
    gpu_enabled: bool = Field(default=False, description="Enable GPU")
    gpu_count: int = Field(default=1, description="Number of GPUs")


# Kubernetes Manifests
class CreateK8sDeploymentInput(BaseModel):
    """Create Kubernetes Deployment YAML."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="Deployment name")
    image: str = Field(..., description="Container image")
    replicas: int = Field(default=1, description="Number of replicas")
    container_port: int = Field(default=8000, description="Container port")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    labels: dict[str, str] | None = Field(default=None, description="Pod labels")
    env: dict[str, str] | None = Field(default=None, description="Environment variables")
    resources: dict[str, Any] | None = Field(default=None, description="Resource requests/limits")


class CreateK8sServiceInput(BaseModel):
    """Create Kubernetes Service YAML."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="Service name")
    selector: dict[str, str] | None = Field(default=None, description="Selector labels")
    port: int = Field(default=80, description="Service port")
    target_port: int = Field(default=8000, description="Target port")
    service_type: str = Field(default="ClusterIP", description="Service type")
    namespace: str = Field(default="default", description="Kubernetes namespace")


class CreateK8sIngressInput(BaseModel):
    """Create Kubernetes Ingress YAML (ALB for EKS)."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="Ingress name")
    host: str = Field(..., description="Ingress host")
    service_name: str = Field(..., description="Service name")
    service_port: int = Field(default=80, description="Service port")
    path: str = Field(default="/", description="Ingress path")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    ingress_class: str = Field(default="alb", description="Ingress class")
    alb_scheme: str = Field(default="internet-facing", description="ALB scheme")
    certificate_arn: str | None = Field(default=None, description="ACM certificate ARN")
    annotations: dict[str, str] | None = Field(default=None, description="Extra annotations")


class CreateK8sHPAInput(BaseModel):
    """Create Kubernetes HPA YAML."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="HPA name")
    deployment_name: str = Field(..., description="Target deployment name")
    min_replicas: int = Field(default=1, description="Minimum replicas")
    max_replicas: int = Field(default=3, description="Maximum replicas")
    target_cpu_utilization: int = Field(default=70, description="Target CPU utilization %")
    namespace: str = Field(default="default", description="Kubernetes namespace")


class CreateK8sConfigMapInput(BaseModel):
    """Create Kubernetes ConfigMap YAML."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="ConfigMap name")
    data: dict[str, str] = Field(..., description="ConfigMap data")
    namespace: str = Field(default="default", description="Kubernetes namespace")


class CreateK8sSecretInput(BaseModel):
    """Create Kubernetes Secret YAML."""

    project_path: str = Field(..., description="Path to the project")
    name: str = Field(..., description="Secret name")
    data: dict[str, str] = Field(..., description="Secret data")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    encode: bool = Field(default=True, description="Base64 encode values")


class GenerateRollbackPlanInput(BaseModel):
    """Generate rollback plan for a deployment target."""

    project_path: str = Field(..., description="Path to the project")
    target: str = Field(..., description="Deployment target (kserve, lambda, docker, etc.)")
    deployment_name: str | None = Field(default=None, description="Deployment name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    error: str | None = Field(default=None, description="Failure reason")


# AWS Tools
class ListEKSClustersInput(BaseModel):
    """List EKS clusters."""

    region: str | None = Field(default=None, description="AWS region")


class UpdateKubeconfigInput(BaseModel):
    """Update kubeconfig for an EKS cluster."""

    cluster_name: str = Field(..., description="EKS cluster name")
    region: str | None = Field(default=None, description="AWS region")


class CreateECRRepoInput(BaseModel):
    """Create an ECR repository."""

    repo_name: str = Field(..., description="ECR repository name")
    region: str | None = Field(default=None, description="AWS region")
    scan_on_push: bool = Field(default=True, description="Enable image scan on push")
    mutable_tags: bool = Field(default=True, description="Allow mutable tags")


class GetECRLoginInput(BaseModel):
    """Get ECR login command."""

    region: str | None = Field(default=None, description="AWS region")


class GenerateIAMPolicyInput(BaseModel):
    """Generate IAM policy for deployment operations."""

    policy_name: str = Field(default="mlops-agent-policy", description="Policy name")
    services: list[str] | None = Field(
        default=None, description="AWS services to include (eks, ecr, lambda)"
    )


class EstimateDeploymentCostInput(BaseModel):
    """Estimate deployment cost based on basic usage inputs."""

    service_type: str = Field(default="lambda", description="Service type (lambda, eks)")
    requests_per_month: int = Field(default=1000000, description="Monthly request count")
    avg_duration_ms: int = Field(default=100, description="Average request duration in ms")
    memory_mb: int = Field(default=1024, description="Lambda memory size in MB")
    eks_node_hours: int = Field(default=720, description="EKS node hours per month")
    region: str | None = Field(default=None, description="AWS region")


class CreateHelmChartInput(BaseModel):
    """Create a Helm chart for Kubernetes deployment."""

    project_path: str = Field(..., description="Path to the project")
    chart_name: str = Field(..., description="Helm chart name")
    image: str = Field(..., description="Container image")
    chart_version: str = Field(default="0.1.0", description="Chart version")
    app_version: str = Field(default="1.0.0", description="Application version")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    container_port: int = Field(default=8000, description="Container port")
    service_port: int = Field(default=80, description="Service port")
    include_ingress: bool = Field(default=False, description="Include ingress template")
    include_hpa: bool = Field(default=False, description="Include HPA template")
    include_configmap: bool = Field(default=False, description="Include ConfigMap template")
    include_secret: bool = Field(default=False, description="Include Secret template")


class RollbackK8sDeploymentInput(BaseModel):
    """Rollback a Kubernetes deployment."""

    project_path: str = Field(..., description="Path to the project")
    deployment_name: str = Field(..., description="Deployment name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    dry_run: bool = Field(default=True, description="Return command without executing")


class RollbackLambdaStackInput(BaseModel):
    """Rollback an AWS Lambda CDK stack."""

    project_path: str = Field(..., description="Path to the project")
    stack_name: str = Field(..., description="CDK stack name")
    dry_run: bool = Field(default=True, description="Return command without executing")


class RollbackDeploymentInput(BaseModel):
    """Rollback a deployment based on target type."""

    project_path: str = Field(..., description="Path to the project")
    target: str = Field(..., description="Target type (kserve, lambda, docker)")
    deployment_name: str | None = Field(default=None, description="Deployment name")
    namespace: str = Field(default="default", description="Kubernetes namespace")
    stack_name: str | None = Field(default=None, description="CDK stack name")
    container_id: str | None = Field(default=None, description="Docker container ID")
    dry_run: bool = Field(default=True, description="Return command without executing")


# ============================================================================
# Tool Implementation Functions
# ============================================================================

# Hydra implementations are imported above and re-exported from this compatibility facade.

# --- MLflow Experiment Tracking Tools ---


def init_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    tags: dict[str, str] | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Initialize MLflow experiment."""
    try:
        import mlflow

        if project_path and tracking_uri is None:
            tracking_uri = str(Path(project_path) / "mlruns")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name, artifact_location=artifact_location, tags=tags
            )
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)

        return {
            "success": True,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "verification_results": [
                {
                    "check_name": "mlflow_experiment_exists",
                    "evidence_type": "declared",
                    "source_step": "initialize_mlflow_experiment",
                    "passed": True,
                    "evidence": f"MLflow experiment '{experiment_name}' is available.",
                }
            ],
            "message": f"Experiment '{experiment_name}' initialized (ID: {experiment_id})",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed. Run: pip install mlflow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def start_mlflow_run(
    experiment_name: str, run_name: str | None = None, tags: dict[str, str] | None = None
) -> dict[str, Any]:
    """Start a new MLflow run."""
    try:
        import mlflow

        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name, tags=tags)

        return {
            "success": True,
            "run_id": run.info.run_id,
            "run_name": run_name or run.info.run_name,
            "experiment_name": experiment_name,
            "artifact_uri": run.info.artifact_uri,
            "message": f"Started run {run.info.run_id}",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_params(params: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Log parameters to MLflow."""
    try:
        import mlflow

        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_params(params)
        else:
            mlflow.log_params(params)

        return {
            "success": True,
            "params_logged": list(params.keys()),
            "message": f"Logged {len(params)} parameters",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_metrics(
    metrics: dict[str, float], step: int | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Log metrics to MLflow."""
    try:
        import mlflow

        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics, step=step)
        else:
            mlflow.log_metrics(metrics, step=step)

        return {
            "success": True,
            "metrics_logged": metrics,
            "step": step,
            "message": f"Logged {len(metrics)} metrics",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_artifact(
    artifact_path: str, artifact_dest: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Log artifact to MLflow."""
    try:
        import mlflow

        path = Path(artifact_path)
        if not path.exists():
            return {"success": False, "error": f"Artifact path {artifact_path} does not exist"}

        if run_id:
            with mlflow.start_run(run_id=run_id):
                if path.is_dir():
                    mlflow.log_artifacts(artifact_path, artifact_dest)
                else:
                    mlflow.log_artifact(artifact_path, artifact_dest)
        else:
            if path.is_dir():
                mlflow.log_artifacts(artifact_path, artifact_dest)
            else:
                mlflow.log_artifact(artifact_path, artifact_dest)

        return {
            "success": True,
            "artifact_path": artifact_path,
            "message": f"Logged artifact from {artifact_path}",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_mlflow_model(
    model_path: str,
    model_name: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register model in MLflow Model Registry."""
    try:
        import mlflow

        model_uri = f"runs:/{run_id}/{model_path}" if run_id else model_path

        result = mlflow.register_model(model_uri, model_name, tags=tags)

        return {
            "success": True,
            "model_name": result.name,
            "model_version": result.version,
            "model_uri": model_uri,
            "message": f"Registered model '{model_name}' version {result.version}",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_best_mlflow_run(
    experiment_name: str, metric_name: str = "accuracy", maximize: bool = True
) -> dict[str, Any]:
    """Get best run from experiment based on metric."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}

        order = "DESC" if maximize else "ASC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1,
        )

        if not runs:
            return {"success": False, "error": "No runs found in experiment"}

        best_run = runs[0]

        return {
            "success": True,
            "run_id": best_run.info.run_id,
            "run_name": best_run.info.run_name,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "best_metric": {metric_name: best_run.data.metrics.get(metric_name)},
            "artifact_uri": best_run.info.artifact_uri,
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def end_mlflow_run(run_id: str | None = None, status: str = "FINISHED") -> dict[str, Any]:
    """End an MLflow run."""
    try:
        import mlflow

        mlflow.end_run(status=status)

        return {"success": True, "status": status, "message": f"Run ended with status {status}"}
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- DVC Data Versioning Tools ---


def init_dvc_repo(project_path: str, no_scm: bool = False) -> dict[str, Any]:
    """Initialize DVC in a repository."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    dvc_config = path / ".dvc" / "config"
    if dvc_config.exists():
        return {
            "success": True,
            "project_path": project_path,
            "dvc_dir": str(path / ".dvc"),
            "verification_results": [
                {
                    "check_name": "dvc_repo_exists",
                    "evidence_type": "observed",
                    "source_step": "initialize_dvc",
                    "passed": True,
                    "evidence": "DVC metadata already exists at .dvc/config.",
                }
            ],
            "message": "DVC already initialized",
        }

    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed. Run: pip install dvc"}

    cmd = ["dvc", "init"]
    if no_scm:
        cmd.append("--no-scm")

    result = run_command(cmd, cwd=project_path)

    if result["success"]:
        return {
            "success": True,
            "project_path": project_path,
            "dvc_dir": str(path / ".dvc"),
            "verification_results": [
                {
                    "check_name": "dvc_repo_exists",
                    "evidence_type": "observed",
                    "source_step": "initialize_dvc",
                    "passed": True,
                    "evidence": "DVC metadata initialized at .dvc/config.",
                }
            ],
            "message": "DVC initialized successfully",
        }

    return result


def configure_dvc_remote(
    project_path: str, remote_name: str = "storage", remote_url: str = None, default: bool = True
) -> dict[str, Any]:
    """Configure DVC remote storage."""
    if not remote_url:
        return {
            "success": True,
            "skipped": True,
            "reason": "No DVC remote URL requested for local setup.",
        }

    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed"}

    # Add remote
    cmd = ["dvc", "remote", "add"]
    if default:
        cmd.append("-d")
    cmd.extend([remote_name, remote_url])

    result = run_command(cmd, cwd=project_path)

    if not result["success"]:
        # Remote might already exist, try modifying
        cmd = ["dvc", "remote", "modify", remote_name, "url", remote_url]
        result = run_command(cmd, cwd=project_path)

    if result["success"]:
        return {
            "success": True,
            "remote_name": remote_name,
            "remote_url": remote_url,
            "is_default": default,
            "message": f"DVC remote '{remote_name}' configured with URL: {remote_url}",
        }

    return result


def add_data_to_dvc(project_path: str, data_path: str) -> dict[str, Any]:
    """Add data to DVC tracking."""
    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed"}

    full_path = Path(project_path) / data_path
    if not full_path.exists():
        return {"success": False, "error": f"Data path {full_path} does not exist"}

    result = run_command(["dvc", "add", data_path], cwd=project_path)

    if result["success"]:
        dvc_file = f"{data_path}.dvc"
        return {
            "success": True,
            "data_path": data_path,
            "dvc_file": dvc_file,
            "message": f"Data added to DVC. Created {dvc_file}",
        }

    return result


def create_dvc_pipeline(
    project_path: str, stages: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create DVC pipeline (dvc.yaml)."""
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    if stages is None:
        train_script = "src/train.py" if (path / "src" / "train.py").exists() else "train.py"
        stages = [
            {
                "name": "train",
                "cmd": f"python {train_script}",
                "deps": [train_script],
            }
        ]

    # Convert stages to DVC format
    dvc_config = {"stages": {}}

    for stage in stages:
        stage_name = stage.get("name", f"stage_{len(dvc_config['stages'])}")
        stage_config = {}

        if "cmd" in stage:
            stage_config["cmd"] = stage["cmd"]
        if "deps" in stage:
            stage_config["deps"] = stage["deps"]
        if "outs" in stage:
            stage_config["outs"] = stage["outs"]
        if "params" in stage:
            stage_config["params"] = stage["params"]
        if "metrics" in stage:
            stage_config["metrics"] = stage["metrics"]
        if "plots" in stage:
            stage_config["plots"] = stage["plots"]

        dvc_config["stages"][stage_name] = stage_config

    dvc_yaml_path = path / "dvc.yaml"
    with open(dvc_yaml_path, "w") as f:
        yaml.dump(dvc_config, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "dvc_yaml_path": str(dvc_yaml_path),
        "stages": list(dvc_config["stages"].keys()),
        "verification_results": [
            {
                "check_name": "dvc_yaml_parseable",
                "evidence_type": "declared",
                "source_step": "create_dvc_yaml",
                "passed": True,
                "evidence": "dvc.yaml generated with at least one pipeline stage.",
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "pipeline_definition",
                    "producing_step": "create_dvc_yaml",
                    "state": "generated",
                    "path": relative_to_project(project_path, dvc_yaml_path),
                }
            ]
        },
        "message": f"DVC pipeline created with {len(stages)} stages",
    }


def dvc_push(project_path: str, remote_name: str | None = None) -> dict[str, Any]:
    """Push data to DVC remote."""
    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed"}

    cmd = ["dvc", "push"]
    if remote_name:
        cmd.extend(["-r", remote_name])

    result = run_command(cmd, cwd=project_path, timeout=300)

    if result["success"]:
        return {
            "success": True,
            "remote": remote_name or "default",
            "message": "Data pushed to remote successfully",
            "output": result["stdout"],
        }

    return result


def dvc_pull(project_path: str, remote_name: str | None = None) -> dict[str, Any]:
    """Pull data from DVC remote."""
    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed"}

    cmd = ["dvc", "pull"]
    if remote_name:
        cmd.extend(["-r", remote_name])

    result = run_command(cmd, cwd=project_path, timeout=300)

    if result["success"]:
        return {
            "success": True,
            "remote": remote_name or "default",
            "message": "Data pulled from remote successfully",
            "output": result["stdout"],
        }

    return result


def dvc_reproduce(
    project_path: str, stages: list[str] | None = None, force: bool = False
) -> dict[str, Any]:
    """Reproduce DVC pipeline."""
    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed"}

    cmd = ["dvc", "repro"]
    if force:
        cmd.append("-f")
    if stages:
        cmd.extend(stages)

    result = run_command(cmd, cwd=project_path, timeout=3600)

    if result["success"]:
        return {
            "success": True,
            "stages": stages or "all",
            "message": "Pipeline reproduced successfully",
            "output": result["stdout"],
        }

    return result


# --- Docker Tools ---


def create_ml_dockerfile(
    project_path: str,
    base_image: str = "python:3.11-slim",
    cuda_version: str | None = None,
    entry_point: str = "train.py",
    requirements_file: str = "requirements.txt",
    expose_port: int | None = None,
) -> dict[str, Any]:
    """Create Dockerfile for ML project."""
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    # Use CUDA base image if specified
    if cuda_version:
        base_image = f"nvidia/cuda:{cuda_version}-runtime-ubuntu22.04"

    dockerfile_content = f"""# MLOps Agent Generated Dockerfile
FROM {base_image}

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

"""

    if cuda_version:
        dockerfile_content += """# Install Python (for CUDA images)
RUN apt-get update && apt-get install -y \\
    python3 python3-pip python3-dev \\
    && rm -rf /var/lib/apt/lists/* \\
    && ln -s /usr/bin/python3 /usr/bin/python

"""

    dockerfile_content += f"""# Install dependencies
COPY {requirements_file} .
RUN pip install --no-cache-dir -r {requirements_file}

# Install additional MLOps tools
RUN pip install --no-cache-dir mlflow hydra-core dvc

# Copy project files
COPY . .

"""

    if expose_port:
        dockerfile_content += f"""# Expose port
EXPOSE {expose_port}

"""

    dockerfile_content += f"""# Set entry point
CMD ["python", "{entry_point}"]
"""

    dockerfile_path = path / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    # Create .dockerignore
    dockerignore_content = """# MLOps Agent Generated .dockerignore
__pycache__/
*.pyc
*.pyo
.git/
.gitignore
.dvc/
*.dvc
.venv/
venv/
.env
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
mlruns/
outputs/
*.log
"""

    dockerignore_path = path / ".dockerignore"
    with open(dockerignore_path, "w") as f:
        f.write(dockerignore_content)

    return {
        "success": True,
        "dockerfile_path": str(dockerfile_path),
        "dockerignore_path": str(dockerignore_path),
        "base_image": base_image,
        "entry_point": entry_point,
        "verification_results": [
            {
                "check_name": "dockerfile_build_evidence",
                "evidence_type": "declared",
                "source_step": "create_dockerfile",
                "passed": True,
                "evidence": "Dockerfile generated for local build validation.",
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "container_definition",
                    "producing_step": "create_dockerfile",
                    "state": "generated",
                    "path": relative_to_project(project_path, dockerfile_path),
                }
            ]
        },
        "message": f"Dockerfile created at {dockerfile_path}",
    }


def build_ml_docker_image(
    project_path: str, image_name: str, tag: str = "latest", dockerfile: str = "Dockerfile"
) -> dict[str, Any]:
    """Build Docker image for ML project."""
    if not check_tool_installed("docker"):
        return {"success": False, "error": "Docker not installed"}

    path = Path(project_path)
    if not (path / dockerfile).exists():
        return {"success": False, "error": f"Dockerfile not found at {path / dockerfile}"}

    full_image_name = f"{image_name}:{tag}"

    result = run_command(
        ["docker", "build", "-t", full_image_name, "-f", dockerfile, "."],
        cwd=project_path,
        timeout=600,
    )

    if result["success"]:
        return {
            "success": True,
            "image_name": full_image_name,
            "message": f"Successfully built image {full_image_name}",
        }

    return result


def run_training_container(
    image_name: str,
    tag: str = "latest",
    gpu: bool = False,
    volumes: dict[str, str] | None = None,
    env_vars: dict[str, str] | None = None,
    command: str | None = None,
) -> dict[str, Any]:
    """Run training in Docker container."""
    if not check_tool_installed("docker"):
        return {"success": False, "error": "Docker not installed"}

    full_image_name = f"{image_name}:{tag}"
    cmd = ["docker", "run", "-d"]

    if gpu:
        cmd.extend(["--gpus", "all"])

    if volumes:
        for host_path, container_path in volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])

    if env_vars:
        for key, value in env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

    cmd.append(full_image_name)

    if command:
        cmd.extend(command.split())

    result = run_command(cmd)

    if result["success"]:
        container_id = result["stdout"].strip()
        return {
            "success": True,
            "container_id": container_id,
            "image": full_image_name,
            "gpu_enabled": gpu,
            "message": f"Container started: {container_id[:12]}",
        }

    return result


def push_docker_image(
    image_name: str, tag: str = "latest", registry: str | None = None
) -> dict[str, Any]:
    """Push Docker image to registry."""
    if not check_tool_installed("docker"):
        return {"success": False, "error": "Docker not installed"}

    full_image_name = f"{image_name}:{tag}"

    if registry:
        remote_name = f"{registry}/{full_image_name}"
        # Tag for remote registry
        tag_result = run_command(["docker", "tag", full_image_name, remote_name])
        if not tag_result["success"]:
            return tag_result
        full_image_name = remote_name

    result = run_command(["docker", "push", full_image_name], timeout=600)

    if result["success"]:
        return {
            "success": True,
            "image": full_image_name,
            "message": f"Successfully pushed {full_image_name}",
        }

    return result


# --- GitHub Actions Tools ---


def create_github_workflow(
    project_path: str,
    workflow_name: str = "ml-pipeline",
    trigger_on: list[str] = None,
    python_version: str = "3.11",
    use_dvc: bool = True,
    use_mlflow: bool = True,
    accuracy_threshold: float | None = None,
) -> dict[str, Any]:
    """Create GitHub Actions workflow for ML pipeline."""
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    trigger_on = trigger_on or ["push", "workflow_dispatch"]

    workflow = {
        "name": "ML Training Pipeline",
        "on": {},
        "env": {"PYTHON_VERSION": python_version},
        "jobs": {"train": {"runs-on": "ubuntu-latest", "steps": []}},
    }

    # Configure triggers
    for trigger in trigger_on:
        if trigger == "push":
            workflow["on"]["push"] = {"branches": ["main", "master"]}
        elif trigger == "pull_request":
            workflow["on"]["pull_request"] = {"branches": ["main", "master"]}
        elif trigger == "workflow_dispatch":
            workflow["on"]["workflow_dispatch"] = {
                "inputs": {
                    "accuracy_threshold": {
                        "description": "Minimum accuracy threshold",
                        "required": False,
                        "default": str(accuracy_threshold or 0.85),
                    }
                }
            }

    steps = workflow["jobs"]["train"]["steps"]

    # Checkout
    steps.append({"name": "Checkout repository", "uses": "actions/checkout@v4"})

    # Setup Python
    steps.append(
        {
            "name": "Set up Python",
            "uses": "actions/setup-python@v5",
            "with": {"python-version": "${{ env.PYTHON_VERSION }}", "cache": "pip"},
        }
    )

    # Install dependencies
    steps.append({"name": "Install dependencies", "run": "pip install -r requirements.txt"})

    # DVC setup
    if use_dvc:
        steps.append({"name": "Setup DVC", "uses": "iterative/setup-dvc@v1"})
        steps.append(
            {
                "name": "Pull data from DVC",
                "run": "dvc pull",
                "env": {
                    "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                    "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_SECRET_ACCESS_KEY }}",
                },
            }
        )

    # MLflow setup
    if use_mlflow:
        workflow["env"]["MLFLOW_TRACKING_URI"] = "${{ secrets.MLFLOW_TRACKING_URI }}"

    # Training step
    train_step = {"name": "Run training", "id": "train", "run": "python train.py"}
    steps.append(train_step)

    # Accuracy check
    if accuracy_threshold:
        steps.append(
            {
                "name": "Check accuracy threshold",
                "run": f"""
ACCURACY=$(cat metrics.json | jq -r '.accuracy')
THRESHOLD=${{{{ github.event.inputs.accuracy_threshold || '{accuracy_threshold}' }}}}
if (( $(echo "$ACCURACY >= $THRESHOLD" | bc -l) )); then
  echo "✅ Accuracy $ACCURACY meets threshold $THRESHOLD"
else
  echo "❌ Accuracy $ACCURACY below threshold $THRESHOLD"
  exit 1
fi
""",
            }
        )

    # Upload artifacts
    steps.append(
        {
            "name": "Upload model artifacts",
            "uses": "actions/upload-artifact@v4",
            "with": {"name": "model", "path": "models/"},
        }
    )

    # Create workflow directory and file
    workflows_dir = ensure_directory(path / ".github" / "workflows")
    workflow_path = workflows_dir / f"{workflow_name}.yml"

    with open(workflow_path, "w") as f:
        yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "workflow_path": str(workflow_path),
        "workflow_name": workflow_name,
        "triggers": trigger_on,
        "features": {
            "dvc": use_dvc,
            "mlflow": use_mlflow,
            "accuracy_threshold": accuracy_threshold,
        },
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "automation_workflow",
                    "producing_step": "create_ci_workflow",
                    "state": "generated",
                    "path": relative_to_project(project_path, workflow_path),
                }
            ]
        },
        "message": f"GitHub Actions workflow created at {workflow_path}",
    }


def add_workflow_step(
    project_path: str,
    workflow_file: str = ".github/workflows/ml-pipeline.yml",
    job_name: str = "train",
    step: dict[str, Any] = None,
) -> dict[str, Any]:
    """Add step to existing GitHub workflow."""
    workflow_path = Path(project_path) / workflow_file

    if not workflow_path.exists():
        return {"success": False, "error": f"Workflow file {workflow_path} does not exist"}

    try:
        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        if job_name not in workflow.get("jobs", {}):
            return {"success": False, "error": f"Job '{job_name}' not found in workflow"}

        workflow["jobs"][job_name]["steps"].append(step)

        with open(workflow_path, "w") as f:
            yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)

        return {
            "success": True,
            "workflow_path": str(workflow_path),
            "step_added": step.get("name", "unnamed"),
            "message": f"Step added to job '{job_name}'",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Training Control Tools ---


CAPSTONE_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def detect_capstone_data_layouts(
    project_path: str,
    dataset_1_path: str,
    dataset_2_path: str,
    completion_mode: str = "local_ready",
    test_size: float = 0.2,
    split_seed: int = 42,
) -> dict[str, Any]:
    """Detect two capstone image-folder datasets without mutating files or DVC state."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}

    datasets = [
        _detect_capstone_image_folder_dataset("dataset_1", dataset_1_path),
        _detect_capstone_image_folder_dataset("dataset_2", dataset_2_path),
    ]
    paths_passed = all(dataset["path_exists"] and dataset["is_directory"] for dataset in datasets)
    layouts_passed = all(dataset["status"] == "succeeded" for dataset in datasets)
    workflow_status = "succeeded" if paths_passed and layouts_passed else "blocked"
    split_manifest_writes_required = layouts_passed and any(
        dataset["layout"] == "class_folders" for dataset in datasets
    )
    split_plans = [
        _capstone_split_plan(path, dataset, float(test_size), int(split_seed))
        for dataset in datasets
        if dataset["status"] == "succeeded" and dataset["layout"] == "class_folders"
    ]
    blocked_dataset_ids = [
        dataset["dataset_id"] for dataset in datasets if dataset["status"] != "succeeded"
    ]
    next_actions = [
        action
        for dataset in datasets
        for action in dataset["next_actions"]
    ]
    artifact_entries = [
        {
            "artifact_type": "capstone_source_dataset",
            "producing_step": "prepare_capstone_data_contract",
            "state": "external",
            "path": dataset["source_path"],
            "metadata": {
                "dataset_id": dataset["dataset_id"],
                "status": dataset["status"],
                "layout": dataset["layout"],
                "class_count": dataset["class_count"],
                "total_image_count": dataset["total_image_count"],
                "source_path_kind": "user_provided_local_or_mounted",
            },
        }
        for dataset in datasets
        if dataset["path_exists"] and dataset["is_directory"]
    ]
    dataset_evidence = {
        "completion_mode": completion_mode,
        "workflow_status": workflow_status,
        "dataset_count": len(datasets),
        "blocked_dataset_ids": blocked_dataset_ids,
        "split_plans": split_plans,
        "datasets": datasets,
    }

    return {
        "success": True,
        "status": workflow_status,
        "completion_mode": completion_mode,
        "datasets": datasets,
        "split_manifest_writes_required": split_manifest_writes_required,
        "split_plans": split_plans,
        "split_seed": int(split_seed),
        "test_size": float(test_size),
        "missing_inputs": sorted(
            {
                missing_input
                for dataset in datasets
                for missing_input in dataset["missing_inputs"]
            }
        ),
        "next_actions": next_actions,
        "verification_results": [
            {
                "check_name": "two_dataset_paths_provided",
                "evidence_type": "observed",
                "source_step": "prepare_capstone_data_contract",
                "passed": paths_passed,
                "evidence": json.dumps(
                    {
                        "dataset_paths": [
                            {
                                "dataset_id": dataset["dataset_id"],
                                "source_path": dataset["source_path"],
                                "path_exists": dataset["path_exists"],
                                "is_directory": dataset["is_directory"],
                                "missing_inputs": dataset["missing_inputs"],
                            }
                            for dataset in datasets
                        ]
                    },
                    sort_keys=True,
                ),
            },
            {
                "check_name": "two_dataset_layouts_supported",
                "evidence_type": "observed",
                "source_step": "prepare_capstone_data_contract",
                "passed": layouts_passed,
                "evidence": json.dumps(dataset_evidence, sort_keys=True),
            },
        ],
        "artifact_manifest": {"entries": artifact_entries},
    }


def _detect_capstone_image_folder_dataset(dataset_id: str, source_path: str) -> dict[str, Any]:
    path = Path(source_path).expanduser()
    record = {
        "dataset_id": dataset_id,
        "status": "blocked",
        "source_path": str(path),
        "layout": "unknown",
        "existing_train_path": None,
        "existing_test_path": None,
        "class_names": [],
        "class_count": 0,
        "per_class_counts": {},
        "split_counts": {},
        "total_image_count": 0,
        "path_exists": path.exists(),
        "is_directory": path.is_dir(),
        "missing_inputs": [],
        "next_actions": [],
        "blocked_reason": None,
    }
    input_name = f"{dataset_id}_path"
    if not source_path:
        record["missing_inputs"].append(input_name)
        record["blocked_reason"] = "missing_dataset_path"
        record["next_actions"].append(f"Provide {input_name} as a local or mounted dataset path.")
        return record
    if not path.exists():
        record["missing_inputs"].append(input_name)
        record["blocked_reason"] = "missing_dataset_path"
        record["next_actions"].append(f"Provide an existing local or mounted path for {input_name}.")
        return record
    if not path.is_dir():
        record["blocked_reason"] = "unsupported_not_directory"
        record["next_actions"].append(
            f"Provide {input_name} as a directory with class-labelled image subdirectories."
        )
        return record

    direct_images = _direct_image_files(path)
    train_dir = _case_insensitive_child_dir(path, "train")
    test_dir = _case_insensitive_child_dir(path, "test")
    if bool(train_dir) != bool(test_dir):
        record["layout"] = "ambiguous_partial_train_test_split"
        record["blocked_reason"] = "ambiguous_layout"
        record["next_actions"].append(
            "Provide both train/ and test/ folders, or provide one unsplit class-folder dataset."
        )
        return record
    if train_dir is not None and test_dir is not None:
        return _detect_existing_train_test_dataset(record, train_dir, test_dir)
    if direct_images:
        record["layout"] = "unsupported_root_images"
        record["blocked_reason"] = "unsupported_layout"
        record["next_actions"].append(
            "Move root-level images under class-labelled subdirectories before detection."
        )
        return record
    return _detect_unsplit_class_folder_dataset(record, path)


def _detect_existing_train_test_dataset(
    record: dict[str, Any],
    train_dir: Path,
    test_dir: Path,
) -> dict[str, Any]:
    train_summary = _class_folder_summary(train_dir)
    test_summary = _class_folder_summary(test_dir)
    record["layout"] = "existing_train_test_split"
    record["existing_train_path"] = str(train_dir)
    record["existing_test_path"] = str(test_dir)
    record["split_counts"] = {
        "train": train_summary["per_class_counts"],
        "test": test_summary["per_class_counts"],
    }
    train_classes = set(train_summary["class_names"])
    test_classes = set(test_summary["class_names"])
    if not train_summary["supported"] or not test_summary["supported"]:
        record["blocked_reason"] = "unsupported_train_test_layout"
        record["missing_inputs"].extend(train_summary["missing_inputs"])
        record["missing_inputs"].extend(test_summary["missing_inputs"])
        record["next_actions"].extend(train_summary["next_actions"])
        record["next_actions"].extend(test_summary["next_actions"])
        return record
    if train_classes != test_classes:
        record["blocked_reason"] = "ambiguous_split_class_mismatch"
        record["next_actions"].append(
            "Ensure train/ and test/ contain the same class-labelled subdirectories."
        )
        return record

    record["status"] = "succeeded"
    record["class_names"] = sorted(train_classes)
    record["class_count"] = len(record["class_names"])
    record["per_class_counts"] = {
        class_name: train_summary["per_class_counts"].get(class_name, 0)
        + test_summary["per_class_counts"].get(class_name, 0)
        for class_name in record["class_names"]
    }
    record["total_image_count"] = sum(record["per_class_counts"].values())
    return record


def generate_capstone_split_manifests(
    project_path: str,
    capstone_data_detection: dict[str, Any],
    test_size: float = 0.2,
    split_seed: int = 42,
    materialize_splits: bool = False,
) -> dict[str, Any]:
    """Write deterministic split manifests for supported capstone datasets only."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}
    if not isinstance(capstone_data_detection, dict):
        return {"success": False, "error": "capstone_data_detection must be structured data"}
    if capstone_data_detection.get("status") != "succeeded":
        evidence = {
            "blocked_reason": "unsupported_or_empty_dataset_evidence",
            "detection_status": capstone_data_detection.get("status"),
            "blocked_dataset_ids": capstone_data_detection.get("blocked_dataset_ids", []),
        }
        return {
            "success": True,
            "status": "blocked",
            "split_manifests": [],
            "missing_inputs": ["supported_capstone_dataset_evidence"],
            "next_actions": [
                "Resolve blocked dataset layout evidence before writing split manifests."
            ],
            "verification_results": [
                _capstone_split_verification_result(
                    "split_evidence_recorded",
                    False,
                    evidence,
                ),
                _capstone_split_verification_result(
                    "dataset_lineage_artifacts_reported",
                    False,
                    evidence,
                    evidence_type="declared",
                ),
            ],
            "artifact_manifest": {"entries": []},
        }
    if not 0 < float(test_size) < 1:
        return {
            "success": True,
            "status": "blocked",
            "split_manifests": [],
            "missing_inputs": ["test_size"],
            "next_actions": ["Provide test_size as a float greater than 0 and less than 1."],
            "verification_results": [
                _capstone_split_verification_result(
                    "split_evidence_recorded",
                    False,
                    {"blocked_reason": "invalid_test_size", "test_size": test_size},
                )
            ],
            "artifact_manifest": {"entries": []},
        }

    split_records: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []
    for dataset in capstone_data_detection.get("datasets", []):
        if dataset.get("status") != "succeeded":
            continue
        if dataset.get("layout") == "existing_train_test_split":
            split_records.append(_existing_split_record(dataset, split_seed, float(test_size)))
            continue
        if dataset.get("layout") != "class_folders":
            continue

        manifest = _build_split_manifest(dataset, float(test_size), int(split_seed))
        manifest_path = (
            path
            / "data"
            / "capstone"
            / dataset["dataset_id"]
            / "split_manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        relative_manifest_path = relative_to_project(str(path), manifest_path)
        split_record = {
            "dataset_id": dataset["dataset_id"],
            "source_path": dataset["source_path"],
            "split_strategy": "manifest",
            "seed": int(split_seed),
            "test_size": float(test_size),
            "train_count": manifest["train_count"],
            "test_count": manifest["test_count"],
            "per_class_counts": manifest["per_class_counts"],
            "split_manifest_path": relative_manifest_path,
            "materialized_train_path": None,
            "materialized_test_path": None,
        }
        split_records.append(split_record)
        artifact_entries.append(
            {
                "artifact_type": "split_manifest",
                "producing_step": "generate_split_manifests",
                "state": "generated",
                "path": relative_manifest_path,
                "checksum": _sha256_file(manifest_path),
                "metadata": {
                    "dataset_id": dataset["dataset_id"],
                    "source_path": dataset["source_path"],
                    "split_strategy": "manifest",
                    "seed": int(split_seed),
                    "test_size": float(test_size),
                    "train_count": manifest["train_count"],
                    "test_count": manifest["test_count"],
                    "per_class_counts": manifest["per_class_counts"],
                },
            }
        )

    materialization_block = (
        {
            "requested": True,
            "status": "blocked",
            "next_action": (
                "Materialized split folders require a separate approval-gated copied-folder "
                "step; no folders were created."
            ),
        }
        if materialize_splits
        else {"requested": False, "status": "not_requested"}
    )
    status = "blocked" if materialize_splits else "succeeded"
    evidence = {
        "split_records": split_records,
        "materialization": materialization_block,
    }
    next_actions = []
    if materialize_splits:
        next_actions.append(materialization_block["next_action"])
    return {
        "success": True,
        "status": status,
        "split_manifests": split_records,
        "materialization": materialization_block,
        "missing_inputs": ["materialized_split_folder_approval"] if materialize_splits else [],
        "next_actions": next_actions,
        "verification_results": [
            _capstone_split_verification_result(
                "split_evidence_recorded",
                bool(split_records) and not materialize_splits,
                evidence,
            ),
            _capstone_split_verification_result(
                "dataset_lineage_artifacts_reported",
                bool(split_records),
                evidence,
                evidence_type="observed" if artifact_entries else "declared",
            ),
        ],
        "artifact_manifest": {"entries": artifact_entries},
    }


def track_capstone_data_package(
    project_path: str,
    capstone_split_manifest_result: dict[str, Any],
    initialize_if_missing: bool = True,
) -> dict[str, Any]:
    """Validate/init local DVC and track generated capstone package paths."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}
    if not isinstance(capstone_split_manifest_result, dict):
        return {"success": False, "error": "capstone_split_manifest_result must be structured data"}

    split_records = capstone_split_manifest_result.get("split_manifests", [])
    if capstone_split_manifest_result.get("status") != "succeeded" or not split_records:
        evidence = {
            "blocked_reason": "missing_generated_split_manifest_evidence",
            "split_status": capstone_split_manifest_result.get("status"),
        }
        return _capstone_dvc_tracking_result(
            status="blocked",
            dvc_repo={"status": "not_checked", "path": str(path)},
            tracked_package_paths=[],
            dvc_tracking_files=[],
            artifact_entries=[],
            next_actions=["Generate approved split manifests before DVC tracking."],
            dvc_repo_passed=False,
            package_tracking_passed=False,
            evidence=evidence,
        )

    package_paths = _capstone_package_paths_from_split_records(path, split_records)
    if not package_paths:
        evidence = {
            "blocked_reason": "no_generated_capstone_package_paths",
            "split_records": split_records,
        }
        return _capstone_dvc_tracking_result(
            status="blocked",
            dvc_repo={"status": "not_checked", "path": str(path)},
            tracked_package_paths=[],
            dvc_tracking_files=[],
            artifact_entries=[],
            next_actions=[
                "Generate split manifests under data/capstone/<dataset_id>/ before DVC tracking."
            ],
            dvc_repo_passed=False,
            package_tracking_passed=False,
            evidence=evidence,
        )

    if not check_tool_installed("dvc"):
        evidence = {"blocked_reason": "missing_dvc_executable", "project_path": str(path)}
        return _capstone_dvc_tracking_result(
            status="blocked",
            dvc_repo={"status": "missing_executable", "path": str(path)},
            tracked_package_paths=[],
            dvc_tracking_files=[],
            artifact_entries=_capstone_lineage_artifacts(path, split_records, []),
            next_actions=["Install DVC in the project environment before tracking data packages."],
            dvc_repo_passed=False,
            package_tracking_passed=False,
            evidence=evidence,
        )

    dvc_repo, repo_failure = _ensure_local_dvc_repo(path, initialize_if_missing)
    if repo_failure is not None:
        artifact_entries = _capstone_lineage_artifacts(path, split_records, [])
        return _capstone_dvc_tracking_result(
            status="failed",
            dvc_repo=dvc_repo,
            tracked_package_paths=[],
            dvc_tracking_files=[],
            artifact_entries=artifact_entries,
            next_actions=[repo_failure],
            dvc_repo_passed=False,
            package_tracking_passed=False,
            evidence={"failure_reason": repo_failure, "dvc_repo": dvc_repo},
        )

    tracked_package_paths: list[str] = []
    dvc_tracking_files: list[str] = []
    add_failures: list[dict[str, Any]] = []
    for package_path in package_paths:
        add_result = run_command(["dvc", "add", package_path], cwd=str(path), timeout=300)
        if not add_result.get("success"):
            add_failures.append({"package_path": package_path, "result": add_result})
            continue
        tracked_package_paths.append(package_path)
        dvc_file = f"{package_path}.dvc"
        dvc_tracking_files.append(dvc_file)

    artifact_entries = _capstone_lineage_artifacts(
        path,
        split_records,
        tracked_package_paths,
        dvc_tracking_files,
        dvc_repo,
    )
    if add_failures:
        return _capstone_dvc_tracking_result(
            status="failed",
            dvc_repo=dvc_repo,
            tracked_package_paths=tracked_package_paths,
            dvc_tracking_files=dvc_tracking_files,
            artifact_entries=artifact_entries,
            next_actions=["Resolve local DVC add failures before continuing."],
            dvc_repo_passed=True,
            package_tracking_passed=False,
            evidence={"add_failures": add_failures, "dvc_repo": dvc_repo},
        )

    evidence = {
        "dvc_repo": dvc_repo,
        "tracked_package_paths": tracked_package_paths,
        "dvc_tracking_files": dvc_tracking_files,
    }
    return _capstone_dvc_tracking_result(
        status="succeeded",
        dvc_repo=dvc_repo,
        tracked_package_paths=tracked_package_paths,
        dvc_tracking_files=dvc_tracking_files,
        artifact_entries=artifact_entries,
        next_actions=[],
        dvc_repo_passed=True,
        package_tracking_passed=bool(tracked_package_paths),
        evidence=evidence,
    )


def configure_validate_capstone_dvc_remote(
    project_path: str,
    completion_mode: str = "local_ready",
    remote_name: str = "capstone",
    remote_url: str | None = None,
    default: bool = True,
    source_step: str = "configure_validate_dvc_remote",
) -> dict[str, Any]:
    """Configure or validate a local/S3 DVC remote without pushing or pulling data."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}
    if completion_mode not in {"local_ready", "capstone_complete"}:
        return {"success": False, "error": f"Unsupported completion_mode: {completion_mode}"}

    if not check_tool_installed("dvc"):
        remote = _capstone_remote_record(remote_name, remote_url, "missing")
        evidence = {
            "blocked_reason": "missing_dvc_executable",
            "remote": remote,
            "credential_capability": None,
        }
        return _capstone_remote_validation_result(
            status="blocked",
            remote=remote,
            verification_results=_capstone_remote_verification_results(
                completion_mode, source_step, False, evidence
            ),
            artifact_entries=[],
            next_actions=["Install DVC before configuring or validating capstone remotes."],
        )

    dvc_config = path / ".dvc" / "config"
    if not dvc_config.exists():
        remote = _capstone_remote_record(remote_name, remote_url, "missing")
        evidence = {
            "blocked_reason": "missing_dvc_repo",
            "remote": remote,
            "credential_capability": None,
        }
        return _capstone_remote_validation_result(
            status="blocked",
            remote=remote,
            verification_results=_capstone_remote_verification_results(
                completion_mode, source_step, False, evidence
            ),
            artifact_entries=[],
            next_actions=["Validate or initialize DVC metadata before remote validation."],
        )

    configure_result: dict[str, Any] | None = None
    if remote_url:
        configure_result = _configure_capstone_dvc_remote(
            path,
            remote_name,
            remote_url,
            default,
        )
        if not configure_result.get("success"):
            remote = _capstone_remote_record(remote_name, remote_url, _remote_type(remote_url))
            evidence = {
                "blocked_reason": "invalid_remote_configuration",
                "remote": remote,
                "dvc_command_status": {
                    "success": False,
                    "returncode": configure_result.get("returncode"),
                },
                "credential_capability": None,
            }
            return _capstone_remote_validation_result(
                status="failed",
                remote=remote,
                verification_results=_capstone_remote_verification_results(
                    completion_mode, source_step, False, evidence
                ),
                artifact_entries=[],
                next_actions=[
                    "Fix the DVC remote URL or existing DVC remote configuration and rerun."
                ],
            )

    resolved_url = remote_url or _read_dvc_remote_url(path, remote_name)
    if not resolved_url:
        remote = _capstone_remote_record(remote_name, None, "missing")
        if completion_mode == "local_ready":
            evidence = {
                "remote": remote,
                "capability": "optional_for_local_ready",
                "credential_capability": None,
            }
            return _capstone_remote_validation_result(
                status="succeeded",
                remote=remote,
                verification_results=[
                    _capstone_remote_verification_result(
                        "local_dvc_remote_validated",
                        "observed",
                        source_step,
                        True,
                        evidence,
                    )
                ],
                artifact_entries=[],
                next_actions=[],
            )
        evidence = {
            "blocked_reason": "missing_s3_remote_url",
            "remote": remote,
            "credential_capability": None,
        }
        return _capstone_remote_validation_result(
            status="blocked",
            remote=remote,
            verification_results=_capstone_remote_verification_results(
                completion_mode, source_step, False, evidence
            ),
            artifact_entries=[],
            next_actions=[
                "Provide dvc_remote_url=s3://bucket/prefix or configure a default S3 DVC remote."
            ],
        )

    remote_type = _remote_type(resolved_url)
    remote = _capstone_remote_record(remote_name, resolved_url, remote_type)
    if remote_type == "local":
        validation = _validate_local_remote_url(resolved_url)
        evidence = {
            "remote": remote,
            "validation": validation,
            "dvc_remote_configured": bool(configure_result),
            "credential_capability": None,
        }
        passed = bool(validation["passed"])
        return _capstone_remote_validation_result(
            status="succeeded" if passed else "blocked",
            remote=remote,
            verification_results=[
                _capstone_remote_verification_result(
                    "local_dvc_remote_validated",
                    "observed",
                    source_step,
                    passed,
                    evidence,
                )
            ],
            artifact_entries=(
                [_capstone_remote_artifact(source_step, remote, validation)]
                if passed
                else []
            ),
            next_actions=[] if passed else validation["next_actions"],
        )

    if remote_type != "s3":
        evidence = {
            "blocked_reason": "unsupported_remote_type",
            "remote": remote,
            "credential_capability": None,
        }
        return _capstone_remote_validation_result(
            status="blocked",
            remote=remote,
            verification_results=_capstone_remote_verification_results(
                completion_mode, source_step, False, evidence
            ),
            artifact_entries=[],
            next_actions=["Use a local path or s3:// URL for the capstone DVC remote."],
        )

    credential_capability = _validate_s3_credential_capability(resolved_url)
    evidence = {
        "remote": remote,
        "dvc_remote_configured": bool(configure_result),
        "credential_capability": credential_capability,
    }
    passed = bool(credential_capability.get("passed"))
    return _capstone_remote_validation_result(
        status="succeeded" if passed else "blocked",
        remote=remote,
        verification_results=[
            _capstone_remote_verification_result(
                "s3_remote_validated",
                "observed",
                source_step,
                passed,
                evidence,
            )
        ],
        artifact_entries=(
            [_capstone_remote_artifact(source_step, remote, credential_capability)]
            if passed
            else []
        ),
        next_actions=credential_capability.get("next_actions", []),
    )


def push_capstone_data(
    project_path: str,
    completion_mode: str = "capstone_complete",
    remote_name: str = "capstone",
    capstone_dvc_remote_result: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
    paths: list[str] | None = None,
    source_step: str = "push_capstone_data",
) -> dict[str, Any]:
    """Run approval-gated DVC push and record observed capstone transfer evidence."""
    return _run_capstone_dvc_transfer(
        project_path=project_path,
        completion_mode=completion_mode,
        direction="push",
        remote_name=remote_name,
        capstone_dvc_remote_result=capstone_dvc_remote_result,
        approval_record=approval_record,
        paths=paths,
        source_step=source_step,
        required_risks=("uses_cloud_credentials",),
    )


def pull_capstone_data(
    project_path: str,
    completion_mode: str = "capstone_complete",
    remote_name: str = "capstone",
    capstone_dvc_remote_result: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
    paths: list[str] | None = None,
    source_step: str = "pull_capstone_data",
) -> dict[str, Any]:
    """Run approval-gated DVC pull and record observed capstone transfer evidence."""
    return _run_capstone_dvc_transfer(
        project_path=project_path,
        completion_mode=completion_mode,
        direction="pull",
        remote_name=remote_name,
        capstone_dvc_remote_result=capstone_dvc_remote_result,
        approval_record=approval_record,
        paths=paths,
        source_step=source_step,
        required_risks=("uses_cloud_credentials", "writes_project_files"),
    )


def _run_capstone_dvc_transfer(
    project_path: str,
    completion_mode: str,
    direction: str,
    remote_name: str,
    capstone_dvc_remote_result: dict[str, Any] | None,
    approval_record: dict[str, Any] | None,
    paths: list[str] | None,
    source_step: str,
    required_risks: tuple[str, ...],
) -> dict[str, Any]:
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}
    if direction not in {"push", "pull"}:
        return {"success": False, "error": f"Unsupported DVC transfer direction: {direction}"}

    remote = _transfer_remote_record(path, remote_name, capstone_dvc_remote_result)
    base_evidence = {
        "transfer_direction": direction,
        "remote": remote,
        "approval_record": _approval_record_evidence(approval_record),
        "paths": paths or [],
    }
    if completion_mode == "local_ready":
        evidence = {
            **base_evidence,
            "blocked_reason": "transfer_deferred_for_local_ready",
            "capability": "deferred_for_local_ready",
        }
        return _capstone_transfer_result(
            status="succeeded",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[
                _capstone_transfer_artifact(source_step, direction, remote, evidence, "deferred")
            ],
            next_actions=[],
        )

    if not _approved_transfer_record_matches(approval_record, source_step, required_risks):
        evidence = {**base_evidence, "blocked_reason": "missing_or_denied_approval"}
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=[
                (
                    f"Record approval as an approved ApprovalRecord for {source_step} "
                    "with risk categories: "
                    f"{', '.join(required_risks)}."
                )
            ],
        )

    if not check_tool_installed("dvc"):
        evidence = {**base_evidence, "blocked_reason": "missing_dvc_executable"}
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=["Install DVC before running capstone data push or pull."],
        )

    if not (path / ".dvc" / "config").exists():
        evidence = {**base_evidence, "blocked_reason": "missing_dvc_repo"}
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=["Validate or initialize DVC metadata before data transfer."],
        )

    if remote["remote_type"] != "s3":
        evidence = {**base_evidence, "blocked_reason": "missing_validated_s3_remote"}
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=[
                "Validate an S3 capstone DVC remote before running capstone data transfer."
            ],
        )

    resolved_url = _read_dvc_remote_url(path, remote_name)
    if not resolved_url:
        evidence = {**base_evidence, "blocked_reason": "missing_dvc_remote_url"}
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=["Configure a DVC remote URL before running capstone data transfer."],
        )

    credential_capability = _validate_s3_credential_capability(resolved_url)
    if not credential_capability.get("passed"):
        evidence = {
            **base_evidence,
            "blocked_reason": "missing_or_invalid_cloud_credential_capability",
            "credential_capability": credential_capability,
        }
        return _capstone_transfer_result(
            status="blocked",
            source_step=source_step,
            passed=False,
            evidence=evidence,
            artifact_entries=[],
            next_actions=credential_capability.get("next_actions", []),
        )

    command = ["dvc", direction, "-r", remote_name]
    command.extend(paths or [])
    started_at = time.monotonic()
    result = run_command(command, cwd=str(path), timeout=600)
    duration_seconds = round(time.monotonic() - started_at, 3)
    evidence = {
        **base_evidence,
        "command": " ".join(command),
        "returncode": result.get("returncode"),
        "stdout_summary": _summarize_transfer_stream(result.get("stdout", "")),
        "stderr_summary": _summarize_transfer_stream(result.get("stderr", "")),
        "duration_seconds": duration_seconds,
        "credential_capability": {
            key: credential_capability.get(key)
            for key in ("status", "identity", "bucket_reachable", "prefix_checked")
        },
    }
    if result.get("success"):
        return _capstone_transfer_result(
            status="succeeded",
            source_step=source_step,
            passed=True,
            evidence=evidence,
            artifact_entries=[
                _capstone_transfer_artifact(source_step, direction, remote, evidence, "validated")
            ],
            next_actions=[],
        )

    evidence["blocked_reason"] = "dvc_transfer_failed"
    return _capstone_transfer_result(
        status="failed",
        source_step=source_step,
        passed=False,
        evidence=evidence,
        artifact_entries=[],
        next_actions=[
            "Inspect DVC transfer output, fix remote permissions or connectivity, and rerun."
        ],
    )


def _transfer_remote_record(
    project_path: Path,
    remote_name: str,
    capstone_dvc_remote_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(capstone_dvc_remote_result, dict):
        remote = capstone_dvc_remote_result.get("remote")
        if isinstance(remote, dict) and remote.get("remote_name"):
            return {
                "remote_name": remote.get("remote_name", remote_name),
                "remote_type": remote.get("remote_type", "missing"),
                "redacted_remote_url": remote.get("redacted_remote_url"),
            }
    remote_url = _read_dvc_remote_url(project_path, remote_name)
    return _capstone_remote_record(remote_name, remote_url, _remote_type(remote_url))


def _approved_transfer_record_matches(
    approval_record: dict[str, Any] | None,
    step_id: str,
    required_risks: tuple[str, ...],
) -> bool:
    if not isinstance(approval_record, dict):
        return False
    status = str(approval_record.get("status", "")).casefold()
    risk_categories = tuple(approval_record.get("risk_categories", ()))
    return (
        approval_record.get("step_id") == step_id
        and status == "approved"
        and risk_categories == required_risks
    )


def _approval_record_evidence(approval_record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(approval_record, dict):
        return None
    return {
        "workflow_run_id": approval_record.get("workflow_run_id"),
        "step_id": approval_record.get("step_id"),
        "risk_categories": approval_record.get("risk_categories", []),
        "status": approval_record.get("status"),
        "approver": approval_record.get("approver"),
        "timestamp": str(approval_record.get("timestamp")),
    }


def _summarize_transfer_stream(value: str, limit: int = 500) -> str:
    redacted = _redacted_evidence(value.strip())
    if not isinstance(redacted, str):
        redacted = str(redacted)
    return redacted[:limit]


def _capstone_transfer_result(
    status: str,
    source_step: str,
    passed: bool,
    evidence: dict[str, Any],
    artifact_entries: list[dict[str, Any]],
    next_actions: list[str],
) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "transfer": {
            "direction": evidence.get("transfer_direction"),
            "remote": evidence.get("remote"),
            "paths": evidence.get("paths", []),
        },
        "missing_inputs": [] if status == "succeeded" else ["approved_s3_transfer_evidence"],
        "next_actions": next_actions,
        "verification_results": [
            {
                "check_name": "s3_transfer_completed",
                "evidence_type": "observed",
                "source_step": source_step,
                "passed": passed,
                "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
            }
        ],
        "artifact_manifest": {"entries": artifact_entries},
    }


def _capstone_transfer_artifact(
    source_step: str,
    direction: str,
    remote: dict[str, Any],
    evidence: dict[str, Any],
    transfer_status: str,
) -> dict[str, Any]:
    return {
        "artifact_type": "capstone_data_transfer",
        "producing_step": source_step,
        "state": "validated",
        "uri": remote.get("redacted_remote_url") or f"dvc-remote://{remote.get('remote_name')}",
        "metadata": {
            "transfer_direction": direction,
            "transfer_status": transfer_status,
            "remote_name": remote.get("remote_name"),
            "remote_type": remote.get("remote_type"),
            "redacted_remote_url": remote.get("redacted_remote_url"),
            "returncode": evidence.get("returncode"),
            "duration_seconds": evidence.get("duration_seconds"),
            "paths": evidence.get("paths", []),
        },
    }


def _configure_capstone_dvc_remote(
    project_path: Path,
    remote_name: str,
    remote_url: str,
    default: bool,
) -> dict[str, Any]:
    cmd = ["dvc", "remote", "add"]
    if default:
        cmd.append("-d")
    cmd.extend([remote_name, remote_url])
    result = run_command(cmd, cwd=str(project_path), timeout=60)
    if result.get("success"):
        return result
    modify_result = run_command(
        ["dvc", "remote", "modify", remote_name, "url", remote_url],
        cwd=str(project_path),
        timeout=60,
    )
    if modify_result.get("success"):
        return modify_result
    return result


def _read_dvc_remote_url(project_path: Path, remote_name: str | None) -> str | None:
    config_path = project_path / ".dvc" / "config"
    if not config_path.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read_string(config_path.read_text())
    except configparser.Error:
        return None
    configured_remote = remote_name
    if not configured_remote and parser.has_section("core"):
        configured_remote = parser.get("core", "remote", fallback=None)
    if not configured_remote:
        configured_remote = parser.get("core", "remote", fallback=None)
    candidate_sections = (
        f'remote "{configured_remote}"',
        f"'remote \"{configured_remote}\"'",
    )
    for section in candidate_sections:
        if parser.has_section(section):
            url = parser.get(section, "url", fallback=None)
            if url:
                return url
    return None


def _remote_type(remote_url: str | None) -> str:
    if not remote_url:
        return "missing"
    parsed = urlparse(remote_url)
    if parsed.scheme == "s3":
        return "s3"
    if parsed.scheme in {"", "file"}:
        return "local"
    return parsed.scheme or "unknown"


def _capstone_remote_record(
    remote_name: str,
    remote_url: str | None,
    remote_type: str,
) -> dict[str, Any]:
    return {
        "remote_name": remote_name,
        "remote_type": remote_type,
        "redacted_remote_url": _redact_remote_url(remote_url),
    }


def _redact_remote_url(remote_url: str | None) -> str | None:
    if not remote_url:
        return None
    parsed = urlparse(remote_url)
    if parsed.scheme == "s3":
        bucket = _redact_token(parsed.netloc)
        path_parts = [
            _redact_token(part)
            for part in parsed.path.split("/")
            if part
        ]
        suffix = "/" + "/".join(path_parts) if path_parts else ""
        return f"s3://{bucket}{suffix}"
    if parsed.scheme and parsed.scheme != "file":
        return f"{parsed.scheme}://<redacted>"
    return str(Path(parsed.path or remote_url).expanduser())


def _redact_token(value: str) -> str:
    if not value:
        return "<redacted>"
    if len(value) <= 4:
        return value[0] + "***"
    return f"{value[:2]}***{value[-2:]}"


def _validate_local_remote_url(remote_url: str) -> dict[str, Any]:
    parsed = urlparse(remote_url)
    path = Path(parsed.path if parsed.scheme == "file" else remote_url).expanduser()
    passed = path.exists() and path.is_dir()
    return {
        "passed": passed,
        "status": "validated" if passed else "invalid_local_remote",
        "path_exists": path.exists(),
        "is_directory": path.is_dir(),
        "next_actions": []
        if passed
        else ["Create the local DVC remote directory or provide an existing directory."],
    }


def _validate_s3_credential_capability(remote_url: str) -> dict[str, Any]:
    if not BOTO3_AVAILABLE:
        return {
            "passed": False,
            "status": "missing_boto3",
            "identity": None,
            "bucket_reachable": False,
            "prefix_checked": False,
            "next_actions": ["Install boto3/botocore to validate S3 credential capability."],
        }
    parsed = urlparse(remote_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    if not bucket:
        return {
            "passed": False,
            "status": "missing_s3_bucket",
            "identity": None,
            "bucket_reachable": False,
            "prefix_checked": False,
            "next_actions": ["Provide an S3 DVC remote URL with a bucket name."],
        }
    try:
        session = boto3.Session()
        sts_client = session.client("sts")
        identity = sts_client.get_caller_identity()
        s3_client = session.client("s3")
        s3_client.head_bucket(Bucket=bucket)
        prefix_checked = False
        if prefix:
            s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
            prefix_checked = True
        return {
            "passed": True,
            "status": "validated",
            "identity": _redact_aws_identity(identity),
            "bucket_reachable": True,
            "prefix_checked": prefix_checked,
            "next_actions": [],
        }
    except NoCredentialsError:
        return {
            "passed": False,
            "status": "missing_cloud_credential_capability",
            "identity": None,
            "bucket_reachable": False,
            "prefix_checked": False,
            "next_actions": [
                "Configure AWS credentials outside Auto-MLOps and rerun validation."
            ],
        }
    except (ClientError, BotoCoreError) as exc:
        return {
            "passed": False,
            "status": "invalid_s3_remote_or_credentials",
            "identity": None,
            "bucket_reachable": False,
            "prefix_checked": False,
            "error_type": type(exc).__name__,
            "next_actions": [
                "Verify AWS credentials and bucket or prefix permissions before rerunning."
            ],
        }


def _redact_aws_identity(identity: dict[str, Any]) -> dict[str, Any]:
    account = str(identity.get("Account", identity.get("account", "")))
    arn = str(identity.get("Arn", identity.get("arn", "")))
    user_id = str(identity.get("UserId", identity.get("user_id", "")))
    redacted_account = _redact_token(account) if account else None
    return {
        "account": redacted_account,
        "arn": re.sub(r"::\d{12}:", f"::{redacted_account or '<redacted>'}:", arn)
        if arn
        else None,
        "user_id": _redact_token(user_id) if user_id else None,
    }


def _capstone_remote_verification_results(
    completion_mode: str,
    source_step: str,
    passed: bool,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    check_name = (
        "s3_remote_validated"
        if completion_mode == "capstone_complete"
        else "local_dvc_remote_validated"
    )
    return [
        _capstone_remote_verification_result(
            check_name,
            "observed",
            source_step,
            passed,
            evidence,
        )
    ]


def _capstone_remote_verification_result(
    check_name: str,
    evidence_type: str,
    source_step: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": evidence_type,
        "source_step": source_step,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _capstone_remote_artifact(
    source_step: str,
    remote: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_type": "capstone_data_remote",
        "producing_step": source_step,
        "state": "validated",
        "uri": remote["redacted_remote_url"] or f"dvc-remote://{remote['remote_name']}",
        "metadata": {
            "remote_name": remote["remote_name"],
            "remote_type": remote["remote_type"],
            "redacted_remote_url": remote["redacted_remote_url"],
            "validation_status": validation.get("status"),
            "bucket_reachable": validation.get("bucket_reachable"),
            "credential_capability_observed": validation.get("passed"),
        },
    }


def _capstone_remote_validation_result(
    status: str,
    remote: dict[str, Any],
    verification_results: list[dict[str, Any]],
    artifact_entries: list[dict[str, Any]],
    next_actions: list[str],
) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "remote": remote,
        "missing_inputs": [] if status == "succeeded" else ["capstone_dvc_remote"],
        "next_actions": next_actions,
        "verification_results": verification_results,
        "artifact_manifest": {"entries": artifact_entries},
    }


def _redacted_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = key.casefold()
            if (
                "secret" in normalized_key
                or "token" in normalized_key
                or "access_key" in normalized_key
            ):
                continue
            if normalized_key == "account" and item is not None:
                redacted[key] = _redact_token(str(item))
                continue
            if normalized_key == "arn" and item is not None:
                redacted[key] = re.sub(
                    r"::\d{12}:",
                    f"::{_redact_token(str(item).split('::', 1)[-1][:12])}:",
                    str(item),
                )
                continue
            if normalized_key in {"user_id", "userid"} and item is not None:
                redacted[key] = _redact_token(str(item))
                continue
            redacted[key] = _redacted_evidence(item)
        return redacted
    if isinstance(value, list):
        return [_redacted_evidence(item) for item in value]
    if isinstance(value, str):
        if "s3://" in value:
            return re.sub(
                r"s3://[^\s\"']+",
                lambda match: _redact_remote_url(match.group(0)),
                value,
            )
        return value
    return value


def _ensure_local_dvc_repo(
    project_path: Path,
    initialize_if_missing: bool,
) -> tuple[dict[str, Any], str | None]:
    dvc_config = project_path / ".dvc" / "config"
    if dvc_config.exists():
        return (
            {
                "status": "validated",
                "project_path": str(project_path),
                "dvc_dir": ".dvc",
                "dvc_config_path": ".dvc/config",
            },
            None,
        )
    if not initialize_if_missing:
        return (
            {
                "status": "missing",
                "project_path": str(project_path),
                "dvc_config_path": ".dvc/config",
            },
            "DVC metadata is missing; approve initialization or initialize DVC manually.",
        )

    init_result = run_command(["dvc", "init", "--no-scm"], cwd=str(project_path), timeout=60)
    if not init_result.get("success"):
        return (
            {
                "status": "init_failed",
                "project_path": str(project_path),
                "dvc_config_path": ".dvc/config",
                "init_result": init_result,
            },
            "DVC initialization failed for the local project.",
        )
    return (
        {
            "status": "initialized",
            "project_path": str(project_path),
            "dvc_dir": ".dvc",
            "dvc_config_path": ".dvc/config",
            "init_stdout": init_result.get("stdout", ""),
        },
        None,
    )


def _capstone_package_paths_from_split_records(
    project_path: Path,
    split_records: list[dict[str, Any]],
) -> list[str]:
    package_paths: list[str] = []
    for split_record in split_records:
        manifest_path = split_record.get("split_manifest_path")
        if not manifest_path:
            continue
        relative_manifest = Path(manifest_path)
        if relative_manifest.is_absolute():
            try:
                relative_manifest = relative_manifest.relative_to(project_path)
            except ValueError:
                continue
        parts = relative_manifest.parts
        if len(parts) < 4 or parts[:2] != ("data", "capstone"):
            continue
        package_path = Path(*parts[:3]).as_posix()
        if not (project_path / package_path).exists():
            continue
        if package_path not in package_paths:
            package_paths.append(package_path)
    return package_paths


def _capstone_lineage_artifacts(
    project_path: Path,
    split_records: list[dict[str, Any]],
    tracked_package_paths: list[str],
    dvc_tracking_files: list[str] | None = None,
    dvc_repo: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    artifact_entries: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for split_record in split_records:
        source_path = split_record.get("source_path")
        if source_path and source_path not in seen_sources:
            artifact_entries.append(
                {
                    "artifact_type": "capstone_source_dataset",
                    "producing_step": "track_capstone_data_package",
                    "state": "external",
                    "path": source_path,
                    "metadata": {
                        "dataset_id": split_record.get("dataset_id"),
                        "source_path_kind": "user_provided_local_or_mounted",
                    },
                }
            )
            seen_sources.add(source_path)
        split_manifest_path = split_record.get("split_manifest_path")
        if split_manifest_path:
            manifest_file = project_path / split_manifest_path
            artifact_entries.append(
                {
                    "artifact_type": "split_manifest",
                    "producing_step": "track_capstone_data_package",
                    "state": "generated",
                    "path": split_manifest_path,
                    "checksum": _sha256_file(manifest_file) if manifest_file.is_file() else None,
                    "metadata": {
                        "dataset_id": split_record.get("dataset_id"),
                        "split_strategy": split_record.get("split_strategy"),
                    },
                }
            )
    for package_path in tracked_package_paths:
        artifact_entries.append(
            {
                "artifact_type": "capstone_data_package",
                "producing_step": "track_capstone_data_package",
                "state": "generated",
                "path": package_path,
                "metadata": {"tracking": "dvc"},
            }
        )
    for dvc_file in dvc_tracking_files or []:
        dvc_file_path = project_path / dvc_file
        artifact_entries.append(
            {
                "artifact_type": "dvc_tracking_file",
                "producing_step": "track_capstone_data_package",
                "state": "generated",
                "path": dvc_file,
                "checksum": _sha256_file(dvc_file_path) if dvc_file_path.is_file() else None,
            }
        )
    if dvc_repo and dvc_repo.get("dvc_config_path"):
        config_path = project_path / dvc_repo["dvc_config_path"]
        artifact_entries.append(
            {
                "artifact_type": "dvc_repo_metadata",
                "producing_step": "track_capstone_data_package",
                "state": "validated" if dvc_repo.get("status") == "validated" else "generated",
                "path": dvc_repo["dvc_config_path"],
                "checksum": _sha256_file(config_path) if config_path.is_file() else None,
                "metadata": {"dvc_repo_status": dvc_repo.get("status")},
            }
        )
    return artifact_entries


def _capstone_dvc_tracking_result(
    status: str,
    dvc_repo: dict[str, Any],
    tracked_package_paths: list[str],
    dvc_tracking_files: list[str],
    artifact_entries: list[dict[str, Any]],
    next_actions: list[str],
    dvc_repo_passed: bool,
    package_tracking_passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    lineage_passed = bool(artifact_entries)
    return {
        "success": True,
        "status": status,
        "dvc_repo": dvc_repo,
        "tracked_package_paths": tracked_package_paths,
        "dvc_tracking_files": dvc_tracking_files,
        "missing_inputs": [] if status == "succeeded" else ["local_dvc_tracking_evidence"],
        "next_actions": next_actions,
        "verification_results": [
            {
                "check_name": "dvc_repo_validated",
                "evidence_type": "observed",
                "source_step": "track_capstone_data_package",
                "passed": dvc_repo_passed,
                "evidence": json.dumps(evidence, sort_keys=True),
            },
            {
                "check_name": "capstone_data_package_tracked",
                "evidence_type": "observed",
                "source_step": "track_capstone_data_package",
                "passed": package_tracking_passed,
                "evidence": json.dumps(evidence, sort_keys=True),
            },
            {
                "check_name": "dataset_lineage_artifacts_reported",
                "evidence_type": "observed",
                "source_step": "track_capstone_data_package",
                "passed": lineage_passed,
                "evidence": json.dumps(
                    {"artifact_count": len(artifact_entries), **evidence},
                    sort_keys=True,
                ),
            },
        ],
        "artifact_manifest": {"entries": artifact_entries},
    }


def _capstone_split_plan(
    project_path: Path,
    dataset: dict[str, Any],
    test_size: float,
    split_seed: int,
) -> dict[str, Any]:
    per_class_counts = {}
    for class_name, total_count in dataset.get("per_class_counts", {}).items():
        planned_test = _planned_test_count(total_count, test_size)
        per_class_counts[class_name] = {
            "train": total_count - planned_test,
            "test": planned_test,
            "total": total_count,
        }
    relative_manifest_path = (
        Path("data") / "capstone" / dataset["dataset_id"] / "split_manifest.json"
    )
    return {
        "dataset_id": dataset["dataset_id"],
        "source_path": dataset["source_path"],
        "split_strategy": "manifest",
        "seed": int(split_seed),
        "test_size": float(test_size),
        "train_count": sum(item["train"] for item in per_class_counts.values()),
        "test_count": sum(item["test"] for item in per_class_counts.values()),
        "per_class_counts": per_class_counts,
        "split_manifest_path": str(relative_manifest_path),
        "absolute_split_manifest_path": str(project_path / relative_manifest_path),
    }


def _planned_test_count(total_count: int, test_size: float) -> int:
    test_count = max(1, round(total_count * test_size))
    if test_count >= total_count and total_count > 1:
        test_count = total_count - 1
    return test_count


def _existing_split_record(
    dataset: dict[str, Any],
    split_seed: int,
    test_size: float,
) -> dict[str, Any]:
    train_counts = dataset.get("split_counts", {}).get("train", {})
    test_counts = dataset.get("split_counts", {}).get("test", {})
    return {
        "dataset_id": dataset["dataset_id"],
        "source_path": dataset["source_path"],
        "split_strategy": "existing",
        "seed": split_seed,
        "test_size": test_size,
        "train_count": sum(train_counts.values()),
        "test_count": sum(test_counts.values()),
        "per_class_counts": {
            class_name: {
                "train": train_counts.get(class_name, 0),
                "test": test_counts.get(class_name, 0),
                "total": train_counts.get(class_name, 0) + test_counts.get(class_name, 0),
            }
            for class_name in dataset.get("class_names", [])
        },
        "split_manifest_path": None,
        "materialized_train_path": dataset.get("existing_train_path"),
        "materialized_test_path": dataset.get("existing_test_path"),
    }


def _build_split_manifest(
    dataset: dict[str, Any],
    test_size: float,
    split_seed: int,
) -> dict[str, Any]:
    source_path = Path(dataset["source_path"])
    train_files: list[dict[str, str]] = []
    test_files: list[dict[str, str]] = []
    per_class_counts: dict[str, dict[str, int]] = {}
    for class_name in dataset.get("class_names", []):
        class_dir = source_path / class_name
        files = sorted(
            _image_files_under(class_dir),
            key=lambda item: item.relative_to(source_path).as_posix(),
        )
        shuffled = list(files)
        random.Random(f"{split_seed}:{class_name}").shuffle(shuffled)
        test_count = _planned_test_count(len(shuffled), test_size)
        test_set = {
            file.relative_to(source_path).as_posix()
            for file in shuffled[:test_count]
        }
        class_train: list[dict[str, str]] = []
        class_test: list[dict[str, str]] = []
        for file in files:
            relative_path = file.relative_to(source_path).as_posix()
            file_record = {
                "class_name": class_name,
                "relative_path": relative_path,
                "source_path": str(file),
            }
            if relative_path in test_set:
                class_test.append(file_record)
            else:
                class_train.append(file_record)
        train_files.extend(class_train)
        test_files.extend(class_test)
        per_class_counts[class_name] = {
            "train": len(class_train),
            "test": len(class_test),
            "total": len(files),
        }

    train_files = sorted(train_files, key=lambda item: (item["class_name"], item["relative_path"]))
    test_files = sorted(test_files, key=lambda item: (item["class_name"], item["relative_path"]))
    return {
        "dataset_id": dataset["dataset_id"],
        "source_path": dataset["source_path"],
        "split_strategy": "manifest",
        "seed": int(split_seed),
        "test_size": float(test_size),
        "train_count": len(train_files),
        "test_count": len(test_files),
        "per_class_counts": per_class_counts,
        "files": {
            "train": train_files,
            "test": test_files,
        },
    }


def _capstone_split_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
    evidence_type: str = "observed",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": evidence_type,
        "source_step": "generate_split_manifests",
        "passed": passed,
        "evidence": json.dumps(evidence, sort_keys=True),
    }


def _detect_unsplit_class_folder_dataset(record: dict[str, Any], path: Path) -> dict[str, Any]:
    summary = _class_folder_summary(path)
    record["layout"] = "class_folders"
    record["missing_inputs"].extend(summary["missing_inputs"])
    record["next_actions"].extend(summary["next_actions"])
    if not summary["supported"]:
        record["blocked_reason"] = summary["blocked_reason"]
        return record

    record["status"] = "succeeded"
    record["class_names"] = summary["class_names"]
    record["class_count"] = len(summary["class_names"])
    record["per_class_counts"] = summary["per_class_counts"]
    record["total_image_count"] = summary["total_image_count"]
    return record


def _class_folder_summary(path: Path) -> dict[str, Any]:
    class_dirs = [
        child
        for child in sorted(path.iterdir(), key=lambda item: item.name.casefold())
        if child.is_dir() and not child.name.startswith(".")
    ]
    if not class_dirs:
        return {
            "supported": False,
            "blocked_reason": "unsupported_empty_dataset",
            "class_names": [],
            "per_class_counts": {},
            "total_image_count": 0,
            "missing_inputs": ["class_label_subdirectories"],
            "next_actions": [
                "Add class-labelled subdirectories with at least one image file per class."
            ],
        }

    per_class_counts = {
        class_dir.name: len(_image_files_under(class_dir))
        for class_dir in class_dirs
    }
    empty_classes = [
        class_name for class_name, image_count in per_class_counts.items() if image_count == 0
    ]
    if empty_classes:
        return {
            "supported": False,
            "blocked_reason": "empty_class_folder",
            "class_names": sorted(per_class_counts),
            "per_class_counts": per_class_counts,
            "total_image_count": sum(per_class_counts.values()),
            "missing_inputs": ["non_empty_class_folders"],
            "next_actions": [
                "Add at least one supported image file to every class-labelled subdirectory."
            ],
        }

    return {
        "supported": True,
        "blocked_reason": None,
        "class_names": sorted(per_class_counts),
        "per_class_counts": per_class_counts,
        "total_image_count": sum(per_class_counts.values()),
        "missing_inputs": [],
        "next_actions": [],
    }


def _case_insensitive_child_dir(path: Path, name: str) -> Path | None:
    for child in path.iterdir():
        if child.is_dir() and child.name.casefold() == name.casefold():
            return child
    return None


def _direct_image_files(path: Path) -> tuple[Path, ...]:
    return tuple(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.casefold() in CAPSTONE_IMAGE_EXTENSIONS
    )


def _image_files_under(path: Path) -> tuple[Path, ...]:
    return tuple(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.casefold() in CAPSTONE_IMAGE_EXTENSIONS
    )


def detect_training_project(project_path: str) -> dict[str, Any]:
    """Detect supported training project shape without running or installing anything."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}

    project_files = _relative_files(path)
    project_dirs = _relative_dirs(path)
    dependency_text = _dependency_signal_text(path)
    config_text = _joined_file_text(
        path,
        [file for file in project_files if _is_config_file(file)],
    )
    python_signal_files = [
        file
        for file in project_files
        if file.endswith(".py")
        and (Path(file).name == "train.py" or file.startswith("src/"))
    ]
    python_text = _joined_file_text(path, python_signal_files)

    training_entrypoints = _detect_training_entrypoints(path, project_files)
    training_entrypoint = training_entrypoints[0] if len(training_entrypoints) == 1 else None
    hydra_config = _detect_hydra_config(path, project_files, training_entrypoint)
    likely_config_files = tuple(
        file
        for file in project_files
        if _is_config_file(file)
    )
    test_files = tuple(
        file
        for file in project_files
        if (
            file.startswith("tests/")
            and file.endswith(".py")
            and Path(file).name.startswith("test_")
        )
    )
    test_command = "python -m pytest tests -q" if test_files else None
    dvc_signals = tuple(
        file
        for file in project_files
        if file == ".dvc/config" or file == "dvc.yaml" or file.endswith(".dvc")
    )
    data_signals = tuple(
        directory
        for directory in project_dirs
        if directory == "data" or directory.startswith("data/")
    )
    output_dirs = tuple(
        directory
        for directory in project_dirs
        if directory in {"outputs", "logs", "mlruns", "checkpoints", "models", "artifacts"}
    )
    checkpoint_artifact_candidates = tuple(
        file
        for file in project_files
        if file.endswith((".ckpt", ".pt", ".pth", ".onnx"))
        and (
            file.startswith("outputs/")
            or file.startswith("checkpoints/")
            or file.startswith("models/")
            or file.startswith("artifacts/")
        )
    )

    has_hydra_signal = (
        "hydra-core" in dependency_text
        or "import hydra" in python_text
        or "@hydra.main" in python_text
    )
    has_pytorch_signal = any(
        token in dependency_text or token in python_text
        for token in ("torch", "pytorch", "lightning", "pytorch-lightning")
    )
    has_lightning_signal = any(
        token in dependency_text or token in python_text
        for token in ("lightning", "pytorch-lightning")
    )
    has_timm_signal = (
        "timm" in dependency_text or "timm" in config_text or "timm" in python_text
    )
    has_dvc_signal = (
        ".dvc/config" in project_files
        or ".dvc" in {Path(file).parts[0] for file in project_files if Path(file).parts}
        or "dvc.yaml" in project_files
        or any(file.endswith(".dvc") for file in project_files)
    )

    framework_family = "unknown"
    if has_lightning_signal:
        framework_family = "pytorch_lightning"
    elif has_pytorch_signal:
        framework_family = "pytorch"

    config_system = "hydra" if has_hydra_signal and likely_config_files else "missing"
    model_library = "timm" if has_timm_signal else "unknown"
    data_versioning = "dvc" if has_dvc_signal else "missing"

    missing_required_pieces: list[str] = []
    next_actions: list[str] = []
    if len(training_entrypoints) > 1:
        missing_required_pieces.append("training_entrypoint")
        next_actions.append(
            "Provide the intended training entrypoint; multiple candidates were observed."
        )
    elif training_entrypoint is None:
        missing_required_pieces.append("training_entrypoint")
        next_actions.append("Add or provide a training entrypoint such as src/train.py.")
    if hydra_config["ambiguous"]:
        missing_required_pieces.append("hydra_config")
        next_actions.append("Provide the intended Hydra config path and config name.")
    elif config_system != "hydra" or hydra_config["config_file"] is None:
        missing_required_pieces.append("hydra_config")
        next_actions.append("Add or provide Hydra configs such as configs/train.yaml.")
    if not has_pytorch_signal:
        missing_required_pieces.append("pytorch_dependency")
        next_actions.append("Add PyTorch or Lightning dependency evidence for Phase 3 support.")
    if not has_timm_signal:
        missing_required_pieces.append("timm_dependency")
        next_actions.append("Add TIMM dependency or model configuration evidence.")
    if not (has_dvc_signal or data_signals):
        missing_required_pieces.append("dvc_or_data_evidence")
        next_actions.append("Add DVC metadata or observed local data path evidence.")
    if test_command is None:
        missing_required_pieces.append("test_command")
        next_actions.append("Add a pytest test that validates the training entrypoint import path.")
    if not (output_dirs or checkpoint_artifact_candidates):
        missing_required_pieces.append("output_artifact_candidates")
        next_actions.append(
            "Add observed output, checkpoint, model, log, or artifact directory evidence."
        )

    status = "supported" if not missing_required_pieces else "blocked"
    confidence = 0.95 if status == "supported" else 0.45
    if model_library != "timm":
        confidence = min(confidence, 0.75)
    if data_versioning != "dvc":
        confidence = min(confidence, 0.8)

    observed_evidence = {
        "project_files": project_files,
        "project_dirs": project_dirs,
        "training_entrypoint_candidates": training_entrypoints,
        "entrypoint_exists": training_entrypoint is not None,
        "hydra_config_path": hydra_config["config_path"],
        "hydra_config_name": hydra_config["config_name"],
        "hydra_config_file": hydra_config["config_file"],
        "hydra_config_files": likely_config_files,
        "dvc_signals": dvc_signals,
        "data_signals": data_signals,
        "dependency_files": tuple(
            file
            for file in project_files
            if file in {"pyproject.toml", "requirements.txt", "requirements-test.txt"}
        ),
        "output_dirs": output_dirs,
        "checkpoint_artifact_candidates": checkpoint_artifact_candidates,
    }
    verification_results = [
        {
            "check_name": "training_project_detected",
            "evidence_type": "observed",
            "source_step": "detect_training_project",
            "passed": True,
            "evidence": (
                f"status={status}; framework_family={framework_family}; "
                f"model_library={model_library}; config_system={config_system}; "
                f"data_versioning={data_versioning}; entrypoint={training_entrypoint}; "
                f"hydra_config={hydra_config['config_file']}; "
                f"missing_required_pieces={', '.join(missing_required_pieces) or 'none'}"
            ),
        }
    ]
    if training_entrypoint is not None:
        verification_results.append(
            {
                "check_name": "training_entrypoint_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": f"entrypoint={training_entrypoint}",
            }
        )
    if hydra_config["config_file"] is not None and not hydra_config["ambiguous"]:
        verification_results.append(
            {
                "check_name": "hydra_config_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": (
                    f"config_path={hydra_config['config_path']}; "
                    f"config_name={hydra_config['config_name']}; "
                    f"config_file={hydra_config['config_file']}"
                ),
            }
        )
    if has_dvc_signal or data_signals:
        verification_results.append(
            {
                "check_name": "dvc_or_data_evidence_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": f"dvc_signals={dvc_signals}; data_signals={data_signals}",
            }
        )
    if has_pytorch_signal and has_timm_signal:
        verification_results.append(
            {
                "check_name": "pytorch_timm_signals_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": f"framework_family={framework_family}; model_library={model_library}",
            }
        )
    if test_command is not None:
        verification_results.append(
            {
                "check_name": "test_command_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": f"test_command={test_command}; test_files={test_files}",
            }
        )
    if output_dirs or checkpoint_artifact_candidates:
        verification_results.append(
            {
                "check_name": "output_artifact_candidates_detected",
                "evidence_type": "observed",
                "source_step": "detect_training_project",
                "passed": True,
                "evidence": (
                    f"output_dirs={output_dirs}; "
                    f"checkpoint_artifact_candidates={checkpoint_artifact_candidates}"
                ),
            }
        )
    artifact_entries = _training_detection_artifact_entries(
        training_entrypoint=training_entrypoint,
        hydra_config_file=hydra_config["config_file"],
        dvc_signals=dvc_signals,
        data_signals=data_signals,
        dependency_files=observed_evidence["dependency_files"],
        test_files=test_files,
        output_dirs=output_dirs,
        checkpoint_artifact_candidates=checkpoint_artifact_candidates,
    )

    return {
        "success": True,
        "status": status,
        "confidence": confidence,
        "framework_family": framework_family,
        "model_library": model_library,
        "config_system": config_system,
        "data_versioning": data_versioning,
        "training_entrypoint": training_entrypoint,
        "training_entrypoint_candidates": list(training_entrypoints),
        "hydra_config_path": hydra_config["config_path"],
        "hydra_config_name": hydra_config["config_name"],
        "hydra_config_file": hydra_config["config_file"],
        "likely_config_files": list(likely_config_files),
        "test_files": list(test_files),
        "test_command": test_command,
        "missing_required_pieces": missing_required_pieces,
        "next_actions": next_actions,
        "observed_evidence": observed_evidence,
        "suggested_validation_command": (
            f"python {training_entrypoint} trainer.fast_dev_run=true"
            if training_entrypoint
            else None
        ),
        "output_dirs": list(output_dirs),
        "checkpoint_artifact_candidates": list(checkpoint_artifact_candidates),
        "verification_results": verification_results,
        "artifact_manifest": {"entries": artifact_entries},
    }


def _relative_files(path: Path) -> tuple[str, ...]:
    return tuple(
        sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.is_file())
    )


def _relative_dirs(path: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(directory.relative_to(path))
            for directory in path.rglob("*")
            if directory.is_dir()
        )
    )


def _detect_training_entrypoints(path: Path, project_files: tuple[str, ...]) -> tuple[str, ...]:
    candidates: list[str] = []
    for file in project_files:
        file_path = Path(file)
        if file_path.suffix != ".py":
            continue
        text = _safe_read_text(path / file).lower()
        if file in {"src/train.py", "train.py"}:
            candidates.append(file)
            continue
        if file_path.name == "train.py" and ("@hydra.main" in text or "import hydra" in text):
            candidates.append(file)
    return tuple(dict.fromkeys(candidates))


def _detect_hydra_config(
    path: Path,
    project_files: tuple[str, ...],
    training_entrypoint: str | None,
) -> dict[str, Any]:
    config_files = tuple(file for file in project_files if _is_config_file(file))
    if training_entrypoint is None:
        return {
            "config_path": None,
            "config_name": None,
            "config_file": None,
            "ambiguous": False,
        }

    entrypoint_text = _safe_read_text(path / training_entrypoint)
    config_path_match = re.search(r"config_path\s*=\s*['\"]([^'\"]+)['\"]", entrypoint_text)
    config_name_match = re.search(r"config_name\s*=\s*['\"]([^'\"]+)['\"]", entrypoint_text)
    if config_path_match and config_name_match:
        config_root = _normalize_entrypoint_relative_path(
            path,
            training_entrypoint,
            config_path_match.group(1),
        )
        config_name = config_name_match.group(1)
        config_file = _resolve_hydra_config_file(config_root, config_name)
        return {
            "config_path": config_root,
            "config_name": _strip_yaml_suffix(config_name),
            "config_file": config_file if config_file in config_files else None,
            "ambiguous": False,
        }

    top_level_configs = tuple(
        file
        for file in config_files
        if file
        in {
            "configs/config.yaml",
            "configs/train.yaml",
            "conf/config.yaml",
            "conf/train.yaml",
        }
    )
    if len(top_level_configs) == 1:
        config_file = top_level_configs[0]
        return {
            "config_path": str(Path(config_file).parent),
            "config_name": _strip_yaml_suffix(Path(config_file).name),
            "config_file": config_file,
            "ambiguous": False,
        }

    return {
        "config_path": None,
        "config_name": None,
        "config_file": None,
        "ambiguous": len(top_level_configs) > 1,
    }


def _dependency_signal_text(path: Path) -> str:
    dependency_files = ("pyproject.toml", "requirements.txt", "requirements-test.txt")
    return "\n".join(_safe_read_text(path / file).lower() for file in dependency_files)


def _joined_file_text(path: Path, relative_files: list[str] | tuple[str, ...]) -> str:
    return "\n".join(_safe_read_text(path / file).lower() for file in relative_files)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def _is_config_file(file: str) -> bool:
    return (
        file.endswith((".yaml", ".yml"))
        and (file.startswith("configs/") or file.startswith("conf/"))
    )


def _normalize_entrypoint_relative_path(
    project_path: Path,
    training_entrypoint: str,
    relative_config_path: str,
) -> str:
    resolved_path = (project_path / training_entrypoint).parent / relative_config_path
    try:
        return str(resolved_path.resolve().relative_to(project_path.resolve()))
    except ValueError:
        return relative_config_path


def _resolve_hydra_config_file(config_path: str, config_name: str) -> str:
    if config_name.endswith((".yaml", ".yml")):
        return str(Path(config_path) / config_name)
    return str(Path(config_path) / f"{config_name}.yaml")


def _strip_yaml_suffix(value: str) -> str:
    return re.sub(r"\.ya?ml$", "", value)


def _training_detection_artifact_entries(
    *,
    training_entrypoint: str | None,
    hydra_config_file: str | None,
    dvc_signals: tuple[str, ...],
    data_signals: tuple[str, ...],
    dependency_files: tuple[str, ...],
    test_files: tuple[str, ...],
    output_dirs: tuple[str, ...],
    checkpoint_artifact_candidates: tuple[str, ...],
) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []

    def add(artifact_type: str, artifact_path: str) -> None:
        entries.append(
            {
                "artifact_type": artifact_type,
                "producing_step": "detect_training_project",
                "state": "external",
                "path": artifact_path,
            }
        )

    if training_entrypoint:
        add("training_entrypoint", training_entrypoint)
    if hydra_config_file:
        add("configuration", hydra_config_file)
    for dvc_signal in dvc_signals:
        add("dvc_metadata", dvc_signal)
    for data_signal in data_signals:
        add("data_evidence", data_signal)
    for dependency_file in dependency_files:
        add("dependency_file", dependency_file)
    for test_file in test_files:
        add("test_suite", test_file)
    for output_dir in output_dirs:
        add("output_directory", output_dir)
    for artifact_candidate in checkpoint_artifact_candidates:
        add("checkpoint_or_model_artifact", artifact_candidate)
    return entries


def run_bounded_training(
    project_path: str,
    training_entrypoint: str,
    hydra_config_path: str | None,
    hydra_config_name: str | None,
    timeout_seconds: int,
    max_epochs: int,
    device: str,
    data_subset: int,
    hydra_overrides: list[str] | None = None,
    target_metric: str = "accuracy",
) -> dict[str, Any]:
    """Run a detected training entrypoint with bounded controls and capture evidence."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}

    missing_controls = [
        name
        for name, value in (
            ("timeout_seconds", timeout_seconds),
            ("max_epochs", max_epochs),
            ("device", device),
            ("data_subset", data_subset),
        )
        if value in (None, "")
    ]
    if missing_controls:
        return _bounded_training_blocked_result(
            missing_controls=missing_controls,
            target_metric=target_metric,
        )

    if not (path / training_entrypoint).is_file():
        return _bounded_training_blocked_result(
            missing_controls=["training_entrypoint"],
            target_metric=target_metric,
        )

    run_dir = path / ".auto_mlops" / "training" / f"run-{int(time.time() * 1000)}"
    log_path = run_dir / "training.log"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    started_at = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_overrides = list(hydra_overrides or [])
    effective_overrides.extend(
        [
            f"trainer.max_epochs={max_epochs}",
            f"device={device}",
            f"data.subset={data_subset}",
        ]
    )
    command = [sys.executable, training_entrypoint, *effective_overrides]
    config_snapshot_relative = _write_training_config_snapshot(
        project_path=path,
        hydra_config_path=hydra_config_path,
        hydra_config_name=hydra_config_name,
        config_snapshot_path=config_snapshot_path,
    )

    try:
        completed = subprocess.run(
            command,
            cwd=path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True

    duration_seconds = time.time() - started_at
    if isinstance(stdout, bytes):
        stdout = stdout.decode(errors="ignore")
    if isinstance(stderr, bytes):
        stderr = stderr.decode(errors="ignore")

    log_path.write_text(
        "\n".join(
            (
                f"command: {' '.join(command)}",
                f"exit_code: {exit_code}",
                f"duration_seconds: {duration_seconds:.6f}",
                "--- stdout ---",
                stdout,
                "--- stderr ---",
                stderr,
            )
        )
    )

    metrics = _parse_training_metrics(stdout, stderr)
    checkpoint_artifact_paths = _find_training_artifacts(path)
    log_relative = relative_to_project(str(path), log_path)
    failure_reasons: list[str] = []
    if timed_out:
        failure_reasons.append(f"command timed out after {timeout_seconds}s")
    elif exit_code != 0:
        failure_reasons.append(f"non-zero exit code {exit_code}")
    if target_metric not in metrics:
        failure_reasons.append(f"missing target metric '{target_metric}'")
    if not checkpoint_artifact_paths:
        failure_reasons.append("missing checkpoint/model artifact")

    status = "succeeded" if not failure_reasons else "failed"
    artifact_entries = [
        {
            "artifact_type": "training_log",
            "producing_step": "run_bounded_training",
            "state": "generated",
            "path": log_relative,
        },
        {
            "artifact_type": "config_snapshot",
            "producing_step": "run_bounded_training",
            "state": "generated",
            "path": config_snapshot_relative,
        },
    ]
    artifact_entries.extend(
        {
            "artifact_type": "checkpoint_or_model_artifact",
            "producing_step": "run_bounded_training",
            "state": "generated",
            "path": artifact_path,
        }
        for artifact_path in checkpoint_artifact_paths
    )

    return {
        "success": True,
        "status": status,
        "command": command,
        "exit_code": exit_code,
        "duration_seconds": duration_seconds,
        "stdout_summary": _summarize_text(stdout),
        "stderr_summary": _summarize_text(stderr),
        "log_path": log_relative,
        "metrics": metrics,
        "target_metric": target_metric,
        "effective_overrides": effective_overrides,
        "config_snapshot_path": config_snapshot_relative,
        "checkpoint_artifact_paths": list(checkpoint_artifact_paths),
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else None,
        "next_actions": _training_failure_next_actions(failure_reasons, log_relative),
        "verification_results": _bounded_training_verification_results(
            controls_present=True,
            command_completed=status == "succeeded" or exit_code == 0,
            metric_captured=target_metric in metrics,
            artifact_captured=bool(checkpoint_artifact_paths),
            run_evidence_captured=True,
            evidence_summary=(
                f"command={' '.join(command)}; exit_code={exit_code}; "
                f"duration_seconds={duration_seconds:.6f}; log_path={log_relative}; "
                f"metrics={metrics}; artifacts={checkpoint_artifact_paths}"
            ),
        ),
        "artifact_manifest": {"entries": artifact_entries},
    }


def _bounded_training_blocked_result(
    missing_controls: list[str],
    target_metric: str,
) -> dict[str, Any]:
    next_actions = [
        "Provide timeout_seconds, max_epochs, device, and data_subset before training."
    ]
    if "training_entrypoint" in missing_controls:
        next_actions = ["Provide a detected training entrypoint before running training."]

    return {
        "success": True,
        "status": "blocked",
        "missing_required_pieces": missing_controls,
        "target_metric": target_metric,
        "failure_reason": f"missing bounded training prerequisites: {', '.join(missing_controls)}",
        "next_actions": next_actions,
        "verification_results": _bounded_training_verification_results(
            controls_present=False,
            command_completed=False,
            metric_captured=False,
            artifact_captured=False,
            run_evidence_captured=False,
            evidence_summary=f"missing_controls={missing_controls}",
        ),
        "artifact_manifest": {"entries": []},
    }


def _write_training_config_snapshot(
    project_path: Path,
    hydra_config_path: str | None,
    hydra_config_name: str | None,
    config_snapshot_path: Path,
) -> str:
    config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    source_config = None
    if hydra_config_path and hydra_config_name:
        source_config = project_path / hydra_config_path / f"{_strip_yaml_suffix(hydra_config_name)}.yaml"
    if source_config is not None and source_config.exists():
        shutil.copyfile(source_config, config_snapshot_path)
    else:
        config_snapshot_path.write_text("config_snapshot: unavailable\n")
    return relative_to_project(str(project_path), config_snapshot_path)


def _parse_training_metrics(*streams: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for stream in streams:
        for line in stream.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                raw_metrics = parsed.get("metrics", parsed)
                if isinstance(raw_metrics, dict):
                    for key, value in raw_metrics.items():
                        if isinstance(value, int | float):
                            metrics[key] = float(value)
                    continue
            for key, value in re.findall(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(-?\d+(?:\.\d+)?)",
                stripped,
            ):
                metrics[key] = float(value)
    return metrics


def _find_training_artifacts(project_path: Path) -> tuple[str, ...]:
    artifact_paths: list[str] = []
    for directory_name in ("checkpoints", "models", "outputs", "artifacts"):
        directory = project_path / directory_name
        if not directory.exists():
            continue
        for artifact_path in directory.rglob("*"):
            if artifact_path.is_file() and artifact_path.suffix in {
                ".ckpt",
                ".pt",
                ".pth",
                ".onnx",
            }:
                artifact_paths.append(relative_to_project(str(project_path), artifact_path))
    return tuple(sorted(artifact_paths))


def _summarize_text(value: str, limit: int = 1000) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "...<truncated>"


def _training_failure_next_actions(failure_reasons: list[str], log_path: str) -> list[str]:
    if not failure_reasons:
        return []
    return [
        f"Inspect {log_path} for the bounded training command output.",
        "Re-run with a small fixture, target metric emission, and checkpoint artifact output.",
    ]


def _bounded_training_verification_results(
    *,
    controls_present: bool,
    command_completed: bool,
    metric_captured: bool,
    artifact_captured: bool,
    run_evidence_captured: bool,
    evidence_summary: str,
) -> list[dict[str, Any]]:
    return [
        {
            "check_name": "bounded_training_controls_present",
            "evidence_type": "observed",
            "source_step": "run_bounded_training",
            "passed": controls_present,
            "evidence": evidence_summary,
        },
        {
            "check_name": "bounded_training_command_completed",
            "evidence_type": "observed",
            "source_step": "run_bounded_training",
            "passed": command_completed,
            "evidence": evidence_summary,
        },
        {
            "check_name": "training_metric_captured",
            "evidence_type": "observed",
            "source_step": "run_bounded_training",
            "passed": metric_captured,
            "evidence": evidence_summary,
        },
        {
            "check_name": "training_artifact_captured",
            "evidence_type": "observed",
            "source_step": "run_bounded_training",
            "passed": artifact_captured,
            "evidence": evidence_summary,
        },
        {
            "check_name": "training_run_evidence_captured",
            "evidence_type": "observed",
            "source_step": "run_bounded_training",
            "passed": run_evidence_captured,
            "evidence": evidence_summary,
        },
    ]


def track_training_in_mlflow(
    project_path: str,
    training_result: dict[str, Any],
    experiment_name: str = "mlops-training",
    tracking_uri: str | None = None,
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log bounded training evidence to a verified local MLflow run."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}

    tracking_uri_result = _resolve_local_mlflow_tracking_uri(path, tracking_uri)
    if not tracking_uri_result["success"]:
        return _mlflow_tracking_blocked_result(
            reason=tracking_uri_result["error"],
            next_action="Use a local path or file:// MLflow tracking URI for this workflow.",
        )
    resolved_tracking_uri = tracking_uri_result["tracking_uri"]

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return _mlflow_tracking_blocked_result(
            reason="MLflow not installed.",
            next_action="Install MLflow in the project environment before tracking training.",
        )

    target_metric = str(training_result.get("target_metric") or "accuracy")
    metrics = {
        key: float(value)
        for key, value in dict(training_result.get("metrics") or {}).items()
        if isinstance(value, int | float)
    }
    checkpoint_paths = list(training_result.get("checkpoint_artifact_paths") or [])
    log_path = training_result.get("log_path")
    config_snapshot_path = training_result.get("config_snapshot_path")
    training_status = str(training_result.get("status") or "unknown")
    run_status = "FINISHED" if training_status == "succeeded" else "FAILED"
    logged_artifacts: list[str] = []

    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)

    run = mlflow.start_run(run_name=run_name)
    run_id = run.info.run_id
    artifact_uri = run.info.artifact_uri
    checkpoint_artifact_uri = None
    try:
        logged_params = _mlflow_training_params(training_result, params)
        for key, value in logged_params.items():
            mlflow.log_param(key, value)
        if metrics:
            mlflow.log_metrics(metrics, step=0)

        for artifact_path, artifact_dest, recorded_path in (
            (log_path, "logs", "training.log"),
            (config_snapshot_path, "configs", "config_snapshot.yaml"),
        ):
            if artifact_path and _log_project_artifact_to_mlflow(path, artifact_path, artifact_dest):
                logged_artifacts.append(recorded_path)

        for checkpoint_path in checkpoint_paths:
            if _log_project_artifact_to_mlflow(path, checkpoint_path, "checkpoints"):
                logged_artifacts.append(checkpoint_path)
                if checkpoint_artifact_uri is None:
                    checkpoint_artifact_uri = (
                        f"{artifact_uri.rstrip('/')}/checkpoints/{Path(checkpoint_path).name}"
                    )

        mlflow.set_tag("training_status", training_status)
        mlflow.set_tag("bounded_training_exit_code", str(training_result.get("exit_code")))
        if training_result.get("failure_reason"):
            mlflow.set_tag("failure_reason", str(training_result["failure_reason"])[:500])
    finally:
        mlflow.end_run(status=run_status)

    verified_run = client.get_run(run_id)
    verified_experiment = client.get_experiment(experiment_id)
    verified_params = dict(verified_run.data.params)
    verified_metrics = dict(verified_run.data.metrics)
    verified_artifacts = client.list_artifacts(run_id)
    artifact_names = {artifact.path for artifact in verified_artifacts}
    has_checkpoint_artifact = any(
        artifact.path.startswith("checkpoints/") or artifact.path == "checkpoints"
        for artifact in verified_artifacts
    )

    verification_results = _mlflow_tracking_verification_results(
        experiment_exists=verified_experiment is not None,
        run_exists=verified_run.info.run_id == run_id,
        tracking_uri_recorded=bool(resolved_tracking_uri),
        artifact_uri_recorded=bool(verified_run.info.artifact_uri),
        params_logged=_mlflow_required_params_logged(verified_params),
        metrics_logged=target_metric in verified_metrics,
        artifacts_logged="logs/training.log" in artifact_names or bool(logged_artifacts),
        checkpoint_logged=has_checkpoint_artifact and checkpoint_artifact_uri is not None,
        run_status_recorded=verified_run.info.status == run_status,
        evidence_summary=(
            f"experiment_id={experiment_id}; run_id={run_id}; "
            f"tracking_uri={resolved_tracking_uri}; artifact_uri={verified_run.info.artifact_uri}; "
            f"run_status={verified_run.info.status}; params={sorted(verified_params)}; "
            f"metrics={verified_metrics}; artifacts={sorted(artifact_names)}"
        ),
    )
    status = "succeeded" if all(item["passed"] for item in verification_results) else "failed"

    artifact_entries = [
        {
            "artifact_type": "mlflow_run",
            "producing_step": "track_training_in_mlflow",
            "state": "generated",
            "uri": verified_run.info.artifact_uri,
        }
    ]
    if checkpoint_artifact_uri:
        artifact_entries.append(
            {
                "artifact_type": "mlflow_checkpoint_or_model_artifact",
                "producing_step": "track_training_in_mlflow",
                "state": "generated",
                "path": checkpoint_paths[0] if checkpoint_paths else None,
                "uri": checkpoint_artifact_uri,
            }
        )

    return {
        "success": True,
        "status": status,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "tracking_uri": resolved_tracking_uri,
        "artifact_uri": verified_run.info.artifact_uri,
        "run_status": verified_run.info.status,
        "params": verified_params,
        "metrics": verified_metrics,
        "logged_artifacts": logged_artifacts,
        "checkpoint_artifact_uri": checkpoint_artifact_uri,
        "failure_reason": None if status == "succeeded" else "MLflow run verification failed.",
        "next_actions": []
        if status == "succeeded"
        else ["Inspect the local MLflow run and verify params, metrics, and artifacts."],
        "verification_results": verification_results,
        "artifact_manifest": {"entries": artifact_entries},
    }


def _resolve_local_mlflow_tracking_uri(
    project_path: Path, tracking_uri: str | None
) -> dict[str, Any]:
    if tracking_uri is None:
        return {"success": True, "tracking_uri": (project_path / "mlruns").resolve().as_uri()}
    if tracking_uri.startswith("file://"):
        return {"success": True, "tracking_uri": tracking_uri}
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", tracking_uri):
        return {
            "success": False,
            "error": f"Remote MLflow tracking URI is not allowed: {tracking_uri}",
        }
    tracking_path = Path(tracking_uri)
    if not tracking_path.is_absolute():
        tracking_path = project_path / tracking_path
    return {"success": True, "tracking_uri": tracking_path.resolve().as_uri()}


def _mlflow_tracking_blocked_result(reason: str, next_action: str) -> dict[str, Any]:
    return {
        "success": True,
        "status": "blocked",
        "failure_reason": reason,
        "next_actions": [next_action],
        "verification_results": _mlflow_tracking_verification_results(
            experiment_exists=False,
            run_exists=False,
            tracking_uri_recorded=False,
            artifact_uri_recorded=False,
            params_logged=False,
            metrics_logged=False,
            artifacts_logged=False,
            checkpoint_logged=False,
            run_status_recorded=False,
            evidence_summary=reason,
        ),
        "artifact_manifest": {"entries": []},
    }


def _mlflow_training_params(
    training_result: dict[str, Any],
    extra_params: dict[str, Any] | None,
) -> dict[str, str]:
    params: dict[str, Any] = dict(extra_params or {})
    params.update(
        {
            "command": " ".join(str(part) for part in training_result.get("command") or ()),
            "effective_overrides": ",".join(
                str(override) for override in training_result.get("effective_overrides") or ()
            ),
            "target_metric": training_result.get("target_metric"),
            "training_status": training_result.get("status"),
            "exit_code": training_result.get("exit_code"),
            "duration_seconds": training_result.get("duration_seconds"),
            "log_path": training_result.get("log_path"),
            "config_snapshot_path": training_result.get("config_snapshot_path"),
            "checkpoint_artifact_paths": ",".join(
                str(path) for path in training_result.get("checkpoint_artifact_paths") or ()
            ),
        }
    )
    return {
        key: _stringify_mlflow_param(value)
        for key, value in params.items()
        if value is not None and value != ""
    }


def _stringify_mlflow_param(value: Any) -> str:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, int | float | bool):
        return str(value)
    return json.dumps(value, sort_keys=True)[:500]


def _log_project_artifact_to_mlflow(
    project_path: Path,
    artifact_path: str,
    artifact_dest: str,
) -> bool:
    path = Path(artifact_path)
    if not path.is_absolute():
        path = project_path / path
    if not path.exists():
        return False

    import mlflow

    if path.is_dir():
        mlflow.log_artifacts(str(path), artifact_dest)
    else:
        mlflow.log_artifact(str(path), artifact_dest)
    return True


def _mlflow_required_params_logged(params: dict[str, str]) -> bool:
    required_params = {
        "command",
        "effective_overrides",
        "target_metric",
        "training_status",
        "exit_code",
        "duration_seconds",
        "log_path",
        "checkpoint_artifact_paths",
        "timeout_seconds",
        "max_epochs",
        "device",
        "data_subset",
    }
    return required_params.issubset(params)


def _mlflow_tracking_verification_results(
    *,
    experiment_exists: bool,
    run_exists: bool,
    tracking_uri_recorded: bool,
    artifact_uri_recorded: bool,
    params_logged: bool,
    metrics_logged: bool,
    artifacts_logged: bool,
    checkpoint_logged: bool,
    run_status_recorded: bool,
    evidence_summary: str,
) -> list[dict[str, Any]]:
    checks = (
        ("mlflow_experiment_exists", experiment_exists),
        ("mlflow_run_exists", run_exists),
        ("mlflow_tracking_uri_recorded", tracking_uri_recorded),
        ("mlflow_artifact_uri_recorded", artifact_uri_recorded),
        ("mlflow_params_logged", params_logged),
        ("mlflow_metrics_logged", metrics_logged),
        ("mlflow_artifacts_logged", artifacts_logged),
        ("mlflow_checkpoint_artifact_logged", checkpoint_logged),
        ("mlflow_run_status_recorded", run_status_recorded),
    )
    return [
        {
            "check_name": check_name,
            "evidence_type": "observed",
            "source_step": "track_training_in_mlflow",
            "passed": passed,
            "evidence": evidence_summary,
        }
        for check_name, passed in checks
    ]


def analyze_training_results(
    project_path: str,
    experiment_name: str,
    target_metric: str = "accuracy",
    target_value: float = 0.85,
) -> dict[str, Any]:
    """Analyze training results and suggest improvements."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=[f"metrics.{target_metric} DESC"]
        )

        if not runs:
            return {"success": False, "error": "No runs found"}

        best_run = runs[0]
        current_value = best_run.data.metrics.get(target_metric, 0)
        gap = target_value - current_value

        analysis = {
            "success": True,
            "best_run_id": best_run.info.run_id,
            "current_metrics": best_run.data.metrics,
            "current_params": best_run.data.params,
            "target_metric": target_metric,
            "target_value": target_value,
            "current_value": current_value,
            "gap": gap,
            "threshold_met": current_value >= target_value,
            "total_runs": len(runs),
        }

        # Generate suggestions based on gap
        suggestions = []

        if gap > 0.1:
            suggestions.extend(
                [
                    "Consider increasing model complexity",
                    "Add more data augmentation",
                    "Try a different architecture",
                    "Increase training epochs significantly",
                ]
            )
        elif gap > 0.05:
            suggestions.extend(
                [
                    "Fine-tune learning rate (try 0.0001 or 0.0005)",
                    "Add regularization (dropout, weight decay)",
                    "Use learning rate scheduling",
                    "Increase batch size if memory allows",
                ]
            )
        elif gap > 0:
            suggestions.extend(
                [
                    "Small hyperparameter adjustments may help",
                    "Try ensemble methods",
                    "Fine-tune for a few more epochs",
                    "Consider test-time augmentation",
                ]
            )

        analysis["suggestions"] = suggestions

        return analysis

    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def suggest_improvements(
    current_metrics: dict[str, float],
    current_config: dict[str, Any],
    target_accuracy: float,
    attempt_number: int = 1,
) -> dict[str, Any]:
    """Suggest improvements based on training results."""
    current_accuracy = current_metrics.get("accuracy", 0)
    gap = target_accuracy - current_accuracy

    suggestions = {
        "success": True,
        "current_accuracy": current_accuracy,
        "target_accuracy": target_accuracy,
        "gap": gap,
        "attempt": attempt_number,
        "config_changes": {},
        "reasoning": [],
    }

    # Learning rate adjustments
    current_lr = current_config.get("learning_rate", 0.001)
    if gap > 0.1:
        # Large gap - try more aggressive changes
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.5
        suggestions["config_changes"]["epochs"] = current_config.get("epochs", 10) * 2
        suggestions["reasoning"].append(
            f"Large accuracy gap ({gap:.2%}). Reducing LR to {current_lr * 0.5} and doubling epochs."
        )
    elif gap > 0.05:
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.7
        suggestions["config_changes"]["epochs"] = int(current_config.get("epochs", 10) * 1.5)
        suggestions["reasoning"].append(
            f"Moderate gap ({gap:.2%}). Adjusting LR to {current_lr * 0.7}."
        )
    else:
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.9
        suggestions["reasoning"].append(
            f"Small gap ({gap:.2%}). Fine-tuning LR to {current_lr * 0.9}."
        )

    # Batch size adjustments based on attempt
    if attempt_number > 1:
        current_batch = current_config.get("batch_size", 32)
        suggestions["config_changes"]["batch_size"] = min(current_batch * 2, 128)
        suggestions["reasoning"].append(
            f"Attempt {attempt_number}: Increasing batch size to {min(current_batch * 2, 128)}."
        )

    # Add regularization on later attempts
    if attempt_number >= 2:
        suggestions["config_changes"]["dropout"] = min(
            current_config.get("dropout", 0.3) + 0.1, 0.5
        )
        suggestions["reasoning"].append("Adding more regularization to prevent overfitting.")

    # Add augmentation suggestion
    if not current_config.get("augmentation", False):
        suggestions["config_changes"]["augmentation"] = True
        suggestions["reasoning"].append("Enabling data augmentation for better generalization.")

    return suggestions


def check_accuracy_threshold(
    experiment_name: str, threshold: float, metric_name: str = "accuracy"
) -> dict[str, Any]:
    """Check if accuracy threshold is met."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)

        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1,
        )

        if not runs:
            return {
                "success": True,
                "threshold_met": False,
                "current_value": 0,
                "threshold": threshold,
                "message": "No runs found in experiment",
            }

        best_run = runs[0]
        current_value = best_run.data.metrics.get(metric_name, 0)
        threshold_met = current_value >= threshold

        return {
            "success": True,
            "threshold_met": threshold_met,
            "current_value": current_value,
            "threshold": threshold,
            "run_id": best_run.info.run_id,
            "gap": threshold - current_value if not threshold_met else 0,
            "message": f"{'✅ Threshold met!' if threshold_met else f'❌ Below threshold by {threshold - current_value:.2%}'}",
        }

    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Data Quality Tools ---


def validate_dataset(
    dataset_path: str,
    dataset_type: str = "auto",
    checks: list[str] | None = None,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Validate ML dataset for quality issues.

    Performs various data quality checks including:
    - Missing values detection
    - Duplicate detection
    - Class balance analysis
    - Data type validation
    - Outlier detection (for numeric data)
    - Image validity (for image datasets)
    """
    path = Path(dataset_path)

    if not path.exists():
        return {"success": False, "error": f"Dataset path {dataset_path} does not exist"}

    # Auto-detect dataset type if needed
    if dataset_type == "auto":
        if path.is_dir():
            # Check if it's an image directory
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            sample_files = list(path.glob("**/*"))[:100]
            image_files = [f for f in sample_files if f.suffix.lower() in image_extensions]
            if len(image_files) > len(sample_files) * 0.5:
                dataset_type = "images"
            else:
                dataset_type = "directory"
        elif path.suffix.lower() == ".csv":
            dataset_type = "csv"
        elif path.suffix.lower() == ".parquet":
            dataset_type = "parquet"
        elif path.suffix.lower() == ".json":
            dataset_type = "json"
        else:
            return {
                "success": False,
                "error": f"Cannot auto-detect dataset type for {path.suffix}",
            }

    validation_results = {
        "success": True,
        "dataset_path": str(path),
        "dataset_type": dataset_type,
        "checks_performed": [],
        "issues": [],
        "warnings": [],
        "statistics": {},
    }

    # Define default checks based on dataset type
    if checks is None:
        if dataset_type in ["csv", "parquet", "json"]:
            checks = ["missing_values", "duplicates", "class_balance", "data_types", "outliers"]
        elif dataset_type == "images":
            checks = ["image_validity", "class_balance"]
        else:
            checks = ["missing_values", "duplicates"]

    try:
        if dataset_type == "images":
            validation_results = _validate_image_dataset(
                path, checks, sample_size, validation_results
            )
        elif dataset_type in ["csv", "parquet", "json"]:
            validation_results = _validate_tabular_dataset(
                path, dataset_type, checks, sample_size, validation_results
            )
        else:
            validation_results["warnings"].append(
                f"Limited validation available for dataset type: {dataset_type}"
            )

        # Determine overall validity
        critical_issues = [
            i for i in validation_results["issues"] if i.get("severity") == "critical"
        ]
        validation_results["is_valid"] = len(critical_issues) == 0
        validation_results["total_issues"] = len(validation_results["issues"])
        validation_results["total_warnings"] = len(validation_results["warnings"])

        if validation_results["is_valid"]:
            validation_results["message"] = "Dataset validation passed"
        else:
            validation_results["message"] = (
                f"Dataset has {len(critical_issues)} critical issue(s) that need attention"
            )

        return validation_results

    except Exception as e:
        return {"success": False, "error": str(e)}


def _validate_image_dataset(
    path: Path,
    checks: list[str],
    sample_size: int | None,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Validate an image dataset directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

    # Collect all image files
    all_images = []
    class_counts = {}

    for item in path.iterdir():
        if item.is_dir():
            # Assume subdirectories are class labels
            class_name = item.name
            class_images = [f for f in item.glob("*") if f.suffix.lower() in image_extensions]
            class_counts[class_name] = len(class_images)
            all_images.extend(class_images)
        elif item.suffix.lower() in image_extensions:
            all_images.append(item)

    results["statistics"]["total_images"] = len(all_images)
    results["statistics"]["classes"] = class_counts if class_counts else None

    # Apply sample size
    if sample_size and len(all_images) > sample_size:
        import random

        all_images = random.sample(all_images, sample_size)
        results["statistics"]["sampled"] = True
        results["statistics"]["sample_size"] = sample_size

    # Check: Image validity
    if "image_validity" in checks:
        results["checks_performed"].append("image_validity")
        invalid_images = []
        corrupted_images = []

        for img_path in all_images:
            try:
                # Try to open and verify the image
                from PIL import Image

                with Image.open(img_path) as img:
                    img.verify()
            except ImportError:
                results["warnings"].append("Pillow not installed - image validity check skipped")
                break
            except Exception as e:
                if "cannot identify" in str(e).lower():
                    invalid_images.append(str(img_path))
                else:
                    corrupted_images.append({"path": str(img_path), "error": str(e)})

        if invalid_images:
            results["issues"].append(
                {
                    "check": "image_validity",
                    "severity": "critical",
                    "message": f"Found {len(invalid_images)} invalid/unreadable images",
                    "details": invalid_images[:10],  # Show first 10
                }
            )

        if corrupted_images:
            results["issues"].append(
                {
                    "check": "image_validity",
                    "severity": "warning",
                    "message": f"Found {len(corrupted_images)} potentially corrupted images",
                    "details": corrupted_images[:10],
                }
            )

    # Check: Class balance
    if "class_balance" in checks and class_counts:
        results["checks_performed"].append("class_balance")
        if len(class_counts) > 1:
            counts = list(class_counts.values())
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            results["statistics"]["class_imbalance_ratio"] = round(imbalance_ratio, 2)

            if imbalance_ratio > 10:
                results["issues"].append(
                    {
                        "check": "class_balance",
                        "severity": "warning",
                        "message": f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)",
                        "details": class_counts,
                    }
                )
            elif imbalance_ratio > 3:
                results["warnings"].append(
                    f"Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)"
                )

    return results


def _validate_tabular_dataset(
    path: Path,
    dataset_type: str,
    checks: list[str],
    sample_size: int | None,
    results: dict[str, Any],
) -> dict[str, Any]:
    """Validate a tabular dataset (CSV, Parquet, JSON)."""
    try:
        import pandas as pd
    except ImportError:
        results["success"] = False
        results["error"] = "pandas not installed. Run: pip install pandas"
        return results

    # Load dataset
    try:
        if dataset_type == "csv":
            df = pd.read_csv(path, nrows=sample_size)
        elif dataset_type == "parquet":
            df = pd.read_parquet(path)
            if sample_size:
                df = df.head(sample_size)
        elif dataset_type == "json":
            df = pd.read_json(path, lines=True, nrows=sample_size)
    except Exception as e:
        results["success"] = False
        results["error"] = f"Failed to load dataset: {str(e)}"
        return results

    results["statistics"]["total_rows"] = len(df)
    results["statistics"]["total_columns"] = len(df.columns)
    results["statistics"]["columns"] = list(df.columns)

    # Check: Missing values
    if "missing_values" in checks:
        results["checks_performed"].append("missing_values")
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0]

        if len(missing_cols) > 0:
            total_missing = missing_cols.sum()
            missing_pct = (total_missing / (len(df) * len(df.columns))) * 100
            results["statistics"]["missing_values_pct"] = round(missing_pct, 2)

            missing_details = {col: int(count) for col, count in missing_cols.items()}

            if missing_pct > 20:
                results["issues"].append(
                    {
                        "check": "missing_values",
                        "severity": "critical",
                        "message": f"High proportion of missing values ({missing_pct:.1f}%)",
                        "details": missing_details,
                    }
                )
            elif missing_pct > 5:
                results["issues"].append(
                    {
                        "check": "missing_values",
                        "severity": "warning",
                        "message": f"Moderate missing values ({missing_pct:.1f}%)",
                        "details": missing_details,
                    }
                )
            else:
                results["warnings"].append(f"Some missing values detected ({missing_pct:.1f}%)")

    # Check: Duplicates
    if "duplicates" in checks:
        results["checks_performed"].append("duplicates")
        duplicate_count = df.duplicated().sum()

        if duplicate_count > 0:
            dup_pct = (duplicate_count / len(df)) * 100
            results["statistics"]["duplicate_rows"] = int(duplicate_count)
            results["statistics"]["duplicate_pct"] = round(dup_pct, 2)

            if dup_pct > 10:
                results["issues"].append(
                    {
                        "check": "duplicates",
                        "severity": "warning",
                        "message": f"High number of duplicate rows ({duplicate_count}, {dup_pct:.1f}%)",
                    }
                )
            else:
                results["warnings"].append(
                    f"Found {duplicate_count} duplicate rows ({dup_pct:.1f}%)"
                )

    # Check: Data types
    if "data_types" in checks:
        results["checks_performed"].append("data_types")
        dtype_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
        results["statistics"]["data_types"] = dtype_info

        # Check for object columns that might be numeric
        for col in df.select_dtypes(include=["object"]).columns:
            # Try to convert to numeric
            numeric_converted = pd.to_numeric(df[col], errors="coerce")
            valid_numeric = numeric_converted.notna().sum()
            if valid_numeric > len(df) * 0.8:
                results["warnings"].append(f"Column '{col}' appears numeric but stored as text")

    # Check: Class balance (for categorical target columns)
    if "class_balance" in checks:
        results["checks_performed"].append("class_balance")
        # Look for common target column names
        target_candidates = ["label", "target", "class", "y", "category"]
        target_col = None

        for col in df.columns:
            if col.lower() in target_candidates:
                target_col = col
                break

        if target_col is None:
            # Use the last column if it's categorical
            last_col = df.columns[-1]
            if df[last_col].dtype == "object" or df[last_col].nunique() < 20:
                target_col = last_col

        if target_col:
            class_counts = df[target_col].value_counts().to_dict()
            results["statistics"]["target_column"] = target_col
            results["statistics"]["class_distribution"] = {
                str(k): int(v) for k, v in class_counts.items()
            }

            if len(class_counts) > 1:
                counts = list(class_counts.values())
                min_count = min(counts)
                max_count = max(counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
                results["statistics"]["class_imbalance_ratio"] = round(imbalance_ratio, 2)

                if imbalance_ratio > 10:
                    results["issues"].append(
                        {
                            "check": "class_balance",
                            "severity": "warning",
                            "message": f"Severe class imbalance in '{target_col}' (ratio: {imbalance_ratio:.1f}:1)",
                            "details": {str(k): int(v) for k, v in class_counts.items()},
                        }
                    )

    # Check: Outliers (for numeric columns)
    if "outliers" in checks:
        results["checks_performed"].append("outliers")
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        outlier_info = {}

        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(df)) * 100
                outlier_info[col] = {
                    "count": int(len(outliers)),
                    "percentage": round(outlier_pct, 2),
                }

        if outlier_info:
            results["statistics"]["outliers"] = outlier_info
            total_outlier_cols = len(outlier_info)

            if total_outlier_cols > len(numeric_cols) * 0.5:
                results["warnings"].append(
                    f"Outliers detected in {total_outlier_cols} of {len(numeric_cols)} numeric columns"
                )

    return results


def create_expectation_suite(
    project_path: str,
    suite_name: str,
    expectations: list[dict[str, Any]],
    output_dir: str = "great_expectations/expectations",
) -> dict[str, Any]:
    """Create a Great Expectations expectation suite for data validation.

    Creates a JSON expectation suite file that can be used with Great Expectations
    for validating ML datasets. The suite defines data quality expectations that
    can be run against datasets to ensure data quality.

    Args:
        project_path: Path to the ML project
        suite_name: Name of the expectation suite
        expectations: List of expectation configurations. Each should have:
            - expectation_type: GE expectation name (e.g., 'expect_column_values_to_not_be_null')
            - column: Column to apply expectation to (optional, depends on expectation type)
            - kwargs: Additional expectation parameters (optional)
            - severity: 'error', 'warning', or 'info' (optional, default: 'error')
            - description: Human-readable description (optional)
        output_dir: Directory to save the suite (relative to project_path)

    Returns:
        Dict with success status and suite details
    """
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    if not expectations:
        return {"success": False, "error": "At least one expectation is required"}

    # Validate and normalize expectations
    normalized_expectations = []
    validation_errors = []

    for i, exp in enumerate(expectations):
        if not isinstance(exp, dict):
            validation_errors.append(f"Expectation {i}: must be a dictionary")
            continue

        exp_type = exp.get("expectation_type")
        if not exp_type:
            validation_errors.append(f"Expectation {i}: missing 'expectation_type'")
            continue

        # Build the normalized expectation in GE format
        ge_expectation = {
            "expectation_type": exp_type,
            "kwargs": {},
            "meta": {},
        }

        # Add column if specified
        column = exp.get("column")
        if column:
            ge_expectation["kwargs"]["column"] = column

        # Add additional kwargs
        extra_kwargs = exp.get("kwargs", {})
        if isinstance(extra_kwargs, dict):
            ge_expectation["kwargs"].update(extra_kwargs)

        # Store metadata
        severity = exp.get("severity", "error")
        if severity not in ("error", "warning", "info"):
            severity = "error"
        ge_expectation["meta"]["severity"] = severity

        description = exp.get("description")
        if description:
            ge_expectation["meta"]["description"] = description
        else:
            # Generate default description
            col_str = f" on column '{column}'" if column else ""
            ge_expectation["meta"]["description"] = f"{exp_type}{col_str}"

        normalized_expectations.append(ge_expectation)

    if validation_errors:
        return {
            "success": False,
            "error": "Invalid expectations",
            "validation_errors": validation_errors,
        }

    # Create the expectation suite structure
    suite = {
        "expectation_suite_name": suite_name,
        "ge_cloud_id": None,
        "expectations": normalized_expectations,
        "meta": {
            "great_expectations_version": "0.18.0",
            "notes": f"Expectation suite created by Auto-MLOps for {suite_name}",
        },
    }

    # Create output directory
    output_path = path / output_dir
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return {"success": False, "error": f"Failed to create output directory: {str(e)}"}

    # Write the suite file
    suite_file = output_path / f"{suite_name}.json"
    try:
        with open(suite_file, "w") as f:
            json.dump(suite, f, indent=2)
    except Exception as e:
        return {"success": False, "error": f"Failed to write suite file: {str(e)}"}

    # Generate summary
    expectation_summary = {}
    for exp in normalized_expectations:
        exp_type = exp["expectation_type"]
        expectation_summary[exp_type] = expectation_summary.get(exp_type, 0) + 1

    return {
        "success": True,
        "suite_name": suite_name,
        "suite_path": str(suite_file),
        "expectation_count": len(normalized_expectations),
        "expectation_types": expectation_summary,
        "message": f"Created expectation suite '{suite_name}' with {len(normalized_expectations)} expectations",
    }


def check_data_quality(
    dataset_path: str,
    expectations: list[dict[str, Any]] | None = None,
    include_statistics: bool = True,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Check data quality using Great Expectations-based validation.

    Validates a dataset against specified expectations or runs basic quality checks.
    Uses the GreatExpectationsValidator from the data_quality module with fallback
    to pandas-based checks if Great Expectations is not installed.

    Args:
        dataset_path: Path to the dataset file (CSV, Parquet, or JSON)
        expectations: Optional list of expectations. Each expectation is a dict with:
            - expectation_type: GE expectation name (e.g., 'expect_column_values_to_not_be_null')
            - column: Column to apply expectation to (optional)
            - kwargs: Additional expectation parameters (optional)
            - severity: 'error', 'warning', or 'info' (optional, default: 'error')
            If not provided, runs basic checks (null values, duplicates, row count)
        include_statistics: Whether to include detailed dataset statistics
        fail_on_error: If True, returns success=False when any ERROR-severity check fails

    Returns:
        Dict with validation results, statistics, and recommendations
    """
    path = Path(dataset_path)

    if not path.exists():
        return {"success": False, "error": f"Dataset path {dataset_path} does not exist"}

    # Determine file type
    suffix = path.suffix.lower()
    if suffix not in (".csv", ".parquet", ".json"):
        return {
            "success": False,
            "error": f"Unsupported file type: {suffix}. Supported types: .csv, .parquet, .json",
        }

    try:
        # Import required modules
        import pandas as pd

        from data_quality.validator import GreatExpectationsValidator

        # Load dataset
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        else:  # .json
            df = pd.read_json(path)

        # Initialize validator
        validator = GreatExpectationsValidator()

        # Configure expectations
        if expectations:
            # Use provided expectations
            for exp in expectations:
                exp_type = exp.get("expectation_type")
                if not exp_type:
                    continue

                column = exp.get("column")
                kwargs = exp.get("kwargs", {})
                severity_str = exp.get("severity", "error").lower()

                # Map severity string to enum
                from data_quality.models import ValidationSeverity

                severity_map = {
                    "error": ValidationSeverity.ERROR,
                    "warning": ValidationSeverity.WARNING,
                    "info": ValidationSeverity.INFO,
                }
                severity = severity_map.get(severity_str, ValidationSeverity.ERROR)

                description = exp.get("description")

                validator.add_expectation(
                    expectation_type=exp_type,
                    column=column,
                    kwargs=kwargs,
                    severity=severity,
                    description=description,
                )
        else:
            # Run basic quality checks if no expectations provided
            from data_quality.models import ValidationSeverity

            # Add basic checks for all columns
            for column in df.columns:
                # Check for null values
                validator.add_not_null_expectation(
                    column=column,
                    mostly=0.95,  # Allow 5% nulls by default
                    severity=ValidationSeverity.WARNING,
                )

            # Check for duplicate rows
            validator.add_no_duplicates_expectation(severity=ValidationSeverity.WARNING)

            # Check row count (at least 1 row)
            validator.add_table_row_count_expectation(
                min_value=1, severity=ValidationSeverity.ERROR
            )

        # Run validation
        report = validator.validate(
            df, dataset_name=path.name, include_statistics=include_statistics
        )

        # Convert report to dict for JSON serialization
        result = {
            "success": True,
            "report_id": report.report_id,
            "dataset_name": report.dataset_name,
            "generated_at": report.generated_at,
            "overall_score": report.overall_score,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "warning_count": report.warning_count,
            "recommendations": report.recommendations,
            "is_valid": report.failed_checks == 0,
        }

        # Add validation results summary
        validation_summary = []
        for vr in report.validation_results:
            validation_summary.append(
                {
                    "rule_name": vr.rule_name,
                    "status": vr.status.value,
                    "severity": vr.severity.value,
                    "column": vr.column,
                    "message": vr.message,
                    "failed_rows": vr.failed_rows,
                    "failed_percentage": vr.failed_percentage,
                }
            )
        result["validation_results"] = validation_summary

        # Add statistics if included
        if include_statistics:
            stats = report.statistics
            result["statistics"] = {
                "row_count": stats.row_count,
                "column_count": stats.column_count,
                "total_cells": stats.total_cells,
                "total_missing": stats.total_missing,
                "missing_percentage": stats.missing_percentage,
                "duplicate_rows": stats.duplicate_rows,
                "duplicate_percentage": stats.duplicate_percentage,
                "memory_usage_bytes": stats.memory_usage_bytes,
                "columns": [
                    {
                        "column_name": col.column_name,
                        "data_type": col.data_type.value,
                        "null_count": col.null_count,
                        "null_percentage": col.null_percentage,
                        "unique_count": col.unique_count,
                        "unique_percentage": col.unique_percentage,
                    }
                    for col in stats.columns
                ],
            }

        # Determine overall success based on fail_on_error flag
        if fail_on_error:
            # Check if any ERROR-severity checks failed
            error_failures = sum(
                1
                for vr in report.validation_results
                if vr.status.value == "failed" and vr.severity.value == "error"
            )
            if error_failures > 0:
                result["success"] = False
                result["message"] = (
                    f"Data quality check failed: {error_failures} error-level check(s) failed"
                )
            else:
                result["message"] = "Data quality check passed"
        else:
            result["message"] = (
                f"Data quality check completed with score {report.overall_score:.1f}/100"
            )

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas and optionally great_expectations.",
        }
    except Exception as e:
        return {"success": False, "error": f"Data quality check failed: {str(e)}"}


def profile_dataset(
    dataset_path: str,
    dataset_name: str | None = None,
    include_column_stats: bool = True,
) -> dict[str, Any]:
    """
    Profile a dataset to get comprehensive statistics.

    Args:
        dataset_path: Path to the dataset file (CSV, Parquet, or JSON)
        dataset_name: Name for the dataset in the report. If not provided, uses filename.
        include_column_stats: Whether to include detailed column-level statistics

    Returns:
        Dictionary with profiling results including statistics and metadata
    """
    try:
        import pandas as pd

        from data_quality import DataProfiler, DatasetStatistics

        path = Path(dataset_path)
        if not path.exists():
            return {"success": False, "error": f"Dataset file {dataset_path} does not exist"}

        # Determine file type and load data
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        elif suffix == ".json":
            df = pd.read_json(dataset_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {suffix}. Supported: .csv, .parquet, .json",
            }

        # Use filename as dataset name if not provided
        name = dataset_name or path.stem

        # Profile the dataset
        profiler = DataProfiler()
        stats: DatasetStatistics = profiler.profile(df, dataset_name=name)

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "dataset_name": name,
            "dataset_path": str(path.absolute()),
            "file_type": suffix[1:],
            "statistics": {
                "row_count": stats.row_count,
                "column_count": stats.column_count,
                "total_cells": stats.total_cells,
                "total_missing": stats.total_missing,
                "missing_percentage": stats.missing_percentage,
                "duplicate_rows": stats.duplicate_rows,
                "duplicate_percentage": stats.duplicate_percentage,
                "memory_usage_bytes": stats.memory_usage_bytes,
                "memory_usage_mb": round(stats.memory_usage_bytes / (1024 * 1024), 2),
            },
        }

        if include_column_stats:
            result["columns"] = [
                {
                    "name": col.column_name,
                    "data_type": col.data_type.value,
                    "total_count": col.total_count,
                    "null_count": col.null_count,
                    "null_percentage": col.null_percentage,
                    "unique_count": col.unique_count,
                    "unique_percentage": col.unique_percentage,
                    "mean": col.mean,
                    "median": col.median,
                    "std": col.std,
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                    "q1": col.q1,
                    "q3": col.q3,
                    "top_values": col.top_values,
                }
                for col in stats.columns
            ]

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas.",
        }
    except Exception as e:
        return {"success": False, "error": f"Dataset profiling failed: {str(e)}"}


def detect_anomalies(
    dataset_path: str,
    dataset_name: str | None = None,
    methods: list[str] | None = None,
    outlier_threshold: float = 1.5,
    zscore_threshold: float = 3.0,
) -> dict[str, Any]:
    """
    Detect anomalies in a dataset using statistical methods.

    Args:
        dataset_path: Path to the dataset file (CSV, Parquet, or JSON)
        dataset_name: Name for the dataset in the report. If not provided, uses filename.
        methods: List of detection methods: 'iqr', 'zscore', 'missing', 'duplicates'.
                 If None, uses all methods.
        outlier_threshold: IQR multiplier for outlier detection (default 1.5)
        zscore_threshold: Z-score threshold for outlier detection (default 3.0)

    Returns:
        Dictionary with anomaly detection results
    """
    try:
        import pandas as pd

        from data_quality import AnomalyDetectionResult, DataProfiler

        path = Path(dataset_path)
        if not path.exists():
            return {"success": False, "error": f"Dataset file {dataset_path} does not exist"}

        # Determine file type and load data
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        elif suffix == ".json":
            df = pd.read_json(dataset_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {suffix}. Supported: .csv, .parquet, .json",
            }

        # Use filename as dataset name if not provided
        name = dataset_name or path.stem

        # Create profiler with custom thresholds
        profiler = DataProfiler(
            outlier_threshold=outlier_threshold,
            zscore_threshold=zscore_threshold,
        )

        # Detect anomalies
        result_data: AnomalyDetectionResult = profiler.detect_anomalies(
            df, dataset_name=name, methods=methods
        )

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "dataset_name": result_data.dataset_name,
            "dataset_path": str(path.absolute()),
            "analyzed_at": result_data.analyzed_at,
            "total_anomalies": result_data.total_anomalies,
            "anomalies_by_type": result_data.anomalies_by_type,
            "affected_rows": result_data.affected_rows,
            "affected_percentage": result_data.affected_percentage,
            "methods_used": methods or ["iqr", "zscore", "missing", "duplicates"],
            "thresholds": {
                "iqr_multiplier": outlier_threshold,
                "zscore_threshold": zscore_threshold,
            },
            "anomalies": [
                {
                    "type": a.anomaly_type.value,
                    "column": a.column,
                    "description": a.description,
                    "severity": a.severity.value,
                    "row_indices_sample": a.row_indices[:10],
                    "total_affected_rows": len(a.row_indices),
                    "value": a.value,
                    "expected_range": a.expected_range,
                }
                for a in result_data.anomalies
            ],
        }

        # Add summary message
        if result_data.total_anomalies == 0:
            result["message"] = "No anomalies detected in the dataset"
        else:
            result["message"] = (
                f"Found {result_data.total_anomalies} anomaly type(s) "
                f"affecting {result_data.affected_rows} rows ({result_data.affected_percentage}%)"
            )

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas and numpy.",
        }
    except Exception as e:
        return {"success": False, "error": f"Anomaly detection failed: {str(e)}"}


def validate_schema(
    dataset_path: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Validate a dataset against a defined schema.

    Args:
        dataset_path: Path to the dataset file (CSV, Parquet, or JSON)
        schema: Schema definition with fields, types, and constraints

    Returns:
        Dictionary with schema validation results
    """
    try:
        import pandas as pd

        from data_quality import DataSchema, DataType, SchemaField
        from data_quality.validators import DataValidator

        path = Path(dataset_path)
        if not path.exists():
            return {"success": False, "error": f"Dataset file {dataset_path} does not exist"}

        # Determine file type and load data
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(dataset_path)
        elif suffix == ".parquet":
            df = pd.read_parquet(dataset_path)
        elif suffix == ".json":
            df = pd.read_json(dataset_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type: {suffix}. Supported: .csv, .parquet, .json",
            }

        # Parse schema definition
        schema_name = schema.get("schema_name", "schema")
        version = schema.get("version", "1.0")
        strict = schema.get("strict", False)
        fields_data = schema.get("fields", [])

        # Map string data types to DataType enum
        type_mapping = {
            "numeric": DataType.NUMERIC,
            "categorical": DataType.CATEGORICAL,
            "text": DataType.TEXT,
            "datetime": DataType.DATETIME,
            "boolean": DataType.BOOLEAN,
            "image": DataType.IMAGE,
            "unknown": DataType.UNKNOWN,
        }

        # Build schema fields
        fields: list[SchemaField] = []
        for field_def in fields_data:
            data_type_str = field_def.get("data_type", "unknown").lower()
            data_type = type_mapping.get(data_type_str, DataType.UNKNOWN)

            field = SchemaField(
                name=field_def["name"],
                data_type=data_type,
                nullable=field_def.get("nullable", True),
                unique=field_def.get("unique", False),
                min_value=field_def.get("min_value"),
                max_value=field_def.get("max_value"),
                allowed_values=field_def.get("allowed_values"),
                pattern=field_def.get("pattern"),
            )
            fields.append(field)

        # Create DataSchema
        data_schema = DataSchema(
            schema_name=schema_name,
            version=version,
            fields=fields,
            strict=strict,
        )

        # Create validator and validate schema
        validator = DataValidator()
        validation_result = validator.validate_schema(df, data_schema)

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "is_valid": validation_result.is_valid,
            "schema_name": validation_result.schema_name,
            "dataset_name": validation_result.dataset_name,
            "dataset_path": str(path.absolute()),
            "validated_at": validation_result.validated_at,
            "missing_columns": validation_result.missing_columns,
            "extra_columns": validation_result.extra_columns,
            "type_mismatches": validation_result.type_mismatches,
            "constraint_violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "status": v.status.value,
                    "severity": v.severity.value,
                    "column": v.column,
                    "message": v.message,
                    "failed_rows": v.failed_rows,
                    "failed_percentage": v.failed_percentage,
                }
                for v in validation_result.constraint_violations
            ],
        }

        # Add summary
        issues = []
        if validation_result.missing_columns:
            issues.append(f"{len(validation_result.missing_columns)} missing columns")
        if validation_result.extra_columns and strict:
            issues.append(f"{len(validation_result.extra_columns)} extra columns")
        if validation_result.type_mismatches:
            issues.append(f"{len(validation_result.type_mismatches)} type mismatches")
        if validation_result.constraint_violations:
            issues.append(f"{len(validation_result.constraint_violations)} constraint violations")

        if validation_result.is_valid:
            result["message"] = "Dataset matches schema"
        else:
            result["message"] = f"Schema validation failed: {', '.join(issues)}"

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas.",
        }
    except KeyError as e:
        return {
            "success": False,
            "error": f"Invalid schema definition: missing required field {str(e)}",
        }
    except Exception as e:
        return {"success": False, "error": f"Schema validation failed: {str(e)}"}


def compare_distributions(
    reference_path: str,
    current_path: str,
    columns: list[str] | None = None,
    significance_level: float = 0.05,
) -> dict[str, Any]:
    """
    Compare distributions between a reference dataset and current dataset for drift detection.

    Args:
        reference_path: Path to the reference dataset file (e.g., training data)
        current_path: Path to the current dataset file to compare against reference
        columns: List of column names to compare. If None, compares all common numeric columns.
        significance_level: Significance level for statistical tests (default 0.05)

    Returns:
        Dictionary with distribution comparison results for drift detection
    """
    try:
        import pandas as pd

        from data_quality import DataProfiler

        ref_path = Path(reference_path)
        cur_path = Path(current_path)

        if not ref_path.exists():
            return {"success": False, "error": f"Reference dataset {reference_path} does not exist"}
        if not cur_path.exists():
            return {"success": False, "error": f"Current dataset {current_path} does not exist"}

        # Load reference dataset
        ref_suffix = ref_path.suffix.lower()
        if ref_suffix == ".csv":
            df_reference = pd.read_csv(reference_path)
        elif ref_suffix == ".parquet":
            df_reference = pd.read_parquet(reference_path)
        elif ref_suffix == ".json":
            df_reference = pd.read_json(reference_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type for reference: {ref_suffix}. Supported: .csv, .parquet, .json",
            }

        # Load current dataset
        cur_suffix = cur_path.suffix.lower()
        if cur_suffix == ".csv":
            df_current = pd.read_csv(current_path)
        elif cur_suffix == ".parquet":
            df_current = pd.read_parquet(current_path)
        elif cur_suffix == ".json":
            df_current = pd.read_json(current_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type for current: {cur_suffix}. Supported: .csv, .parquet, .json",
            }

        # Compare distributions
        profiler = DataProfiler()
        anomalies = profiler.compare_distributions(
            df_reference,
            df_current,
            columns=columns,
        )

        # Filter anomalies by significance level
        significant_shifts = []
        for a in anomalies:
            if a.value and isinstance(a.value, dict):
                p_value = a.value.get("p_value", 1.0)
                if p_value < significance_level:
                    significant_shifts.append(
                        {
                            "column": a.column,
                            "description": a.description,
                            "severity": a.severity.value,
                            "ks_statistic": a.value.get("ks_statistic"),
                            "p_value": p_value,
                            "reference_mean": a.value.get("ref_mean"),
                            "current_mean": a.value.get("cur_mean"),
                            "drift_detected": True,
                        }
                    )

        # Determine columns compared
        import numpy as np

        if columns is None:
            ref_numeric = set(df_reference.select_dtypes(include=[np.number]).columns)
            cur_numeric = set(df_current.select_dtypes(include=[np.number]).columns)
            columns_compared = list(ref_numeric & cur_numeric)
        else:
            columns_compared = columns

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "reference_dataset": str(ref_path.absolute()),
            "current_dataset": str(cur_path.absolute()),
            "reference_rows": len(df_reference),
            "current_rows": len(df_current),
            "columns_compared": columns_compared,
            "significance_level": significance_level,
            "total_shifts_detected": len(significant_shifts),
            "shifts": significant_shifts,
            "drift_detected": len(significant_shifts) > 0,
        }

        # Add summary message
        if len(significant_shifts) == 0:
            result["message"] = "No significant distribution drift detected"
        else:
            drifted_cols = [s["column"] for s in significant_shifts]
            result["message"] = (
                f"Distribution drift detected in {len(significant_shifts)} column(s): {', '.join(drifted_cols)}"
            )

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas, numpy, and scipy.",
        }
    except Exception as e:
        return {"success": False, "error": f"Distribution comparison failed: {str(e)}"}


# --- Monitoring Tools ---


def detect_data_drift(
    reference_path: str,
    current_path: str,
    feature_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    drift_threshold: float = 0.1,
    dataset_name: str = "dataset",
) -> dict[str, Any]:
    """
    Detect data drift between reference and current datasets using Evidently AI.

    Uses the DriftDetector from the monitoring module which leverages Evidently AI
    for comprehensive drift detection with support for multiple statistical tests.

    Args:
        reference_path: Path to the reference dataset file (e.g., training data)
        current_path: Path to the current dataset file to compare against reference
        feature_columns: List of column names to check for drift. If None, checks all columns.
        categorical_columns: Explicitly specify which columns are categorical.
        numerical_columns: Explicitly specify which columns are numerical.
        drift_threshold: Threshold for drift detection (0-1, lower = stricter). Default 0.1.
        dataset_name: Name for the dataset in the report.

    Returns:
        Dictionary with drift detection results including:
        - overall_drift_detected: Whether overall drift was detected
        - drift_share: Proportion of features with drift
        - severity: Drift severity level (none, low, medium, high, critical)
        - feature_results: Per-feature drift analysis
        - recommendations: Suggested actions based on drift analysis
    """
    try:
        import pandas as pd

        from monitoring import DriftDetector

        ref_path = Path(reference_path)
        cur_path = Path(current_path)

        if not ref_path.exists():
            return {"success": False, "error": f"Reference dataset {reference_path} does not exist"}
        if not cur_path.exists():
            return {"success": False, "error": f"Current dataset {current_path} does not exist"}

        # Load reference dataset
        ref_suffix = ref_path.suffix.lower()
        if ref_suffix == ".csv":
            df_reference = pd.read_csv(reference_path)
        elif ref_suffix == ".parquet":
            df_reference = pd.read_parquet(reference_path)
        elif ref_suffix == ".json":
            df_reference = pd.read_json(reference_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type for reference: {ref_suffix}. Supported: .csv, .parquet, .json",
            }

        # Load current dataset
        cur_suffix = cur_path.suffix.lower()
        if cur_suffix == ".csv":
            df_current = pd.read_csv(current_path)
        elif cur_suffix == ".parquet":
            df_current = pd.read_parquet(current_path)
        elif cur_suffix == ".json":
            df_current = pd.read_json(current_path)
        else:
            return {
                "success": False,
                "error": f"Unsupported file type for current: {cur_suffix}. Supported: .csv, .parquet, .json",
            }

        # Initialize drift detector
        detector = DriftDetector(drift_threshold=drift_threshold)

        # Detect drift
        report = detector.detect_drift(
            reference_data=df_reference,
            current_data=df_current,
            feature_columns=feature_columns,
            dataset_name=dataset_name,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
        )

        # Build feature results for output
        feature_results_output = []
        for fr in report.feature_results:
            feature_results_output.append(
                {
                    "feature_name": fr.feature_name,
                    "drift_detected": fr.drift_detected,
                    "drift_score": fr.drift_score,
                    "stattest_name": fr.stattest_name,
                    "stattest_threshold": fr.stattest_threshold,
                    "p_value": fr.p_value,
                    "reference_distribution": fr.reference_distribution,
                    "current_distribution": fr.current_distribution,
                }
            )

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "report_id": report.report_id,
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp.isoformat(),
            "drift_type": report.drift_type.value,
            "overall_drift_detected": report.overall_drift_detected,
            "drift_share": report.drift_share,
            "severity": report.severity.value,
            "reference_rows": report.reference_rows,
            "current_rows": report.current_rows,
            "feature_results": feature_results_output,
            "recommendations": report.recommendations,
            "evidently_available": detector.evidently_available,
        }

        # Add drifted features summary
        drifted_features = [fr.feature_name for fr in report.feature_results if fr.drift_detected]
        result["drifted_features"] = drifted_features
        result["total_features_checked"] = len(report.feature_results)
        result["drifted_features_count"] = len(drifted_features)

        # Add summary message
        if not report.overall_drift_detected:
            result["message"] = f"No significant data drift detected in {dataset_name}"
        else:
            result["message"] = (
                f"Data drift detected in {dataset_name}: "
                f"{len(drifted_features)}/{len(report.feature_results)} features drifted "
                f"(severity: {report.severity.value})"
            )

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install pandas and scipy.",
        }
    except Exception as e:
        return {"success": False, "error": f"Drift detection failed: {str(e)}"}


def monitor_model_performance(
    model_name: str,
    y_true: list[float | int],
    y_pred: list[float | int],
    y_prob: list[list[float]] | None = None,
    task_type: str = "classification",
    model_version: str | None = None,
    degradation_threshold: float = 0.05,
    baseline_metrics: dict[str, float] | None = None,
    metrics_to_check: list[str] | None = None,
    record_snapshot: bool = True,
    storage_path: str | None = None,
) -> dict[str, Any]:
    """
    Monitor model performance metrics and detect degradation.

    Uses the ModelMonitor from the monitoring module to calculate metrics,
    track performance over time, detect degradation, and provide health status.

    Args:
        model_name: Name of the model to monitor
        y_true: Ground truth labels/values
        y_pred: Predicted labels/values
        y_prob: Prediction probabilities (for classification)
        task_type: "classification" or "regression"
        model_version: Version of the model
        degradation_threshold: Threshold for degradation detection (0-1)
        baseline_metrics: Baseline metrics to compare against
        metrics_to_check: List of metrics to evaluate for health status
        record_snapshot: Whether to record this evaluation as a performance snapshot
        storage_path: Path to save/load performance history (JSON file)

    Returns:
        Dictionary with performance monitoring results including:
        - metrics: Calculated performance metrics
        - health_status: Overall health status (healthy, warning, critical)
        - baseline_comparison: Comparison to baseline if provided
        - recommendations: Suggested actions based on performance
    """
    try:
        import numpy as np

        from monitoring import HealthStatus, ModelMonitor
        from monitoring.models import ModelMetrics

        # Convert inputs to numpy arrays
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        y_prob_arr = np.array(y_prob) if y_prob is not None else None

        # Validate inputs
        if len(y_true_arr) != len(y_pred_arr):
            return {
                "success": False,
                "error": f"Length mismatch: y_true has {len(y_true_arr)} samples, "
                f"y_pred has {len(y_pred_arr)} samples",
            }

        if len(y_true_arr) == 0:
            return {"success": False, "error": "Empty input arrays provided"}

        # Create baseline metrics object if provided
        baseline = None
        if baseline_metrics:
            baseline = ModelMetrics(**baseline_metrics)

        # Initialize monitor
        monitor = ModelMonitor(
            model_name=model_name,
            model_version=model_version,
            degradation_threshold=degradation_threshold,
            baseline_metrics=baseline,
        )

        # Load existing snapshots if storage path provided
        snapshots_loaded = 0
        if storage_path:
            snapshots_loaded = monitor.load_snapshots(storage_path)

        # Calculate metrics
        metrics = monitor.calculate_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_prob=y_prob_arr,
            task_type=task_type,
        )

        # Record snapshot if requested
        snapshot_id = None
        if record_snapshot:
            snapshot = monitor.record_snapshot(
                metrics=metrics,
                sample_size=len(y_true_arr),
            )
            snapshot_id = snapshot.snapshot_id

            # Save snapshots if storage path provided
            if storage_path:
                monitor.save_snapshots(storage_path)

        # Get health status
        health = monitor.get_health_status(
            metrics_to_check=metrics_to_check,
            days=7,
        )

        # Compare to baseline if provided
        baseline_comparison = None
        if baseline_metrics:
            baseline_comparison = monitor.compare_to_baseline(metrics)

        # Build metrics output
        metrics_output: dict[str, Any] = {}
        if task_type == "classification":
            if metrics.accuracy is not None:
                metrics_output["accuracy"] = metrics.accuracy
            if metrics.precision is not None:
                metrics_output["precision"] = metrics.precision
            if metrics.recall is not None:
                metrics_output["recall"] = metrics.recall
            if metrics.f1_score is not None:
                metrics_output["f1_score"] = metrics.f1_score
            if metrics.auc_roc is not None:
                metrics_output["auc_roc"] = metrics.auc_roc
            if metrics.log_loss is not None:
                metrics_output["log_loss"] = metrics.log_loss
        else:  # regression
            if metrics.mse is not None:
                metrics_output["mse"] = metrics.mse
            if metrics.rmse is not None:
                metrics_output["rmse"] = metrics.rmse
            if metrics.mae is not None:
                metrics_output["mae"] = metrics.mae
            if metrics.r2_score is not None:
                metrics_output["r2_score"] = metrics.r2_score

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "model_name": model_name,
            "model_version": model_version,
            "task_type": task_type,
            "sample_size": len(y_true_arr),
            "metrics": metrics_output,
            "health_status": (
                health["status"].value
                if isinstance(health["status"], HealthStatus)
                else health["status"]
            ),
            "health_message": health["message"],
            "health_details": {
                "metrics": health["metrics"],
                "issues": health.get("issues", []),
                "warnings": health.get("warnings", []),
            },
            "recommendations": health.get("recommendations", []),
            "snapshot_count": health.get("snapshot_count", 0),
        }

        # Add snapshot info if recorded
        if snapshot_id:
            result["snapshot_id"] = snapshot_id
            result["snapshot_recorded"] = True

        # Add baseline comparison if available
        if baseline_comparison:
            result["baseline_comparison"] = baseline_comparison

        # Add storage info
        if storage_path:
            result["storage_path"] = storage_path
            result["snapshots_loaded"] = snapshots_loaded

        # Add summary message
        if health["status"] == HealthStatus.HEALTHY:
            result["message"] = f"Model '{model_name}' is performing well"
        elif health["status"] == HealthStatus.WARNING:
            result["message"] = f"Model '{model_name}' shows warning signs: {health['message']}"
        elif health["status"] == HealthStatus.CRITICAL:
            result["message"] = f"Model '{model_name}' has critical issues: {health['message']}"
        else:
            result["message"] = f"Model '{model_name}' status unknown: {health['message']}"

        return result

    except ImportError as e:
        return {
            "success": False,
            "error": f"Required module not available: {str(e)}. Install numpy and scikit-learn.",
        }
    except Exception as e:
        return {"success": False, "error": f"Model performance monitoring failed: {str(e)}"}


def setup_alerting(
    project_path: str,
    alert_name: str,
    alert_type: str = "threshold",
    metrics: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    notification_channels: list[str] | None = None,
    notification_config: dict[str, Any] | None = None,
    evaluation_window: str = "5m",
    cooldown_period: str = "15m",
    severity: str = "warning",
    enabled: bool = True,
) -> dict[str, Any]:
    """
    Setup alerting configuration for model monitoring.

    Creates alerting rules and notification configurations for monitoring
    model performance metrics. Supports threshold-based, anomaly detection,
    drift detection, and composite alerts.

    Args:
        project_path: Path to the project
        alert_name: Name for the alerting configuration
        alert_type: Type of alert (threshold, anomaly, drift, composite)
        metrics: List of metrics to monitor
        thresholds: Threshold values for each metric
        notification_channels: Notification channels to use
        notification_config: Configuration for each notification channel
        evaluation_window: Time window for metric evaluation
        cooldown_period: Minimum time between repeated alerts
        severity: Alert severity level
        enabled: Whether the alert is enabled

    Returns:
        Dictionary with alerting configuration details including:
        - config_path: Path to the created configuration file
        - alert_rules: Generated alert rules
        - notification_setup: Notification channel configurations
    """
    try:
        path = Path(project_path)
        if not path.exists():
            return {"success": False, "error": f"Project path {project_path} does not exist"}

        # Set defaults
        if metrics is None:
            metrics = ["accuracy", "latency"]
        if notification_channels is None:
            notification_channels = ["email"]
        if thresholds is None:
            thresholds = {}

        # Validate alert_type
        valid_alert_types = ["threshold", "anomaly", "drift", "composite"]
        if alert_type not in valid_alert_types:
            return {
                "success": False,
                "error": f"Invalid alert_type '{alert_type}'. Must be one of: {valid_alert_types}",
            }

        # Validate severity
        valid_severities = ["info", "warning", "critical"]
        if severity not in valid_severities:
            return {
                "success": False,
                "error": f"Invalid severity '{severity}'. Must be one of: {valid_severities}",
            }

        # Validate notification channels
        valid_channels = ["email", "slack", "pagerduty", "webhook"]
        for channel in notification_channels:
            if channel not in valid_channels:
                return {
                    "success": False,
                    "error": f"Invalid notification channel '{channel}'. Must be one of: {valid_channels}",
                }

        # Create alerting directory structure
        alerting_dir = ensure_directory(path / "monitoring" / "alerting")
        configs_dir = ensure_directory(alerting_dir / "configs")
        rules_dir = ensure_directory(alerting_dir / "rules")

        # Generate alert rules based on alert_type
        alert_rules = []
        for metric in metrics:
            rule = {
                "name": f"{alert_name}_{metric}",
                "metric": metric,
                "alert_type": alert_type,
                "evaluation_window": evaluation_window,
                "cooldown_period": cooldown_period,
                "severity": severity,
                "enabled": enabled,
            }

            if alert_type == "threshold":
                if metric in thresholds:
                    rule["threshold"] = thresholds[metric]
                    rule["condition"] = f"{metric} < {thresholds[metric]}"
                else:
                    # Default thresholds for common metrics
                    default_thresholds = {
                        "accuracy": 0.9,
                        "precision": 0.85,
                        "recall": 0.85,
                        "f1_score": 0.85,
                        "latency": 100,  # ms
                        "error_rate": 0.05,
                        "throughput": 100,  # requests/sec
                    }
                    if metric in default_thresholds:
                        rule["threshold"] = default_thresholds[metric]
                        if metric in ["latency", "error_rate"]:
                            rule["condition"] = f"{metric} > {default_thresholds[metric]}"
                        else:
                            rule["condition"] = f"{metric} < {default_thresholds[metric]}"
            elif alert_type == "anomaly":
                rule["detection_method"] = "zscore"
                rule["sensitivity"] = 2.5  # z-score threshold
                rule["condition"] = f"anomaly detected in {metric}"
            elif alert_type == "drift":
                rule["drift_threshold"] = thresholds.get(metric, 0.1)
                rule["statistical_test"] = "ks_test"
                rule["condition"] = f"drift detected in {metric}"
            elif alert_type == "composite":
                rule["sub_rules"] = [
                    {
                        "metric": metric,
                        "condition": "threshold",
                        "threshold": thresholds.get(metric),
                    }
                ]
                rule["operator"] = "AND"

            alert_rules.append(rule)

        # Generate notification configurations
        notification_setup = {}
        for channel in notification_channels:
            channel_config = {"enabled": True, "severity_filter": severity}

            if notification_config and channel in notification_config:
                channel_config.update(notification_config[channel])
            else:
                # Default configurations
                if channel == "email":
                    channel_config["recipients"] = ["mlops-alerts@example.com"]
                    channel_config["subject_template"] = "[{severity}] {alert_name}: {metric} alert"
                elif channel == "slack":
                    channel_config["webhook_url"] = "${SLACK_WEBHOOK_URL}"
                    channel_config["channel"] = "#ml-alerts"
                    channel_config["mention_on_critical"] = True
                elif channel == "pagerduty":
                    channel_config["routing_key"] = "${PAGERDUTY_ROUTING_KEY}"
                    channel_config["severity_mapping"] = {
                        "info": "info",
                        "warning": "warning",
                        "critical": "critical",
                    }
                elif channel == "webhook":
                    channel_config["url"] = "${ALERT_WEBHOOK_URL}"
                    channel_config["method"] = "POST"
                    channel_config["headers"] = {"Content-Type": "application/json"}

            notification_setup[channel] = channel_config

        # Create the alerting configuration
        alert_config = {
            "name": alert_name,
            "version": "1.0",
            "description": f"Alerting configuration for {alert_name}",
            "enabled": enabled,
            "alert_type": alert_type,
            "metrics": metrics,
            "evaluation_window": evaluation_window,
            "cooldown_period": cooldown_period,
            "severity": severity,
            "rules": alert_rules,
            "notifications": notification_setup,
            "metadata": {
                "created_by": "mlops-agent",
                "project_path": str(path),
            },
        }

        # Write configuration file
        config_filename = f"{alert_name.lower().replace(' ', '_')}_alerting.yaml"
        config_path = configs_dir / config_filename
        with open(config_path, "w") as f:
            yaml.dump(alert_config, f, default_flow_style=False, sort_keys=False)

        # Write individual rule files for each metric
        rules_written = []
        for rule in alert_rules:
            rule_filename = f"{rule['name'].lower().replace(' ', '_')}.yaml"
            rule_path = rules_dir / rule_filename
            with open(rule_path, "w") as f:
                yaml.dump(rule, f, default_flow_style=False, sort_keys=False)
            rules_written.append(str(rule_path))

        # Create a sample alerting runner script
        runner_script = f'''#!/usr/bin/env python3
"""
Alerting Runner for {alert_name}

Auto-generated by MLOps Agent
Load this configuration and run alerting checks.
"""
import yaml
from pathlib import Path


def load_alerting_config(config_path: str) -> dict:
    """Load alerting configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_alert_condition(rule: dict, current_value: float) -> bool:
    """Check if alert condition is met."""
    alert_type = rule.get("alert_type", "threshold")

    if alert_type == "threshold":
        threshold = rule.get("threshold")
        if threshold is None:
            return False

        # Check condition based on metric type
        metric = rule.get("metric", "")
        if metric in ["latency", "error_rate"]:
            return current_value > threshold
        else:
            return current_value < threshold

    return False


def send_notification(channel: str, config: dict, alert_data: dict) -> bool:
    """Send notification through specified channel."""
    print(f"[{{channel.upper()}}] Alert: {{alert_data}}")
    # Implement actual notification logic here
    return True


def run_alerting_check(config_path: str, metrics_values: dict):
    """Run alerting checks against current metric values."""
    config = load_alerting_config(config_path)

    if not config.get("enabled", True):
        print("Alerting is disabled")
        return

    alerts_triggered = []
    for rule in config.get("rules", []):
        if not rule.get("enabled", True):
            continue

        metric = rule.get("metric")
        if metric not in metrics_values:
            continue

        current_value = metrics_values[metric]
        if check_alert_condition(rule, current_value):
            alert_data = {{
                "name": rule.get("name"),
                "metric": metric,
                "current_value": current_value,
                "threshold": rule.get("threshold"),
                "severity": rule.get("severity"),
            }}
            alerts_triggered.append(alert_data)

            # Send notifications
            for channel, channel_config in config.get("notifications", {{}}).items():
                if channel_config.get("enabled", True):
                    send_notification(channel, channel_config, alert_data)

    return alerts_triggered


if __name__ == "__main__":
    # Example usage
    config_path = "{config_path}"
    sample_metrics = {{
        "accuracy": 0.85,  # Below threshold
        "latency": 150,    # Above threshold
    }}
    alerts = run_alerting_check(config_path, sample_metrics)
    print(f"Triggered alerts: {{len(alerts)}}")
'''

        runner_path = alerting_dir / f"run_{alert_name.lower().replace(' ', '_')}_alerting.py"
        with open(runner_path, "w") as f:
            f.write(runner_script)

        # Build result
        result: dict[str, Any] = {
            "success": True,
            "alert_name": alert_name,
            "alert_type": alert_type,
            "config_path": str(config_path),
            "rules_dir": str(rules_dir),
            "runner_script": str(runner_path),
            "rules_count": len(alert_rules),
            "rules_written": rules_written,
            "metrics_monitored": metrics,
            "notification_channels": notification_channels,
            "alert_rules": alert_rules,
            "notification_setup": notification_setup,
            "evaluation_window": evaluation_window,
            "cooldown_period": cooldown_period,
            "severity": severity,
            "enabled": enabled,
            "message": f"Alerting configuration '{alert_name}' created successfully with {len(alert_rules)} rules",
            "next_steps": [
                f"Review configuration at {config_path}",
                "Set environment variables for notification channels (SLACK_WEBHOOK_URL, etc.)",
                f"Run alerting checks with: python {runner_path}",
                "Integrate with your monitoring pipeline",
            ],
        }

        return result

    except Exception as e:
        return {"success": False, "error": f"Alerting setup failed: {str(e)}"}


# --- Deployment Tools (Phase 4) ---


def load_template(template_path: str) -> str:
    """Load a template file."""
    path = Path(__file__).parent / template_path
    if path.exists():
        return path.read_text()
    return ""


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Simple template rendering with ${var} syntax."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", str(value))
    return result


# LitServe Tools
def create_litserve_api(
    project_path: str,
    model_path: str,
    model_name: str,
    model_type: str = "image_classifier",
    class_labels: list[str] | None = None,
    source_step: str = "generate_litserve_api",
) -> dict[str, Any]:
    """Create LitServe API for model serving."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    # Create deployment directory
    deploy_dir = ensure_directory(path / "deployment" / "litserve")

    # Generate class name from model name
    class_name = "".join(word.capitalize() for word in model_name.replace("-", "_").split("_"))
    model_suffix = Path(model_path).suffix.lower()
    if model_type in {"tabular_regressor", "tabular_classifier", "sklearn"} or model_suffix in {
        ".pkl",
        ".joblib",
    }:
        scaler_path = str(Path(model_path).with_name("scaler" + model_suffix))
        server_code = f'''"""LitServe Server for {model_name}

Auto-generated by MLOps Agent
"""
import os
import pickle
from pathlib import Path
from typing import Any

import litserve as ls
import litserve.server as _litserve_server
import numpy as np

_litserve_server._MCP_AVAILABLE = False
DEFAULT_PORT = 8000


def load_artifact(path: Path) -> Any:
    if path.suffix == ".joblib":
        import joblib

        return joblib.load(path)
    with open(path, "rb") as artifact_file:
        return pickle.load(artifact_file)


class {class_name}API(ls.LitAPI):
    """LitServe API for {model_name}"""

    def setup(self, device: str) -> None:
        """Initialize sklearn model and optional scaler."""
        self.device = device
        project_root = Path(__file__).resolve().parents[2]
        self.model = load_artifact(project_root / "{model_path}")
        scaler_file = project_root / "{scaler_path}"
        self.scaler = load_artifact(scaler_file) if scaler_file.exists() else None

    def decode_request(self, request: dict) -> np.ndarray:
        """Convert HTTP request to tabular model input."""
        values = request.get("instances", request.get("features", request.get("input", request)))
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim > 1:
            if array.shape[0] != 1:
                raise ValueError("Send one tabular row per /predict request.")
            array = array.reshape(-1)
        if self.scaler is not None:
            array = self.scaler.transform(array.reshape(1, -1))[0]
        return array

    def predict(self, x: np.ndarray) -> Any:
        """Run sklearn prediction."""
        return self.model.predict(x)

    def encode_response(self, output: Any) -> dict:
        """Convert model output to HTTP response."""
        return {{"predictions": np.asarray(output).tolist()}}


if __name__ == "__main__":
    api = {class_name}API(max_batch_size=64, batch_timeout=0.05)
    server = ls.LitServer(
        api,
        accelerator="auto",
        workers_per_device=4
    )
    server.run(port=int(os.environ.get("LITSERVE_PORT", DEFAULT_PORT)))
'''

        server_path = deploy_dir / "server.py"
        with open(server_path, "w") as f:
            f.write(server_code)

        requirements = """litserve>=0.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
"""
        req_path = deploy_dir / "requirements.txt"
        with open(req_path, "w") as f:
            f.write(requirements)

        return {
            "success": True,
            "server_path": str(server_path),
            "requirements_path": str(req_path),
            "class_name": f"{class_name}API",
            "verification_results": [
                {
                    "check_name": "litserve_app_artifact_ready",
                    "evidence_type": "declared",
                    "source_step": source_step,
                    "passed": True,
                    "evidence": "LitServe tabular sklearn server generated.",
                }
            ],
            "artifact_manifest": {
                "entries": [
                    {
                        "artifact_type": "serving_application",
                        "producing_step": source_step,
                        "state": "generated",
                        "path": relative_to_project(project_path, server_path),
                    }
                ]
            },
            "message": f"LitServe API created at {deploy_dir}",
        }

    # Prepare template variables based on model type
    if model_type == "image_classifier":
        transforms_code = """from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])"""
        decode_code = """image_data = request.get("image")
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)"""
        encode_code = """probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        label = self.labels[predicted.item()] if self.labels else str(predicted.item())
        return {"class": label, "confidence": confidence.item()}"""
        labels_code = f"self.labels = {class_labels or []}"
    else:
        transforms_code = "# Add custom transforms here"
        decode_code = "return request"
        encode_code = "return {'output': output.tolist()}"
        labels_code = "self.labels = None"

    # Generate server code
    server_code = f'''"""LitServe Server for {model_name}

Auto-generated by MLOps Agent
"""
import base64
import io
import os
from typing import Any

import litserve as ls
import litserve.server as _litserve_server
import torch
from PIL import Image

_litserve_server._MCP_AVAILABLE = False
DEFAULT_PORT = 8000


class {class_name}API(ls.LitAPI):
    """LitServe API for {model_name}"""

    def setup(self, device: str) -> None:
        """Initialize model and components."""
        self.device = device
        self.model = torch.jit.load("{model_path}")
        self.model = self.model.to(device)
        self.model.eval()

        {transforms_code}
        {labels_code}

    def decode_request(self, request: dict) -> Any:
        """Convert HTTP request to model input."""
        {decode_code}

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run model inference."""
        with torch.no_grad():
            return self.model(x)

    def encode_response(self, output: torch.Tensor) -> dict:
        """Convert model output to HTTP response."""
        {encode_code}


if __name__ == "__main__":
    api = {class_name}API(max_batch_size=64, batch_timeout=0.05)
    server = ls.LitServer(
        api,
        accelerator="auto",
        workers_per_device=4
    )
    server.run(port=int(os.environ.get("LITSERVE_PORT", DEFAULT_PORT)))
'''

    # Write server file
    server_path = deploy_dir / "server.py"
    with open(server_path, "w") as f:
        f.write(server_code)

    # Write requirements
    requirements = """litserve>=0.2.0
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
"""
    req_path = deploy_dir / "requirements.txt"
    with open(req_path, "w") as f:
        f.write(requirements)

    return {
        "success": True,
        "server_path": str(server_path),
        "requirements_path": str(req_path),
        "class_name": f"{class_name}API",
        "verification_results": [
            {
                "check_name": "litserve_app_artifact_ready",
                "evidence_type": "declared",
                "source_step": source_step,
                "passed": True,
                "evidence": (
                    "LitServe server and requirements files were generated for local preflight."
                ),
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "serving_application",
                    "producing_step": source_step,
                    "state": "generated",
                    "path": relative_to_project(project_path, server_path),
                }
            ]
        },
        "message": f"LitServe API created at {deploy_dir}",
    }


def select_or_create_model_artifact(
    project_path: str,
    model_path: str | None = None,
    model_name: str = "model",
    create_placeholder: bool = True,
) -> dict[str, Any]:
    """Select or create a local model artifact for LitServe preflight."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    candidates = []
    if model_path:
        candidates.append(path / model_path)
    candidates.extend(
        candidate
        for pattern in ("models/*.pt", "model/*.pt", "*.pt", "models/*.pth", "model/*.pth")
        for candidate in path.glob(pattern)
    )

    selected = next((candidate for candidate in candidates if candidate.exists()), None)
    state = "selected"
    if selected is None:
        if not create_placeholder:
            return {"success": False, "error": "No local model artifact found"}
        selected = path / "models" / f"{model_name.replace('-', '_')}_preflight.pt"
        ensure_directory(selected.parent)
        selected.write_text("Local LitServe preflight placeholder; replace with a real model.\n")

    relative_path = relative_to_project(project_path, selected)
    return {
        "success": True,
        "model_path": relative_path,
        "verification_results": [
            {
                "check_name": "model_artifact_selected",
                "evidence_type": "declared",
                "source_step": "select_or_create_model_artifact",
                "passed": True,
                "evidence": f"Model artifact selected for local preflight: {relative_path}.",
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "model_artifact",
                    "producing_step": "select_or_create_model_artifact",
                    "state": state,
                    "path": relative_path,
                }
            ]
        },
        "message": f"Model artifact selected for preflight: {relative_path}",
    }


def generate_litserve_dockerfile(
    project_path: str,
    server_path: str = "deployment/litserve/server.py",
    requirements_file: str = "deployment/litserve/requirements.txt",
    port: int = 8000,
) -> dict[str, Any]:
    """Generate or validate a Dockerfile for local LitServe preflight."""
    result = create_ml_dockerfile(
        project_path=project_path,
        base_image="python:3.11-slim",
        cuda_version=None,
        entry_point=server_path,
        requirements_file=requirements_file,
        expose_port=port,
    )
    if not result.get("success"):
        return result

    dockerfile_path = result["dockerfile_path"]
    return {
        **result,
        "verification_results": [
            {
                "check_name": "dockerfile_artifact_ready",
                "evidence_type": "declared",
                "source_step": "generate_or_validate_dockerfile",
                "passed": True,
                "evidence": "Dockerfile generated for local LitServe preflight; image not built.",
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "container_definition",
                    "producing_step": "generate_or_validate_dockerfile",
                    "state": "generated",
                    "path": relative_to_project(project_path, dockerfile_path),
                }
            ]
        },
    }


def record_litserve_launch_command(
    project_path: str,
    server_path: str = "deployment/litserve/server.py",
    port: int = 8000,
) -> dict[str, Any]:
    """Record the LitServe launch command without starting a server."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    command = f"python {server_path}"
    return {
        "success": True,
        "launch_command": command,
        "port": port,
        "verification_results": [
            {
                "check_name": "launch_command_recorded",
                "evidence_type": "declared",
                "source_step": "record_launch_command",
                "passed": True,
                "evidence": f"Launch command recorded but not executed: {command}.",
            }
        ],
        "message": "LitServe launch command recorded without starting a server.",
    }


def record_litserve_missing_live_evidence(project_path: str) -> dict[str, Any]:
    """Record live evidence intentionally absent from a local LitServe preflight."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    missing_live_evidence = {
        "gpu_detection": "not_checked_in_local_preflight",
        "server_start": "not_started_in_local_preflight",
        "/health": "not_called_in_local_preflight",
        "/predict": "not_called_in_local_preflight",
        "endpoint_url": "not_available_without_server_start",
    }
    return {
        "success": True,
        "missing_live_evidence": missing_live_evidence,
        "verification_results": [
            {
                "check_name": "missing_live_evidence_recorded",
                "evidence_type": "declared",
                "source_step": "record_missing_live_evidence",
                "passed": True,
                "evidence": ", ".join(missing_live_evidence),
            }
        ],
        "message": "Missing live deployment evidence recorded for preflight-only success.",
    }


def detect_runtime_environment(project_path: str) -> dict[str, Any]:
    """Record local runtime context without provisioning cloud resources."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    return {
        "success": True,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "project_path": str(path),
        "message": "Runtime environment recorded on the current machine.",
    }


def detect_gpu_cuda(project_path: str) -> dict[str, Any]:
    """Detect GPU availability from observed nvidia-smi and PyTorch CUDA checks."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    nvidia_smi = {
        "available": False,
        "success": False,
        "stdout": "",
        "stderr": "nvidia-smi not found",
    }
    if check_tool_installed("nvidia-smi"):
        result = run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            cwd=str(path),
            timeout=10,
        )
        nvidia_smi = {
            "available": result.get("success", False) and bool(result.get("stdout")),
            "success": result.get("success", False),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr") or result.get("error", ""),
        }

    torch_cuda: dict[str, Any] = {"available": False, "error": None}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_cuda = {
            "available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception as exc:
        torch_cuda = {"available": False, "error": str(exc)}

    gpu_available = bool(nvidia_smi["available"] or torch_cuda["available"])
    evidence = {"nvidia_smi": nvidia_smi, "torch_cuda": torch_cuda}
    return {
        "success": True,
        "gpu_available": gpu_available,
        "gpu_evidence": evidence,
        "verification_results": [
            {
                "check_name": "gpu_detection_recorded",
                "evidence_type": "observed",
                "source_step": "detect_gpu_cuda",
                "passed": gpu_available,
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        ],
        "message": (
            "GPU detected from observed runtime evidence."
            if gpu_available
            else "No GPU detected from nvidia-smi or PyTorch CUDA."
        ),
    }


def _load_pickle_or_joblib_artifact(path: Path) -> Any:
    if path.suffix.lower() == ".joblib":
        import joblib

        return joblib.load(path)
    with open(path, "rb") as artifact_file:
        return pickle.load(artifact_file)


def _infer_tabular_feature_count(project_path: str) -> int | None:
    path = Path(project_path)
    for relative_path in (
        "outputs/scaler.pkl",
        "outputs/scaler.joblib",
        "outputs/model.pkl",
        "outputs/model.joblib",
    ):
        artifact_path = path / relative_path
        if not artifact_path.exists():
            continue
        try:
            artifact = _load_pickle_or_joblib_artifact(artifact_path)
        except Exception:
            continue
        feature_count = getattr(artifact, "n_features_in_", None)
        if isinstance(feature_count, int) and feature_count > 0:
            return feature_count
    return None


def _default_litserve_prediction_payload(project_path: str) -> dict[str, Any]:
    feature_count = _infer_tabular_feature_count(project_path) or 1
    return {"input": [0.0] * feature_count}


def select_best_model_artifact(
    project_path: str,
    model_path: str | None = None,
    latest_run: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
    metric_name: str | None = None,
    metric_direction: str | None = None,
    threshold: float | None = None,
    tie_policy: str | None = None,
) -> dict[str, Any]:
    """Select a model artifact for deterministic training comparison or LitServe deployment."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deterministic_inputs = (
        latest_run,
        baseline,
        metric_name,
        metric_direction,
        threshold,
        tie_policy,
    )
    if any(value is not None for value in deterministic_inputs):
        return _select_model_artifact_from_metric_comparison(
            project_path=path,
            latest_run=latest_run,
            baseline=baseline,
            metric_name=metric_name,
            metric_direction=metric_direction,
            threshold=threshold,
            tie_policy=tie_policy,
        )

    candidates = []
    if model_path:
        candidates.append(path / model_path)
    candidates.extend(
        candidate
        for pattern in (
            "outputs/model.pkl",
            "outputs/model.joblib",
            "outputs/*.pkl",
            "outputs/*.joblib",
            "models/model.pt",
            "models/model.pth",
            "models/model.pkl",
            "models/model.joblib",
            "models/*_preflight.pt",
            "models/*.pt",
            "models/*.pth",
            "models/*.pkl",
            "models/*.joblib",
            "model/*.pt",
            "model/*.pth",
            "model/*.pkl",
            "model/*.joblib",
            "*.pt",
            "*.pth",
            "*.pkl",
            "*.joblib",
        )
        for candidate in path.glob(pattern)
    )
    selected = next((candidate for candidate in candidates if candidate.exists()), None)
    if selected is None:
        return {
            "success": False,
            "error": "No model artifact or LitServe preflight artifact found.",
        }

    relative_path = relative_to_project(project_path, selected)
    model_type = (
        "tabular_regressor"
        if selected.suffix.lower() in {".pkl", ".joblib"}
        else "torch"
    )
    return {
        "success": True,
        "model_path": relative_path,
        "model_type": model_type,
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "model_artifact",
                    "producing_step": "select_best_model_artifact",
                    "state": "selected",
                    "path": relative_path,
                }
            ]
        },
        "message": f"Model artifact selected for LitServe GPU deployment: {relative_path}",
    }


def _select_model_artifact_from_metric_comparison(
    *,
    project_path: Path,
    latest_run: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    metric_name: str | None,
    metric_direction: str | None,
    threshold: float | None,
    tie_policy: str | None,
) -> dict[str, Any]:
    missing_inputs = [
        name
        for name, value in (
            ("latest_run", latest_run),
            ("baseline", baseline),
            ("metric_name", metric_name),
            ("metric_direction", metric_direction),
            ("threshold", threshold),
            ("tie_policy", tie_policy),
        )
        if value in (None, "")
    ]
    if missing_inputs:
        return _model_selection_blocked_result(
            missing_inputs=missing_inputs,
            reason=f"Missing model selection inputs: {', '.join(missing_inputs)}",
        )
    if metric_direction not in {"maximize", "minimize"}:
        return _model_selection_blocked_result(
            missing_inputs=["metric_direction"],
            reason="metric_direction must be 'maximize' or 'minimize'.",
        )
    if tie_policy not in {"keep_baseline", "select_latest"}:
        return _model_selection_blocked_result(
            missing_inputs=["tie_policy"],
            reason="tie_policy must be 'keep_baseline' or 'select_latest'.",
        )

    assert latest_run is not None
    assert baseline is not None
    assert metric_name is not None
    assert metric_direction is not None
    assert threshold is not None
    assert tie_policy is not None

    latest_value = _metric_value_from_run(latest_run, metric_name)
    baseline_value = _baseline_metric_value(baseline, metric_name)
    if latest_value is None:
        return _model_selection_blocked_result(
            missing_inputs=["latest_metric"],
            reason=f"Latest run is missing metric '{metric_name}'.",
        )
    if baseline_value is None:
        return _model_selection_blocked_result(
            missing_inputs=["baseline_metric"],
            reason=f"Baseline is missing metric '{metric_name}'.",
        )

    latest_artifact = _artifact_reference_from_run(latest_run)
    baseline_artifact = _artifact_reference_from_run(baseline)
    if latest_artifact is None:
        return _model_selection_blocked_result(
            missing_inputs=["candidate_artifact"],
            reason="Latest run is missing checkpoint/model artifact evidence.",
        )
    if baseline_artifact is None:
        return _model_selection_blocked_result(
            missing_inputs=["baseline_artifact"],
            reason="Baseline is missing selected checkpoint/model artifact evidence.",
        )

    latest_run_status = latest_run.get("run_status")
    latest_run_complete = latest_run_status in (None, "", "FINISHED")
    improvement = (
        latest_value - baseline_value
        if metric_direction == "maximize"
        else baseline_value - latest_value
    )
    comparison_result = {
        "metric_name": metric_name,
        "metric_direction": metric_direction,
        "baseline_value": baseline_value,
        "latest_value": latest_value,
        "threshold": threshold,
        "improvement": improvement,
        "tie_policy": tie_policy,
    }
    beats_baseline = latest_run_complete and (
        improvement > threshold or (improvement == threshold and tie_policy == "select_latest")
    )
    selected_source = "latest" if beats_baseline else "baseline"
    selected_run = latest_run if beats_baseline else baseline
    selected_artifact = latest_artifact if beats_baseline else baseline_artifact
    selected_value = latest_value if beats_baseline else baseline_value
    selected_path = selected_artifact.get("path")
    selected_uri = selected_artifact.get("uri")
    checksum = _artifact_checksum(project_path, selected_path)
    decision = "select_latest" if beats_baseline else "keep_baseline"

    evidence_summary = (
        f"decision={decision}; metric_name={metric_name}; direction={metric_direction}; "
        f"baseline_value={baseline_value}; latest_value={latest_value}; "
        f"threshold={threshold}; improvement={improvement}; "
        f"selected_artifact={selected_path or selected_uri}; "
        f"source_run_id={selected_run.get('run_id')}"
    )
    artifact_entry = {
        "artifact_type": "model_artifact",
        "producing_step": "select_best_model_artifact",
        "state": "selected",
        "path": selected_path,
        "uri": selected_uri,
        "checksum": checksum,
        "metadata": {
            "source_run_id": selected_run.get("run_id"),
            "metric_name": metric_name,
            "metric_value": selected_value,
            "comparison_result": comparison_result,
            "decision": decision,
        },
    }

    return {
        "success": True,
        "status": "selected_latest" if beats_baseline else "kept_baseline",
        "decision": decision,
        "selected_source": selected_source,
        "model_path": selected_path,
        "model_uri": selected_uri,
        "model_type": _model_type_from_artifact(selected_path or selected_uri),
        "source_run_id": selected_run.get("run_id"),
        "metric": {
            "name": metric_name,
            "direction": metric_direction,
            "value": selected_value,
        },
        "comparison_result": comparison_result,
        "discard_reason": None if beats_baseline else _model_selection_discard_reason(
            latest_run_complete=latest_run_complete,
            latest_run_status=latest_run_status,
        ),
        "keep_baseline_reason": None
        if beats_baseline
        else "Baseline artifact remains selected.",
        "verification_results": _model_selection_verification_results(
            inputs_present=True,
            baseline_recorded=True,
            metric_compared=True,
            candidate_artifact_verified=True,
            artifact_selected=True,
            evidence_summary=evidence_summary,
        ),
        "artifact_manifest": {"entries": [artifact_entry]},
    }


def _model_selection_blocked_result(
    missing_inputs: list[str],
    reason: str,
) -> dict[str, Any]:
    return {
        "success": True,
        "status": "blocked",
        "decision": "blocked",
        "missing_required_pieces": missing_inputs,
        "failure_reason": reason,
        "next_actions": [
            "Provide metric_name, metric_direction, threshold, tie_policy, baseline, and candidate artifact evidence."
        ],
        "verification_results": _model_selection_verification_results(
            inputs_present=False,
            baseline_recorded=False,
            metric_compared=False,
            candidate_artifact_verified=False,
            artifact_selected=False,
            evidence_summary=reason,
        ),
        "artifact_manifest": {"entries": []},
    }


def _model_selection_discard_reason(
    *,
    latest_run_complete: bool,
    latest_run_status: Any,
) -> str:
    if not latest_run_complete:
        return f"Latest run status is {latest_run_status}; keeping baseline artifact."
    return "Latest run did not beat baseline under the configured threshold and tie policy."


def _metric_value_from_run(run: dict[str, Any], metric_name: str) -> float | None:
    metrics = run.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get(metric_name), int | float):
        return float(metrics[metric_name])
    if isinstance(run.get("metric_value"), int | float):
        return float(run["metric_value"])
    return None


def _baseline_metric_value(baseline: dict[str, Any], metric_name: str) -> float | None:
    if isinstance(baseline.get("metric_value"), int | float):
        return float(baseline["metric_value"])
    return _metric_value_from_run(baseline, metric_name)


def _artifact_reference_from_run(run: dict[str, Any]) -> dict[str, str] | None:
    for path_key in ("artifact_path", "model_path", "checkpoint_artifact_path"):
        if run.get(path_key):
            return {"path": str(run[path_key])}
    for uri_key in ("checkpoint_artifact_uri", "model_uri", "artifact_uri"):
        if run.get(uri_key):
            return {"uri": str(run[uri_key])}
    raw_manifest = run.get("artifact_manifest")
    entries = raw_manifest.get("entries", ()) if isinstance(raw_manifest, dict) else ()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("artifact_type") in {
            "model_artifact",
            "checkpoint_or_model_artifact",
            "mlflow_checkpoint_or_model_artifact",
        }:
            reference: dict[str, str] = {}
            if entry.get("path"):
                reference["path"] = str(entry["path"])
            if entry.get("uri"):
                reference["uri"] = str(entry["uri"])
            if reference:
                return reference
    return None


def _artifact_checksum(project_path: Path, artifact_path: str | None) -> str | None:
    if not artifact_path:
        return None
    path = Path(artifact_path)
    if not path.is_absolute():
        path = project_path / path
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _model_type_from_artifact(artifact: str | None) -> str:
    if artifact is None:
        return "unknown"
    suffix = Path(artifact).suffix.lower()
    if suffix in {".pkl", ".joblib"}:
        return "tabular_regressor"
    return "torch"


def _model_selection_verification_results(
    *,
    inputs_present: bool,
    baseline_recorded: bool,
    metric_compared: bool,
    candidate_artifact_verified: bool,
    artifact_selected: bool,
    evidence_summary: str,
) -> list[dict[str, Any]]:
    checks = (
        ("model_selection_inputs_present", inputs_present),
        ("model_selection_baseline_recorded", baseline_recorded),
        ("model_selection_metric_compared", metric_compared),
        ("model_selection_candidate_artifact_verified", candidate_artifact_verified),
        ("model_artifact_selected", artifact_selected),
    )
    return [
        {
            "check_name": check_name,
            "evidence_type": "observed",
            "source_step": "select_best_model_artifact",
            "passed": passed,
            "evidence": evidence_summary,
        }
        for check_name, passed in checks
    ]


DEFAULT_CAPSTONE_STAGES = (
    "setup",
    "data",
    "train",
    "container_ci",
    "deploy",
    "monitor",
    "report",
)
DEFAULT_CAPSTONE_IMPLEMENTED_SUBWORKFLOWS = (
    "setup_pipeline",
    "prepare_capstone_data",
    "detect_training_project",
    "train_and_track",
    "prepare_capstone_container_ci",
    "deploy_litserve_preflight",
    "deploy_litserve_gpu",
)
DEFAULT_CAPSTONE_BLOCKED_SUBWORKFLOWS = ("train_until_better",)
DEFAULT_CAPSTONE_DEFERRED_CAPABILITIES = (
    {
        "stage": "data",
        "capability": "S3 DVC remote automation",
        "reason": "Remote DVC/S3 credential setup and automation are later-phase work.",
        "later_phase_pointer": "Future data/versioning issue",
    },
    {
        "stage": "deploy",
        "capability": "KServe/Helm/ArgoCD",
        "reason": "Production Kubernetes serving, charting, and GitOps rollout are not implemented.",
        "later_phase_pointer": "Future production deployment issue",
    },
    {
        "stage": "deploy",
        "capability": "HuggingFace Spaces",
        "reason": "Spaces publishing remains outside the current verified workflow set.",
        "later_phase_pointer": "Future demo deployment issue",
    },
    {
        "stage": "deploy",
        "capability": "AWS Lambda serverless",
        "reason": "Serverless packaging and deployment are not implemented by this skeleton.",
        "later_phase_pointer": "Future serverless deployment issue",
    },
    {
        "stage": "monitor",
        "capability": "stress tests",
        "reason": "Load and stress testing workflows are not part of the current registry contract.",
        "later_phase_pointer": "Future monitoring and validation issue",
    },
    {
        "stage": "monitor",
        "capability": "frontend",
        "reason": "Workflow timeline and endpoint cards are deferred UI work.",
        "later_phase_pointer": "Future frontend issue",
    },
    {
        "stage": "report",
        "capability": "final report",
        "reason": "Final report generation is not implemented in this phase.",
        "later_phase_pointer": "Future reporting issue",
    },
    {
        "stage": "report",
        "capability": "video",
        "reason": "Video generation and publication are not implemented in this phase.",
        "later_phase_pointer": "Future reporting issue",
    },
)


DATA_STAGE_EVIDENCE_PATH = ".auto_mlops/capstone/data_stage_evidence.json"


def record_capstone_data_stage_evidence(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    capstone_data_detection: dict[str, Any] | None = None,
    capstone_split_manifest_result: dict[str, Any] | None = None,
    capstone_data_package_result: dict[str, Any] | None = None,
    capstone_data_remote_result: dict[str, Any] | None = None,
    capstone_data_push_result: dict[str, Any] | None = None,
    capstone_data_pull_result: dict[str, Any] | None = None,
    verification_results: list[dict[str, Any]] | None = None,
    artifact_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write durable Phase 4 data-stage evidence for capstone handoff."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not path.is_dir():
        return {"success": False, "error": f"Project path {project_path} is not a directory"}

    workflow_inputs = workflow_inputs or {}
    capstone_data_detection = capstone_data_detection or {}
    capstone_split_manifest_result = capstone_split_manifest_result or {}
    capstone_data_package_result = capstone_data_package_result or {}
    capstone_data_remote_result = capstone_data_remote_result or {}
    capstone_data_push_result = capstone_data_push_result or {}
    capstone_data_pull_result = capstone_data_pull_result or {}
    verification_results = verification_results or []
    artifact_entries = _artifact_entries_from_manifest_payload(artifact_manifest)

    completion_mode = str(
        workflow_inputs.get(
            "completion_mode",
            capstone_data_detection.get("completion_mode", "local_ready"),
        )
    )
    transfer_result = (
        capstone_data_pull_result
        if capstone_data_pull_result.get("status") == "succeeded"
        else capstone_data_push_result
    )
    datasets = _data_stage_dataset_records(
        capstone_data_detection,
        capstone_split_manifest_result,
    )
    dvc_state = _data_stage_dvc_state(
        completion_mode,
        capstone_data_package_result,
        capstone_data_remote_result,
        transfer_result,
    )
    blocked_capabilities = _data_stage_blocked_capabilities(
        completion_mode,
        datasets,
        capstone_split_manifest_result,
        capstone_data_package_result,
        capstone_data_remote_result,
        transfer_result,
    )
    status = _data_stage_status(
        completion_mode,
        datasets,
        capstone_split_manifest_result,
        capstone_data_package_result,
        capstone_data_remote_result,
        transfer_result,
    )

    evidence_path = path / DATA_STAGE_EVIDENCE_PATH
    evidence_entry = {
        "artifact_type": "data_stage_evidence",
        "producing_step": "record_data_stage_evidence",
        "state": "generated",
        "path": DATA_STAGE_EVIDENCE_PATH,
        "metadata": {
            "schema_version": "phase4.data_stage_evidence.v1",
            "completion_mode": completion_mode,
            "status": status,
        },
    }
    merged_entries = _dedupe_artifact_entries([*artifact_entries, evidence_entry])
    evidence = {
        "schema_version": "phase4.data_stage_evidence.v1",
        "created_at": _utc_now_iso(),
        "workflow_id": "prepare_capstone_data",
        "status": status,
        "completion_mode": completion_mode,
        "datasets": datasets,
        "dvc": dvc_state,
        "blocked_capabilities": blocked_capabilities,
        "verification_results": verification_results,
        "artifact_manifest": {"entries": merged_entries},
    }
    redacted_evidence = _redacted_evidence(evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(json.dumps(redacted_evidence, indent=2, sort_keys=True) + "\n")

    lineage_artifacts_reported = any(
        entry.get("artifact_type")
        in {
            "capstone_source_dataset",
            "split_manifest",
            "capstone_data_package",
            "dvc_tracking_file",
            "dvc_repo_metadata",
            "capstone_data_remote",
            "capstone_data_transfer",
        }
        for entry in merged_entries
    )
    verification_evidence = {
        "path": DATA_STAGE_EVIDENCE_PATH,
        "status": status,
        "completion_mode": completion_mode,
        "dataset_count": len(datasets),
        "blocked_capabilities": blocked_capabilities,
    }
    return {
        "success": True,
        "status": status,
        "evidence_path": DATA_STAGE_EVIDENCE_PATH,
        "data_stage_evidence": redacted_evidence,
        "missing_inputs": [] if status == "succeeded" else ["complete_data_stage_evidence"],
        "next_actions": [
            capability["reason"] for capability in blocked_capabilities if capability.get("reason")
        ],
        "verification_results": [
            {
                "check_name": "data_stage_evidence_artifact_reported",
                "evidence_type": "observed",
                "source_step": "record_data_stage_evidence",
                "passed": True,
                "evidence": json.dumps(_redacted_evidence(verification_evidence), sort_keys=True),
            },
            {
                "check_name": "dataset_lineage_artifacts_reported",
                "evidence_type": "observed",
                "source_step": "record_data_stage_evidence",
                "passed": lineage_artifacts_reported,
                "evidence": json.dumps(
                    {
                        "artifact_count": len(merged_entries),
                        "lineage_artifacts_reported": lineage_artifacts_reported,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "artifact_manifest": {"entries": [evidence_entry]},
    }


def _data_stage_dataset_records(
    detection: dict[str, Any],
    split_result: dict[str, Any],
) -> list[dict[str, Any]]:
    split_by_dataset = {
        split.get("dataset_id"): split
        for split in split_result.get("split_manifests", [])
        if isinstance(split, dict) and split.get("dataset_id")
    }
    datasets: list[dict[str, Any]] = []
    for dataset in detection.get("datasets", []):
        if not isinstance(dataset, dict):
            continue
        split = split_by_dataset.get(dataset.get("dataset_id"), {})
        datasets.append(
            {
                "dataset_id": dataset.get("dataset_id"),
                "status": dataset.get("status", "blocked"),
                "source_path": dataset.get("source_path"),
                "layout": dataset.get("layout"),
                "missing_inputs": dataset.get("missing_inputs", []),
                "next_actions": dataset.get("next_actions", []),
                "class_count": dataset.get("class_count", 0),
                "total_image_count": dataset.get("total_image_count", 0),
                "split": {
                    "strategy": split.get("split_strategy"),
                    "seed": split.get("seed"),
                    "test_size": split.get("test_size"),
                    "train_count": split.get("train_count"),
                    "test_count": split.get("test_count"),
                    "per_class_counts": split.get("per_class_counts", {}),
                    "split_manifest_path": split.get("split_manifest_path"),
                    "materialized_train_path": split.get("materialized_train_path"),
                    "materialized_test_path": split.get("materialized_test_path"),
                },
                "artifacts": [
                    value
                    for value in (
                        split.get("split_manifest_path"),
                        split.get("materialized_train_path"),
                        split.get("materialized_test_path"),
                    )
                    if value
                ],
            }
        )
    return datasets


def _data_stage_dvc_state(
    completion_mode: str,
    package_result: dict[str, Any],
    remote_result: dict[str, Any],
    transfer_result: dict[str, Any],
) -> dict[str, Any]:
    transfer = transfer_result.get("transfer") if isinstance(transfer_result, dict) else None
    if not isinstance(transfer, dict):
        transfer = {}
    transfer_status = transfer_result.get("status", "missing")
    if completion_mode == "local_ready" and transfer_status == "missing":
        transfer_status = "deferred"
    return {
        "tracked_paths": package_result.get("tracked_package_paths", []),
        "dvc_tracking_files": package_result.get("dvc_tracking_files", []),
        "repo": package_result.get("dvc_repo", {}),
        "remote": remote_result.get(
            "remote",
            {
                "remote_name": "capstone",
                "remote_type": "missing",
                "redacted_remote_url": None,
            },
        ),
        "remote_validation_status": remote_result.get("status", "missing"),
        "transfer": {
            "status": transfer_status,
            "direction": transfer.get("direction"),
            "remote": transfer.get("remote"),
            "paths": transfer.get("paths", []),
            "pushed": transfer_result.get("status") == "succeeded"
            and transfer.get("direction") == "push",
            "pulled": transfer_result.get("status") == "succeeded"
            and transfer.get("direction") == "pull",
        },
    }


def _data_stage_status(
    completion_mode: str,
    datasets: list[dict[str, Any]],
    split_result: dict[str, Any],
    package_result: dict[str, Any],
    remote_result: dict[str, Any],
    transfer_result: dict[str, Any],
) -> str:
    local_ready = (
        len(datasets) == 2
        and all(dataset.get("status") == "succeeded" for dataset in datasets)
        and split_result.get("status") == "succeeded"
        and package_result.get("status") == "succeeded"
    )
    if not local_ready:
        return "blocked"
    if completion_mode != "capstone_complete":
        return "succeeded"
    remote = remote_result.get("remote", {})
    capstone_complete = (
        remote_result.get("status") == "succeeded"
        and remote.get("remote_type") == "s3"
        and transfer_result.get("status") == "succeeded"
    )
    return "succeeded" if capstone_complete else "blocked"


def _data_stage_blocked_capabilities(
    completion_mode: str,
    datasets: list[dict[str, Any]],
    split_result: dict[str, Any],
    package_result: dict[str, Any],
    remote_result: dict[str, Any],
    transfer_result: dict[str, Any],
) -> list[dict[str, Any]]:
    blocked: list[dict[str, Any]] = []
    if len(datasets) != 2 or any(dataset.get("status") != "succeeded" for dataset in datasets):
        blocked.append(
            {
                "stage": "data",
                "capability": "two_supported_capstone_datasets",
                "reason": "Exactly two supported dataset entries are required.",
                "later_phase_pointer": "Phase 4 dataset detection rerun",
            }
        )
    if split_result.get("status") != "succeeded":
        blocked.append(
            {
                "stage": "data",
                "capability": "split_evidence_recorded",
                "reason": "Split manifest or existing split evidence is incomplete.",
                "later_phase_pointer": "Phase 4 split manifest rerun",
            }
        )
    if package_result.get("status") != "succeeded":
        blocked.append(
            {
                "stage": "data",
                "capability": "capstone_data_package_tracked",
                "reason": "DVC tracking evidence for the capstone data package is incomplete.",
                "later_phase_pointer": "Phase 4 DVC tracking rerun",
            }
        )
    remote = remote_result.get("remote", {})
    if completion_mode == "local_ready":
        blocked.append(
            {
                "stage": "data",
                "capability": "s3_transfer_completed",
                "reason": "S3 transfer is deferred for local_ready and still required for capstone completeness.",
                "later_phase_pointer": "Phase 4 capstone_complete rerun",
            }
        )
        return blocked
    if remote_result.get("status") != "succeeded" or remote.get("remote_type") != "s3":
        blocked.append(
            {
                "stage": "data",
                "capability": "s3_remote_validated",
                "reason": "Capstone-complete data stage requires validated S3 DVC remote evidence.",
                "later_phase_pointer": "Phase 4 S3 remote validation rerun",
            }
        )
    if transfer_result.get("status") != "succeeded":
        blocked.append(
            {
                "stage": "data",
                "capability": "s3_transfer_completed",
                "reason": "Capstone-complete data stage requires approved successful S3 transfer evidence.",
                "later_phase_pointer": "Phase 4 DVC push/pull rerun",
            }
        )
    return blocked


def _artifact_entries_from_manifest_payload(artifact_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(artifact_manifest, dict):
        return []
    raw_entries = artifact_manifest.get("entries", [])
    if isinstance(raw_entries, dict):
        raw_entries = [raw_entries]
    return [entry for entry in raw_entries if isinstance(entry, dict)]


def _dedupe_artifact_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any, Any]] = set()
    for entry in entries:
        key = (
            entry.get("artifact_type"),
            entry.get("producing_step"),
            entry.get("path"),
            entry.get("uri"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_capstone_data_stage_evidence(project_path: Path) -> dict[str, Any]:
    evidence_path = project_path / DATA_STAGE_EVIDENCE_PATH
    if not evidence_path.exists():
        return {
            "status": "missing",
            "stage_status": "blocked",
            "completed": False,
            "evidence_artifact": DATA_STAGE_EVIDENCE_PATH,
            "blocked_capabilities": [
                {
                    "stage": "data",
                    "capability": "data_stage_evidence_artifact_reported",
                    "reason": (
                        "Run prepare_capstone_data before claiming capstone data-stage "
                        "completion."
                    ),
                    "later_phase_pointer": "Phase 4 data-stage evidence issue",
                }
            ],
            "artifact_entry": None,
        }
    try:
        evidence = json.loads(evidence_path.read_text())
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "stage_status": "blocked",
            "completed": False,
            "evidence_artifact": DATA_STAGE_EVIDENCE_PATH,
            "blocked_capabilities": [
                {
                    "stage": "data",
                    "capability": "data_stage_evidence_parseable",
                    "reason": "Data-stage evidence artifact exists but is not valid JSON.",
                    "later_phase_pointer": "Phase 4 data-stage evidence rerun",
                }
            ],
            "artifact_entry": None,
        }
    blocked_capabilities = evidence.get("blocked_capabilities", [])
    if not isinstance(blocked_capabilities, list):
        blocked_capabilities = []
    status = evidence.get("status", "blocked")
    completion_mode = evidence.get("completion_mode")
    completed = status == "succeeded" and completion_mode in {
        "local_ready",
        "capstone_complete",
    }
    return {
        "status": "completed" if completed else "blocked",
        "raw_status": status,
        "stage_status": "completed" if completed else "blocked",
        "completed": completed,
        "completion_mode": completion_mode,
        "evidence_artifact": DATA_STAGE_EVIDENCE_PATH,
        "dataset_count": len(evidence.get("datasets", []))
        if isinstance(evidence.get("datasets"), list)
        else 0,
        "blocked_capabilities": blocked_capabilities,
        "artifact_entry": {
            "artifact_type": "data_stage_evidence",
            "producing_step": "record_capstone_orchestrator_skeleton",
            "state": "validated",
            "path": DATA_STAGE_EVIDENCE_PATH,
            "checksum": _sha256_file(evidence_path),
            "metadata": {
                "source_workflow_id": "prepare_capstone_data",
                "data_stage_status": status,
                "completion_mode": completion_mode,
            },
        },
    }


def _load_capstone_container_ci_evidence(
    project_path: Path,
    *,
    upstream_ready: bool,
) -> dict[str, Any]:
    evidence_path = project_path / CONTAINER_CI_EVIDENCE_PATH
    missing_capability = {
        "stage": "container_ci",
        "capability": "container_ci_evidence_artifact_reported",
        "reason": "Run prepare_capstone_container_ci to produce durable container/CI evidence.",
        "later_phase_pointer": "Phase 5 prepare_capstone_container_ci Issue 7",
    }
    if not evidence_path.exists():
        if upstream_ready:
            return {
                "status": "missing",
                "stage_status": "blocked",
                "completed": False,
                "evidence_artifact": CONTAINER_CI_EVIDENCE_PATH,
                "blocked_capabilities": [missing_capability],
                "deferred_capabilities": [],
                "artifact_entry": None,
            }
        return {
            "status": "missing",
            "stage_status": "deferred",
            "completed": False,
            "evidence_artifact": CONTAINER_CI_EVIDENCE_PATH,
            "blocked_capabilities": [],
            "deferred_capabilities": [
                {
                    **missing_capability,
                    "reason": (
                        "Container/CI evidence is deferred until data and selected model "
                        "artifact evidence are ready."
                    ),
                }
            ],
            "artifact_entry": None,
        }
    try:
        evidence = json.loads(evidence_path.read_text())
    except json.JSONDecodeError:
        return {
            "status": "blocked",
            "stage_status": "blocked",
            "completed": False,
            "evidence_artifact": CONTAINER_CI_EVIDENCE_PATH,
            "blocked_capabilities": [
                {
                    "stage": "container_ci",
                    "capability": "container_ci_evidence_parseable",
                    "reason": "Container/CI evidence artifact exists but is not valid JSON.",
                    "later_phase_pointer": "Phase 5 container/CI evidence rerun",
                }
            ],
            "deferred_capabilities": [],
            "artifact_entry": None,
        }

    schema_ok = (
        evidence.get("schema_version") == "phase5.container_ci_evidence.v1"
        and evidence.get("workflow_id") == "prepare_capstone_container_ci"
        and isinstance(evidence.get("artifact_manifest"), dict)
        and isinstance(evidence.get("verification_results"), list)
    )
    checks_ok = _container_ci_handoff_checks_passed(evidence)
    status = evidence.get("status", "blocked")
    completion_mode = evidence.get("completion_mode")
    completed = schema_ok and status == "succeeded" and checks_ok
    blocked_capabilities = evidence.get("blocked_capabilities", [])
    deferred_capabilities = evidence.get("deferred_capabilities", [])
    if not isinstance(blocked_capabilities, list):
        blocked_capabilities = []
    if not isinstance(deferred_capabilities, list):
        deferred_capabilities = []
    if not completed:
        blocked_capabilities = [
            *blocked_capabilities,
            {
                "stage": "container_ci",
                "capability": "container_ci_success_contract_satisfied",
                "reason": (
                    "Container/CI evidence exists but required structured verification "
                    "or artifact manifest evidence is incomplete."
                ),
                "later_phase_pointer": "Phase 5 prepare_capstone_container_ci rerun",
            },
        ]
    return {
        "status": "completed" if completed else "blocked",
        "raw_status": status,
        "stage_status": "completed" if completed else "blocked",
        "completed": completed,
        "completion_mode": completion_mode,
        "evidence_artifact": CONTAINER_CI_EVIDENCE_PATH,
        "next_phase_readiness": evidence.get("next_phase_readiness", {}),
        "blocked_capabilities": blocked_capabilities,
        "deferred_capabilities": deferred_capabilities,
        "artifact_entry": {
            "artifact_type": "container_ci_evidence",
            "producing_step": "record_capstone_orchestrator_skeleton",
            "state": "validated" if completed else "blocked",
            "path": CONTAINER_CI_EVIDENCE_PATH,
            "checksum": _sha256_file(evidence_path),
            "metadata": {
                "source_workflow_id": "prepare_capstone_container_ci",
                "container_ci_status": status,
                "completion_mode": completion_mode,
            },
        },
    }


def _container_ci_handoff_checks_passed(evidence: dict[str, Any]) -> bool:
    results = evidence.get("verification_results", [])
    if not isinstance(results, list):
        return False
    checks = {
        result.get("check_name"): result.get("passed")
        for result in results
        if isinstance(result, dict)
    }
    entries = evidence.get("artifact_manifest", {}).get("entries", [])
    if not isinstance(entries, list):
        return False
    artifact_types = {entry.get("artifact_type") for entry in entries if isinstance(entry, dict)}
    common = (
        checks.get("container_ci_evidence_artifact_reported") is True
        and checks.get("capstone_ci_workflow_validated") is True
        and {"container_ci_evidence", "github_actions_workflow", "container_build_spec"}.issubset(
            artifact_types
        )
    )
    if evidence.get("completion_mode") != "container_capstone_complete":
        return common
    capstone_checks = (
        "registry_push_succeeded",
        "pushed_image_reference_reported",
        "capstone_ci_registry_usage_validated",
    )
    return common and all(checks.get(check) is True for check in capstone_checks)


def prepare_capstone_container_ci_contract(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record Phase 5 Issue 1 blocked/deferred evidence without running container tooling."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    optional_inputs = (
        "data_stage_evidence_path",
        "local_model_artifact_path",
        "mlflow_run_id",
        "mlflow_best_artifact_path",
        "registry_target",
        "image_name",
        "image_tag",
        "ci_workflow_path",
    )
    missing_inputs = [
        input_name for input_name in optional_inputs if not inputs.get(input_name)
    ]
    deferred_capabilities = [
        {
            "capability": "resolve_upstream_container_evidence",
            "issue": "Phase 5 Issue 2",
            "reason": "Upstream data, training, MLflow, and model artifact evidence resolution is deferred.",
        },
        {
            "capability": "build_smoke_check_container_image",
            "issue": "Phase 5 Issue 4",
            "reason": "Docker image build and smoke checks are deferred.",
        },
        {
            "capability": "configure_validate_registry_target",
            "issue": "Phase 5 Issue 5",
            "reason": "Registry target configuration and validation are deferred.",
        },
        {
            "capability": "approval_gated_registry_login_push",
            "issue": "Phase 5 Issue 6",
            "reason": "Registry login and push are deferred until approval-gated implementation.",
        },
        {
            "capability": "record_container_ci_evidence_handoff",
            "issue": "Phase 5 Issue 7",
            "reason": "container_ci_evidence.json writing is deferred to Phase 5 Issue 7.",
        },
    ]
    next_actions = [
        "Provide upstream data-stage, training, MLflow, or local model artifact evidence before Issue 2.",
        "Implement Phase 5 Issue 2 before resolving upstream evidence.",
        "Do not generate Dockerfiles, run Docker, configure registries, mutate secrets, or write CI workflows in Issue 1.",
    ]
    evidence_payload = {
        "workflow_id": "prepare_capstone_container_ci",
        "status": "blocked",
        "completion_mode": completion_mode,
        "missing_inputs": missing_inputs,
        "deferred_capabilities": deferred_capabilities,
        "next_actions": next_actions,
    }
    return {
        "success": True,
        "status": "blocked",
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "missing_inputs": missing_inputs,
        "deferred_capabilities": deferred_capabilities,
        "next_actions": next_actions,
        "verification_results": [
            {
                "check_name": "upstream_evidence_resolved",
                "evidence_type": "observed",
                "source_step": "prepare_capstone_container_ci_contract",
                "passed": False,
                "evidence": json.dumps(evidence_payload, sort_keys=True),
            }
        ],
    }


CONTAINER_UPSTREAM_SOURCE_STEP = "resolve_upstream_container_evidence"
TRAINING_EVIDENCE_CANDIDATE_PATHS = (
    ".auto_mlops/capstone/training_evidence.json",
    ".auto_mlops/capstone/training_mlflow_evidence.json",
    ".auto_mlops/training_evidence.json",
    "outputs/training_evidence.json",
    "outputs/mlflow_tracking_result.json",
    "outputs/model_selection_result.json",
)


def resolve_capstone_container_upstream_evidence(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve Phase 5 upstream evidence without running training, Docker, CI, or registry work."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    if completion_mode not in {"container_local_ready", "container_capstone_complete"}:
        return {
            "success": True,
            "status": "blocked",
            "completion_mode": completion_mode,
            "blocked_capabilities": [
                {
                    "capability": "completion_mode",
                    "reason": "completion_mode must be container_local_ready or container_capstone_complete.",
                    "next_action": "Provide a valid Container CI Completion Mode.",
                }
            ],
            "verification_results": [
                _container_upstream_verification_result(
                    "upstream_evidence_resolved",
                    False,
                    {"completion_mode": completion_mode, "reason": "invalid_completion_mode"},
                )
            ],
            "artifact_manifest": {"entries": []},
        }

    data_stage = _resolve_container_data_stage_evidence(path, inputs)
    training_evidence = _load_container_training_evidence(path)
    local_model_artifact = _resolve_local_model_artifact(path, inputs)
    mlflow_best_artifact = _resolve_mlflow_best_artifact(path, inputs, training_evidence)
    training_lineage = _resolve_training_lineage(data_stage, training_evidence)

    local_model_available = local_model_artifact["status"] == "resolved"
    mlflow_best_available = mlflow_best_artifact["status"] == "resolved"
    data_capstone_complete = data_stage["status"] == "resolved" and (
        data_stage.get("completion_mode") == "capstone_complete"
    )
    training_lineage_available = training_lineage["status"] == "resolved"

    upstream_evidence = {
        "data_stage": data_stage["evidence"],
        "training_lineage": training_lineage["evidence"],
        "mlflow_best_artifact": mlflow_best_artifact["evidence"],
        "local_model_artifact": local_model_artifact["evidence"],
    }
    artifact_entries = _container_upstream_artifact_entries(
        data_stage=data_stage,
        local_model_artifact=local_model_artifact,
        mlflow_best_artifact=mlflow_best_artifact,
    )

    blocked_capabilities: list[dict[str, Any]] = []
    deferred_capabilities: list[dict[str, Any]] = []

    if completion_mode == "container_local_ready":
        if data_stage["status"] != "resolved":
            deferred_capabilities.append(
                _container_upstream_capability(
                    "data_stage_evidence_artifact_reported",
                    "Run prepare_capstone_data to produce durable data-stage evidence.",
                    "Phase 4 prepare_capstone_data",
                )
            )
            upstream_evidence["data_stage"]["status"] = "deferred"
        if not mlflow_best_available and local_model_available:
            deferred_capabilities.append(
                _container_upstream_capability(
                    "mlflow_best_artifact_verified",
                    "Run train_and_track and select a MLflow-linked best artifact.",
                    "Phase 3 train_and_track",
                )
            )
            upstream_evidence["mlflow_best_artifact"]["status"] = "deferred"
        if not (local_model_available or mlflow_best_available):
            blocked_capabilities.append(
                _container_upstream_capability(
                    "local_or_mlflow_model_artifact_resolved",
                    "Provide local_model_artifact_path or structured MLflow best artifact evidence.",
                    "Phase 5 Issue 2 upstream evidence",
                )
            )
    else:
        if not data_capstone_complete:
            upstream_evidence["data_stage"]["status"] = "blocked"
            blocked_capabilities.append(
                _container_upstream_capability(
                    "data_stage_evidence_artifact_reported",
                    "Run prepare_capstone_data with completion_mode=capstone_complete.",
                    "Phase 4 prepare_capstone_data capstone_complete",
                )
            )
        if not mlflow_best_available:
            upstream_evidence["mlflow_best_artifact"]["status"] = "blocked"
            blocked_capabilities.append(
                _container_upstream_capability(
                    "mlflow_best_artifact_verified",
                    "Provide structured MLflow-linked best artifact evidence.",
                    "Phase 3 train_and_track/select_best_model_artifact",
                )
            )
        if not training_lineage_available:
            upstream_evidence["training_lineage"]["status"] = "blocked"
            blocked_capabilities.append(
                _container_upstream_capability(
                    "training_lineage_verified",
                    "Provide structured training evidence tied to data_stage_evidence.json.",
                    "Phase 3 train_and_track with data-stage lineage",
                )
            )

    resolved = not blocked_capabilities
    verification_results = _container_upstream_verification_results(
        completion_mode=completion_mode,
        upstream_evidence=upstream_evidence,
        local_model_available=local_model_available,
        mlflow_best_available=mlflow_best_available,
        data_capstone_complete=data_capstone_complete,
        training_lineage_available=training_lineage_available,
        resolved=resolved,
    )
    workflow_input_overrides = {
        "local_model_artifact_available": local_model_available,
        "mlflow_best_artifact_available": mlflow_best_available,
    }
    return {
        "success": True,
        "status": "resolved" if resolved else "blocked",
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "upstream_evidence": _redacted_evidence(upstream_evidence),
        "blocked_capabilities": _redacted_evidence(blocked_capabilities),
        "deferred_capabilities": _redacted_evidence(deferred_capabilities),
        "workflow_input_overrides": workflow_input_overrides,
        "verification_results": verification_results,
        "artifact_manifest": {"entries": _redacted_evidence(artifact_entries)},
        "next_actions": _container_upstream_next_actions(
            blocked_capabilities,
            deferred_capabilities,
        ),
    }


RUNTIME_IMAGE_SPEC_STEP = "generate_validate_runtime_image_spec"
CONTAINER_BUILD_STEP = "build_smoke_check_container_image"
RUNTIME_IMAGE_INTENDED_ROLES = ("ci", "training_validation", "inference_validation")
RUNTIME_IMAGE_BOUNDED_COMMANDS = (
    "python -m pytest tests -q",
    "python - <<'PY'\nfrom pathlib import Path\nassert Path('/app').exists()\nPY",
)
RUNTIME_IMAGE_SAFE_BASE = "python:3.11-slim"


def generate_validate_capstone_runtime_image_spec(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate or validate a conservative Capstone Runtime Image spec without Docker."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    dockerfile_path = path / "Dockerfile"
    dockerignore_path = path / ".dockerignore"
    dependency_context = _capstone_runtime_dependency_context(path)
    existing_dockerfile = dockerfile_path.exists()
    approved = _runtime_image_write_approved(approval_record)

    action = "validated_existing" if existing_dockerfile else "blocked_write_requires_approval"
    dockerfile_content = dockerfile_path.read_text() if existing_dockerfile else ""
    dockerignore_content = (
        dockerignore_path.read_text() if dockerignore_path.exists() else _default_dockerignore()
    )

    if not existing_dockerfile and approved:
        dockerfile_content = _generate_capstone_runtime_dockerfile(dependency_context)
        dockerfile_path.write_text(dockerfile_content)
        if not dockerignore_path.exists():
            dockerignore_path.write_text(dockerignore_content)
        action = "generated"

    base_image_decision = _runtime_image_base_image_decision(
        dockerfile_content=dockerfile_content,
        workflow_inputs=inputs,
    )
    build_spec_evidence = {
        "action": action,
        "build_spec_path": "Dockerfile",
        "base_image_decision": base_image_decision,
        "intended_roles": list(RUNTIME_IMAGE_INTENDED_ROLES),
        "bounded_commands": list(RUNTIME_IMAGE_BOUNDED_COMMANDS),
        "dependency_context": dependency_context,
        "docker_execution": {
            "build_attempted": False,
            "run_attempted": False,
            "reason": "Phase 5 Issue 3 records build-spec evidence only.",
        },
    }
    if action == "blocked_write_requires_approval":
        build_spec_evidence["approval_required"] = {
            "step_id": RUNTIME_IMAGE_SPEC_STEP,
            "risk_categories": ["writes_project_files"],
        }
    if approved and action == "generated":
        build_spec_evidence["approval_record"] = _runtime_image_approval_evidence(
            approval_record
        )

    secret_safety = _runtime_image_secret_safety(
        build_spec_path="Dockerfile",
        base_image=base_image_decision["selected_base_image"],
        dependency_files=dependency_context["dependency_files"],
        bounded_commands=RUNTIME_IMAGE_BOUNDED_COMMANDS,
        dockerfile_content=dockerfile_content,
    )
    blocked_capabilities: list[dict[str, Any]] = []
    if action == "blocked_write_requires_approval":
        blocked_capabilities.append(
            {
                "capability": "write_container_build_spec",
                "reason": "Writing Dockerfile requires an Approval Gate.",
                "required_risk_categories": ["writes_project_files"],
                "next_action": (
                    "Record approval for generate_validate_runtime_image_spec "
                    "before writing Dockerfile."
                ),
            }
        )
    if not secret_safety["passed"]:
        blocked_capabilities.append(
            {
                "capability": "secret_safe_container_build_spec",
                "reason": "Container build spec contains secret-like or local-only evidence.",
                "next_action": "Remove secret-like Dockerfile content before continuing.",
            }
        )

    artifact_entries = _runtime_image_artifact_entries(
        action=action,
        dependency_source=dependency_context["selected_dependency_source"],
        include_dockerignore=(action == "generated" and dockerignore_path.exists()),
    )
    verification_results = [
        _runtime_image_verification_result(
            "container_build_spec_reported",
            action != "blocked_write_requires_approval" and secret_safety["passed"],
            build_spec_evidence,
            evidence_type="observed" if action == "validated_existing" else "declared",
        ),
        _runtime_image_verification_result(
            "dependency_context_reported",
            True,
            dependency_context,
            evidence_type="observed",
        ),
        _runtime_image_verification_result(
            "secret_safety_validated",
            secret_safety["passed"],
            secret_safety,
            evidence_type="observed",
        ),
        _runtime_image_verification_result(
            "container_artifact_manifest_reported",
            bool(artifact_entries),
            {"entries": artifact_entries},
            evidence_type="observed" if action == "validated_existing" else "declared",
        ),
    ]
    status = "validated" if action == "validated_existing" else action
    if blocked_capabilities:
        status = "blocked"
    return {
        "success": True,
        "status": status,
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": inputs.get("completion_mode", "container_local_ready"),
        "container": {
            "runtime_image": _redacted_evidence(build_spec_evidence),
            "dependency_context": dependency_context,
            "secret_safety": secret_safety,
        },
        "blocked_capabilities": _redacted_evidence(blocked_capabilities),
        "deferred_capabilities": [
            {
                "capability": "build_smoke_check_container_image",
                "issue": "Phase 5 Issue 4",
                "reason": "Docker image build and smoke checks are deferred.",
            },
            {
                "capability": "configure_validate_registry_target",
                "issue": "Phase 5 Issue 5",
                "reason": "Registry target validation is deferred.",
            },
            {
                "capability": "approval_gated_registry_login_push",
                "issue": "Phase 5 Issue 6",
                "reason": "Registry login and push are deferred.",
            },
            {
                "capability": "record_container_ci_evidence_handoff",
                "issue": "Phase 5 Issue 7",
                "reason": "container_ci_evidence.json writing is deferred to Phase 5 Issue 7.",
            },
        ],
        "verification_results": verification_results,
        "artifact_manifest": {"entries": _redacted_evidence(artifact_entries)},
        "next_actions": _runtime_image_next_actions(blocked_capabilities),
    }


def _capstone_runtime_dependency_context(project_path: Path) -> dict[str, Any]:
    discovered = [
        name
        for name in ("uv.lock", "pyproject.toml", "requirements.txt", "setup.py")
        if (project_path / name).exists()
    ]
    if "uv.lock" in discovered or "pyproject.toml" in discovered:
        selected_source = "uv_or_pyproject"
        install_strategy = "uv_sync_frozen" if "uv.lock" in discovered else "pip_install_editable"
    elif "requirements.txt" in discovered:
        selected_source = "requirements"
        install_strategy = "pip_requirements"
    elif "setup.py" in discovered:
        selected_source = "setup_py"
        install_strategy = "pip_install_editable"
    else:
        selected_source = "none"
        install_strategy = "no_dependency_file_detected"
    return {
        "dependency_files": discovered,
        "selected_dependency_source": selected_source,
        "install_strategy": install_strategy,
        "priority_order": ["uv.lock/pyproject.toml", "requirements.txt", "setup.py"],
    }


def _runtime_image_write_approved(approval_record: dict[str, Any] | None) -> bool:
    if not isinstance(approval_record, dict):
        return False
    return (
        approval_record.get("step_id") == RUNTIME_IMAGE_SPEC_STEP
        and approval_record.get("status") == "approved"
        and approval_record.get("risk_categories") == ["writes_project_files"]
    )


def _runtime_image_approval_evidence(
    approval_record: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(approval_record, dict):
        return {}
    return {
        "step_id": approval_record.get("step_id"),
        "risk_categories": approval_record.get("risk_categories", []),
        "status": approval_record.get("status"),
        "approver": approval_record.get("approver"),
        "timestamp": approval_record.get("timestamp"),
    }


def _generate_capstone_runtime_dockerfile(dependency_context: dict[str, Any]) -> str:
    install_strategy = dependency_context["install_strategy"]
    lines = [
        f"FROM {RUNTIME_IMAGE_SAFE_BASE}",
        "ENV PYTHONDONTWRITEBYTECODE=1",
        "ENV PYTHONUNBUFFERED=1",
        "WORKDIR /app",
        "RUN python -m pip install --upgrade pip",
        "COPY . .",
    ]
    if install_strategy == "uv_sync_frozen":
        lines.append("RUN python -m pip install uv && uv sync --frozen --no-dev")
    elif install_strategy == "pip_requirements":
        lines.append("RUN python -m pip install -r requirements.txt")
    elif install_strategy == "pip_install_editable":
        lines.append("RUN python -m pip install -e .")
    else:
        lines.append(
            "RUN python - <<'PY'\n"
            "from pathlib import Path\n"
            "assert Path('/app').exists()\n"
            "PY"
        )
    lines.extend(
        [
            'LABEL org.opencontainers.image.title="Capstone Runtime Image"',
            'LABEL auto_mlops.phase="5-runtime-image-spec"',
            'CMD ["python", "-m", "pytest", "tests", "-q"]',
            "",
        ]
    )
    return "\n".join(lines)


def _default_dockerignore() -> str:
    return "\n".join(
        [
            ".env",
            ".env.*",
            ".venv/",
            "__pycache__/",
            ".pytest_cache/",
            ".mypy_cache/",
            ".ruff_cache/",
            ".git/",
            "data/",
            ".auto_mlops/",
            "",
        ]
    )


def _runtime_image_base_image_decision(
    dockerfile_content: str,
    workflow_inputs: dict[str, Any],
) -> dict[str, Any]:
    selected_base_image = RUNTIME_IMAGE_SAFE_BASE
    match = re.search(r"^\s*FROM\s+([^\s]+)", dockerfile_content, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        selected_base_image = match.group(1)
    cuda_requested = bool(workflow_inputs.get("cuda_required"))
    cuda_observed = bool(re.search(r"\bcuda\b|nvidia/cuda", dockerfile_content, flags=re.IGNORECASE))
    return {
        "selected_base_image": selected_base_image,
        "python_version_floor": "3.10",
        "cuda_base_allowed": cuda_requested or cuda_observed,
        "cuda_base_observed": cuda_observed,
        "decision_reason": (
            "Existing Dockerfile base image referenced."
            if dockerfile_content
            else "Default conservative Python 3.10+ base image selected."
        ),
    }


def _runtime_image_secret_safety(
    *,
    build_spec_path: str,
    base_image: str,
    dependency_files: list[str],
    bounded_commands: tuple[str, ...],
    dockerfile_content: str,
) -> dict[str, Any]:
    fields = {
        "build_spec_path": build_spec_path,
        "base_image": base_image,
        "dependency_files": " ".join(dependency_files),
        "bounded_commands": "\n".join(bounded_commands),
        "dockerfile_content": dockerfile_content,
    }
    violations: list[dict[str, str]] = []
    secret_key_pattern = re.compile(
        r"\b(?:AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|[A-Z0-9_]*(?:SECRET|TOKEN|ACCESS_KEY)[A-Z0-9_]*)\b"
    )
    absolute_source_dataset_pattern = re.compile(r"/(?:home|Users|mnt|media|data)/[^\s\"']+")
    for field, value in fields.items():
        secret_match = secret_key_pattern.search(value)
        if secret_match:
            violations.append(
                {
                    "field": field,
                    "rule": "secret_like_key",
                    "match": secret_match.group(0),
                }
            )
        if ".env" in value and field == "dockerfile_content":
            violations.append(
                {
                    "field": field,
                    "rule": "env_file_reference",
                    "match": ".env",
                }
            )
        path_match = absolute_source_dataset_pattern.search(value)
        if path_match:
            violations.append(
                {
                    "field": field,
                    "rule": "absolute_source_dataset_path",
                    "match": "<redacted-path>",
                }
            )
    return {
        "passed": not violations,
        "checked_fields": [
            "build_spec_path",
            "base_image",
            "dependency_files",
            "bounded_commands",
            "dockerfile_content",
        ],
        "violations": violations,
    }


def _runtime_image_artifact_entries(
    *,
    action: str,
    dependency_source: str,
    include_dockerignore: bool,
) -> list[dict[str, Any]]:
    if action == "blocked_write_requires_approval":
        return []
    state = "validated" if action == "validated_existing" else "generated"
    entries: list[dict[str, Any]] = [
        {
            "artifact_type": "container_build_spec",
            "producing_step": RUNTIME_IMAGE_SPEC_STEP,
            "state": state,
            "path": "Dockerfile",
            "metadata": {
                "action": action,
                "dependency_source": dependency_source,
                "intended_roles": list(RUNTIME_IMAGE_INTENDED_ROLES),
            },
        }
    ]
    if include_dockerignore:
        entries.append(
            {
                "artifact_type": "container_build_context_ignore",
                "producing_step": RUNTIME_IMAGE_SPEC_STEP,
                "state": "generated",
                "path": ".dockerignore",
                "metadata": {"action": "generated_secret_safety_ignore_rules"},
            }
        )
    return entries


def _runtime_image_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
    *,
    evidence_type: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": evidence_type,
        "source_step": RUNTIME_IMAGE_SPEC_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _runtime_image_next_actions(blocked_capabilities: list[dict[str, Any]]) -> list[str]:
    if blocked_capabilities:
        return [item["next_action"] for item in blocked_capabilities if item.get("next_action")]
    return [
        "Proceed to Phase 5 Issue 4 for Docker availability, image build, and smoke-check evidence.",
        "Do not run Docker, configure registries, write CI workflows, or write container_ci_evidence.json in Issue 3.",
    ]


def build_smoke_check_capstone_container_image(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    capstone_runtime_image_spec_result: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
    smoke_approval_record: dict[str, Any] | None = None,
    build_timeout_seconds: int = 300,
    smoke_timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Record Docker availability, approved image build, and bounded smoke evidence."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    build_spec_ready = _container_build_spec_ready(path, capstone_runtime_image_spec_result)
    docker_evidence = _detect_container_docker_availability()
    docker_available = bool(docker_evidence["available"])
    image_tag = _container_image_tag(inputs)
    container: dict[str, Any] = {
        "docker": docker_evidence,
        "build_spec": {
            "ready": build_spec_ready,
            "path": "Dockerfile",
            "source_status": (capstone_runtime_image_spec_result or {}).get("status"),
        },
        "image_build": {
            "attempted": False,
            "approved": False,
            "image_tag": image_tag,
            "command": None,
        },
        "smoke_check": {
            "attempted": False,
            "approved": False,
            "passed": False,
            "command": None,
        },
    }
    blocked_capabilities: list[dict[str, Any]] = []
    deferred_capabilities: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []

    verification_results = [
        _container_build_verification_result(
            "docker_availability_reported",
            True,
            docker_evidence,
        )
    ]
    if completion_mode == "container_capstone_complete":
        verification_results.append(
            _container_build_verification_result(
                "docker_available",
                docker_available,
                docker_evidence,
            )
        )

    if not build_spec_ready:
        blocked_capabilities.append(
            {
                "capability": "container_build_spec_reported",
                "reason": "Docker image build requires a validated or generated Dockerfile.",
                "next_action": "Complete generate_validate_runtime_image_spec before running docker build.",
            }
        )

    if not docker_available:
        if completion_mode == "container_local_ready" and build_spec_ready:
            deferred_capabilities.append(
                {
                    "capability": "image_build_deferred_reported",
                    "reason": "Docker is unavailable; local readiness may continue from validated build spec evidence.",
                    "later_phase_pointer": "Run this step again on a host with Docker available.",
                }
            )
            verification_results.append(
                _container_build_verification_result(
                    "image_build_deferred_reported",
                    True,
                    {
                        "docker_available": False,
                        "build_spec_ready": True,
                        "reason": "docker_unavailable",
                    },
                    evidence_type="observed",
                )
            )
            status = "deferred"
        elif completion_mode == "container_capstone_complete":
            blocked_capabilities.append(
                {
                    "capability": "docker_available",
                    "reason": "Docker is required for container_capstone_complete image build evidence.",
                    "next_action": "Run prepare_capstone_container_ci on a host with Docker available.",
                }
            )
            status = "blocked"
        else:
            status = "blocked"
        return _container_build_result(
            status=status if not blocked_capabilities else "blocked",
            completion_mode=completion_mode,
            container=container,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    build_approved = _container_step_approval_approved(approval_record, "builds_image")
    container["image_build"]["approved"] = build_approved
    if not build_approved:
        blocked_capabilities.append(
            {
                "capability": "image_build_approved",
                "reason": "Docker image build requires an Approval Gate.",
                "required_risk_categories": ["builds_image"],
                "next_action": (
                    "Record approval for build_smoke_check_container_image before running docker build."
                ),
            }
        )
        verification_results.append(
            _container_build_verification_result(
                "image_build_attempt_reported",
                False,
                {"attempted": False, "reason": "missing_builds_image_approval"},
            )
        )
        return _container_build_result(
            status="blocked",
            completion_mode=completion_mode,
            container=container,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    if blocked_capabilities:
        verification_results.append(
            _container_build_verification_result(
                "image_build_attempt_reported",
                False,
                {"attempted": False, "reason": "build_spec_not_ready"},
            )
        )
        return _container_build_result(
            status="blocked",
            completion_mode=completion_mode,
            container=container,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    build_evidence = _run_container_docker_build(path, image_tag, build_timeout_seconds)
    container["image_build"].update(build_evidence)
    verification_results.append(
        _container_build_verification_result(
            "image_build_attempt_reported",
            build_evidence["attempted"],
            build_evidence,
        )
    )
    verification_results.append(
        _container_build_verification_result(
            "image_build_succeeded",
            build_evidence["return_code"] == 0,
            build_evidence,
        )
    )
    if build_evidence["return_code"] != 0:
        return _container_build_result(
            status="failed",
            completion_mode=completion_mode,
            container=container,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    image_id = _inspect_container_image_id(image_tag)
    container["image_build"]["image_id"] = image_id
    artifact_entries.append(_container_image_artifact_entry(image_tag, image_id))

    smoke_approved = _container_step_approval_approved(
        smoke_approval_record,
        "executes_project_code",
    )
    container["smoke_check"]["approved"] = smoke_approved
    if not smoke_approved:
        if completion_mode == "container_capstone_complete":
            blocked_capabilities.append(
                {
                    "capability": "container_smoke_check_approved",
                    "reason": "Container smoke check executes project code and requires approval.",
                    "required_risk_categories": ["executes_project_code"],
                    "next_action": (
                        "Record approval for build_smoke_check_container_image before running container smoke check."
                    ),
                }
            )
            status = "blocked"
        else:
            deferred_capabilities.append(
                {
                    "capability": "container_smoke_check_passed",
                    "reason": "Smoke check was not approved for local readiness.",
                    "later_phase_pointer": "Approve bounded smoke check when runtime execution evidence is needed.",
                }
            )
            status = "succeeded"
        return _container_build_result(
            status=status,
            completion_mode=completion_mode,
            container=container,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    smoke_evidence = _run_container_smoke_check(image_tag, smoke_timeout_seconds)
    container["smoke_check"].update(smoke_evidence)
    verification_results.append(
        _container_build_verification_result(
            "container_smoke_check_passed",
            smoke_evidence["passed"],
            smoke_evidence,
        )
    )
    return _container_build_result(
        status="succeeded" if smoke_evidence["passed"] else "failed",
        completion_mode=completion_mode,
        container=container,
        blocked_capabilities=blocked_capabilities,
        deferred_capabilities=deferred_capabilities,
        verification_results=verification_results,
        artifact_entries=artifact_entries,
    )


def _container_build_spec_ready(
    project_path: Path,
    capstone_runtime_image_spec_result: dict[str, Any] | None,
) -> bool:
    if not (project_path / "Dockerfile").exists():
        return False
    if not isinstance(capstone_runtime_image_spec_result, dict):
        return True
    return capstone_runtime_image_spec_result.get("status") in {
        "validated",
        "generated",
        "succeeded",
    }


def _detect_container_docker_availability() -> dict[str, Any]:
    docker_path = shutil.which("docker")
    if not docker_path:
        return {
            "available": False,
            "docker_path": None,
            "version": None,
            "context": None,
            "return_code": None,
            "reason": "docker_cli_not_found",
        }
    command = ["docker", "version", "--format", "{{.Server.Version}}"]
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
        duration_seconds = round(time.monotonic() - started_at, 3)
        version = (result.stdout or "").strip() or None
        return {
            "available": result.returncode == 0,
            "docker_path": docker_path,
            "version": version,
            "context": "local",
            "command": command,
            "duration_seconds": duration_seconds,
            "return_code": result.returncode,
            "stdout_summary": _container_command_summary(result.stdout),
            "stderr_summary": _container_command_summary(result.stderr),
            "reason": None if result.returncode == 0 else "docker_version_failed",
        }
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "docker_path": docker_path,
            "version": None,
            "context": "local",
            "command": command,
            "return_code": None,
            "reason": "docker_version_timeout",
        }


def _container_image_tag(workflow_inputs: dict[str, Any]) -> str:
    image_name = workflow_inputs.get("image_name") or "capstone-runtime"
    image_tag = workflow_inputs.get("image_tag") or "local"
    return f"{image_name}:{image_tag}"


def _container_step_approval_approved(
    approval_record: dict[str, Any] | None,
    risk_category: str,
) -> bool:
    if not isinstance(approval_record, dict):
        return False
    return (
        approval_record.get("step_id") == CONTAINER_BUILD_STEP
        and approval_record.get("status") == "approved"
        and risk_category in approval_record.get("risk_categories", [])
    )


def _run_container_docker_build(
    project_path: Path,
    image_tag: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = ["docker", "build", "--pull=false", "-f", "Dockerfile", "-t", image_tag, "."]
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration_seconds = round(time.monotonic() - started_at, 3)
        return {
            "attempted": True,
            "command": command,
            "duration_seconds": duration_seconds,
            "return_code": result.returncode,
            "stdout_summary": _container_command_summary(result.stdout),
            "stderr_summary": _container_command_summary(result.stderr),
            "image_tag": image_tag,
        }
    except subprocess.TimeoutExpired:
        duration_seconds = round(time.monotonic() - started_at, 3)
        return {
            "attempted": True,
            "command": command,
            "duration_seconds": duration_seconds,
            "return_code": 124,
            "stdout_summary": "",
            "stderr_summary": f"docker build timed out after {timeout_seconds}s",
            "image_tag": image_tag,
        }


def _inspect_container_image_id(image_tag: str) -> str | None:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def _run_container_smoke_check(image_tag: str, timeout_seconds: int) -> dict[str, Any]:
    command = [
        "docker",
        "run",
        "--rm",
        image_tag,
        "python",
        "-c",
        "from pathlib import Path; assert Path('/app').exists(); print('smoke ok')",
    ]
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        duration_seconds = round(time.monotonic() - started_at, 3)
        return {
            "attempted": True,
            "command": command,
            "duration_seconds": duration_seconds,
            "return_code": result.returncode,
            "stdout_summary": _container_command_summary(result.stdout),
            "stderr_summary": _container_command_summary(result.stderr),
            "passed": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        duration_seconds = round(time.monotonic() - started_at, 3)
        return {
            "attempted": True,
            "command": command,
            "duration_seconds": duration_seconds,
            "return_code": 124,
            "stdout_summary": "",
            "stderr_summary": f"container smoke check timed out after {timeout_seconds}s",
            "passed": False,
        }


def _container_command_summary(value: str | None, limit: int = 2000) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _container_image_artifact_entry(image_tag: str, image_id: str | None) -> dict[str, Any]:
    return {
        "artifact_type": "container_image",
        "producing_step": CONTAINER_BUILD_STEP,
        "state": "external",
        "uri": f"docker://{image_tag}",
        "metadata": {
            "image_id": image_id,
            "image_tag": image_tag,
        },
    }


def _container_build_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
    *,
    evidence_type: str = "observed",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": evidence_type,
        "source_step": CONTAINER_BUILD_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _container_build_result(
    *,
    status: str,
    completion_mode: str,
    container: dict[str, Any],
    blocked_capabilities: list[dict[str, Any]],
    deferred_capabilities: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
    artifact_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "container": _redacted_evidence(container),
        "blocked_capabilities": _redacted_evidence(blocked_capabilities),
        "deferred_capabilities": _redacted_evidence(deferred_capabilities),
        "verification_results": verification_results,
        "artifact_manifest": {"entries": _redacted_evidence(artifact_entries)},
        "workflow_input_overrides": {
            "docker_available": bool(container.get("docker", {}).get("available")),
        },
        "next_actions": [
            item["next_action"]
            for item in blocked_capabilities
            if item.get("next_action")
        ],
    }


REGISTRY_TARGET_STEP = "configure_validate_registry_target"
REGISTRY_IMAGE_NAME_PATTERN = re.compile(
    r"^[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*$"
)
REGISTRY_IMAGE_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


def configure_validate_capstone_registry_target(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    capstone_container_build_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate Phase 5 registry target evidence without login or push."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    target_result = _resolve_container_registry_target(
        path,
        inputs,
        capstone_container_build_result,
    )
    registry = target_result["registry"]
    blocked_capabilities: list[dict[str, Any]] = []
    deferred_capabilities: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []

    if registry["status"] == "missing":
        capability = {
            "capability": "registry_target_validated",
            "reason": "No registry target was declared and GitHub repository context was not detected.",
            "next_action": (
                "Provide registry_target or run from a GitHub repository so GHCR can be selected."
            ),
        }
        if completion_mode == "container_capstone_complete":
            blocked_capabilities.append(capability)
            status = "blocked"
        else:
            deferred_capabilities.append(capability)
            status = "deferred"
    elif registry["status"] == "invalid":
        blocked_capabilities.append(
            {
                "capability": "registry_target_validated",
                "reason": registry["validation_error"],
                "next_action": "Provide a valid GHCR, DockerHub, or ECR image reference.",
            }
        )
        status = "failed"
    else:
        auth_capability = _detect_container_registry_auth_capability(registry["provider"])
        registry["auth_capability"] = auth_capability
        artifact_entries.append(_container_registry_artifact_entry(registry))
        if not auth_capability["available"]:
            capability = {
                "capability": "registry_auth_capability_verified",
                "reason": "Registry authentication capability was not detected locally.",
                "next_action": _container_registry_auth_next_action(registry["provider"]),
            }
            if completion_mode == "container_capstone_complete":
                blocked_capabilities.append(capability)
                status = "blocked"
            else:
                deferred_capabilities.append(capability)
                status = "deferred"
        else:
            status = "validated"

    target_valid = registry["status"] == "validated"
    auth_available = bool(registry.get("auth_capability", {}).get("available"))
    secret_safety = _container_registry_secret_safety(registry)
    if not secret_safety["passed"]:
        blocked_capabilities.append(
            {
                "capability": "secret_safety_validated",
                "reason": "Registry target evidence contains secret-like material.",
                "next_action": "Remove raw credentials from registry inputs.",
            }
        )
        status = "failed"

    verification_results = [
        _container_registry_verification_result(
            "registry_target_validated",
            target_valid,
            registry,
        ),
        _container_registry_verification_result(
            "registry_auth_capability_verified",
            auth_available,
            registry.get("auth_capability", {"available": False, "source": None}),
        ),
        _container_registry_verification_result(
            "secret_safety_validated",
            secret_safety["passed"],
            secret_safety,
        ),
    ]
    return _container_registry_result(
        status=status,
        completion_mode=completion_mode,
        registry=registry,
        blocked_capabilities=blocked_capabilities,
        deferred_capabilities=deferred_capabilities,
        verification_results=verification_results,
        artifact_entries=artifact_entries,
    )


def _resolve_container_registry_target(
    project_path: Path,
    workflow_inputs: dict[str, Any],
    capstone_container_build_result: dict[str, Any] | None,
) -> dict[str, Any]:
    explicit_target = workflow_inputs.get("registry_target")
    image_name, image_tag = _container_registry_image_name_tag(
        workflow_inputs,
        capstone_container_build_result,
    )
    if explicit_target:
        return _parse_explicit_container_registry_target(str(explicit_target), image_name, image_tag)

    github_repo = _github_repo_from_local_context(project_path)
    if not github_repo:
        return {
            "registry": {
                "status": "missing",
                "provider": None,
                "provider_support": None,
                "registry_url": None,
                "redacted_image_reference": None,
                "image_name": image_name,
                "image_tag": image_tag,
                "auth_capability": {"available": False, "source": None, "checked": []},
            }
        }

    owner, _repo = github_repo
    image_path = f"{owner.casefold()}/{image_name}"
    image_reference = f"ghcr.io/{image_path}:{image_tag}"
    registry = _validated_registry_payload(
        provider="ghcr",
        provider_support="first_class_default",
        registry_url="ghcr.io",
        image_reference=image_reference,
        image_name=image_name,
        image_tag=image_tag,
        explicit=False,
    )
    return {"registry": registry}


def _container_registry_image_name_tag(
    workflow_inputs: dict[str, Any],
    capstone_container_build_result: dict[str, Any] | None,
) -> tuple[str, str]:
    image_tag_from_build = None
    if isinstance(capstone_container_build_result, dict):
        image_tag_from_build = (
            capstone_container_build_result.get("container", {})
            .get("image_build", {})
            .get("image_tag")
        )
    image_name = workflow_inputs.get("image_name")
    image_tag = workflow_inputs.get("image_tag")
    if (not image_name or not image_tag) and isinstance(image_tag_from_build, str):
        parsed_name, parsed_tag = _split_container_image_tag(image_tag_from_build)
        image_name = image_name or parsed_name
        image_tag = image_tag or parsed_tag
    return str(image_name or "capstone-runtime").casefold(), str(image_tag or "local")


def _split_container_image_tag(image_reference: str) -> tuple[str, str]:
    last_segment = image_reference.rsplit("/", 1)[-1]
    if ":" in last_segment:
        name, tag = last_segment.rsplit(":", 1)
        return name, tag
    return last_segment, "local"


def _parse_explicit_container_registry_target(
    registry_target: str,
    default_image_name: str,
    default_image_tag: str,
) -> dict[str, Any]:
    target = registry_target.strip()
    parsed = urlparse(f"//{target}" if "://" not in target else target)
    reference = parsed.netloc + parsed.path if parsed.netloc else parsed.path
    reference = reference.lstrip("/")
    provider = None
    provider_support = "declared_target"
    registry_url = None
    image_reference = reference
    if reference.startswith("ghcr.io/"):
        provider = "ghcr"
        provider_support = "first_class_declared"
        registry_url = "ghcr.io"
    elif re.match(r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/", reference):
        provider = "ecr"
        registry_url = reference.split("/", 1)[0]
    elif reference.startswith("docker.io/"):
        provider = "dockerhub"
        registry_url = "docker.io"
    elif reference.count("/") == 1 and _looks_like_dockerhub_namespace(reference):
        provider = "dockerhub"
        registry_url = "docker.io"
        image_reference = f"docker.io/{reference}"

    if provider is None:
        return {
            "registry": {
                "status": "invalid",
                "provider": None,
                "provider_support": None,
                "registry_url": None,
                "redacted_image_reference": _redact_container_image_reference(reference),
                "image_name": default_image_name,
                "image_tag": default_image_tag,
                "validation_error": "registry_target must be a GHCR, DockerHub, or ECR image reference.",
                "auth_capability": {"available": False, "source": None, "checked": []},
            }
        }

    image_path = image_reference.split("/", 1)[1] if "/" in image_reference else default_image_name
    if ":" in image_path.rsplit("/", 1)[-1]:
        image_name, image_tag = image_path.rsplit(":", 1)
    else:
        image_name, image_tag = image_path, default_image_tag
        image_reference = f"{image_reference}:{image_tag}"
    return {
        "registry": _validated_registry_payload(
            provider=provider,
            provider_support=provider_support,
            registry_url=registry_url,
            image_reference=image_reference,
            image_name=image_name,
            image_tag=image_tag,
            explicit=True,
        )
    }


def _validated_registry_payload(
    *,
    provider: str,
    provider_support: str,
    registry_url: str,
    image_reference: str,
    image_name: str,
    image_tag: str,
    explicit: bool,
) -> dict[str, Any]:
    validation_error = _container_registry_validation_error(image_name, image_tag)
    return {
        "status": "invalid" if validation_error else "validated",
        "provider": provider,
        "provider_support": provider_support,
        "registry_url": registry_url,
        "redacted_registry_url": _redact_container_image_reference(registry_url),
        "redacted_image_reference": _redact_container_image_reference(image_reference),
        "image_name": image_name,
        "image_tag": image_tag,
        "explicit_target": explicit,
        "validation_error": validation_error,
        "login_attempted": False,
        "push_attempted": False,
        "auth_capability": {"available": False, "source": None, "checked": []},
    }


def _container_registry_validation_error(image_name: str, image_tag: str) -> str | None:
    if not REGISTRY_IMAGE_NAME_PATTERN.fullmatch(image_name):
        return "image name must use lowercase Docker repository syntax."
    if not REGISTRY_IMAGE_TAG_PATTERN.fullmatch(image_tag):
        return "image tag must use Docker tag syntax."
    return None


def _looks_like_dockerhub_namespace(reference: str) -> bool:
    namespace = reference.split("/", 1)[0]
    return "." not in namespace and ":" not in namespace and namespace != "localhost"


def _github_repo_from_local_context(project_path: Path) -> tuple[str, str] | None:
    git_config = project_path / ".git" / "config"
    if not git_config.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(git_config)
    except configparser.Error:
        return None
    if not parser.has_section('remote "origin"') or not parser.has_option(
        'remote "origin"', "url"
    ):
        return None
    return _github_repo_from_url(parser.get('remote "origin"', "url"))


def _github_repo_from_url(remote_url: str) -> tuple[str, str] | None:
    match = re.search(r"github\.com[:/](?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?$", remote_url)
    if not match:
        return None
    return match.group("owner"), match.group("repo")


def _detect_container_registry_auth_capability(provider: str | None) -> dict[str, Any]:
    env_by_provider = {
        "ghcr": ("GITHUB_TOKEN", "GHCR_TOKEN"),
        "dockerhub": ("DOCKER_USERNAME", "DOCKER_PASSWORD", "DOCKERHUB_TOKEN"),
        "ecr": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"),
    }
    checked = list(env_by_provider.get(provider or "", ()))
    if provider == "ghcr":
        available = bool(os.environ.get("GITHUB_TOKEN") or os.environ.get("GHCR_TOKEN"))
    elif provider == "dockerhub":
        available = bool(
            os.environ.get("DOCKER_USERNAME")
            and (os.environ.get("DOCKER_PASSWORD") or os.environ.get("DOCKERHUB_TOKEN"))
        )
    elif provider == "ecr":
        available = bool(
            os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
    else:
        available = False
    return {
        "available": available,
        "source": "environment" if available else None,
        "checked": checked,
        "login_attempted": False,
        "remote_probe_attempted": False,
    }


def _container_registry_auth_next_action(provider: str | None) -> str:
    if provider == "ghcr":
        return "Provide GHCR-capable GitHub token evidence before approval-gated push."
    if provider == "dockerhub":
        return "Provide DockerHub username and token evidence before approval-gated push."
    if provider == "ecr":
        return "Provide AWS ECR credential capability evidence before approval-gated push."
    return "Provide registry authentication capability evidence before approval-gated push."


def _container_registry_secret_safety(registry: dict[str, Any]) -> dict[str, Any]:
    serialized = json.dumps(registry, sort_keys=True)
    secret_patterns = (
        r"ghp_[A-Za-z0-9_]+",
        r"github_pat_[A-Za-z0-9_]+",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(password|token|secret)=\S+",
    )
    violations = [
        {"rule": "secret_like_value", "match": "<redacted-secret>"}
        for pattern in secret_patterns
        if re.search(pattern, serialized)
    ]
    return {
        "passed": not violations,
        "checked_fields": [
            "registry_url",
            "redacted_image_reference",
            "image_name",
            "image_tag",
            "auth_capability",
        ],
        "violations": violations,
    }


def _redact_container_image_reference(image_reference: str | None) -> str | None:
    if not image_reference:
        return None
    redacted = re.sub(r"(?i)(password|token|secret)=[^/@:]+", r"\1=<redacted>", image_reference)
    redacted = re.sub(r"://[^:@/\s]+:[^@/\s]+@", "://<redacted>@", redacted)
    redacted = re.sub(
        r"\b\d{12}(?=\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com)",
        "<redacted-account>",
        redacted,
    )
    return redacted


def _container_registry_artifact_entry(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": "container_registry_target",
        "producing_step": REGISTRY_TARGET_STEP,
        "state": "selected",
        "uri": f"registry://{registry['redacted_image_reference']}",
        "metadata": {
            "provider": registry["provider"],
            "image_name": registry["image_name"],
            "image_tag": registry["image_tag"],
            "auth_capability_available": bool(
                registry.get("auth_capability", {}).get("available")
            ),
        },
    }


def _container_registry_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": "observed",
        "source_step": REGISTRY_TARGET_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _container_registry_result(
    *,
    status: str,
    completion_mode: str,
    registry: dict[str, Any],
    blocked_capabilities: list[dict[str, Any]],
    deferred_capabilities: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
    artifact_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "registry": _redacted_evidence(registry),
        "blocked_capabilities": _redacted_evidence(blocked_capabilities),
        "deferred_capabilities": _redacted_evidence(deferred_capabilities),
        "verification_results": verification_results,
        "artifact_manifest": {"entries": _redacted_evidence(artifact_entries)},
        "workflow_input_overrides": {
            "registry_target_validated": registry.get("status") == "validated",
            "registry_auth_capability_available": bool(
                registry.get("auth_capability", {}).get("available")
            ),
        },
        "next_actions": [
            item["next_action"]
            for item in (*blocked_capabilities, *deferred_capabilities)
            if item.get("next_action")
        ],
    }


REGISTRY_PUSH_STEP = "approval_gated_registry_login_push"


def approval_gated_capstone_registry_login_push(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    capstone_registry_target_result: dict[str, Any] | None = None,
    capstone_container_build_result: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
    push_timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run approval-gated registry login and image push without recording secrets."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    if completion_mode == "container_local_ready":
        deferred = [
            {
                "capability": "approval_gated_registry_login_push",
                "reason": "Registry push is deferred for container_local_ready.",
                "next_action": (
                    "Use completion_mode=container_capstone_complete with validated "
                    "registry target, auth capability, approval, and local image evidence."
                ),
            }
        ]
        return _container_registry_push_result(
            status="deferred",
            completion_mode=completion_mode,
            registry={"push_attempted": False, "login_attempted": False},
            blocked_capabilities=[],
            deferred_capabilities=deferred,
            verification_results=[],
            artifact_entries=[],
        )

    registry = _registry_payload_for_push(path, inputs, capstone_registry_target_result)
    blocked_capabilities: list[dict[str, Any]] = []
    deferred_capabilities: list[dict[str, Any]] = []
    artifact_entries: list[dict[str, Any]] = []
    verification_results: list[dict[str, Any]] = []

    if registry.get("status") != "validated":
        blocked_capabilities.append(
            {
                "capability": "registry_target_validated",
                "reason": registry.get("validation_error")
                or "Registry target is missing or has not been validated.",
                "next_action": "Run configure_validate_capstone_registry_target successfully first.",
            }
        )
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=None,
                push_evidence={
                    "blocked_reason": "registry_target_not_validated",
                    "push_attempted": False,
                },
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="blocked",
            completion_mode=completion_mode,
            registry=registry,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    required_risks = _registry_push_required_risk_categories(registry.get("provider"))
    approval_evidence = _registry_push_approval_evidence(approval_record, required_risks)
    if not approval_evidence["approved"]:
        blocked_capabilities.append(
            {
                "capability": "registry_push_approved",
                "reason": approval_evidence["reason"],
                "required_risk_categories": list(required_risks),
                "next_action": (
                    "Record approval for approval_gated_registry_login_push before "
                    "registry login or push may run."
                ),
            }
        )
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=approval_evidence,
                push_evidence={
                    "blocked_reason": "missing_or_denied_approval",
                    "push_attempted": False,
                    "login_attempted": False,
                },
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="blocked",
            completion_mode=completion_mode,
            registry=registry,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    auth_capability = registry.get("auth_capability")
    if not isinstance(auth_capability, dict) or not auth_capability.get("available"):
        blocked_capabilities.append(
            {
                "capability": "registry_auth_capability_verified",
                "reason": "Registry authentication capability was not available.",
                "next_action": _container_registry_auth_next_action(registry.get("provider")),
            }
        )
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=approval_evidence,
                push_evidence={
                    "blocked_reason": "missing_registry_auth_capability",
                    "push_attempted": False,
                    "login_attempted": False,
                },
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="blocked",
            completion_mode=completion_mode,
            registry=registry,
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    local_image = _registry_push_local_image_reference(inputs, capstone_container_build_result)
    local_image_evidence = _inspect_local_container_image(local_image)
    if not local_image_evidence["available"]:
        blocked_capabilities.append(
            {
                "capability": "local_container_image_available",
                "reason": "Local image required for registry push was not found.",
                "next_action": "Run build_smoke_check_capstone_container_image successfully first.",
            }
        )
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=approval_evidence,
                push_evidence={
                    "blocked_reason": "local_image_missing",
                    "local_image": local_image,
                    "local_image_inspect": local_image_evidence,
                    "push_attempted": False,
                },
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="blocked",
            completion_mode=completion_mode,
            registry={**registry, "local_image": local_image},
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    image_reference = _registry_push_image_reference(registry, inputs)
    login_evidence = _run_container_registry_login(registry, push_timeout_seconds)
    registry["login_attempted"] = True
    if login_evidence["return_code"] != 0:
        blocked_capabilities.append(
            {
                "capability": "registry_auth_capability_verified",
                "reason": "Registry login failed.",
                "next_action": "Inspect registry login output and refresh credentials outside Auto-MLOps.",
            }
        )
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=approval_evidence,
                push_evidence={
                    "blocked_reason": "registry_login_failed",
                    "login": login_evidence,
                    "push_attempted": False,
                },
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="failed",
            completion_mode=completion_mode,
            registry={**registry, "login": login_evidence},
            blocked_capabilities=blocked_capabilities,
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
        )

    tag_evidence = _run_registry_command(
        ["docker", "tag", local_image, image_reference],
        timeout_seconds=60,
    )
    if tag_evidence["return_code"] != 0:
        push_evidence = {
            "action": "docker tag",
            "registry_type": registry.get("provider"),
            "redacted_image_reference": registry.get("redacted_image_reference"),
            "blocked_reason": "registry_image_tag_failed",
            "tag": tag_evidence,
            "push_attempted": False,
        }
        verification_results.extend(
            _registry_push_verification_results(
                registry=registry,
                approval_evidence=approval_evidence,
                push_evidence=push_evidence,
                push_succeeded=False,
                pushed_reference_reported=False,
            )
        )
        return _container_registry_push_result(
            status="failed",
            completion_mode=completion_mode,
            registry={**registry, "login": login_evidence, "tag": tag_evidence},
            blocked_capabilities=[],
            deferred_capabilities=deferred_capabilities,
            verification_results=verification_results,
            artifact_entries=artifact_entries,
            next_actions=["Inspect docker tag output before retrying registry push."],
        )

    push_evidence = _run_registry_command(
        ["docker", "push", image_reference],
        timeout_seconds=push_timeout_seconds,
    )
    registry["push_attempted"] = True
    digest = _inspect_pushed_container_digest(image_reference) or _digest_from_registry_push_output(
        push_evidence
    )
    push_succeeded = push_evidence["return_code"] == 0
    observed_push_evidence = {
        "action": "docker push",
        "command": f"docker push {registry.get('redacted_image_reference')}",
        "registry_type": registry.get("provider"),
        "redacted_image_reference": registry.get("redacted_image_reference"),
        "return_code": push_evidence["return_code"],
        "status": "succeeded" if push_succeeded else "failed",
        "stdout_summary": push_evidence["stdout_summary"],
        "stderr_summary": push_evidence["stderr_summary"],
        "duration_seconds": push_evidence["duration_seconds"],
        "pushed_image_reference": registry.get("redacted_image_reference")
        if push_succeeded
        else None,
        "digest": digest if push_succeeded else None,
        "login": login_evidence,
        "tag": tag_evidence,
    }
    if not push_succeeded:
        observed_push_evidence["blocked_reason"] = "registry_push_failed"

    verification_results.extend(
        _registry_push_verification_results(
            registry=registry,
            approval_evidence=approval_evidence,
            push_evidence=observed_push_evidence,
            push_succeeded=push_succeeded,
            pushed_reference_reported=push_succeeded,
        )
    )
    if push_succeeded:
        artifact_entries.append(_container_registry_push_artifact_entry(registry, digest))
        status = "succeeded"
        next_actions: list[str] = []
    else:
        status = "failed"
        next_actions = ["Inspect registry push output and retry after resolving the failure."]

    return _container_registry_push_result(
        status=status,
        completion_mode=completion_mode,
        registry={
            **registry,
            "login": login_evidence,
            "tag": tag_evidence,
            "push": observed_push_evidence,
            "pushed_image_reference": registry.get("redacted_image_reference")
            if push_succeeded
            else None,
            "digest": digest if push_succeeded else None,
        },
        blocked_capabilities=blocked_capabilities,
        deferred_capabilities=deferred_capabilities,
        verification_results=verification_results,
        artifact_entries=artifact_entries,
        next_actions=next_actions,
    )


def _registry_payload_for_push(
    project_path: Path,
    workflow_inputs: dict[str, Any],
    capstone_registry_target_result: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(capstone_registry_target_result, dict):
        registry = capstone_registry_target_result.get("registry")
        if isinstance(registry, dict):
            return dict(registry)
    target_result = _resolve_container_registry_target(project_path, workflow_inputs, None)
    registry = dict(target_result["registry"])
    if registry.get("status") == "validated":
        registry["auth_capability"] = _detect_container_registry_auth_capability(
            registry.get("provider")
        )
    return registry


def _registry_push_required_risk_categories(provider: str | None) -> tuple[str, ...]:
    risks = ["uses_remote_service_credentials", "pushes_registry"]
    if provider == "ecr":
        risks.append("uses_cloud_credentials")
    return tuple(risks)


def _registry_push_approval_evidence(
    approval_record: dict[str, Any] | None,
    required_risks: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(approval_record, dict):
        return {
            "approved": False,
            "reason": "missing_approval_record",
            "required_risk_categories": list(required_risks),
            "approval_record": None,
        }
    risk_categories = tuple(approval_record.get("risk_categories", ()))
    required_present = all(risk in risk_categories for risk in required_risks)
    approved = (
        approval_record.get("step_id") == REGISTRY_PUSH_STEP
        and approval_record.get("status") == "approved"
        and required_present
    )
    reason = None
    if approval_record.get("status") == "denied":
        reason = "approval_denied"
    elif not required_present:
        reason = "approval_record_missing_required_risks"
    elif approval_record.get("step_id") != REGISTRY_PUSH_STEP:
        reason = "approval_record_wrong_step"
    elif approval_record.get("status") != "approved":
        reason = "approval_record_not_approved"
    return {
        "approved": approved,
        "reason": reason,
        "required_risk_categories": list(required_risks),
        "approval_record": _redacted_evidence(
            {
                "workflow_run_id": approval_record.get("workflow_run_id"),
                "step_id": approval_record.get("step_id"),
                "risk_categories": list(risk_categories),
                "status": approval_record.get("status"),
                "approver": approval_record.get("approver"),
                "timestamp": approval_record.get("timestamp"),
            }
        ),
    }


def _registry_push_local_image_reference(
    workflow_inputs: dict[str, Any],
    capstone_container_build_result: dict[str, Any] | None,
) -> str:
    if isinstance(capstone_container_build_result, dict):
        image_tag = (
            capstone_container_build_result.get("container", {})
            .get("image_build", {})
            .get("image_tag")
        )
        if isinstance(image_tag, str) and image_tag:
            return image_tag
    return _container_image_tag(workflow_inputs)


def _inspect_local_container_image(image_reference: str) -> dict[str, Any]:
    if not shutil.which("docker"):
        return {
            "available": False,
            "image_reference": image_reference,
            "return_code": None,
            "reason": "docker_cli_not_found",
        }
    result = _run_registry_command(
        ["docker", "image", "inspect", image_reference],
        timeout_seconds=30,
    )
    return {
        "available": result["return_code"] == 0,
        "image_reference": image_reference,
        "return_code": result["return_code"],
        "stdout_summary": result["stdout_summary"],
        "stderr_summary": result["stderr_summary"],
        "duration_seconds": result["duration_seconds"],
        "reason": None if result["return_code"] == 0 else "docker_image_inspect_failed",
    }


def _registry_push_image_reference(
    registry: dict[str, Any],
    workflow_inputs: dict[str, Any],
) -> str:
    image_reference = workflow_inputs.get("registry_target") or registry.get("image_reference")
    image_reference = image_reference or registry.get("redacted_image_reference")
    if not isinstance(image_reference, str) or not image_reference:
        registry_url = registry.get("registry_url")
        image_name = registry.get("image_name")
        image_tag = registry.get("image_tag")
        image_reference = f"{registry_url}/{image_name}:{image_tag}"
    return image_reference


def _run_container_registry_login(
    registry: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    provider = registry.get("provider")
    registry_url = registry.get("registry_url")
    if provider == "ecr":
        region = _ecr_region_from_registry_url(str(registry_url or ""))
        password_result = _run_registry_command(
            ["aws", "ecr", "get-login-password", "--region", region],
            timeout_seconds=60,
        )
        if password_result["return_code"] != 0:
            return {
                **password_result,
                "action": "aws ecr get-login-password",
                "registry_type": "ecr",
                "redacted_login_target": registry.get("redacted_registry_url"),
            }
        return _run_registry_command(
            [
                "docker",
                "login",
                str(registry_url),
                "--username",
                "AWS",
                "--password-stdin",
            ],
            timeout_seconds=timeout_seconds,
            input_text=password_result.get("stdout_summary", ""),
            evidence_command=(
                f"aws ecr get-login-password --region {region} | "
                f"docker login {registry.get('redacted_registry_url')} "
                "--username AWS --password-stdin"
            ),
            extra={
                "action": "aws ecr login",
                "registry_type": "ecr",
                "redacted_login_target": registry.get("redacted_registry_url"),
            },
        )

    username = "oauth2"
    secret_value = None
    if provider == "ghcr":
        username = os.environ.get("GITHUB_ACTOR") or "oauth2"
        secret_value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GHCR_TOKEN")
    elif provider == "dockerhub":
        username = os.environ.get("DOCKER_USERNAME") or ""
        secret_value = os.environ.get("DOCKER_PASSWORD") or os.environ.get("DOCKERHUB_TOKEN")
    if not secret_value:
        return {
            "action": "docker login",
            "registry_type": provider,
            "redacted_login_target": registry.get("redacted_registry_url"),
            "command": (
                f"docker login {registry.get('redacted_registry_url')} "
                "--username <redacted> --password-stdin"
            ),
            "return_code": 1,
            "stdout_summary": "",
            "stderr_summary": "registry credential material was not available",
            "duration_seconds": 0,
        }
    return _run_registry_command(
        [
            "docker",
            "login",
            str(registry_url),
            "--username",
            username,
            "--password-stdin",
        ],
        timeout_seconds=timeout_seconds,
        input_text=secret_value,
        evidence_command=(
            f"docker login {registry.get('redacted_registry_url')} "
            "--username <redacted> --password-stdin"
        ),
        extra={
            "action": "docker login",
            "registry_type": provider,
            "redacted_login_target": registry.get("redacted_registry_url"),
        },
    )


def _ecr_region_from_registry_url(registry_url: str) -> str:
    match = re.search(r"\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com$", registry_url)
    return match.group(1) if match else os.environ.get("AWS_REGION", "us-east-1")


def _run_registry_command(
    command: list[str],
    *,
    timeout_seconds: int,
    input_text: str | None = None,
    evidence_command: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            input=input_text,
        )
        return_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        return_code = 124
        stdout = ""
        stderr = f"command timed out after {timeout_seconds}s"
    except FileNotFoundError:
        return_code = 127
        stdout = ""
        stderr = f"command not found: {command[0]}"
    duration_seconds = round(time.monotonic() - started_at, 3)
    payload = {
        "command": evidence_command or _registry_command_summary(command),
        "return_code": return_code,
        "stdout_summary": _redact_registry_command_output(stdout),
        "stderr_summary": _redact_registry_command_output(stderr),
        "duration_seconds": duration_seconds,
    }
    if extra:
        payload.update(extra)
    return payload


def _registry_command_summary(command: list[str]) -> str:
    redacted_parts = ["<redacted>" if "password" in part.casefold() else part for part in command]
    return " ".join(redacted_parts)


def _redact_registry_command_output(value: str | None) -> str:
    text = _container_command_summary(value)
    patterns = (
        r"ghp_[A-Za-z0-9_]+",
        r"github_pat_[A-Za-z0-9_]+",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(password|token|secret)=\S+",
        r"(?i)(Bearer\s+)[A-Za-z0-9._-]+",
    )
    for pattern in patterns:
        text = re.sub(pattern, "<redacted-secret>", text)
    return _redact_container_image_reference(text) or ""


def _inspect_pushed_container_digest(image_reference: str) -> str | None:
    try:
        result = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image_reference,
                "--format",
                "{{index .RepoDigests 0}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "").strip()
    if "@sha256:" in output:
        return output.rsplit("@", 1)[-1]
    if output.startswith("sha256:"):
        return output
    return None


def _digest_from_registry_push_output(push_evidence: dict[str, Any]) -> str | None:
    combined = " ".join(
        str(push_evidence.get(key, "")) for key in ("stdout_summary", "stderr_summary")
    )
    match = re.search(r"sha256:[a-fA-F0-9]{8,64}", combined)
    return match.group(0) if match else None


def _registry_push_verification_results(
    *,
    registry: dict[str, Any],
    approval_evidence: dict[str, Any] | None,
    push_evidence: dict[str, Any],
    push_succeeded: bool,
    pushed_reference_reported: bool,
) -> list[dict[str, Any]]:
    secret_safety = _container_registry_push_secret_safety(
        registry=registry,
        approval_evidence=approval_evidence,
        push_evidence=push_evidence,
    )
    auth_capability = registry.get("auth_capability", {})
    return [
        _container_registry_push_verification_result(
            "registry_auth_capability_verified",
            bool(isinstance(auth_capability, dict) and auth_capability.get("available")),
            auth_capability if isinstance(auth_capability, dict) else {},
        ),
        _container_registry_push_verification_result(
            "registry_push_approved",
            bool(approval_evidence and approval_evidence.get("approved")),
            approval_evidence or {"approved": False, "reason": "missing_approval_record"},
        ),
        _container_registry_push_verification_result(
            "registry_push_succeeded",
            push_succeeded,
            push_evidence,
        ),
        _container_registry_push_verification_result(
            "pushed_image_reference_reported",
            pushed_reference_reported,
            {
                "pushed_image_reference": push_evidence.get("pushed_image_reference"),
                "digest": push_evidence.get("digest"),
                "redacted_image_reference": registry.get("redacted_image_reference"),
            },
        ),
        _container_registry_push_verification_result(
            "secret_safety_validated",
            secret_safety["passed"],
            secret_safety,
        ),
    ]


def _container_registry_push_secret_safety(
    *,
    registry: dict[str, Any],
    approval_evidence: dict[str, Any] | None,
    push_evidence: dict[str, Any],
) -> dict[str, Any]:
    serialized = json.dumps(
        _redacted_evidence(
            {
                "registry": registry,
                "approval": approval_evidence or {},
                "push": push_evidence,
            }
        ),
        sort_keys=True,
    )
    secret_patterns = (
        r"ghp_[A-Za-z0-9_]+",
        r"github_pat_[A-Za-z0-9_]+",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(password|token|secret)=\S+",
    )
    violations = [
        {"rule": "secret_like_value", "match": "<redacted-secret>"}
        for pattern in secret_patterns
        if re.search(pattern, serialized)
    ]
    return {
        "passed": not violations,
        "checked_fields": ["registry", "approval", "push"],
        "violations": violations,
    }


def _container_registry_push_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": "observed",
        "source_step": REGISTRY_PUSH_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _container_registry_push_artifact_entry(
    registry: dict[str, Any],
    digest: str | None,
) -> dict[str, Any]:
    return {
        "artifact_type": "container_registry_push",
        "producing_step": REGISTRY_PUSH_STEP,
        "state": "external",
        "uri": f"registry://{registry.get('redacted_image_reference')}",
        "checksum": digest,
        "metadata": {
            "provider": registry.get("provider"),
            "pushed_image_reference": registry.get("redacted_image_reference"),
            "digest": digest,
        },
    }


def _container_registry_push_result(
    *,
    status: str,
    completion_mode: str,
    registry: dict[str, Any],
    blocked_capabilities: list[dict[str, Any]],
    deferred_capabilities: list[dict[str, Any]],
    verification_results: list[dict[str, Any]],
    artifact_entries: list[dict[str, Any]],
    next_actions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "registry": _redacted_evidence(registry),
        "blocked_capabilities": _redacted_evidence(blocked_capabilities),
        "deferred_capabilities": _redacted_evidence(deferred_capabilities),
        "verification_results": verification_results,
        "artifact_manifest": {"entries": _redacted_evidence(artifact_entries)},
        "workflow_input_overrides": {
            "registry_push_succeeded": status == "succeeded",
            "pushed_image_reference_available": status == "succeeded",
        },
        "next_actions": next_actions
        if next_actions is not None
        else [
            item["next_action"]
            for item in (*blocked_capabilities, *deferred_capabilities)
            if item.get("next_action")
        ],
    }


CONTAINER_CI_RECORD_STEP = "record_container_ci_evidence_handoff"
CONTAINER_CI_EVIDENCE_PATH = ".auto_mlops/capstone/container_ci_evidence.json"
CAPSTONE_CI_WORKFLOW_PATH = ".github/workflows/capstone-ci.yml"
PHASE6_DEFERRED_CAPABILITIES = (
    "kserve_deployment",
    "helm_packaging",
    "argocd_gitops",
    "eks_provisioning",
)


def record_capstone_container_ci_evidence_handoff(
    project_path: str,
    workflow_inputs: dict[str, Any] | None = None,
    capstone_container_upstream_evidence: dict[str, Any] | None = None,
    capstone_runtime_image_spec_result: dict[str, Any] | None = None,
    capstone_container_build_result: dict[str, Any] | None = None,
    capstone_registry_target_result: dict[str, Any] | None = None,
    capstone_registry_push_result: dict[str, Any] | None = None,
    verification_results: list[dict[str, Any]] | None = None,
    artifact_manifest: dict[str, Any] | None = None,
    approval_record: dict[str, Any] | None = None,
    remote_ci_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Phase 5 Capstone CI Evidence and durable container/CI handoff."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    if not _container_ci_write_approved(approval_record):
        blocked = [
            {
                "capability": "capstone_ci_workflow_write_approved",
                "reason": "Writing capstone CI workflow and evidence requires an Approval Gate.",
                "required_risk_categories": ["writes_project_files"],
                "next_action": (
                    "Record approval for record_container_ci_evidence_handoff before "
                    "writing .github/workflows/capstone-ci.yml."
                ),
            }
        ]
        return {
            "success": True,
            "status": "blocked",
            "workflow_id": "prepare_capstone_container_ci",
            "completion_mode": (workflow_inputs or {}).get(
                "completion_mode",
                "container_local_ready",
            ),
            "blocked_capabilities": blocked,
            "deferred_capabilities": [],
            "verification_results": [
                _container_ci_verification_result(
                    "capstone_ci_workflow_reported",
                    False,
                    {"reason": "missing_writes_project_files_approval"},
                    evidence_type="declared",
                ),
                _container_ci_verification_result(
                    "container_ci_evidence_artifact_reported",
                    False,
                    {"reason": "missing_writes_project_files_approval"},
                ),
            ],
            "artifact_manifest": {"entries": []},
            "next_actions": [blocked[0]["next_action"]],
        }

    inputs = workflow_inputs or {}
    completion_mode = inputs.get("completion_mode", "container_local_ready")
    existing_results = list(verification_results or [])
    existing_entries = _artifact_entries_from_manifest_payload(artifact_manifest)
    step_results = [
        capstone_container_upstream_evidence,
        capstone_runtime_image_spec_result,
        capstone_container_build_result,
        capstone_registry_target_result,
        capstone_registry_push_result,
    ]
    for step_result in step_results:
        existing_results.extend(_verification_results_from_tool_payload(step_result))
        existing_entries.extend(_artifact_entries_from_manifest_payload(_manifest_from_result(step_result)))

    workflow_path = path / CAPSTONE_CI_WORKFLOW_PATH
    workflow_text = _generate_capstone_ci_workflow_yaml(
        completion_mode=completion_mode,
        registry_result=capstone_registry_target_result,
        registry_push_result=capstone_registry_push_result,
    )
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.write_text(workflow_text)

    ci_validation = _validate_capstone_ci_workflow(workflow_text, completion_mode)
    remote_run = _container_ci_remote_run_evidence(remote_ci_evidence)
    if remote_run["status"] != "succeeded":
        remote_deferred = [
            {
                "capability": "remote_github_actions_run_evidence",
                "reason": "Remote GitHub Actions run evidence is unavailable in Phase 5.",
                "next_action": (
                    "Inspect GitHub Actions after pushing the generated workflow, if remote "
                    "CI evidence is required."
                ),
            }
        ]
    else:
        remote_deferred = []

    container = _container_ci_container_section(
        capstone_runtime_image_spec_result,
        capstone_container_build_result,
    )
    registry = _container_ci_registry_section(
        capstone_registry_target_result,
        capstone_registry_push_result,
    )
    upstream = _container_ci_upstream_section(capstone_container_upstream_evidence)
    ci = {
        "workflow_path": CAPSTONE_CI_WORKFLOW_PATH,
        "workflow_generated": True,
        "yaml_parse": ci_validation["yaml_parse"],
        "bounded_repo_local_commands": ci_validation["bounded_repo_local_commands"],
        "required_checks": ci_validation["required_checks"],
        "registry_usage": ci_validation["registry_usage"],
        "remote_run": remote_run,
        "act_simulation": {"status": "deferred", "reason": "act simulation is not required."},
    }

    artifact_entries = _dedupe_artifact_entries(
        [
            *existing_entries,
            {
                "artifact_type": "github_actions_workflow",
                "producing_step": CONTAINER_CI_RECORD_STEP,
                "state": "generated",
                "path": CAPSTONE_CI_WORKFLOW_PATH,
                "checksum": _sha256_file(workflow_path),
                "metadata": {"workflow_name": "Capstone CI"},
            },
            *_container_ci_image_artifact_entries(registry, container),
            {
                "artifact_type": "container_ci_evidence",
                "producing_step": CONTAINER_CI_RECORD_STEP,
                "state": "generated",
                "path": CONTAINER_CI_EVIDENCE_PATH,
                "metadata": {
                    "schema_version": "phase5.container_ci_evidence.v1",
                    "completion_mode": completion_mode,
                },
            },
        ]
    )
    handoff_results = [
        _container_ci_verification_result(
            "container_ci_evidence_artifact_reported",
            True,
            {"path": CONTAINER_CI_EVIDENCE_PATH},
        ),
        _container_ci_verification_result(
            "container_artifact_manifest_reported",
            bool(artifact_entries),
            {"artifact_count": len(artifact_entries)},
        ),
        _container_ci_verification_result(
            "capstone_ci_workflow_reported",
            True,
            {"path": CAPSTONE_CI_WORKFLOW_PATH},
            evidence_type="declared",
        ),
        _container_ci_verification_result(
            "capstone_ci_workflow_validated",
            ci_validation["passed"],
            ci_validation,
        ),
        _container_ci_verification_result(
            "secret_safety_validated",
            ci_validation["secret_safety"]["passed"],
            ci_validation["secret_safety"],
        ),
    ]
    if completion_mode == "container_capstone_complete":
        handoff_results.append(
            _container_ci_verification_result(
                "capstone_ci_registry_usage_validated",
                ci_validation["registry_usage"]["secret_backed"],
                ci_validation["registry_usage"],
            )
        )

    all_results = [*existing_results, *handoff_results]
    blocked_capabilities = _dedupe_capabilities(
        [
            *_capabilities_from_result(capstone_container_upstream_evidence, "blocked_capabilities"),
            *_capabilities_from_result(capstone_runtime_image_spec_result, "blocked_capabilities"),
            *_capabilities_from_result(capstone_container_build_result, "blocked_capabilities"),
            *_capabilities_from_result(capstone_registry_target_result, "blocked_capabilities"),
            *_capabilities_from_result(capstone_registry_push_result, "blocked_capabilities"),
        ]
    )
    deferred_capabilities = _dedupe_capabilities(
        [
            *_capabilities_from_result(capstone_container_upstream_evidence, "deferred_capabilities"),
            *_capabilities_from_result(capstone_runtime_image_spec_result, "deferred_capabilities"),
            *_capabilities_from_result(capstone_container_build_result, "deferred_capabilities"),
            *_capabilities_from_result(capstone_registry_target_result, "deferred_capabilities"),
            *_capabilities_from_result(capstone_registry_push_result, "deferred_capabilities"),
            *remote_deferred,
        ]
    )
    next_phase_readiness = _container_ci_next_phase_readiness(
        completion_mode=completion_mode,
        registry=registry,
        container=container,
        ci=ci,
        blocked_capabilities=blocked_capabilities,
        deferred_capabilities=deferred_capabilities,
    )
    status = _container_ci_status_from_evidence(
        completion_mode=completion_mode,
        verification_results=all_results,
        artifact_entries=artifact_entries,
    )

    evidence_path = path / CONTAINER_CI_EVIDENCE_PATH
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = {
        "schema_version": "phase5.container_ci_evidence.v1",
        "created_at": _utc_now_iso(),
        "workflow_id": "prepare_capstone_container_ci",
        "status": status,
        "completion_mode": completion_mode,
        "upstream_evidence": upstream,
        "container": container,
        "registry": registry,
        "ci": ci,
        "approval_records": [_container_ci_approval_record(approval_record)],
        "blocked_capabilities": blocked_capabilities,
        "deferred_capabilities": deferred_capabilities,
        "verification_results": all_results,
        "artifact_manifest": {"entries": artifact_entries},
        "next_phase_readiness": next_phase_readiness,
    }
    redacted = _redacted_evidence(evidence)
    evidence_path.write_text(json.dumps(redacted, indent=2, sort_keys=True) + "\n")

    return {
        "success": True,
        "status": status,
        "workflow_id": "prepare_capstone_container_ci",
        "completion_mode": completion_mode,
        "evidence_path": CONTAINER_CI_EVIDENCE_PATH,
        "container_ci_evidence": redacted,
        "upstream_evidence": upstream,
        "container": container,
        "registry": registry,
        "ci": ci,
        "approval_records": [_container_ci_approval_record(approval_record)],
        "blocked_capabilities": blocked_capabilities,
        "deferred_capabilities": deferred_capabilities,
        "next_phase_readiness": next_phase_readiness,
        "verification_results": handoff_results,
        "artifact_manifest": {
            "entries": [
                entry for entry in artifact_entries if entry.get("producing_step") == CONTAINER_CI_RECORD_STEP
            ]
        },
        "next_actions": next_phase_readiness["missing_blockers"],
    }


def _container_ci_write_approved(approval_record: dict[str, Any] | None) -> bool:
    return bool(
        isinstance(approval_record, dict)
        and approval_record.get("step_id") == CONTAINER_CI_RECORD_STEP
        and approval_record.get("status") == "approved"
        and approval_record.get("risk_categories") == ["writes_project_files"]
    )


def _manifest_from_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    return result.get("artifact_manifest") if isinstance(result, dict) else None


def _verification_results_from_tool_payload(
    result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    raw_results = result.get("verification_results", [])
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
    return [item for item in raw_results if isinstance(item, dict)]


def _generate_capstone_ci_workflow_yaml(
    *,
    completion_mode: str,
    registry_result: dict[str, Any] | None,
    registry_push_result: dict[str, Any] | None,
) -> str:
    image_reference = _container_ci_image_reference(registry_result, registry_push_result)
    push_enabled = "true" if completion_mode == "container_capstone_complete" else "false"
    return (
        "name: Capstone CI\n"
        "\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "\n"
        "permissions:\n"
        "  contents: read\n"
        "  packages: write\n"
        "\n"
        "env:\n"
        "  COMPLETION_MODE: " + str(completion_mode) + "\n"
        "  IMAGE_REF: " + str(image_reference or "ghcr.io/${{ github.repository }}:capstone") + "\n"
        "  RUN_IMAGE_BUILD: \"false\"\n"
        "  PUSH_IMAGE: \"" + push_enabled + "\"\n"
        "\n"
        "jobs:\n"
        "  capstone-ci:\n"
        "    runs-on: ubuntu-latest\n"
        "    timeout-minutes: 20\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: \"3.11\"\n"
        "      - name: Install bounded dependencies\n"
        "        run: |\n"
        "          python -m pip install --upgrade pip\n"
        "          if [ -f requirements.txt ]; then python -m pip install -r requirements.txt; "
        "elif [ -f pyproject.toml ]; then python -m pip install -e .; fi\n"
        "      - name: Run selected tests\n"
        "        run: python -m pytest tests/unit/test_workflow_registry.py "
        "tests/unit/test_agent_loop.py tests/unit/test_execute_step.py -q\n"
        "      - name: Validate data-stage evidence\n"
        "        run: |\n"
        "          python - <<'PY'\n"
        "          import json\n"
        "          from pathlib import Path\n"
        "          path = Path('.auto_mlops/capstone/data_stage_evidence.json')\n"
        "          if path.exists():\n"
        "              data = json.loads(path.read_text())\n"
        "              assert data.get('schema_version') == 'phase4.data_stage_evidence.v1'\n"
        "          PY\n"
        "      - name: Validate training or best artifact evidence\n"
        "        run: |\n"
        "          python - <<'PY'\n"
        "          import json\n"
        "          from pathlib import Path\n"
        "          candidates = [\n"
        "              Path('.auto_mlops/capstone/training_evidence.json'),\n"
        "              Path('.auto_mlops/capstone/training_mlflow_evidence.json'),\n"
        "              Path('outputs/model_selection_result.json'),\n"
        "          ]\n"
        "          for candidate in candidates:\n"
        "              if candidate.exists():\n"
        "                  json.loads(candidate.read_text())\n"
        "                  break\n"
        "          PY\n"
        "      - name: Optional image build\n"
        "        if: env.RUN_IMAGE_BUILD == 'true'\n"
        "        run: docker build --pull=false -t \"$IMAGE_REF\" .\n"
        "      - name: Optional registry login\n"
        "        if: env.PUSH_IMAGE == 'true'\n"
        "        run: echo \"${{ secrets.GITHUB_TOKEN }}\" | docker login ghcr.io "
        "-u \"${{ github.actor }}\" --password-stdin\n"
        "      - name: Optional registry push\n"
        "        if: env.PUSH_IMAGE == 'true'\n"
        "        run: docker push \"$IMAGE_REF\"\n"
    )


def _container_ci_image_reference(
    registry_result: dict[str, Any] | None,
    registry_push_result: dict[str, Any] | None,
) -> str | None:
    for result in (registry_push_result, registry_result):
        if not isinstance(result, dict):
            continue
        registry = result.get("registry")
        if not isinstance(registry, dict):
            continue
        reference = registry.get("pushed_image_reference") or registry.get(
            "redacted_image_reference"
        )
        if isinstance(reference, str) and reference:
            return reference
    return None


def _validate_capstone_ci_workflow(workflow_text: str, completion_mode: str) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(workflow_text)
        yaml_ok = isinstance(parsed, dict) and "jobs" in parsed
        yaml_error = None
    except yaml.YAMLError as exc:
        parsed = {}
        yaml_ok = False
        yaml_error = str(exc)
    commands = _capstone_ci_run_commands(parsed if isinstance(parsed, dict) else {})
    bounded = _capstone_ci_bounded_commands(commands)
    required_checks = {
        "tests": any("pytest" in command for command in commands),
        "data_stage_evidence_validation": any(
            "data_stage_evidence.json" in command for command in commands
        ),
        "training_or_best_artifact_validation": any(
            "training_evidence" in command or "model_selection_result" in command
            for command in commands
        ),
        "optional_image_build": any("docker build" in command for command in commands),
        "full_training_deferred": not any(
            forbidden in command.casefold()
            for command in commands
            for forbidden in ("max_epochs", "hpo", "learning-rate finder", "dvc repro train")
        ),
    }
    registry_usage = _capstone_ci_registry_usage(workflow_text, completion_mode)
    secret_safety = _capstone_ci_secret_safety(workflow_text)
    return {
        "passed": yaml_ok
        and bounded["passed"]
        and all(required_checks.values())
        and secret_safety["passed"]
        and (completion_mode != "container_capstone_complete" or registry_usage["secret_backed"]),
        "yaml_parse": {"passed": yaml_ok, "error": yaml_error},
        "bounded_repo_local_commands": bounded,
        "required_checks": required_checks,
        "registry_usage": registry_usage,
        "secret_safety": secret_safety,
    }


def _capstone_ci_run_commands(parsed_workflow: dict[str, Any]) -> list[str]:
    commands: list[str] = []
    jobs = parsed_workflow.get("jobs", {})
    if not isinstance(jobs, dict):
        return commands
    for job in jobs.values():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                commands.append(step["run"])
    return commands


def _capstone_ci_bounded_commands(commands: list[str]) -> dict[str, Any]:
    forbidden_patterns = (
        r"\bcurl\b",
        r"\bwget\b",
        r"\bsudo\b",
        r"\bkubectl\b",
        r"\bhelm\b",
        r"\bargocd\b",
        r"\beksctl\b",
        r"\bterraform\b",
        r"\brm\s+-rf\s+/",
    )
    violations = []
    for command in commands:
        for pattern in forbidden_patterns:
            if re.search(pattern, command, flags=re.IGNORECASE):
                violations.append({"command": command, "rule": pattern})
    return {
        "passed": not violations and bool(commands),
        "command_count": len(commands),
        "violations": violations,
        "policy": "Commands must be bounded CI checks against repository files.",
    }


def _capstone_ci_registry_usage(workflow_text: str, completion_mode: str) -> dict[str, Any]:
    includes_push = "docker push" in workflow_text
    uses_github_secrets = "secrets.GITHUB_TOKEN" in workflow_text
    plaintext_credential = bool(
        re.search(
            r"(?i)(ghp_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|password\s*[:=]\s*[^\s$])",
            workflow_text,
        )
    )
    return {
        "includes_registry_push": includes_push,
        "uses_github_secrets": uses_github_secrets,
        "plaintext_credentials_detected": plaintext_credential,
        "secret_backed": (
            includes_push
            and uses_github_secrets
            and not plaintext_credential
            and completion_mode == "container_capstone_complete"
        )
        or completion_mode != "container_capstone_complete",
    }


def _capstone_ci_secret_safety(workflow_text: str) -> dict[str, Any]:
    secret_patterns = (
        r"ghp_[A-Za-z0-9_]+",
        r"github_pat_[A-Za-z0-9_]+",
        r"AKIA[0-9A-Z]{16}",
        r"(?i)(password|token|secret)\s*[:=]\s*(?!\$\{\{\s*secrets\.)[A-Za-z0-9._-]+",
    )
    violations = [
        {"rule": "plaintext_secret", "match": "<redacted-secret>"}
        for pattern in secret_patterns
        if re.search(pattern, workflow_text)
    ]
    return {
        "passed": not violations,
        "checked_fields": ["capstone-ci.yml"],
        "violations": violations,
    }


def _container_ci_remote_run_evidence(remote_ci_evidence: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(remote_ci_evidence, dict) and remote_ci_evidence.get("status"):
        return _redacted_evidence(remote_ci_evidence)
    return {
        "status": "deferred",
        "required": False,
        "reason": "Remote GitHub Actions run evidence was not inspected in Phase 5.",
    }


def _container_ci_container_section(
    image_spec_result: dict[str, Any] | None,
    build_result: dict[str, Any] | None,
) -> dict[str, Any]:
    image_spec_container = (
        image_spec_result.get("container", {}) if isinstance(image_spec_result, dict) else {}
    )
    build_container = build_result.get("container", {}) if isinstance(build_result, dict) else {}
    return _redacted_evidence(
        {
            "runtime_image": image_spec_container.get("runtime_image", {}),
            "dependency_context": image_spec_container.get("dependency_context", {}),
            "secret_safety": image_spec_container.get("secret_safety", {}),
            "build": build_container,
            "status": build_result.get("status") if isinstance(build_result, dict) else "missing",
        }
    )


def _container_ci_registry_section(
    registry_result: dict[str, Any] | None,
    registry_push_result: dict[str, Any] | None,
) -> dict[str, Any]:
    target = registry_result.get("registry", {}) if isinstance(registry_result, dict) else {}
    push = registry_push_result.get("registry", {}) if isinstance(registry_push_result, dict) else {}
    return _redacted_evidence(
        {
            "target": target,
            "push": push,
            "target_status": registry_result.get("status")
            if isinstance(registry_result, dict)
            else "missing",
            "push_status": registry_push_result.get("status")
            if isinstance(registry_push_result, dict)
            else "deferred",
        }
    )


def _container_ci_upstream_section(upstream_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(upstream_result, dict):
        return {"status": "missing", "upstream_evidence": {}}
    return _redacted_evidence(
        {
            "status": upstream_result.get("status"),
            "upstream_evidence": upstream_result.get("upstream_evidence", {}),
            "blocked_capabilities": upstream_result.get("blocked_capabilities", []),
            "deferred_capabilities": upstream_result.get("deferred_capabilities", []),
        }
    )


def _container_ci_image_artifact_entries(
    registry: dict[str, Any],
    container: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = []
    image_reference = registry.get("push", {}).get("pushed_image_reference") or registry.get(
        "target", {}
    ).get("redacted_image_reference")
    digest = registry.get("push", {}).get("digest") or registry.get("target", {}).get("digest")
    local_image = container.get("build", {}).get("image_build", {}).get("image_id")
    if image_reference:
        entries.append(
            {
                "artifact_type": "container_image_reference",
                "producing_step": CONTAINER_CI_RECORD_STEP,
                "state": "selected",
                "uri": f"registry://{image_reference}",
                "checksum": digest,
                "metadata": {"local_image_id": local_image},
            }
        )
    return entries


def _container_ci_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
    *,
    evidence_type: str = "observed",
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": evidence_type,
        "source_step": CONTAINER_CI_RECORD_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _capabilities_from_result(
    result: dict[str, Any] | None,
    key: str,
) -> list[dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    capabilities = result.get(key, [])
    return [item for item in capabilities if isinstance(item, dict)]


def _dedupe_capabilities(capabilities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for capability in capabilities:
        key = (capability.get("stage"), capability.get("capability"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(capability)
    return deduped


def _container_ci_next_phase_readiness(
    *,
    completion_mode: str,
    registry: dict[str, Any],
    container: dict[str, Any],
    ci: dict[str, Any],
    blocked_capabilities: list[dict[str, Any]],
    deferred_capabilities: list[dict[str, Any]],
) -> dict[str, Any]:
    image_reference = registry.get("push", {}).get("pushed_image_reference") or registry.get(
        "target", {}
    ).get("redacted_image_reference")
    digest = registry.get("push", {}).get("digest") or registry.get("target", {}).get("digest")
    missing_blockers = [
        item.get("capability")
        for item in (*blocked_capabilities, *deferred_capabilities)
        if item.get("capability")
    ]
    return {
        "completion_mode": completion_mode,
        "image": {
            "reference": image_reference,
            "digest": digest,
            "local_image_id": container.get("build", {}).get("image_build", {}).get("image_id"),
            "ready_for_phase6": bool(image_reference or digest),
        },
        "registry_push_status": registry.get("push_status"),
        "ci_validation_status": "validated"
        if ci.get("yaml_parse", {}).get("passed")
        else "blocked",
        "missing_blockers": missing_blockers,
        "deferred_capabilities": list(PHASE6_DEFERRED_CAPABILITIES),
        "phase6_claims": {
            "kserve_deployment_complete": False,
            "helm_packaging_complete": False,
            "argocd_gitops_complete": False,
            "eks_provisioning_complete": False,
        },
    }


def _container_ci_status_from_evidence(
    *,
    completion_mode: str,
    verification_results: list[dict[str, Any]],
    artifact_entries: list[dict[str, Any]],
) -> str:
    checks = {
        result.get("check_name"): result.get("passed")
        for result in verification_results
        if isinstance(result, dict)
    }
    artifact_types = {
        entry.get("artifact_type") for entry in artifact_entries if isinstance(entry, dict)
    }
    common_passed = all(
        checks.get(check_name) is True
        for check_name in (
            "upstream_evidence_resolved",
            "container_build_spec_reported",
            "dependency_context_reported",
            "container_ci_evidence_artifact_reported",
            "container_artifact_manifest_reported",
            "capstone_ci_workflow_reported",
            "capstone_ci_workflow_validated",
            "secret_safety_validated",
            "docker_availability_reported",
        )
    )
    has_model = checks.get("local_model_artifact_resolved") is True or checks.get(
        "mlflow_best_artifact_resolved"
    ) is True
    has_evidence_artifacts = {
        "container_ci_evidence",
        "github_actions_workflow",
        "container_build_spec",
    }.issubset(artifact_types)
    if completion_mode == "container_local_ready":
        build_reported = (
            checks.get("image_build_attempt_reported") is True
            or checks.get("image_build_deferred_reported") is True
        )
        return (
            "succeeded"
            if common_passed and has_model and build_reported and has_evidence_artifacts
            else "blocked"
        )

    capstone_required = all(
        checks.get(check_name) is True
        for check_name in (
            "data_stage_capstone_complete_verified",
            "mlflow_best_artifact_verified",
            "training_lineage_verified",
            "docker_available",
            "image_build_succeeded",
            "container_smoke_check_passed",
            "registry_target_validated",
            "registry_auth_capability_verified",
            "registry_push_approved",
            "registry_push_succeeded",
            "pushed_image_reference_reported",
            "capstone_ci_registry_usage_validated",
        )
    )
    failed_runtime = any(
        result.get("passed") is False
        and result.get("check_name")
        in {"image_build_succeeded", "container_smoke_check_passed"}
        for result in verification_results
        if isinstance(result, dict)
    )
    if failed_runtime:
        return "failed"
    return "succeeded" if common_passed and capstone_required and has_evidence_artifacts else "blocked"


def _container_ci_approval_record(approval_record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(approval_record, dict):
        return {}
    return _redacted_evidence(
        {
            "workflow_run_id": approval_record.get("workflow_run_id"),
            "step_id": approval_record.get("step_id"),
            "risk_categories": approval_record.get("risk_categories", []),
            "status": approval_record.get("status"),
            "approver": approval_record.get("approver"),
            "timestamp": approval_record.get("timestamp"),
        }
    )


def _resolve_container_data_stage_evidence(
    project_path: Path,
    workflow_inputs: dict[str, Any],
) -> dict[str, Any]:
    evidence_path = _container_evidence_path(
        project_path,
        workflow_inputs.get("data_stage_evidence_path"),
        DATA_STAGE_EVIDENCE_PATH,
    )
    relative_path = relative_to_project(str(project_path), evidence_path)
    missing = {
        "status": "blocked",
        "capability": "data_stage_evidence_artifact_reported",
        "path": relative_path,
        "reason": "data_stage_evidence.json is missing.",
    }
    if not evidence_path.exists():
        return {"status": "missing", "evidence": missing, "artifact_entry": None}
    try:
        evidence = json.loads(evidence_path.read_text())
    except json.JSONDecodeError:
        missing["reason"] = "data_stage_evidence.json is not valid JSON."
        return {"status": "blocked", "evidence": missing, "artifact_entry": None}

    required_ok = (
        evidence.get("schema_version") == "phase4.data_stage_evidence.v1"
        and evidence.get("workflow_id") == "prepare_capstone_data"
        and evidence.get("status") == "succeeded"
        and evidence.get("completion_mode") in {"local_ready", "capstone_complete"}
        and isinstance(evidence.get("datasets"), list)
        and len(evidence.get("datasets", [])) >= 2
        and isinstance(evidence.get("dvc"), dict)
        and isinstance(evidence.get("artifact_manifest"), dict)
    )
    status = "resolved" if required_ok else "blocked"
    redacted = _redacted_evidence(evidence)
    data_evidence = {
        "status": status,
        "path": relative_path,
        "schema_version": evidence.get("schema_version"),
        "workflow_id": evidence.get("workflow_id"),
        "raw_status": evidence.get("status"),
        "completion_mode": evidence.get("completion_mode"),
        "dataset_count": len(evidence.get("datasets", []))
        if isinstance(evidence.get("datasets"), list)
        else 0,
        "dvc": redacted.get("dvc") if isinstance(redacted, dict) else None,
        "blocked_capabilities": redacted.get("blocked_capabilities", [])
        if isinstance(redacted, dict)
        else [],
    }
    artifact_entry = {
        "artifact_type": "data_stage_evidence",
        "producing_step": CONTAINER_UPSTREAM_SOURCE_STEP,
        "state": "validated",
        "path": relative_path,
        "checksum": _sha256_file(evidence_path),
        "metadata": {
            "source_workflow_id": "prepare_capstone_data",
            "data_stage_status": evidence.get("status"),
            "completion_mode": evidence.get("completion_mode"),
        },
    }
    return {
        "status": status,
        "completion_mode": evidence.get("completion_mode"),
        "evidence": data_evidence,
        "artifact_entry": artifact_entry if required_ok else None,
    }


def _container_evidence_path(
    project_path: Path,
    configured_path: Any,
    default_relative_path: str,
) -> Path:
    if isinstance(configured_path, str) and configured_path:
        path = Path(configured_path)
        return path if path.is_absolute() else project_path / path
    return project_path / default_relative_path


def _load_container_training_evidence(project_path: Path) -> dict[str, Any] | None:
    for relative_path in TRAINING_EVIDENCE_CANDIDATE_PATHS:
        candidate = project_path / relative_path
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and _has_structured_training_evidence(payload):
            payload["_evidence_path"] = relative_path
            return payload
    return None


def _has_structured_training_evidence(payload: dict[str, Any]) -> bool:
    return isinstance(payload.get("artifact_manifest"), dict) or isinstance(
        payload.get("verification_results"),
        list,
    )


def _resolve_local_model_artifact(
    project_path: Path,
    workflow_inputs: dict[str, Any],
) -> dict[str, Any]:
    configured = workflow_inputs.get("local_model_artifact_path")
    if not isinstance(configured, str) or not configured:
        return {
            "status": "missing",
            "evidence": {
                "status": "missing",
                "capability": "local_model_artifact_resolved",
                "reason": "local_model_artifact_path was not provided.",
            },
            "artifact_entry": None,
        }
    artifact_path = Path(configured)
    full_path = artifact_path if artifact_path.is_absolute() else project_path / artifact_path
    if not full_path.is_file():
        return {
            "status": "blocked",
            "evidence": {
                "status": "blocked",
                "path": _redact_path_if_sensitive(configured),
                "reason": "local_model_artifact_path does not reference an existing file.",
            },
            "artifact_entry": None,
        }
    relative_path = relative_to_project(str(project_path), full_path)
    artifact_entry = {
        "artifact_type": "model_artifact",
        "producing_step": CONTAINER_UPSTREAM_SOURCE_STEP,
        "state": "selected",
        "path": _redact_path_if_sensitive(relative_path),
        "checksum": _artifact_checksum(project_path, relative_path),
        "metadata": {"source": "local_model_artifact_fallback"},
    }
    return {
        "status": "resolved",
        "evidence": {
            "status": "resolved",
            "path": _redact_path_if_sensitive(relative_path),
            "model_type": _model_type_from_artifact(relative_path),
            "source": "local_model_artifact_fallback",
        },
        "artifact_entry": artifact_entry,
    }


def _resolve_mlflow_best_artifact(
    project_path: Path,
    workflow_inputs: dict[str, Any],
    training_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest_artifact = _best_artifact_from_training_manifest(training_evidence)
    if manifest_artifact is not None:
        artifact_entry = _container_model_artifact_entry(project_path, manifest_artifact)
        return {
            "status": "resolved",
            "evidence": {
                "status": "resolved",
                "source": "structured_artifact_manifest",
                "training_evidence_path": training_evidence.get("_evidence_path")
                if training_evidence
                else None,
                "path": artifact_entry.get("path"),
                "uri": artifact_entry.get("uri"),
                "source_run_id": (artifact_entry.get("metadata") or {}).get("source_run_id"),
            },
            "artifact_entry": artifact_entry,
        }

    configured_path = workflow_inputs.get("mlflow_best_artifact_path")
    run_id = workflow_inputs.get("mlflow_run_id")
    if isinstance(configured_path, str) and configured_path and run_id:
        artifact_path = Path(configured_path)
        full_path = artifact_path if artifact_path.is_absolute() else project_path / artifact_path
        if full_path.exists():
            relative_path = relative_to_project(str(project_path), full_path)
            artifact_entry = {
                "artifact_type": "model_artifact",
                "producing_step": CONTAINER_UPSTREAM_SOURCE_STEP,
                "state": "selected",
                "path": _redact_path_if_sensitive(relative_path),
                "checksum": _artifact_checksum(project_path, relative_path),
                "metadata": {
                    "source": "mlflow_best_artifact_input",
                    "source_run_id": str(run_id),
                },
            }
            return {
                "status": "resolved",
                "evidence": {
                    "status": "resolved",
                    "source": "workflow_input",
                    "path": _redact_path_if_sensitive(relative_path),
                    "source_run_id": str(run_id),
                },
                "artifact_entry": artifact_entry,
            }
    return {
        "status": "missing",
        "evidence": {
            "status": "missing",
            "capability": "mlflow_best_artifact_verified",
            "reason": "No structured MLflow-linked best artifact evidence was found.",
        },
        "artifact_entry": None,
    }


def _best_artifact_from_training_manifest(
    training_evidence: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(training_evidence, dict):
        return None
    if not _training_verification_passed(training_evidence, "model_artifact_selected"):
        return None
    if not _training_has_mlflow_link(training_evidence):
        return None
    entries = training_evidence.get("artifact_manifest", {}).get("entries", ())
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("artifact_type") in {
            "model_artifact",
            "checkpoint_or_model_artifact",
            "mlflow_checkpoint_or_model_artifact",
        } and (entry.get("path") or entry.get("uri")):
            return entry
    return None


def _training_has_mlflow_link(training_evidence: dict[str, Any]) -> bool:
    if _training_verification_passed(training_evidence, "mlflow_run_exists"):
        return True
    entries = training_evidence.get("artifact_manifest", {}).get("entries", ())
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        if entry.get("artifact_type") == "mlflow_run" or metadata.get("source_run_id"):
            return True
    return False


def _container_model_artifact_entry(
    project_path: Path,
    source_entry: dict[str, Any],
) -> dict[str, Any]:
    path_value = source_entry.get("path")
    uri_value = source_entry.get("uri")
    redacted_path = _redact_path_if_sensitive(str(path_value)) if path_value else None
    redacted_uri = _redact_uri_if_sensitive(str(uri_value)) if uri_value else None
    metadata = source_entry.get("metadata") if isinstance(source_entry.get("metadata"), dict) else {}
    source_run_id = metadata.get("source_run_id")
    return {
        "artifact_type": "model_artifact",
        "producing_step": CONTAINER_UPSTREAM_SOURCE_STEP,
        "state": "selected",
        "path": redacted_path,
        "uri": redacted_uri,
        "checksum": _artifact_checksum(project_path, path_value) if path_value else None,
        "metadata": _redacted_evidence(
            {
                "source": "structured_artifact_manifest",
                "source_run_id": source_run_id,
                "source_producing_step": source_entry.get("producing_step"),
            }
        ),
    }


def _resolve_training_lineage(
    data_stage: dict[str, Any],
    training_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(training_evidence, dict):
        return {
            "status": "missing",
            "evidence": {
                "status": "missing",
                "capability": "training_lineage_verified",
                "reason": "No structured training evidence artifact was found.",
            },
        }
    lineage_reference = training_evidence.get("data_stage_evidence")
    lineage_path = None
    if isinstance(lineage_reference, dict):
        lineage_path = lineage_reference.get("path")
    entries = training_evidence.get("artifact_manifest", {}).get("entries", ())
    if lineage_path is None and isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and entry.get("artifact_type") == "data_stage_evidence":
                lineage_path = entry.get("path")
                break
    lineage_ok = (
        lineage_path == DATA_STAGE_EVIDENCE_PATH
        and _training_verification_passed_any(
            training_evidence,
            ("training_command_completed", "bounded_training_completed", "mlflow_run_exists"),
        )
    )
    return {
        "status": "resolved" if lineage_ok else "blocked",
        "evidence": {
            "status": "resolved" if lineage_ok else "blocked",
            "training_evidence_path": training_evidence.get("_evidence_path"),
            "data_stage_evidence_path": lineage_path,
            "data_stage_status": data_stage["evidence"].get("status"),
            "reason": None
            if lineage_ok
            else "Training evidence is not tied to data_stage_evidence.json.",
        },
    }


def _training_verification_passed(training_evidence: dict[str, Any], check_name: str) -> bool:
    return _training_verification_passed_any(training_evidence, (check_name,))


def _training_verification_passed_any(
    training_evidence: dict[str, Any],
    check_names: tuple[str, ...],
) -> bool:
    results = training_evidence.get("verification_results", ())
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("check_name") in check_names and result.get("passed") is True:
            return True
    return False


def _container_upstream_artifact_entries(
    *,
    data_stage: dict[str, Any],
    local_model_artifact: dict[str, Any],
    mlflow_best_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for source in (data_stage, mlflow_best_artifact, local_model_artifact):
        entry = source.get("artifact_entry")
        if isinstance(entry, dict):
            entries.append({key: value for key, value in entry.items() if value is not None})
    return _dedupe_artifact_entries(entries)


def _container_upstream_verification_results(
    *,
    completion_mode: str,
    upstream_evidence: dict[str, Any],
    local_model_available: bool,
    mlflow_best_available: bool,
    data_capstone_complete: bool,
    training_lineage_available: bool,
    resolved: bool,
) -> list[dict[str, Any]]:
    results = [
        _container_upstream_verification_result(
            "upstream_evidence_resolved",
            resolved,
            upstream_evidence,
        )
    ]
    if completion_mode == "container_local_ready":
        if local_model_available:
            results.append(
                _container_upstream_verification_result(
                    "local_model_artifact_resolved",
                    True,
                    upstream_evidence["local_model_artifact"],
                )
            )
        elif not mlflow_best_available:
            results.append(
                _container_upstream_verification_result(
                    "local_model_artifact_resolved",
                    False,
                    upstream_evidence["local_model_artifact"],
                )
            )
        if mlflow_best_available:
            results.append(
                _container_upstream_verification_result(
                    "mlflow_best_artifact_resolved",
                    True,
                    upstream_evidence["mlflow_best_artifact"],
                )
            )
        elif not local_model_available:
            results.append(
                _container_upstream_verification_result(
                    "mlflow_best_artifact_resolved",
                    False,
                    upstream_evidence["mlflow_best_artifact"],
                )
            )
        return results
    results.extend(
        [
            _container_upstream_verification_result(
                "data_stage_capstone_complete_verified",
                data_capstone_complete,
                upstream_evidence["data_stage"],
            ),
            _container_upstream_verification_result(
                "mlflow_best_artifact_verified",
                mlflow_best_available,
                upstream_evidence["mlflow_best_artifact"],
            ),
            _container_upstream_verification_result(
                "training_lineage_verified",
                training_lineage_available,
                upstream_evidence["training_lineage"],
            ),
        ]
    )
    return results


def _container_upstream_verification_result(
    check_name: str,
    passed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "evidence_type": "observed",
        "source_step": CONTAINER_UPSTREAM_SOURCE_STEP,
        "passed": passed,
        "evidence": json.dumps(_redacted_evidence(evidence), sort_keys=True),
    }


def _container_upstream_capability(
    capability: str,
    reason: str,
    later_phase_pointer: str,
) -> dict[str, Any]:
    return {
        "stage": "container_ci",
        "capability": capability,
        "reason": reason,
        "later_phase_pointer": later_phase_pointer,
    }


def _container_upstream_next_actions(
    blocked_capabilities: list[dict[str, Any]],
    deferred_capabilities: list[dict[str, Any]],
) -> list[str]:
    next_actions = [
        item["reason"] for item in (*blocked_capabilities, *deferred_capabilities)
    ]
    next_actions.append(
        "Do not generate Dockerfiles, run Docker, configure registries, mutate secrets, "
        "or write CI workflows in Phase 5 Issue 2."
    )
    return next_actions


def _redact_path_if_sensitive(path: str) -> str:
    parts = Path(path).parts
    if any(
        any(marker in part.casefold() for marker in ("secret", "token", "password", "access_key"))
        for part in parts
    ):
        return "<redacted-path>"
    return path


def _redact_uri_if_sensitive(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        return _redact_remote_url(uri) or f"{parsed.scheme}://<redacted>"
    return _redact_path_if_sensitive(uri)


def record_capstone_orchestrator_skeleton(
    project_path: str,
    declared_stages: list[str] | tuple[str, ...] | None = None,
    implemented_subworkflows: list[str] | tuple[str, ...] | None = None,
    blocked_subworkflows: list[str] | tuple[str, ...] | None = None,
    selected_model_artifact_path: str | None = None,
    endpoint_url: str | None = None,
) -> dict[str, Any]:
    """Record the Capstone Orchestrator skeleton without faking full capstone success."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    declared = list(declared_stages or DEFAULT_CAPSTONE_STAGES)
    implemented = list(implemented_subworkflows or DEFAULT_CAPSTONE_IMPLEMENTED_SUBWORKFLOWS)
    blocked_workflows = list(blocked_subworkflows or DEFAULT_CAPSTONE_BLOCKED_SUBWORKFLOWS)
    data_stage = _load_capstone_data_stage_evidence(path)
    selected_model_artifact = (
        {"path": selected_model_artifact_path, "state": "available"}
        if selected_model_artifact_path
        else None
    )
    container_ci_stage = _load_capstone_container_ci_evidence(
        path,
        upstream_ready=bool(data_stage["completed"] and selected_model_artifact),
    )
    blocked_stages = [
        {
            "stage": "train",
            "capability": workflow_id,
            "reason": (
                f"Workflow '{workflow_id}' is not registered on this branch; the "
                "orchestrator must not claim iterative improvement success."
            ),
            "later_phase_pointer": "Future train-and-improve issue",
        }
        for workflow_id in blocked_workflows
    ]
    blocked_stages.extend(data_stage["blocked_capabilities"])
    blocked_stages.extend(container_ci_stage["blocked_capabilities"])
    deferred_capabilities = list(DEFAULT_CAPSTONE_DEFERRED_CAPABILITIES)
    deferred_capabilities.extend(container_ci_stage["deferred_capabilities"])
    implemented_records = [
        {
            "workflow_id": workflow_id,
            "status": "implemented",
            "completion_rule": "A stage may be complete only after this workflow's SuccessContract succeeds.",
        }
        for workflow_id in implemented
    ]
    stage_statuses = [
        {
            "stage": stage,
            "status": data_stage["stage_status"]
            if stage == "data"
            else container_ci_stage["stage_status"]
            if stage == "container_ci"
            else "available"
            if stage in {"setup", "train", "deploy"}
            else "deferred",
            "completion_rule": "contract-derived",
        }
        for stage in declared
    ]
    completed_stages = ["data"] if data_stage["completed"] else []
    if container_ci_stage["completed"]:
        completed_stages.append("container_ci")
    endpoint_evidence = (
        {"endpoint_url": endpoint_url, "state": "available"} if endpoint_url else None
    )
    next_actions = [
        "Run prepare_capstone_data and require its durable data-stage evidence before downstream capstone stages.",
        "Run setup_pipeline and require its SuccessContract to succeed before marking setup complete.",
        "Run train_and_track with explicit bounded controls and model-selection inputs.",
        "Add train_until_better before claiming iterative improvement stage completion.",
        "Provide selected model artifact evidence before executing LitServe deployment workflows.",
        "Replace each deferred capability with a registry-owned workflow in a later issue.",
    ]

    plan_path = path / ".auto_mlops" / "capstone" / "orchestrator_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_entry = {
        "artifact_type": "capstone_orchestrator_plan",
        "producing_step": "record_capstone_orchestrator_skeleton",
        "state": "generated",
        "path": relative_to_project(str(path), plan_path),
        "metadata": {
            "declared_stages": declared,
            "status": "blocked",
            "deferred_capability_count": len(DEFAULT_CAPSTONE_DEFERRED_CAPABILITIES),
        },
    }
    artifact_entries = [artifact_entry]
    if data_stage["artifact_entry"] is not None:
        artifact_entries.append(data_stage["artifact_entry"])
    if container_ci_stage["artifact_entry"] is not None:
        artifact_entries.append(container_ci_stage["artifact_entry"])
    plan = {
        "workflow_id": "build_capstone_pipeline",
        "name": "Capstone Orchestrator",
        "status": "blocked",
        "declared_stages": declared,
        "stage_statuses": stage_statuses,
        "completed_stages": completed_stages,
        "blocked_stages": blocked_stages,
        "data_stage": {
            key: value
            for key, value in data_stage.items()
            if key not in {"artifact_entry", "blocked_capabilities"}
        },
        "container_ci_stage": {
            key: value
            for key, value in container_ci_stage.items()
            if key
            not in {
                "artifact_entry",
                "blocked_capabilities",
                "deferred_capabilities",
            }
        },
        "deferred_capabilities": deferred_capabilities,
        "deferred_stages": deferred_capabilities,
        "implemented_subworkflows": implemented_records,
        "selected_model_artifact": selected_model_artifact,
        "endpoint_evidence": endpoint_evidence,
        "artifact_manifest": {"entries": artifact_entries},
        "next_actions": next_actions,
    }
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True))

    verification_results = [
        {
            "check_name": "capstone_stage_plan_recorded",
            "evidence_type": "declared",
            "source_step": "record_capstone_orchestrator_skeleton",
            "passed": True,
            "evidence": json.dumps(
                {"declared_stages": declared, "stage_statuses": stage_statuses},
                sort_keys=True,
            ),
        },
        {
            "check_name": "implemented_subworkflows_referenced",
            "evidence_type": "declared",
            "source_step": "record_capstone_orchestrator_skeleton",
            "passed": True,
            "evidence": json.dumps(
                {
                    "implemented_subworkflows": implemented,
                    "blocked_subworkflows": blocked_workflows,
                },
                sort_keys=True,
            ),
        },
        {
            "check_name": "deferred_capabilities_recorded",
            "evidence_type": "declared",
            "source_step": "record_capstone_orchestrator_skeleton",
            "passed": True,
            "evidence": json.dumps(list(DEFAULT_CAPSTONE_DEFERRED_CAPABILITIES), sort_keys=True),
        },
    ]

    return {
        "success": True,
        **plan,
        "verification_results": verification_results,
        "message": (
            "Capstone Orchestrator skeleton recorded; full capstone remains blocked until "
            "future capabilities have registry-owned workflows and contract evidence."
        ),
    }


def record_litserve_image_build_skipped(project_path: str) -> dict[str, Any]:
    """Record that Docker image build is optional and skipped by default."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    return {
        "success": True,
        "image_build": "skipped",
        "message": "Docker image build is optional for this runtime slice and was not run.",
    }


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def _find_available_port(host: str, requested_port: int, attempts: int = 25) -> int:
    for candidate_port in range(requested_port, requested_port + attempts):
        if _is_port_available(host, candidate_port):
            return candidate_port
    raise RuntimeError(
        f"No available port found from {requested_port} to {requested_port + attempts - 1}"
    )


def start_litserve_server(
    project_path: str,
    server_path: str = "deployment/litserve/server.py",
    port: int = 8000,
    host: str = "127.0.0.1",
    log_path: str = "deployment/litserve/server.log",
    startup_wait_seconds: float = 2.0,
) -> dict[str, Any]:
    """Start LitServe server and record observed process evidence."""
    path = Path(project_path)
    server_file = path / server_path
    if not server_file.exists():
        return {"success": False, "error": f"LitServe server not found: {server_path}"}

    log_file = path / log_path
    ensure_directory(log_file.parent)
    try:
        actual_port = _find_available_port(host, port)
    except RuntimeError as exc:
        return {"success": False, "error": str(exc)}

    command = [sys.executable, str(server_file)]
    env = os.environ.copy()
    env["LITSERVE_PORT"] = str(actual_port)
    log_handle = open(log_file, "a")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(path),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    time.sleep(startup_wait_seconds)
    running = process.poll() is None
    endpoint_url = f"http://{host}:{actual_port}"
    evidence = {
        "pid": process.pid,
        "running": running,
        "returncode": process.returncode,
        "command": " ".join(command),
        "endpoint_url": endpoint_url,
        "requested_port": port,
        "port": actual_port,
        "log_path": relative_to_project(project_path, log_file),
    }
    return {
        "success": True,
        "process_id": process.pid,
        "endpoint_url": endpoint_url,
        "port": actual_port,
        "server_start_command": " ".join(command),
        "log_path": relative_to_project(project_path, log_file),
        "verification_results": [
            {
                "check_name": "server_start_command_recorded",
                "evidence_type": "observed",
                "source_step": "start_litserve_server",
                "passed": running,
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        ],
        "message": "LitServe server start attempted and process evidence recorded.",
    }


def test_litserve_health_endpoint(
    project_path: str,
    endpoint_url: str = "http://127.0.0.1:8000",
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Call /health and record observed HTTP evidence."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    url = endpoint_url.rstrip("/") + "/health"
    evidence: dict[str, Any] = {"url": url, "attempts": 0}
    passed = False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        evidence["attempts"] += 1
        try:
            with urllib.request.urlopen(url, timeout=min(2.0, timeout_seconds)) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                evidence.update({"status_code": response.status, "body": body})
                passed = 200 <= response.status < 300
                if passed:
                    break
        except urllib.error.URLError as exc:
            evidence["error"] = str(exc)
        if time.monotonic() < deadline:
            time.sleep(0.5)

    return {
        "success": True,
        "health_passed": passed,
        "verification_results": [
            {
                "check_name": "health_result_recorded",
                "evidence_type": "observed",
                "source_step": "test_health_endpoint",
                "passed": passed,
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        ],
        "message": "LitServe /health check recorded.",
    }


def test_litserve_prediction_endpoint(
    project_path: str,
    endpoint_url: str = "http://127.0.0.1:8000",
    sample_input: dict[str, Any] | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Call /predict and record observed HTTP evidence."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    url = endpoint_url.rstrip("/") + "/predict"
    request_payload = (
        _default_litserve_prediction_payload(project_path)
        if sample_input in (None, {"input": [0.0]})
        else sample_input
    )
    payload = json.dumps(request_payload).encode("utf-8")
    evidence: dict[str, Any] = {"url": url, "request": request_payload, "attempts": 0}
    passed = False
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() <= deadline:
        evidence["attempts"] += 1
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(2.0, timeout_seconds)) as response:
                body = response.read(4096).decode("utf-8", errors="replace")
                evidence.update({"status_code": response.status, "body": body})
                passed = 200 <= response.status < 300
                if passed:
                    break
        except urllib.error.URLError as exc:
            evidence["error"] = str(exc)
        if time.monotonic() < deadline:
            time.sleep(0.5)

    return {
        "success": True,
        "prediction_passed": passed,
        "verification_results": [
            {
                "check_name": "prediction_result_recorded",
                "evidence_type": "observed",
                "source_step": "test_prediction_endpoint",
                "passed": passed,
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        ],
        "message": "LitServe /predict check recorded.",
    }


def capture_litserve_logs_and_endpoint(
    project_path: str,
    endpoint_url: str = "http://127.0.0.1:8000",
    log_path: str = "deployment/litserve/server.log",
) -> dict[str, Any]:
    """Record endpoint URL and server log location as observed deployment evidence."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    log_file = path / log_path
    evidence = {
        "endpoint_url": endpoint_url,
        "log_path": relative_to_project(project_path, log_file),
        "log_exists": log_file.exists(),
    }
    return {
        "success": True,
        "endpoint_url": endpoint_url,
        "log_path": relative_to_project(project_path, log_file),
        "verification_results": [
            {
                "check_name": "endpoint_url_recorded",
                "evidence_type": "observed",
                "source_step": "capture_logs_and_endpoint",
                "passed": bool(endpoint_url),
                "evidence": json.dumps(evidence, sort_keys=True),
            }
        ],
        "message": "LitServe endpoint URL and log path recorded.",
    }


def record_litserve_gpu_rollback_readiness(
    project_path: str,
    process_id: int | None = None,
    port: int = 8000,
) -> dict[str, Any]:
    """Record process cleanup and manual Lambda Cloud stop instructions."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    command = f"kill {process_id}" if process_id else f"lsof -ti:{port} | xargs -r kill"
    manual_instruction = (
        "Stop the user-started Lambda Cloud instance manually from the Lambda Cloud console "
        "when deployment testing is complete."
    )
    return {
        "success": True,
        "rollback_plan": {
            "command": command,
            "documented_target": manual_instruction,
        },
        "message": "LitServe process cleanup and manual Lambda stop instruction recorded.",
    }


def configure_litserver(
    project_path: str,
    max_batch_size: int = 64,
    batch_timeout: float = 0.05,
    workers_per_device: int = 4,
    accelerator: str = "auto",
    port: int = 8000,
) -> dict[str, Any]:
    """Configure LitServe server settings."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "litserve"
    server_path = deploy_dir / "server.py"

    if not server_path.exists():
        return {
            "success": False,
            "error": "LitServe server.py not found. Run create_litserve_api first.",
        }

    # Read and update server configuration
    content = server_path.read_text()

    # Update server configuration
    old_config = 'accelerator="auto"'
    new_config = f'accelerator="{accelerator}"'
    content = content.replace(old_config, new_config)

    content = content.replace("max_batch_size=64", f"max_batch_size={max_batch_size}")
    content = content.replace("batch_timeout=0.05", f"batch_timeout={batch_timeout}")
    content = content.replace("workers_per_device=4", f"workers_per_device={workers_per_device}")
    content = content.replace("port=8000", f"port={port}")
    content = content.replace("DEFAULT_PORT = 8000", f"DEFAULT_PORT = {port}")

    with open(server_path, "w") as f:
        f.write(content)

    return {
        "success": True,
        "server_path": str(server_path),
        "config": {
            "max_batch_size": max_batch_size,
            "batch_timeout": batch_timeout,
            "workers_per_device": workers_per_device,
            "accelerator": accelerator,
            "port": port,
        },
        "message": "LitServe configuration updated",
    }


# Gradio Tools
def create_gradio_interface(
    project_path: str,
    model_path: str,
    model_name: str,
    interface_type: str = "image_classifier",
    title: str = "ML Model Demo",
    description: str | None = None,
    examples: list[str] | None = None,
    share: bool = False,
) -> dict[str, Any]:
    """Create Gradio interface for model demo."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "gradio")
    class_name = "".join(word.capitalize() for word in model_name.replace("-", "_").split("_"))

    # Configure based on interface type
    if interface_type == "image_classifier":
        inputs = 'gr.Image(type="pil")'
        outputs = "gr.Label(num_top_classes=5)"
        predict_code = """image = self.transform(image).unsqueeze(0).to(self.device)
        output = self.model(image)
        probs = torch.softmax(output, dim=1)[0]
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}"""
        setup_code = """from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.labels = ["class_0", "class_1"]  # Update with actual labels"""
        input_params = "image"
    elif interface_type == "text_classifier":
        inputs = 'gr.Textbox(lines=3, placeholder="Enter text...")'
        outputs = "gr.Label(num_top_classes=5)"
        predict_code = """# Tokenize and predict
        return {"positive": 0.8, "negative": 0.2}  # Update with actual prediction"""
        setup_code = "# Add tokenizer setup here"
        input_params = "text"
    else:
        inputs = "gr.Textbox()"
        outputs = "gr.JSON()"
        predict_code = "return {'result': 'prediction'}"
        setup_code = "pass"
        input_params = "input_data"

    app_code = f'''"""Gradio Interface for {model_name}

Auto-generated by MLOps Agent
"""
import gradio as gr
import torch
from PIL import Image
from typing import Any


class {class_name}:
    """Model wrapper for Gradio interface"""

    def __init__(self, model_path: str = "{model_path}"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        {setup_code}

    @torch.no_grad()
    def predict(self, {input_params}) -> Any:
        """Run prediction."""
        {predict_code}


# Create model instance
model = {class_name}()

# Create Gradio interface
demo = gr.Interface(
    fn=model.predict,
    inputs={inputs},
    outputs={outputs},
    title="{title}",
    description="{description or f'Demo for {model_name}'}",
    examples={examples or []},
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share={share})
'''

    app_path = deploy_dir / "app.py"
    with open(app_path, "w") as f:
        f.write(app_code)

    # Requirements
    requirements = """gradio>=4.0.0
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
"""
    req_path = deploy_dir / "requirements.txt"
    with open(req_path, "w") as f:
        f.write(requirements)

    return {
        "success": True,
        "app_path": str(app_path),
        "requirements_path": str(req_path),
        "interface_type": interface_type,
        "message": f"Gradio interface created at {deploy_dir}",
    }


def deploy_to_huggingface(
    project_path: str, space_name: str, hf_token: str | None = None, private: bool = False
) -> dict[str, Any]:
    """Deploy Gradio app to Hugging Face Spaces."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "gradio"

    if not (deploy_dir / "app.py").exists():
        return {
            "success": False,
            "error": "Gradio app.py not found. Run create_gradio_interface first.",
        }

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        return {
            "success": False,
            "error": "HF_TOKEN not provided. Set environment variable or pass hf_token.",
        }

    # Create README for HF Spaces
    readme_content = f"""---
title: {space_name}
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# {space_name}

Auto-generated by MLOps Agent.
"""

    readme_path = deploy_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    # Create deployment script
    deploy_script = f"""#!/bin/bash
# Deploy to Hugging Face Spaces
# Run: bash deploy_hf.sh

cd {deploy_dir}

# Initialize git if needed
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# Add HF remote
git remote remove hf 2>/dev/null || true
git remote add hf https://huggingface.co/spaces/$HF_USERNAME/{space_name}

# Add and commit
git add .
git commit -m "Deploy to HF Spaces"

# Push
git push hf main --force

echo "Deployed to: https://huggingface.co/spaces/$HF_USERNAME/{space_name}"
"""

    script_path = deploy_dir / "deploy_hf.sh"
    with open(script_path, "w") as f:
        f.write(deploy_script)

    return {
        "success": True,
        "space_name": space_name,
        "deploy_dir": str(deploy_dir),
        "deploy_script": str(script_path),
        "readme_path": str(readme_path),
        "message": f"HF Spaces deployment prepared. Run: bash {script_path}",
    }


# FastAPI + Lambda Tools
def create_fastapi_app(
    project_path: str,
    model_path: str,
    model_name: str,
    endpoint_type: str = "image",
    title: str = "ML Inference API",
) -> dict[str, Any]:
    """Create FastAPI application for model serving."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "fastapi_lambda")

    # Configure based on endpoint type
    if endpoint_type == "image":
        endpoint_code = """@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    \"\"\"Run prediction on uploaded image.\"\"\"
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            confidence, predicted = torch.max(probs, 0)

        return {
            "class": int(predicted),
            "confidence": float(confidence),
            "probabilities": probs.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""
        setup_code = """from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])"""
    else:
        endpoint_code = """@app.post("/predict")
async def predict(data: dict):
    \"\"\"Run prediction on input data.\"\"\"
    try:
        # Process input
        with torch.no_grad():
            output = model(data)
        return {"result": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""
        setup_code = "# Add custom preprocessing here"

    app_code = f'''"""FastAPI Application for {model_name} - Lambda Ready

Auto-generated by MLOps Agent
"""
import io
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="{title}", description="Inference API for {model_name}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device("cpu")  # Lambda is CPU-only
model = torch.jit.load("{model_path}")
model.to(device)
model.eval()

{setup_code}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {{"status": "healthy", "model": "{model_name}"}}


{endpoint_code}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    app_path = deploy_dir / "app.py"
    with open(app_path, "w") as f:
        f.write(app_code)

    # Requirements
    requirements = """fastapi>=0.109.0
uvicorn[standard]>=0.27.0
mangum>=0.17.0
torch>=2.1.0
torchvision>=0.16.0
Pillow>=10.0.0
python-multipart>=0.0.6
"""
    req_path = deploy_dir / "requirements.txt"
    with open(req_path, "w") as f:
        f.write(requirements)

    return {
        "success": True,
        "app_path": str(app_path),
        "requirements_path": str(req_path),
        "endpoint_type": endpoint_type,
        "message": f"FastAPI app created at {deploy_dir}",
    }


def create_lambda_dockerfile(
    project_path: str, python_version: str = "3.11", model_file: str = "model.pt", port: int = 8080
) -> dict[str, Any]:
    """Create Dockerfile for AWS Lambda deployment."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "fastapi_lambda"

    if not deploy_dir.exists():
        return {"success": False, "error": "FastAPI deployment directory not found."}

    dockerfile_content = f"""# Lambda-ready Dockerfile
# Auto-generated by MLOps Agent

FROM public.ecr.aws/docker/library/python:{python_version}-slim

# Copy Lambda Web Adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT={port}
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /var/task

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py ./
COPY {model_file} ./

CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app
"""

    dockerfile_path = deploy_dir / "Dockerfile"
    with open(dockerfile_path, "w") as f:
        f.write(dockerfile_content)

    # Create .dockerignore
    dockerignore = """__pycache__/
*.pyc
.git/
.env
*.log
"""
    with open(deploy_dir / ".dockerignore", "w") as f:
        f.write(dockerignore)

    return {
        "success": True,
        "dockerfile_path": str(dockerfile_path),
        "python_version": python_version,
        "port": port,
        "message": f"Lambda Dockerfile created at {dockerfile_path}",
    }


def generate_cdk_stack(
    project_path: str,
    stack_name: str,
    model_name: str,
    memory_size: int = 1024,
    timeout: int = 30,
    stage: str = "prod",
) -> dict[str, Any]:
    """Generate AWS CDK stack for Lambda deployment."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "fastapi_lambda"

    if not deploy_dir.exists():
        return {"success": False, "error": "FastAPI deployment directory not found."}

    stack_class = "".join(word.capitalize() for word in stack_name.replace("-", "_").split("_"))

    cdk_code = f'''"""AWS CDK Stack for {model_name} Lambda Deployment

Auto-generated by MLOps Agent
Deploy with: cdk deploy
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_ecr_assets as ecr_assets,
    aws_apigateway as apigw,
    aws_logs as logs,
    CfnOutput,
)
from constructs import Construct


class {stack_class}Stack(Stack):
    """CDK Stack for {model_name}"""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Build Docker image
        docker_image = ecr_assets.DockerImageAsset(
            self, "{model_name}Image",
            directory=".",
        )

        # Lambda function
        inference_function = lambda_.DockerImageFunction(
            self, "{model_name}Function",
            code=lambda_.DockerImageCode.from_ecr(
                repository=docker_image.repository,
                tag_or_digest=docker_image.asset_hash,
            ),
            memory_size={memory_size},
            timeout=Duration.seconds({timeout}),
            environment={{"MODEL_NAME": "{model_name}"}},
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # API Gateway
        api = apigw.LambdaRestApi(
            self, "{model_name}Api",
            handler=inference_function,
            proxy=True,
            deploy_options=apigw.StageOptions(stage_name="{stage}"),
        )

        CfnOutput(self, "ApiEndpoint", value=api.url)
        CfnOutput(self, "FunctionArn", value=inference_function.function_arn)
'''

    cdk_path = deploy_dir / "cdk_stack.py"
    with open(cdk_path, "w") as f:
        f.write(cdk_code)

    # Create cdk.json
    cdk_json = {
        "app": "python3 cdk_stack.py",
        "context": {"@aws-cdk/core:stackRelativeExports": True},
    }
    with open(deploy_dir / "cdk.json", "w") as f:
        json.dump(cdk_json, f, indent=2)

    return {
        "success": True,
        "cdk_stack_path": str(cdk_path),
        "stack_name": stack_name,
        "memory_size": memory_size,
        "timeout": timeout,
        "message": f"CDK stack created. Deploy with: cd {deploy_dir} && cdk deploy",
    }


# TorchServe Tools
def create_torchserve_handler(
    project_path: str, model_path: str, model_name: str, handler_type: str = "image_classifier"
) -> dict[str, Any]:
    """Create TorchServe custom handler."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "torchserve")
    class_name = "".join(word.capitalize() for word in model_name.replace("-", "_").split("_"))
    model_file = Path(model_path).name

    # Configure based on handler type
    if handler_type == "image_classifier":
        init_code = """from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])"""
        preprocess_code = """images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert("RGB")
            images.append(self.transform(image))
        return torch.stack(images).to(self.device)"""
        postprocess_code = """probs = torch.softmax(inference_output, dim=1)
        results = []
        for prob in probs:
            conf, pred = torch.max(prob, 0)
            results.append({"class": int(pred), "confidence": float(conf)})
        return results"""
    else:
        init_code = "pass"
        preprocess_code = "return data"
        postprocess_code = "return inference_output.tolist()"

    handler_code = f'''"""TorchServe Handler for {model_name}

Auto-generated by MLOps Agent
"""
import torch
import io
import logging
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class {class_name}Handler(BaseHandler):
    """TorchServe handler for {model_name}"""

    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context) -> None:
        """Initialize model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        self.model = torch.jit.load(f"{{model_dir}}/{model_file}")
        self.model.to(self.device)
        self.model.eval()

        {init_code}

        self.initialized = True
        logger.info(f"Model loaded on {{self.device}}")

    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        """Preprocess input data."""
        {preprocess_code}

    def inference(self, data: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Run inference."""
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, inference_output: torch.Tensor) -> List[Dict[str, Any]]:
        """Postprocess output."""
        {postprocess_code}
'''

    handler_path = deploy_dir / "handler.py"
    with open(handler_path, "w") as f:
        f.write(handler_code)

    return {
        "success": True,
        "handler_path": str(handler_path),
        "handler_type": handler_type,
        "class_name": f"{class_name}Handler",
        "message": f"TorchServe handler created at {handler_path}",
    }


def create_mar_archive(
    project_path: str,
    model_name: str,
    model_file: str,
    handler_file: str = "handler.py",
    version: str = "1.0",
    extra_files: list[str] | None = None,
) -> dict[str, Any]:
    """Create TorchServe MAR (Model Archive) file."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "torchserve"

    if not deploy_dir.exists():
        return {"success": False, "error": "TorchServe deployment directory not found."}

    # Create MAR build script
    extra_files_arg = ""
    if extra_files:
        extra_files_arg = f'--extra-files "{",".join(extra_files)}"'

    script_content = f"""#!/bin/bash
# Create MAR archive for {model_name}
# Auto-generated by MLOps Agent

set -e

MODEL_NAME="{model_name}"
VERSION="{version}"
MODEL_FILE="{model_file}"
HANDLER="{handler_file}"
EXPORT_PATH="./model-store"

mkdir -p $EXPORT_PATH

echo "Creating MAR archive..."
torch-model-archiver \\
    --model-name "$MODEL_NAME" \\
    --version "$VERSION" \\
    --serialized-file "$MODEL_FILE" \\
    --handler "$HANDLER" \\
    --export-path "$EXPORT_PATH" \\
    {extra_files_arg} \\
    --force

echo "MAR created: $EXPORT_PATH/$MODEL_NAME.mar"
echo "Run: torchserve --start --model-store $EXPORT_PATH --models $MODEL_NAME=$MODEL_NAME.mar"
"""

    script_path = deploy_dir / "create_mar.sh"
    with open(script_path, "w") as f:
        f.write(script_content)

    # Make executable
    os.chmod(script_path, 0o755)

    return {
        "success": True,
        "script_path": str(script_path),
        "model_name": model_name,
        "version": version,
        "message": f"MAR build script created. Run: bash {script_path}",
    }


def generate_torchserve_config(
    project_path: str,
    model_name: str,
    inference_port: int = 8080,
    management_port: int = 8081,
    metrics_port: int = 8082,
    workers: int = 1,
) -> dict[str, Any]:
    """Generate TorchServe configuration."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "torchserve"

    if not deploy_dir.exists():
        ensure_directory(deploy_dir)

    config_content = f"""# TorchServe Configuration for {model_name}
# Auto-generated by MLOps Agent

inference_address=http://0.0.0.0:{inference_port}
management_address=http://0.0.0.0:{management_port}
metrics_address=http://0.0.0.0:{metrics_port}

number_of_netty_threads=4
job_queue_size=100
model_store=./model-store
load_models={model_name}

default_workers_per_model={workers}
max_request_size=6553500

async_logging=true
"""

    config_path = deploy_dir / "config.properties"
    with open(config_path, "w") as f:
        f.write(config_content)

    return {
        "success": True,
        "config_path": str(config_path),
        "ports": {
            "inference": inference_port,
            "management": management_port,
            "metrics": metrics_port,
        },
        "message": f"TorchServe config created at {config_path}",
    }


# KServe Tools
def create_inference_service_yaml(
    project_path: str,
    service_name: str,
    model_name: str,
    storage_uri: str,
    namespace: str = "default",
    runtime: str = "pytorch",
    min_replicas: int = 1,
    max_replicas: int = 5,
) -> dict[str, Any]:
    """Create KServe InferenceService YAML."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "kserve")

    # Map runtime to KServe predictor
    runtime_map = {
        "pytorch": "pytorch",
        "tensorflow": "tensorflow",
        "sklearn": "sklearn",
        "xgboost": "xgboost",
        "onnx": "onnx",
    }
    predictor_runtime = runtime_map.get(runtime, "pytorch")

    yaml_content = f"""# KServe InferenceService for {model_name}
# Auto-generated by MLOps Agent

apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {service_name}
  namespace: {namespace}
  labels:
    app: {service_name}
spec:
  predictor:
    minReplicas: {min_replicas}
    maxReplicas: {max_replicas}
    {predictor_runtime}:
      storageUri: "{storage_uri}"
      resources:
        limits:
          cpu: "1"
          memory: "2Gi"
        requests:
          cpu: "100m"
          memory: "256Mi"
"""

    yaml_path = deploy_dir / "inference_service.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return {
        "success": True,
        "yaml_path": str(yaml_path),
        "service_name": service_name,
        "namespace": namespace,
        "runtime": predictor_runtime,
        "message": f"KServe InferenceService YAML created. Apply with: kubectl apply -f {yaml_path}",
    }


def generate_kserve_config(
    project_path: str,
    service_name: str,
    min_replicas: int = 1,
    max_replicas: int = 5,
    target_utilization: int = 80,
    gpu_enabled: bool = False,
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Generate KServe configuration."""
    path = Path(project_path)
    deploy_dir = path / "deployment" / "kserve"

    if not deploy_dir.exists():
        ensure_directory(deploy_dir)

    gpu_config = ""
    if gpu_enabled:
        gpu_config = f"""
gpu:
  enabled: true
  count: {gpu_count}
  type: "nvidia.com/gpu"
"""

    config_content = f"""# KServe Configuration for {service_name}
# Auto-generated by MLOps Agent

service:
  name: {service_name}

scaling:
  minReplicas: {min_replicas}
  maxReplicas: {max_replicas}
  targetUtilization: {target_utilization}
  scaleToZero: true
  scaleToZeroGracePeriod: "30s"

resources:
  cpu:
    request: "100m"
    limit: "1"
  memory:
    request: "256Mi"
    limit: "2Gi"
{gpu_config}
"""

    config_path = deploy_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(config_content)

    return {
        "success": True,
        "config_path": str(config_path),
        "scaling": {
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "target_utilization": target_utilization,
        },
        "gpu_enabled": gpu_enabled,
        "message": f"KServe config created at {config_path}",
    }


# --- Kubernetes Manifest Tools ---


def create_k8s_deployment_yaml(
    project_path: str,
    name: str,
    image: str,
    replicas: int = 1,
    container_port: int = 8000,
    namespace: str = "default",
    labels: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create Kubernetes Deployment YAML."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "k8s")
    labels = labels or {"app": name}
    env_list = [{"name": k, "value": str(v)} for k, v in (env or {}).items()]

    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name, "namespace": namespace, "labels": labels},
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": name,
                            "image": image,
                            "ports": [{"containerPort": container_port}],
                            "env": env_list,
                        }
                    ]
                },
            },
        },
    }

    if resources:
        deployment["spec"]["template"]["spec"]["containers"][0]["resources"] = resources

    yaml_path = deploy_dir / "deployment.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(deployment, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "deployment_path": str(yaml_path),
        "message": f"Deployment YAML created at {yaml_path}",
    }


def create_k8s_service_yaml(
    project_path: str,
    name: str,
    selector: dict[str, str] | None = None,
    port: int = 80,
    target_port: int = 8000,
    service_type: str = "ClusterIP",
    namespace: str = "default",
) -> dict[str, Any]:
    """Create Kubernetes Service YAML."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    deploy_dir = ensure_directory(path / "deployment" / "k8s")
    selector = selector or {"app": name}
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "type": service_type,
            "selector": selector,
            "ports": [{"port": port, "targetPort": target_port}],
        },
    }

    yaml_path = deploy_dir / "service.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(service, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "service_path": str(yaml_path),
        "message": f"Service YAML created at {yaml_path}",
    }


def create_k8s_ingress_yaml(
    project_path: str,
    name: str,
    host: str,
    service_name: str,
    service_port: int = 80,
    path: str = "/",
    namespace: str = "default",
    ingress_class: str = "alb",
    alb_scheme: str = "internet-facing",
    certificate_arn: str | None = None,
    annotations: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create Kubernetes Ingress YAML (ALB annotations for EKS)."""
    base_annotations = {
        "kubernetes.io/ingress.class": ingress_class,
        "alb.ingress.kubernetes.io/scheme": alb_scheme,
        "alb.ingress.kubernetes.io/target-type": "ip",
    }
    if certificate_arn:
        base_annotations["alb.ingress.kubernetes.io/certificate-arn"] = certificate_arn
        base_annotations["alb.ingress.kubernetes.io/listen-ports"] = '[{"HTTPS":443}]'
    if annotations:
        base_annotations.update(annotations)

    ingress = {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "Ingress",
        "metadata": {"name": name, "namespace": namespace, "annotations": base_annotations},
        "spec": {
            "rules": [
                {
                    "host": host,
                    "http": {
                        "paths": [
                            {
                                "path": path,
                                "pathType": "Prefix",
                                "backend": {
                                    "service": {
                                        "name": service_name,
                                        "port": {"number": service_port},
                                    }
                                },
                            }
                        ]
                    },
                }
            ]
        },
    }

    deploy_dir = ensure_directory(Path(project_path) / "deployment" / "k8s")
    yaml_path = deploy_dir / "ingress.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(ingress, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "ingress_path": str(yaml_path),
        "message": f"Ingress YAML created at {yaml_path}",
    }


def create_k8s_hpa_yaml(
    project_path: str,
    name: str,
    deployment_name: str,
    min_replicas: int = 1,
    max_replicas: int = 3,
    target_cpu_utilization: int = 70,
    namespace: str = "default",
) -> dict[str, Any]:
    """Create Kubernetes HPA YAML."""
    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": deployment_name,
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": target_cpu_utilization,
                        },
                    },
                }
            ],
        },
    }

    deploy_dir = ensure_directory(Path(project_path) / "deployment" / "k8s")
    yaml_path = deploy_dir / "hpa.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(hpa, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "hpa_path": str(yaml_path),
        "message": f"HPA YAML created at {yaml_path}",
    }


def create_k8s_configmap_yaml(
    project_path: str,
    name: str,
    data: dict[str, str],
    namespace: str = "default",
) -> dict[str, Any]:
    """Create Kubernetes ConfigMap YAML."""
    configmap = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": name, "namespace": namespace},
        "data": {k: str(v) for k, v in data.items()},
    }

    deploy_dir = ensure_directory(Path(project_path) / "deployment" / "k8s")
    yaml_path = deploy_dir / "configmap.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(configmap, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "configmap_path": str(yaml_path),
        "message": f"ConfigMap YAML created at {yaml_path}",
    }


def create_k8s_secret_yaml(
    project_path: str,
    name: str,
    data: dict[str, str],
    namespace: str = "default",
    encode: bool = True,
) -> dict[str, Any]:
    """Create Kubernetes Secret YAML."""
    secret = {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": name, "namespace": namespace},
        "type": "Opaque",
    }

    if encode:
        secret["data"] = {
            k: base64.b64encode(str(v).encode("utf-8")).decode("utf-8")
            for k, v in data.items()
        }
    else:
        secret["stringData"] = {k: str(v) for k, v in data.items()}

    deploy_dir = ensure_directory(Path(project_path) / "deployment" / "k8s")
    yaml_path = deploy_dir / "secret.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(secret, f, default_flow_style=False, sort_keys=False)

    return {
        "success": True,
        "secret_path": str(yaml_path),
        "message": f"Secret YAML created at {yaml_path}",
    }


def generate_rollback_plan(
    project_path: str,
    target: str,
    deployment_name: str | None = None,
    namespace: str = "default",
    error: str | None = None,
) -> dict[str, Any]:
    """Generate a rollback plan with suggested commands."""
    path = Path(project_path)
    rollback_dir = ensure_directory(path / "deployment" / "rollback")
    plan_path = rollback_dir / f"rollback_{target}.md"

    commands = []
    if target in {"kserve", "k8s", "kubernetes"}:
        name = deployment_name or "your-deployment"
        commands.append(f"kubectl rollout undo deployment/{name} -n {namespace}")
    elif target in {"lambda", "fastapi_lambda"}:
        commands.append("cd deployment/fastapi_lambda && cdk destroy")
    elif target in {"docker"}:
        commands.append("docker ps")
        commands.append("docker stop <container_id> && docker rm <container_id>")
    else:
        commands.append("Review deployment artifacts and undo changes manually.")

    content = [
        f"# Rollback Plan ({target})",
        "",
        f"Error: {error or 'N/A'}",
        "",
        "## Suggested Commands",
        "",
    ]
    content.extend([f"- `{cmd}`" for cmd in commands])
    plan_path.write_text("\n".join(content))

    return {
        "success": True,
        "rollback_plan_path": str(plan_path),
        "commands": commands,
        "message": f"Rollback plan created at {plan_path}",
    }


# --- AWS Automation Tools ---


def _ensure_boto3() -> dict[str, Any] | None:
    if not BOTO3_AVAILABLE:
        return {"success": False, "error": "boto3 not installed. Add boto3 to dependencies."}
    return None


def list_eks_clusters(region: str | None = None) -> dict[str, Any]:
    """List EKS clusters in a region."""
    missing = _ensure_boto3()
    if missing:
        return missing
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    try:
        client = boto3.client("eks", region_name=region)
        clusters = client.list_clusters().get("clusters", [])
        return {"success": True, "region": region, "clusters": clusters}
    except Exception as e:
        return {"success": False, "error": str(e)}


def update_kubeconfig(cluster_name: str, region: str | None = None) -> dict[str, Any]:
    """Update kubeconfig for an EKS cluster using AWS CLI."""
    if not check_tool_installed("aws"):
        return {"success": False, "error": "AWS CLI not installed. Install awscli."}
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    cmd = ["aws", "eks", "update-kubeconfig", "--name", cluster_name, "--region", region]
    result = run_command(cmd)
    if result["success"]:
        return {
            "success": True,
            "message": f"kubeconfig updated for {cluster_name} in {region}",
            "output": result.get("stdout", ""),
        }
    return result


def create_ecr_repo(
    repo_name: str,
    region: str | None = None,
    scan_on_push: bool = True,
    mutable_tags: bool = True,
) -> dict[str, Any]:
    """Create or get an ECR repository."""
    missing = _ensure_boto3()
    if missing:
        return missing
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("ecr", region_name=region)
    try:
        response = client.create_repository(
            repositoryName=repo_name,
            imageScanningConfiguration={"scanOnPush": scan_on_push},
            imageTagMutability="MUTABLE" if mutable_tags else "IMMUTABLE",
        )
    except ClientError as e:
        if getattr(e, "response", {}).get("Error", {}).get("Code") == "RepositoryAlreadyExistsException":
            response = client.describe_repositories(repositoryNames=[repo_name])
        else:
            return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}

    repo = response.get("repository", response.get("repositories", [{}])[0])
    return {
        "success": True,
        "repository": repo,
        "repository_uri": repo.get("repositoryUri"),
        "message": f"ECR repository ready: {repo.get('repositoryUri')}",
    }


def get_ecr_login(region: str | None = None) -> dict[str, Any]:
    """Get ECR login command."""
    missing = _ensure_boto3()
    if missing:
        return missing
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    try:
        client = boto3.client("ecr", region_name=region)
        token = client.get_authorization_token()
        auth_data = token.get("authorizationData", [{}])[0]
        auth_token = auth_data.get("authorizationToken", "")
        proxy_endpoint = auth_data.get("proxyEndpoint", "")
        user_pass = base64.b64decode(auth_token).decode("utf-8") if auth_token else ":"
        username, password = user_pass.split(":", 1)
        login_cmd = f"docker login -u {username} -p {password} {proxy_endpoint}"
        return {"success": True, "login_command": login_cmd, "registry": proxy_endpoint}
    except Exception as e:
        return {"success": False, "error": str(e)}


def generate_iam_policy(
    policy_name: str = "mlops-agent-policy", services: list[str] | None = None
) -> dict[str, Any]:
    """Generate a least-privilege IAM policy document."""
    services = services or ["eks", "ecr", "lambda"]
    statements = []
    if "eks" in services:
        statements.append(
            {
                "Effect": "Allow",
                "Action": [
                    "eks:ListClusters",
                    "eks:DescribeCluster",
                    "eks:UpdateClusterConfig",
                ],
                "Resource": "*",
            }
        )
    if "ecr" in services:
        statements.append(
            {
                "Effect": "Allow",
                "Action": [
                    "ecr:CreateRepository",
                    "ecr:DescribeRepositories",
                    "ecr:GetAuthorizationToken",
                    "ecr:BatchGetImage",
                ],
                "Resource": "*",
            }
        )
    if "lambda" in services:
        statements.append(
            {
                "Effect": "Allow",
                "Action": [
                    "lambda:CreateFunction",
                    "lambda:UpdateFunctionCode",
                    "lambda:GetFunction",
                    "lambda:DeleteFunction",
                ],
                "Resource": "*",
            }
        )

    policy = {
        "Version": "2012-10-17",
        "Statement": statements,
    }
    return {
        "success": True,
        "policy_name": policy_name,
        "policy_document": policy,
    }


def estimate_deployment_cost(
    service_type: str = "lambda",
    requests_per_month: int = 1000000,
    avg_duration_ms: int = 100,
    memory_mb: int = 1024,
    eks_node_hours: int = 720,
    region: str | None = None,
) -> dict[str, Any]:
    """Estimate monthly cost for basic deployment usage."""
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    if service_type == "lambda":
        gb_seconds = (memory_mb / 1024) * (avg_duration_ms / 1000) * requests_per_month
        compute_cost = gb_seconds * 0.0000166667  # approx $/GB-s
        request_cost = (requests_per_month / 1_000_000) * 0.20
        total = compute_cost + request_cost
        return {
            "success": True,
            "service_type": service_type,
            "region": region,
            "estimated_monthly_cost_usd": round(total, 2),
            "details": {
                "compute_cost": round(compute_cost, 2),
                "request_cost": round(request_cost, 2),
            },
            "notes": "Estimate uses public Lambda pricing defaults.",
        }

    hourly_rate = 0.0832  # t3.medium approx in us-east-1
    total = eks_node_hours * hourly_rate
    return {
        "success": True,
        "service_type": service_type,
        "region": region,
        "estimated_monthly_cost_usd": round(total, 2),
        "details": {"node_hours": eks_node_hours, "hourly_rate": hourly_rate},
        "notes": "Estimate uses a single t3.medium on-demand node.",
    }


def create_helm_chart(
    project_path: str,
    chart_name: str,
    image: str,
    chart_version: str = "0.1.0",
    app_version: str = "1.0.0",
    namespace: str = "default",
    container_port: int = 8000,
    service_port: int = 80,
    include_ingress: bool = False,
    include_hpa: bool = False,
    include_configmap: bool = False,
    include_secret: bool = False,
) -> dict[str, Any]:
    """Create a Helm chart for Kubernetes deployment."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    chart_dir = ensure_directory(path / "deployment" / "helm" / chart_name)
    templates_dir = ensure_directory(chart_dir / "templates")

    chart_yaml = f"""apiVersion: v2
name: {chart_name}
description: Helm chart generated by MLOps Agent
type: application
version: {chart_version}
appVersion: "{app_version}"
"""

    values_yaml = f"""replicaCount: 1

image:
  repository: "{image}"
  pullPolicy: IfNotPresent

namespace: "{namespace}"

service:
  type: ClusterIP
  port: {service_port}

containerPort: {container_port}

ingress:
  enabled: {str(include_ingress).lower()}
  className: alb
  host: example.com
  path: /

hpa:
  enabled: {str(include_hpa).lower()}
  minReplicas: 1
  maxReplicas: 3
  targetCPUUtilizationPercentage: 70

configmap:
  enabled: {str(include_configmap).lower()}
  data: {{}}

secret:
  enabled: {str(include_secret).lower()}
  data: {{}}

resources: {{}}
"""

    deployment_tpl = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Chart.Name }}
  namespace: {{ .Values.namespace }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Chart.Name }}
  template:
    metadata:
      labels:
        app: {{ .Chart.Name }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: {{ .Values.image.repository }}
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: {{ .Values.containerPort }}
          {{- if .Values.configmap.enabled }}
          envFrom:
            - configMapRef:
                name: {{ .Chart.Name }}-config
          {{- end }}
          {{- if .Values.secret.enabled }}
          envFrom:
            - secretRef:
                name: {{ .Chart.Name }}-secret
          {{- end }}
"""

    service_tpl = """apiVersion: v1
kind: Service
metadata:
  name: {{ .Chart.Name }}
  namespace: {{ .Values.namespace }}
spec:
  type: {{ .Values.service.type }}
  selector:
    app: {{ .Chart.Name }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.containerPort }}
"""

    ingress_tpl = """{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ .Chart.Name }}
  namespace: {{ .Values.namespace }}
  annotations:
    kubernetes.io/ingress.class: {{ .Values.ingress.className }}
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
spec:
  rules:
    - host: {{ .Values.ingress.host }}
      http:
        paths:
          - path: {{ .Values.ingress.path }}
            pathType: Prefix
            backend:
              service:
                name: {{ .Chart.Name }}
                port:
                  number: {{ .Values.service.port }}
{{- end }}
"""

    hpa_tpl = """{{- if .Values.hpa.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ .Chart.Name }}
  namespace: {{ .Values.namespace }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ .Chart.Name }}
  minReplicas: {{ .Values.hpa.minReplicas }}
  maxReplicas: {{ .Values.hpa.maxReplicas }}
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.hpa.targetCPUUtilizationPercentage }}
{{- end }}
"""

    configmap_tpl = """{{- if .Values.configmap.enabled }}
apiVersion: v1
kind: ConfigMap
metadata:
  name: {{ .Chart.Name }}-config
  namespace: {{ .Values.namespace }}
data:
{{- range $key, $val := .Values.configmap.data }}
  {{ $key }}: {{ $val | quote }}
{{- end }}
{{- end }}
"""

    secret_tpl = """{{- if .Values.secret.enabled }}
apiVersion: v1
kind: Secret
metadata:
  name: {{ .Chart.Name }}-secret
  namespace: {{ .Values.namespace }}
type: Opaque
stringData:
{{- range $key, $val := .Values.secret.data }}
  {{ $key }}: {{ $val | quote }}
{{- end }}
{{- end }}
"""

    (chart_dir / "Chart.yaml").write_text(chart_yaml)
    (chart_dir / "values.yaml").write_text(values_yaml)
    (templates_dir / "deployment.yaml").write_text(deployment_tpl)
    (templates_dir / "service.yaml").write_text(service_tpl)

    if include_ingress:
        (templates_dir / "ingress.yaml").write_text(ingress_tpl)
    if include_hpa:
        (templates_dir / "hpa.yaml").write_text(hpa_tpl)
    if include_configmap:
        (templates_dir / "configmap.yaml").write_text(configmap_tpl)
    if include_secret:
        (templates_dir / "secret.yaml").write_text(secret_tpl)

    return {
        "success": True,
        "chart_path": str(chart_dir),
        "message": f"Helm chart created at {chart_dir}",
    }


def rollback_k8s_deployment(
    project_path: str,
    deployment_name: str,
    namespace: str = "default",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Rollback a Kubernetes deployment using kubectl."""
    cmd = ["kubectl", "rollout", "undo", f"deployment/{deployment_name}", "-n", namespace]
    if dry_run:
        return {"success": True, "dry_run": True, "command": " ".join(cmd)}
    if not check_tool_installed("kubectl"):
        return {"success": False, "error": "kubectl not installed"}
    result = run_command(cmd)
    return {"success": result.get("success", False), "output": result}


def rollback_lambda_stack(
    project_path: str,
    stack_name: str,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Rollback an AWS Lambda CDK stack (destroy)."""
    deploy_dir = Path(project_path) / "deployment" / "fastapi_lambda"
    cmd = ["cdk", "destroy", stack_name, "--force"]
    if dry_run:
        return {"success": True, "dry_run": True, "command": " ".join(cmd), "cwd": str(deploy_dir)}
    if not check_tool_installed("cdk"):
        return {"success": False, "error": "cdk not installed"}
    result = run_command(cmd, cwd=str(deploy_dir))
    return {"success": result.get("success", False), "output": result}


def rollback_deployment(
    project_path: str,
    target: str,
    deployment_name: str | None = None,
    namespace: str = "default",
    stack_name: str | None = None,
    container_id: str | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Rollback a deployment based on target type."""
    target = target.lower()
    if target in {"kserve", "k8s", "kubernetes"}:
        name = deployment_name or "your-deployment"
        return rollback_k8s_deployment(project_path, name, namespace, dry_run)
    if target in {"lambda", "fastapi_lambda"}:
        name = stack_name or "your-stack"
        return rollback_lambda_stack(project_path, name, dry_run)
    if target in {"docker"}:
        if not container_id:
            container_id = "<container_id>"
        cmd = f"docker stop {container_id} && docker rm {container_id}"
        return {"success": True, "dry_run": True, "command": cmd}
    return {
        "success": True,
        "dry_run": True,
        "command": "Review deployment artifacts and undo changes manually.",
    }


# ============================================================================
# MCP Server Setup
# ============================================================================

app = Server("mlops-agent-server")


configure_hydra_dependencies(
    HydraDependencies(
        ensure_directory=lambda path: ensure_directory(path),
        relative_to_project=lambda project_path, artifact_path: relative_to_project(
            project_path, artifact_path
        ),
    )
)
TOOL_REGISTRY = build_tool_registry(sys.modules[__name__])


@app.list_tools()
async def list_tools():
    """List all available MLOps tools from the declarative registry."""
    return TOOL_REGISTRY.catalog()


@app.call_tool()
async def call_tool(name: str, arguments: Any):
    """Validate and dispatch a tool call through the declarative registry."""
    return await TOOL_REGISTRY.call_mcp(name, arguments)


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
