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
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except Exception:
    boto3 = None
    ClientError = Exception
    BOTO3_AVAILABLE = False

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import BaseModel, Field

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


def ensure_directory(path: str) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def relative_to_project(project_path: str, artifact_path: str | Path) -> str:
    """Return a project-relative artifact path when possible."""
    path = Path(artifact_path)
    try:
        return str(path.relative_to(Path(project_path)))
    except ValueError:
        return str(path)


# ============================================================================
# Pydantic Input Models
# ============================================================================

# --- Hydra Configuration Tools ---


class AnalyzeProjectConfigInput(BaseModel):
    """Analyze project for configuration needs."""

    project_path: str = Field(..., description="Path to the ML project")


class CreateHydraConfigInput(BaseModel):
    """Create Hydra configuration structure."""

    project_path: str = Field(..., description="Path to the ML project")
    config_name: str = Field(default="config", description="Name of main config file")
    ml_model_config: dict[str, Any] | None = Field(default=None, description="Model configuration")
    training_config: dict[str, Any] | None = Field(
        default=None, description="Training configuration"
    )
    data_config: dict[str, Any] | None = Field(default=None, description="Data configuration")


class UpdateHydraConfigInput(BaseModel):
    """Update existing Hydra configuration."""

    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(
        default="configs/config.yaml", description="Relative path to config file"
    )
    updates: dict[str, Any] = Field(..., description="Dictionary of updates to apply")


class ValidateHydraConfigInput(BaseModel):
    """Validate Hydra configuration."""

    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(
        default="configs/config.yaml", description="Relative path to config file"
    )


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


class ConfigureLitserverInput(BaseModel):
    """Configure LitServe server settings."""

    project_path: str = Field(..., description="Path to the project")
    max_batch_size: int = Field(default=64, description="Maximum batch size for inference")
    batch_timeout: float = Field(default=0.05, description="Batch timeout in seconds")
    workers_per_device: int = Field(default=4, description="Number of workers per device")
    accelerator: str = Field(default="auto", description="Accelerator: cpu, gpu, auto")
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

# --- Hydra Configuration Tools ---


def analyze_project_config(project_path: str) -> dict[str, Any]:
    """Analyze project structure for configuration needs."""
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    analysis = {
        "has_hydra": (path / "configs").exists(),
        "has_config_yaml": (path / "configs" / "config.yaml").exists(),
        "has_requirements": (path / "requirements.txt").exists(),
        "has_train_script": (path / "train.py").exists(),
        "has_model_dir": (path / "model").exists() or (path / "models").exists(),
        "python_files": [f.name for f in path.glob("*.py")],
        "config_files": [f.name for f in path.glob("**/*.yaml")]
        + [f.name for f in path.glob("**/*.yml")],
    }

    # Detect framework
    requirements_path = path / "requirements.txt"
    if requirements_path.exists():
        content = requirements_path.read_text().lower()
        analysis["framework"] = {
            "pytorch": "torch" in content or "pytorch" in content,
            "tensorflow": "tensorflow" in content,
            "sklearn": "scikit-learn" in content or "sklearn" in content,
            "hydra": "hydra-core" in content,
            "mlflow": "mlflow" in content,
            "dvc": "dvc" in content,
        }

    analysis["success"] = True
    analysis["recommendations"] = []

    if not analysis["has_hydra"]:
        analysis["recommendations"].append("Create configs/ directory for Hydra configuration")
    if not analysis["has_requirements"]:
        analysis["recommendations"].append("Add requirements.txt for dependencies")
    if not analysis["has_train_script"]:
        analysis["recommendations"].append("Create train.py as main entry point")

    return analysis


def create_hydra_config(
    project_path: str,
    config_name: str = "config",
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    data_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create Hydra configuration structure."""
    path = Path(project_path)

    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    # Default configurations
    default_model = model_config or {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "dropout": 0.5,
    }

    default_training = training_config or {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "early_stopping": {"patience": 5, "min_delta": 0.001},
    }

    default_data = data_config or {
        "train_path": "data/train",
        "val_path": "data/val",
        "test_path": "data/test",
        "num_workers": 4,
        "augmentation": True,
    }

    # Create config directories
    configs_dir = ensure_directory(path / "configs")
    ensure_directory(configs_dir / "model")
    ensure_directory(configs_dir / "training")
    ensure_directory(configs_dir / "data")

    created_files = []

    # Create main config
    main_config = {
        "defaults": [{"model": "default"}, {"training": "default"}, {"data": "default"}, "_self_"],
        "experiment_name": "${model.name}_${training.optimizer}_lr${training.learning_rate}",
        "seed": 42,
        "device": "cuda",
        "output_dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "mlflow": {"tracking_uri": "mlruns", "experiment_name": "${experiment_name}"},
    }

    config_path = configs_dir / f"{config_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(main_config, f, default_flow_style=False, sort_keys=False)
    created_files.append(str(config_path))

    # Create model config
    model_path = configs_dir / "model" / "default.yaml"
    with open(model_path, "w") as f:
        yaml.dump(default_model, f, default_flow_style=False)
    created_files.append(str(model_path))

    # Create training config
    training_path = configs_dir / "training" / "default.yaml"
    with open(training_path, "w") as f:
        yaml.dump(default_training, f, default_flow_style=False)
    created_files.append(str(training_path))

    # Create data config
    data_path = configs_dir / "data" / "default.yaml"
    with open(data_path, "w") as f:
        yaml.dump(default_data, f, default_flow_style=False)
    created_files.append(str(data_path))

    return {
        "success": True,
        "created_files": created_files,
        "config_dir": str(configs_dir),
        "verification_results": [
            {
                "check_name": "hydra_config_validates",
                "evidence_type": "declared",
                "source_step": "create_or_validate_hydra_config",
                "passed": True,
                "evidence": f"Hydra config generated at {relative_to_project(project_path, config_path)}.",
            }
        ],
        "artifact_manifest": {
            "entries": [
                {
                    "artifact_type": "configuration",
                    "producing_step": "create_or_validate_hydra_config",
                    "state": "generated",
                    "path": relative_to_project(project_path, config_path),
                }
            ]
        },
        "message": f"Hydra configuration created at {configs_dir}",
    }


def update_hydra_config(
    project_path: str, config_path: str = "configs/config.yaml", updates: dict[str, Any] = None
) -> dict[str, Any]:
    """Update existing Hydra configuration."""
    full_path = Path(project_path) / config_path

    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}

    try:
        with open(full_path) as f:
            config = yaml.safe_load(f)

        # Deep update
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        deep_update(config, updates or {})

        with open(full_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        return {
            "success": True,
            "config_path": str(full_path),
            "updated_config": config,
            "message": f"Configuration updated at {full_path}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def validate_hydra_config(
    project_path: str, config_path: str = "configs/config.yaml"
) -> dict[str, Any]:
    """Validate Hydra configuration."""
    full_path = Path(project_path) / config_path

    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}

    issues = []
    warnings = []

    try:
        with open(full_path) as f:
            config = yaml.safe_load(f)

        # Check for required fields
        if "defaults" not in config:
            warnings.append("No 'defaults' section found - Hydra composition may not work")

        # Check for common issues
        if isinstance(config.get("defaults"), list):
            for default in config["defaults"]:
                if isinstance(default, dict):
                    for key, value in default.items():
                        if key not in ["_self_"]:
                            sub_config_path = Path(project_path) / "configs" / key / f"{value}.yaml"
                            if not sub_config_path.exists():
                                issues.append(f"Missing config file: {sub_config_path}")

        return {
            "success": len(issues) == 0,
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "config": config,
        }
    except yaml.YAMLError as e:
        return {"success": False, "error": f"Invalid YAML: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
) -> dict[str, Any]:
    """Create LitServe API for model serving."""
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}

    # Create deployment directory
    deploy_dir = ensure_directory(path / "deployment" / "litserve")

    # Generate class name from model name
    class_name = "".join(word.capitalize() for word in model_name.replace("-", "_").split("_"))

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
import torch
import litserve as ls
from PIL import Image
import io
import base64
from typing import Any


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
    api = {class_name}API()
    server = ls.LitServer(
        api,
        accelerator="auto",
        max_batch_size=64,
        batch_timeout=0.05,
        workers_per_device=4
    )
    server.run(port=8000)
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
        "message": f"LitServe API created at {deploy_dir}",
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


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MLOps tools."""
    return [
        # Hydra Configuration Tools
        Tool(
            name="analyze_project_config",
            description="Analyze ML project structure for configuration needs (Hydra, requirements, scripts)",
            inputSchema=AnalyzeProjectConfigInput.model_json_schema(),
        ),
        Tool(
            name="create_hydra_config",
            description="Create Hydra configuration structure with model, training, and data configs",
            inputSchema=CreateHydraConfigInput.model_json_schema(),
        ),
        Tool(
            name="update_hydra_config",
            description="Update existing Hydra configuration with new values",
            inputSchema=UpdateHydraConfigInput.model_json_schema(),
        ),
        Tool(
            name="validate_hydra_config",
            description="Validate Hydra configuration for errors and missing files",
            inputSchema=ValidateHydraConfigInput.model_json_schema(),
        ),
        # MLflow Experiment Tracking Tools
        Tool(
            name="init_mlflow_experiment",
            description="Initialize MLflow experiment with tracking URI and tags",
            inputSchema=InitMLflowExperimentInput.model_json_schema(),
        ),
        Tool(
            name="start_mlflow_run",
            description="Start a new MLflow run in an experiment",
            inputSchema=StartMLflowRunInput.model_json_schema(),
        ),
        Tool(
            name="log_mlflow_params",
            description="Log parameters to MLflow run",
            inputSchema=LogMLflowParamsInput.model_json_schema(),
        ),
        Tool(
            name="log_mlflow_metrics",
            description="Log metrics to MLflow run with optional step",
            inputSchema=LogMLflowMetricsInput.model_json_schema(),
        ),
        Tool(
            name="log_mlflow_artifact",
            description="Log artifact file or directory to MLflow",
            inputSchema=LogMLflowArtifactInput.model_json_schema(),
        ),
        Tool(
            name="register_mlflow_model",
            description="Register model in MLflow Model Registry",
            inputSchema=RegisterMLflowModelInput.model_json_schema(),
        ),
        Tool(
            name="get_best_mlflow_run",
            description="Get best run from experiment based on metric",
            inputSchema=GetBestMLflowRunInput.model_json_schema(),
        ),
        Tool(
            name="end_mlflow_run",
            description="End an MLflow run with status",
            inputSchema=EndMLflowRunInput.model_json_schema(),
        ),
        # DVC Data Versioning Tools
        Tool(
            name="init_dvc_repo",
            description="Initialize DVC in a repository",
            inputSchema=InitDVCRepoInput.model_json_schema(),
        ),
        Tool(
            name="configure_dvc_remote",
            description="Configure DVC remote storage (S3, GCS, Azure, etc.)",
            inputSchema=ConfigureDVCRemoteInput.model_json_schema(),
        ),
        Tool(
            name="add_data_to_dvc",
            description="Add data file or directory to DVC tracking",
            inputSchema=AddDataToDVCInput.model_json_schema(),
        ),
        Tool(
            name="create_dvc_pipeline",
            description="Create DVC pipeline with stages (dvc.yaml)",
            inputSchema=CreateDVCPipelineInput.model_json_schema(),
        ),
        Tool(
            name="dvc_push",
            description="Push data to DVC remote storage",
            inputSchema=DVCPushInput.model_json_schema(),
        ),
        Tool(
            name="dvc_pull",
            description="Pull data from DVC remote storage",
            inputSchema=DVCPullInput.model_json_schema(),
        ),
        Tool(
            name="dvc_reproduce",
            description="Reproduce DVC pipeline (run stages)",
            inputSchema=DVCReproduceInput.model_json_schema(),
        ),
        # Docker Tools
        Tool(
            name="create_ml_dockerfile",
            description="Create Dockerfile for ML project with GPU support option",
            inputSchema=CreateMLDockerfileInput.model_json_schema(),
        ),
        Tool(
            name="build_ml_docker_image",
            description="Build Docker image for ML project",
            inputSchema=BuildMLDockerImageInput.model_json_schema(),
        ),
        Tool(
            name="run_training_container",
            description="Run training in Docker container with GPU and volume support",
            inputSchema=RunTrainingContainerInput.model_json_schema(),
        ),
        Tool(
            name="push_docker_image",
            description="Push Docker image to registry",
            inputSchema=PushDockerImageInput.model_json_schema(),
        ),
        # GitHub Actions Tools
        Tool(
            name="create_github_workflow",
            description="Create GitHub Actions workflow for ML pipeline with DVC, MLflow, and accuracy checks",
            inputSchema=CreateGitHubWorkflowInput.model_json_schema(),
        ),
        Tool(
            name="add_workflow_step",
            description="Add step to existing GitHub Actions workflow",
            inputSchema=AddWorkflowStepInput.model_json_schema(),
        ),
        # Training Control Tools
        Tool(
            name="analyze_training_results",
            description="Analyze training results and suggest improvements",
            inputSchema=AnalyzeTrainingResultsInput.model_json_schema(),
        ),
        Tool(
            name="suggest_improvements",
            description="Suggest configuration improvements based on current metrics",
            inputSchema=SuggestImprovementsInput.model_json_schema(),
        ),
        Tool(
            name="check_accuracy_threshold",
            description="Check if accuracy threshold is met in experiment",
            inputSchema=CheckAccuracyThresholdInput.model_json_schema(),
        ),
        # Data Quality Tools
        Tool(
            name="validate_dataset",
            description="Validate ML dataset for quality issues (missing values, duplicates, class balance, outliers, image validity)",
            inputSchema=ValidateDatasetInput.model_json_schema(),
        ),
        Tool(
            name="create_expectation_suite",
            description="Create a Great Expectations expectation suite for data validation with customizable expectations",
            inputSchema=CreateExpectationSuiteInput.model_json_schema(),
        ),
        Tool(
            name="check_data_quality",
            description="Check data quality using Great Expectations-based validation. Validates datasets against expectations or runs basic quality checks (nulls, duplicates, row count). Returns quality score, validation results, and recommendations.",
            inputSchema=CheckDataQualityInput.model_json_schema(),
        ),
        Tool(
            name="profile_dataset",
            description="Profile a dataset to get comprehensive statistics including row/column counts, missing values, duplicates, memory usage, and per-column statistics (mean, median, std, quartiles for numeric; top values for categorical).",
            inputSchema=ProfileDatasetInput.model_json_schema(),
        ),
        Tool(
            name="detect_anomalies",
            description="Detect anomalies in a dataset using statistical methods (IQR outliers, Z-score outliers, missing value patterns, duplicate rows). Returns detailed anomaly information with affected rows and severity levels.",
            inputSchema=DetectAnomaliesInput.model_json_schema(),
        ),
        Tool(
            name="validate_schema",
            description="Validate a dataset against a defined schema. Checks for missing columns, extra columns (in strict mode), type mismatches, and constraint violations (nullability, uniqueness, value ranges, patterns).",
            inputSchema=ValidateSchemaInput.model_json_schema(),
        ),
        Tool(
            name="compare_distributions",
            description="Compare distributions between a reference dataset and current dataset for drift detection. Uses Kolmogorov-Smirnov test to identify significant distribution shifts in numeric columns.",
            inputSchema=CompareDistributionsInput.model_json_schema(),
        ),
        # Monitoring Tools
        Tool(
            name="detect_data_drift",
            description="Detect data drift between reference and current datasets using Evidently AI. Provides comprehensive drift analysis with per-feature results, severity levels, and recommendations. Supports both numerical and categorical columns with configurable thresholds.",
            inputSchema=DetectDataDriftInput.model_json_schema(),
        ),
        Tool(
            name="monitor_model_performance",
            description="Monitor model performance metrics and detect degradation. Calculates classification metrics (accuracy, precision, recall, F1, AUC-ROC) or regression metrics (MSE, RMSE, MAE, R²), tracks performance over time, compares to baseline, and provides health status with recommendations.",
            inputSchema=MonitorModelPerformanceInput.model_json_schema(),
        ),
        Tool(
            name="setup_alerting",
            description="Setup alerting configuration for model monitoring. Creates threshold, anomaly, drift, or composite alerts with configurable notification channels (email, Slack, PagerDuty, webhook). Generates alert rules, notification configs, and runner scripts.",
            inputSchema=SetupAlertingInput.model_json_schema(),
        ),
        # Deployment Tools (Phase 4)
        # LitServe
        Tool(
            name="create_litserve_api",
            description="Create LitServe API for high-throughput model serving with batching and GPU support",
            inputSchema=CreateLitserveAPIInput.model_json_schema(),
        ),
        Tool(
            name="configure_litserver",
            description="Configure LitServe server settings (batch size, workers, accelerator)",
            inputSchema=ConfigureLitserverInput.model_json_schema(),
        ),
        # Gradio
        Tool(
            name="create_gradio_interface",
            description="Create Gradio interface for quick model demos and prototypes",
            inputSchema=CreateGradioInterfaceInput.model_json_schema(),
        ),
        Tool(
            name="deploy_to_huggingface",
            description="Deploy Gradio app to Hugging Face Spaces",
            inputSchema=DeployToHuggingfaceInput.model_json_schema(),
        ),
        # FastAPI + Lambda
        Tool(
            name="create_fastapi_app",
            description="Create FastAPI application for serverless model serving",
            inputSchema=CreateFastAPIAppInput.model_json_schema(),
        ),
        Tool(
            name="create_lambda_dockerfile",
            description="Create Dockerfile for AWS Lambda deployment with Lambda Web Adapter",
            inputSchema=CreateLambdaDockerfileInput.model_json_schema(),
        ),
        Tool(
            name="generate_cdk_stack",
            description="Generate AWS CDK stack for Lambda deployment with API Gateway",
            inputSchema=GenerateCDKStackInput.model_json_schema(),
        ),
        # TorchServe
        Tool(
            name="create_torchserve_handler",
            description="Create TorchServe custom handler for enterprise model serving",
            inputSchema=CreateTorchserveHandlerInput.model_json_schema(),
        ),
        Tool(
            name="create_mar_archive",
            description="Create TorchServe MAR (Model Archive) build script",
            inputSchema=CreateMARArchiveInput.model_json_schema(),
        ),
        Tool(
            name="generate_torchserve_config",
            description="Generate TorchServe configuration (ports, workers)",
            inputSchema=GenerateTorchserveConfigInput.model_json_schema(),
        ),
        # KServe
        Tool(
            name="create_inference_service_yaml",
            description="Create KServe InferenceService YAML for Kubernetes deployment",
            inputSchema=CreateInferenceServiceYAMLInput.model_json_schema(),
        ),
        Tool(
            name="generate_kserve_config",
            description="Generate KServe scaling and resource configuration",
            inputSchema=GenerateKServeConfigInput.model_json_schema(),
        ),
        # Kubernetes Manifests
        Tool(
            name="create_k8s_deployment_yaml",
            description="Create Kubernetes Deployment YAML",
            inputSchema=CreateK8sDeploymentInput.model_json_schema(),
        ),
        Tool(
            name="create_k8s_service_yaml",
            description="Create Kubernetes Service YAML",
            inputSchema=CreateK8sServiceInput.model_json_schema(),
        ),
        Tool(
            name="create_k8s_ingress_yaml",
            description="Create Kubernetes Ingress YAML (ALB annotations for EKS)",
            inputSchema=CreateK8sIngressInput.model_json_schema(),
        ),
        Tool(
            name="create_k8s_hpa_yaml",
            description="Create Kubernetes HPA YAML",
            inputSchema=CreateK8sHPAInput.model_json_schema(),
        ),
        Tool(
            name="create_k8s_configmap_yaml",
            description="Create Kubernetes ConfigMap YAML",
            inputSchema=CreateK8sConfigMapInput.model_json_schema(),
        ),
        Tool(
            name="create_k8s_secret_yaml",
            description="Create Kubernetes Secret YAML",
            inputSchema=CreateK8sSecretInput.model_json_schema(),
        ),
        Tool(
            name="generate_rollback_plan",
            description="Generate rollback plan for a deployment target",
            inputSchema=GenerateRollbackPlanInput.model_json_schema(),
        ),
        # AWS Automation
        Tool(
            name="list_eks_clusters",
            description="List EKS clusters in a region",
            inputSchema=ListEKSClustersInput.model_json_schema(),
        ),
        Tool(
            name="update_kubeconfig",
            description="Update kubeconfig for an EKS cluster using AWS CLI",
            inputSchema=UpdateKubeconfigInput.model_json_schema(),
        ),
        Tool(
            name="create_ecr_repo",
            description="Create or get an ECR repository",
            inputSchema=CreateECRRepoInput.model_json_schema(),
        ),
        Tool(
            name="get_ecr_login",
            description="Get ECR docker login command",
            inputSchema=GetECRLoginInput.model_json_schema(),
        ),
        Tool(
            name="generate_iam_policy",
            description="Generate least-privilege IAM policy",
            inputSchema=GenerateIAMPolicyInput.model_json_schema(),
        ),
        Tool(
            name="estimate_deployment_cost",
            description="Estimate monthly deployment cost for Lambda/EKS",
            inputSchema=EstimateDeploymentCostInput.model_json_schema(),
        ),
        Tool(
            name="create_helm_chart",
            description="Create Helm chart for Kubernetes deployment",
            inputSchema=CreateHelmChartInput.model_json_schema(),
        ),
        Tool(
            name="rollback_k8s_deployment",
            description="Rollback a Kubernetes deployment using kubectl",
            inputSchema=RollbackK8sDeploymentInput.model_json_schema(),
        ),
        Tool(
            name="rollback_lambda_stack",
            description="Rollback an AWS Lambda CDK stack",
            inputSchema=RollbackLambdaStackInput.model_json_schema(),
        ),
        Tool(
            name="rollback_deployment",
            description="Rollback a deployment based on target type",
            inputSchema=RollbackDeploymentInput.model_json_schema(),
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls for MLOps operations."""

    try:
        # Hydra Configuration Tools
        if name == "analyze_project_config":
            input_data = AnalyzeProjectConfigInput(**arguments)
            result = analyze_project_config(input_data.project_path)

        elif name == "create_hydra_config":
            input_data = CreateHydraConfigInput(**arguments)
            result = create_hydra_config(
                input_data.project_path,
                input_data.config_name,
                input_data.ml_model_config,
                input_data.training_config,
                input_data.data_config,
            )

        elif name == "update_hydra_config":
            input_data = UpdateHydraConfigInput(**arguments)
            result = update_hydra_config(
                input_data.project_path, input_data.config_path, input_data.updates
            )

        elif name == "validate_hydra_config":
            input_data = ValidateHydraConfigInput(**arguments)
            result = validate_hydra_config(input_data.project_path, input_data.config_path)

        # MLflow Tools
        elif name == "init_mlflow_experiment":
            input_data = InitMLflowExperimentInput(**arguments)
            result = init_mlflow_experiment(
                input_data.experiment_name,
                input_data.tracking_uri,
                input_data.artifact_location,
                input_data.tags,
            )

        elif name == "start_mlflow_run":
            input_data = StartMLflowRunInput(**arguments)
            result = start_mlflow_run(
                input_data.experiment_name, input_data.run_name, input_data.tags
            )

        elif name == "log_mlflow_params":
            input_data = LogMLflowParamsInput(**arguments)
            result = log_mlflow_params(input_data.params, input_data.run_id)

        elif name == "log_mlflow_metrics":
            input_data = LogMLflowMetricsInput(**arguments)
            result = log_mlflow_metrics(input_data.metrics, input_data.step, input_data.run_id)

        elif name == "log_mlflow_artifact":
            input_data = LogMLflowArtifactInput(**arguments)
            result = log_mlflow_artifact(
                input_data.artifact_path, input_data.artifact_dest, input_data.run_id
            )

        elif name == "register_mlflow_model":
            input_data = RegisterMLflowModelInput(**arguments)
            result = register_mlflow_model(
                input_data.model_path, input_data.model_name, input_data.run_id, input_data.tags
            )

        elif name == "get_best_mlflow_run":
            input_data = GetBestMLflowRunInput(**arguments)
            result = get_best_mlflow_run(
                input_data.experiment_name, input_data.metric_name, input_data.maximize
            )

        elif name == "end_mlflow_run":
            input_data = EndMLflowRunInput(**arguments)
            result = end_mlflow_run(input_data.run_id, input_data.status)

        # DVC Tools
        elif name == "init_dvc_repo":
            input_data = InitDVCRepoInput(**arguments)
            result = init_dvc_repo(input_data.project_path, input_data.no_scm)

        elif name == "configure_dvc_remote":
            input_data = ConfigureDVCRemoteInput(**arguments)
            result = configure_dvc_remote(
                input_data.project_path,
                input_data.remote_name,
                input_data.remote_url,
                input_data.default,
            )

        elif name == "add_data_to_dvc":
            input_data = AddDataToDVCInput(**arguments)
            result = add_data_to_dvc(input_data.project_path, input_data.data_path)

        elif name == "create_dvc_pipeline":
            input_data = CreateDVCPipelineInput(**arguments)
            result = create_dvc_pipeline(input_data.project_path, input_data.stages)

        elif name == "dvc_push":
            input_data = DVCPushInput(**arguments)
            result = dvc_push(input_data.project_path, input_data.remote_name)

        elif name == "dvc_pull":
            input_data = DVCPullInput(**arguments)
            result = dvc_pull(input_data.project_path, input_data.remote_name)

        elif name == "dvc_reproduce":
            input_data = DVCReproduceInput(**arguments)
            result = dvc_reproduce(input_data.project_path, input_data.stages, input_data.force)

        # Docker Tools
        elif name == "create_ml_dockerfile":
            input_data = CreateMLDockerfileInput(**arguments)
            result = create_ml_dockerfile(
                input_data.project_path,
                input_data.base_image,
                input_data.cuda_version,
                input_data.entry_point,
                input_data.requirements_file,
                input_data.expose_port,
            )

        elif name == "build_ml_docker_image":
            input_data = BuildMLDockerImageInput(**arguments)
            result = build_ml_docker_image(
                input_data.project_path,
                input_data.image_name,
                input_data.tag,
                input_data.dockerfile,
            )

        elif name == "run_training_container":
            input_data = RunTrainingContainerInput(**arguments)
            result = run_training_container(
                input_data.image_name,
                input_data.tag,
                input_data.gpu,
                input_data.volumes,
                input_data.env_vars,
                input_data.command,
            )

        elif name == "push_docker_image":
            input_data = PushDockerImageInput(**arguments)
            result = push_docker_image(input_data.image_name, input_data.tag, input_data.registry)

        # GitHub Actions Tools
        elif name == "create_github_workflow":
            input_data = CreateGitHubWorkflowInput(**arguments)
            result = create_github_workflow(
                input_data.project_path,
                input_data.workflow_name,
                input_data.trigger_on,
                input_data.python_version,
                input_data.use_dvc,
                input_data.use_mlflow,
                input_data.accuracy_threshold,
            )

        elif name == "add_workflow_step":
            input_data = AddWorkflowStepInput(**arguments)
            result = add_workflow_step(
                input_data.project_path,
                input_data.workflow_file,
                input_data.job_name,
                input_data.step,
            )

        # Training Control Tools
        elif name == "analyze_training_results":
            input_data = AnalyzeTrainingResultsInput(**arguments)
            result = analyze_training_results(
                input_data.project_path,
                input_data.experiment_name,
                input_data.target_metric,
                input_data.target_value,
            )

        elif name == "suggest_improvements":
            input_data = SuggestImprovementsInput(**arguments)
            result = suggest_improvements(
                input_data.current_metrics,
                input_data.current_config,
                input_data.target_accuracy,
                input_data.attempt_number,
            )

        elif name == "check_accuracy_threshold":
            input_data = CheckAccuracyThresholdInput(**arguments)
            result = check_accuracy_threshold(
                input_data.experiment_name, input_data.threshold, input_data.metric_name
            )

        # Data Quality Tools
        elif name == "validate_dataset":
            input_data = ValidateDatasetInput(**arguments)
            result = validate_dataset(
                input_data.dataset_path,
                input_data.dataset_type,
                input_data.checks,
                input_data.sample_size,
            )

        elif name == "create_expectation_suite":
            input_data = CreateExpectationSuiteInput(**arguments)
            result = create_expectation_suite(
                input_data.project_path,
                input_data.suite_name,
                input_data.expectations,
                input_data.output_dir,
            )

        elif name == "check_data_quality":
            input_data = CheckDataQualityInput(**arguments)
            result = check_data_quality(
                input_data.dataset_path,
                input_data.expectations,
                input_data.include_statistics,
                input_data.fail_on_error,
            )

        elif name == "profile_dataset":
            input_data = ProfileDatasetInput(**arguments)
            result = profile_dataset(
                input_data.dataset_path,
                input_data.dataset_name,
                input_data.include_column_stats,
            )

        elif name == "detect_anomalies":
            input_data = DetectAnomaliesInput(**arguments)
            result = detect_anomalies(
                input_data.dataset_path,
                input_data.dataset_name,
                input_data.methods,
                input_data.outlier_threshold,
                input_data.zscore_threshold,
            )

        elif name == "validate_schema":
            input_data = ValidateSchemaInput(**arguments)
            result = validate_schema(
                input_data.dataset_path,
                input_data.schema_definition,
            )

        elif name == "compare_distributions":
            input_data = CompareDistributionsInput(**arguments)
            result = compare_distributions(
                input_data.reference_path,
                input_data.current_path,
                input_data.columns,
                input_data.significance_level,
            )

        # Monitoring Tools
        elif name == "detect_data_drift":
            input_data = DetectDataDriftInput(**arguments)
            result = detect_data_drift(
                input_data.reference_path,
                input_data.current_path,
                input_data.feature_columns,
                input_data.categorical_columns,
                input_data.numerical_columns,
                input_data.drift_threshold,
                input_data.dataset_name,
            )

        elif name == "monitor_model_performance":
            input_data = MonitorModelPerformanceInput(**arguments)
            result = monitor_model_performance(
                input_data.model_name,
                input_data.y_true,
                input_data.y_pred,
                input_data.y_prob,
                input_data.task_type,
                input_data.model_version,
                input_data.degradation_threshold,
                input_data.baseline_metrics,
                input_data.metrics_to_check,
                input_data.record_snapshot,
                input_data.storage_path,
            )

        elif name == "setup_alerting":
            input_data = SetupAlertingInput(**arguments)
            result = setup_alerting(
                input_data.project_path,
                input_data.alert_name,
                input_data.alert_type,
                input_data.metrics,
                input_data.thresholds,
                input_data.notification_channels,
                input_data.notification_config,
                input_data.evaluation_window,
                input_data.cooldown_period,
                input_data.severity,
                input_data.enabled,
            )

        # Deployment Tools (Phase 4)
        # LitServe
        elif name == "create_litserve_api":
            input_data = CreateLitserveAPIInput(**arguments)
            result = create_litserve_api(
                input_data.project_path,
                input_data.model_path,
                input_data.model_name,
                input_data.model_type,
                input_data.class_labels,
            )

        elif name == "configure_litserver":
            input_data = ConfigureLitserverInput(**arguments)
            result = configure_litserver(
                input_data.project_path,
                input_data.max_batch_size,
                input_data.batch_timeout,
                input_data.workers_per_device,
                input_data.accelerator,
                input_data.port,
            )

        # Gradio
        elif name == "create_gradio_interface":
            input_data = CreateGradioInterfaceInput(**arguments)
            result = create_gradio_interface(
                input_data.project_path,
                input_data.model_path,
                input_data.model_name,
                input_data.interface_type,
                input_data.title,
                input_data.description,
                input_data.examples,
                input_data.share,
            )

        elif name == "deploy_to_huggingface":
            input_data = DeployToHuggingfaceInput(**arguments)
            result = deploy_to_huggingface(
                input_data.project_path,
                input_data.space_name,
                input_data.hf_token,
                input_data.private,
            )

        # FastAPI + Lambda
        elif name == "create_fastapi_app":
            input_data = CreateFastAPIAppInput(**arguments)
            result = create_fastapi_app(
                input_data.project_path,
                input_data.model_path,
                input_data.model_name,
                input_data.endpoint_type,
                input_data.title,
            )

        elif name == "create_lambda_dockerfile":
            input_data = CreateLambdaDockerfileInput(**arguments)
            result = create_lambda_dockerfile(
                input_data.project_path,
                input_data.python_version,
                input_data.model_file,
                input_data.port,
            )

        elif name == "generate_cdk_stack":
            input_data = GenerateCDKStackInput(**arguments)
            result = generate_cdk_stack(
                input_data.project_path,
                input_data.stack_name,
                input_data.model_name,
                input_data.memory_size,
                input_data.timeout,
                input_data.stage,
            )

        # TorchServe
        elif name == "create_torchserve_handler":
            input_data = CreateTorchserveHandlerInput(**arguments)
            result = create_torchserve_handler(
                input_data.project_path,
                input_data.model_path,
                input_data.model_name,
                input_data.handler_type,
            )

        elif name == "create_mar_archive":
            input_data = CreateMARArchiveInput(**arguments)
            result = create_mar_archive(
                input_data.project_path,
                input_data.model_name,
                input_data.model_file,
                input_data.handler_file,
                input_data.version,
                input_data.extra_files,
            )

        elif name == "generate_torchserve_config":
            input_data = GenerateTorchserveConfigInput(**arguments)
            result = generate_torchserve_config(
                input_data.project_path,
                input_data.model_name,
                input_data.inference_port,
                input_data.management_port,
                input_data.metrics_port,
                input_data.workers,
            )

        # KServe
        elif name == "create_inference_service_yaml":
            input_data = CreateInferenceServiceYAMLInput(**arguments)
            result = create_inference_service_yaml(
                input_data.project_path,
                input_data.service_name,
                input_data.model_name,
                input_data.storage_uri,
                input_data.namespace,
                input_data.runtime,
                input_data.min_replicas,
                input_data.max_replicas,
            )

        elif name == "generate_kserve_config":
            input_data = GenerateKServeConfigInput(**arguments)
            result = generate_kserve_config(
                input_data.project_path,
                input_data.service_name,
                input_data.min_replicas,
                input_data.max_replicas,
                input_data.target_utilization,
                input_data.gpu_enabled,
                input_data.gpu_count,
            )

        # Kubernetes Manifests
        elif name == "create_k8s_deployment_yaml":
            input_data = CreateK8sDeploymentInput(**arguments)
            result = create_k8s_deployment_yaml(
                input_data.project_path,
                input_data.name,
                input_data.image,
                input_data.replicas,
                input_data.container_port,
                input_data.namespace,
                input_data.labels,
                input_data.env,
                input_data.resources,
            )

        elif name == "create_k8s_service_yaml":
            input_data = CreateK8sServiceInput(**arguments)
            result = create_k8s_service_yaml(
                input_data.project_path,
                input_data.name,
                input_data.selector,
                input_data.port,
                input_data.target_port,
                input_data.service_type,
                input_data.namespace,
            )

        elif name == "create_k8s_ingress_yaml":
            input_data = CreateK8sIngressInput(**arguments)
            result = create_k8s_ingress_yaml(
                input_data.project_path,
                input_data.name,
                input_data.host,
                input_data.service_name,
                input_data.service_port,
                input_data.path,
                input_data.namespace,
                input_data.ingress_class,
                input_data.alb_scheme,
                input_data.certificate_arn,
                input_data.annotations,
            )

        elif name == "create_k8s_hpa_yaml":
            input_data = CreateK8sHPAInput(**arguments)
            result = create_k8s_hpa_yaml(
                input_data.project_path,
                input_data.name,
                input_data.deployment_name,
                input_data.min_replicas,
                input_data.max_replicas,
                input_data.target_cpu_utilization,
                input_data.namespace,
            )

        elif name == "create_k8s_configmap_yaml":
            input_data = CreateK8sConfigMapInput(**arguments)
            result = create_k8s_configmap_yaml(
                input_data.project_path,
                input_data.name,
                input_data.data,
                input_data.namespace,
            )

        elif name == "create_k8s_secret_yaml":
            input_data = CreateK8sSecretInput(**arguments)
            result = create_k8s_secret_yaml(
                input_data.project_path,
                input_data.name,
                input_data.data,
                input_data.namespace,
                input_data.encode,
            )

        elif name == "generate_rollback_plan":
            input_data = GenerateRollbackPlanInput(**arguments)
            result = generate_rollback_plan(
                input_data.project_path,
                input_data.target,
                input_data.deployment_name,
                input_data.namespace,
                input_data.error,
            )

        # AWS Automation
        elif name == "list_eks_clusters":
            input_data = ListEKSClustersInput(**arguments)
            result = list_eks_clusters(input_data.region)

        elif name == "update_kubeconfig":
            input_data = UpdateKubeconfigInput(**arguments)
            result = update_kubeconfig(input_data.cluster_name, input_data.region)

        elif name == "create_ecr_repo":
            input_data = CreateECRRepoInput(**arguments)
            result = create_ecr_repo(
                input_data.repo_name,
                input_data.region,
                input_data.scan_on_push,
                input_data.mutable_tags,
            )

        elif name == "get_ecr_login":
            input_data = GetECRLoginInput(**arguments)
            result = get_ecr_login(input_data.region)

        elif name == "generate_iam_policy":
            input_data = GenerateIAMPolicyInput(**arguments)
            result = generate_iam_policy(input_data.policy_name, input_data.services)

        elif name == "estimate_deployment_cost":
            input_data = EstimateDeploymentCostInput(**arguments)
            result = estimate_deployment_cost(
                input_data.service_type,
                input_data.requests_per_month,
                input_data.avg_duration_ms,
                input_data.memory_mb,
                input_data.eks_node_hours,
                input_data.region,
            )

        elif name == "create_helm_chart":
            input_data = CreateHelmChartInput(**arguments)
            result = create_helm_chart(
                input_data.project_path,
                input_data.chart_name,
                input_data.image,
                input_data.chart_version,
                input_data.app_version,
                input_data.namespace,
                input_data.container_port,
                input_data.service_port,
                input_data.include_ingress,
                input_data.include_hpa,
                input_data.include_configmap,
                input_data.include_secret,
            )

        elif name == "rollback_k8s_deployment":
            input_data = RollbackK8sDeploymentInput(**arguments)
            result = rollback_k8s_deployment(
                input_data.project_path,
                input_data.deployment_name,
                input_data.namespace,
                input_data.dry_run,
            )

        elif name == "rollback_lambda_stack":
            input_data = RollbackLambdaStackInput(**arguments)
            result = rollback_lambda_stack(
                input_data.project_path,
                input_data.stack_name,
                input_data.dry_run,
            )

        elif name == "rollback_deployment":
            input_data = RollbackDeploymentInput(**arguments)
            result = rollback_deployment(
                input_data.project_path,
                input_data.target,
                input_data.deployment_name,
                input_data.namespace,
                input_data.stack_name,
                input_data.container_id,
                input_data.dry_run,
            )

        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    except Exception as e:
        error_result = {"error": str(e), "tool": name}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
