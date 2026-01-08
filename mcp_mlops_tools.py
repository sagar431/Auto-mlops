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
import json
import os
import subprocess
import shutil
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from pydantic import BaseModel, Field


# ============================================================================
# Helper Functions
# ============================================================================

def run_command(cmd: List[str], cwd: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode
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
    ml_model_config: Optional[Dict[str, Any]] = Field(default=None, description="Model configuration")
    training_config: Optional[Dict[str, Any]] = Field(default=None, description="Training configuration")
    data_config: Optional[Dict[str, Any]] = Field(default=None, description="Data configuration")


class UpdateHydraConfigInput(BaseModel):
    """Update existing Hydra configuration."""
    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(default="configs/config.yaml", description="Relative path to config file")
    updates: Dict[str, Any] = Field(..., description="Dictionary of updates to apply")


class ValidateHydraConfigInput(BaseModel):
    """Validate Hydra configuration."""
    project_path: str = Field(..., description="Path to the ML project")
    config_path: str = Field(default="configs/config.yaml", description="Relative path to config file")


# --- MLflow Experiment Tracking Tools ---

class InitMLflowExperimentInput(BaseModel):
    """Initialize MLflow experiment."""
    experiment_name: str = Field(..., description="Name of the experiment")
    tracking_uri: Optional[str] = Field(default=None, description="MLflow tracking URI")
    artifact_location: Optional[str] = Field(default=None, description="Artifact storage location")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Experiment tags")


class LogMLflowParamsInput(BaseModel):
    """Log parameters to MLflow."""
    run_id: Optional[str] = Field(default=None, description="Run ID (uses active run if not specified)")
    params: Dict[str, Any] = Field(..., description="Parameters to log")


class LogMLflowMetricsInput(BaseModel):
    """Log metrics to MLflow."""
    run_id: Optional[str] = Field(default=None, description="Run ID (uses active run if not specified)")
    metrics: Dict[str, float] = Field(..., description="Metrics to log")
    step: Optional[int] = Field(default=None, description="Step number for the metrics")


class LogMLflowArtifactInput(BaseModel):
    """Log artifact to MLflow."""
    artifact_path: str = Field(..., description="Local path to artifact file or directory")
    artifact_dest: Optional[str] = Field(default=None, description="Destination path in artifact store")
    run_id: Optional[str] = Field(default=None, description="Run ID (uses active run if not specified)")


class RegisterMLflowModelInput(BaseModel):
    """Register model in MLflow Model Registry."""
    model_path: str = Field(..., description="Path to the model artifact")
    model_name: str = Field(..., description="Name for the registered model")
    run_id: Optional[str] = Field(default=None, description="Run ID containing the model")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Model tags")


class GetBestMLflowRunInput(BaseModel):
    """Get best run from experiment based on metric."""
    experiment_name: str = Field(..., description="Name of the experiment")
    metric_name: str = Field(default="accuracy", description="Metric to optimize")
    maximize: bool = Field(default=True, description="Whether to maximize the metric")


class StartMLflowRunInput(BaseModel):
    """Start a new MLflow run."""
    experiment_name: str = Field(..., description="Name of the experiment")
    run_name: Optional[str] = Field(default=None, description="Name for the run")
    tags: Optional[Dict[str, str]] = Field(default=None, description="Run tags")


class EndMLflowRunInput(BaseModel):
    """End an MLflow run."""
    run_id: Optional[str] = Field(default=None, description="Run ID to end")
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
    stages: List[Dict[str, Any]] = Field(..., description="List of pipeline stages")


class DVCPushInput(BaseModel):
    """Push data to DVC remote."""
    project_path: str = Field(..., description="Path to the project")
    remote_name: Optional[str] = Field(default=None, description="Remote name (uses default if not specified)")


class DVCPullInput(BaseModel):
    """Pull data from DVC remote."""
    project_path: str = Field(..., description="Path to the project")
    remote_name: Optional[str] = Field(default=None, description="Remote name (uses default if not specified)")


class DVCReproduceInput(BaseModel):
    """Reproduce DVC pipeline."""
    project_path: str = Field(..., description="Path to the project")
    stages: Optional[List[str]] = Field(default=None, description="Specific stages to reproduce")
    force: bool = Field(default=False, description="Force reproduction even if up-to-date")


# --- Docker Tools ---

class CreateMLDockerfileInput(BaseModel):
    """Create Dockerfile for ML project."""
    project_path: str = Field(..., description="Path to the project")
    base_image: str = Field(default="python:3.11-slim", description="Base Docker image")
    cuda_version: Optional[str] = Field(default=None, description="CUDA version if GPU support needed")
    entry_point: str = Field(default="train.py", description="Training script entry point")
    requirements_file: str = Field(default="requirements.txt", description="Requirements file path")
    expose_port: Optional[int] = Field(default=None, description="Port to expose")


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
    volumes: Optional[Dict[str, str]] = Field(default=None, description="Volume mappings")
    env_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variables")
    command: Optional[str] = Field(default=None, description="Override command")


class PushDockerImageInput(BaseModel):
    """Push Docker image to registry."""
    image_name: str = Field(..., description="Docker image name")
    tag: str = Field(default="latest", description="Image tag")
    registry: Optional[str] = Field(default=None, description="Registry URL")


# --- GitHub Actions Tools ---

class CreateGitHubWorkflowInput(BaseModel):
    """Create GitHub Actions workflow for ML pipeline."""
    project_path: str = Field(..., description="Path to the project")
    workflow_name: str = Field(default="ml-pipeline", description="Workflow name")
    trigger_on: List[str] = Field(default=["push", "workflow_dispatch"], description="Trigger events")
    python_version: str = Field(default="3.11", description="Python version")
    use_dvc: bool = Field(default=True, description="Include DVC steps")
    use_mlflow: bool = Field(default=True, description="Include MLflow tracking")
    accuracy_threshold: Optional[float] = Field(default=None, description="Accuracy threshold for CI")


class AddWorkflowStepInput(BaseModel):
    """Add step to existing GitHub workflow."""
    project_path: str = Field(..., description="Path to the project")
    workflow_file: str = Field(default=".github/workflows/ml-pipeline.yml", description="Workflow file path")
    job_name: str = Field(default="train", description="Job to add step to")
    step: Dict[str, Any] = Field(..., description="Step configuration")


class TriggerGitHubWorkflowInput(BaseModel):
    """Trigger GitHub Actions workflow (via API)."""
    repo: str = Field(..., description="Repository in format owner/repo")
    workflow_id: str = Field(..., description="Workflow file name or ID")
    ref: str = Field(default="main", description="Branch/tag/SHA to run workflow on")
    inputs: Optional[Dict[str, str]] = Field(default=None, description="Workflow inputs")


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
    current_metrics: Dict[str, float] = Field(..., description="Current metrics")
    current_config: Dict[str, Any] = Field(..., description="Current configuration")
    target_accuracy: float = Field(..., description="Target accuracy")
    attempt_number: int = Field(default=1, description="Current attempt number")


class CheckAccuracyThresholdInput(BaseModel):
    """Check if accuracy threshold is met."""
    experiment_name: str = Field(..., description="MLflow experiment name")
    threshold: float = Field(..., description="Accuracy threshold")
    metric_name: str = Field(default="accuracy", description="Metric name to check")


# ============================================================================
# Tool Implementation Functions
# ============================================================================

# --- Hydra Configuration Tools ---

def analyze_project_config(project_path: str) -> Dict[str, Any]:
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
        "config_files": [f.name for f in path.glob("**/*.yaml")] + [f.name for f in path.glob("**/*.yml")],
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
    model_config: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    data_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create Hydra configuration structure."""
    path = Path(project_path)
    
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    
    # Default configurations
    default_model = model_config or {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 2,
        "dropout": 0.5
    }
    
    default_training = training_config or {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "cosine",
        "early_stopping": {
            "patience": 5,
            "min_delta": 0.001
        }
    }
    
    default_data = data_config or {
        "train_path": "data/train",
        "val_path": "data/val",
        "test_path": "data/test",
        "num_workers": 4,
        "augmentation": True
    }
    
    # Create config directories
    configs_dir = ensure_directory(path / "configs")
    ensure_directory(configs_dir / "model")
    ensure_directory(configs_dir / "training")
    ensure_directory(configs_dir / "data")
    
    created_files = []
    
    # Create main config
    main_config = {
        "defaults": [
            {"model": "default"},
            {"training": "default"},
            {"data": "default"},
            "_self_"
        ],
        "experiment_name": "${model.name}_${training.optimizer}_lr${training.learning_rate}",
        "seed": 42,
        "device": "cuda",
        "output_dir": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
        "mlflow": {
            "tracking_uri": "mlruns",
            "experiment_name": "${experiment_name}"
        }
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
        "message": f"Hydra configuration created at {configs_dir}"
    }


def update_hydra_config(
    project_path: str,
    config_path: str = "configs/config.yaml",
    updates: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Update existing Hydra configuration."""
    full_path = Path(project_path) / config_path
    
    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    
    try:
        with open(full_path, "r") as f:
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
            "message": f"Configuration updated at {full_path}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def validate_hydra_config(
    project_path: str,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """Validate Hydra configuration."""
    full_path = Path(project_path) / config_path
    
    if not full_path.exists():
        return {"success": False, "error": f"Config file {full_path} does not exist"}
    
    issues = []
    warnings = []
    
    try:
        with open(full_path, "r") as f:
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
            "config": config
        }
    except yaml.YAMLError as e:
        return {"success": False, "error": f"Invalid YAML: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- MLflow Experiment Tracking Tools ---

def init_mlflow_experiment(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    artifact_location: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Initialize MLflow experiment."""
    try:
        import mlflow
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
                tags=tags
            )
        else:
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        return {
            "success": True,
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "tracking_uri": mlflow.get_tracking_uri(),
            "message": f"Experiment '{experiment_name}' initialized (ID: {experiment_id})"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed. Run: pip install mlflow"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def start_mlflow_run(
    experiment_name: str,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
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
            "message": f"Started run {run.info.run_id}"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_params(
    params: Dict[str, Any],
    run_id: Optional[str] = None
) -> Dict[str, Any]:
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
            "message": f"Logged {len(params)} parameters"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
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
            "message": f"Logged {len(metrics)} metrics"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def log_mlflow_artifact(
    artifact_path: str,
    artifact_dest: Optional[str] = None,
    run_id: Optional[str] = None
) -> Dict[str, Any]:
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
            "message": f"Logged artifact from {artifact_path}"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def register_mlflow_model(
    model_path: str,
    model_name: str,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
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
            "message": f"Registered model '{model_name}' version {result.version}"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_best_mlflow_run(
    experiment_name: str,
    metric_name: str = "accuracy",
    maximize: bool = True
) -> Dict[str, Any]:
    """Get best run from experiment based on metric."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}
        
        order = "DESC" if maximize else "ASC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1
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
            "artifact_uri": best_run.info.artifact_uri
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def end_mlflow_run(
    run_id: Optional[str] = None,
    status: str = "FINISHED"
) -> Dict[str, Any]:
    """End an MLflow run."""
    try:
        import mlflow
        
        mlflow.end_run(status=status)
        
        return {
            "success": True,
            "status": status,
            "message": f"Run ended with status {status}"
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- DVC Data Versioning Tools ---

def init_dvc_repo(
    project_path: str,
    no_scm: bool = False
) -> Dict[str, Any]:
    """Initialize DVC in a repository."""
    if not check_tool_installed("dvc"):
        return {"success": False, "error": "DVC not installed. Run: pip install dvc"}
    
    path = Path(project_path)
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    
    cmd = ["dvc", "init"]
    if no_scm:
        cmd.append("--no-scm")
    
    result = run_command(cmd, cwd=project_path)
    
    if result["success"]:
        return {
            "success": True,
            "project_path": project_path,
            "dvc_dir": str(path / ".dvc"),
            "message": "DVC initialized successfully"
        }
    
    return result


def configure_dvc_remote(
    project_path: str,
    remote_name: str = "storage",
    remote_url: str = None,
    default: bool = True
) -> Dict[str, Any]:
    """Configure DVC remote storage."""
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
            "message": f"DVC remote '{remote_name}' configured with URL: {remote_url}"
        }
    
    return result


def add_data_to_dvc(
    project_path: str,
    data_path: str
) -> Dict[str, Any]:
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
            "message": f"Data added to DVC. Created {dvc_file}"
        }
    
    return result


def create_dvc_pipeline(
    project_path: str,
    stages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create DVC pipeline (dvc.yaml)."""
    path = Path(project_path)
    
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    
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
        "message": f"DVC pipeline created with {len(stages)} stages"
    }


def dvc_push(
    project_path: str,
    remote_name: Optional[str] = None
) -> Dict[str, Any]:
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
            "output": result["stdout"]
        }
    
    return result


def dvc_pull(
    project_path: str,
    remote_name: Optional[str] = None
) -> Dict[str, Any]:
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
            "output": result["stdout"]
        }
    
    return result


def dvc_reproduce(
    project_path: str,
    stages: Optional[List[str]] = None,
    force: bool = False
) -> Dict[str, Any]:
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
            "output": result["stdout"]
        }
    
    return result


# --- Docker Tools ---

def create_ml_dockerfile(
    project_path: str,
    base_image: str = "python:3.11-slim",
    cuda_version: Optional[str] = None,
    entry_point: str = "train.py",
    requirements_file: str = "requirements.txt",
    expose_port: Optional[int] = None
) -> Dict[str, Any]:
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
        "message": f"Dockerfile created at {dockerfile_path}"
    }


def build_ml_docker_image(
    project_path: str,
    image_name: str,
    tag: str = "latest",
    dockerfile: str = "Dockerfile"
) -> Dict[str, Any]:
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
        timeout=600
    )
    
    if result["success"]:
        return {
            "success": True,
            "image_name": full_image_name,
            "message": f"Successfully built image {full_image_name}"
        }
    
    return result


def run_training_container(
    image_name: str,
    tag: str = "latest",
    gpu: bool = False,
    volumes: Optional[Dict[str, str]] = None,
    env_vars: Optional[Dict[str, str]] = None,
    command: Optional[str] = None
) -> Dict[str, Any]:
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
            "message": f"Container started: {container_id[:12]}"
        }
    
    return result


def push_docker_image(
    image_name: str,
    tag: str = "latest",
    registry: Optional[str] = None
) -> Dict[str, Any]:
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
            "message": f"Successfully pushed {full_image_name}"
        }
    
    return result


# --- GitHub Actions Tools ---

def create_github_workflow(
    project_path: str,
    workflow_name: str = "ml-pipeline",
    trigger_on: List[str] = None,
    python_version: str = "3.11",
    use_dvc: bool = True,
    use_mlflow: bool = True,
    accuracy_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Create GitHub Actions workflow for ML pipeline."""
    path = Path(project_path)
    
    if not path.exists():
        return {"success": False, "error": f"Project path {project_path} does not exist"}
    
    trigger_on = trigger_on or ["push", "workflow_dispatch"]
    
    workflow = {
        "name": "ML Training Pipeline",
        "on": {},
        "env": {
            "PYTHON_VERSION": python_version
        },
        "jobs": {
            "train": {
                "runs-on": "ubuntu-latest",
                "steps": []
            }
        }
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
                        "default": str(accuracy_threshold or 0.85)
                    }
                }
            }
    
    steps = workflow["jobs"]["train"]["steps"]
    
    # Checkout
    steps.append({
        "name": "Checkout repository",
        "uses": "actions/checkout@v4"
    })
    
    # Setup Python
    steps.append({
        "name": "Set up Python",
        "uses": "actions/setup-python@v5",
        "with": {
            "python-version": "${{ env.PYTHON_VERSION }}",
            "cache": "pip"
        }
    })
    
    # Install dependencies
    steps.append({
        "name": "Install dependencies",
        "run": "pip install -r requirements.txt"
    })
    
    # DVC setup
    if use_dvc:
        steps.append({
            "name": "Setup DVC",
            "uses": "iterative/setup-dvc@v1"
        })
        steps.append({
            "name": "Pull data from DVC",
            "run": "dvc pull",
            "env": {
                "AWS_ACCESS_KEY_ID": "${{ secrets.AWS_ACCESS_KEY_ID }}",
                "AWS_SECRET_ACCESS_KEY": "${{ secrets.AWS_SECRET_ACCESS_KEY }}"
            }
        })
    
    # MLflow setup
    if use_mlflow:
        workflow["env"]["MLFLOW_TRACKING_URI"] = "${{ secrets.MLFLOW_TRACKING_URI }}"
    
    # Training step
    train_step = {
        "name": "Run training",
        "id": "train",
        "run": "python train.py"
    }
    steps.append(train_step)
    
    # Accuracy check
    if accuracy_threshold:
        steps.append({
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
"""
        })
    
    # Upload artifacts
    steps.append({
        "name": "Upload model artifacts",
        "uses": "actions/upload-artifact@v4",
        "with": {
            "name": "model",
            "path": "models/"
        }
    })
    
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
            "accuracy_threshold": accuracy_threshold
        },
        "message": f"GitHub Actions workflow created at {workflow_path}"
    }


def add_workflow_step(
    project_path: str,
    workflow_file: str = ".github/workflows/ml-pipeline.yml",
    job_name: str = "train",
    step: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Add step to existing GitHub workflow."""
    workflow_path = Path(project_path) / workflow_file
    
    if not workflow_path.exists():
        return {"success": False, "error": f"Workflow file {workflow_path} does not exist"}
    
    try:
        with open(workflow_path, "r") as f:
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
            "message": f"Step added to job '{job_name}'"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Training Control Tools ---

def analyze_training_results(
    project_path: str,
    experiment_name: str,
    target_metric: str = "accuracy",
    target_value: float = 0.85
) -> Dict[str, Any]:
    """Analyze training results and suggest improvements."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}
        
        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{target_metric} DESC"]
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
            "total_runs": len(runs)
        }
        
        # Generate suggestions based on gap
        suggestions = []
        
        if gap > 0.1:
            suggestions.extend([
                "Consider increasing model complexity",
                "Add more data augmentation",
                "Try a different architecture",
                "Increase training epochs significantly"
            ])
        elif gap > 0.05:
            suggestions.extend([
                "Fine-tune learning rate (try 0.0001 or 0.0005)",
                "Add regularization (dropout, weight decay)",
                "Use learning rate scheduling",
                "Increase batch size if memory allows"
            ])
        elif gap > 0:
            suggestions.extend([
                "Small hyperparameter adjustments may help",
                "Try ensemble methods",
                "Fine-tune for a few more epochs",
                "Consider test-time augmentation"
            ])
        
        analysis["suggestions"] = suggestions
        
        return analysis
        
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def suggest_improvements(
    current_metrics: Dict[str, float],
    current_config: Dict[str, Any],
    target_accuracy: float,
    attempt_number: int = 1
) -> Dict[str, Any]:
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
        "reasoning": []
    }
    
    # Learning rate adjustments
    current_lr = current_config.get("learning_rate", 0.001)
    if gap > 0.1:
        # Large gap - try more aggressive changes
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.5
        suggestions["config_changes"]["epochs"] = current_config.get("epochs", 10) * 2
        suggestions["reasoning"].append(f"Large accuracy gap ({gap:.2%}). Reducing LR to {current_lr * 0.5} and doubling epochs.")
    elif gap > 0.05:
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.7
        suggestions["config_changes"]["epochs"] = int(current_config.get("epochs", 10) * 1.5)
        suggestions["reasoning"].append(f"Moderate gap ({gap:.2%}). Adjusting LR to {current_lr * 0.7}.")
    else:
        suggestions["config_changes"]["learning_rate"] = current_lr * 0.9
        suggestions["reasoning"].append(f"Small gap ({gap:.2%}). Fine-tuning LR to {current_lr * 0.9}.")
    
    # Batch size adjustments based on attempt
    if attempt_number > 1:
        current_batch = current_config.get("batch_size", 32)
        suggestions["config_changes"]["batch_size"] = min(current_batch * 2, 128)
        suggestions["reasoning"].append(f"Attempt {attempt_number}: Increasing batch size to {min(current_batch * 2, 128)}.")
    
    # Add regularization on later attempts
    if attempt_number >= 2:
        suggestions["config_changes"]["dropout"] = min(current_config.get("dropout", 0.3) + 0.1, 0.5)
        suggestions["reasoning"].append("Adding more regularization to prevent overfitting.")
    
    # Add augmentation suggestion
    if not current_config.get("augmentation", False):
        suggestions["config_changes"]["augmentation"] = True
        suggestions["reasoning"].append("Enabling data augmentation for better generalization.")
    
    return suggestions


def check_accuracy_threshold(
    experiment_name: str,
    threshold: float,
    metric_name: str = "accuracy"
) -> Dict[str, Any]:
    """Check if accuracy threshold is met."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return {"success": False, "error": f"Experiment '{experiment_name}' not found"}
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if not runs:
            return {
                "success": True,
                "threshold_met": False,
                "current_value": 0,
                "threshold": threshold,
                "message": "No runs found in experiment"
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
            "message": f"{'✅ Threshold met!' if threshold_met else f'❌ Below threshold by {threshold - current_value:.2%}'}"
        }
        
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
            inputSchema=AnalyzeProjectConfigInput.model_json_schema()
        ),
        Tool(
            name="create_hydra_config",
            description="Create Hydra configuration structure with model, training, and data configs",
            inputSchema=CreateHydraConfigInput.model_json_schema()
        ),
        Tool(
            name="update_hydra_config",
            description="Update existing Hydra configuration with new values",
            inputSchema=UpdateHydraConfigInput.model_json_schema()
        ),
        Tool(
            name="validate_hydra_config",
            description="Validate Hydra configuration for errors and missing files",
            inputSchema=ValidateHydraConfigInput.model_json_schema()
        ),
        
        # MLflow Experiment Tracking Tools
        Tool(
            name="init_mlflow_experiment",
            description="Initialize MLflow experiment with tracking URI and tags",
            inputSchema=InitMLflowExperimentInput.model_json_schema()
        ),
        Tool(
            name="start_mlflow_run",
            description="Start a new MLflow run in an experiment",
            inputSchema=StartMLflowRunInput.model_json_schema()
        ),
        Tool(
            name="log_mlflow_params",
            description="Log parameters to MLflow run",
            inputSchema=LogMLflowParamsInput.model_json_schema()
        ),
        Tool(
            name="log_mlflow_metrics",
            description="Log metrics to MLflow run with optional step",
            inputSchema=LogMLflowMetricsInput.model_json_schema()
        ),
        Tool(
            name="log_mlflow_artifact",
            description="Log artifact file or directory to MLflow",
            inputSchema=LogMLflowArtifactInput.model_json_schema()
        ),
        Tool(
            name="register_mlflow_model",
            description="Register model in MLflow Model Registry",
            inputSchema=RegisterMLflowModelInput.model_json_schema()
        ),
        Tool(
            name="get_best_mlflow_run",
            description="Get best run from experiment based on metric",
            inputSchema=GetBestMLflowRunInput.model_json_schema()
        ),
        Tool(
            name="end_mlflow_run",
            description="End an MLflow run with status",
            inputSchema=EndMLflowRunInput.model_json_schema()
        ),
        
        # DVC Data Versioning Tools
        Tool(
            name="init_dvc_repo",
            description="Initialize DVC in a repository",
            inputSchema=InitDVCRepoInput.model_json_schema()
        ),
        Tool(
            name="configure_dvc_remote",
            description="Configure DVC remote storage (S3, GCS, Azure, etc.)",
            inputSchema=ConfigureDVCRemoteInput.model_json_schema()
        ),
        Tool(
            name="add_data_to_dvc",
            description="Add data file or directory to DVC tracking",
            inputSchema=AddDataToDVCInput.model_json_schema()
        ),
        Tool(
            name="create_dvc_pipeline",
            description="Create DVC pipeline with stages (dvc.yaml)",
            inputSchema=CreateDVCPipelineInput.model_json_schema()
        ),
        Tool(
            name="dvc_push",
            description="Push data to DVC remote storage",
            inputSchema=DVCPushInput.model_json_schema()
        ),
        Tool(
            name="dvc_pull",
            description="Pull data from DVC remote storage",
            inputSchema=DVCPullInput.model_json_schema()
        ),
        Tool(
            name="dvc_reproduce",
            description="Reproduce DVC pipeline (run stages)",
            inputSchema=DVCReproduceInput.model_json_schema()
        ),
        
        # Docker Tools
        Tool(
            name="create_ml_dockerfile",
            description="Create Dockerfile for ML project with GPU support option",
            inputSchema=CreateMLDockerfileInput.model_json_schema()
        ),
        Tool(
            name="build_ml_docker_image",
            description="Build Docker image for ML project",
            inputSchema=BuildMLDockerImageInput.model_json_schema()
        ),
        Tool(
            name="run_training_container",
            description="Run training in Docker container with GPU and volume support",
            inputSchema=RunTrainingContainerInput.model_json_schema()
        ),
        Tool(
            name="push_docker_image",
            description="Push Docker image to registry",
            inputSchema=PushDockerImageInput.model_json_schema()
        ),
        
        # GitHub Actions Tools
        Tool(
            name="create_github_workflow",
            description="Create GitHub Actions workflow for ML pipeline with DVC, MLflow, and accuracy checks",
            inputSchema=CreateGitHubWorkflowInput.model_json_schema()
        ),
        Tool(
            name="add_workflow_step",
            description="Add step to existing GitHub Actions workflow",
            inputSchema=AddWorkflowStepInput.model_json_schema()
        ),
        
        # Training Control Tools
        Tool(
            name="analyze_training_results",
            description="Analyze training results and suggest improvements",
            inputSchema=AnalyzeTrainingResultsInput.model_json_schema()
        ),
        Tool(
            name="suggest_improvements",
            description="Suggest configuration improvements based on current metrics",
            inputSchema=SuggestImprovementsInput.model_json_schema()
        ),
        Tool(
            name="check_accuracy_threshold",
            description="Check if accuracy threshold is met in experiment",
            inputSchema=CheckAccuracyThresholdInput.model_json_schema()
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
                input_data.data_config
            )
        
        elif name == "update_hydra_config":
            input_data = UpdateHydraConfigInput(**arguments)
            result = update_hydra_config(
                input_data.project_path,
                input_data.config_path,
                input_data.updates
            )
        
        elif name == "validate_hydra_config":
            input_data = ValidateHydraConfigInput(**arguments)
            result = validate_hydra_config(
                input_data.project_path,
                input_data.config_path
            )
        
        # MLflow Tools
        elif name == "init_mlflow_experiment":
            input_data = InitMLflowExperimentInput(**arguments)
            result = init_mlflow_experiment(
                input_data.experiment_name,
                input_data.tracking_uri,
                input_data.artifact_location,
                input_data.tags
            )
        
        elif name == "start_mlflow_run":
            input_data = StartMLflowRunInput(**arguments)
            result = start_mlflow_run(
                input_data.experiment_name,
                input_data.run_name,
                input_data.tags
            )
        
        elif name == "log_mlflow_params":
            input_data = LogMLflowParamsInput(**arguments)
            result = log_mlflow_params(input_data.params, input_data.run_id)
        
        elif name == "log_mlflow_metrics":
            input_data = LogMLflowMetricsInput(**arguments)
            result = log_mlflow_metrics(
                input_data.metrics,
                input_data.step,
                input_data.run_id
            )
        
        elif name == "log_mlflow_artifact":
            input_data = LogMLflowArtifactInput(**arguments)
            result = log_mlflow_artifact(
                input_data.artifact_path,
                input_data.artifact_dest,
                input_data.run_id
            )
        
        elif name == "register_mlflow_model":
            input_data = RegisterMLflowModelInput(**arguments)
            result = register_mlflow_model(
                input_data.model_path,
                input_data.model_name,
                input_data.run_id,
                input_data.tags
            )
        
        elif name == "get_best_mlflow_run":
            input_data = GetBestMLflowRunInput(**arguments)
            result = get_best_mlflow_run(
                input_data.experiment_name,
                input_data.metric_name,
                input_data.maximize
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
                input_data.default
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
            result = dvc_reproduce(
                input_data.project_path,
                input_data.stages,
                input_data.force
            )
        
        # Docker Tools
        elif name == "create_ml_dockerfile":
            input_data = CreateMLDockerfileInput(**arguments)
            result = create_ml_dockerfile(
                input_data.project_path,
                input_data.base_image,
                input_data.cuda_version,
                input_data.entry_point,
                input_data.requirements_file,
                input_data.expose_port
            )
        
        elif name == "build_ml_docker_image":
            input_data = BuildMLDockerImageInput(**arguments)
            result = build_ml_docker_image(
                input_data.project_path,
                input_data.image_name,
                input_data.tag,
                input_data.dockerfile
            )
        
        elif name == "run_training_container":
            input_data = RunTrainingContainerInput(**arguments)
            result = run_training_container(
                input_data.image_name,
                input_data.tag,
                input_data.gpu,
                input_data.volumes,
                input_data.env_vars,
                input_data.command
            )
        
        elif name == "push_docker_image":
            input_data = PushDockerImageInput(**arguments)
            result = push_docker_image(
                input_data.image_name,
                input_data.tag,
                input_data.registry
            )
        
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
                input_data.accuracy_threshold
            )
        
        elif name == "add_workflow_step":
            input_data = AddWorkflowStepInput(**arguments)
            result = add_workflow_step(
                input_data.project_path,
                input_data.workflow_file,
                input_data.job_name,
                input_data.step
            )
        
        # Training Control Tools
        elif name == "analyze_training_results":
            input_data = AnalyzeTrainingResultsInput(**arguments)
            result = analyze_training_results(
                input_data.project_path,
                input_data.experiment_name,
                input_data.target_metric,
                input_data.target_value
            )
        
        elif name == "suggest_improvements":
            input_data = SuggestImprovementsInput(**arguments)
            result = suggest_improvements(
                input_data.current_metrics,
                input_data.current_config,
                input_data.target_accuracy,
                input_data.attempt_number
            )
        
        elif name == "check_accuracy_threshold":
            input_data = CheckAccuracyThresholdInput(**arguments)
            result = check_accuracy_threshold(
                input_data.experiment_name,
                input_data.threshold,
                input_data.metric_name
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
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
