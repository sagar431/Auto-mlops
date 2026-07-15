"""Basic MLflow tool implementations and declarative registrations."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType
from typing import Any

from ..compatibility import RootModuleHandler
from ..registry import ToolSpec
from ..schemas.mlflow import (
    EndMLflowRunInput,
    GetBestMLflowRunInput,
    InitMLflowExperimentInput,
    LogMLflowArtifactInput,
    LogMLflowMetricsInput,
    LogMLflowParamsInput,
    RegisterMLflowModelInput,
    StartMLflowRunInput,
)
from .mlflow_dependencies import MLflowDependencies, MLflowFilesystem

_configured_dependencies = MLflowDependencies()
_dependency_override: ContextVar[MLflowDependencies | None] = ContextVar(
    "mlflow_dependency_override", default=None
)


def configure_dependencies(dependencies: MLflowDependencies) -> None:
    """Configure the immutable dependency baseline for the root facade."""
    global _configured_dependencies
    _configured_dependencies = dependencies


@contextmanager
def use_dependencies(dependencies: MLflowDependencies) -> Iterator[None]:
    """Temporarily inject MLflow dependencies and restore prior state."""
    token = _dependency_override.set(dependencies)
    try:
        yield
    finally:
        _dependency_override.reset(token)


def _dependencies() -> MLflowDependencies:
    return _dependency_override.get() or _configured_dependencies


def init_mlflow_experiment(
    experiment_name: str,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
    tags: dict[str, str] | None = None,
    project_path: str | None = None,
) -> dict[str, Any]:
    """Initialize MLflow experiment."""
    try:
        mlflow = _dependencies().sdk_loader()

        if project_path and tracking_uri is None:
            tracking_uri = str(Path(project_path) / "mlruns")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def start_mlflow_run(
    experiment_name: str, run_name: str | None = None, tags: dict[str, str] | None = None
) -> dict[str, Any]:
    """Start a new MLflow run."""
    try:
        mlflow = _dependencies().sdk_loader()
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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def log_mlflow_params(params: dict[str, Any], run_id: str | None = None) -> dict[str, Any]:
    """Log parameters to MLflow."""
    try:
        mlflow = _dependencies().sdk_loader()
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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def log_mlflow_metrics(
    metrics: dict[str, float], step: int | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Log metrics to MLflow."""
    try:
        mlflow = _dependencies().sdk_loader()
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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _log_artifact(
    mlflow: Any,
    filesystem: MLflowFilesystem,
    artifact_path: str,
    artifact_dest: str | None,
) -> None:
    if filesystem.is_directory(artifact_path):
        mlflow.log_artifacts(artifact_path, artifact_dest)
    else:
        mlflow.log_artifact(artifact_path, artifact_dest)


def log_mlflow_artifact(
    artifact_path: str, artifact_dest: str | None = None, run_id: str | None = None
) -> dict[str, Any]:
    """Log artifact to MLflow."""
    try:
        dependencies = _dependencies()
        mlflow = dependencies.sdk_loader()
        if not dependencies.filesystem.exists(artifact_path):
            return {
                "success": False,
                "error": f"Artifact path {artifact_path} does not exist",
            }

        if run_id:
            with mlflow.start_run(run_id=run_id):
                _log_artifact(mlflow, dependencies.filesystem, artifact_path, artifact_dest)
        else:
            _log_artifact(mlflow, dependencies.filesystem, artifact_path, artifact_dest)

        return {
            "success": True,
            "artifact_path": artifact_path,
            "message": f"Logged artifact from {artifact_path}",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def register_mlflow_model(
    model_path: str,
    model_name: str,
    run_id: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Register model in MLflow Model Registry."""
    try:
        mlflow = _dependencies().sdk_loader()
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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def get_best_mlflow_run(
    experiment_name: str, metric_name: str = "accuracy", maximize: bool = True
) -> dict[str, Any]:
    """Get best run from experiment based on metric."""
    try:
        client = _dependencies().client_factory()
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
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def end_mlflow_run(run_id: str | None = None, status: str = "FINISHED") -> dict[str, Any]:
    """End an MLflow run."""
    try:
        mlflow = _dependencies().sdk_loader()
        mlflow.end_run(status=status)
        return {
            "success": True,
            "status": status,
            "message": f"Run ended with status {status}",
        }
    except ImportError:
        return {"success": False, "error": "MLflow not installed"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def tool_specs(root_module: ModuleType) -> tuple[ToolSpec, ...]:
    """Return basic MLflow ToolSpecs with dynamic root handler resolution."""
    definitions = (
        (
            "init_mlflow_experiment",
            "Initialize MLflow experiment with tracking URI and tags",
            InitMLflowExperimentInput,
        ),
        (
            "start_mlflow_run",
            "Start a new MLflow run in an experiment",
            StartMLflowRunInput,
        ),
        ("log_mlflow_params", "Log parameters to MLflow run", LogMLflowParamsInput),
        (
            "log_mlflow_metrics",
            "Log metrics to MLflow run with optional step",
            LogMLflowMetricsInput,
        ),
        (
            "log_mlflow_artifact",
            "Log artifact file or directory to MLflow",
            LogMLflowArtifactInput,
        ),
        (
            "register_mlflow_model",
            "Register model in MLflow Model Registry",
            RegisterMLflowModelInput,
        ),
        (
            "get_best_mlflow_run",
            "Get best run from experiment based on metric",
            GetBestMLflowRunInput,
        ),
        ("end_mlflow_run", "End an MLflow run with status", EndMLflowRunInput),
    )
    return tuple(
        ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            handler=RootModuleHandler(root_module, name),
        )
        for name, description, input_model in definitions
    )
