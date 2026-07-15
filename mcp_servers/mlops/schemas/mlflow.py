"""Input schemas for basic MLflow experiment-tracking MCP tools."""

from typing import Any

from pydantic import BaseModel, Field


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
