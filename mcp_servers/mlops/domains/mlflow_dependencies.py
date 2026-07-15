"""Lazy dependency boundary for the basic MLflow MCP domain."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class MLflowSDK(Protocol):
    """MLflow module operations used by the basic handlers."""

    def set_tracking_uri(self, tracking_uri: str) -> None: ...

    def get_experiment_by_name(self, experiment_name: str) -> Any: ...

    def create_experiment(
        self,
        experiment_name: str,
        artifact_location: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str: ...

    def set_experiment(self, experiment_name: str) -> None: ...

    def get_tracking_uri(self) -> str: ...

    def start_run(self, **kwargs: Any) -> Any: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...

    def log_artifact(self, artifact_path: str, artifact_dest: str | None = None) -> None: ...

    def log_artifacts(self, artifact_path: str, artifact_dest: str | None = None) -> None: ...

    def register_model(
        self, model_uri: str, model_name: str, tags: dict[str, str] | None = None
    ) -> Any: ...

    def end_run(self, status: str = "FINISHED") -> None: ...


class MLflowClient(Protocol):
    """Tracking-client operations used by best-run lookup."""

    def get_experiment_by_name(self, experiment_name: str) -> Any: ...

    def search_runs(self, **kwargs: Any) -> list[Any]: ...


class MLflowFilesystem(Protocol):
    """Artifact-path operations used before logging local artifacts."""

    def exists(self, path: str | Path) -> bool: ...

    def is_directory(self, path: str | Path) -> bool: ...


@dataclass(frozen=True)
class LocalMLflowFilesystem:
    """Real local artifact-path implementation."""

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def is_directory(self, path: str | Path) -> bool:
        return Path(path).is_dir()


def load_mlflow_sdk() -> MLflowSDK:
    """Import MLflow only when a handler actually needs the SDK."""
    import mlflow

    return mlflow


def create_mlflow_client() -> MLflowClient:
    """Import and construct MlflowClient only for best-run lookup."""
    from mlflow.tracking import MlflowClient

    return MlflowClient()


@dataclass(frozen=True)
class MLflowDependencies:
    """Immutable, lazily evaluated dependencies for basic MLflow handlers."""

    sdk_loader: Callable[[], MLflowSDK] = load_mlflow_sdk
    client_factory: Callable[[], MLflowClient] = create_mlflow_client
    filesystem: MLflowFilesystem = field(default_factory=LocalMLflowFilesystem)
