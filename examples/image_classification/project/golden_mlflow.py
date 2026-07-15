"""Local SQLite-backed MLflow tracking for the golden DVC training slice."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

try:
    from .golden_data import load_and_verify_manifest, sha256_file
    from .model import load_golden_checkpoint
except ImportError:  # Support direct execution from the project directory.
    from golden_data import load_and_verify_manifest, sha256_file
    from model import load_golden_checkpoint

EXPERIMENT_NAME = "golden-image-classification"
DVC_PIPELINE = "examples/image_classification/project/dvc.yaml"
DVC_STAGE = "train_golden"
DEFAULT_STORAGE_DIR = Path(__file__).resolve().parent / ".mlflow"
REQUIRED_ARTIFACTS = {
    "checkpoint/model.pt",
    "config/training_config.json",
    "dataset/manifest.json",
    "lineage/lineage.json",
    "metrics/metrics.json",
}


class GoldenMLflowError(ValueError):
    """Raised when local MLflow evidence violates the golden-slice contract."""


def _tracking_paths(storage_dir: str | Path) -> tuple[Path, Path, str]:
    root = Path(storage_dir).resolve()
    database_path = root / "tracking.db"
    artifact_dir = root / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return database_path, artifact_dir, f"sqlite:///{database_path.as_posix()}"


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise GoldenMLflowError(f"{description} was not found")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GoldenMLflowError(f"{description} is unreadable") from exc
    if not isinstance(value, dict):
        raise GoldenMLflowError(f"{description} must contain a JSON object")
    return value


def _git_commit(project_dir: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        return None
    try:
        int(commit, 16)
    except ValueError:
        return None
    return commit


def _collect_evidence(artifact_dir: Path, dataset_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    dataset_dir = dataset_dir.resolve()
    checkpoint_path = artifact_dir / "model.pt"
    config_path = artifact_dir / "training_config.json"
    metrics_path = artifact_dir / "metrics.json"
    lineage_path = artifact_dir / "lineage.json"
    manifest_path = dataset_dir / "manifest.json"

    loaded = load_golden_checkpoint(checkpoint_path)
    config = _read_json(config_path, "Golden training configuration")
    metrics_document = _read_json(metrics_path, "Golden metrics document")
    lineage = _read_json(lineage_path, "Golden lineage document")
    manifest, verified_lineage = load_and_verify_manifest(dataset_dir, manifest_path)
    final_metrics = metrics_document.get("final")
    if not isinstance(final_metrics, dict):
        raise GoldenMLflowError("Golden metrics document has no final metrics")

    if loaded.training_config != config:
        raise GoldenMLflowError("Checkpoint training configuration conflicts with training_config.json")
    if loaded.metrics != final_metrics:
        raise GoldenMLflowError("Checkpoint metrics conflict with metrics.json")
    if loaded.dataset_lineage != lineage:
        raise GoldenMLflowError("Checkpoint dataset lineage conflicts with lineage.json")
    if lineage != verified_lineage:
        raise GoldenMLflowError("Recorded lineage conflicts with the verified dataset manifest")
    if lineage.get("dataset_checksum") != manifest.get("dataset_checksum"):
        raise GoldenMLflowError("Dataset checksum conflicts with the dataset manifest")
    manifest_checksum = sha256_file(manifest_path)
    if lineage.get("manifest_checksum") != manifest_checksum:
        raise GoldenMLflowError("Manifest checksum conflicts with recorded lineage")

    return {
        "paths": {
            "checkpoint/model.pt": checkpoint_path,
            "config/training_config.json": config_path,
            "dataset/manifest.json": manifest_path,
            "lineage/lineage.json": lineage_path,
            "metrics/metrics.json": metrics_path,
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_checksum": manifest_checksum,
        "checkpoint": loaded,
        "config": config,
        "metrics": {key: float(value) for key, value in final_metrics.items()},
        "lineage": lineage,
        "manifest": manifest,
    }


def _expected_params(evidence: dict[str, Any]) -> dict[str, str]:
    config = evidence["config"]
    checkpoint = evidence["checkpoint"]
    return {
        "seed": str(config["seed"]),
        "epochs": str(config["epochs"]),
        "batch_size": str(config["batch_size"]),
        "learning_rate": str(config["learning_rate"]),
        "architecture": checkpoint.architecture,
        "image_size": str(checkpoint.image_size),
        "class_names": json.dumps(list(checkpoint.class_names), separators=(",", ":")),
        "train_samples": str(config["train_samples"]),
        "validation_samples": str(config["validation_samples"]),
    }


def _expected_tags(evidence: dict[str, Any]) -> dict[str, str]:
    checkpoint = evidence["checkpoint"]
    lineage = evidence["lineage"]
    return {
        "golden.dataset_schema": str(lineage["schema_version"]),
        "golden.dataset_checksum": str(lineage["dataset_checksum"]),
        "golden.manifest_checksum": evidence["manifest_checksum"],
        "golden.checkpoint_schema": checkpoint.schema_version,
        "golden.checkpoint_sha256": evidence["checkpoint_sha256"],
        "golden.dvc_pipeline": DVC_PIPELINE,
        "golden.dvc_stage": DVC_STAGE,
    }


def _artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    observed: set[str] = set()
    for artifact in client.list_artifacts(run_id, path):
        if artifact.is_dir:
            observed.update(_artifact_paths(client, run_id, artifact.path))
        else:
            observed.add(artifact.path)
    return observed


def log_golden_mlflow_run(
    artifact_dir: str | Path,
    dataset_dir: str | Path,
    duration_seconds: float,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    experiment_name: str = EXPERIMENT_NAME,
) -> dict[str, Any]:
    """Validate deterministic outputs, log one local run, and verify it immediately."""
    if duration_seconds < 0:
        raise GoldenMLflowError("Training duration must be non-negative")
    artifact_root = Path(artifact_dir).resolve()
    dataset_root = Path(dataset_dir).resolve()
    evidence = _collect_evidence(artifact_root, dataset_root)
    database_path, mlflow_artifact_dir, tracking_uri = _tracking_paths(storage_dir)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=mlflow_artifact_dir.as_uri(),
            tags={"golden.contract": "dvc-checkpoint-lineage.v1"},
        )
    else:
        if experiment.artifact_location != mlflow_artifact_dir.as_uri():
            raise GoldenMLflowError(
                "Existing MLflow experiment artifact location conflicts with local storage"
            )
        experiment_id = experiment.experiment_id

    tags = _expected_tags(evidence)
    commit = _git_commit(Path(__file__).resolve().parent)
    if commit is not None:
        tags["golden.git_commit"] = commit
    run = client.create_run(
        experiment_id,
        tags={**tags, "mlflow.runName": "golden-dvc-training"},
        run_name="golden-dvc-training",
    )
    run_id = run.info.run_id
    try:
        for key, value in _expected_params(evidence).items():
            client.log_param(run_id, key, value)
        client.log_metric(
            run_id, "validation_accuracy", evidence["metrics"]["validation_accuracy"]
        )
        client.log_metric(run_id, "validation_loss", evidence["metrics"]["validation_loss"])
        client.log_metric(run_id, "training_duration_seconds", float(duration_seconds))
        for artifact_path, local_path in evidence["paths"].items():
            destination = str(Path(artifact_path).parent)
            client.log_artifact(run_id, str(local_path), destination)
        client.set_terminated(run_id, status=RunStatus.to_string(RunStatus.FINISHED))
    except Exception:
        client.set_terminated(run_id, status=RunStatus.to_string(RunStatus.FAILED))
        raise

    verified = verify_golden_mlflow_run(
        artifact_dir=artifact_root,
        dataset_dir=dataset_root,
        storage_dir=storage_dir,
        experiment_name=experiment_name,
        run_id=run_id,
    )
    return {
        **verified,
        "database_path": str(database_path),
        "artifact_root": str(mlflow_artifact_dir),
    }


def verify_golden_mlflow_run(
    artifact_dir: str | Path,
    dataset_dir: str | Path,
    storage_dir: str | Path = DEFAULT_STORAGE_DIR,
    experiment_name: str = EXPERIMENT_NAME,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Query local MLflow and prove that a run matches current DVC evidence."""
    evidence = _collect_evidence(Path(artifact_dir), Path(dataset_dir))
    _, _, tracking_uri = _tracking_paths(storage_dir)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise GoldenMLflowError(f"MLflow experiment '{experiment_name}' was not found")
    if run_id is None:
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise GoldenMLflowError("No golden MLflow run was found")
        run = runs[0]
        run_id = run.info.run_id
    else:
        try:
            run = client.get_run(run_id)
        except Exception as exc:
            raise GoldenMLflowError(f"MLflow run '{run_id}' was not found") from exc
    if run.info.experiment_id != experiment.experiment_id:
        raise GoldenMLflowError("MLflow run belongs to a different experiment")
    if run.info.status != "FINISHED":
        raise GoldenMLflowError("MLflow run is not finished successfully")

    expected_params = _expected_params(evidence)
    for key, expected in expected_params.items():
        if run.data.params.get(key) != expected:
            raise GoldenMLflowError(f"MLflow parameter '{key}' conflicts with training evidence")
    for key in ("validation_accuracy", "validation_loss"):
        if key not in run.data.metrics or run.data.metrics[key] != evidence["metrics"][key]:
            raise GoldenMLflowError(f"MLflow metric '{key}' conflicts with training evidence")
    duration = run.data.metrics.get("training_duration_seconds")
    if duration is None or duration < 0:
        raise GoldenMLflowError("MLflow training duration metric is missing or invalid")

    expected_tags = _expected_tags(evidence)
    for key, expected in expected_tags.items():
        if run.data.tags.get(key) != expected:
            raise GoldenMLflowError(f"MLflow tag '{key}' conflicts with deterministic evidence")
    git_commit = run.data.tags.get("golden.git_commit")
    if git_commit is not None:
        try:
            valid_commit = len(git_commit) == 40 and int(git_commit, 16) >= 0
        except ValueError:
            valid_commit = False
        if not valid_commit:
            raise GoldenMLflowError("MLflow Git commit tag is invalid")

    observed_artifacts = _artifact_paths(client, run_id)
    missing_artifacts = REQUIRED_ARTIFACTS - observed_artifacts
    if missing_artifacts:
        missing = ", ".join(sorted(missing_artifacts))
        raise GoldenMLflowError(f"MLflow run is missing required artifacts: {missing}")
    with tempfile.TemporaryDirectory(prefix="golden-mlflow-verify-") as download_dir:
        downloaded = Path(
            client.download_artifacts(run_id, "checkpoint/model.pt", dst_path=download_dir)
        )
        if sha256_file(downloaded) != evidence["checkpoint_sha256"]:
            raise GoldenMLflowError("Logged checkpoint artifact SHA-256 conflicts with local checkpoint")

    return {
        "status": "verified",
        "experiment_name": experiment.name,
        "experiment_id": experiment.experiment_id,
        "run_id": run_id,
        "tracking_uri": tracking_uri,
        "params": {key: run.data.params[key] for key in sorted(expected_params)},
        "metrics": {
            key: run.data.metrics[key]
            for key in ("training_duration_seconds", "validation_accuracy", "validation_loss")
        },
        "tags": {
            key: run.data.tags[key]
            for key in sorted({*expected_tags, "golden.git_commit"})
            if key in run.data.tags
        },
        "artifacts": sorted(REQUIRED_ARTIFACTS),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("track", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/dvc-golden"))
        subparser.add_argument("--dataset-dir", type=Path, default=Path("data/golden"))
        subparser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE_DIR)
        subparser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
        if command == "track":
            subparser.add_argument("--duration-seconds", type=float, required=True)
        else:
            subparser.add_argument("--run-id")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "track":
            result = log_golden_mlflow_run(
                args.artifact_dir,
                args.dataset_dir,
                args.duration_seconds,
                args.storage_dir,
                args.experiment_name,
            )
        else:
            result = verify_golden_mlflow_run(
                args.artifact_dir,
                args.dataset_dir,
                args.storage_dir,
                args.experiment_name,
                args.run_id,
            )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
