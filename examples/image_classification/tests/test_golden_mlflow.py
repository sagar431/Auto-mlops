"""Hermetic tests for local MLflow evidence linked to the golden DVC slice."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from golden_data import DatasetConfig, prepare_golden_dataset
from golden_mlflow import (
    DVC_PIPELINE,
    DVC_STAGE,
    EXPERIMENT_NAME,
    REQUIRED_ARTIFACTS,
    GoldenMLflowError,
    log_golden_mlflow_run,
    verify_golden_mlflow_run,
)
from golden_train import TrainingConfig, train_golden
from mlflow.tracking import MlflowClient

PROJECT_DIR = Path(__file__).resolve().parents[1] / "project"


def _training_evidence(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    dataset_dir = tmp_path / "dataset"
    artifact_dir = tmp_path / "training-artifacts"
    prepare_golden_dataset(
        dataset_dir,
        DatasetConfig(seed=17, train_samples=16, validation_samples=8),
    )
    result = train_golden(
        artifact_dir,
        TrainingConfig(
            seed=17,
            epochs=1,
            train_samples=16,
            validation_samples=8,
            batch_size=4,
            learning_rate=0.05,
        ),
        dataset_dir=dataset_dir,
        manifest_path=dataset_dir / "manifest.json",
    )
    return dataset_dir, artifact_dir, result


def test_local_mlflow_run_logs_and_verifies_complete_lineage(tmp_path):
    dataset_dir, artifact_dir, training = _training_evidence(tmp_path)
    storage_dir = tmp_path / "tracking"

    result = log_golden_mlflow_run(
        artifact_dir,
        dataset_dir,
        float(training["duration_seconds"]),
        storage_dir,
    )

    assert result["status"] == "verified"
    assert result["experiment_name"] == EXPERIMENT_NAME
    assert result["run_id"]
    assert Path(result["database_path"]).is_file()
    assert Path(result["artifact_root"]).is_dir()
    assert result["params"] == {
        "architecture": "tiny_color_cnn_v1",
        "batch_size": "4",
        "class_names": '["red","blue"]',
        "epochs": "1",
        "image_size": "16",
        "learning_rate": "0.05",
        "seed": "17",
        "train_samples": "16",
        "validation_samples": "8",
    }
    assert result["metrics"]["validation_accuracy"] == pytest.approx(1.0)
    assert result["metrics"]["validation_loss"] >= 0
    assert result["metrics"]["training_duration_seconds"] >= 0
    lineage = json.loads(Path(training["lineage_path"]).read_text())
    assert result["tags"]["golden.dataset_schema"] == "golden-red-blue-dataset.v1"
    assert result["tags"]["golden.dataset_checksum"] == lineage["dataset_checksum"]
    assert result["tags"]["golden.manifest_checksum"] == lineage["manifest_checksum"]
    assert result["tags"]["golden.checkpoint_schema"] == "golden-image-classifier.v1"
    assert len(result["tags"]["golden.checkpoint_sha256"]) == 64
    assert result["tags"]["golden.dvc_pipeline"] == DVC_PIPELINE
    assert result["tags"]["golden.dvc_stage"] == DVC_STAGE
    assert set(result["artifacts"]) == REQUIRED_ARTIFACTS

    client = MlflowClient(tracking_uri=result["tracking_uri"])
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    assert experiment is not None
    assert client.get_run(result["run_id"]).info.status == "FINISHED"


def test_latest_run_lookup_uses_isolated_temporary_mlflow_storage(tmp_path):
    dataset_dir, artifact_dir, training = _training_evidence(tmp_path)
    first_storage = tmp_path / "first-tracking"
    second_storage = tmp_path / "second-tracking"
    first = log_golden_mlflow_run(
        artifact_dir, dataset_dir, float(training["duration_seconds"]), first_storage
    )
    second = log_golden_mlflow_run(
        artifact_dir, dataset_dir, float(training["duration_seconds"]), second_storage
    )

    assert first["run_id"] != second["run_id"]
    assert first["tracking_uri"] != second["tracking_uri"]
    assert verify_golden_mlflow_run(
        artifact_dir, dataset_dir, first_storage
    )["run_id"] == first["run_id"]
    assert verify_golden_mlflow_run(
        artifact_dir, dataset_dir, second_storage
    )["run_id"] == second["run_id"]


def test_tracking_rejects_lineage_tampered_after_checkpoint_creation(tmp_path):
    dataset_dir, artifact_dir, training = _training_evidence(tmp_path)
    lineage_path = Path(training["lineage_path"])
    lineage = json.loads(lineage_path.read_text())
    lineage["dataset_checksum"] = "0" * 64
    lineage_path.write_text(json.dumps(lineage))

    with pytest.raises(GoldenMLflowError, match="conflicts with lineage.json"):
        log_golden_mlflow_run(artifact_dir, dataset_dir, 0.1, tmp_path / "tracking")


def test_verifier_rejects_conflicting_mlflow_checkpoint_tag(tmp_path):
    dataset_dir, artifact_dir, training = _training_evidence(tmp_path)
    result = log_golden_mlflow_run(
        artifact_dir,
        dataset_dir,
        float(training["duration_seconds"]),
        tmp_path / "tracking",
    )
    client = MlflowClient(tracking_uri=result["tracking_uri"])
    client.set_tag(result["run_id"], "golden.checkpoint_sha256", "0" * 64)

    with pytest.raises(GoldenMLflowError, match="checkpoint_sha256.*conflicts"):
        verify_golden_mlflow_run(
            artifact_dir,
            dataset_dir,
            tmp_path / "tracking",
            run_id=result["run_id"],
        )


def test_file_backed_training_cli_always_tracks_and_verifies_mlflow(tmp_path):
    dataset_dir = tmp_path / "dataset"
    prepare_golden_dataset(
        dataset_dir,
        DatasetConfig(seed=17, train_samples=16, validation_samples=8),
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "golden_train.py"),
            "--dataset-dir",
            str(dataset_dir),
            "--manifest",
            str(dataset_dir / "manifest.json"),
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--seed",
            "17",
            "--epochs",
            "1",
            "--train-samples",
            "16",
            "--validation-samples",
            "8",
            "--batch-size",
            "4",
            "--learning-rate",
            "0.05",
            "--mlflow-storage-dir",
            str(tmp_path / "tracking"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    result = json.loads(completed.stdout.splitlines()[-1])
    assert result["status"] == "succeeded"
    assert result["mlflow"]["status"] == "verified"
    assert result["mlflow"]["experiment_name"] == EXPERIMENT_NAME


def test_dvc_training_stage_declares_local_mlflow_dependency_without_tracking_outputs():
    pipeline = yaml.safe_load((PROJECT_DIR / "dvc.yaml").read_text())
    stage = pipeline["stages"]["train_golden"]

    assert "golden_mlflow.py" in stage["deps"]
    assert "--mlflow-storage-dir .mlflow" in stage["cmd"]
    assert all(".mlflow" not in output for output in stage["outs"])
