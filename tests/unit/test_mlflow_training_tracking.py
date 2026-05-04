import mlflow
from mlflow.tracking import MlflowClient

import mcp_mlops_tools


def test_track_training_in_mlflow_logs_captured_training_evidence(tmp_path):
    tracking_uri = (tmp_path / "mlruns").as_uri()
    log_path = tmp_path / "train.log"
    checkpoint_path = tmp_path / "checkpoints" / "epoch=1.ckpt"
    log_path.write_text("val_accuracy=0.91\n")
    checkpoint_path.parent.mkdir()
    checkpoint_path.write_bytes(b"checkpoint")

    result = mcp_mlops_tools.track_training_in_mlflow(
        project_path=str(tmp_path),
        experiment_name="phase3-train",
        tracking_uri=tracking_uri,
        params={
            "hydra_overrides": ["trainer.max_epochs=1"],
            "training_controls": {"timeout_seconds": 30, "device": "cpu"},
        },
        metrics={"val_accuracy": 0.91, "train_loss": 0.12},
        artifacts=[str(log_path), str(checkpoint_path)],
        checkpoint_path=str(checkpoint_path),
    )

    assert result["success"] is True
    assert result["tracking_uri"] == tracking_uri
    assert result["run_id"]
    assert result["artifact_uri"]
    assert result["verification_results"][0]["check_name"] == "mlflow_run_exists"
    assert result["verification_results"][0]["evidence_type"] == "observed"
    assert result["artifact_manifest"]["entries"][0]["uri"].startswith(result["artifact_uri"])

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(result["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.params["training_controls.timeout_seconds"] == "30"
    assert run.data.params["hydra_overrides.0"] == "trainer.max_epochs=1"
    assert run.data.metrics["val_accuracy"] == 0.91
    assert {artifact.path for artifact in client.list_artifacts(result["run_id"])} == {
        "artifacts"
    }


def test_track_training_in_mlflow_fails_without_metrics_or_artifacts(tmp_path):
    missing_metrics = mcp_mlops_tools.track_training_in_mlflow(
        project_path=str(tmp_path),
        experiment_name="phase3-train",
        tracking_uri=(tmp_path / "mlruns-missing-metrics").as_uri(),
        metrics={},
        artifacts=[str(tmp_path)],
    )

    assert missing_metrics["success"] is False
    assert "metrics" in missing_metrics["error"]

    missing_artifacts = mcp_mlops_tools.track_training_in_mlflow(
        project_path=str(tmp_path),
        experiment_name="phase3-train",
        tracking_uri=(tmp_path / "mlruns-missing-artifacts").as_uri(),
        metrics={"val_accuracy": 0.91},
        artifacts=[],
    )

    assert missing_artifacts["success"] is False
    assert "artifact" in missing_artifacts["error"]
