#!/usr/bin/env python3
"""
Tests for agent/agent_loop.py - MLOps agent orchestration loop.

Run with: pytest tests/unit/test_agent_loop.py -v
"""

import json
import pickle
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import mcp_mlops_tools
from agent.agent_loop import (
    AgentLoop,
    Route,
    StepExecutionError,
    StepExecutionTracker,
    StepType,
    run_mlops_agent,
)
from mcp_mlops_tools import (
    _find_available_port,
    create_litserve_api,
    select_best_model_artifact,
)
from mcp_mlops_tools import (
    test_litserve_prediction_endpoint as call_litserve_prediction_endpoint,
)
from workflow.registry import (
    ApprovalRecord,
    ArtifactManifest,
    ArtifactManifestEntry,
    VerificationResult,
    WorkflowStatus,
)


def _write_tiny_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")


def test_select_best_model_artifact_finds_training_output_pickle(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "model.pkl").write_bytes(b"pickle-model")

    result = select_best_model_artifact(str(tmp_path))

    assert result["success"] is True
    assert result["model_path"] == "outputs/model.pkl"
    assert result["model_type"] == "tabular_regressor"
    assert result["artifact_manifest"]["entries"][0]["path"] == "outputs/model.pkl"


def test_detect_capstone_data_layouts_blocks_empty_class_folder(tmp_path):
    dataset_1 = tmp_path / "source_one"
    dataset_2 = tmp_path / "source_two"
    _write_tiny_image(dataset_1 / "cats" / "cat-1.jpg")
    (dataset_2 / "empty_class").mkdir(parents=True)

    result = mcp_mlops_tools.detect_capstone_data_layouts(
        project_path=str(tmp_path),
        dataset_1_path=str(dataset_1),
        dataset_2_path=str(dataset_2),
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["datasets"][0]["status"] == "succeeded"
    assert result["datasets"][1]["status"] == "blocked"
    assert result["datasets"][1]["blocked_reason"] == "empty_class_folder"
    assert result["datasets"][1]["missing_inputs"] == ["non_empty_class_folders"]
    layout_result = next(
        item
        for item in result["verification_results"]
        if item["check_name"] == "two_dataset_layouts_supported"
    )
    assert layout_result["passed"] is False
    assert "empty_class_folder" in layout_result["evidence"]


def test_generate_capstone_split_manifests_writes_deterministic_manifest(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "source_one"
    dataset_2 = tmp_path / "source_two"
    for class_name in ("cats", "dogs"):
        for index in range(5):
            _write_tiny_image(dataset_1 / class_name / f"{class_name}-{index}.jpg")
            _write_tiny_image(dataset_2 / class_name / f"{class_name}-{index}.jpg")
    detection = mcp_mlops_tools.detect_capstone_data_layouts(
        project_path=str(project_path),
        dataset_1_path=str(dataset_1),
        dataset_2_path=str(dataset_2),
    )

    first = mcp_mlops_tools.generate_capstone_split_manifests(
        project_path=str(project_path),
        capstone_data_detection=detection,
        test_size=0.4,
        split_seed=7,
    )
    first_manifest = (
        project_path / first["split_manifests"][0]["split_manifest_path"]
    ).read_text()
    second = mcp_mlops_tools.generate_capstone_split_manifests(
        project_path=str(project_path),
        capstone_data_detection=detection,
        test_size=0.4,
        split_seed=7,
    )

    assert first["success"] is True
    assert first["status"] == "succeeded"
    assert first_manifest == (
        project_path / second["split_manifests"][0]["split_manifest_path"]
    ).read_text()
    manifest = json.loads(first_manifest)
    assert manifest["dataset_id"] == "dataset_1"
    assert manifest["source_path"] == str(dataset_1)
    assert manifest["split_strategy"] == "manifest"
    assert manifest["seed"] == 7
    assert manifest["test_size"] == 0.4
    assert manifest["train_count"] == 6
    assert manifest["test_count"] == 4
    assert manifest["per_class_counts"] == {
        "cats": {"train": 3, "test": 2, "total": 5},
        "dogs": {"train": 3, "test": 2, "total": 5},
    }
    assert [item["class_name"] for item in manifest["files"]["train"]] == [
        "cats",
        "cats",
        "cats",
        "dogs",
        "dogs",
        "dogs",
    ]
    assert first["artifact_manifest"]["entries"][0]["artifact_type"] == "split_manifest"
    verification_by_name = {
        item["check_name"]: item for item in first["verification_results"]
    }
    assert verification_by_name["split_evidence_recorded"]["passed"] is True
    assert verification_by_name["dataset_lineage_artifacts_reported"]["passed"] is True
    assert (dataset_1 / "cats" / "cats-0.jpg").exists()
    assert not (project_path / "data" / "capstone" / "dataset_1" / "train").exists()


def test_track_capstone_data_package_adds_split_manifests_to_dvc(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    dvc_config = project_path / ".dvc" / "config"
    dvc_config.parent.mkdir()
    dvc_config.write_text("[core]\n")
    manifest_path = project_path / "data" / "capstone" / "dataset_1" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"dataset_id": "dataset_1"}\n')
    source_dataset = tmp_path / "source_one"
    _write_tiny_image(source_dataset / "cats" / "cat-1.jpg")
    commands = []

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        if cmd == ["dvc", "add", "data/capstone/dataset_1"]:
            (project_path / "data" / "capstone" / "dataset_1.dvc").write_text(
                "outs:\n- path: data/capstone/dataset_1\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)

    result = mcp_mlops_tools.track_capstone_data_package(
        project_path=str(project_path),
        capstone_split_manifest_result={
            "status": "succeeded",
            "split_manifests": [
                {
                    "dataset_id": "dataset_1",
                    "source_path": str(source_dataset),
                    "split_strategy": "manifest",
                    "split_manifest_path": "data/capstone/dataset_1/split_manifest.json",
                    "materialized_train_path": None,
                    "materialized_test_path": None,
                }
            ],
        },
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert commands == [(["dvc", "add", "data/capstone/dataset_1"], str(project_path), 300)]
    assert result["dvc_repo"]["status"] == "validated"
    assert result["tracked_package_paths"] == ["data/capstone/dataset_1"]
    assert result["dvc_tracking_files"] == ["data/capstone/dataset_1.dvc"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["dvc_repo_validated"]["passed"] is True
    assert verification_by_name["capstone_data_package_tracked"]["passed"] is True
    entries = result["artifact_manifest"]["entries"]
    assert {
        (entry["artifact_type"], entry["state"], entry["path"])
        for entry in entries
    } >= {
        ("capstone_source_dataset", "external", str(source_dataset)),
        ("split_manifest", "generated", "data/capstone/dataset_1/split_manifest.json"),
        ("capstone_data_package", "generated", "data/capstone/dataset_1"),
        ("dvc_tracking_file", "generated", "data/capstone/dataset_1.dvc"),
    }


def test_track_capstone_data_package_blocks_when_dvc_is_missing(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    manifest_path = project_path / "data" / "capstone" / "dataset_1" / "split_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"dataset_id": "dataset_1"}\n')

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: False)

    result = mcp_mlops_tools.track_capstone_data_package(
        project_path=str(project_path),
        capstone_split_manifest_result={
            "status": "succeeded",
            "split_manifests": [
                {
                    "dataset_id": "dataset_1",
                    "source_path": str(tmp_path / "source_one"),
                    "split_strategy": "manifest",
                    "split_manifest_path": "data/capstone/dataset_1/split_manifest.json",
                }
            ],
        },
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["dvc_repo"]["status"] == "missing_executable"
    assert "Install DVC" in " ".join(result["next_actions"])
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["dvc_repo_validated"]["passed"] is False
    assert verification_by_name["capstone_data_package_tracked"]["passed"] is False
    assert not (project_path / ".dvc").exists()


def test_configure_validate_capstone_dvc_remote_configures_local_remote(
    tmp_path, monkeypatch
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    dvc_config = project_path / ".dvc" / "config"
    dvc_config.parent.mkdir()
    dvc_config.write_text("[core]\n")
    local_remote = tmp_path / "dvc-remote"
    local_remote.mkdir()
    commands = []

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        if cmd[:3] == ["dvc", "remote", "add"]:
            dvc_config.write_text(
                '[core]\n    remote = capstone\n[\'remote "capstone"\']\n'
                f"    url = {local_remote}\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)

    result = mcp_mlops_tools.configure_validate_capstone_dvc_remote(
        project_path=str(project_path),
        completion_mode="local_ready",
        remote_name="capstone",
        remote_url=str(local_remote),
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert result["remote"]["remote_type"] == "local"
    assert commands == [
        (
            ["dvc", "remote", "add", "-d", "capstone", str(local_remote)],
            str(project_path),
            60,
        )
    ]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["local_dvc_remote_validated"]["passed"] is True
    manifest_entry = result["artifact_manifest"]["entries"][0]
    assert manifest_entry["artifact_type"] == "capstone_data_remote"
    assert manifest_entry["state"] == "validated"
    assert manifest_entry["metadata"]["remote_type"] == "local"


def test_configure_validate_capstone_dvc_remote_blocks_missing_aws_credentials(
    tmp_path, monkeypatch
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    dvc_config = project_path / ".dvc" / "config"
    dvc_config.parent.mkdir()
    dvc_config.write_text("[core]\n")
    secret_url = "s3://secret-capstone-bucket/path?AWS_SECRET_ACCESS_KEY=do-not-record"

    def fake_run_command(cmd, cwd=None, timeout=60):
        if cmd[:3] == ["dvc", "remote", "add"]:
            dvc_config.write_text(
                "[core]\n    remote = capstone\n"
                '[\'remote "capstone"\']\n'
                f"    url = {secret_url}\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": False,
            "status": "missing_cloud_credential_capability",
            "identity": None,
            "bucket_reachable": False,
            "next_actions": [
                "Configure AWS credentials outside Auto-MLOps and rerun validation."
            ],
        },
    )

    result = mcp_mlops_tools.configure_validate_capstone_dvc_remote(
        project_path=str(project_path),
        completion_mode="capstone_complete",
        remote_name="capstone",
        remote_url=secret_url,
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["remote"]["remote_type"] == "s3"
    assert "missing_cloud_credential_capability" in serialized
    assert "do-not-record" not in serialized
    assert secret_url not in serialized
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["s3_remote_validated"]["passed"] is False
    assert "Configure AWS credentials" in " ".join(result["next_actions"])


def _approved_transfer_record(step_id, risks):
    return {
        "workflow_run_id": "run-123",
        "step_id": step_id,
        "risk_categories": risks,
        "status": "approved",
        "approver": "ops@example.com",
        "timestamp": "2026-05-05T00:00:00+00:00",
    }


def test_push_capstone_data_blocks_without_approval(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a\n"
    )
    commands = []

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: commands.append(cmd)
        or {"success": True, "stdout": "ok", "stderr": "", "returncode": 0},
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        capstone_dvc_remote_result={
            "status": "succeeded",
            "remote": {
                "remote_name": "capstone",
                "remote_type": "s3",
                "redacted_remote_url": "s3://se***et/te***-a",
            },
        },
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert commands == []
    assert "approval" in " ".join(result["next_actions"])
    verification = result["verification_results"][0]
    assert verification["check_name"] == "s3_transfer_completed"
    assert verification["source_step"] == "push_capstone_data"
    assert verification["passed"] is False


def test_push_capstone_data_denied_approval_blocks_without_transfer(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a\n"
    )
    commands = []

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: commands.append(cmd)
        or {"success": True, "stdout": "ok", "stderr": "", "returncode": 0},
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        approval_record={
            **_approved_transfer_record(
                "push_capstone_data",
                ["uses_cloud_credentials"],
            ),
            "status": "denied",
        },
    )
    evidence = json.loads(result["verification_results"][0]["evidence"])

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert commands == []
    assert evidence["blocked_reason"] == "missing_or_denied_approval"
    assert evidence["approval_record"]["status"] == "denied"
    assert result["verification_results"][0]["passed"] is False


def test_push_capstone_data_records_observed_transfer_evidence(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a/capstone\n"
    )
    commands = []

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        return {
            "success": True,
            "stdout": "2 files pushed to s3://secret-capstone-bucket/team-a/capstone",
            "stderr": "",
            "returncode": 0,
        }

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": True,
            "status": "validated",
            "identity": {"account": "12***12"},
            "bucket_reachable": True,
            "prefix_checked": True,
            "next_actions": [],
        },
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        paths=["data/capstone/dataset_1"],
        approval_record=_approved_transfer_record(
            "push_capstone_data",
            ["uses_cloud_credentials"],
        ),
    )
    serialized = json.dumps(result, sort_keys=True)

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert commands == [
        (
            ["dvc", "push", "-r", "capstone", "data/capstone/dataset_1"],
            str(project_path),
            600,
        )
    ]
    assert "secret-capstone-bucket" not in serialized
    assert "s3://se***et/te***-a/ca***ne" in serialized
    verification = result["verification_results"][0]
    evidence = json.loads(verification["evidence"])
    assert verification["check_name"] == "s3_transfer_completed"
    assert verification["source_step"] == "push_capstone_data"
    assert verification["passed"] is True
    assert evidence["command"] == "dvc push -r capstone data/capstone/dataset_1"
    assert evidence["returncode"] == 0
    assert evidence["paths"] == ["data/capstone/dataset_1"]
    assert result["artifact_manifest"]["entries"][0]["artifact_type"] == "capstone_data_transfer"


def test_pull_capstone_data_blocks_missing_credentials(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a\n"
    )
    commands = []

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: commands.append(cmd)
        or {"success": True, "stdout": "ok", "stderr": "", "returncode": 0},
    )
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": False,
            "status": "missing_cloud_credential_capability",
            "identity": None,
            "bucket_reachable": False,
            "prefix_checked": False,
            "next_actions": [
                "Configure AWS credentials outside Auto-MLOps and rerun validation."
            ],
        },
    )

    result = mcp_mlops_tools.pull_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        approval_record=_approved_transfer_record(
            "pull_capstone_data",
            ["uses_cloud_credentials", "writes_project_files"],
        ),
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert commands == []
    assert "Configure AWS credentials" in " ".join(result["next_actions"])
    assert result["verification_results"][0]["passed"] is False


def test_push_capstone_data_blocks_when_dvc_is_missing(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    commands = []

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: False)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: commands.append(cmd)
        or {"success": True, "stdout": "ok", "stderr": "", "returncode": 0},
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        approval_record=_approved_transfer_record(
            "push_capstone_data",
            ["uses_cloud_credentials"],
        ),
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert commands == []
    assert "Install DVC" in " ".join(result["next_actions"])
    assert result["verification_results"][0]["passed"] is False


def test_push_capstone_data_blocks_without_validated_s3_remote(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text("[core]\n")
    commands = []

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: commands.append(cmd)
        or {"success": True, "stdout": "ok", "stderr": "", "returncode": 0},
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        capstone_dvc_remote_result={
            "status": "blocked",
            "remote": {
                "remote_name": "capstone",
                "remote_type": "missing",
                "redacted_remote_url": None,
            },
        },
        approval_record=_approved_transfer_record(
            "push_capstone_data",
            ["uses_cloud_credentials"],
        ),
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert commands == []
    assert "Validate an S3" in " ".join(result["next_actions"])
    assert result["verification_results"][0]["passed"] is False


def test_push_capstone_data_failed_transfer_records_structured_next_action(
    tmp_path, monkeypatch
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a\n"
    )

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(
        mcp_mlops_tools,
        "run_command",
        lambda cmd, cwd=None, timeout=60: {
            "success": False,
            "stdout": "",
            "stderr": "Access denied for s3://secret-capstone-bucket/team-a",
            "returncode": 1,
        },
    )
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": True,
            "status": "validated",
            "identity": {"account": "12***12"},
            "bucket_reachable": True,
            "prefix_checked": True,
            "next_actions": [],
        },
    )

    result = mcp_mlops_tools.push_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        approval_record=_approved_transfer_record(
            "push_capstone_data",
            ["uses_cloud_credentials"],
        ),
    )
    evidence = json.loads(result["verification_results"][0]["evidence"])

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["verification_results"][0]["passed"] is False
    assert evidence["returncode"] == 1
    assert evidence["blocked_reason"] == "dvc_transfer_failed"
    assert "secret-capstone-bucket" not in json.dumps(result, sort_keys=True)
    assert "Inspect DVC transfer output" in " ".join(result["next_actions"])


def test_pull_capstone_data_records_observed_transfer_evidence(tmp_path, monkeypatch):
    project_path = tmp_path / "project"
    project_path.mkdir()
    (project_path / ".dvc").mkdir()
    (project_path / ".dvc" / "config").write_text(
        "[core]\n    remote = capstone\n"
        '[\'remote "capstone"\']\n'
        "    url = s3://secret-capstone-bucket/team-a/capstone\n"
    )
    commands = []

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        return {
            "success": True,
            "stdout": "1 file pulled",
            "stderr": "",
            "returncode": 0,
        }

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": True,
            "status": "validated",
            "identity": {"account": "12***12"},
            "bucket_reachable": True,
            "prefix_checked": True,
            "next_actions": [],
        },
    )

    result = mcp_mlops_tools.pull_capstone_data(
        project_path=str(project_path),
        remote_name="capstone",
        approval_record=_approved_transfer_record(
            "pull_capstone_data",
            ["uses_cloud_credentials", "writes_project_files"],
        ),
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert commands == [(["dvc", "pull", "-r", "capstone"], str(project_path), 600)]
    verification = result["verification_results"][0]
    evidence = json.loads(verification["evidence"])
    assert verification["source_step"] == "pull_capstone_data"
    assert verification["passed"] is True
    assert evidence["transfer_direction"] == "pull"
    assert result["artifact_manifest"]["entries"][0]["metadata"]["transfer_direction"] == "pull"


@pytest.mark.asyncio
async def test_prepare_capstone_data_local_remote_configuration_requires_write_approval(
    tmp_path,
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    local_remote = tmp_path / "dvc-remote"
    local_remote.mkdir()
    agent = AgentLoop()
    agent._initialize_session("Prepare capstone data", str(project_path), 0.85)
    agent.workflow_selection = agent.workflow_registry.select_workflow("Prepare capstone data")
    agent.ctx.globals["workflow_inputs"] = {
        "project_path": str(project_path),
        "completion_mode": "local_ready",
        "dvc_remote_name": "capstone",
        "dvc_remote_url": str(local_remote),
    }

    validation = await agent._validate_registry_step_approval("configure_validate_dvc_remote")

    assert validation.status is WorkflowStatus.BLOCKED
    assert [risk.value for risk in validation.risk_categories] == ["writes_project_files"]
    assert "approval" in validation.next_action


@pytest.mark.asyncio
async def test_prepare_capstone_data_s3_remote_configuration_requires_cloud_approval(
    tmp_path,
):
    project_path = tmp_path / "project"
    project_path.mkdir()
    agent = AgentLoop()
    agent._initialize_session("Prepare capstone data", str(project_path), 0.85)
    agent.workflow_selection = agent.workflow_registry.select_workflow("Prepare capstone data")
    agent.ctx.globals["workflow_inputs"] = {
        "project_path": str(project_path),
        "completion_mode": "capstone_complete",
        "dvc_remote_name": "capstone",
        "dvc_remote_url": "s3://capstone-bucket/team-a",
    }

    validation = await agent._validate_registry_step_approval("configure_validate_dvc_remote")

    assert validation.status is WorkflowStatus.BLOCKED
    assert [risk.value for risk in validation.risk_categories] == [
        "writes_project_files",
        "uses_cloud_credentials",
    ]
    assert "approval" in validation.next_action


@pytest.mark.asyncio
async def test_prepare_capstone_data_push_requires_cloud_approval(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    agent = AgentLoop()
    agent._initialize_session("Prepare capstone data", str(project_path), 0.85)
    agent.workflow_selection = agent.workflow_registry.select_workflow("Prepare capstone data")
    agent.ctx.globals["workflow_inputs"] = {
        "project_path": str(project_path),
        "completion_mode": "capstone_complete",
        "dvc_transfer_direction": "push",
    }

    validation = await agent._validate_registry_step_approval("push_capstone_data")

    assert validation.status is WorkflowStatus.BLOCKED
    assert [risk.value for risk in validation.risk_categories] == ["uses_cloud_credentials"]
    assert "approval" in validation.next_action


@pytest.mark.asyncio
async def test_prepare_capstone_data_pull_requires_cloud_and_write_approval(tmp_path):
    project_path = tmp_path / "project"
    project_path.mkdir()
    agent = AgentLoop()
    agent._initialize_session("Prepare capstone data", str(project_path), 0.85)
    agent.workflow_selection = agent.workflow_registry.select_workflow("Prepare capstone data")
    agent.ctx.globals["workflow_inputs"] = {
        "project_path": str(project_path),
        "completion_mode": "capstone_complete",
        "dvc_transfer_direction": "pull",
    }

    validation = await agent._validate_registry_step_approval("pull_capstone_data")

    assert validation.status is WorkflowStatus.BLOCKED
    assert [risk.value for risk in validation.risk_categories] == [
        "uses_cloud_credentials",
        "writes_project_files",
    ]
    assert "approval" in validation.next_action


def test_select_best_model_artifact_selects_latest_run_that_beats_baseline(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "baseline.ckpt").write_text("baseline")
    (checkpoints_dir / "latest.ckpt").write_text("latest")

    result = select_best_model_artifact(
        project_path=str(tmp_path),
        latest_run={
            "run_id": "latest-run",
            "metrics": {"accuracy": 0.91},
            "artifact_path": "checkpoints/latest.ckpt",
        },
        baseline={
            "run_id": "baseline-run",
            "metric_value": 0.87,
            "artifact_path": "checkpoints/baseline.ckpt",
        },
        metric_name="accuracy",
        metric_direction="maximize",
        threshold=0.01,
        tie_policy="keep_baseline",
    )

    assert result["success"] is True
    assert result["status"] == "selected_latest"
    assert result["decision"] == "select_latest"
    assert result["model_path"] == "checkpoints/latest.ckpt"
    assert result["source_run_id"] == "latest-run"
    assert result["comparison_result"] == {
        "metric_name": "accuracy",
        "metric_direction": "maximize",
        "baseline_value": 0.87,
        "latest_value": 0.91,
        "threshold": 0.01,
        "improvement": pytest.approx(0.04),
        "tie_policy": "keep_baseline",
    }
    manifest_entry = result["artifact_manifest"]["entries"][0]
    assert manifest_entry["artifact_type"] == "model_artifact"
    assert manifest_entry["state"] == "selected"
    assert manifest_entry["path"] == "checkpoints/latest.ckpt"
    assert manifest_entry["checksum"]
    assert manifest_entry["metadata"] == {
        "source_run_id": "latest-run",
        "metric_name": "accuracy",
        "metric_value": 0.91,
        "comparison_result": result["comparison_result"],
        "decision": "select_latest",
    }
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["model_selection_inputs_present"]["passed"] is True
    assert verification_by_name["model_selection_metric_compared"]["passed"] is True
    assert verification_by_name["model_selection_candidate_artifact_verified"]["passed"] is True
    assert verification_by_name["model_artifact_selected"]["passed"] is True


def test_select_best_model_artifact_keeps_baseline_when_latest_is_worse(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "baseline.ckpt").write_text("baseline")
    (checkpoints_dir / "latest.ckpt").write_text("latest")

    result = select_best_model_artifact(
        project_path=str(tmp_path),
        latest_run={
            "run_id": "latest-run",
            "metrics": {"accuracy": 0.84},
            "artifact_path": "checkpoints/latest.ckpt",
        },
        baseline={
            "run_id": "baseline-run",
            "metric_value": 0.87,
            "artifact_path": "checkpoints/baseline.ckpt",
        },
        metric_name="accuracy",
        metric_direction="maximize",
        threshold=0.01,
        tie_policy="keep_baseline",
    )

    assert result["success"] is True
    assert result["status"] == "kept_baseline"
    assert result["decision"] == "keep_baseline"
    assert result["model_path"] == "checkpoints/baseline.ckpt"
    assert result["source_run_id"] == "baseline-run"
    assert result["discard_reason"]
    assert result["keep_baseline_reason"]
    manifest_entry = result["artifact_manifest"]["entries"][0]
    assert manifest_entry["state"] == "selected"
    assert manifest_entry["path"] == "checkpoints/baseline.ckpt"
    assert manifest_entry["metadata"]["decision"] == "keep_baseline"
    assert manifest_entry["metadata"]["source_run_id"] == "baseline-run"
    assert manifest_entry["metadata"]["comparison_result"]["improvement"] == pytest.approx(-0.03)


def test_select_best_model_artifact_blocks_without_explicit_comparison_inputs(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "latest.ckpt").write_text("latest")

    result = select_best_model_artifact(
        project_path=str(tmp_path),
        latest_run={
            "run_id": "latest-run",
            "metrics": {"accuracy": 0.91},
            "artifact_path": "checkpoints/latest.ckpt",
        },
        metric_name="accuracy",
        threshold=0.01,
        tie_policy="keep_baseline",
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["decision"] == "blocked"
    assert set(result["missing_required_pieces"]) == {"baseline", "metric_direction"}
    assert "Missing model selection inputs" in result["failure_reason"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["model_selection_inputs_present"]["passed"] is False
    assert verification_by_name["model_artifact_selected"]["passed"] is False
    assert result["artifact_manifest"]["entries"] == []


def test_select_best_model_artifact_blocks_without_candidate_artifact(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "baseline.ckpt").write_text("baseline")

    result = select_best_model_artifact(
        project_path=str(tmp_path),
        latest_run={
            "run_id": "latest-run",
            "metrics": {"accuracy": 0.91},
        },
        baseline={
            "run_id": "baseline-run",
            "metric_value": 0.87,
            "artifact_path": "checkpoints/baseline.ckpt",
        },
        metric_name="accuracy",
        metric_direction="maximize",
        threshold=0.01,
        tie_policy="keep_baseline",
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["missing_required_pieces"] == ["candidate_artifact"]
    assert "checkpoint/model artifact" in result["failure_reason"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["model_selection_candidate_artifact_verified"]["passed"] is False
    assert verification_by_name["model_artifact_selected"]["passed"] is False


def test_record_capstone_orchestrator_skeleton_records_deferred_capabilities(tmp_path):
    result = mcp_mlops_tools.record_capstone_orchestrator_skeleton(
        project_path=str(tmp_path),
        selected_model_artifact_path="models/model.pt",
        endpoint_url="http://127.0.0.1:8000",
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["declared_stages"] == [
        "setup",
        "data",
        "train",
        "deploy",
        "monitor",
        "report",
    ]
    assert result["completed_stages"] == []
    assert {
        "setup_pipeline",
        "detect_training_project",
        "train_and_track",
        "deploy_litserve_preflight",
        "deploy_litserve_gpu",
    }.issubset({item["workflow_id"] for item in result["implemented_subworkflows"]})
    assert result["blocked_stages"][0]["capability"] == "train_until_better"
    deferred_names = {item["capability"] for item in result["deferred_capabilities"]}
    assert {
        "S3 DVC remote automation",
        "KServe/Helm/ArgoCD",
        "HuggingFace Spaces",
        "AWS Lambda serverless",
        "stress tests",
        "frontend",
        "final report",
        "video",
    }.issubset(deferred_names)
    assert result["selected_model_artifact"] == {
        "path": "models/model.pt",
        "state": "available",
    }
    assert result["endpoint_evidence"] == {
        "endpoint_url": "http://127.0.0.1:8000",
        "state": "available",
    }
    assert result["next_actions"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["capstone_stage_plan_recorded"]["passed"] is True
    assert verification_by_name["implemented_subworkflows_referenced"]["passed"] is True
    assert verification_by_name["deferred_capabilities_recorded"]["passed"] is True
    assert "capstone_pipeline_ready" not in verification_by_name
    manifest_entry = result["artifact_manifest"]["entries"][0]
    assert manifest_entry["artifact_type"] == "capstone_orchestrator_plan"
    assert manifest_entry["state"] == "generated"
    assert manifest_entry["path"] == ".auto_mlops/capstone/orchestrator_plan.json"
    assert (tmp_path / manifest_entry["path"]).exists()


def test_record_capstone_data_stage_evidence_writes_durable_redacted_artifact(tmp_path):
    evidence_result = mcp_mlops_tools.record_capstone_data_stage_evidence(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "local_ready"},
        capstone_data_detection={
            "status": "succeeded",
            "completion_mode": "local_ready",
            "datasets": [
                {
                    "dataset_id": "dataset_1",
                    "status": "succeeded",
                    "source_path": str(tmp_path / "dataset_one"),
                    "layout": "class_folders",
                    "missing_inputs": [],
                    "next_actions": [],
                    "class_count": 2,
                    "total_image_count": 4,
                },
                {
                    "dataset_id": "dataset_2",
                    "status": "succeeded",
                    "source_path": str(tmp_path / "dataset_two"),
                    "layout": "class_folders",
                    "missing_inputs": [],
                    "next_actions": [],
                    "class_count": 2,
                    "total_image_count": 4,
                },
            ],
        },
        capstone_split_manifest_result={
            "status": "succeeded",
            "split_manifests": [
                {
                    "dataset_id": "dataset_1",
                    "split_strategy": "manifest",
                    "seed": 42,
                    "test_size": 0.2,
                    "train_count": 2,
                    "test_count": 2,
                    "split_manifest_path": "data/capstone/dataset_1/split_manifest.json",
                },
                {
                    "dataset_id": "dataset_2",
                    "split_strategy": "manifest",
                    "seed": 42,
                    "test_size": 0.2,
                    "train_count": 2,
                    "test_count": 2,
                    "split_manifest_path": "data/capstone/dataset_2/split_manifest.json",
                },
            ],
        },
        capstone_data_package_result={
            "status": "succeeded",
            "dvc_repo": {"status": "initialized", "dvc_config_path": ".dvc/config"},
            "tracked_package_paths": ["data/capstone/dataset_1", "data/capstone/dataset_2"],
            "dvc_tracking_files": [
                "data/capstone/dataset_1.dvc",
                "data/capstone/dataset_2.dvc",
            ],
        },
        capstone_data_remote_result={
            "status": "succeeded",
            "remote": {
                "remote_name": "capstone",
                "remote_type": "s3",
                "redacted_remote_url": "s3://se***et/te***-a",
            },
        },
        capstone_data_push_result={
            "status": "succeeded",
            "transfer": {
                "direction": "push",
                "remote": {
                    "remote_name": "capstone",
                    "remote_type": "s3",
                    "redacted_remote_url": "s3://se***et/te***-a",
                },
                "paths": ["data/capstone/dataset_1", "data/capstone/dataset_2"],
            },
        },
        verification_results=[
            {
                "check_name": "two_dataset_paths_provided",
                "evidence_type": "observed",
                "source_step": "prepare_capstone_data_contract",
                "passed": True,
                "evidence": "{}",
            }
        ],
        artifact_manifest={
            "entries": [
                {
                    "artifact_type": "split_manifest",
                    "producing_step": "generate_split_manifests",
                    "state": "generated",
                    "path": "data/capstone/dataset_1/split_manifest.json",
                }
            ]
        },
    )

    evidence_path = tmp_path / ".auto_mlops" / "capstone" / "data_stage_evidence.json"
    evidence = json.loads(evidence_path.read_text())
    assert evidence_result["status"] == "succeeded"
    assert evidence["schema_version"] == "phase4.data_stage_evidence.v1"
    assert evidence["workflow_id"] == "prepare_capstone_data"
    assert evidence["completion_mode"] == "local_ready"
    assert [dataset["status"] for dataset in evidence["datasets"]] == [
        "succeeded",
        "succeeded",
    ]
    assert evidence["dvc"]["remote"]["remote_type"] == "s3"
    assert "secret" not in json.dumps(evidence, sort_keys=True)
    assert {
        (entry["artifact_type"], entry["path"])
        for entry in evidence["artifact_manifest"]["entries"]
    } >= {
        ("data_stage_evidence", ".auto_mlops/capstone/data_stage_evidence.json"),
        ("split_manifest", "data/capstone/dataset_1/split_manifest.json"),
    }
    verification_by_name = {
        item["check_name"]: item for item in evidence_result["verification_results"]
    }
    assert verification_by_name["data_stage_evidence_artifact_reported"]["passed"] is True
    assert verification_by_name["dataset_lineage_artifacts_reported"]["passed"] is True


def _write_capstone_data_stage_evidence(
    project_path,
    *,
    completion_mode="capstone_complete",
    status="succeeded",
):
    evidence_dir = project_path / ".auto_mlops" / "capstone"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "data_stage_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4.data_stage_evidence.v1",
                "workflow_id": "prepare_capstone_data",
                "status": status,
                "completion_mode": completion_mode,
                "datasets": [
                    {"dataset_id": "dataset_1", "status": "succeeded"},
                    {"dataset_id": "dataset_2", "status": "succeeded"},
                ],
                "dvc": {
                    "remote": {"remote_type": "s3"},
                    "transfer": {"status": "succeeded"},
                },
                "blocked_capabilities": [],
                "verification_results": [
                    {
                        "check_name": "data_stage_evidence_artifact_reported",
                        "evidence_type": "observed",
                        "source_step": "record_data_stage_evidence",
                        "passed": True,
                        "evidence": "{}",
                    }
                ],
                "artifact_manifest": {"entries": []},
            },
            sort_keys=True,
        )
    )
    return evidence_path


def test_resolve_capstone_container_upstream_local_ready_uses_local_model_fallback(
    tmp_path,
):
    model_path = tmp_path / "models" / "best.pt"
    model_path.parent.mkdir()
    model_path.write_text("model")

    result = mcp_mlops_tools.resolve_capstone_container_upstream_evidence(
        project_path=str(tmp_path),
        workflow_inputs={
            "completion_mode": "container_local_ready",
            "local_model_artifact_path": "models/best.pt",
        },
    )

    assert result["success"] is True
    assert result["status"] == "resolved"
    assert result["completion_mode"] == "container_local_ready"
    assert result["upstream_evidence"]["local_model_artifact"]["status"] == "resolved"
    assert result["upstream_evidence"]["data_stage"]["status"] == "deferred"
    assert result["upstream_evidence"]["mlflow_best_artifact"]["status"] == "deferred"
    assert {
        item["capability"] for item in result["deferred_capabilities"]
    } >= {
        "data_stage_evidence_artifact_reported",
        "mlflow_best_artifact_verified",
    }
    assert result["workflow_input_overrides"] == {
        "local_model_artifact_available": True,
        "mlflow_best_artifact_available": False,
    }
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["upstream_evidence_resolved"]["passed"] is True
    assert verification_by_name["local_model_artifact_resolved"]["passed"] is True
    assert "data_stage_capstone_complete_verified" not in verification_by_name
    manifest_entries = result["artifact_manifest"]["entries"]
    assert manifest_entries == [
        {
            "artifact_type": "model_artifact",
            "producing_step": "resolve_upstream_container_evidence",
            "state": "selected",
            "path": "models/best.pt",
            "checksum": manifest_entries[0]["checksum"],
            "metadata": {"source": "local_model_artifact_fallback"},
        }
    ]
    assert not (tmp_path / ".auto_mlops" / "capstone" / "container_ci_evidence.json").exists()


def test_resolve_capstone_container_upstream_capstone_complete_blocks_missing_data_stage(
    tmp_path,
):
    result = mcp_mlops_tools.resolve_capstone_container_upstream_evidence(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["upstream_evidence"]["data_stage"]["status"] == "blocked"
    assert result["upstream_evidence"]["mlflow_best_artifact"]["status"] == "blocked"
    assert {
        item["capability"] for item in result["blocked_capabilities"]
    } >= {
        "data_stage_evidence_artifact_reported",
        "mlflow_best_artifact_verified",
        "training_lineage_verified",
    }
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["upstream_evidence_resolved"]["passed"] is False
    assert verification_by_name["data_stage_capstone_complete_verified"]["passed"] is False
    assert verification_by_name["mlflow_best_artifact_verified"]["passed"] is False
    assert verification_by_name["training_lineage_verified"]["passed"] is False
    assert result["artifact_manifest"]["entries"] == []


def test_resolve_capstone_container_upstream_capstone_complete_requires_capstone_data(
    tmp_path,
):
    _write_capstone_data_stage_evidence(tmp_path, completion_mode="local_ready")
    model_path = tmp_path / "checkpoints" / "best.ckpt"
    model_path.parent.mkdir()
    model_path.write_text("checkpoint")
    training_evidence_path = tmp_path / ".auto_mlops" / "capstone" / "training_evidence.json"
    training_evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "phase3.training_evidence.v1",
                "workflow_id": "train_and_track",
                "status": "succeeded",
                "data_stage_evidence": {
                    "path": ".auto_mlops/capstone/data_stage_evidence.json"
                },
                "verification_results": [
                    {"check_name": "mlflow_run_exists", "passed": True},
                    {"check_name": "model_artifact_selected", "passed": True},
                    {"check_name": "training_command_completed", "passed": True},
                ],
                "artifact_manifest": {
                    "entries": [
                        {
                            "artifact_type": "model_artifact",
                            "producing_step": "select_best_model_artifact",
                            "state": "selected",
                            "path": "checkpoints/best.ckpt",
                            "metadata": {
                                "source_run_id": "run-1",
                                "api_token": "SECRET_TOKEN_VALUE",
                            },
                        }
                    ]
                },
            },
            sort_keys=True,
        )
    )

    result = mcp_mlops_tools.resolve_capstone_container_upstream_evidence(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
    )

    assert result["status"] == "blocked"
    assert result["upstream_evidence"]["data_stage"]["completion_mode"] == "local_ready"
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["data_stage_capstone_complete_verified"]["passed"] is False
    assert verification_by_name["mlflow_best_artifact_verified"]["passed"] is True
    assert verification_by_name["training_lineage_verified"]["passed"] is True
    assert "SECRET_TOKEN_VALUE" not in json.dumps(result, sort_keys=True)


def test_resolve_capstone_container_upstream_does_not_infer_from_prose_summary(tmp_path):
    _write_capstone_data_stage_evidence(tmp_path)
    training_evidence_path = tmp_path / ".auto_mlops" / "capstone" / "training_evidence.json"
    training_evidence_path.write_text(
        json.dumps(
            {
                "workflow_id": "train_and_track",
                "status": "succeeded",
                "summary": "MLflow run exists and the best model is checkpoints/best.ckpt.",
                "latest_run": "run-1",
            },
            sort_keys=True,
        )
    )

    result = mcp_mlops_tools.resolve_capstone_container_upstream_evidence(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
    )

    assert result["status"] == "blocked"
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["data_stage_capstone_complete_verified"]["passed"] is True
    assert verification_by_name["mlflow_best_artifact_verified"]["passed"] is False
    assert verification_by_name["training_lineage_verified"]["passed"] is False
    assert result["upstream_evidence"]["mlflow_best_artifact"]["status"] == "blocked"
    assert result["upstream_evidence"]["training_lineage"]["status"] == "blocked"


def test_resolve_capstone_container_upstream_capstone_complete_accepts_structured_training_evidence(
    tmp_path,
):
    _write_capstone_data_stage_evidence(tmp_path)
    model_path = tmp_path / "checkpoints" / "best.ckpt"
    model_path.parent.mkdir()
    model_path.write_text("checkpoint")
    training_evidence_path = tmp_path / ".auto_mlops" / "capstone" / "training_evidence.json"
    training_evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "phase3.training_evidence.v1",
                "workflow_id": "train_and_track",
                "status": "succeeded",
                "data_stage_evidence": {
                    "path": ".auto_mlops/capstone/data_stage_evidence.json"
                },
                "verification_results": [
                    {"check_name": "mlflow_run_exists", "passed": True},
                    {"check_name": "model_artifact_selected", "passed": True},
                    {"check_name": "training_command_completed", "passed": True},
                ],
                "artifact_manifest": {
                    "entries": [
                        {
                            "artifact_type": "mlflow_run",
                            "producing_step": "track_training_in_mlflow",
                            "state": "generated",
                            "uri": "file:///tmp/mlruns/1/run-1/artifacts",
                            "metadata": {"run_id": "run-1"},
                        },
                        {
                            "artifact_type": "model_artifact",
                            "producing_step": "select_best_model_artifact",
                            "state": "selected",
                            "path": "checkpoints/best.ckpt",
                            "metadata": {"source_run_id": "run-1"},
                        },
                    ]
                },
            },
            sort_keys=True,
        )
    )

    result = mcp_mlops_tools.resolve_capstone_container_upstream_evidence(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
    )

    assert result["success"] is True
    assert result["status"] == "resolved"
    assert result["blocked_capabilities"] == []
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["upstream_evidence_resolved"]["passed"] is True
    assert verification_by_name["data_stage_capstone_complete_verified"]["passed"] is True
    assert verification_by_name["mlflow_best_artifact_verified"]["passed"] is True
    assert verification_by_name["training_lineage_verified"]["passed"] is True
    assert {
        (entry["artifact_type"], entry["state"], entry.get("path") or entry.get("uri"))
        for entry in result["artifact_manifest"]["entries"]
    } >= {
        ("data_stage_evidence", "validated", ".auto_mlops/capstone/data_stage_evidence.json"),
        ("model_artifact", "selected", "checkpoints/best.ckpt"),
    }


def _evidence_by_check(result: dict, check_name: str) -> dict:
    raw_evidence = next(
        item["evidence"]
        for item in result["verification_results"]
        if item["check_name"] == check_name
    )
    return json.loads(raw_evidence)


def test_generate_validate_capstone_runtime_image_spec_detects_dependency_priority(
    tmp_path,
):
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")
    (tmp_path / "requirements.txt").write_text("torch\n")
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
    )

    assert result["success"] is True
    assert result["status"] == "validated"
    dependency_context = _evidence_by_check(result, "dependency_context_reported")
    assert dependency_context["selected_dependency_source"] == "uv_or_pyproject"
    assert dependency_context["dependency_files"] == [
        "uv.lock",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
    ]
    assert dependency_context["install_strategy"] == "uv_sync_frozen"


def test_generate_validate_capstone_runtime_image_spec_references_existing_dockerfile(
    tmp_path,
):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY requirements.txt ./requirements.txt\n"
        "RUN python -m pip install -r requirements.txt\n"
    )
    (tmp_path / "requirements.txt").write_text("pytest\n")

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
    )

    assert result["status"] == "validated"
    assert dockerfile.read_text().startswith("FROM python:3.11-slim")
    build_spec = _evidence_by_check(result, "container_build_spec_reported")
    assert build_spec["action"] == "validated_existing"
    assert build_spec["build_spec_path"] == "Dockerfile"
    assert build_spec["base_image_decision"]["selected_base_image"] == "python:3.11-slim"
    assert build_spec["intended_roles"] == [
        "ci",
        "training_validation",
        "inference_validation",
    ]
    assert result["artifact_manifest"]["entries"] == [
        {
            "artifact_type": "container_build_spec",
            "producing_step": "generate_validate_runtime_image_spec",
            "state": "validated",
            "path": "Dockerfile",
            "metadata": {
                "action": "validated_existing",
                "dependency_source": "requirements",
                "intended_roles": [
                    "ci",
                    "training_validation",
                    "inference_validation",
                ],
            },
        }
    ]


def test_generate_validate_capstone_runtime_image_spec_blocks_write_without_approval(
    tmp_path,
):
    (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
    )

    assert result["status"] == "blocked"
    assert not (tmp_path / "Dockerfile").exists()
    assert result["blocked_capabilities"] == [
        {
            "capability": "write_container_build_spec",
            "reason": "Writing Dockerfile requires an Approval Gate.",
            "required_risk_categories": ["writes_project_files"],
            "next_action": (
                "Record approval for generate_validate_runtime_image_spec before writing Dockerfile."
            ),
        }
    ]
    build_spec = _evidence_by_check(result, "container_build_spec_reported")
    assert build_spec["action"] == "blocked_write_requires_approval"
    assert build_spec["approval_required"]["risk_categories"] == ["writes_project_files"]


def test_generate_validate_capstone_runtime_image_spec_generates_conservative_dockerfile_with_approval(
    tmp_path,
):
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n")

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
        approval_record={
            "workflow_run_id": "run-123",
            "step_id": "generate_validate_runtime_image_spec",
            "risk_categories": ["writes_project_files"],
            "status": "approved",
            "approver": "tester",
            "timestamp": "2026-05-05T00:00:00+00:00",
        },
    )

    dockerfile_text = (tmp_path / "Dockerfile").read_text()
    assert result["status"] == "generated"
    assert dockerfile_text.startswith("FROM python:3.11-slim\n")
    assert "nvidia/cuda" not in dockerfile_text
    assert "COPY .env" not in dockerfile_text
    assert "/home/" not in dockerfile_text
    assert "uv sync --frozen" in dockerfile_text
    build_spec = _evidence_by_check(result, "container_build_spec_reported")
    assert build_spec["action"] == "generated"
    assert build_spec["bounded_commands"] == [
        "python -m pytest tests -q",
        "python - <<'PY'\nfrom pathlib import Path\nassert Path('/app').exists()\nPY",
    ]
    assert build_spec["approval_record"]["status"] == "approved"
    secret_safety = _evidence_by_check(result, "secret_safety_validated")
    assert secret_safety == {
        "passed": True,
        "checked_fields": [
            "build_spec_path",
            "base_image",
            "dependency_files",
            "bounded_commands",
            "dockerfile_content",
        ],
        "violations": [],
    }


def test_generate_validate_capstone_runtime_image_spec_reports_structured_secret_safety(
    tmp_path,
):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    (tmp_path / ".env").write_text("AWS_SECRET_ACCESS_KEY=do-not-copy\n")
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "ENV AWS_SECRET_ACCESS_KEY=do-not-copy\n"
    )

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
    )

    secret_safety = _evidence_by_check(result, "secret_safety_validated")
    assert result["status"] == "blocked"
    assert secret_safety["passed"] is False
    assert secret_safety["violations"] == [
        {
            "field": "dockerfile_content",
            "rule": "secret_like_key",
            "match": "AWS_SECRET_ACCESS_KEY",
        }
    ]
    assert "do-not-copy" not in json.dumps(result)


def test_generate_validate_capstone_runtime_image_spec_rejects_absolute_dataset_paths(
    tmp_path,
):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\nCOPY /home/ubuntu/source-data /app/data\n"
    )

    result = mcp_mlops_tools.generate_validate_capstone_runtime_image_spec(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
    )

    secret_safety = _evidence_by_check(result, "secret_safety_validated")
    assert result["status"] == "blocked"
    assert secret_safety["violations"] == [
        {
            "field": "dockerfile_content",
            "rule": "absolute_source_dataset_path",
            "match": "<redacted-path>",
        }
    ]


def test_build_smoke_check_capstone_container_image_defers_local_ready_without_docker(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: None)

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
        capstone_runtime_image_spec_result={"status": "validated"},
    )

    assert result["success"] is True
    assert result["status"] == "deferred"
    assert result["container"]["docker"]["available"] is False
    assert result["blocked_capabilities"] == []
    assert {
        item["capability"] for item in result["deferred_capabilities"]
    } == {"image_build_deferred_reported"}
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["docker_availability_reported"]["passed"] is True
    assert verification_by_name["image_build_deferred_reported"]["passed"] is True
    assert result["artifact_manifest"]["entries"] == []


def test_build_smoke_check_capstone_container_image_blocks_capstone_complete_without_docker(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: None)

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
        capstone_runtime_image_spec_result={"status": "validated"},
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["container"]["docker"]["available"] is False
    assert {
        item["capability"] for item in result["blocked_capabilities"]
    } == {"docker_available"}
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["docker_available"]["passed"] is False


def test_build_smoke_check_capstone_container_image_requires_build_approval(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(
        mcp_mlops_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Docker version 25.0.0", stderr=""
        ),
    )

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_local_ready"},
        capstone_runtime_image_spec_result={"status": "validated"},
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["container"]["docker"]["available"] is True
    assert result["container"]["image_build"]["attempted"] is False
    assert result["blocked_capabilities"] == [
        {
            "capability": "image_build_approved",
            "reason": "Docker image build requires an Approval Gate.",
            "required_risk_categories": ["builds_image"],
            "next_action": "Record approval for build_smoke_check_container_image before running docker build.",
        }
    ]


def test_build_smoke_check_capstone_container_image_records_successful_build_and_smoke(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: "/usr/bin/docker")
    commands: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[:2] == ["docker", "version"]:
            return SimpleNamespace(returncode=0, stdout="Docker version 25.0.0", stderr="")
        if cmd[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=0, stdout="Successfully built abc123", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="sha256:abc123\n", stderr="")
        if cmd[:2] == ["docker", "run"]:
            return SimpleNamespace(returncode=0, stdout="smoke ok", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(mcp_mlops_tools.subprocess, "run", fake_run)

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={
            "completion_mode": "container_capstone_complete",
            "image_name": "capstone-runtime",
            "image_tag": "test",
        },
        capstone_runtime_image_spec_result={"status": "validated"},
        approval_record={
            "step_id": "build_smoke_check_container_image",
            "risk_categories": ["builds_image"],
            "status": "approved",
        },
        smoke_approval_record={
            "step_id": "build_smoke_check_container_image",
            "risk_categories": ["executes_project_code"],
            "status": "approved",
        },
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert ["docker", "build", "--pull=false", "-f", "Dockerfile", "-t", "capstone-runtime:test", "."] in commands
    assert result["container"]["image_build"]["image_tag"] == "capstone-runtime:test"
    assert result["container"]["image_build"]["image_id"] == "sha256:abc123"
    assert result["container"]["image_build"]["return_code"] == 0
    assert "duration_seconds" in result["container"]["image_build"]
    assert result["container"]["image_build"]["stdout_summary"] == "Successfully built abc123"
    assert result["container"]["smoke_check"]["passed"] is True
    assert result["container"]["smoke_check"]["return_code"] == 0
    assert "duration_seconds" in result["container"]["smoke_check"]
    assert result["container"]["smoke_check"]["stdout_summary"] == "smoke ok"
    assert not any("/health" in " ".join(command) for command in commands)
    assert not any("/predict" in " ".join(command) for command in commands)
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["docker_available"]["passed"] is True
    assert verification_by_name["image_build_succeeded"]["passed"] is True
    assert verification_by_name["container_smoke_check_passed"]["passed"] is True
    assert result["artifact_manifest"]["entries"] == [
        {
            "artifact_type": "container_image",
            "producing_step": "build_smoke_check_container_image",
            "state": "external",
            "uri": "docker://capstone-runtime:test",
            "metadata": {
                "image_id": "sha256:abc123",
                "image_tag": "capstone-runtime:test",
            },
        }
    ]


def test_build_smoke_check_capstone_container_image_requires_smoke_approval(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "version"]:
            return SimpleNamespace(returncode=0, stdout="Docker version 25.0.0", stderr="")
        if cmd[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=0, stdout="Successfully built abc123", stderr="")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="sha256:abc123\n", stderr="")
        if cmd[:2] == ["docker", "run"]:
            raise AssertionError("smoke check must not run without executes_project_code approval")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(mcp_mlops_tools.subprocess, "run", fake_run)

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
        capstone_runtime_image_spec_result={"status": "validated"},
        approval_record={
            "step_id": "build_smoke_check_container_image",
            "risk_categories": ["builds_image"],
            "status": "approved",
        },
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert result["container"]["smoke_check"]["attempted"] is False
    assert result["blocked_capabilities"] == [
        {
            "capability": "container_smoke_check_approved",
            "reason": "Container smoke check executes project code and requires approval.",
            "required_risk_categories": ["executes_project_code"],
            "next_action": (
                "Record approval for build_smoke_check_container_image before running container smoke check."
            ),
        }
    ]
    assert "container_smoke_check_passed" not in {
        item["check_name"] for item in result["verification_results"]
    }


def test_build_smoke_check_capstone_container_image_fails_from_structured_build_evidence(
    tmp_path, monkeypatch
):
    (tmp_path / "Dockerfile").write_text("FROM python:3.11-slim\n")
    monkeypatch.setattr(mcp_mlops_tools.shutil, "which", lambda name: "/usr/bin/docker")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "version"]:
            return SimpleNamespace(returncode=0, stdout="Docker version 25.0.0", stderr="")
        if cmd[:2] == ["docker", "build"]:
            return SimpleNamespace(returncode=17, stdout="build output", stderr="build failed")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(mcp_mlops_tools.subprocess, "run", fake_run)

    result = mcp_mlops_tools.build_smoke_check_capstone_container_image(
        project_path=str(tmp_path),
        workflow_inputs={"completion_mode": "container_capstone_complete"},
        capstone_runtime_image_spec_result={"status": "validated"},
        approval_record={
            "step_id": "build_smoke_check_container_image",
            "risk_categories": ["builds_image"],
            "status": "approved",
        },
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["container"]["image_build"]["return_code"] == 17
    assert result["container"]["image_build"]["stderr_summary"] == "build failed"
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["image_build_succeeded"]["passed"] is False


def test_record_capstone_orchestrator_skeleton_references_data_stage_evidence(tmp_path):
    evidence_dir = tmp_path / ".auto_mlops" / "capstone"
    evidence_dir.mkdir(parents=True)
    evidence_path = evidence_dir / "data_stage_evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": "phase4.data_stage_evidence.v1",
                "workflow_id": "prepare_capstone_data",
                "status": "succeeded",
                "completion_mode": "local_ready",
                "datasets": [
                    {"dataset_id": "dataset_1", "status": "succeeded"},
                    {"dataset_id": "dataset_2", "status": "succeeded"},
                ],
                "dvc": {
                    "remote": {"remote_type": "missing"},
                    "transfer": {"status": "deferred"},
                },
                "blocked_capabilities": [
                    {
                        "stage": "data",
                        "capability": "s3_transfer_completed",
                        "reason": "S3 transfer evidence is missing for capstone completion.",
                        "later_phase_pointer": "Phase 4 capstone_complete rerun",
                    }
                ],
                "verification_results": [],
                "artifact_manifest": {"entries": []},
            },
            sort_keys=True,
        )
    )

    result = mcp_mlops_tools.record_capstone_orchestrator_skeleton(project_path=str(tmp_path))

    assert "data" in result["completed_stages"]
    assert result["data_stage"]["status"] == "completed"
    assert result["data_stage"]["evidence_artifact"] == (
        ".auto_mlops/capstone/data_stage_evidence.json"
    )
    assert any(
        entry["artifact_type"] == "data_stage_evidence"
        for entry in result["artifact_manifest"]["entries"]
    )
    assert any(
        blocked["capability"] == "s3_transfer_completed"
        for blocked in result["blocked_stages"]
    )


def test_create_litserve_api_generates_tabular_server_for_pickle_artifact(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "model.pkl").write_bytes(b"pickle-model")
    (outputs_dir / "scaler.pkl").write_bytes(b"pickle-scaler")

    result = create_litserve_api(
        project_path=str(tmp_path),
        model_path="outputs/model.pkl",
        model_name="model",
    )

    server_path = tmp_path / "deployment" / "litserve" / "server.py"
    server_code = server_path.read_text()
    assert result["success"] is True
    assert "pickle.load" in server_code
    assert "torch.jit.load" not in server_code
    assert "outputs/scaler.pkl" in server_code
    assert "_litserve_server._MCP_AVAILABLE = False" in server_code
    assert "ModelAPI(max_batch_size=64, batch_timeout=0.05)" in server_code
    assert "server = ls.LitServer(\n        api,\n        accelerator=\"auto\",\n        workers_per_device=4" in server_code
    assert "array = self.scaler.transform(array.reshape(1, -1))[0]" in server_code


def test_litserve_prediction_endpoint_uses_artifact_feature_count_for_default_payload(
    tmp_path, monkeypatch
):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "scaler.pkl").write_bytes(
        pickle.dumps(SimpleNamespace(n_features_in_=8))
    )
    captured: dict[str, dict] = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit):
            return b'{"predictions":[0.0]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = call_litserve_prediction_endpoint(str(tmp_path))

    assert result["prediction_passed"] is True
    assert captured["payload"] == {"input": [0.0] * 8}


def test_litserve_health_endpoint_retries_until_ready(monkeypatch):
    attempts = {"count": 0}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, limit):
            return b"ok"

    def fake_urlopen(url, timeout):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise mcp_mlops_tools.urllib.error.URLError("connection refused")
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = mcp_mlops_tools.test_litserve_health_endpoint(
        "/home/ubuntu/Auto-mlops",
        endpoint_url="http://127.0.0.1:8001",
        timeout_seconds=3.0,
    )

    assert result["health_passed"] is True
    assert attempts["count"] == 2


def test_find_available_port_skips_bound_requested_port(monkeypatch):
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_is_port_available",
        lambda host, port: port == 8001,
    )

    selected_port = _find_available_port("127.0.0.1", 8000, attempts=2)
    assert selected_port == 8001


def _write_session_06_training_project(project_path):
    configs = project_path / "configs"
    (configs / "model").mkdir(parents=True)
    (configs / "data").mkdir()
    (project_path / "src" / "models").mkdir(parents=True)
    (project_path / "outputs").mkdir()
    (project_path / "checkpoints").mkdir()
    (project_path / "tests" / "test_train").mkdir(parents=True)
    (project_path / ".dvc").mkdir()

    (configs / "config.yaml").write_text("model:\n  lr: 1e-3\n")
    (configs / "train.yaml").write_text(
        "defaults:\n"
        "  - model: timm_classify\n"
        "  - data: cat_dog\n"
        "trainer:\n"
        "  max_epochs: 5\n"
    )
    (configs / "model" / "timm_classify.yaml").write_text(
        "_target_: src.models.timmclassifier.TimmClassifier\n"
        "model_name: resnet18\n"
    )
    (configs / "data" / "cat_dog.yaml").write_text(
        "_target_: src.data.datamodule.CatDogDataModule\n"
        "data_dir: data/catdog_test\n"
    )
    (project_path / "src" / "train.py").write_text(
        "import hydra\n\n"
        "@hydra.main(version_base='1.3', config_path='../configs', config_name='train.yaml')\n"
        "def main(cfg):\n"
        "    return None\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    (project_path / "src" / "models" / "timmclassifier.py").write_text(
        "import timm\n"
        "import torch\n"
        "import lightning as L\n"
    )
    (project_path / "pyproject.toml").write_text(
        "[project]\n"
        "dependencies = [\n"
        "  'hydra-core==1.3.2',\n"
        "  'lightning==2.5.0',\n"
        "  'timm==1.0.14',\n"
        "  'torch==2.6.0',\n"
        "]\n"
    )
    (project_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n")
    (project_path / "data.dvc").write_text("outs:\n  - path: data/catdog_test\n")
    (project_path / ".dvc" / "config").write_text("[core]\n")
    (project_path / "tests" / "test_train" / "test_training.py").write_text(
        "def test_training_entrypoint_imports():\n"
        "    assert True\n"
    )


def _write_bounded_training_fixture(project_path, script_body):
    _write_session_06_training_project(project_path)
    (project_path / "src" / "train.py").write_text(script_body)


def test_detect_training_project_recognizes_session_06_shape(tmp_path):
    _write_session_06_training_project(tmp_path)

    result = mcp_mlops_tools.detect_training_project(str(tmp_path))

    assert result["success"] is True
    assert result["status"] == "supported"
    assert result["framework_family"] == "pytorch_lightning"
    assert result["model_library"] == "timm"
    assert result["config_system"] == "hydra"
    assert result["data_versioning"] == "dvc"
    assert result["training_entrypoint"] == "src/train.py"
    assert "configs/train.yaml" in result["likely_config_files"]
    assert "tests/test_train/test_training.py" in result["test_files"]
    assert result["missing_required_pieces"] == []
    assert result["next_actions"] == []
    assert result["test_command"] == "python -m pytest tests -q"
    assert {
        "training_entrypoint_detected",
        "hydra_config_detected",
        "dvc_or_data_evidence_detected",
        "pytorch_timm_signals_detected",
        "test_command_detected",
        "output_artifact_candidates_detected",
    }.issubset({item["check_name"] for item in result["verification_results"]})
    manifest_entries = result["artifact_manifest"]["entries"]
    assert {
        (entry["artifact_type"], entry["state"], entry["path"])
        for entry in manifest_entries
    }.issuperset(
        {
            ("training_entrypoint", "external", "src/train.py"),
            ("configuration", "external", "configs/train.yaml"),
            ("test_suite", "external", "tests/test_train/test_training.py"),
        }
    )


def test_detect_training_project_blocks_ambiguous_entrypoints(tmp_path):
    _write_session_06_training_project(tmp_path)
    (tmp_path / "train.py").write_text(
        "import hydra\n"
        "@hydra.main(version_base='1.3', config_path='configs', config_name='train')\n"
        "def main(cfg):\n"
        "    return None\n"
    )

    result = mcp_mlops_tools.detect_training_project(str(tmp_path))

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert set(result["training_entrypoint_candidates"]) == {"src/train.py", "train.py"}
    assert "training_entrypoint" in result["missing_required_pieces"]
    assert "multiple candidates" in " ".join(result["next_actions"])
    assert "training_entrypoint_detected" not in {
        item["check_name"] for item in result["verification_results"]
    }


def test_run_bounded_training_captures_metrics_logs_duration_and_artifacts(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "import json\n"
        "from pathlib import Path\n"
        "Path('checkpoints').mkdir(exist_ok=True)\n"
        "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
        "print(json.dumps({'metrics': {'accuracy': 0.91, 'loss': 0.12}}))\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=["trainer.fast_dev_run=true"],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert result["exit_code"] == 0
    assert result["metrics"]["accuracy"] == 0.91
    assert result["duration_seconds"] >= 0
    assert result["command"][0].endswith("python")
    assert "trainer.max_epochs=1" in result["effective_overrides"]
    assert "device=cpu" in result["effective_overrides"]
    assert "data.subset=4" in result["effective_overrides"]
    assert result["log_path"].endswith("training.log")
    assert result["config_snapshot_path"].endswith("config_snapshot.yaml")
    assert "checkpoints/model.ckpt" in result["checkpoint_artifact_paths"]

    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["bounded_training_controls_present"]["passed"] is True
    assert verification_by_name["bounded_training_command_completed"]["passed"] is True
    assert verification_by_name["training_metric_captured"]["passed"] is True
    assert verification_by_name["training_artifact_captured"]["passed"] is True
    assert verification_by_name["training_run_evidence_captured"]["passed"] is True
    assert {
        (entry["artifact_type"], entry["state"], entry["path"])
        for entry in result["artifact_manifest"]["entries"]
    }.issuperset(
        {
            ("training_log", "generated", result["log_path"]),
            ("config_snapshot", "generated", result["config_snapshot_path"]),
            ("checkpoint_or_model_artifact", "generated", "checkpoints/model.ckpt"),
        }
    )


def test_run_bounded_training_records_nonzero_exit_as_failed(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "import sys\n"
        "print('about to fail')\n"
        "print('boom', file=sys.stderr)\n"
        "raise SystemExit(3)\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=[],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["exit_code"] == 3
    assert "non-zero" in result["failure_reason"]
    assert "about to fail" in result["stdout_summary"]
    assert "boom" in result["stderr_summary"]
    assert result["next_actions"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["bounded_training_command_completed"]["passed"] is False


def test_run_bounded_training_zero_exit_without_metric_or_artifact_does_not_succeed(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "print('finished without metric or artifact')\n",
    )

    result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=[],
        target_metric="accuracy",
    )

    assert result["success"] is True
    assert result["status"] == "failed"
    assert result["exit_code"] == 0
    assert "target metric" in result["failure_reason"]
    assert "checkpoint" in result["failure_reason"]
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["training_metric_captured"]["passed"] is False
    assert verification_by_name["training_artifact_captured"]["passed"] is False


def test_track_training_in_mlflow_logs_bounded_training_evidence_to_local_run(tmp_path):
    _write_bounded_training_fixture(
        tmp_path,
        "import json\n"
        "from pathlib import Path\n"
        "Path('checkpoints').mkdir(exist_ok=True)\n"
        "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
        "print(json.dumps({'metrics': {'accuracy': 0.94, 'loss': 0.08}}))\n",
    )
    training_result = mcp_mlops_tools.run_bounded_training(
        project_path=str(tmp_path),
        training_entrypoint="src/train.py",
        hydra_config_path="configs",
        hydra_config_name="train",
        timeout_seconds=10,
        max_epochs=1,
        device="cpu",
        data_subset=4,
        hydra_overrides=["trainer.fast_dev_run=true"],
        target_metric="accuracy",
    )

    result = mcp_mlops_tools.track_training_in_mlflow(
        project_path=str(tmp_path),
        training_result=training_result,
        experiment_name="issue-0003",
        params={
            "timeout_seconds": 10,
            "max_epochs": 1,
            "device": "cpu",
            "data_subset": 4,
        },
    )

    assert result["success"] is True
    assert result["status"] == "succeeded"
    assert result["experiment_id"]
    assert result["run_id"]
    assert result["tracking_uri"].startswith("file:")
    assert result["artifact_uri"].startswith("file:")
    assert result["run_status"] == "FINISHED"
    assert "trainer.fast_dev_run=true" in result["params"]["effective_overrides"]
    assert "trainer.max_epochs=1" in result["params"]["effective_overrides"]
    assert result["params"]["timeout_seconds"] == "10"
    assert result["metrics"]["accuracy"] == 0.94
    assert "training.log" in result["logged_artifacts"]
    assert result["checkpoint_artifact_uri"].endswith("checkpoints/model.ckpt")

    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    for check_name in (
        "mlflow_experiment_exists",
        "mlflow_run_exists",
        "mlflow_tracking_uri_recorded",
        "mlflow_artifact_uri_recorded",
        "mlflow_params_logged",
        "mlflow_metrics_logged",
        "mlflow_artifacts_logged",
        "mlflow_checkpoint_artifact_logged",
        "mlflow_run_status_recorded",
    ):
        assert verification_by_name[check_name]["passed"] is True
    assert {
        (entry["artifact_type"], entry["state"], entry.get("uri"))
        for entry in result["artifact_manifest"]["entries"]
    }.issuperset(
        {
            ("mlflow_run", "generated", result["artifact_uri"]),
            (
                "mlflow_checkpoint_or_model_artifact",
                "generated",
                result["checkpoint_artifact_uri"],
            ),
        }
    )


def test_track_training_in_mlflow_blocks_remote_tracking_uri(tmp_path):
    _write_session_06_training_project(tmp_path)

    result = mcp_mlops_tools.track_training_in_mlflow(
        project_path=str(tmp_path),
        training_result={
            "status": "succeeded",
            "metrics": {"accuracy": 0.9},
            "target_metric": "accuracy",
            "checkpoint_artifact_paths": ["checkpoints/model.ckpt"],
        },
        experiment_name="issue-0003",
        tracking_uri="https://mlflow.example.com",
        params={
            "timeout_seconds": 10,
            "max_epochs": 1,
            "device": "cpu",
            "data_subset": 4,
        },
    )

    assert result["success"] is True
    assert result["status"] == "blocked"
    assert "Remote MLflow tracking URI is not allowed" in result["failure_reason"]
    assert "local path or file://" in " ".join(result["next_actions"])
    verification_by_name = {
        item["check_name"]: item for item in result["verification_results"]
    }
    assert verification_by_name["mlflow_run_exists"]["passed"] is False


# ============================================================================
# Route Constants Tests
# ============================================================================


class TestRouteConstants:
    """Tests for Route constants."""

    def test_route_summarize(self):
        """Test SUMMARIZE route constant."""
        assert Route.SUMMARIZE == "summarize"

    def test_route_decision(self):
        """Test DECISION route constant."""
        assert Route.DECISION == "decision"

    def test_route_improve(self):
        """Test IMPROVE route constant."""
        assert Route.IMPROVE == "improve"

    def test_route_deploy(self):
        """Test DEPLOY route constant."""
        assert Route.DEPLOY == "deploy"


# ============================================================================
# StepType Constants Tests
# ============================================================================


class TestStepTypeConstants:
    """Tests for StepType constants."""

    def test_step_type_root(self):
        """Test ROOT step type constant."""
        assert StepType.ROOT == "ROOT"

    def test_step_type_code(self):
        """Test CODE step type constant."""
        assert StepType.CODE == "CODE"

    def test_step_type_improve(self):
        """Test IMPROVE step type constant."""
        assert StepType.IMPROVE == "IMPROVE"

    def test_step_type_deploy(self):
        """Test DEPLOY step type constant."""
        assert StepType.DEPLOY == "DEPLOY"


# ============================================================================
# StepExecutionError Tests
# ============================================================================


class TestStepExecutionError:
    """Tests for StepExecutionError exception."""

    def test_create_step_execution_error(self):
        """Test creating StepExecutionError."""
        error = StepExecutionError("step_1", "Something went wrong")
        assert error.step_id == "step_1"
        assert error.error_message == "Something went wrong"
        assert "step_1" in str(error)
        assert "Something went wrong" in str(error)

    def test_step_execution_error_inheritance(self):
        """Test StepExecutionError is an Exception."""
        error = StepExecutionError("step_1", "Error")
        assert isinstance(error, Exception)

    def test_step_execution_error_message_format(self):
        """Test error message format."""
        error = StepExecutionError("test_step", "Test error")
        expected = "Step 'test_step' failed: Test error"
        assert str(error) == expected


# ============================================================================
# StepExecutionTracker Tests
# ============================================================================


class TestStepExecutionTracker:
    """Tests for StepExecutionTracker class."""

    def test_create_tracker_with_defaults(self):
        """Test creating tracker with default values."""
        tracker = StepExecutionTracker()
        assert tracker.max_steps == 15
        assert tracker.max_retries == 3
        assert tracker.tries == 0
        assert tracker.attempts == {}

    def test_create_tracker_with_custom_values(self):
        """Test creating tracker with custom values."""
        tracker = StepExecutionTracker(max_steps=10, max_retries=5)
        assert tracker.max_steps == 10
        assert tracker.max_retries == 5

    def test_increment(self):
        """Test incrementing tries counter."""
        tracker = StepExecutionTracker()
        assert tracker.tries == 0
        tracker.increment()
        assert tracker.tries == 1
        tracker.increment()
        assert tracker.tries == 2

    def test_record_failure(self):
        """Test recording step failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 1
        tracker.record_failure("step_1")
        assert tracker.attempts["step_1"] == 2
        tracker.record_failure("step_2")
        assert tracker.attempts["step_2"] == 1

    def test_retry_step_id_no_failures(self):
        """Test retry step ID with no failures."""
        tracker = StepExecutionTracker()
        assert tracker.retry_step_id("step_1") == "step_1"

    def test_retry_step_id_with_failures(self):
        """Test retry step ID generation with failures."""
        tracker = StepExecutionTracker()
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F1"
        tracker.record_failure("step_1")
        assert tracker.retry_step_id("step_1") == "step_1F2"

    def test_should_continue_within_limits(self):
        """Test should_continue when within limits."""
        tracker = StepExecutionTracker(max_steps=3)
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True
        tracker.increment()
        assert tracker.should_continue() is True

    def test_should_continue_at_limit(self):
        """Test should_continue when at limit."""
        tracker = StepExecutionTracker(max_steps=3)
        tracker.increment()
        tracker.increment()
        tracker.increment()
        assert tracker.should_continue() is False

    def test_has_exceeded_retries_within_limit(self):
        """Test has_exceeded_retries when within limit."""
        tracker = StepExecutionTracker(max_retries=2)
        assert tracker.has_exceeded_retries("step_1") is False
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is False

    def test_has_exceeded_retries_at_limit(self):
        """Test has_exceeded_retries when at limit."""
        tracker = StepExecutionTracker(max_retries=2)
        tracker.record_failure("step_1")
        tracker.record_failure("step_1")
        assert tracker.has_exceeded_retries("step_1") is True

    def test_is_circuit_open_initially_false(self):
        """Test circuit is initially closed."""
        tracker = StepExecutionTracker()
        assert tracker.is_circuit_open() is False

    def test_get_circuit_stats(self):
        """Test getting circuit stats."""
        tracker = StepExecutionTracker()
        stats = tracker.get_circuit_stats()
        assert "state" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "failed_calls" in stats
        assert "rejected_calls" in stats
        assert "consecutive_failures" in stats
        assert "consecutive_successes" in stats
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0

    def test_reset_circuit(self):
        """Test resetting circuit breaker."""
        tracker = StepExecutionTracker()
        tracker.circuit_breaker._stats.total_calls = 100
        tracker.circuit_breaker._stats.failed_calls = 50
        tracker.reset_circuit()
        stats = tracker.get_circuit_stats()
        assert stats["total_calls"] == 0
        assert stats["failed_calls"] == 0

    def test_circuit_breaker_property(self):
        """Test circuit breaker property access."""
        tracker = StepExecutionTracker()
        assert tracker.circuit_breaker is not None
        assert tracker.circuit_breaker.name == "step_execution"


# ============================================================================
# AgentLoop Initialization Tests
# ============================================================================


class TestAgentLoopInit:
    """Tests for AgentLoop initialization."""

    @pytest.fixture
    def mock_prompts_dir(self, tmp_path):
        """Create temporary prompts directory with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text(
            "Improve: target={target_accuracy} current={current_accuracy}"
        )

        return str(prompts_dir)

    def test_create_agent_loop_with_defaults(self, mock_prompts_dir):
        """Test creating AgentLoop with default values."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        assert agent.status == "idle"
        assert agent.profile == "default"
        assert agent.on_event is None
        assert agent.tools_module is None

    def test_create_agent_loop_with_custom_profile(self, mock_prompts_dir):
        """Test creating AgentLoop with custom profile."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir, profile="custom")
        assert agent.profile == "custom"

    def test_create_agent_loop_with_on_event(self, mock_prompts_dir):
        """Test creating AgentLoop with event callback."""

        async def callback(event_type, data):
            pass

        agent = AgentLoop(prompts_dir=mock_prompts_dir, on_event=callback)
        assert agent.on_event is callback

    def test_create_agent_loop_with_tools_module(self, mock_prompts_dir):
        """Test creating AgentLoop with custom tools module."""
        mock_tools = MagicMock()
        agent = AgentLoop(prompts_dir=mock_prompts_dir, tools_module=mock_tools)
        assert agent.tools_module is mock_tools

    def test_load_prompt_returns_content(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns file content."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        prompt_file = tmp_path / "test.txt"
        prompt_file.write_text("Test content")
        result = agent._load_prompt(prompt_file)
        assert result == "Test content"

    def test_load_prompt_missing_file_returns_empty(self, mock_prompts_dir, tmp_path):
        """Test _load_prompt returns empty string for missing file."""
        agent = AgentLoop(prompts_dir=mock_prompts_dir)
        result = agent._load_prompt(tmp_path / "nonexistent.txt")
        assert result == ""


# ============================================================================
# AgentLoop Session Initialization Tests
# ============================================================================


class TestAgentLoopSessionInit:
    """Tests for AgentLoop session initialization."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception: {query}")
        (prompts_dir / "decision_prompt.txt").write_text("Decision: {query}")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_initialize_session_creates_session_id(self, agent):
        """Test that _initialize_session creates a session ID."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session_id is not None
        assert len(agent.session_id) > 0

    def test_initialize_session_sets_query(self, agent):
        """Test that _initialize_session sets the query."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.query == "Test query"

    def test_initialize_session_creates_context(self, agent):
        """Test that _initialize_session creates context manager."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.ctx is not None
        assert agent.ctx.project_path == "/test/path"

    def test_initialize_session_sets_accuracy_threshold(self, agent):
        """Test that _initialize_session sets accuracy threshold."""
        agent._initialize_session("Test query", "/test/path", 0.90)
        assert agent.ctx.experiment_state.target_accuracy == 0.90

    def test_initialize_session_creates_agent_session(self, agent):
        """Test that _initialize_session creates AgentSession."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.session is not None
        assert agent.session.original_query == "Test query"

    def test_initialize_session_initializes_placeholders(self, agent):
        """Test that _initialize_session initializes placeholders."""
        agent._initialize_session("Test query", "/test/path", 0.85)
        assert agent.p_out == {}
        assert agent.code_variants == {}
        assert agent.next_step_id == "0"
        assert agent.final_output == ""


# ============================================================================
# AgentLoop Routing Tests
# ============================================================================


class TestAgentLoopRouting:
    """Tests for AgentLoop routing logic."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_should_summarize_true_when_goal_achieved(self, agent):
        """Test _should_summarize returns True when goal achieved."""
        agent.p_out = {"original_goal_achieved": True, "route": "decision"}
        assert agent._should_summarize() is True

    def test_should_summarize_true_when_route_is_summarize(self, agent):
        """Test _should_summarize returns True when route is summarize."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.SUMMARIZE}
        assert agent._should_summarize() is True

    def test_should_summarize_false_when_not_achieved(self, agent):
        """Test _should_summarize returns False when not achieved."""
        agent.p_out = {"original_goal_achieved": False, "route": Route.DECISION}
        assert agent._should_summarize() is False

    def test_needs_improvement_when_below_threshold(self, agent):
        """Test _needs_improvement returns True when below threshold."""
        agent._initialize_session("Test", "/test", 0.90)
        agent.ctx.experiment_state.current_accuracy = 0.80
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is True

    def test_needs_improvement_false_when_at_threshold(self, agent):
        """Test _needs_improvement returns False when at threshold."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.current_accuracy = 0.85
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_improvement_false_when_no_accuracy(self, agent):
        """Test _needs_improvement returns False when no accuracy recorded."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.experiment_state.stage = "evaluation"
        assert agent._needs_improvement() is False

    def test_needs_deployment_when_route_is_deploy(self, agent):
        """Test _needs_deployment returns True when route is deploy."""
        agent.p_out = {"route": Route.DEPLOY}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_stage_is_deploy(self, agent):
        """Test _needs_deployment returns True when stage is deploy."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "deploy"}
        assert agent._needs_deployment() is True

    def test_needs_deployment_when_deployment_target_set(self, agent):
        """Test _needs_deployment returns True when deployment target set."""
        agent.p_out = {"route": Route.DECISION, "entities": {"deployment_target": "gradio"}}
        assert agent._needs_deployment() is True

    def test_needs_deployment_false_when_not_deploying(self, agent):
        """Test _needs_deployment returns False when not deploying."""
        agent.p_out = {"route": Route.DECISION, "pipeline_stage": "training", "entities": {}}
        assert agent._needs_deployment() is False


# ============================================================================
# AgentLoop Event Emission Tests
# ============================================================================


class TestAgentLoopEventEmission:
    """Tests for AgentLoop event emission."""

    @pytest.fixture
    def agent_with_event_handler(self, tmp_path):
        """Create AgentLoop with event handler."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=capture_event)
        agent._events = events
        return agent

    @pytest.mark.asyncio
    async def test_emit_calls_callback(self, agent_with_event_handler):
        """Test _emit calls the event callback."""
        await agent_with_event_handler._emit("test_event", {"key": "value"})
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["type"] == "test_event"
        assert agent_with_event_handler._events[0]["data"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_with_empty_data(self, agent_with_event_handler):
        """Test _emit with empty data."""
        await agent_with_event_handler._emit("test_event")
        assert len(agent_with_event_handler._events) == 1
        assert agent_with_event_handler._events[0]["data"] == {}

    @pytest.mark.asyncio
    async def test_emit_handles_callback_error(self, tmp_path):
        """Test _emit handles callback error gracefully."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        async def failing_callback(event_type, data):
            raise ValueError("Callback failed")

        agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=failing_callback)
        # Should not raise
        await agent._emit("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_emit_without_callback(self, tmp_path):
        """Test _emit without callback does nothing."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        # Should not raise
        await agent._emit("test_event", {"key": "value"})


# ============================================================================
# AgentLoop Pick Next Step Tests
# ============================================================================


class TestAgentLoopPickNextStep:
    """Tests for AgentLoop _pick_next_step method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    def test_pick_next_step_returns_first_pending(self, agent):
        """Test _pick_next_step returns first pending step."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        result = agent._pick_next_step()
        assert result == "0"

    def test_pick_next_step_skips_completed(self, agent):
        """Test _pick_next_step skips completed steps."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.add_step("1", "Second step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result == "1"

    def test_pick_next_step_returns_none_when_all_completed(self, agent):
        """Test _pick_next_step returns None when all completed."""
        agent._initialize_session("Test", "/test", 0.85)
        agent.ctx.add_step("0", "First step", "CODE", from_node="ROOT")
        agent.ctx.mark_step_completed("0")
        result = agent._pick_next_step()
        assert result is None

    def test_pick_next_step_returns_none_when_no_steps(self, agent):
        """Test _pick_next_step returns None when no steps."""
        agent._initialize_session("Test", "/test", 0.85)
        result = agent._pick_next_step()
        assert result is None


# ============================================================================
# AgentLoop Run Integration Tests
# ============================================================================


class TestAgentLoopRun:
    """Integration tests for AgentLoop.run method."""

    @pytest.fixture
    def mock_agent(self, tmp_path, mock_llm):
        """Create AgentLoop with mocked dependencies."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")

        agent = AgentLoop(prompts_dir=str(prompts_dir))
        return agent

    @pytest.mark.asyncio
    async def test_run_sets_status_to_running(self, mock_agent, mock_llm):
        """Test run sets status to running."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path")
            # Status should be success at the end
            assert mock_agent.status in ("running", "success")

    @pytest.mark.asyncio
    async def test_run_initializes_session(self, mock_agent, mock_llm):
        """Test run initializes session."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Test summary"}
            await mock_agent.run("Test query", "/test/path", 0.85)
            assert mock_agent.session is not None
            assert mock_agent.ctx is not None

    @pytest.mark.asyncio
    async def test_run_returns_summary_when_goal_achieved(self, mock_agent, mock_llm):
        """Test run returns summary when goal achieved."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with patch.object(mock_agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Final summary"}
            result = await mock_agent.run("Test query", "/test/path")
            assert result == "Final summary"
            assert mock_agent.status == "success"

    @pytest.mark.asyncio
    async def test_run_selects_setup_workflow_before_perception_or_decision(self, mock_agent):
        """Test registry workflow selection runs before prompt-authored planning."""
        expected_step_ids = [
            step.step_id for step in mock_agent.workflow_registry.get("setup_pipeline").steps
        ]

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before selection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Set up MLOps for this project", "/test/path")

        assert mock_agent.workflow_selection.workflow_id == "setup_pipeline"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.get_pending_steps() == expected_step_ids
        for step in mock_agent.workflow_registry.get("setup_pipeline").steps:
            runtime_step = mock_agent.ctx.graph.nodes[step.step_id]["data"]
            assert runtime_step.tool in step.tool_functions
        assert "setup_pipeline" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_projects_litserve_gpu_workflow_and_blocks_before_risky_actions(
        self, mock_agent
    ):
        """Test LitServe GPU runtime uses registry steps and approval gates."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        expected_step_ids = [
            step.step_id for step in mock_agent.workflow_registry.get("deploy_litserve_gpu").steps
        ]

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before selection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        approval_events = [event for event in events if event["type"] == "approval_required"]
        assert mock_agent.workflow_selection.workflow_id == "deploy_litserve_gpu"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.get_pending_steps() == expected_step_ids
        assert approval_events
        assert approval_events[0]["data"]["workflow_id"] == "deploy_litserve_gpu"
        assert approval_events[0]["data"]["step_id"] == "detect_gpu_cuda"
        assert approval_events[0]["data"]["risk_categories"] == ["uses_gpu"]
        assert "Approval required" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_blocks_missing_entrypoint_or_config_before_training(
        self, mock_agent, tmp_path
    ):
        """Test training requests use registry detection and block unsupported project shape."""
        project_path = tmp_path / "incomplete-training-project"
        project_path.mkdir()
        (project_path / "pyproject.toml").write_text("[project]\ndependencies = ['torch']\n")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run before training detection"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Detect this training project",
                str(project_path),
            )

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert completed_tool_steps == ["detect_training_project"]
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "training_entrypoint" in result
        assert "hydra_config" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_succeeds_from_supported_detection(
        self, mock_agent, tmp_path
    ):
        """Test supported Phase 3 training projects complete the detection contract."""
        _write_session_06_training_project(tmp_path)

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training detection must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run("Detect this training project", str(tmp_path))

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert completed_tool_steps == ["detect_training_project"]
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in result
        assert "framework_family=pytorch_lightning" in result
        assert "entrypoint=src/train.py" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_detect_training_project_blocks_missing_project_path(self, mock_agent):
        """Test detection workflow blocks before tools when project_path is missing."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while detection is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Detect this training project")

        assert mock_agent.workflow_selection.workflow_id == "detect_training_project"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("project_path",)
        assert "project_path" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_blocks_missing_bounded_controls(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track blocks before tools when bounded controls are missing."""
        _write_session_06_training_project(tmp_path)

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while training is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Train this project", str(tmp_path))

        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert set(mock_agent.workflow_selection.missing_inputs) == {
            "timeout_seconds",
            "max_epochs",
            "device",
            "data_subset",
            "metric_name",
            "metric_direction",
            "threshold",
            "tie_policy",
            "baseline_metric",
            "baseline_artifact_path",
        }
        assert "missing_inputs" in result
        assert "timeout_seconds" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_data_blocks_missing_project_path_and_datasets(
        self, mock_agent
    ):
        """Test prepare_capstone_data blocks before tools when required inputs are absent."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while data prep is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Prepare capstone data")

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_data"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == (
            "project_path",
            "dataset_1_path",
            "dataset_2_path",
        )
        assert "missing_inputs" in result
        assert "dataset_1_path" in result
        assert "dataset_2_path" in result
        assert "next_action" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_data_blocks_invalid_completion_mode(
        self, mock_agent, tmp_path
    ):
        """Test prepare_capstone_data rejects unsupported completion modes before execution."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while data prep is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "Prepare capstone data dataset_1_path=/data/one "
                "dataset_2_path=/data/two completion_mode=production",
                str(tmp_path),
            )

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_data"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("completion_mode",)
        assert "completion_mode" in result
        assert "local_ready" in result
        assert "capstone_complete" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_container_ci_blocks_missing_project_path(
        self, mock_agent
    ):
        """Test prepare_capstone_container_ci blocks before tools without project_path."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError(
                    "perception should not run while container CI prep is blocked"
                ),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("prepare capstone container CI")

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_container_ci"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("project_path",)
        assert "missing_inputs" in result
        assert "project_path" in result
        assert "next_action" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_container_ci_blocks_invalid_completion_mode(
        self, mock_agent, tmp_path
    ):
        """Test prepare_capstone_container_ci rejects unsupported completion modes."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError(
                    "perception should not run while container CI prep is blocked"
                ),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "prepare capstone container CI completion_mode=production",
                str(tmp_path),
            )

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_container_ci"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("completion_mode",)
        assert "completion_mode" in result
        assert "container_local_ready" in result
        assert "container_capstone_complete" in result
        assert mock_agent.ctx.get_pending_steps() == []
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_container_ci_resolves_upstream_then_blocks_later_issues(
        self, mock_agent, tmp_path
    ):
        """Test Issue 2 runtime resolves upstream evidence before later Phase 5 work."""
        model_path = tmp_path / "models" / "best.pt"
        model_path.parent.mkdir()
        model_path.write_text("model")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry container CI prep must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "prepare capstone container CI completion_mode=container_local_ready "
                "local_model_artifact_path=models/best.pt",
                str(tmp_path),
            )

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_container_ci"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.globals["workflow_inputs"].items() >= {
            "project_path": str(tmp_path),
            "completion_mode": "container_local_ready",
            "data_stage_evidence_path": None,
            "local_model_artifact_path": "models/best.pt",
            "mlflow_run_id": None,
            "mlflow_best_artifact_path": None,
            "registry_target": None,
            "image_name": None,
            "image_tag": None,
            "ci_workflow_path": None,
            "local_model_artifact_available": True,
            "mlflow_best_artifact_available": False,
        }.items()
        assert isinstance(mock_agent.ctx.globals["workflow_inputs"]["docker_available"], bool)
        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert completed_tool_steps == [
            "prepare_capstone_container_ci_contract",
            "resolve_upstream_container_evidence",
            "generate_validate_runtime_image_spec",
            "build_smoke_check_container_image",
        ]
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "upstream_evidence_resolved:observed:passed" in result
        assert "local_model_artifact_resolved:observed:passed" in result
        assert "container_build_spec_reported:declared:failed" in result
        assert "dependency_context_reported:observed:passed" in result
        assert "secret_safety_validated:observed:passed" in result
        assert "docker_availability_reported:observed:passed" in result
        assert "data_stage_evidence_artifact_reported" in result
        assert "mlflow_best_artifact_verified" in result
        assert mock_agent.ctx.globals["capstone_runtime_image_spec"]["blocked_capabilities"] == [
            {
                "capability": "write_container_build_spec",
                "reason": "Writing Dockerfile requires an Approval Gate.",
                "required_risk_categories": ["writes_project_files"],
                "next_action": (
                    "Record approval for generate_validate_runtime_image_spec "
                    "before writing Dockerfile."
                ),
            }
        ]
        assert (
            mock_agent.ctx.globals["capstone_container_build_smoke_check"][
                "blocked_capabilities"
            ][0]["capability"]
            == "container_build_spec_reported"
        )
        assert "container_ci_evidence.json writing is deferred to Phase 5 Issue 7" in result
        assert "build_ml_docker_image" not in [
            step["tool"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert not (tmp_path / "Dockerfile").exists()
        assert not (tmp_path / ".auto_mlops" / "capstone" / "container_ci_evidence.json").exists()
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_data_detects_two_valid_image_folder_datasets(
        self, mock_agent, tmp_path
    ):
        """Test Issue 2 records read-only dataset layout evidence for two datasets."""
        dataset_1 = tmp_path / "source_one"
        dataset_2 = tmp_path / "source_two"
        _write_tiny_image(dataset_1 / "cats" / "cat-1.jpg")
        _write_tiny_image(dataset_1 / "dogs" / "dog-1.png")
        _write_tiny_image(dataset_2 / "apple" / "apple-1.jpeg")
        _write_tiny_image(dataset_2 / "banana" / "banana-1.webp")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run for registry data prep"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Prepare capstone data "
                f"dataset_1_path={dataset_1} dataset_2_path={dataset_2}",
                str(tmp_path),
            )

        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_data"
        assert mock_agent.workflow_selection.status is WorkflowStatus.PENDING
        assert mock_agent.ctx.globals["workflow_inputs"] == {
            "project_path": str(tmp_path),
            "completion_mode": "local_ready",
            "dataset_1_path": str(dataset_1),
            "dataset_2_path": str(dataset_2),
            "test_size": 0.2,
            "split_seed": 42,
            "materialize_splits": False,
            "dvc_transfer_direction": "push",
        }
        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert completed_tool_steps == ["prepare_capstone_data_contract"]
        assert "Approval required before executing workflow step 'generate_split_manifests'" in result
        assert "writes_project_files" in result
        assert "dataset_1" in result
        assert "dataset_2" in result
        assert "split_manifest.json" in result
        verification_results = mock_agent.ctx.globals["verification_results"]
        assert [
            verification_result.check_name for verification_result in verification_results
        ] == ["two_dataset_paths_provided", "two_dataset_layouts_supported"]
        artifact_manifest = mock_agent.ctx.globals["artifact_manifest"]
        assert [entry.artifact_type for entry in artifact_manifest.entries] == [
            "capstone_source_dataset",
            "capstone_source_dataset",
        ]
        assert all(entry.state.value == "external" for entry in artifact_manifest.entries)
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_data_blocks_unsupported_dataset_without_mutation(
        self, mock_agent, tmp_path
    ):
        """Test unsupported layouts block with dataset-level evidence and no writes."""
        dataset_1 = tmp_path / "source_one"
        dataset_2 = tmp_path / "unsupported"
        _write_tiny_image(dataset_1 / "cats" / "cat-1.jpg")
        dataset_2.mkdir()
        before_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run for registry data prep"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Prepare capstone data "
                f"dataset_1_path={dataset_1} dataset_2_path={dataset_2}",
                str(tmp_path),
            )

        after_paths = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
        assert before_paths == after_paths
        assert mock_agent.workflow_selection.workflow_id == "prepare_capstone_data"
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "unsupported_empty_dataset" in result
        assert "Add class-labelled subdirectories" in result
        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.BLOCKED
        assert contract_status.failed_checks == ()
        verification_results = mock_agent.ctx.globals["verification_results"]
        layout_result = next(
            result
            for result in verification_results
            if result.check_name == "two_dataset_layouts_supported"
        )
        assert layout_result.passed is False
        assert "dataset_2" in layout_result.evidence
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_prepare_capstone_data_detects_existing_train_test_layout(
        self, mock_agent, tmp_path
    ):
        """Test existing train/test folders are recorded as split evidence candidates."""
        dataset_1 = tmp_path / "source_one"
        dataset_2 = tmp_path / "source_two"
        _write_tiny_image(dataset_1 / "train" / "cats" / "cat-train.jpg")
        _write_tiny_image(dataset_1 / "test" / "cats" / "cat-test.jpg")
        _write_tiny_image(dataset_2 / "train" / "apple" / "apple-train.jpg")
        _write_tiny_image(dataset_2 / "test" / "apple" / "apple-test.jpg")

        with (
            patch.object(mock_agent.perception, "run", new_callable=AsyncMock) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Prepare capstone data "
                f"dataset_1_path={dataset_1} dataset_2_path={dataset_2}",
                str(tmp_path),
            )

        assert "existing_train_test_split" in result
        assert str(dataset_2 / "train") in result
        assert str(dataset_2 / "test") in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_build_capstone_pipeline_records_deferred_capabilities(
        self, mock_agent, tmp_path
    ):
        """Test Capstone Orchestrator skeleton blocks honestly on future work."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry capstone must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run("Build full capstone pipeline", str(tmp_path))

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "build_capstone_pipeline"
        assert completed_tool_steps == ["record_capstone_orchestrator_skeleton"]
        assert mock_agent.status == "paused"
        assert "build_capstone_pipeline final workflow status derived from SuccessContract" in result
        assert "contract_status: blocked" in result
        assert "capstone_pipeline_ready" in result
        assert "train_until_better" in result
        assert "S3 DVC remote automation" in result
        assert "KServe/Helm/ArgoCD" in result
        assert "final report" in result
        assert "video" in result
        assert ".auto_mlops/capstone/orchestrator_plan.json" in result
        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.BLOCKED
        assert [failure.check_name for failure in contract_status.missing_evidence] == [
            "capstone_pipeline_ready"
        ]
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_blocks_missing_detection_evidence_before_training(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track stops after unsupported detection evidence."""
        project_path = tmp_path / "incomplete-training-project"
        project_path.mkdir()
        (project_path / "pyproject.toml").write_text("[project]\ndependencies = ['torch']\n")
        executed_step_ids = []

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            executed_step_ids.append(step_id)
            if step_id == "detect_training_project":
                result = mcp_mlops_tools.detect_training_project(str(project_path))
                return True, {
                    "success": True,
                    "result": result,
                    "step_id": step_id,
                    "tool": tool,
                }
            raise AssertionError(f"training executed without detection evidence: {step_id}")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch(
                "agent.agent_loop.execute_step",
                new_callable=AsyncMock,
                side_effect=execute_registry_step,
            ),
        ):
            result = await mock_agent.run(
                "Train this project timeout 30 max epochs 1 device cpu subset 8 "
                "metric accuracy maximize threshold 0.01 tie keep_baseline "
                "baseline metric 0.80 baseline artifact checkpoints/baseline.ckpt",
                str(project_path),
            )

        assert executed_step_ids == ["detect_training_project"]
        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in result
        assert "training_entrypoint_detected" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_succeeds_from_bounded_training_evidence(
        self, mock_agent, tmp_path
    ):
        """Test train_and_track completes when detection and bounded training evidence exist."""
        _write_bounded_training_fixture(
            tmp_path,
            "# config_path='../configs', config_name='train.yaml'\n"
            "import json\n"
            "from pathlib import Path\n"
            "Path('checkpoints').mkdir(exist_ok=True)\n"
            "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
            "print(json.dumps({'metrics': {'accuracy': 0.93}}))\n",
        )
        (tmp_path / "checkpoints" / "baseline.ckpt").write_text("baseline")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Train this project timeout 30 max epochs 1 device cpu subset 8 "
                "metric accuracy maximize threshold 0.01 tie keep_baseline "
                "baseline metric 0.80 baseline artifact checkpoints/baseline.ckpt",
                str(tmp_path),
            )

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert mock_agent.workflow_selection.workflow_id == "train_and_track"
        assert completed_tool_steps == [
            "detect_training_project",
            "run_bounded_training",
            "track_training_in_mlflow",
            "select_best_model_artifact",
        ]
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in result
        assert "accuracy" in result
        assert "checkpoints/model.ckpt" in result
        assert "mlflow_run_exists" in result
        assert "model_artifact_selected" in result
        assert "run_id" in result
        selected_entries = [
            entry
            for entry in mock_agent.ctx.globals["artifact_manifest"].entries
            if entry.producing_step == "select_best_model_artifact"
        ]
        assert selected_entries[0].state.value == "selected"
        assert selected_entries[0].metadata["decision"] == "select_latest"
        assert selected_entries[0].metadata["source_run_id"]
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_train_and_track_records_failed_training_in_mlflow(
        self, mock_agent, tmp_path
    ):
        """Test failed bounded training still records a failed MLflow run."""
        _write_bounded_training_fixture(
            tmp_path,
            "# config_path='../configs', config_name='train.yaml'\n"
            "import json\n"
            "from pathlib import Path\n"
            "Path('checkpoints').mkdir(exist_ok=True)\n"
            "Path('checkpoints/model.ckpt').write_text('checkpoint')\n"
            "print(json.dumps({'metrics': {'accuracy': 0.31}}))\n"
            "raise SystemExit(3)\n",
        )
        (tmp_path / "checkpoints" / "baseline.ckpt").write_text("baseline")

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry training must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
        ):
            result = await mock_agent.run(
                "Train this project timeout 30 max epochs 1 device cpu subset 8 "
                "metric accuracy maximize threshold 0.01 tie keep_baseline "
                "baseline metric 0.80 baseline artifact checkpoints/baseline.ckpt",
                str(tmp_path),
            )

        completed_tool_steps = [
            step["index"] for step in mock_agent.ctx.get_completed_steps() if step["tool"]
        ]
        assert completed_tool_steps == [
            "detect_training_project",
            "run_bounded_training",
            "track_training_in_mlflow",
            "select_best_model_artifact",
        ]
        assert mock_agent.status == "failed"
        assert "contract_status: failed" in result
        assert "bounded_training_command_completed" in result
        assert "run_status" in result
        assert "FAILED" in result
        assert "keep_baseline" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_succeeds_from_observed_runtime_evidence(
        self, mock_agent
    ):
        """Test LitServe GPU success is derived from observed live evidence."""
        mock_agent.auto_approve = True

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            payloads = {
                "detect_runtime_environment": {"success": True},
                "detect_gpu_cuda": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "gpu_detection_recorded",
                            "evidence_type": "observed",
                            "source_step": "detect_gpu_cuda",
                            "passed": True,
                            "evidence": "nvidia-smi observed NVIDIA A10",
                        }
                    ],
                },
                "select_best_model_artifact": {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "model_artifact",
                                "producing_step": "select_best_model_artifact",
                                "state": "selected",
                                "path": "models/model.pt",
                            }
                        ]
                    },
                },
                "generate_litserve_api": {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "serving_application",
                                "producing_step": "generate_litserve_api",
                                "state": "generated",
                                "path": "deployment/litserve/server.py",
                            }
                        ]
                    },
                },
                "configure_litserve_gpu_runtime": {"success": True},
                "create_dockerfile": {"success": True},
                "build_image_if_available": {"success": True},
                "start_litserve_server": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "server_start_command_recorded",
                            "evidence_type": "observed",
                            "source_step": "start_litserve_server",
                            "passed": True,
                            "evidence": "process 123 started: python deployment/litserve/server.py",
                        }
                    ],
                },
                "test_health_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "health_result_recorded",
                            "evidence_type": "observed",
                            "source_step": "test_health_endpoint",
                            "passed": True,
                            "evidence": "GET /health returned 200",
                        }
                    ],
                },
                "test_prediction_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "prediction_result_recorded",
                            "evidence_type": "observed",
                            "source_step": "test_prediction_endpoint",
                            "passed": True,
                            "evidence": "POST /predict returned sample prediction",
                        }
                    ],
                },
                "capture_logs_and_endpoint": {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "endpoint_url_recorded",
                            "evidence_type": "observed",
                            "source_step": "capture_logs_and_endpoint",
                            "passed": True,
                            "evidence": "endpoint_url=http://127.0.0.1:8000",
                        }
                    ],
                },
                "write_monitoring_and_rollback_report": {
                    "success": True,
                    "rollback_plan": {
                        "command": "kill 123",
                        "documented_target": "Stop the user-started Lambda Cloud instance manually.",
                    },
                },
            }
            return (
                True,
                {
                    "success": True,
                    "result": payloads[step_id],
                    "step_id": step_id,
                    "tool": tool,
                },
            )

        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry runtime must skip perception"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch(
                "agent.agent_loop.execute_step",
                new_callable=AsyncMock,
                side_effect=execute_registry_step,
            ) as mock_execute_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert mock_agent.workflow_selection.workflow_id == "deploy_litserve_gpu"
        assert mock_agent.status == "success"
        assert "deploy_litserve_gpu final workflow status derived from SuccessContract" in result
        assert "contract_status: succeeded" in result
        assert "workflow_status: succeeded" in result
        assert "endpoint_url=http://127.0.0.1:8000" in result
        assert "Stop the user-started Lambda Cloud instance manually" in result
        assert mock_execute_step.await_count == len(
            mock_agent.workflow_registry.get("deploy_litserve_gpu").steps
        )
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_failed_gpu_check_stops_before_server_start(
        self, mock_agent
    ):
        """Test failed observed GPU evidence fails safely before deployment actions."""
        mock_agent.auto_approve = True
        executed_step_ids = []

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            executed_step_ids.append(step_id)
            if step_id == "detect_runtime_environment":
                return True, {
                    "success": True,
                    "result": {"success": True},
                    "step_id": step_id,
                    "tool": tool,
                }
            if step_id == "detect_gpu_cuda":
                return True, {
                    "success": True,
                    "result": {
                        "success": True,
                        "verification_results": [
                            {
                                "check_name": "gpu_detection_recorded",
                                "evidence_type": "observed",
                                "source_step": "detect_gpu_cuda",
                                "passed": False,
                                "evidence": "nvidia-smi unavailable and torch CUDA unavailable",
                            }
                        ],
                    },
                    "step_id": step_id,
                    "tool": tool,
                }
            raise AssertionError(f"unsafe step executed after failed GPU check: {step_id}")

        with patch(
            "agent.agent_loop.execute_step",
            new_callable=AsyncMock,
            side_effect=execute_registry_step,
        ):
            result = await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert executed_step_ids == ["detect_runtime_environment", "detect_gpu_cuda"]
        assert mock_agent.status == "failed"
        assert "contract_status: failed" in result
        assert "gpu_detection_recorded" in result
        assert "start_litserve_server" not in executed_step_ids

    @pytest.mark.asyncio
    async def test_run_litserve_gpu_uses_selected_pickle_artifact_for_litserve_api(
        self, mock_agent
    ):
        """Test selected sklearn artifacts are passed into LitServe generation."""
        mock_agent.auto_approve = True
        generate_args = None
        runtime_args_by_step = {}

        async def execute_registry_step(step_id, tool, args, ctx, tools_module):
            nonlocal generate_args
            if step_id == "detect_runtime_environment":
                payload = {"success": True}
            elif step_id == "detect_gpu_cuda":
                payload = {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": "gpu_detection_recorded",
                            "evidence_type": "observed",
                            "source_step": "detect_gpu_cuda",
                            "passed": True,
                            "evidence": "nvidia-smi observed NVIDIA A10",
                        }
                    ],
                }
            elif step_id == "select_best_model_artifact":
                payload = {
                    "success": True,
                    "model_path": "outputs/model.pkl",
                    "model_type": "tabular_regressor",
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "model_artifact",
                                "producing_step": "select_best_model_artifact",
                                "state": "selected",
                                "path": "outputs/model.pkl",
                            }
                        ]
                    },
                }
            elif step_id == "generate_litserve_api":
                generate_args = args
                payload = {
                    "success": True,
                    "artifact_manifest": {
                        "entries": [
                            {
                                "artifact_type": "serving_application",
                                "producing_step": "generate_litserve_api",
                                "state": "generated",
                                "path": "deployment/litserve/server.py",
                            }
                        ]
                    },
                }
            else:
                runtime_args_by_step[step_id] = args
                payload = {
                    "success": True,
                    "verification_results": [
                        {
                            "check_name": {
                                "start_litserve_server": "server_start_command_recorded",
                                "test_health_endpoint": "health_result_recorded",
                                "test_prediction_endpoint": "prediction_result_recorded",
                                "capture_logs_and_endpoint": "endpoint_url_recorded",
                            }.get(step_id, "unused"),
                            "evidence_type": "observed",
                            "source_step": step_id,
                            "passed": True,
                            "evidence": f"{step_id} evidence",
                        }
                    ],
                }
                if step_id == "start_litserve_server":
                    payload["endpoint_url"] = "http://127.0.0.1:8001"
                    payload["process_id"] = 123
                    payload["port"] = 8001
                if step_id == "write_monitoring_and_rollback_report":
                    payload = {
                        "success": True,
                        "rollback_plan": {"command": "kill 123"},
                    }
            return (
                True,
                {"success": True, "result": payload, "step_id": step_id, "tool": tool},
            )

        with patch(
            "agent.agent_loop.execute_step",
            new_callable=AsyncMock,
            side_effect=execute_registry_step,
        ):
            await mock_agent.run(
                "Deploy this model on Lambda Labs GPU with LitServe",
                "/test/path",
            )

        assert generate_args["model_path"] == "outputs/model.pkl"
        assert generate_args["model_type"] == "tabular_regressor"
        assert runtime_args_by_step["test_health_endpoint"]["endpoint_url"] == "http://127.0.0.1:8001"
        assert (
            runtime_args_by_step["test_prediction_endpoint"]["endpoint_url"]
            == "http://127.0.0.1:8001"
        )
        assert (
            runtime_args_by_step["capture_logs_and_endpoint"]["endpoint_url"]
            == "http://127.0.0.1:8001"
        )
        assert runtime_args_by_step["write_monitoring_and_rollback_report"]["process_id"] == 123
        assert runtime_args_by_step["write_monitoring_and_rollback_report"]["port"] == 8001

    @pytest.mark.asyncio
    async def test_run_blocks_setup_workflow_missing_project_path_with_clarifying_question(
        self, mock_agent
    ):
        """Test missing setup workflow inputs block before planning or tool execution."""
        with (
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("perception should not run while workflow is blocked"),
            ) as mock_perception,
            patch.object(mock_agent.decision, "run", new_callable=AsyncMock) as mock_decision,
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
        ):
            result = await mock_agent.run("Set up MLOps for this project")

        assert mock_agent.workflow_selection.workflow_id == "setup_pipeline"
        assert mock_agent.workflow_selection.status is WorkflowStatus.BLOCKED
        assert mock_agent.workflow_selection.missing_inputs == ("project_path",)
        assert mock_agent.ctx.get_pending_steps() == []
        assert "missing_inputs" in result
        assert "project_path" in result
        assert "What project path should I set up MLOps for?" in result
        mock_perception.assert_not_awaited()
        mock_decision.assert_not_awaited()
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_gated_step_without_approval_blocks_before_execute_step(
        self, mock_agent
    ):
        """Test setup workflow approval gates block before tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.next_step_id = workflow_step.step_id

        with patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step:
            await mock_agent._execute_steps_loop()

        approval_events = [
            event for event in events if event["type"] == "approval_required"
        ]
        assert mock_agent.status == "paused"
        assert approval_events
        assert approval_events[0]["data"]["workflow_id"] == "setup_pipeline"
        assert approval_events[0]["data"]["step_id"] == "create_or_validate_hydra_config"
        assert approval_events[0]["data"]["risk_categories"] == ["writes_project_files"]
        assert approval_events[0]["data"]["next_action"]
        assert "Approval required" in mock_agent.final_output
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_denied_approval_blocks_before_execute_step(self, mock_agent):
        """Test denied setup workflow approval prevents tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="denied",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step:
            await mock_agent._execute_steps_loop()

        denied_events = [event for event in events if event["type"] == "approval_denied"]
        assert mock_agent.status == "failed"
        assert denied_events
        assert denied_events[0]["data"]["workflow_id"] == "setup_pipeline"
        assert denied_events[0]["data"]["step_id"] == "create_or_validate_hydra_config"
        assert denied_events[0]["data"]["risk_categories"] == ["writes_project_files"]
        assert "Approval denied" in mock_agent.final_output
        assert "ops@example.com" in mock_agent.final_output
        mock_execute_step.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_approved_record_allows_execute_step(self, mock_agent):
        """Test approved setup workflow approval allows tool execution."""
        events = []

        async def capture_event(event_type, data):
            events.append({"type": event_type, "data": data})

        mock_agent.on_event = capture_event
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        approval_events = [
            event
            for event in events
            if event["type"] in {"approval_required", "approval_denied"}
        ]
        assert approval_events == []
        mock_execute_step.assert_awaited_once()
        assert mock_execute_step.await_args.kwargs["step_id"] == workflow_step.step_id

    @pytest.mark.asyncio
    async def test_setup_pipeline_approved_step_skips_post_step_perception(self, mock_agent):
        """Test registry setup steps do not require LLM perception after execution."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                side_effect=AssertionError("registry setup step must skip post-step perception"),
            ) as mock_perception,
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        mock_execute_step.assert_awaited_once()
        mock_perception.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_setup_pipeline_captures_structured_verification_result(self, mock_agent):
        """Test setup workflow records structured verification evidence from step results."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {
                    "success": True,
                    "step_id": workflow_step.step_id,
                    "result": {
                        "verification_results": [
                            {
                                "check_name": "hydra_config_validates",
                                "evidence_type": "observed",
                                "source_step": workflow_step.step_id,
                                "passed": True,
                                "evidence": "validated conf/config.yaml",
                            }
                        ]
                    },
                },
            )
            await mock_agent._execute_steps_loop()

        verification_results = mock_agent.ctx.globals["verification_results"]
        assert verification_results == (
            VerificationResult(
                check_name="hydra_config_validates",
                evidence_type="observed",
                source_step=workflow_step.step_id,
                passed=True,
                evidence="validated conf/config.yaml",
            ),
        )

    @pytest.mark.asyncio
    async def test_setup_pipeline_captures_artifact_manifest_entry(self, mock_agent):
        """Test setup workflow records explicit generated artifact evidence from step results."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "create_or_validate_hydra_config"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {
                    "success": True,
                    "step_id": workflow_step.step_id,
                    "result": {
                        "artifact_manifest": {
                            "entries": [
                                {
                                    "artifact_type": "configuration",
                                    "producing_step": workflow_step.step_id,
                                    "state": "validated",
                                    "path": "conf/config.yaml",
                                }
                            ]
                        }
                    },
                },
            )
            await mock_agent._execute_steps_loop()

        assert mock_agent.ctx.globals["artifact_manifest"] == ArtifactManifest(
            entries=(
                ArtifactManifestEntry(
                    artifact_type="configuration",
                    producing_step=workflow_step.step_id,
                    state="validated",
                    path="conf/config.yaml",
                ),
            )
        )

    @pytest.mark.asyncio
    async def test_setup_pipeline_missing_evidence_blocks_prompt_success(self, mock_agent):
        """Test setup workflow missing evidence blocks prompt-authored success."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        workflow_step = mock_agent.workflow_registry.get("setup_pipeline").step_by_id(
            "analyze_project_structure"
        )
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.SUMMARIZE, "original_goal_achieved": True},
            ),
            patch.object(
                mock_agent.summarizer,
                "summarize",
                new_callable=AsyncMock,
                side_effect=AssertionError("summarizer must not decide setup success"),
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.BLOCKED
        assert contract_status.failed_checks == ()
        assert contract_status.missing_evidence
        assert mock_agent.ctx.globals["workflow_status"] is WorkflowStatus.BLOCKED
        assert mock_agent.status == "paused"
        assert "contract_status: blocked" in mock_agent.final_output
        assert "missing_evidence" in mock_agent.final_output

    @pytest.mark.asyncio
    async def test_setup_pipeline_passing_contract_succeeds_from_structured_evidence(
        self, mock_agent
    ):
        """Test setup workflow succeeds only from passing contract evidence."""
        mock_agent._initialize_session("Set up MLOps for this project", "/test/path", 0.85)
        mock_agent.workflow_selection = mock_agent.workflow_registry.select_workflow(
            "Set up MLOps for this project"
        )
        template = mock_agent.workflow_registry.get("setup_pipeline")
        workflow_step = template.step_by_id("create_ci_workflow")
        mock_agent.ctx.add_step(
            step_id=workflow_step.step_id,
            description=workflow_step.description,
            step_type=StepType.CODE,
            tool=workflow_step.tool_functions[0],
            args={},
            from_node=StepType.ROOT,
        )
        mock_agent.ctx.globals["approval_records"] = (
            ApprovalRecord(
                workflow_run_id=mock_agent.session_id,
                step_id=workflow_step.step_id,
                risk_categories=("writes_project_files",),
                status="approved",
                approver="ops@example.com",
                timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            ),
        )
        mock_agent.ctx.globals["verification_results"] = tuple(
            VerificationResult(
                check_name=check.name,
                evidence_type="declared",
                source_step=check.source_step,
                passed=True,
                evidence=f"{check.name} evidence",
            )
            for check in template.success_contract.checks
            if check.name != "generated_files_reported"
        )
        mock_agent.ctx.globals["artifact_manifest"] = ArtifactManifest(
            entries=(
                ArtifactManifestEntry(
                    artifact_type="configuration",
                    producing_step="create_or_validate_hydra_config",
                    state="generated",
                    path="conf/config.yaml",
                ),
                ArtifactManifestEntry(
                    artifact_type="pipeline_definition",
                    producing_step="create_dvc_yaml",
                    state="generated",
                    path="dvc.yaml",
                ),
                ArtifactManifestEntry(
                    artifact_type="container_definition",
                    producing_step="create_dockerfile",
                    state="generated",
                    path="Dockerfile",
                ),
                ArtifactManifestEntry(
                    artifact_type="automation_workflow",
                    producing_step="create_ci_workflow",
                    state="generated",
                    path=".github/workflows/mlops.yml",
                ),
            )
        )
        mock_agent.next_step_id = workflow_step.step_id

        with (
            patch("agent.agent_loop.execute_step", new_callable=AsyncMock) as mock_execute_step,
            patch.object(
                mock_agent.perception,
                "run",
                new_callable=AsyncMock,
                return_value={"route": Route.DECISION, "original_goal_achieved": False},
            ),
            patch.object(
                mock_agent.summarizer,
                "summarize",
                new_callable=AsyncMock,
                side_effect=AssertionError("summarizer must not decide setup success"),
            ),
        ):
            mock_execute_step.return_value = (
                True,
                {"success": True, "step_id": workflow_step.step_id},
            )
            await mock_agent._execute_steps_loop()

        contract_status = mock_agent.ctx.globals["contract_status"]
        assert contract_status.status is WorkflowStatus.SUCCEEDED
        assert contract_status.missing_evidence == ()
        assert contract_status.failed_checks == ()
        assert mock_agent.ctx.globals["workflow_status"] is WorkflowStatus.SUCCEEDED
        assert mock_agent.status == "success"
        assert "contract_status: succeeded" in mock_agent.final_output


# ============================================================================
# run_mlops_agent Convenience Function Tests
# ============================================================================


class TestRunMlopsAgent:
    """Tests for run_mlops_agent convenience function."""

    @pytest.mark.asyncio
    async def test_run_mlops_agent_creates_agent(self, mock_llm):
        """Test run_mlops_agent creates and runs agent."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            result = await run_mlops_agent("Test query", "/test/path", 0.85)
            assert result == "Test result"

    @pytest.mark.asyncio
    async def test_run_mlops_agent_passes_event_callback(self, mock_llm):
        """Test run_mlops_agent passes event callback."""
        mock_llm.json_response = {
            "entities": {},
            "pipeline_stage": "setup",
            "route": "summarize",
            "original_goal_achieved": True,
            "confidence": 0.9,
            "reasoning": "Goal achieved",
        }

        events = []

        async def capture_event(event_type, data):
            events.append(event_type)

        with (
            patch(
                "agent.agent_loop.Summarizer.summarize", new_callable=AsyncMock
            ) as mock_summarize,
            patch("agent.agent_loop.Perception.run", new_callable=AsyncMock) as mock_perception,
        ):
            mock_perception.return_value = {
                "entities": {},
                "pipeline_stage": "setup",
                "route": "summarize",
                "original_goal_achieved": True,
                "confidence": 0.9,
                "reasoning": "Goal achieved",
            }
            mock_summarize.return_value = {"summary_markdown": "Test result"}
            await run_mlops_agent("Test query", on_event=capture_event)
            assert "status" in events
            assert "phase" in events


# ============================================================================
# AgentLoop Handle Failure Tests
# ============================================================================


class TestAgentLoopHandleFailure:
    """Tests for AgentLoop _handle_failure method."""

    @pytest.fixture
    def agent(self, tmp_path):
        """Create AgentLoop with test prompts."""
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "perception_prompt.txt").write_text("Perception")
        (prompts_dir / "decision_prompt.txt").write_text("Decision")
        (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
        (prompts_dir / "improvement_prompt.txt").write_text("Improve")
        return AgentLoop(prompts_dir=str(prompts_dir))

    @pytest.mark.asyncio
    async def test_handle_failure_sets_status_failed(self, agent, mock_llm):
        """Test _handle_failure preserves failed status while generating a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.status == "failed"
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_marks_session_completed(self, agent, mock_llm):
        """Test _handle_failure marks session as completed with failure."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            await agent._handle_failure()
            assert agent.session.status == "failed"

    @pytest.mark.asyncio
    async def test_handle_failure_returns_summary(self, agent, mock_llm):
        """Test _handle_failure still generates a summary."""
        mock_llm.json_response = {"summary_markdown": "Failure summary"}

        agent._initialize_session("Test", "/test", 0.85)
        agent.p_out = {}

        with patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum:
            mock_sum.return_value = {"summary_markdown": "Failure summary"}
            result = await agent._handle_failure()
            assert result == "Failure summary"
