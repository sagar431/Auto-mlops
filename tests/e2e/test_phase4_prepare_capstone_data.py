from unittest.mock import AsyncMock, patch

import pytest

import mcp_mlops_tools
from agent.agent_loop import AgentLoop
from workflow.registry import WorkflowStatus


def _write_tiny_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image")


@pytest.mark.asyncio
async def test_prepare_capstone_data_detects_two_image_folder_datasets_read_only(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    _write_tiny_image(dataset_1 / "train" / "cats" / "cat-train.jpg")
    _write_tiny_image(dataset_1 / "test" / "cats" / "cat-test.jpg")
    _write_tiny_image(dataset_2 / "train" / "apple" / "apple-train.jpg")
    _write_tiny_image(dataset_2 / "test" / "apple" / "apple-test.jpg")
    agent = AgentLoop(prompts_dir=str(prompts_dir))

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2}",
            str(project_path),
    )

    assert agent.workflow_selection.workflow_id == "prepare_capstone_data"
    assert agent.status == "paused"
    assert "Approval required before executing workflow step 'track_capstone_data_package'" in result
    assert "two_dataset_paths_provided:observed:passed" in result
    assert "two_dataset_layouts_supported:observed:passed" in result
    assert "existing_train_test_split" in result
    assert not (project_path / ".auto_mlops").exists()
    assert not (project_path / ".dvc").exists()
    assert not (project_path / "data" / "capstone").exists()


@pytest.mark.asyncio
async def test_prepare_capstone_data_blocks_for_split_manifest_write_approval(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    for dataset in (dataset_1, dataset_2):
        _write_tiny_image(dataset / "cats" / "cat-1.jpg")
        _write_tiny_image(dataset / "dogs" / "dog-1.jpg")
    events = []

    async def on_event(event_type, data):
        events.append({"type": event_type, "data": data})

    agent = AgentLoop(prompts_dir=str(prompts_dir), on_event=on_event)

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2} "
            "test_size=0.5 split_seed=11",
            str(project_path),
        )

    approval_events = [event for event in events if event["type"] == "approval_required"]
    assert agent.workflow_selection.workflow_id == "prepare_capstone_data"
    assert agent.status == "paused"
    assert approval_events
    assert approval_events[0]["data"]["step_id"] == "generate_split_manifests"
    assert approval_events[0]["data"]["risk_categories"] == ["writes_project_files"]
    assert "Approval required before executing workflow step 'generate_split_manifests'" in result
    assert "split_manifest.json" in result
    assert '"seed": 11' in result
    assert '"test_size": 0.5' in result
    assert "two_dataset_layouts_supported" in {
        item.check_name for item in agent.ctx.globals["verification_results"]
    }
    assert not (project_path / "data" / "capstone").exists()


@pytest.mark.asyncio
async def test_prepare_capstone_data_approved_run_generates_split_manifests_only(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    for dataset in (dataset_1, dataset_2):
        for class_name in ("cats", "dogs"):
            for index in range(3):
                _write_tiny_image(dataset / class_name / f"{class_name}-{index}.jpg")

    agent = AgentLoop(prompts_dir=str(prompts_dir), auto_approve=True)

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2} "
            "test_size=0.34 split_seed=19",
            str(project_path),
        )

    manifest_1 = project_path / "data" / "capstone" / "dataset_1" / "split_manifest.json"
    manifest_2 = project_path / "data" / "capstone" / "dataset_2" / "split_manifest.json"
    assert manifest_1.exists()
    assert manifest_2.exists()
    assert agent.status == "paused"
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.BLOCKED
    assert "split_evidence_recorded:observed:passed" in result
    assert "dataset_lineage_artifacts_reported:observed:passed" in result
    artifact_manifest = agent.ctx.globals["artifact_manifest"]
    assert {
        (entry.artifact_type, entry.state.value, entry.path)
        for entry in artifact_manifest.entries
        if entry.artifact_type == "split_manifest"
    } == {
        ("split_manifest", "generated", "data/capstone/dataset_1/split_manifest.json"),
        ("split_manifest", "generated", "data/capstone/dataset_2/split_manifest.json"),
    }
    assert (dataset_1 / "cats" / "cats-0.jpg").exists()
    assert not (project_path / ".dvc").exists()
    assert not (project_path / ".auto_mlops" / "capstone" / "data_stage_evidence.json").exists()
    assert not (project_path / "data" / "capstone" / "dataset_1" / "train").exists()


@pytest.mark.asyncio
async def test_prepare_capstone_data_dvc_tracks_capstone_package_after_approval(
    tmp_path, monkeypatch
):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    for dataset in (dataset_1, dataset_2):
        for class_name in ("cats", "dogs"):
            for index in range(3):
                _write_tiny_image(dataset / class_name / f"{class_name}-{index}.jpg")
    commands = []

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        if cmd == ["dvc", "init", "--no-scm"]:
            (project_path / ".dvc").mkdir(exist_ok=True)
            (project_path / ".dvc" / "config").write_text("[core]\n")
        elif cmd[:2] == ["dvc", "add"]:
            package_path = cmd[2]
            (project_path / f"{package_path}.dvc").write_text(
                f"outs:\n- path: {package_path}\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    agent = AgentLoop(prompts_dir=str(prompts_dir), auto_approve=True)

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2}",
            str(project_path),
        )

    assert (project_path / ".dvc" / "config").exists()
    assert (project_path / "data" / "capstone" / "dataset_1.dvc").exists()
    assert (project_path / "data" / "capstone" / "dataset_2.dvc").exists()
    assert commands == [
        (["dvc", "init", "--no-scm"], str(project_path), 60),
        (["dvc", "add", "data/capstone/dataset_1"], str(project_path), 300),
        (["dvc", "add", "data/capstone/dataset_2"], str(project_path), 300),
    ]
    assert all(str(dataset_1) not in " ".join(command) for command, _, _ in commands)
    assert all(str(dataset_2) not in " ".join(command) for command, _, _ in commands)
    assert agent.status == "paused"
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.BLOCKED
    assert "dvc_repo_validated:observed:passed" in result
    assert "capstone_data_package_tracked:observed:passed" in result
    assert "s3_remote_validated" not in result
    assert not (project_path / ".auto_mlops" / "capstone" / "data_stage_evidence.json").exists()
    artifact_manifest = agent.ctx.globals["artifact_manifest"]
    assert {
        (entry.artifact_type, entry.state.value, entry.path)
        for entry in artifact_manifest.entries
        if entry.artifact_type in {"capstone_data_package", "dvc_tracking_file"}
    } >= {
        ("capstone_data_package", "generated", "data/capstone/dataset_1"),
        ("capstone_data_package", "generated", "data/capstone/dataset_2"),
        ("dvc_tracking_file", "generated", "data/capstone/dataset_1.dvc"),
        ("dvc_tracking_file", "generated", "data/capstone/dataset_2.dvc"),
    }


@pytest.mark.asyncio
async def test_prepare_capstone_data_capstone_complete_pushes_after_approval(
    tmp_path, monkeypatch
):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    for dataset in (dataset_1, dataset_2):
        for class_name in ("cats", "dogs"):
            for index in range(3):
                _write_tiny_image(dataset / class_name / f"{class_name}-{index}.jpg")
    commands = []
    raw_s3_url = "s3://secret-capstone-bucket/team-a/capstone"

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        if cmd == ["dvc", "init", "--no-scm"]:
            (project_path / ".dvc").mkdir(exist_ok=True)
            (project_path / ".dvc" / "config").write_text("[core]\n")
        elif cmd[:2] == ["dvc", "add"]:
            package_path = cmd[2]
            (project_path / f"{package_path}.dvc").write_text(
                f"outs:\n- path: {package_path}\n"
            )
        elif cmd[:3] == ["dvc", "remote", "add"]:
            (project_path / ".dvc" / "config").write_text(
                "[core]\n    remote = capstone\n"
                '[\'remote "capstone"\']\n'
                f"    url = {raw_s3_url}\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": True,
            "status": "validated",
            "identity": {
                "account": "123456789012",
                "arn": "arn:aws:iam::123456789012:user/capstone",
                "user_id": "AIDAEXAMPLE",
            },
            "bucket_reachable": True,
            "prefix_checked": True,
            "next_actions": [],
        },
    )
    agent = AgentLoop(prompts_dir=str(prompts_dir), auto_approve=True)

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2} "
            "completion_mode=capstone_complete "
            f"dvc_remote_name=capstone dvc_remote_url={raw_s3_url}",
            str(project_path),
        )

    command_names = [command[0][:2] for command in commands]
    assert ["dvc", "push"] in command_names
    assert ["dvc", "pull"] not in command_names
    assert "s3_remote_validated:observed:passed" in result
    assert "s3_transfer_completed:observed:passed" in result
    assert raw_s3_url not in result
    assert "secret-capstone-bucket" not in result
    assert "123456789012" not in result
    assert "AIDAEXAMPLE" not in result
    assert agent.status == "paused"
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.BLOCKED
    assert not (project_path / ".auto_mlops" / "capstone" / "data_stage_evidence.json").exists()
    remote_entries = [
        entry
        for entry in agent.ctx.globals["artifact_manifest"].entries
        if entry.artifact_type == "capstone_data_remote"
    ]
    assert remote_entries
    assert remote_entries[0].state.value == "validated"
    assert remote_entries[0].metadata["remote_type"] == "s3"
    transfer_entries = [
        entry
        for entry in agent.ctx.globals["artifact_manifest"].entries
        if entry.artifact_type == "capstone_data_transfer"
    ]
    assert transfer_entries
    assert transfer_entries[0].producing_step == "push_capstone_data"
    assert transfer_entries[0].metadata["transfer_direction"] == "push"


@pytest.mark.asyncio
async def test_prepare_capstone_data_capstone_complete_pulls_when_requested(
    tmp_path, monkeypatch
):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    dataset_1 = tmp_path / "dataset_one"
    dataset_2 = tmp_path / "dataset_two"
    for dataset in (dataset_1, dataset_2):
        for class_name in ("cats", "dogs"):
            for index in range(3):
                _write_tiny_image(dataset / class_name / f"{class_name}-{index}.jpg")
    commands = []
    raw_s3_url = "s3://secret-capstone-bucket/team-a/capstone"

    def fake_run_command(cmd, cwd=None, timeout=60):
        commands.append((cmd, cwd, timeout))
        if cmd == ["dvc", "init", "--no-scm"]:
            (project_path / ".dvc").mkdir(exist_ok=True)
            (project_path / ".dvc" / "config").write_text("[core]\n")
        elif cmd[:2] == ["dvc", "add"]:
            package_path = cmd[2]
            (project_path / f"{package_path}.dvc").write_text(
                f"outs:\n- path: {package_path}\n"
            )
        elif cmd[:3] == ["dvc", "remote", "add"]:
            (project_path / ".dvc" / "config").write_text(
                "[core]\n    remote = capstone\n"
                '[\'remote "capstone"\']\n'
                f"    url = {raw_s3_url}\n"
            )
        return {"success": True, "stdout": "ok", "stderr": "", "returncode": 0}

    monkeypatch.setattr(mcp_mlops_tools, "check_tool_installed", lambda tool: tool == "dvc")
    monkeypatch.setattr(mcp_mlops_tools, "run_command", fake_run_command)
    monkeypatch.setattr(
        mcp_mlops_tools,
        "_validate_s3_credential_capability",
        lambda remote_url: {
            "passed": True,
            "status": "validated",
            "identity": {
                "account": "123456789012",
                "arn": "arn:aws:iam::123456789012:user/capstone",
                "user_id": "AIDAEXAMPLE",
            },
            "bucket_reachable": True,
            "prefix_checked": True,
            "next_actions": [],
        },
    )
    agent = AgentLoop(prompts_dir=str(prompts_dir), auto_approve=True)

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Phase 4 registry path must not call decision"),
        ),
    ):
        result = await agent.run(
            "Prepare capstone data "
            f"dataset_1_path={dataset_1} dataset_2_path={dataset_2} "
            "completion_mode=capstone_complete "
            "dvc_transfer_direction=pull "
            f"dvc_remote_name=capstone dvc_remote_url={raw_s3_url}",
            str(project_path),
        )

    command_names = [command[0][:2] for command in commands]
    assert ["dvc", "push"] not in command_names
    assert ["dvc", "pull"] in command_names
    assert "s3_transfer_completed:observed:passed" in result
    transfer_entries = [
        entry
        for entry in agent.ctx.globals["artifact_manifest"].entries
        if entry.artifact_type == "capstone_data_transfer"
    ]
    assert transfer_entries[0].producing_step == "pull_capstone_data"
    assert transfer_entries[0].metadata["transfer_direction"] == "pull"
