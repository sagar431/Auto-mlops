from unittest.mock import AsyncMock, patch

import pytest

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
    _write_tiny_image(dataset_1 / "cats" / "cat-1.jpg")
    _write_tiny_image(dataset_1 / "dogs" / "dog-1.jpg")
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
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.BLOCKED
    assert "two_dataset_paths_provided:observed:passed" in result
    assert "two_dataset_layouts_supported:observed:passed" in result
    assert "existing_train_test_split" in result
    assert not (project_path / ".auto_mlops").exists()
    assert not (project_path / ".dvc").exists()
    assert not (project_path / "data" / "capstone").exists()
