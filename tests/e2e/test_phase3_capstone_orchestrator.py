from unittest.mock import AsyncMock, patch

import pytest

from agent.agent_loop import AgentLoop
from workflow.registry import WorkflowStatus


@pytest.mark.asyncio
async def test_capstone_orchestrator_records_skeleton_and_blocks_future_capabilities(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")

    project_path = tmp_path / "project"
    project_path.mkdir()
    agent = AgentLoop(prompts_dir=str(prompts_dir))

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Capstone registry path must not call perception"),
        ),
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("Capstone registry path must not call decision"),
        ),
    ):
        result = await agent.run("Build full capstone pipeline", str(project_path))

    plan_path = project_path / ".auto_mlops" / "capstone" / "orchestrator_plan.json"
    assert plan_path.exists()
    assert agent.workflow_selection.workflow_id == "build_capstone_pipeline"
    assert agent.status == "paused"
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.BLOCKED
    assert "contract_status: blocked" in result
    assert "capstone_pipeline_ready" in result
    assert "train_until_better" in result
    assert "S3 DVC remote automation" in result
    assert "AWS Lambda serverless" in result
    assert "final report" in result
