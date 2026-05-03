import os
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from agent.agent_loop import AgentLoop
from workflow.registry import ArtifactManifest, VerificationResult, WorkflowStatus


def _prompts_dir(tmp_path: Path) -> str:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(parents=True)
    (prompts_dir / "perception_prompt.txt").write_text("Perception")
    (prompts_dir / "decision_prompt.txt").write_text("Decision")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize")
    (prompts_dir / "improvement_prompt.txt").write_text("Improve")
    return str(prompts_dir)


def _local_ml_project(tmp_path: Path) -> Path:
    project = tmp_path / "local_python_ml_project"
    (project / "src").mkdir(parents=True)
    (project / "data").mkdir()
    (project / "configs").mkdir()
    (project / "src" / "train.py").write_text(
        "def train():\n"
        "    return {'accuracy': 0.91}\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    train()\n"
    )
    (project / "data" / "train.csv").write_text("feature,label\n1,0\n2,1\n")
    (project / "requirements.txt").write_text("scikit-learn\n")
    return project


def _local_setup_tools(call_order: list[str]) -> ModuleType:
    tools = ModuleType("phase1_local_setup_tools")

    def analyze_project_config(project_path: str) -> dict:
        call_order.append("analyze_project_config")
        project = Path(project_path)
        return {
            "success": True,
            "framework": "python",
            "has_train_script": (project / "src" / "train.py").exists(),
            "has_data": (project / "data" / "train.csv").exists(),
        }

    def create_hydra_config(project_path: str) -> dict:
        call_order.append("create_hydra_config")
        config_path = Path(project_path) / "configs" / "config.yaml"
        config_path.write_text("defaults:\n  - _self_\ntraining:\n  seed: 42\n")
        return {
            "success": True,
            "verification_results": [
                {
                    "check_name": "hydra_config_validates",
                    "evidence_type": "observed",
                    "source_step": "create_or_validate_hydra_config",
                    "passed": True,
                    "evidence": "Hydra config parsed from configs/config.yaml.",
                }
            ],
            "artifact_manifest": {
                "entries": [
                    {
                        "artifact_type": "configuration",
                        "producing_step": "create_or_validate_hydra_config",
                        "state": "generated",
                        "path": "configs/config.yaml",
                    }
                ]
            },
        }

    def init_dvc_repo(project_path: str, no_scm: bool = False) -> dict:
        call_order.append("init_dvc_repo")
        dvc_config = Path(project_path) / ".dvc" / "config"
        dvc_config.parent.mkdir()
        dvc_config.write_text(f"[core]\n    no_scm = {str(no_scm).lower()}\n")
        return {
            "success": True,
            "verification_results": [
                {
                    "check_name": "dvc_repo_exists",
                    "evidence_type": "observed",
                    "source_step": "initialize_dvc",
                    "passed": True,
                    "evidence": "Local .dvc/config exists; no remote configured.",
                }
            ],
        }

    def configure_dvc_remote(project_path: str) -> dict:
        call_order.append("configure_dvc_remote")
        return {
            "success": True,
            "skipped": True,
            "reason": "No DVC remote requested for local Phase 1 setup.",
        }

    def add_data_to_dvc(project_path: str, data_path: str = "data/train.csv") -> dict:
        call_order.append("add_data_to_dvc")
        tracked_path = Path(project_path) / data_path
        tracked_path.with_name(f"{tracked_path.name}.dvc").write_text(
            f"outs:\n  - path: {tracked_path.name}\n"
        )
        return {"success": True, "tracked_path": data_path}

    def create_dvc_pipeline(project_path: str) -> dict:
        call_order.append("create_dvc_pipeline")
        dvc_yaml = Path(project_path) / "dvc.yaml"
        dvc_yaml.write_text("stages:\n  train:\n    cmd: python src/train.py\n")
        return {
            "success": True,
            "verification_results": [
                {
                    "check_name": "dvc_yaml_parseable",
                    "evidence_type": "observed",
                    "source_step": "create_dvc_yaml",
                    "passed": True,
                    "evidence": "dvc.yaml contains a train stage.",
                }
            ],
            "artifact_manifest": {
                "entries": [
                    {
                        "artifact_type": "pipeline_definition",
                        "producing_step": "create_dvc_yaml",
                        "state": "generated",
                        "path": "dvc.yaml",
                    }
                ]
            },
        }

    def init_mlflow_experiment(project_path: str, experiment_name: str = "mlops") -> dict:
        call_order.append("init_mlflow_experiment")
        (Path(project_path) / "mlruns").mkdir()
        return {
            "success": True,
            "verification_results": [
                {
                    "check_name": "mlflow_experiment_exists",
                    "evidence_type": "declared",
                    "source_step": "initialize_mlflow_experiment",
                    "passed": True,
                    "evidence": "Local mlruns directory initialized.",
                }
            ],
        }

    def create_ml_dockerfile(project_path: str) -> dict:
        call_order.append("create_ml_dockerfile")
        dockerfile = Path(project_path) / "Dockerfile"
        dockerfile.write_text("FROM python:3.12-slim\nCOPY . /app\n")
        return {
            "success": True,
            "verification_results": [
                {
                    "check_name": "dockerfile_build_evidence",
                    "evidence_type": "declared",
                    "source_step": "create_dockerfile",
                    "passed": True,
                    "evidence": "Dockerfile generated for local validation; image not built.",
                }
            ],
            "artifact_manifest": {
                "entries": [
                    {
                        "artifact_type": "container_definition",
                        "producing_step": "create_dockerfile",
                        "state": "generated",
                        "path": "Dockerfile",
                    }
                ]
            },
        }

    def create_github_workflow(project_path: str) -> dict:
        call_order.append("create_github_workflow")
        workflow = Path(project_path) / ".github" / "workflows" / "mlops.yml"
        workflow.parent.mkdir(parents=True)
        workflow.write_text("name: mlops\non: [push]\njobs: {}\n")
        return {
            "success": True,
            "artifact_manifest": {
                "entries": [
                    {
                        "artifact_type": "automation_workflow",
                        "producing_step": "create_ci_workflow",
                        "state": "generated",
                        "path": ".github/workflows/mlops.yml",
                    }
                ]
            },
        }

    tools.analyze_project_config = analyze_project_config
    tools.create_hydra_config = create_hydra_config
    tools.init_dvc_repo = init_dvc_repo
    tools.configure_dvc_remote = configure_dvc_remote
    tools.add_data_to_dvc = add_data_to_dvc
    tools.create_dvc_pipeline = create_dvc_pipeline
    tools.init_mlflow_experiment = init_mlflow_experiment
    tools.create_ml_dockerfile = create_ml_dockerfile
    tools.create_github_workflow = create_github_workflow
    return tools


@pytest.mark.asyncio
async def test_local_setup_pipeline_requires_approval_then_reaches_contract_success(tmp_path):
    project = _local_ml_project(tmp_path)
    query = f"Set up MLOps for this local Python ML project at {project}"

    blocked_call_order: list[str] = []
    blocked_agent = AgentLoop(
        prompts_dir=_prompts_dir(tmp_path / "blocked"),
        tools_module=_local_setup_tools(blocked_call_order),
    )

    with (
        patch.object(
            blocked_agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("registry setup selection should run before perception"),
        ),
        patch.object(
            blocked_agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("setup_pipeline must not use prompt-authored plans"),
        ),
    ):
        blocked_output = await blocked_agent.run(query=query, project_path=str(project))

    assert blocked_agent.workflow_selection.workflow_id == "setup_pipeline"
    assert blocked_agent.status == "paused"
    assert "Approval required before executing workflow step" in blocked_output
    assert "create_or_validate_hydra_config" in blocked_output
    assert blocked_call_order == []

    successful_call_order: list[str] = []
    events: list[dict] = []

    async def capture_event(event_type: str, data: dict) -> None:
        events.append({"type": event_type, "data": data})

    agent = AgentLoop(
        prompts_dir=_prompts_dir(tmp_path / "success"),
        tools_module=_local_setup_tools(successful_call_order),
        on_event=capture_event,
        auto_approve=True,
    )

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("registry setup execution must skip post-step perception"),
        ) as mock_perception,
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("setup_pipeline must not use prompt-authored plans"),
        ),
        patch.object(
            agent.summarizer,
            "summarize",
            new_callable=AsyncMock,
            side_effect=AssertionError("SuccessContract must derive setup_pipeline status"),
        ),
    ):
        output = await agent.run(query=query, project_path=str(project))

    template = agent.workflow_registry.get("setup_pipeline")
    assert agent.workflow_selection.workflow_id == "setup_pipeline"
    mock_perception.assert_not_awaited()
    assert [event["type"] for event in events].count("step_complete") == len(template.steps)
    assert successful_call_order == [
        step.tool_functions[0] for step in template.steps if step.tool_functions
    ]

    contract_status = agent.ctx.globals["contract_status"]
    assert contract_status.status is WorkflowStatus.SUCCEEDED
    assert contract_status.missing_evidence == ()
    assert contract_status.failed_checks == ()
    assert agent.ctx.globals["workflow_status"] is WorkflowStatus.SUCCEEDED
    assert agent.status == "success"

    verification_results = agent.ctx.globals["verification_results"]
    assert all(isinstance(result, VerificationResult) for result in verification_results)
    assert {result.check_name for result in verification_results} == {
        "hydra_config_validates",
        "dvc_repo_exists",
        "dvc_yaml_parseable",
        "mlflow_experiment_exists",
        "dockerfile_build_evidence",
    }

    artifact_manifest = agent.ctx.globals["artifact_manifest"]
    assert isinstance(artifact_manifest, ArtifactManifest)
    assert {(entry.artifact_type, entry.path) for entry in artifact_manifest.entries} == {
        ("configuration", "configs/config.yaml"),
        ("pipeline_definition", "dvc.yaml"),
        ("container_definition", "Dockerfile"),
        ("automation_workflow", ".github/workflows/mlops.yml"),
    }

    approval_records = agent.ctx.globals["approval_records"]
    assert {record.step_id for record in approval_records} == {
        gate.step_id for gate in template.approval_gates
    }
    assert all(record.status.value == "approved" for record in approval_records)

    assert "workflow_status: succeeded" in output
    assert "contract_status: succeeded" in output
    assert "missing_evidence: none" in output
    assert "failed_checks: none" in output
    assert "artifacts:" in output
    assert "configs/config.yaml" in output
    assert ".github/workflows/mlops.yml" in output
    assert "approvals:" in output


@pytest.mark.asyncio
async def test_local_setup_pipeline_real_tools_reach_contract_success_without_perception(
    tmp_path, monkeypatch
):
    project = _local_ml_project(tmp_path)
    query = f"Set up MLOps for this local Python ML project at {project}"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dvc = fake_bin / "dvc"
    fake_dvc.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "import sys\n"
        "cwd = Path.cwd()\n"
        "args = sys.argv[1:]\n"
        "if args and args[0] == 'init':\n"
        "    (cwd / '.dvc').mkdir(exist_ok=True)\n"
        "    (cwd / '.dvc' / 'config').write_text('[core]\\n    no_scm = true\\n')\n"
        "    raise SystemExit(0)\n"
        "if len(args) >= 2 and args[0] == 'add':\n"
        "    for rel_path in args[1:]:\n"
        "        (cwd / f'{rel_path}.dvc').write_text('outs:\\n  - path: ' + rel_path + '\\n')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(0)\n"
    )
    fake_dvc.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")

    import mcp_mlops_tools

    agent = AgentLoop(
        prompts_dir=_prompts_dir(tmp_path / "real-tools"),
        tools_module=mcp_mlops_tools,
        auto_approve=True,
    )

    with (
        patch.object(
            agent.perception,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("registry setup execution must skip post-step perception"),
        ) as mock_perception,
        patch.object(
            agent.decision,
            "run",
            new_callable=AsyncMock,
            side_effect=AssertionError("setup_pipeline must not use prompt-authored plans"),
        ),
        patch.object(
            agent.summarizer,
            "summarize",
            new_callable=AsyncMock,
            side_effect=AssertionError("SuccessContract must derive setup_pipeline status"),
        ),
    ):
        output = await agent.run(query=query, project_path=str(project))

    mock_perception.assert_not_awaited()
    assert agent.ctx.globals["contract_status"].status is WorkflowStatus.SUCCEEDED
    assert "workflow_status: succeeded" in output
    assert "missing_evidence: none" in output
    assert "failed_checks: none" in output
    assert "configs/config.yaml" in output
    assert "dvc.yaml" in output
    assert "Dockerfile" in output
    assert ".github/workflows/ml-pipeline.yml" in output
