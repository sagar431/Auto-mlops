import pytest

from workflow.registry import WorkflowRegistry, get_workflow_registry


def test_registry_returns_setup_pipeline_by_id():
    registry = get_workflow_registry()

    template = registry.get("setup_pipeline")

    assert template.workflow_id == "setup_pipeline"
    assert template.name == "Setup Pipeline"


def test_setup_pipeline_declares_ordered_workflow_steps():
    template = get_workflow_registry().get("setup_pipeline")

    assert [step.step_id for step in template.steps] == [
        "analyze_project_structure",
        "create_or_validate_hydra_config",
        "initialize_dvc",
        "configure_dvc_remote",
        "add_data_to_dvc",
        "create_dvc_yaml",
        "initialize_mlflow_experiment",
        "create_dockerfile",
        "create_ci_workflow",
    ]
    assert [step.order for step in template.steps] == list(range(1, 10))


def test_setup_pipeline_declares_inputs_separately_from_step_args():
    template = get_workflow_registry().get("setup_pipeline")

    assert [workflow_input.name for workflow_input in template.required_inputs] == ["project_path"]
    assert all("project_path" not in step.default_args for step in template.steps)


def test_setup_pipeline_declares_success_contract_check_names():
    template = get_workflow_registry().get("setup_pipeline")

    assert [check.name for check in template.success_contract.checks] == [
        "hydra_config_validates",
        "dvc_repo_exists",
        "dvc_yaml_parseable",
        "mlflow_experiment_exists",
        "dockerfile_build_evidence",
        "generated_files_reported",
    ]


def test_setup_pipeline_declares_artifact_requirements_as_data():
    template = get_workflow_registry().get("setup_pipeline")

    assert [artifact.name for artifact in template.artifact_requirements] == [
        "hydra_config",
        "dvc_yaml",
        "dockerfile",
        "ci_workflow",
    ]


def test_workflow_step_is_separate_from_tool_function():
    template = get_workflow_registry().get("setup_pipeline")
    hydra_step = template.step_by_id("create_or_validate_hydra_config")

    assert hydra_step.tool_functions == ("create_hydra_config", "validate_hydra_config")
    assert hydra_step.step_id != hydra_step.tool_functions[0]


def test_registry_rejects_fake_templates():
    template = get_workflow_registry().get("setup_pipeline")
    fake_template = template.__class__(
        workflow_id="fake",
        name="Fake",
        description="No contract detail.",
        required_inputs=template.required_inputs,
        steps=(),
        success_contract=template.success_contract.__class__(checks=()),
        artifact_requirements=(),
    )

    with pytest.raises(ValueError, match="Fake Template"):
        WorkflowRegistry([fake_template])
