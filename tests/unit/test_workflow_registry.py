import pytest

from workflow.registry import WorkflowRegistry, WorkflowStatus, get_workflow_registry


def test_registry_returns_setup_pipeline_by_id():
    registry = get_workflow_registry()

    template = registry.get("setup_pipeline")

    assert template.workflow_id == "setup_pipeline"
    assert template.name == "Setup Pipeline"


def test_registry_contains_exactly_phase_0_templates():
    registry = get_workflow_registry()

    assert registry.workflow_ids == (
        "setup_pipeline",
        "deploy_litserve_gpu",
        "deploy_gpu_inference",
        "deploy_gradio_demo",
        "deploy_kserve_production",
    )
    for excluded_workflow_id in (
        "rollback",
        "monitor_and_alert",
        "train_and_track",
        "train_until_better",
    ):
        with pytest.raises(KeyError):
            registry.get(excluded_workflow_id)


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


def test_deploy_litserve_gpu_requires_observed_runtime_evidence():
    template = get_workflow_registry().get("deploy_litserve_gpu")

    observed_checks = {
        check.name
        for check in template.success_contract.checks
        if check.evidence_type == "observed"
    }

    assert {
        "gpu_detection_recorded",
        "server_start_command_recorded",
        "health_result_recorded",
        "prediction_result_recorded",
    }.issubset(observed_checks)


def test_deploy_gpu_inference_declares_backend_branches():
    template = get_workflow_registry().get("deploy_gpu_inference")

    assert [branch.name for branch in template.branches] == [
        "litserve",
        "gradio",
        "vllm",
        "kserve",
        "torchserve",
        "fastapi_lambda_cpu",
    ]
    assert all(branch.selection_rule for branch in template.branches)


def test_deployment_templates_declare_routing_and_approval_metadata():
    registry = get_workflow_registry()
    litserve = registry.get("deploy_litserve_gpu")
    gpu_inference = registry.get("deploy_gpu_inference")
    kserve = registry.get("deploy_kserve_production")

    assert "Lambda Labs GPU" in litserve.routing_aliases
    assert "AWS Lambda serverless" in litserve.negative_routing_rules
    assert "serve this LLM with vLLM" in gpu_inference.routing_aliases
    assert "KServe canary rollout" in kserve.routing_aliases

    litserve_risk_categories = {
        risk
        for gate in litserve.approval_gates
        for risk in gate.risk_categories
    }
    assert {
        "writes_project_files",
        "builds_image",
        "starts_server",
        "exposes_port",
    }.issubset(litserve_risk_categories)


def test_select_workflow_returns_structured_litserve_gpu_decision():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Deploy this model on Lambda Labs GPU")

    assert selection.workflow_id == "deploy_litserve_gpu"
    assert selection.confidence >= 0.8
    assert selection.matched_aliases == ("Lambda Labs GPU",)
    assert "deploy_gpu_inference" in selection.rejected_workflows
    assert selection.missing_inputs == ()
    assert "Lambda Labs GPU" in selection.selection_reason


def test_select_workflow_records_gpu_inference_branch_evidence():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Serve this LLM with vLLM")

    assert selection.workflow_id == "deploy_gpu_inference"
    assert selection.status is WorkflowStatus.PENDING
    assert selection.matched_aliases == ("serve this LLM with vLLM",)
    assert selection.matched_branches == ("vllm",)
    assert "vllm" in selection.selection_reason


def test_select_workflow_blocks_ambiguous_request_instead_of_setup_fallback():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Deploy my model")

    assert selection.status is WorkflowStatus.BLOCKED
    assert selection.workflow_id is None
    assert selection.confidence < 0.5
    assert selection.missing_inputs == ("workflow_intent",)
    assert "setup_pipeline" not in selection.rejected_workflows
    assert "No registry routing alias matched" in selection.selection_reason


def test_select_workflow_blocks_conflicting_alias_matches():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Create a Gradio demo and deploy to Kubernetes")

    assert selection.status is WorkflowStatus.BLOCKED
    assert selection.workflow_id is None
    assert selection.confidence < 0.5
    assert selection.matched_aliases == ("Create a Gradio demo", "deploy to Kubernetes")
    assert selection.rejected_workflows == ("deploy_gradio_demo", "deploy_kserve_production")
    assert selection.missing_inputs == ("workflow_intent",)
    assert "Multiple registry routing aliases matched" in selection.selection_reason


def test_deployment_templates_are_non_fake_ordered_templates():
    registry = get_workflow_registry()

    for workflow_id in (
        "deploy_litserve_gpu",
        "deploy_gpu_inference",
        "deploy_gradio_demo",
        "deploy_kserve_production",
    ):
        template = registry.get(workflow_id)

        assert [workflow_input.name for workflow_input in template.required_inputs] == [
            "project_path"
        ]
        assert [step.order for step in template.steps] == list(range(1, len(template.steps) + 1))
        assert template.success_contract.checks
        assert template.routing_aliases
        assert template.approval_gates


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
