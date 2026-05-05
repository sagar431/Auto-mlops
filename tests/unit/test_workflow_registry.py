from datetime import datetime, timezone

import pytest

from workflow.registry import (
    ApprovalGate,
    ApprovalRecord,
    ApprovalStatus,
    ArtifactManifest,
    ArtifactManifestEntry,
    DeploymentCheckResult,
    DeploymentReport,
    GpuEvidence,
    LatencySummary,
    RiskCategory,
    RollbackPlan,
    VerificationResult,
    WorkflowRegistry,
    WorkflowStatus,
    get_workflow_registry,
)


def test_registry_returns_setup_pipeline_by_id():
    registry = get_workflow_registry()

    template = registry.get("setup_pipeline")

    assert template.workflow_id == "setup_pipeline"
    assert template.name == "Setup Pipeline"


def test_registry_contains_phase_3_training_detection_template():
    registry = get_workflow_registry()

    assert registry.workflow_ids == (
        "setup_pipeline",
        "detect_training_project",
        "train_and_track",
        "deploy_litserve_preflight",
        "deploy_litserve_gpu",
        "deploy_gpu_inference",
        "deploy_gradio_demo",
        "deploy_kserve_production",
    )
    for excluded_workflow_id in (
        "rollback",
        "monitor_and_alert",
        "train_until_better",
    ):
        with pytest.raises(KeyError):
            registry.get(excluded_workflow_id)


def test_detect_training_project_declares_detection_only_template():
    template = get_workflow_registry().get("detect_training_project")

    assert template.workflow_id == "detect_training_project"
    assert [workflow_input.name for workflow_input in template.required_inputs] == [
        "project_path"
    ]
    assert [step.step_id for step in template.steps] == ["detect_training_project"]
    assert template.steps[0].tool_functions == ("detect_training_project",)
    assert [check.name for check in template.success_contract.checks] == [
        "training_project_detected",
        "training_entrypoint_detected",
        "hydra_config_detected",
        "dvc_or_data_evidence_detected",
        "pytorch_timm_signals_detected",
        "test_command_detected",
        "output_artifact_candidates_detected",
    ]
    assert all(check.evidence_type == "observed" for check in template.success_contract.checks)


def test_train_and_track_declares_bounded_training_template():
    template = get_workflow_registry().get("train_and_track")

    assert template.workflow_id == "train_and_track"
    assert [workflow_input.name for workflow_input in template.required_inputs] == [
        "project_path",
        "timeout_seconds",
        "max_epochs",
        "device",
        "data_subset",
    ]
    assert [step.step_id for step in template.steps] == [
        "detect_training_project",
        "run_bounded_training",
    ]
    assert template.step_by_id("detect_training_project").tool_functions == (
        "detect_training_project",
    )
    assert template.step_by_id("run_bounded_training").tool_functions == (
        "run_bounded_training",
    )
    assert [check.name for check in template.success_contract.checks] == [
        "training_project_detected",
        "training_entrypoint_detected",
        "hydra_config_detected",
        "bounded_training_controls_present",
        "bounded_training_command_completed",
        "training_metric_captured",
        "training_artifact_captured",
        "training_run_evidence_captured",
    ]
    assert all(check.evidence_type == "observed" for check in template.success_contract.checks)


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


def test_artifact_manifest_entry_requires_path_or_uri_and_known_state():
    entry = ArtifactManifestEntry(
        path="conf/config.yaml",
        artifact_type="configuration",
        producing_step="create_or_validate_hydra_config",
        state="generated",
    )

    assert entry.path == "conf/config.yaml"
    assert entry.uri is None
    assert entry.state == "generated"

    with pytest.raises(ValueError, match="path or uri"):
        ArtifactManifestEntry(
            artifact_type="configuration",
            producing_step="create_or_validate_hydra_config",
            state="generated",
        )

    with pytest.raises(ValueError, match="Unknown artifact state"):
        ArtifactManifestEntry(
            path="conf/config.yaml",
            artifact_type="configuration",
            producing_step="create_or_validate_hydra_config",
            state="reported",
        )


def test_artifact_manifest_rejects_unknown_producing_step_for_workflow():
    registry = get_workflow_registry()
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                path="conf/config.yaml",
                artifact_type="configuration",
                producing_step="not_a_workflow_step",
                state="generated",
            ),
        )
    )

    with pytest.raises(ValueError, match="Unknown artifact producing step"):
        registry.validate_artifact_manifest("setup_pipeline", manifest)


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
        "endpoint_url_recorded",
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


def test_gated_step_without_matching_approval_record_is_blocked():
    registry = get_workflow_registry()

    validation = registry.validate_step_approval(
        workflow_id="deploy_litserve_gpu",
        workflow_run_id="run-123",
        step_id="start_litserve_server",
        approval_records=(),
    )

    assert validation.status is WorkflowStatus.BLOCKED
    assert validation.step_id == "start_litserve_server"
    assert validation.risk_categories == ("starts_server", "exposes_port")
    assert validation.approval_record is None
    assert "approval" in validation.next_action


def test_gated_step_with_matching_approved_record_is_pending():
    registry = get_workflow_registry()
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    approval_record = ApprovalRecord(
        workflow_run_id="run-123",
        step_id="start_litserve_server",
        risk_categories=("starts_server", "exposes_port"),
        status="approved",
        approver="ops@example.com",
        timestamp=timestamp,
    )

    validation = registry.validate_step_approval(
        workflow_id="deploy_litserve_gpu",
        workflow_run_id="run-123",
        step_id="start_litserve_server",
        approval_records=(approval_record,),
    )

    assert approval_record.status is ApprovalStatus.APPROVED
    assert approval_record.approver == "ops@example.com"
    assert approval_record.timestamp == timestamp
    assert validation.status is WorkflowStatus.PENDING
    assert validation.approval_record == approval_record
    assert validation.next_action == "Approval satisfied; step may run."


def test_gated_step_with_matching_denied_record_fails_with_structured_reason():
    registry = get_workflow_registry()
    approval_record = ApprovalRecord(
        workflow_run_id="run-123",
        step_id="start_litserve_server",
        risk_categories=("starts_server", "exposes_port"),
        status="denied",
        approver="ops@example.com",
        timestamp=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    )

    validation = registry.validate_step_approval(
        workflow_id="deploy_litserve_gpu",
        workflow_run_id="run-123",
        step_id="start_litserve_server",
        approval_records=(approval_record,),
    )

    assert validation.status is WorkflowStatus.FAILED
    assert validation.risk_categories == ("starts_server", "exposes_port")
    assert validation.approval_record == approval_record
    assert "denied" in validation.next_action
    assert "ops@example.com" in validation.next_action


def test_approval_gate_uses_controlled_risk_categories():
    gate = ApprovalGate(
        step_id="start_server",
        risk_categories=("starts_server", "exposes_port"),
    )

    assert gate.risk_categories == (RiskCategory.STARTS_SERVER, RiskCategory.EXPOSES_PORT)

    with pytest.raises(ValueError, match="Unknown risk category"):
        ApprovalGate(
            step_id="delete_cluster",
            risk_categories=("deletes_cluster",),
        )


def test_select_workflow_returns_structured_litserve_gpu_decision():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Deploy this model on Lambda Labs GPU")

    assert selection.workflow_id == "deploy_litserve_gpu"
    assert selection.confidence >= 0.8
    assert selection.matched_aliases == ("Lambda Labs GPU",)
    assert "deploy_gpu_inference" in selection.rejected_workflows
    assert selection.missing_inputs == ()
    assert "Lambda Labs GPU" in selection.selection_reason


def test_select_workflow_routes_litserve_local_preflight_to_preflight_contract():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Prepare LitServe deployment locally")
    template = registry.get("deploy_litserve_preflight")

    assert selection.workflow_id == "deploy_litserve_preflight"
    assert selection.status is WorkflowStatus.PENDING
    assert selection.matched_aliases == ("Prepare LitServe deployment locally",)
    assert "deploy_litserve_gpu" in selection.rejected_workflows

    checks_by_name = {check.name: check for check in template.success_contract.checks}
    assert set(checks_by_name) == {
        "model_artifact_selected",
        "litserve_app_artifact_ready",
        "dockerfile_artifact_ready",
        "launch_command_recorded",
        "missing_live_evidence_recorded",
    }
    assert all(check.evidence_type != "observed" for check in checks_by_name.values())
    assert [gate.step_id for gate in template.approval_gates] == [
        "generate_or_validate_litserve_app",
        "generate_or_validate_dockerfile",
        "future_server_start",
        "future_docker_build",
        "future_cloud_gpu_launch",
    ]


def test_litserve_preflight_succeeds_from_local_artifacts_and_declared_missing_live_evidence():
    registry = get_workflow_registry()
    verification_results = (
        VerificationResult(
            check_name="launch_command_recorded",
            evidence_type="declared",
            source_step="record_launch_command",
            passed=True,
            evidence="python deployment/litserve/server.py",
        ),
        VerificationResult(
            check_name="missing_live_evidence_recorded",
            evidence_type="declared",
            source_step="record_missing_live_evidence",
            passed=True,
            evidence="gpu_detection, server_start, /health, /predict, endpoint_url missing",
        ),
    )
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                path="models/model_preflight.pt",
                artifact_type="model_artifact",
                producing_step="select_or_create_model_artifact",
                state="selected",
            ),
            ArtifactManifestEntry(
                path="deployment/litserve/server.py",
                artifact_type="serving_application",
                producing_step="generate_or_validate_litserve_app",
                state="generated",
            ),
            ArtifactManifestEntry(
                path="Dockerfile",
                artifact_type="container_definition",
                producing_step="generate_or_validate_dockerfile",
                state="generated",
            ),
        )
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_preflight",
        verification_results=verification_results,
        artifact_manifest=manifest,
    )

    assert validation.status is WorkflowStatus.SUCCEEDED
    assert validation.missing_evidence == ()
    assert validation.failed_checks == ()

    with pytest.raises(ValueError, match="Unknown artifact producing step"):
        registry.validate_success_contract(
            "deploy_litserve_gpu",
            verification_results=verification_results,
            artifact_manifest=manifest,
        )


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


def test_select_workflow_routes_natural_language_training_request():
    registry = get_workflow_registry()

    selection = registry.select_workflow("Train this project")

    assert selection.workflow_id == "train_and_track"
    assert selection.status is WorkflowStatus.PENDING
    assert selection.matched_aliases == ("train this project",)
    assert selection.missing_inputs == ()
    assert "train_and_track" in selection.selection_reason


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


@pytest.mark.parametrize(
    (
        "prompt",
        "workflow_id",
        "matched_alias",
        "rejected_workflows",
        "required_contract_checks",
    ),
    (
        (
            "Deploy this model on Lambda Labs GPU",
            "deploy_litserve_gpu",
            "Lambda Labs GPU",
            ("deploy_litserve_preflight", "deploy_gpu_inference"),
            (
                "gpu_detection_recorded",
                "server_start_command_recorded",
                "health_result_recorded",
                "prediction_result_recorded",
                "rollback_plan_exists",
            ),
        ),
        (
            "Create a Gradio demo",
            "deploy_gradio_demo",
            "Create a Gradio demo",
            (),
            (
                "app_file_exists",
                "launch_command_exists",
                "sample_prediction_path_documented",
                "rollback_plan_exists",
            ),
        ),
        (
            "Deploy to KServe with canary rollout",
            "deploy_kserve_production",
            "Deploy to KServe with canary rollout",
            (),
            (
                "kubernetes_manifests_validate",
                "registry_image_reference_exists",
                "canary_rollback_plan_exists",
                "dry_run_result_recorded",
            ),
        ),
        (
            "Run this model and tell me if GPU is being used",
            "deploy_gpu_inference",
            "run this model and tell me if GPU is being used",
            (),
            (
                "gpu_cuda_status_recorded",
                "gpu_utilization_evidence_captured",
                "rollback_plan_exists",
            ),
        ),
        (
            "Detect this training project",
            "detect_training_project",
            "detect this training project",
            (),
            (
                "training_project_detected",
                "training_entrypoint_detected",
                "hydra_config_detected",
            ),
        ),
        (
            "Set up MLOps for this project",
            "setup_pipeline",
            "Set up MLOps",
            (),
            (
                "hydra_config_validates",
                "dvc_repo_exists",
                "generated_files_reported",
            ),
        ),
    ),
)
def test_phase_0_prompt_contract_matrix_selects_expected_workflow(
    prompt,
    workflow_id,
    matched_alias,
    rejected_workflows,
    required_contract_checks,
):
    registry = get_workflow_registry()

    selection = registry.select_workflow(prompt)
    template = registry.get(workflow_id)

    assert selection.status is WorkflowStatus.PENDING
    assert selection.workflow_id == workflow_id
    assert selection.matched_aliases == (matched_alias,)
    assert selection.rejected_workflows == rejected_workflows
    assert selection.missing_inputs == ()
    assert [workflow_input.name for workflow_input in template.required_inputs] == [
        "project_path"
    ]
    declared_checks = {check.name for check in template.success_contract.checks}
    assert set(required_contract_checks).issubset(declared_checks)


def test_phase_0_prompt_contract_matrix_keeps_lambda_labs_gpu_distinct_from_aws_lambda():
    registry = get_workflow_registry()

    lambda_labs_selection = registry.select_workflow("Deploy this model on Lambda Labs GPU")
    aws_lambda_selection = registry.select_workflow("Deploy this model to AWS Lambda serverless")

    assert lambda_labs_selection.workflow_id == "deploy_litserve_gpu"
    assert "deploy_gpu_inference" in lambda_labs_selection.rejected_workflows
    assert aws_lambda_selection.status is WorkflowStatus.BLOCKED
    assert aws_lambda_selection.workflow_id is None
    assert "deploy_litserve_gpu" in aws_lambda_selection.rejected_workflows
    assert "deploy_kserve_production" in aws_lambda_selection.rejected_workflows
    assert aws_lambda_selection.missing_inputs == ("workflow_intent",)


@pytest.mark.parametrize(
    ("workflow_id", "observed_check_names"),
    (
        (
            "deploy_litserve_gpu",
            (
                "gpu_detection_recorded",
                "server_start_command_recorded",
                "health_result_recorded",
                "prediction_result_recorded",
            ),
        ),
        (
            "deploy_gpu_inference",
            (
                "gpu_cuda_status_recorded",
                "server_start_command_recorded",
                "health_check_passes",
                "prediction_test_passes",
                "gpu_utilization_evidence_captured",
                "latency_metrics_recorded",
            ),
        ),
        (
            "deploy_kserve_production",
            (
                "kubernetes_manifests_validate",
                "dry_run_result_recorded",
            ),
        ),
    ),
)
def test_phase_0_deployment_contract_matrix_requires_observed_runtime_checks(
    workflow_id,
    observed_check_names,
):
    template = get_workflow_registry().get(workflow_id)
    checks_by_name = {check.name: check for check in template.success_contract.checks}

    for check_name in observed_check_names:
        assert checks_by_name[check_name].evidence_type == "observed"


def test_success_contract_blocks_missing_required_evidence():
    registry = get_workflow_registry()

    validation = registry.validate_success_contract("deploy_litserve_gpu", verification_results=())

    assert validation.status is WorkflowStatus.BLOCKED
    assert validation.missing_evidence
    assert validation.failed_checks == ()
    assert validation.missing_evidence[0].check_name == "gpu_detection_recorded"
    assert validation.missing_evidence[0].expected_evidence_type == "observed"
    assert validation.missing_evidence[0].source_step == "detect_gpu_cuda"
    assert validation.missing_evidence[0].next_action


def test_declared_evidence_does_not_satisfy_observed_contract_check():
    registry = get_workflow_registry()
    declared_health_result = VerificationResult(
        check_name="health_result_recorded",
        evidence_type="declared",
        source_step="test_health_endpoint",
        passed=True,
        evidence="summary says /health passed",
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=(declared_health_result,),
    )

    health_failure = next(
        failure
        for failure in validation.missing_evidence
        if failure.check_name == "health_result_recorded"
    )
    assert validation.status is WorkflowStatus.BLOCKED
    assert health_failure.expected_evidence_type == "observed"
    assert health_failure.source_step == "test_health_endpoint"
    assert health_failure.actual_evidence == (declared_health_result,)
    assert "/health" in health_failure.actual_evidence[0].evidence


def test_declared_endpoint_evidence_does_not_satisfy_litserve_gpu_contract():
    registry = get_workflow_registry()
    declared_endpoint_result = VerificationResult(
        check_name="endpoint_url_recorded",
        evidence_type="declared",
        source_step="capture_logs_and_endpoint",
        passed=True,
        evidence="endpoint_url=http://127.0.0.1:8000",
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=(declared_endpoint_result,),
    )

    endpoint_failure = next(
        failure
        for failure in validation.missing_evidence
        if failure.check_name == "endpoint_url_recorded"
    )
    assert validation.status is WorkflowStatus.BLOCKED
    assert endpoint_failure.expected_evidence_type == "observed"
    assert endpoint_failure.source_step == "capture_logs_and_endpoint"
    assert endpoint_failure.actual_evidence == (declared_endpoint_result,)


def test_failed_required_check_derives_failed_workflow_status():
    registry = get_workflow_registry()
    template = registry.get("deploy_litserve_gpu")
    verification_results = tuple(
        VerificationResult(
            check_name=check.name,
            evidence_type="observed" if check.evidence_type == "observed" else "declared",
            source_step=check.source_step,
            passed=check.name != "prediction_result_recorded",
            evidence=f"{check.name} evidence",
        )
        for check in template.success_contract.checks
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=verification_results,
    )

    assert validation.status is WorkflowStatus.FAILED
    assert validation.missing_evidence == ()
    assert [failure.check_name for failure in validation.failed_checks] == [
        "prediction_result_recorded"
    ]
    assert validation.failed_checks[0].next_action


def test_setup_pipeline_generated_files_contract_uses_artifact_manifest():
    registry = get_workflow_registry()
    template = registry.get("setup_pipeline")
    verification_results = tuple(
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
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                path="conf/config.yaml",
                artifact_type="configuration",
                producing_step="create_or_validate_hydra_config",
                state="generated",
            ),
            ArtifactManifestEntry(
                path="dvc.yaml",
                artifact_type="pipeline_definition",
                producing_step="create_dvc_yaml",
                state="generated",
            ),
            ArtifactManifestEntry(
                path="Dockerfile",
                artifact_type="container_definition",
                producing_step="create_dockerfile",
                state="generated",
            ),
            ArtifactManifestEntry(
                path=".github/workflows/mlops.yml",
                artifact_type="automation_workflow",
                producing_step="create_ci_workflow",
                state="generated",
            ),
        )
    )

    validation = registry.validate_success_contract(
        "setup_pipeline",
        verification_results=verification_results,
        artifact_manifest=manifest,
    )

    assert validation.status is WorkflowStatus.SUCCEEDED
    assert validation.missing_evidence == ()
    assert validation.failed_checks == ()


def test_setup_pipeline_generated_files_contract_blocks_missing_manifest_entries():
    registry = get_workflow_registry()
    template = registry.get("setup_pipeline")
    verification_results = tuple(
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
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                path="conf/config.yaml",
                artifact_type="configuration",
                producing_step="create_or_validate_hydra_config",
                state="generated",
            ),
        )
    )

    validation = registry.validate_success_contract(
        "setup_pipeline",
        verification_results=verification_results,
        artifact_manifest=manifest,
    )

    assert validation.status is WorkflowStatus.BLOCKED
    assert [failure.check_name for failure in validation.missing_evidence] == [
        "generated_files_reported"
    ]
    assert validation.missing_evidence[0].actual_evidence == manifest.entries
    assert "dvc_yaml" in validation.missing_evidence[0].next_action


def test_litserve_deployment_contract_requires_selected_model_and_generated_serving_artifact():
    registry = get_workflow_registry()
    template = registry.get("deploy_litserve_gpu")
    verification_results = tuple(
        VerificationResult(
            check_name=check.name,
            evidence_type="observed" if check.evidence_type == "observed" else "declared",
            source_step=check.source_step,
            passed=True,
            evidence=f"{check.name} evidence",
        )
        for check in template.success_contract.checks
        if check.name != "litserve_files_generated"
    )
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                uri="models:/classifier/Production",
                artifact_type="model_artifact",
                producing_step="select_best_model_artifact",
                state="selected",
            ),
        )
    )

    blocked_validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=verification_results,
        artifact_manifest=manifest,
    )

    assert blocked_validation.status is WorkflowStatus.BLOCKED
    assert blocked_validation.missing_evidence[0].check_name == "litserve_files_generated"
    assert "litserve_api" in blocked_validation.missing_evidence[0].next_action

    complete_manifest = ArtifactManifest(
        entries=(
            *manifest.entries,
            ArtifactManifestEntry(
                path="serving/litserve_api.py",
                artifact_type="serving_application",
                producing_step="generate_litserve_api",
                state="generated",
            ),
        )
    )

    passed_validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=verification_results,
        artifact_manifest=complete_manifest,
    )

    assert passed_validation.status is WorkflowStatus.SUCCEEDED


def test_litserve_deployment_success_blocks_missing_rollback_readiness():
    registry = get_workflow_registry()
    template = registry.get("deploy_litserve_gpu")
    verification_results = tuple(
        VerificationResult(
            check_name=check.name,
            evidence_type="observed" if check.evidence_type == "observed" else "declared",
            source_step=check.source_step,
            passed=True,
            evidence=f"{check.name} evidence",
        )
        for check in template.success_contract.checks
        if check.name not in {"litserve_files_generated", "rollback_plan_exists"}
    )
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                uri="models:/classifier/Production",
                artifact_type="model_artifact",
                producing_step="select_best_model_artifact",
                state="selected",
            ),
            ArtifactManifestEntry(
                path="serving/litserve_api.py",
                artifact_type="serving_application",
                producing_step="generate_litserve_api",
                state="generated",
            ),
        )
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=verification_results,
        artifact_manifest=manifest,
    )

    assert validation.status is WorkflowStatus.BLOCKED
    assert [failure.check_name for failure in validation.missing_evidence] == [
        "rollback_plan_exists"
    ]
    assert validation.missing_evidence[0].source_step == "write_monitoring_and_rollback_report"


def test_declared_rollback_plan_satisfies_litserve_rollback_readiness():
    registry = get_workflow_registry()
    template = registry.get("deploy_litserve_gpu")
    verification_results = tuple(
        VerificationResult(
            check_name=check.name,
            evidence_type="observed" if check.evidence_type == "observed" else "declared",
            source_step=check.source_step,
            passed=True,
            evidence=f"{check.name} evidence",
        )
        for check in template.success_contract.checks
        if check.name not in {"litserve_files_generated", "rollback_plan_exists"}
    )
    manifest = ArtifactManifest(
        entries=(
            ArtifactManifestEntry(
                uri="models:/classifier/Production",
                artifact_type="model_artifact",
                producing_step="select_best_model_artifact",
                state="selected",
            ),
            ArtifactManifestEntry(
                path="serving/litserve_api.py",
                artifact_type="serving_application",
                producing_step="generate_litserve_api",
                state="generated",
            ),
        )
    )

    validation = registry.validate_success_contract(
        "deploy_litserve_gpu",
        verification_results=verification_results,
        artifact_manifest=manifest,
        rollback_plan=RollbackPlan(documented_target="models:/classifier/Previous"),
    )

    assert validation.status is WorkflowStatus.SUCCEEDED
    assert validation.missing_evidence == ()
    assert validation.failed_checks == ()


def test_deployment_templates_require_rollback_readiness_contract():
    registry = get_workflow_registry()

    expected_sources = {
        "deploy_litserve_gpu": ("rollback_plan_exists", "write_monitoring_and_rollback_report"),
        "deploy_gpu_inference": ("rollback_plan_exists", "generate_rollback_plan"),
        "deploy_gradio_demo": ("rollback_plan_exists", "document_rollback_target"),
        "deploy_kserve_production": ("canary_rollback_plan_exists", "prepare_rollback"),
    }

    for workflow_id, (check_name, source_step) in expected_sources.items():
        template = registry.get(workflow_id)
        rollback_check = next(
            check for check in template.success_contract.checks if check.name == check_name
        )

        assert rollback_check.evidence_type == "declared_or_observed"
        assert rollback_check.source_step == source_step


def test_deployment_report_is_structured_evidence_with_rollback_plan():
    registry = get_workflow_registry()
    contract_status = registry.validate_success_contract(
        "deploy_gpu_inference",
        verification_results=tuple(
            VerificationResult(
                check_name=check.name,
                evidence_type="observed" if check.evidence_type == "observed" else "declared",
                source_step=check.source_step,
                passed=True,
                evidence=f"{check.name} evidence",
            )
            for check in registry.get("deploy_gpu_inference").success_contract.checks
        ),
    )

    report = DeploymentReport(
        workflow_id="deploy_gpu_inference",
        target="lambda-gpu-vm",
        selected_backend="vllm",
        endpoint_url="http://127.0.0.1:8000",
        server_start_command="python -m vllm.entrypoints.openai.api_server",
        health_result=DeploymentCheckResult(passed=True, evidence={"status_code": 200}),
        prediction_result=DeploymentCheckResult(
            passed=True,
            evidence={"sample_output_shape": [1, 3]},
        ),
        latency_summary=LatencySummary(p50_ms=12.5, p95_ms=31.0, sample_count=20),
        gpu_evidence=GpuEvidence(
            available=True,
            device_name="NVIDIA A10",
            cuda_version="12.1",
            utilization_percent=42.0,
        ),
        artifacts=ArtifactManifest(entries=()),
        approvals=(),
        rollback_plan=RollbackPlan(command="systemctl restart previous-vllm.service"),
        contract_status=contract_status,
    )

    assert report.rollback_plan.command == "systemctl restart previous-vllm.service"
    assert report.health_result.evidence["status_code"] == 200
    assert report.contract_status.status is WorkflowStatus.SUCCEEDED


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


def test_litserve_gpu_template_declares_executable_runtime_tools():
    template = get_workflow_registry().get("deploy_litserve_gpu")

    tools_by_step = {step.step_id: step.tool_functions for step in template.steps}

    assert tools_by_step == {
        "detect_runtime_environment": ("detect_runtime_environment",),
        "detect_gpu_cuda": ("detect_gpu_cuda",),
        "select_best_model_artifact": ("select_best_model_artifact",),
        "generate_litserve_api": ("create_litserve_api",),
        "configure_litserve_gpu_runtime": ("configure_litserver",),
        "create_dockerfile": ("create_ml_dockerfile",),
        "build_image_if_available": ("record_litserve_image_build_skipped",),
        "start_litserve_server": ("start_litserve_server",),
        "test_health_endpoint": ("test_litserve_health_endpoint",),
        "test_prediction_endpoint": ("test_litserve_prediction_endpoint",),
        "capture_logs_and_endpoint": ("capture_litserve_logs_and_endpoint",),
        "write_monitoring_and_rollback_report": ("record_litserve_gpu_rollback_readiness",),
    }

    approval_gates = {
        gate.step_id: gate.risk_categories for gate in template.approval_gates
    }
    assert approval_gates["detect_gpu_cuda"] == ("uses_gpu",)
    assert approval_gates["generate_litserve_api"] == ("writes_project_files",)
    assert approval_gates["create_dockerfile"] == ("writes_project_files",)
    assert approval_gates["build_image_if_available"] == ("builds_image",)
    assert approval_gates["start_litserve_server"] == ("starts_server", "exposes_port")


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
