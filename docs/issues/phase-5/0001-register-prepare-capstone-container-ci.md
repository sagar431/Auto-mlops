# Issue 0001: Register `prepare_capstone_container_ci`

## Title

Register `prepare_capstone_container_ci` as the Phase 5 **Capstone Container And CI Automation Workflow**.

## Goal

Add a first-class **Workflow Template** for Phase 5 without implementing Docker, registry, or CI behavior yet. The registry must declare stable **Workflow Inputs**, routing aliases, **Container CI Completion Mode** values, **Success Contract** branches, blocked behavior, and approval boundaries so later Phase 5 issues can add evidence-producing steps without changing the workflow identity.

## Dependencies

None. This is the first Phase 5 implementation issue.

## Expected Behavior

- Register `prepare_capstone_container_ci` in the **Workflow Registry**.
- Declare the generic **Workflow Input** `completion_mode` with Phase 5-specific values:
  - `container_local_ready`
  - `container_capstone_complete`
- Declare workflow inputs for project path and optional local model artifact, MLflow run or best artifact selectors, registry target, image name, image tag, and CI workflow path.
- Add routing aliases for natural-language requests such as:
  - "prepare capstone container CI"
  - "create capstone Docker and CI evidence"
  - "package capstone runtime image"
  - "prepare container_ci_evidence"
- Add negative routing rules so Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment, frontend, final report, and stress-test requests do not select this workflow.
- Add the ordered **Workflow Template** skeleton for the seven Phase 5 milestones, with later steps allowed to remain blocked or deferred until implemented.
- Declare **Success Contract** branches for `container_local_ready` and `container_capstone_complete`.
- Missing required inputs must produce structured `blocked` **Workflow Status** with **Contract Failure** records, not fallback to a generic workflow.
- `build_capstone_pipeline` must not be changed in this issue except for any registry references needed to recognise the future stage name.

## Success Contract / Evidence Expectations

Common contract checks should be declared but may initially be unsatisfied until later issues implement their source steps:

- `upstream_evidence_resolved`
- `container_build_spec_reported`
- `dependency_context_reported`
- `container_ci_evidence_artifact_reported`
- `container_artifact_manifest_reported`
- `capstone_ci_workflow_reported`
- `capstone_ci_workflow_validated`
- `secret_safety_validated`

`container_local_ready` branch checks:

- `local_model_artifact_resolved` or `mlflow_best_artifact_resolved`
- `docker_availability_reported`
- Conditional `image_build_attempt_reported` when Docker is available
- Conditional `image_build_deferred_reported` when Docker is unavailable

`container_capstone_complete` branch checks:

- `data_stage_capstone_complete_verified`
- `mlflow_best_artifact_verified`
- `training_lineage_verified`
- `docker_available`
- `image_build_succeeded`
- `container_smoke_check_passed`
- `registry_target_validated`
- `registry_auth_capability_verified`
- `registry_push_approved`
- `registry_push_succeeded`
- `pushed_image_reference_reported`
- `capstone_ci_registry_usage_validated`

Avoid vague checks such as `docker_ready`.

## Approval Gates

- Read-only registry selection and input validation do not require an **Approval Gate**.
- Declare future **Approval Gates** for steps that will write project files, build images, execute container smoke commands, use remote service credentials, or push registry images.
- This issue should not execute gated steps.

## Tests To Add

- Registry tests assert `prepare_capstone_container_ci` exists and is not a **Fake Template**.
- Registry tests assert `completion_mode` accepts only `container_local_ready` and `container_capstone_complete`.
- Prompt/routing tests assert container/CI prompts select `prepare_capstone_container_ci`.
- Negative routing tests assert Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS, endpoint deployment, stress-test, frontend, final report, and video prompts do not select this workflow.
- Contract tests assert both completion-mode branches declare the expected checks and conditional checks.
- Runtime tests assert missing required inputs block with structured **Contract Failure** records.

## Out Of Scope

- Dockerfile generation, image build, image smoke checks, registry validation, registry login or push, CI workflow generation, and `container_ci_evidence.json` writing.
- Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment checks, stress tests, frontend timeline changes, final report or video generation, secret mutation, and unapproved registry push.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -k capstone -q`
- `.venv/bin/python -m pytest tests/unit/test_agent_loop.py -k capstone -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase3_capstone_orchestrator.py -q`
- `.venv/bin/python -m ruff check workflow agent tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/e2e/test_phase3_capstone_orchestrator.py`
