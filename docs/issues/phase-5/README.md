# Phase 5 Implementation Issues

## Status

Planned. Phase 5 follows the newer capstone sequence from `CONTEXT.md` and the Phase 4 issue track rather than the older `PRD.md` phase numbering. `PRD.md` remains unedited.

## Goal

Phase 5 adds the **Capstone Container And CI Automation Workflow** for the **North-Star Capstone Workflow**. The workflow should produce Docker/container evidence, registry target evidence, approval-gated registry push evidence, **Capstone CI Evidence**, and a durable `.auto_mlops/capstone/container_ci_evidence.json` handoff artifact for `build_capstone_pipeline`.

The goal is to prove the container/CI handoff. Phase 5 does not deploy, load test, generate reports, provision infrastructure, or manage secrets.

## Issues

The planned Phase 5 milestones are:

1. [Register `prepare_capstone_container_ci`](./0001-register-prepare-capstone-container-ci.md)
2. [Resolve upstream evidence](./0002-resolve-upstream-evidence.md)
3. [Generate or validate default **Capstone Runtime Image** build spec](./0003-generate-validate-runtime-image-spec.md)
4. [Build and smoke-check image when Docker is available](./0004-build-smoke-check-image.md)
5. [Configure and validate registry target](./0005-configure-validate-registry-target.md)
6. [Approval-gated registry login/push](./0006-approval-gated-registry-login-push.md)
7. [Generate/validate **Capstone CI Evidence** and write `.auto_mlops/capstone/container_ci_evidence.json` with orchestrator handoff](./0007-container-ci-evidence-and-orchestrator-handoff.md)

## Dependency Order

- Issue 1 has no blockers and establishes the **Workflow Template**, **Workflow Inputs**, routing aliases, `completion_mode` values, local and capstone-complete contract branches, and blocked behavior.
- Issue 2 depends on issue 1 because upstream evidence must resolve against registered workflow inputs and contract branches.
- Issue 3 depends on issue 2 because the default **Capstone Runtime Image** build spec should reflect resolved data, training, MLflow, and model artifact evidence where available.
- Issue 4 depends on issue 3 because local build and smoke evidence require a generated or validated build spec.
- Issue 5 depends on issue 4 because registry target validation should reference a buildable image tag or planned image reference.
- Issue 6 depends on issue 5 because registry login and push should run only after registry target validation and approval checks are in place.
- Issue 7 depends on issues 1 through 6 because the durable evidence artifact and `build_capstone_pipeline` handoff must reflect completed, blocked, and deferred container/CI evidence from the implemented slices.

## Success Contract Boundaries

Common checks for both `container_local_ready` and `container_capstone_complete`:

- `upstream_evidence_resolved`
- `container_build_spec_reported`
- `dependency_context_reported`
- `container_ci_evidence_artifact_reported`
- `container_artifact_manifest_reported`
- `capstone_ci_workflow_reported`
- `capstone_ci_workflow_validated`
- `secret_safety_validated`

Checks for `container_local_ready`:

- `local_model_artifact_resolved` or `mlflow_best_artifact_resolved`
- `docker_availability_reported`
- `image_build_attempt_reported` when Docker is available
- `image_build_deferred_reported` when Docker is unavailable

Checks for `container_capstone_complete`:

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

Phase 5 should avoid vague checks such as `docker_ready`; the contract should expose the exact evidence layer that passed, blocked, or failed.

`registry_push_approved` and `registry_push_succeeded` are separate checks. `registry_push_approved` requires an **Approval Record** for the push step with the expected risk categories. `registry_push_succeeded` requires observed registry command or API evidence that the image push completed. The evidence artifact should record approval evidence separately from execution evidence.

## Evidence Artifact Shape

`.auto_mlops/capstone/container_ci_evidence.json` is the durable handoff artifact for Phase 5. It should include:

- `schema_version`
- `created_at`
- `workflow_id`
- `status`
- `completion_mode`
- `upstream_evidence`
- `container`
- `registry`
- `ci`
- `next_phase_readiness`
- `blocked_capabilities`
- `deferred_capabilities`
- `verification_results`
- `artifact_manifest`

`next_phase_readiness` should remain narrow. It records image reference or digest readiness, registry push status, CI validation status, missing blockers for Kubernetes or GitOps, and deferred capabilities such as `kserve_deployment`, `helm_packaging`, `argocd_gitops`, and `eks_provisioning`.

`next_phase_readiness` must not generate Kubernetes manifests or claim KServe, Helm, ArgoCD, GitOps, or EKS provisioning are complete. It only records whether Phase 5 produced enough evidence for Phase 6 to start cleanly.

## Deferred Boundaries

Phase 5 excludes:

- Kubernetes, KServe, Helm, ArgoCD, and GitOps.
- EKS cluster provisioning, `kubectl` validation, and InferenceService generation.
- Full LitServe or KServe endpoint startup and health/predict checks.
- Frontend timeline changes.
- Final report or video generation.
- Stress tests and load tests.
- Full training in CI.
- HPO or learning-rate finder automation in CI.
- Broad framework-specific Dockerfile generation.
- Multi-image packaging.
- CUDA-optimized image variants unless explicitly required for the default runtime image.
- Mandatory remote GitHub Actions execution.
- Mandatory `act` simulation.
- Secret creation or mutation, including writing GitHub secrets, registry tokens, AWS keys, or `.env`.
- Automatic registry push without approval.
- DockerHub or ECR deep automation beyond declared target support unless a specific issue adds it.
