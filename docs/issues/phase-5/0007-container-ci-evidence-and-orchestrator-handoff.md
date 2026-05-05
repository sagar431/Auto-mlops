# Issue 0007: Container CI Evidence And Orchestrator Handoff

## Title

Write `.auto_mlops/capstone/container_ci_evidence.json` and expose container/CI state to `build_capstone_pipeline`.

## Goal

Finish Phase 5 by writing the durable **Container CI Evidence Artifact**, generating or validating **Capstone CI Evidence**, recording the artifact in the **Artifact Manifest**, and updating `build_capstone_pipeline` to reference container/CI stage state without owning Docker, registry, or CI implementation details.

## Dependencies

- Depends on issues 0001 through 0006 because the durable evidence artifact and orchestrator handoff must reflect registered workflow contracts, upstream evidence, build-spec evidence, image build/smoke evidence, registry target evidence, and approval-gated push evidence.

## Expected Behavior

- Generate or validate `.github/workflows/capstone-ci.yml` or equivalent **Capstone CI Evidence**.
- Validate workflow YAML parsing.
- Validate referenced CI commands are bounded and repo-local.
- Include checks for:
  - tests
  - **Data Stage Evidence Artifact** validation
  - training, MLflow, or best artifact validation
  - optional image build
  - secret-safe registry usage when registry push is included
- Do not run full training in CI by default.
- Do not require remote GitHub Actions run evidence for Phase 5. Include remote run evidence when available; otherwise record it as blocked/deferred.
- Do not require `act` simulation.
- Inspect existing `.github/workflows/*` read-only. Generate dedicated `.github/workflows/capstone-ci.yml` by default and never delete or rewrite unrelated workflows.
- Write `.auto_mlops/capstone/container_ci_evidence.json` with:
  - `schema_version`: `phase5.container_ci_evidence.v1`
  - `created_at`
  - `workflow_id`: `prepare_capstone_container_ci`
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
- Include the evidence file itself in the **Artifact Manifest**.
- Include generated Dockerfile/container files, generated GitHub Actions workflow, image reference/digest when available, and upstream data/training evidence references in the **Artifact Manifest**.
- Include narrow `next_phase_readiness` for Phase 6 handoff:
  - image reference/digest readiness
  - registry push status
  - CI validation status
  - missing blockers for Kubernetes/GitOps
  - deferred capabilities: `kserve_deployment`, `helm_packaging`, `argocd_gitops`, `eks_provisioning`
- `next_phase_readiness` must not generate Kubernetes manifests or claim KServe, Helm, ArgoCD, GitOps, or EKS provisioning are complete.
- Update `build_capstone_pipeline` to:
  - reference the durable container/CI evidence artifact through the **Artifact Manifest**
  - report missing container/CI evidence as blocked when upstream data/training evidence exists
  - report missing container/CI evidence as deferred when upstream stages are not ready
  - report `container_local_ready` as container-local readiness only
  - report `container_capstone_complete` as completed only when required structured checks pass
  - never mark Kubernetes/GitOps complete from Phase 5 evidence

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `container_ci_evidence_artifact_reported`: **Observed Evidence**
- `container_artifact_manifest_reported`: **Observed Evidence**
- `capstone_ci_workflow_reported`: **Declared Evidence** or **Observed Evidence**
- `capstone_ci_workflow_validated`: **Observed Evidence**
- `capstone_ci_registry_usage_validated`: **Observed Evidence** for `container_capstone_complete`
- `secret_safety_validated`: **Observed Evidence**

The final artifact must derive status from **Verification Results**, **Artifact Manifest**, and **Success Contract** evidence, not prose.

## Approval Gates

- Reading existing workflows and validating YAML syntax is read-only and does not require approval.
- Writing `.github/workflows/capstone-ci.yml` requires an **Approval Gate** with `writes_project_files`.
- Adapting an existing `capstone-ci.yml` requires an **Approval Gate** with `writes_project_files`.
- Authenticated GitHub API or CLI inspection for remote CI evidence requires an **Approval Gate** with **Remote Service Credential Use**.
- CI workflow content must use GitHub secrets for registry login or push and must not embed plaintext credentials.

## Tests To Add

- Tests that `container_ci_evidence.json` is written with schema version, workflow id, status, completion mode, upstream, container, registry, CI, next-phase readiness, blocked/deferred capabilities, verification results, and artifact manifest.
- Tests that the evidence artifact includes itself in the **Artifact Manifest**.
- Tests that generated Dockerfile/container files, GitHub Actions workflow, image references/digests, and upstream evidence references appear in the **Artifact Manifest** when available.
- Tests that generated CI YAML parses successfully and uses bounded repo-local commands.
- Tests that CI workflow registry usage uses GitHub secrets and never plaintext credentials.
- Tests that missing remote GitHub Actions evidence is blocked/deferred, not fake-succeeded.
- Tests that `build_capstone_pipeline` references container/CI evidence when present.
- Tests that `build_capstone_pipeline` reports missing container/CI evidence as blocked or deferred based on upstream evidence readiness.
- Tests that Kubernetes/GitOps capabilities remain deferred and are never marked complete from Phase 5 evidence.

## Out Of Scope

- Running full training in CI by default, mandatory remote GitHub Actions execution, mandatory `act` simulation, frontend changes, final report or video generation, stress tests, load tests, endpoint deployment checks, Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, Kubernetes manifests, secret creation or mutation, and unapproved registry push.
- Moving Docker, registry, or CI implementation details into `build_capstone_pipeline`.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -k container -q`
- `.venv/bin/python -m pytest tests/unit/test_agent_loop.py -k container -q`
- `.venv/bin/python -m pytest tests/unit/test_execute_step.py -k container -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase5_prepare_capstone_container_ci.py -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase3_capstone_orchestrator.py -q`
- `.venv/bin/python -m ruff check workflow agent action mcp_mlops_tools.py tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/unit/test_execute_step.py tests/e2e/test_phase5_prepare_capstone_container_ci.py tests/e2e/test_phase3_capstone_orchestrator.py`
