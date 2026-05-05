# Issue 0002: Resolve Upstream Evidence

## Title

Resolve data-stage, training, MLflow, and model artifact evidence for `prepare_capstone_container_ci`.

## Goal

Add the upstream evidence resolver for the Phase 5 workflow. The resolver must determine whether the container/CI stage can proceed from durable **Data Stage Evidence Artifact**, MLflow or best artifact evidence, and local model artifact fallback evidence, then record structured blocked or deferred evidence without claiming capstone completeness from weak inputs.

## Dependencies

- Depends on issue 0001 because upstream evidence must resolve against the registered **Workflow Template**, **Workflow Inputs**, and **Container CI Completion Mode** branches.

## Expected Behavior

- Read `.auto_mlops/capstone/data_stage_evidence.json` when present and validate its schema, workflow id, status, completion mode, dataset entries, DVC evidence, blocked capabilities, and **Artifact Manifest** references.
- Resolve training, MLflow, and best artifact evidence from Phase 3-compatible sources already present in the project.
- Resolve a local model artifact fallback only for `container_local_ready`.
- For `container_local_ready`, allow Docker/CI asset preparation to continue when upstream capstone evidence is missing, but record the missing data-stage or MLflow/best-artifact evidence as deferred in `upstream_evidence`.
- For `container_capstone_complete`, block when:
  - `data_stage_evidence.json` is missing
  - the data stage is not capstone-complete
  - MLflow-linked best artifact evidence is missing
  - training lineage cannot be tied back to data-stage evidence
- Redact paths or URIs that could expose secrets. Local source dataset paths may be referenced as upstream evidence only as already allowed by the data-stage artifact; do not embed secrets or `.env` contents.
- Produce **Verification Results** and **Contract Failure** records from structured evidence, not prose.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `upstream_evidence_resolved`: **Observed Evidence**
- `local_model_artifact_resolved`: **Observed Evidence** when a local model artifact fallback is used for `container_local_ready`
- `mlflow_best_artifact_resolved`: **Observed Evidence** when MLflow or best artifact evidence is available
- `data_stage_capstone_complete_verified`: **Observed Evidence** for `container_capstone_complete`
- `mlflow_best_artifact_verified`: **Observed Evidence** for `container_capstone_complete`
- `training_lineage_verified`: **Observed Evidence** or structured blocked evidence for `container_capstone_complete`

The resolver must distinguish:

- Completed upstream evidence
- Deferred upstream evidence allowed by `container_local_ready`
- Blocked upstream evidence required by `container_capstone_complete`

## Approval Gates

- Reading durable evidence files, MLflow metadata, local model artifact metadata, and project configuration is read-only and does not require an **Approval Gate**.
- This issue must not run training, mutate MLflow state, move model artifacts, download datasets, or write secrets.

## Tests To Add

- Unit tests for missing `data_stage_evidence.json` in both completion modes.
- Unit tests for local model artifact fallback in `container_local_ready`.
- Unit tests that `container_capstone_complete` blocks without capstone-complete data evidence.
- Unit tests that `container_capstone_complete` blocks without MLflow-linked best artifact evidence.
- Tests that deferred upstream gaps are represented in `upstream_evidence`, `blocked_capabilities`, or `deferred_capabilities` as appropriate.
- Tests that no success is inferred from prose summaries or latest-run lookup alone.
- Tests that secret-like values are redacted from upstream evidence.

## Out Of Scope

- Generating Dockerfiles, building images, validating registry targets, pushing images, generating CI workflows, or writing the final `container_ci_evidence.json`.
- Broad changes to `train_and_track` behavior unless a minimal contract-facing evidence reader is required.
- Full training, HPO, LR finder, dataset downloads, Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment checks, frontend, final report, video generation, stress tests, secret mutation, and unapproved registry push.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -k container -q`
- `.venv/bin/python -m pytest tests/unit/test_agent_loop.py -k container -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase3_capstone_orchestrator.py -q`
- `.venv/bin/python -m ruff check workflow agent mcp_mlops_tools.py tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/e2e/test_phase3_capstone_orchestrator.py`
