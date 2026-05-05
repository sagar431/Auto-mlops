# Issue 0007: Data Stage Evidence And Orchestrator Handoff

## Title

Write `.auto_mlops/capstone/data_stage_evidence.json` and expose data-stage state to `build_capstone_pipeline`.

## Goal

Finish Phase 4 by writing a durable **Data Stage Evidence Artifact** and making the Capstone Orchestrator reference it through the **Artifact Manifest** instead of relying on latest-run lookup or summary text.

## Expected Behavior

- Write `.auto_mlops/capstone/data_stage_evidence.json`.
- Include schema metadata:
  - `schema_version`: `phase4.data_stage_evidence.v1`
  - `created_at`
  - `workflow_id`: `prepare_capstone_data`
  - `status`
- Include dataset-level records with:
  - `dataset_id`
  - `status`
  - `source_path`
  - `layout`
  - `missing_inputs`
  - `next_actions`
  - split strategy, seed, counts, paths, and artifacts
- Include DVC state:
  - tracked paths
  - remote type
  - redacted remote URL
  - validation status
  - pushed/pulled flags
- Include `blocked_capabilities`, **Verification Results**, and **Artifact Manifest** entries.
- Include the data-stage evidence file itself in the **Artifact Manifest**.
- Include split manifests and DVC files in the **Artifact Manifest** when generated.
- Update `build_capstone_pipeline` so the data stage references this evidence artifact and records completed, blocked, or deferred data-stage state.
- Avoid broad `train_and_track` behavior changes unless a minimal contract-facing handoff check is required.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `data_stage_evidence_artifact_reported`: **Observed Evidence**
- `dataset_lineage_artifacts_reported`: **Declared Evidence** or **Observed Evidence**

For capstone-complete success, the evidence file must contain exactly two dataset entries and successful local-ready plus S3 evidence. For partial runs, it may contain one or zero dataset entries only when the workflow status is `blocked`.

`build_capstone_pipeline` should:

- Reference the data-stage evidence artifact through the **Artifact Manifest**.
- Treat local-ready evidence as a completed local data stage.
- Treat missing S3 transfer evidence as blocked for capstone completion.
- Continue to record later capabilities as deferred instead of claiming full capstone success.

## Tests To Add

- `prepare_capstone_data` writes `.auto_mlops/capstone/data_stage_evidence.json` with schema version and created timestamp.
- The evidence artifact contains dataset-level status, missing inputs, next actions, split evidence, DVC state, and blocked capabilities.
- The **Artifact Manifest** includes the data-stage evidence file itself.
- The **Artifact Manifest** includes generated split manifests and DVC files when present.
- Evidence output redacts S3 URLs and contains no secret material.
- `build_capstone_pipeline` references data-stage evidence when present.
- `build_capstone_pipeline` records missing S3 transfer as blocked rather than succeeded for capstone-complete state.

## Out Of Scope

- Final capstone report generation.
- Video generation.
- Frontend, report, or demo deployment capabilities.
- Training, HPO, learning-rate finder, or model selection changes beyond minimal handoff evidence.
- Docker, CI/CD, Kubernetes, KServe, Helm, ArgoCD, AWS Lambda, or HuggingFace Spaces.
- Latest-run lookup as the only orchestrator handoff.

## Verification Commands

- `pytest tests/unit/test_workflow_registry.py -k capstone`
- `pytest tests/unit/test_agent_loop.py -k capstone`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py`
- `ruff check workflow agent mcp_mlops_tools.py tests`

## Dependency / Blocker Notes

Blocked by [0001](./0001-register-prepare-capstone-data.md), [0002](./0002-detect-two-image-folder-datasets.md), [0003](./0003-generate-split-manifests.md), [0004](./0004-dvc-track-capstone-data-package.md), [0005](./0005-configure-and-validate-dvc-remote.md), and [0006](./0006-approval-gated-dvc-push-pull.md).
