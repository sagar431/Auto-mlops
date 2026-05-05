# Issue 0004: Create And DVC-Track Capstone Data Package

## Title

Create and DVC-track `data/capstone/<dataset_id>/` packages.

## Goal

Make `prepare_capstone_data` produce a DVC-tracked **Capstone Data Package** for each dataset without blindly tracking raw source datasets.

## Expected Behavior

- Validate or initialize DVC metadata for the project through registry-owned steps.
- Require an **Approval Gate** before DVC repo initialization or update.
- Create `data/capstone/<dataset_id>/` package directories when approved.
- DVC-track generated split manifests.
- DVC-track materialized train/test folders only when they were generated for downstream training.
- Record raw source dataset paths as `external` **Artifact Manifest** entries by default.
- Record DVC files and tracked package paths as generated or validated **Artifact Manifest** entries.
- Preserve blocked behavior when DVC is unavailable or project file writes are not approved.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `capstone_data_package_tracked`: **Observed Evidence**
- `dvc_repo_validated`: **Observed Evidence**
- `dataset_lineage_artifacts_reported`: **Declared Evidence** or **Observed Evidence**

Evidence should include:

- DVC repo path and validation status.
- Tracked package paths.
- Generated `.dvc` files or `dvc.yaml` references when applicable.
- Checksums when available.
- Source dataset `external` artifact entries.
- Generated split manifest artifact entries.
- **Approval Records** for project file writes and DVC state changes.

## Tests To Add

- Generated split manifests are added to DVC tracking under `data/capstone/<dataset_id>/`.
- Raw source dataset directories are not DVC-tracked by default.
- Materialized train/test folders are DVC-tracked only when generated.
- Missing DVC executable or invalid DVC repo blocks with next actions.
- DVC initialization or update requires approval before execution.
- The **Artifact Manifest** includes source datasets, split manifests, and DVC tracking artifacts with correct states.

## Out Of Scope

- DVC remote configuration.
- DVC push or pull.
- S3 credentials or bucket validation.
- Dataset downloads.
- Moving source dataset files.
- Training changes.

## Verification Commands

- `pytest tests/unit/test_agent_loop.py -k dvc`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py -k dvc_track`
- `python -m tests.root_migrated.test_mlops_tools --tool dvc`
- `ruff check mcp_mlops_tools.py agent workflow tests`

## Dependency / Blocker Notes

Blocked by [0003](./0003-generate-split-manifests.md).
