# Issue 0001: Register `prepare_capstone_data`

## Title

Register `prepare_capstone_data` as the Phase 4 Capstone Data Automation Workflow.

## Goal

Add a first-class **Workflow Template** for reproducible capstone data lineage. This slice should make `prepare_capstone_data` selectable, contract-owned, and honest about missing dataset inputs before any data inspection, DVC mutation, or transfer behavior exists.

## Expected Behavior

- Add `prepare_capstone_data` to the **Workflow Registry**.
- Declare **Workflow Inputs**:
  - `project_path`
  - `dataset_1_path`
  - `dataset_2_path`
  - `completion_mode`, with allowed values `local_ready` and `capstone_complete`
- Add routing aliases such as "prepare capstone data", "set up capstone data", and "version capstone datasets".
- Block before step execution when either dataset path is missing.
- Distinguish **Local Data Readiness** from **Capstone Data Completeness** in the contract model.
- Ensure the workflow cannot be marked `succeeded` from summary text or prompt-authored planning.

## Success Contract / Evidence Expectations

The initial **Success Contract** should define contract branches, even if later checks are not implemented yet.

Local-ready checks:

- `two_dataset_paths_provided`: **Observed Evidence**
- `two_dataset_layouts_supported`: **Observed Evidence**
- `split_evidence_recorded`: **Observed Evidence**
- `capstone_data_package_tracked`: **Observed Evidence**
- `dvc_repo_validated`: **Observed Evidence**
- `data_stage_evidence_artifact_reported`: **Observed Evidence**
- `dataset_lineage_artifacts_reported`: **Declared Evidence** or **Observed Evidence**

Capstone-complete checks:

- All local-ready checks.
- `s3_remote_validated`: **Observed Evidence**
- `s3_transfer_completed`: **Observed Evidence**

This issue only needs to register the contract shape and block honestly when inputs are missing. Later issues fill in the evidence.

## Tests To Add

- `get_workflow_registry().get("prepare_capstone_data")` returns a real **Workflow Template**.
- The template declares `project_path`, `dataset_1_path`, `dataset_2_path`, and `completion_mode` as **Workflow Inputs**.
- The template rejects unknown `completion_mode` values.
- Missing dataset inputs produce a blocked **Workflow Selection** with missing input names.
- Natural-language prompts for capstone data preparation select `prepare_capstone_data`, not `setup_pipeline`.
- The registry does not expose a **Fake Template** with no contract checks.

## Out Of Scope

- Dataset layout detection.
- Split manifest generation.
- DVC tracking.
- DVC remote configuration.
- DVC push or pull.
- Writing `.auto_mlops/capstone/data_stage_evidence.json`.
- Editing `PRD.md`.

## Verification Commands

- `pytest tests/unit/test_workflow_registry.py`
- `pytest tests/unit/test_agent_loop.py -k prepare_capstone_data`
- `ruff check workflow agent tests`

## Dependency / Blocker Notes

None. This issue must land before all other Phase 4 issues.
