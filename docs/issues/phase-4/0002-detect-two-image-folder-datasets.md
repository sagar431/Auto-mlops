# Issue 0002: Detect Two Image-Folder Datasets

## Title

Detect two user-provided canonical image-folder datasets with dataset-level status.

## Goal

Make `prepare_capstone_data` inspect the two supplied dataset paths and produce **Observed Evidence** for supported or unsupported **Canonical Image Folder Datasets** without writing files or mutating DVC state.

## Expected Behavior

- Validate that `dataset_1_path` and `dataset_2_path` exist as local or mounted paths.
- Detect canonical image-folder layouts with class-labelled subdirectories.
- Detect existing train/test folder layouts when present.
- Return dataset-level status for each dataset:
  - `succeeded` when the layout is supported.
  - `blocked` when the path is missing, ambiguous, unsupported, empty, or lacks class-labelled subdirectories.
- Return `missing_inputs` and `next_actions` per dataset.
- Record original source dataset paths as external **Artifact Manifest** candidates, not as DVC-tracked package paths.
- Keep the whole slice as **Read-Only Data Inspection** with no **Approval Gate**.

## Success Contract / Evidence Expectations

This issue should satisfy or fail these checks with structured **Verification Results**:

- `two_dataset_paths_provided`: passed only when both paths exist.
- `two_dataset_layouts_supported`: passed only when both dataset layouts are supported.

Evidence should include:

- Dataset id.
- Source path.
- Layout classification.
- Existing train path when discovered.
- Existing test path when discovered.
- Class names or class count.
- Image count summaries.
- Missing inputs and next actions for blocked datasets.

## Tests To Add

- Two valid image-folder datasets pass detection and produce dataset-level `succeeded` statuses.
- A missing `dataset_2_path` blocks the workflow before any mutation step.
- A dataset without class-labelled subdirectories blocks with a clear unsupported-layout reason.
- An empty class folder blocks or fails the layout check with a next action.
- Existing train/test folders are detected as existing split evidence candidates.
- Detection does not create files, initialize DVC, or request approval.

## Out Of Scope

- Downloading datasets.
- Supporting CSV, parquet, JSON, Hugging Face datasets, Kaggle APIs, or arbitrary custom loaders.
- Creating train/test splits.
- Copying, moving, deleting, or mutating source dataset files.
- DVC tracking or remote configuration.
- Training changes.

## Verification Commands

- `pytest tests/unit/test_workflow_registry.py -k prepare_capstone_data`
- `pytest tests/unit/test_agent_loop.py -k capstone_data`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py -k detect`
- `ruff check workflow agent mcp_mlops_tools.py tests`

## Dependency / Blocker Notes

Blocked by [0001](./0001-register-prepare-capstone-data.md).
