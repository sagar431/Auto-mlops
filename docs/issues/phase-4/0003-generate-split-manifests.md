# Issue 0003: Generate Split Manifests

## Title

Generate deterministic split manifests for detected capstone datasets.

## Goal

Add the split layer for `prepare_capstone_data`: produce deterministic **Split Manifests** for each supported dataset, preserve source dataset layout, and require **Approval Gates** before project file writes or optional materialized train/test folders.

## Expected Behavior

- For each supported dataset, detect whether train/test split evidence already exists.
- When no split exists, propose a deterministic split plan with:
  - fixed seed
  - split ratio
  - per-class counts
  - output manifest path
- Block at an **Approval Gate** before writing a split manifest.
- Generate `data/capstone/<dataset_id>/split_manifest.json` after approval.
- Never move original dataset files.
- Materialize copied train/test folders only when a workflow input or downstream training requirement asks for physical split folders.
- Block at a separate **Approval Gate** before copying files into materialized split folders.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `split_evidence_recorded`: **Observed Evidence**
- `dataset_lineage_artifacts_reported`: **Declared Evidence** or **Observed Evidence** for generated split manifests

Evidence should include:

- Dataset id.
- Split strategy: `existing`, `manifest`, or `copied_folders`.
- Seed.
- Split ratio.
- Train count.
- Test count.
- Per-class counts.
- Split manifest path.
- Materialized train/test paths when generated.
- **Approval Records** for split manifest writes and materialized folder writes.

## Tests To Add

- A dataset without train/test folders blocks for approval before writing a split manifest.
- Approved split manifest generation creates deterministic output with stable ordering and fixed seed.
- Re-running split generation with the same seed produces the same manifest.
- Existing train/test layouts record split evidence without copying or moving files.
- Materialized folders are not created unless explicitly requested and approved.
- Source dataset files are never moved or deleted.

## Out Of Scope

- Data augmentation.
- Data cleaning, deduplication, or quality scoring beyond layout and split counts.
- Cross-validation folds.
- Training loader rewrites.
- DVC tracking.
- S3 remote configuration or transfer.

## Verification Commands

- `pytest tests/unit/test_agent_loop.py -k split_manifest`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py -k split`
- `ruff check mcp_mlops_tools.py agent workflow tests`

## Dependency / Blocker Notes

Blocked by [0002](./0002-detect-two-image-folder-datasets.md).
