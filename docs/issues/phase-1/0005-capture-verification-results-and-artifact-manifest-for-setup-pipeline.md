# Issue 0005: Capture Verification Results And Artifact Manifest For setup_pipeline

## Title

Record structured **Verification Results** and an **Artifact Manifest** for setup execution.

## Problem

The setup workflow cannot be marked successful from prose summaries or tool success flags alone. Phase 1 needs each relevant setup step to produce structured evidence that can satisfy the `setup_pipeline` **Success Contract**.

## Scope

- Capture **Verification Results** for local setup checks.
- Record an **Artifact Manifest** for generated, validated, or selected setup artifacts.
- Require observed validation where local checks are feasible.
- Use artifact manifest entries for generated-file reporting.
- Model Docker evidence as conditional: observed validation/build evidence when Docker is available, otherwise declared build command/report evidence with a clear prerequisite note.

## Out Of Scope

- Real training execution.
- Cloud DVC remote setup unless explicitly supplied and approval-gated.
- Computing checksums for every artifact unless already practical.
- Deployment reports or rollback execution.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `agent/contextManager.py`
- `summarization/summarizer.py`
- `tests/unit/test_workflow_contracts.py`
- `tests/unit/test_artifact_manifest.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- Hydra validation creates a passing observed **Verification Result** when config validates.
- DVC repository and `dvc.yaml` checks record observed evidence.
- Generated Hydra, DVC, Dockerfile, and CI artifacts appear in the **Artifact Manifest**.
- Docker unavailable does not produce fake observed build evidence.

## Acceptance Criteria

- `setup_pipeline` evidence is structured and testable.
- Generated-file reporting can be derived from the **Artifact Manifest**.
- Declared evidence is not used where the contract requires observed validation.
- Failed validation produces a structured failure with next action.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_contracts.py tests/unit/test_artifact_manifest.py tests/unit/test_phase1_workflow_runtime.py -q
```
