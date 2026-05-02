# Issue 0005: Add Artifact Manifest Schema And Contract Checks

## Title

Add Artifact Manifest schema for generated, validated, selected, and external artifacts.

## Problem

The PRD requires generated files to be listed in final reports, but a prose list is not enough for contract validation. Phase 0 needs a structured **Artifact Manifest** that can satisfy success contract checks.

## Scope

- Add **Artifact Manifest** and **Artifact State** schema.
- Support artifact states `generated`, `validated`, `selected`, and `external`.
- Tie manifest entries to producing **Workflow Steps**.
- Add contract checks that require specific artifact entries for `setup_pipeline` and deployment templates.

## Out Of Scope

- Computing checksums for every historical artifact.
- Real file generation.
- Frontend artifact rendering.
- Full persistence layer migration unless the implementation already needs one.

## Files Likely Touched

- `workflow/`
- `agent/contextManager.py`
- `summarization/`
- `tests/unit/test_artifact_manifest.py`
- `tests/unit/test_workflow_contracts.py`

## Tests To Write First

- Manifest entry validates required path or URI, artifact type, producing step, and state.
- Invalid artifact state is rejected.
- `setup_pipeline` contract fails when expected generated files are absent from the manifest.
- Deployment contract can require a selected model artifact and generated serving artifact.

## Acceptance Criteria

- Artifact manifests are structured and validated.
- Contract validation can use manifest contents.
- Generated-file reporting can be derived from manifest data.
- No free-form artifact status labels are accepted.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_artifact_manifest.py tests/unit/test_workflow_contracts.py -q
```

