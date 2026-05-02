# Issue 0001: Add Workflow Registry Core And Setup Pipeline Template

## Title

Add Workflow Registry core models and the first testable `setup_pipeline` template.

## Problem

Phase 0 requires deterministic workflow skeletons owned by code, but the current system relies on prompt-generated `plan_graph` skeletons and a flat tool list. There is no code-owned **Workflow Registry** or **Workflow Template** contract to prevent invented steps.

## Scope

- Add core registry/domain types for **Workflow Registry**, **Workflow Template**, **Workflow Step**, **Workflow Input**, **Success Contract**, and **Workflow Status**.
- Add a registry lookup API for `workflow_id`.
- Add the first real template, `setup_pipeline`, with ordered steps, required inputs, success contract checks, and artifact requirements as data.
- Add validation that registry entries are not **Fake Templates**.

## Out Of Scope

- Replacing the full agent loop.
- Adding deployment workflow templates.
- Executing tools from the registry.
- Real Hydra, DVC, MLflow, Docker, or GitHub Actions validation.

## Files Likely Touched

- `agent/` or new `workflow/` package
- `decision/decision.py`
- `tests/unit/`
- `CONTEXT.md` only if terminology gaps are found

## Tests To Write First

- Registry returns `setup_pipeline` by id.
- `setup_pipeline` steps are ordered and deterministic.
- Required inputs are declared separately from step args.
- A template missing steps or success contract checks is rejected as a fake template.

## Acceptance Criteria

- `setup_pipeline` exists in a code-owned registry.
- The registry exposes deterministic template data without calling an LLM.
- Tests prove the template cannot be empty or contract-free.
- No future placeholder templates are added.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -q
```

