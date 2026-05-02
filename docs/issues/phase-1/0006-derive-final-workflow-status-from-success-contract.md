# Issue 0006: Derive Final Workflow Status From SuccessContract

## Title

Make `setup_pipeline` final status runtime-owned and contract-derived.

## Problem

The current agent can mark success from perception or summarization. Phase 1 needs the **Workflow Runtime** to be the only authority for final `setup_pipeline` status, using **SuccessContract** validation over captured evidence.

## Scope

- Validate the selected template's **Success Contract** after setup execution or blocking.
- Set final workflow status from `ContractValidation`.
- Return `succeeded` only when all required checks pass.
- Return `blocked` for missing input, missing approval, or missing prerequisite/evidence.
- Return `failed` when a verification check ran and failed.
- Ensure final summary reports contract status, failed checks, missing evidence, artifact manifest, and next actions.

## Out Of Scope

- Rewriting all summarization paths.
- Contract-derived status for deployment or training workflows.
- Persistent `WorkflowRun` database state.
- UI timeline or status rendering.

## Files Likely Touched

- `agent/agent_loop.py`
- `workflow/registry.py`
- `summarization/summarizer.py`
- `tests/unit/test_workflow_contracts.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- Missing setup verification evidence yields `blocked`, not `succeeded`.
- Failed Hydra or DVC validation yields `failed`.
- Passing required checks yields `succeeded`.
- Summarizer cannot override contract status.

## Acceptance Criteria

- **SuccessContract** validation decides final `setup_pipeline` status.
- Final output reflects structured contract validation.
- `original_goal_achieved` or prompt summary cannot mark setup success without passing checks.
- Missing evidence and failed checks include deterministic next actions.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_contracts.py tests/unit/test_phase1_workflow_runtime.py -q
```
