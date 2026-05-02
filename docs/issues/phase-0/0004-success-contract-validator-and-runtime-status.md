# Issue 0004: Add Success Contract Validator And Runtime-Owned Status

## Title

Validate Success Contracts from structured Verification Results.

## Problem

Phase 0 says workflows cannot be marked successful without contract verification fields, but success is still effectively prompt or summary driven. The runtime needs to derive **Workflow Status** from **Verification Results**.

## Scope

- Add **Verification Result** and **Contract Failure** types.
- Support **Observed Evidence** versus **Declared Evidence**.
- Add a contract validator that returns `succeeded`, `blocked`, or `failed` from registry checks.
- Report `missing_evidence` and `failed_checks` with expected evidence type, source step, actual evidence, and next action.
- Prevent declared evidence from satisfying live-runtime checks.

## Out Of Scope

- Real HTTP checks.
- Real GPU utilization capture.
- Real Docker or Kubernetes checks.
- Rewriting summarization output.

## Files Likely Touched

- `workflow/`
- `agent/agent_loop.py`
- `summarization/`
- `tests/unit/test_workflow_contracts.py`

## Tests To Write First

- A workflow with missing required checks cannot be `succeeded`.
- Declared `/health` evidence does not satisfy an observed `/health` check.
- Missing GPU evidence for `deploy_litserve_gpu` yields `blocked` or `failed`, not `succeeded`.
- Contract failure output includes missing evidence and next action.

## Acceptance Criteria

- Runtime-owned status is computed from contract validation.
- `succeeded` is only returned when all required checks pass.
- `blocked` is returned for missing inputs, approvals, or environment prerequisites.
- Contract failure data is structured and testable.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_contracts.py -q
```

