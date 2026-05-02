# Issue 0004: Enforce setup_pipeline Approval Gates Before execute_step

## Title

Require registry-owned **Approval Records** before risky setup steps call `execute_step`.

## Problem

Phase 0 modeled **Approval Gates**, but the current agent approval flow is coarse and deployment-oriented. Phase 1 needs step-level approval enforcement for `setup_pipeline` write operations before the runtime calls `execute_step`.

## Scope

- Add or verify `setup_pipeline` **Approval Gates** for write operations.
- Validate approval immediately before each gated step executes.
- Block a gated step when no matching **Approval Record** exists.
- Fail or stop execution when approval is denied.
- Preserve read-only analysis and pure validation as ungated where no write occurs.
- Emit approval-required events with step id and risk categories.

## Out Of Scope

- New frontend approval UI.
- Persistent database-backed approval records.
- Broad permission policy or user-management redesign.
- Approval enforcement for deployment templates beyond preserving Phase 0 metadata.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `agent/approval.py`
- `tests/unit/test_workflow_approval.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- A gated `setup_pipeline` write step without an approval record blocks and does not call `execute_step`.
- A matching approved **Approval Record** allows the step to run.
- A denied approval prevents the step from running and returns a structured reason.
- Read-only analysis runs without approval.

## Acceptance Criteria

- Approval gates are declared at the workflow-step level.
- `execute_step` is never called for a blocked gated step.
- Blocking output includes `step_id`, risk categories, and next action.
- Existing coarse approval behavior does not override registry-owned gates for `setup_pipeline`.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_approval.py tests/unit/test_phase1_workflow_runtime.py -q
```
