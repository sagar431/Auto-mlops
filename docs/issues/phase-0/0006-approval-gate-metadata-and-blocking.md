# Issue 0006: Add Approval Gate Metadata And Blocking Semantics

## Title

Add step-level Approval Gates with auditable Approval Records.

## Problem

The PRD requires approval before risky actions, but approval needs to be registry-owned and auditable. A generic chat confirmation is not enough to unblock a risky **Workflow Step**.

## Scope

- Add **Approval Gate**, **Risk Category**, and **Approval Record** schema.
- Attach approval metadata to risky registry steps.
- Block execution of gated steps until an approval record exists.
- Support risk categories such as `writes_project_files`, `installs_packages`, `starts_server`, `builds_image`, `pushes_registry`, `uses_cloud_credentials`, and `exposes_port`.

## Out Of Scope

- New UI approval flows.
- Real cloud credential checks.
- Executing build, server, push, or deployment steps.
- Authentication or user-management redesign.

## Files Likely Touched

- `agent/approval.py`
- `agent/agent_loop.py`
- `workflow/`
- `tests/unit/test_agent_loop.py`
- `tests/unit/test_workflow_approval.py`

## Tests To Write First

- A gated step without an approval record blocks the workflow.
- Approval records include workflow run id, step id, risk categories, approval status, approver when available, and timestamp.
- Denied approval keeps the workflow blocked or failed with structured reason.
- Registry templates expose expected risk categories on risky steps.

## Acceptance Criteria

- Approval gates are declared at the workflow-step level.
- Approval records are required before gated steps execute.
- Blocking status and next action are deterministic.
- Existing approval tests still pass or are updated to assert approval records.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_approval.py tests/unit/test_agent_loop.py -q
```

