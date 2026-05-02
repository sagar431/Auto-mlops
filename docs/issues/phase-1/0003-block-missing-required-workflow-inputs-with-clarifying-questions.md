# Issue 0003: Block Missing Required Workflow Inputs With Clarifying Questions

## Title

Convert missing **Workflow Inputs** into blocked runs and concrete clarifying questions.

## Problem

The `setup_pipeline` template declares `project_path` as a required **Workflow Input**, but the current agent can fall back to generic planning when required inputs are missing. Phase 1 needs missing required inputs to block before perception, decision planning, or tool execution.

## Scope

- Merge explicit runtime inputs such as the `project_path` argument into selected workflow input state.
- Detect missing required inputs from the selected **Workflow Template**.
- Return `WorkflowStatus.BLOCKED` when required inputs are missing.
- Emit or return a deterministic blocking reason and one concrete clarifying question.
- For `setup_pipeline`, ask for the project path when absent.

## Out Of Scope

- Asking for optional inputs up front, such as DVC remote, data path, experiment name, Docker preference, or CI provider.
- Full conversational resume or persisted form state.
- Database migrations for workflow input state.
- Free-form prompt fallback when required inputs are missing.

## Files Likely Touched

- `agent/agent_loop.py`
- `workflow/registry.py`
- `agent/contextManager.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- `setup_pipeline` without `project_path` becomes `blocked`.
- The blocked response asks: "What project path should I set up MLOps for?"
- No perception, decision, or tool execution runs while required inputs are missing.
- Optional setup details do not block the workflow before the relevant step needs them.

## Acceptance Criteria

- Missing required **Workflow Inputs** block before execution.
- Blocking output includes missing input names and a clear next action.
- The final status is runtime-owned, not summarizer-authored.
- A supplied `project_path` unblocks the selected `setup_pipeline` workflow.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_phase1_workflow_runtime.py -q
```
