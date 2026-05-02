# Issue 0001: Select Workflow Before Perception Or Decision

## Title

Run **Workflow Selection** before prompt-authored planning or tool execution.

## Problem

The current agent loop begins with perception and then asks the decision module to generate a `plan_graph`. Phase 1 needs `WorkflowRegistry.select_workflow(query)` to be the first routing authority so supported workflows are chosen from registry data instead of prompt-authored route fields.

## Scope

- Call `WorkflowRegistry.select_workflow(query)` early in `AgentLoop.run`.
- Store and emit the structured **Workflow Selection** for the run.
- Enrich selection with explicit runtime inputs such as `project_path`.
- Block ambiguous, conflicting, or low-confidence selection before perception, decision planning, or tool execution.
- Preserve perception only as an observation source after workflow selection.

## Out Of Scope

- Full replacement of perception for all workflows.
- Broad prompt-routing quality work beyond Phase 1 scenarios.
- Executing deployment, training, or multi-workflow plans.
- Editing `PRD.md`.

## Files Likely Touched

- `agent/agent_loop.py`
- `workflow/registry.py`
- `decision/decision.py`
- `tests/unit/test_agent_loop.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- `AgentLoop.run` selects a workflow before calling perception or decision planning.
- Ambiguous selection blocks and does not call `execute_step`.
- `setup_pipeline` selection with a supplied `project_path` proceeds to runtime template projection.
- Perception output cannot change the selected workflow id.

## Acceptance Criteria

- Natural-language selection is represented by a structured **Workflow Selection**.
- Prompt-authored `route`, `pipeline_stage`, or `deployment_target` is not the authority for `setup_pipeline`.
- Blocked selection returns a deterministic reason and next action.
- No tool executes before successful workflow selection and required input validation.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_agent_loop.py tests/unit/test_phase1_workflow_runtime.py -q
```
