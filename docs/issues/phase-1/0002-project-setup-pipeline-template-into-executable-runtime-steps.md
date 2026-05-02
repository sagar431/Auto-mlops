# Issue 0002: Project setup_pipeline Template Into Executable Runtime Steps

## Title

Execute `setup_pipeline` from the registered **Workflow Template** instead of an LLM-authored skeleton.

## Problem

Phase 0 created a deterministic `setup_pipeline` template, but the current runtime still executes nodes from the decision module's prompt-authored `plan_graph`. Phase 1 needs a runtime projection that turns registry-owned steps into executable nodes without letting the LLM add, remove, reorder, rename, or skip workflow steps.

## Scope

- Load the selected `setup_pipeline` **Workflow Template** from the registry.
- Project template steps into the existing execution graph or a workflow-runtime equivalent.
- Bind each **Workflow Step** only to its allowed `tool_functions`.
- Allow constrained argument filling for registered steps.
- Prevent prompt output from changing step order, step ids, required steps, approval gates, or success checks.
- Keep the existing generic decision module available for non-Phase-1 paths only if needed.

## Out Of Scope

- Registry execution for deployment templates.
- Parallel or branch execution.
- Multi-workflow orchestration such as a capstone pipeline.
- New tool implementations unrelated to `setup_pipeline`.

## Files Likely Touched

- `agent/agent_loop.py`
- `workflow/registry.py`
- `agent/contextManager.py`
- `decision/decision.py`
- `tests/unit/test_phase1_workflow_runtime.py`

## Tests To Write First

- Runtime creates executable nodes in the same order as `setup_pipeline.steps`.
- Runtime rejects or ignores LLM output that attempts to add, remove, reorder, or rename setup steps.
- Each executable node uses only a tool function declared on its registry step.
- Missing step arguments block rather than inventing an unsafe tool call.

## Acceptance Criteria

- `setup_pipeline` execution shape comes from `WorkflowTemplate.steps`.
- The LLM may only provide constrained observations or arguments.
- The runtime can report the selected template id and projected step ids.
- Prompt-authored `plan_graph` is not used as the skeleton for `setup_pipeline`.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_phase1_workflow_runtime.py -q
```
