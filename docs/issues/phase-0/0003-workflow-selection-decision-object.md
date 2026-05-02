# Issue 0003: Add Structured Workflow Selection Decision Object

## Title

Add structured Workflow Selection for natural-language routing.

## Problem

Routing currently depends on prompt output such as `route`, `pipeline_stage`, and `deployment_target`. Phase 0 needs a debuggable **Workflow Selection** object that records why a workflow was selected and which workflows were rejected.

## Scope

- Add a **Workflow Selection** type with `workflow_id`, `confidence`, `matched_aliases`, `rejected_workflows`, `missing_inputs`, and `selection_reason`.
- Add deterministic selection logic over registry **Routing Aliases** and **Negative Routing Rules**.
- Block low-confidence or conflicting selection instead of falling back to generic setup.
- Route “Deploy this model on Lambda Labs GPU” directly to `deploy_litserve_gpu`.

## Out Of Scope

- Full replacement of perception prompts.
- LLM prompt tuning beyond the minimal integration needed for tests.
- Runtime execution of selected templates.
- Real deployment checks.

## Files Likely Touched

- `perception/`
- `decision/decision.py`
- `agent/agent_loop.py`
- `workflow/`
- `tests/unit/test_perception.py`
- `tests/unit/test_decision.py`

## Tests To Write First

- “Deploy this model on Lambda Labs GPU” selects `deploy_litserve_gpu`.
- Lambda Labs GPU rejects AWS Lambda/FastAPI Lambda workflow paths.
- “Serve this LLM with vLLM” selects `deploy_gpu_inference` with the appropriate branch evidence.
- Ambiguous input returns blocked selection with missing inputs or a clarifying question.

## Acceptance Criteria

- Route selection returns a structured object, not only a string id.
- Selection records matched aliases and rejected workflows.
- Ambiguous selection does not start a generic workflow.
- Existing baseline route tests continue to pass or are intentionally updated to assert the new object.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_perception.py tests/unit/test_decision.py -q
```

