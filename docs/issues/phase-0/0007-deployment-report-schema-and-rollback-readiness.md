# Issue 0007: Add Deployment Report Schema And Rollback Readiness Contract

## Title

Add structured Deployment Report schema with rollback readiness.

## Problem

Deployment success should be backed by structured evidence, not markdown-only summaries. Phase 0 also requires deployment workflows to include rollback readiness even when rollback execution is out of scope.

## Scope

- Add **Deployment Report** schema as structured output.
- Include target, selected backend, endpoint URL, server command, health result, prediction result, latency summary, GPU evidence, artifacts, approvals, rollback plan, and contract status fields where applicable.
- Add **Rollback Plan** as a structured contract field for deployment workflows.
- Render markdown from structured deployment report data only if needed by existing summarization paths.

## Out Of Scope

- Frontend endpoint card.
- Real endpoint execution.
- Real rollback execution.
- Kubernetes apply or rollback commands.

## Files Likely Touched

- `workflow/`
- `summarization/summarizer.py`
- `prompts/summarizer_prompt.txt`
- `tests/unit/test_deployment_report.py`
- `tests/unit/test_workflow_contracts.py`

## Tests To Write First

- Deployment report validates structured fields.
- Markdown rendering, if present, reflects structured data.
- Deployment success fails when rollback plan is missing.
- Declared rollback plan can satisfy rollback readiness without executing rollback.

## Acceptance Criteria

- Deployment report is structured first.
- Deployment contracts can require rollback readiness.
- Markdown summaries cannot invent success outside structured contract status.
- Tests cover missing rollback plan failure.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_deployment_report.py tests/unit/test_workflow_contracts.py -q
```

