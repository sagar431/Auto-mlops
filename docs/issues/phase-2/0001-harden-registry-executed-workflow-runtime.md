# Issue 0001: Harden Registry-Executed Workflow Runtime

## Title

Skip post-step LLM perception by default for registry-executed **Workflow Templates**.

## Problem

Phase 1 manually verified that local `setup_pipeline` succeeds, but the smoke run took 190.2 seconds because Gemini/perception was called after deterministic registry steps. One call hit a Gemini 503 and fell back to Gemini Flash even though the registered setup workflow had already selected its ordered steps and derives status from **Success Contract** evidence.

Phase 2 deployment workflows will be more expensive and less reliable if deterministic registry execution still depends on LLM availability after workflow selection.

## Scope

- Make registry-executed **Workflow Templates** skip post-step LLM perception by default.
- Continue advancing deterministic workflow steps by registry order.
- Continue deriving registry workflow status from **Success Contract** validation.
- Allow a registry-owned workflow or step to opt into post-step LLM perception only when explicitly declared.
- Preserve existing non-registry agent behavior, including perception, decision, summarizer, improvement, and deployment routing behavior.
- Keep `setup_pipeline` as the proving workflow for the first runtime-hardening slice.

## Out Of Scope

- Implementing LitServe, Gradio, GPU, cloud, Docker build, Kubernetes, or deployment workflows.
- Removing perception from old non-registry agent paths.
- Editing `PRD.md`.
- Adding brittle timing assertions to regular unit tests.

## Files Likely Touched

- `agent/agent_loop.py`
- `workflow/registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/test_phase1_setup_pipeline.py`

## Tests To Write First

- A registry-executed `setup_pipeline` run with approvals completes without awaiting post-step `perception.run`.
- A registry-executed `setup_pipeline` run still captures **Verification Results**, **Artifact Manifest** entries, **Approval Records**, and contract-derived final status.
- Existing non-registry agent tests still cover perception, decision, and summarizer behavior.
- If a future registry step explicitly opts into post-step perception, the runtime can call it only for that declared case.

## Acceptance Criteria

- 0 post-step LLM perception calls during registry step execution.
- No dependency on Gemini/OpenAI availability after workflow selection for deterministic registry execution.
- Local `setup_pipeline` CLI smoke test should finish under 30 seconds on the current dev machine.
- 5 consecutive local smoke runs should end with:
  - `workflow_status: succeeded`
  - `contract_status: succeeded`
  - `missing_evidence: none`
  - `failed_checks: none`
- The under-30-second target is recorded as a manual smoke-test metric or non-default slow/performance test, not a brittle default unit-test assertion.
- Existing non-registry tests continue to exercise perception, decision, and summarizer behavior.

## Recommended Test Commands

```bash
.venv/bin/python -m pytest tests/e2e/test_phase1_setup_pipeline.py -q
.venv/bin/python -m pytest tests/unit/test_agent_loop.py tests/unit/test_workflow_registry.py -q
.venv/bin/python -m ruff check agent workflow action tests/e2e/test_phase1_setup_pipeline.py tests/unit/test_agent_loop.py tests/unit/test_workflow_registry.py
```

## Manual Smoke Command

```bash
.venv/bin/python cli.py "Set up MLOps for this local Python ML project" --project /tmp/phase1_manual_ml_project --approve
```
