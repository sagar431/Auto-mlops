# Issue 0007: Add Local setup_pipeline End-to-End Test Fixture

## Title

Prove the first usable Phase 1 setup workflow end to end with a local fixture.

## Problem

Phase 1 needs one usable product path, not only unit tests for registry models. The first E2E should prove that a natural-language request can select and execute the local `setup_pipeline` workflow with approval blocking, verification evidence, artifact reporting, and contract-derived status.

## Scope

- Add a small local Python ML project fixture.
- Run the prompt "Set up MLOps for this local Python ML project" with a supplied `project_path`.
- Assert selection of `setup_pipeline`.
- Assert registry-projected executable steps.
- Assert approval blocking for write steps and success after approvals are supplied.
- Assert Hydra, DVC, MLflow, Dockerfile, and CI workflow artifacts or validations are recorded.
- Assert final status comes from **SuccessContract** validation.

## Out Of Scope

- Real training runs.
- GPU, LitServe, Gradio, KServe, vLLM, TorchServe, or endpoint checks.
- Cloud credentials, S3 remotes, registry push, or Kubernetes contexts.
- Frontend workflow timeline.
- Full capstone orchestration.

## Files Likely Touched

- `tests/e2e/`
- `tests/fixtures/`
- `agent/agent_loop.py`
- `workflow/registry.py`
- `mcp_mlops_tools.py`

## Tests To Write First

- The local E2E blocks for approval before the first write step.
- With approvals supplied, the fixture workflow reaches contract-derived `succeeded`.
- Without `project_path`, the same request blocks with a clarifying question.
- A validation failure produces `failed` with structured next action.

## Acceptance Criteria

- A real local setup path is usable from natural language to final report.
- The E2E does not require network, GPU, Docker daemon, cloud credentials, or Kubernetes.
- Final output includes workflow id, status, verification results, artifact manifest, approvals, and next actions.
- The test guards against reintroducing prompt-authored setup skeletons.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/e2e/test_setup_pipeline_phase1.py -q
```
