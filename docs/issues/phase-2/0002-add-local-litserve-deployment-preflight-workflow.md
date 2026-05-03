# Issue 0002: Add Local LitServe Deployment Preflight Workflow

## Title

Add a separate `deploy_litserve_preflight` **Workflow Template** for local LitServe validation.

## Problem

`deploy_litserve_gpu` requires observed live evidence: GPU detection, server start, `/health`, `/predict`, endpoint URL, and rollback or stop readiness. Local file generation cannot satisfy that full deployment contract.

Phase 2 needs a local-first path that validates deployability before spending Lambda Cloud GPU time. This must not weaken `deploy_litserve_gpu` into accepting preflight-only success.

## Scope

- Add a separate `deploy_litserve_preflight` **Workflow Template**.
- Define a preflight **Success Contract** based on declared or locally validated evidence:
  - LitServe app generated.
  - Dockerfile generated or validated.
  - model artifact selected.
  - launch command recorded.
- Generate or validate files locally on CPU.
- Record generated/validated artifacts in the **Artifact Manifest**.
- Declare approval gates needed before future risky actions such as server start, port exposure, Docker build, cloud credentials, or GPU usage.
- Keep `deploy_litserve_gpu` success limited to observed live deployment evidence.

## Out Of Scope

- Starting a Lambda Cloud GPU instance.
- Using Lambda Cloud API credentials.
- Requiring or detecting a GPU.
- Starting a live LitServe server unless explicitly approved for a local-only check.
- Calling `/health` or `/predict` as required full deployment evidence.
- Docker image build by default.
- Cloud registry push, Kubernetes, KServe, TorchServe, vLLM, or Gradio.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/`

## Tests To Write First

- Natural-language local preflight request selects `deploy_litserve_preflight`, not `deploy_litserve_gpu`.
- `deploy_litserve_preflight` has a different **Success Contract** from `deploy_litserve_gpu`.
- Preflight success requires LitServe app evidence, Dockerfile evidence, selected model artifact evidence, and launch command evidence.
- Preflight does not require GPU detection, server start, `/health`, `/predict`, or endpoint evidence.
- `deploy_litserve_gpu` still cannot succeed from preflight-only declared evidence.
- Approval gates are declared for future risky actions.

## Acceptance Criteria

- `deploy_litserve_preflight` exists as a real registry **Workflow Template**, not a fake placeholder.
- It can succeed with declared/local evidence only for preflight obligations.
- It records generated or validated artifacts.
- It clearly reports missing live deployment evidence when relevant.
- It never marks full `deploy_litserve_gpu` as succeeded without observed live checks.
- No cloud credentials, GPU instance, registry push, Kubernetes context, or deployment server is required for the default test path.

## Recommended Test Commands

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py -q
.venv/bin/python -m pytest tests/e2e/ -q
.venv/bin/python -m ruff check agent workflow action tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/e2e/
```
