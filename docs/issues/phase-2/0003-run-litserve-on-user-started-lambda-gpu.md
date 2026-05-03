# Issue 0003: Run LitServe On User-Started Lambda GPU

## Title

Execute the narrow `deploy_litserve_gpu` workflow on an already-running Lambda Cloud GPU instance.

## Problem

The first Phase 2 capstone must prove a real LitServe deployment on Lambda Cloud GPU without wasting GPU credits during planning or local file generation. The first GPU execution slice should assume the user has already started the Lambda GPU instance, then run the agent inside that instance to collect observed deployment evidence.

Cloud API provisioning and lifecycle automation are deferred so the first capstone can focus on verified deployment behavior.

## Scope

- Execute `deploy_litserve_gpu` on a user-started Lambda Cloud GPU machine.
- Require explicit approval before server start, port exposure, GPU actions, cloud credential use, or optional Docker build.
- Detect GPU with `nvidia-smi` or PyTorch CUDA.
- Select the model artifact produced or validated by preflight.
- Start LitServe directly in the project environment.
- Test `/health` and record the observed result.
- Test `/predict` with sample input and record the observed result.
- Record endpoint URL, server start command, logs or log path, artifacts, approvals, and rollback/stop readiness.
- Record cleanup or stop-instance instructions so GPU credits are not wasted.
- Keep Docker image build optional and approval-gated.

## Out Of Scope

- Starting, stopping, or provisioning Lambda Cloud instances via API.
- Autonomous cloud/GPU actions without explicit approval.
- Docker build by default.
- Pushing images to ECR, Docker Hub, or another registry.
- Kubernetes, KServe, EKS, canary rollout.
- TorchServe, MAR archives, model workers.
- General `deploy_gpu_inference` backend selection.
- Gradio demo deployment.
- AWS Lambda serverless.
- Training improvement loops, HPO, or `train_until_better`.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/`

## Tests To Write First

- Without approval, `deploy_litserve_gpu` blocks before server start or port exposure.
- Without observed GPU evidence, `deploy_litserve_gpu` ends `blocked` or `failed` with a clear next action.
- With mocked observed GPU, server, health, and prediction evidence, the workflow reaches contract-derived `succeeded`.
- Declared evidence cannot satisfy GPU detection, server start, `/health`, `/predict`, or endpoint checks.
- Final output includes workflow status, contract status, missing evidence or failed checks when relevant, artifacts, approvals, endpoint URL, and rollback/stop readiness.

## Acceptance Criteria

- GPU detection is captured as **Observed Evidence** from `nvidia-smi` or PyTorch CUDA.
- LitServe app evidence is captured.
- Server start is captured as **Observed Evidence**.
- `/health` returns OK and is captured as **Observed Evidence**.
- `/predict` works with sample input and is captured as **Observed Evidence**.
- Endpoint URL is recorded.
- Rollback or stop-instance readiness is recorded.
- Cleanup/stop instructions are included to protect the user's Lambda Cloud GPU budget.
- Full deployment success is impossible without observed live evidence.

## Manual Smoke Preconditions

- User has started a Lambda Cloud GPU instance.
- Agent runs inside that instance.
- Project dependencies are installed.
- The project has a selected model artifact or a fixture artifact from preflight.
- User explicitly approves server start, port exposure, GPU actions, and any optional Docker build.

## Recommended Test Commands

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py -q
.venv/bin/python -m pytest tests/e2e/ -q
.venv/bin/python -m ruff check agent workflow action tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/e2e/
```
