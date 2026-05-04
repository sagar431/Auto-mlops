# Phase 2 Implementation Issues

## Status

Complete. Issues 0001, 0002, and 0003 have landed on `main`; the first real `deploy_litserve_gpu` path succeeded on a user-started Lambda Cloud GPU instance with observed live deployment evidence.

Phase 2 targets the first real Lambda Cloud GPU LitServe capstone, but starts by hardening the registry runtime so deterministic workflow execution is fast and reliable before GPU time is spent.

Important constraint: the user has a Lambda Cloud GPU plan/budget around $1000. Phase 2 should target real Lambda Cloud GPU deployment with LitServe, but must avoid wasting GPU time. Local generation and validation happen first, cloud/GPU actions require explicit approval, and cleanup or stop-instance instructions are required.

`PRD.md` is intentionally not edited by these issues. Phase 2 Issue 1 is a prerequisite hardening slice before the PRD's LitServe capstone work continues.

## Issues

1. [Complete] [Harden Registry-Executed Workflow Runtime](./0001-harden-registry-executed-workflow-runtime.md)
2. [Complete] [Add Local LitServe Deployment Preflight Workflow](./0002-add-local-litserve-deployment-preflight-workflow.md)
3. [Complete] [Run LitServe On User-Started Lambda GPU](./0003-run-litserve-on-user-started-lambda-gpu.md)

## Dependency Order

- 0001 has no blockers and must land before new deployment workflow execution.
- 0002 depends on 0001.
- 0003 depends on 0001 and 0002.

## Phase 2 Boundaries

Phase 2 includes:

- Registry runtime hardening for deterministic workflow execution.
- A separate `deploy_litserve_preflight` **Workflow Template** for local validation.
- A narrow `deploy_litserve_gpu` path for a user-started Lambda Cloud GPU instance.
- Explicit approval gates before risky server, port, cloud, or GPU actions.
- Observed evidence for live deployment success.
- Rollback, cleanup, or stop-instance readiness.

Phase 2 excludes:

- Kubernetes, KServe, EKS, and canary rollout.
- TorchServe, MAR archives, model version workers.
- General `deploy_gpu_inference` backend selection across LitServe, Gradio, vLLM, KServe, TorchServe, or FastAPI Lambda CPU.
- `deploy_gradio_demo`.
- Lambda Cloud API provisioning, instance start automation, or automatic billing lifecycle control in the first GPU execution issue.
- AWS Lambda serverless.
- Cloud registry push, ECR, or Docker Hub push.
- Training improvement loops, HPO, or `train_until_better`.
- Persistent DB workflow records, multi-user governance, and frontend endpoint cards.
- Autonomous cloud/GPU actions without explicit approval.
- Marking deployment succeeded from declared evidence where observed evidence is required.
