# Auto-MLOps Agent Product PRD

## Product Goal

Build a production-like Auto-MLOps agent that can complete verified workflows from a natural-language request, not only generate files.

The first product-quality journey:

> A user gives an ML project path. The agent analyzes the project, configures MLOps, validates or runs training, packages the best model, deploys to a selected target, tests the endpoint, sets monitoring/rollback, and reports exact artifacts, commands, metrics, and URLs.

The product should initially optimize for reliability over breadth. A smaller number of verified paths is more valuable than many untested tools.

## Product Boundary

### Initial Supported Paths

1. **Local or Lambda Cloud GPU deployment**
   - Primary serving target: LitServe
   - Success proof: GPU detected, LitServe app generated, server started, `/health` and `/predict` tested.

2. **Demo deployment**
   - Primary serving target: Gradio
   - Success proof: UI generated, local launch command works, sample prediction path documented.

3. **Kubernetes production deployment**
   - Start with KServe or TorchServe, not both.
   - Recommended first choice: KServe if the user asks for Kubernetes/EKS/canary; TorchServe if the capstone requirement is MAR/model versioning.

### Explicit Non-Goals For Early Phases

- Perfect support for every ML framework.
- Full cloud provisioning across all providers.
- Autonomous production deploys without approval.
- Broad general-purpose AI workspace features unrelated to MLOps.

## Course Graph Alignment

The EMLO course graph identifies the workflows the product should implement first:

- **Reproducible Training Workflow**: Hydra/OmegaConf, DVC, MLflow, HPO or learning-rate finder.
- **Cloud ML Deployment Foundations**: Docker, registry push, AWS/EKS basics.
- **LitServe Serving Lifecycle**: LitAPI, batching, streaming, GPU acceleration, autoscaling.
- **KServe/Knative Canary Stack**: InferenceService, revisions, canary, scale-to-zero, rollback.
- **Capstone Pipeline**: DVC with S3, HPO/LR finder, Docker registry, CI/CD, model serving, monitoring, report.
- **TorchServe Pipeline**: handler, MAR archive, workers, model versioning, metrics.

## External Architecture References

### Hermes Agent

Hermes should be used as an architectural reference, not merged wholesale.

Useful ideas to adapt:

- Skill system for reusable procedures.
- Toolsets and permissions.
- Persistent memory/session search.
- Subagent delegation for parallel workstreams.
- Gateway/scheduled automation ideas.
- Context compression and long-running agent ergonomics.

Do not import the whole Hermes runtime in early phases. Auto-MLOps should remain a focused MLOps product with its own workflow contracts.

### Autoresearch Pattern

Use the autoresearch pattern for measurable ML improvement loops:

1. Establish baseline metric.
2. Propose one bounded change.
3. Run a fixed-budget experiment.
4. Compare metric against baseline.
5. Keep the change only if it improves the metric.
6. Log the result and repeat until budget or target is reached.

For Auto-MLOps this becomes:

```text
train_until_better:
  baseline -> hydra override -> dvc repro/train -> mlflow metric -> compare -> keep/discard
```

The agent must not make unbounded edits to user projects. Each experiment needs a budget, metric, diff, and rollback point.

## Core Workflows

### 1. setup_pipeline

Purpose: create a reproducible MLOps foundation.

Steps:

1. Analyze project structure.
2. Create or validate Hydra config.
3. Initialize DVC.
4. Configure DVC remote when requested.
5. Add data to DVC.
6. Create `dvc.yaml`.
7. Initialize MLflow experiment.
8. Create Dockerfile.
9. Create CI workflow.

Success contract:

- Hydra config validates.
- DVC repo exists.
- `dvc.yaml` exists and is parseable.
- MLflow experiment exists.
- Dockerfile builds or build command/report is produced.
- Generated files are listed in the final report.

### 2. train_and_track

Purpose: run or validate training and select a deployable model.

Steps:

1. Start MLflow run.
2. Run training or call configured training command.
3. Log parameters, metrics, and artifacts.
4. Check threshold metric.
5. Select best run.
6. Register or mark best model artifact.

Success contract:

- MLflow run exists.
- Metric is captured.
- Best model path is known.
- If threshold fails, improvement workflow is proposed.

### 3. train_until_better

Purpose: autoresearch-style controlled improvement.

Steps:

1. Establish baseline.
2. Generate one Hydra override set.
3. Run fixed-budget experiment.
4. Compare metric.
5. Keep or discard change.
6. Record decision, diff, metric, and cost.

Success contract:

- Every attempt has metric, config, duration, and decision.
- No repeated failed change.
- Stop condition is explicit: target met, max attempts, budget exhausted, or data/model issue detected.

### 4. deploy_litserve_gpu

Purpose: first production-quality capstone flow for Lambda Cloud GPU or local GPU VM.

Steps:

1. Detect runtime environment.
2. Detect GPU/CUDA.
3. Select best model artifact.
4. Generate LitServe API.
5. Configure batching/workers/GPU.
6. Create Dockerfile.
7. Build image if Docker is available.
8. Start LitServe server.
9. Test `/health`.
10. Test `/predict` with sample input.
11. Capture logs and endpoint.
12. Write monitoring and rollback report.

Success contract:

- GPU detection result recorded.
- LitServe files generated.
- Server start command recorded.
- `/health` result recorded.
- `/predict` result recorded.
- Endpoint URL recorded.
- Rollback plan exists.

### 4A. deploy_gpu_inference

Purpose: general agentic GPU inference workflow for natural-language deployment requests.

User intent examples:

```text
Deploy this classifier on Lambda GPU.
Serve this LLM with vLLM and optimize latency.
Run this model and tell me if GPU is being used.
Why is my inference slow?
Can you reduce p95 latency under 100ms?
Rollback to previous working model.
```

The user should not need to know Docker commands, CUDA details, ports, batching, or endpoint testing. The agent acts like a deployment engineer: it inspects the environment, selects the serving backend, requests approval for risky actions, executes the workflow, verifies the endpoint, and reports exact evidence.

Steps:

1. Detect runtime environment.
2. Detect GPU.
3. Detect CUDA and driver compatibility.
4. Inspect model artifact and framework.
5. Select serving backend.
6. Generate serving app.
7. Configure GPU runtime, batching, workers, and ports.
8. Start server.
9. Test health endpoint.
10. Test prediction endpoint.
11. Collect latency metrics.
12. Check GPU utilization.
13. Generate report.
14. Generate rollback plan.

Backend selection rules:

- Image/classic PyTorch model -> LitServe.
- Demo/UI request -> Gradio.
- LLM -> vLLM first; consider SGLang or TensorRT-LLM for advanced optimization.
- Kubernetes/EKS request -> KServe.
- Enterprise PyTorch model registry or MAR requirement -> TorchServe.
- AWS Lambda CPU serverless -> FastAPI Lambda only when GPU is not required.

Success contract:

- Deployment target is selected with reasoning.
- GPU/CUDA status is recorded.
- Generated serving files are listed.
- Server start command is recorded.
- Health check passes.
- Prediction test passes with sample input.
- GPU utilization evidence is captured when available.
- Latency metrics include at least p50 and p95 when the endpoint is live.
- Endpoint URL is reported.
- Rollback command or plan exists.

Example interaction:

```text
User:
Deploy /home/ubuntu/catdog-classifier on this Lambda GPU.

Agent:
I found a PyTorch image classifier.
GPU detected: NVIDIA A10, CUDA 12.4.
Recommended serving target: LitServe.
I need approval to create serving files, build an image, start a server on port 8000, and run health/predict tests.

Final:
Deployment complete.
Endpoint: http://146.x.x.x:8000/predict
Model: models/best.pt
GPU: NVIDIA A10, 24GB
Latency: p50 38ms, p95 74ms
Artifacts: serve_litserve.py, Dockerfile, deployment_report.md, rollback.sh
```

### 5. deploy_gradio_demo

Purpose: quick demo for a trained model.

Steps:

1. Select model artifact.
2. Generate Gradio app.
3. Run or validate launch command.
4. Test sample prediction path.
5. Optionally prepare Hugging Face Spaces package.

Success contract:

- App file exists.
- Launch command exists.
- Sample input/output path is documented.

### 6. deploy_kserve_production

Purpose: Kubernetes/EKS production deployment path.

Steps:

1. Detect Kubernetes context.
2. Detect EKS cluster when AWS is used.
3. Create or identify ECR repo.
4. Build and push image.
5. Generate InferenceService.
6. Generate HPA/ingress/config/secrets as needed.
7. Run `kubectl apply --dry-run=server`.
8. Prepare canary rollout.
9. Prepare rollback.
10. Record monitoring endpoints.

Success contract:

- Kubernetes manifests validate.
- Registry image reference exists.
- Canary/rollback plan exists.
- Dry-run result is recorded.

### 7. monitor_and_alert

Purpose: make deployment observable.

Steps:

1. Define service health contract.
2. Capture baseline metrics.
3. Configure drift/performance checks.
4. Generate alert config.
5. Store monitoring report.

Success contract:

- Health/readiness contract exists.
- Baseline metrics exist.
- Alert policy or setup instructions exist.

### 8. rollback

Purpose: restore a known working version.

Steps:

1. Identify previous deployment.
2. Identify previous model artifact/image.
3. Run target-specific rollback command or generate exact command.
4. Verify health/predict after rollback.
5. Store rollback event.

Success contract:

- Rollback target is known.
- Rollback command is available or executed.
- Post-rollback verification is recorded.

## Deterministic Workflow Templates

The LLM should select workflows and fill arguments. It should not invent the skeleton from scratch for common flows.

Example:

```python
WORKFLOWS = {
    "deploy_gpu_inference": [
        "detect_runtime_environment",
        "detect_gpu",
        "detect_cuda",
        "inspect_model_artifact",
        "select_serving_backend",
        "generate_serving_app",
        "configure_gpu_runtime",
        "start_server",
        "test_http_endpoint",
        "test_prediction_endpoint",
        "collect_latency_metrics",
        "check_gpu_utilization",
        "generate_deployment_report",
        "generate_rollback_plan",
    ],
    "litserve_gpu_deploy": [
        "detect_runtime_environment",
        "detect_gpu",
        "analyze_project_config",
        "get_best_mlflow_run",
        "create_litserve_api",
        "configure_litserver",
        "create_ml_dockerfile",
        "build_ml_docker_image",
        "start_litserve_server",
        "test_http_endpoint",
        "test_prediction_endpoint",
        "setup_alerting",
        "generate_rollback_plan",
    ],
    "setup_pipeline": [
        "analyze_project_config",
        "create_hydra_config",
        "validate_hydra_config",
        "init_dvc_repo",
        "configure_dvc_remote",
        "add_data_to_dvc",
        "create_dvc_pipeline",
        "init_mlflow_experiment",
        "create_ml_dockerfile",
        "create_github_workflow",
    ],
}
```

## Required New Tooling

### Environment Detection

- `detect_runtime_environment`
- `detect_gpu`
- `detect_cuda`
- `detect_docker`
- `detect_kubernetes`
- `detect_cloud_provider`
- `check_required_ports`
- `inspect_model_artifact`
- `detect_model_framework`

### Verification And Smoke Tests

- `start_litserve_server`
- `start_inference_server`
- `test_http_endpoint`
- `test_prediction_endpoint`
- `validate_k8s_manifests`
- `kubectl_apply_dry_run`
- `docker_run_healthcheck`
- `check_container_logs`
- `validate_github_workflow`
- `confirm_mlflow_run_exists`
- `collect_latency_metrics`
- `check_gpu_utilization`

### Workflow Runtime

- `select_workflow_template`
- `select_serving_backend`
- `generate_serving_app`
- `configure_gpu_runtime`
- `generate_deployment_report`
- `validate_workflow_contract`
- `persist_workflow_event`
- `persist_artifact_manifest`
- `create_checkpoint`
- `restore_checkpoint`

### Controlled Experiment Loop

- `create_experiment_baseline`
- `propose_hydra_override`
- `run_fixed_budget_experiment`
- `compare_experiment_metric`
- `keep_or_discard_change`

## Product Phases

### Phase 0 — Product Boundary And Contracts

Goal: stop tool sprawl and define what “done” means.

Deliverables:

- Define supported workflows and success contracts.
- Add workflow template registry.
- Add artifact manifest schema.
- Add deployment report schema.
- Add prompt tests for natural-language route/target selection.

Acceptance criteria:

- A workflow cannot be marked successful without contract verification fields.
- Lambda Labs GPU is always routed to LitServe, not AWS Lambda.
- Each workflow has a deterministic skeleton.

### Phase 1 — Verified Setup Pipeline

Goal: make the foundation workflow reliable.

Deliverables:

- Implement `setup_pipeline`.
- Add Hydra validation after config creation.
- Add DVC validation after repo/pipeline creation.
- Add MLflow experiment existence check.
- Add Dockerfile syntax/build validation where Docker exists.
- Add GitHub workflow YAML validation.

Acceptance criteria:

- Natural-language prompt “set up MLOps for this project” creates a verified plan.
- Generated files are listed with pass/fail verification results.
- Failing verification produces a fix suggestion.

### Phase 2 — Lambda Cloud GPU LitServe Capstone

Goal: build the first production-quality demo.

Deliverables:

- Implement runtime/GPU/CUDA/Docker/port detection.
- Implement `deploy_litserve_gpu`.
- Add `start_litserve_server`.
- Add `/health` and `/predict` smoke tests.
- Add monitoring/rollback report.
- Add E2E test using a small image classifier fixture.

Acceptance criteria:

- On Lambda Cloud GPU instance, the agent detects GPU.
- LitServe app is generated.
- Server starts.
- `/health` passes.
- `/predict` returns a valid response for sample image.
- Final report includes endpoint, logs, model artifact, verification results, and rollback command.

### Phase 2A — Agentic GPU Inference UX

Goal: make GPU inference feel like a natural-language product rather than a script runner.

Deliverables:

- Implement `deploy_gpu_inference` as the general GPU inference workflow.
- Add backend selection for LitServe, Gradio, vLLM, KServe, TorchServe, and FastAPI Lambda CPU.
- Add model artifact inspection and framework detection.
- Add GPU utilization capture through `nvidia-smi` or equivalent.
- Add latency summary with p50 and p95 for live endpoints.
- Add approval prompt before server start, port exposure, package install, image build, or cloud deployment.
- Add final endpoint card in the frontend.

Acceptance criteria:

- “Deploy this classifier on Lambda GPU” selects LitServe and verifies health/predict.
- “Serve this LLM with vLLM” selects vLLM and reports token/latency metrics when implemented.
- “Run this model and tell me if GPU is being used” returns GPU utilization evidence.
- “Why is my inference slow?” produces latency, logs, batching, GPU, and model-size diagnostics.
- “Rollback to previous working model” uses the stored rollback plan.

### Phase 3 — Train And Improve Loop

Goal: add controlled autoresearch-style improvement.

Deliverables:

- Implement `train_and_track`.
- Implement `train_until_better`.
- Add fixed experiment budgets.
- Add Hydra override tracking.
- Add MLflow metric comparison.
- Add keep/discard decision with checkpoint rollback.

Acceptance criteria:

- Agent establishes a baseline before changing config.
- Each attempt logs metric, config, duration, and decision.
- Worse changes are discarded.
- Best model artifact is selected for deployment.

### Phase 4 — Demo And Product UX

Goal: make the agent understandable and usable.

Deliverables:

- Implement `deploy_gradio_demo`.
- Add frontend workflow timeline.
- Show current stage, tool status, approvals, artifacts, endpoint URL, test results, failure reason, retry/rollback actions.
- Add event streaming over WebSocket with session ownership.

Acceptance criteria:

- User can watch the workflow progress from request to report.
- Approval gates appear before risky steps.
- Failed steps show exact reason and next action.

### Phase 5 — Kubernetes Production Path

Goal: support one serious Kubernetes path.

Deliverables:

- Choose KServe as first Kubernetes path unless TorchServe is required by capstone.
- Add Kubernetes/EKS detection.
- Add ECR image flow.
- Generate InferenceService, HPA, ingress, config, secret templates.
- Add server-side dry-run validation.
- Add canary and rollback plan.

Acceptance criteria:

- “Deploy to KServe with canary rollout” produces correct target and plan.
- Manifests pass validation.
- Rollback and monitoring steps are included.

### Phase 6 — Production Governance

Goal: harden the system for real users.

Deliverables:

- Persist sessions, events, approvals, tool results, artifacts, deployments, endpoint URLs, model versions, and rollback plans.
- Ensure API keys/users are DB-backed.
- Enforce session ownership on HTTP and WebSocket paths.
- Add rate limits, usage metering, and cost estimates.
- Add tool permission policy.

Acceptance criteria:

- Users cannot access other users’ sessions.
- Every deployment has an audit trail.
- Every risky tool has approval and permission metadata.

## Natural Language Test Matrix

Create tests for these prompts:

| Prompt | Expected Target | Required Assertions |
| --- | --- | --- |
| “Deploy this model on Lambda Labs GPU” | LitServe | no FastAPI Lambda, GPU detection, health/predict smoke tests |
| “Run this model and tell me if GPU is being used” | deploy_gpu_inference | GPU utilization evidence, selected backend, endpoint test |
| “Serve this LLM with vLLM and optimize latency” | vLLM | no LitServe/TorchServe unless requested, token metrics, latency report |
| “Why is my inference slow?” | inference diagnostics | p50/p95, logs, GPU utilization, batching/config suggestions |
| “Create a Gradio demo” | Gradio | app generated, launch command, sample prediction path |
| “Set up DVC with S3” | setup_pipeline | DVC init, remote config, data add, validation |
| “Train until 90% accuracy” | train_until_better | baseline, MLflow metric, improvement loop, stop condition |
| “Deploy to KServe with canary rollout” | KServe | InferenceService, canary, dry-run, rollback |
| “Build full capstone pipeline” | capstone workflow | Hydra, DVC, MLflow, Docker, CI/CD, serving, report |

## Data Model Additions

Minimum persistent records:

- `WorkflowRun`: id, user_id, session_id, workflow_name, status, started_at, finished_at.
- `WorkflowStep`: id, workflow_run_id, tool, args_hash, status, output_summary, error, started_at, finished_at.
- `ArtifactManifest`: workflow_run_id, path, type, checksum, created_by_step.
- `DeploymentRecord`: workflow_run_id, target, endpoint_url, model_version, image_uri, status.
- `VerificationResult`: workflow_run_id, step_id, check_name, passed, evidence.
- `ApprovalRecord`: workflow_run_id, step_id, user_id, approved, timestamp.
- `ExperimentAttempt`: workflow_run_id, attempt, config_overrides, metric, decision, duration_sec.

## UX Requirements

The frontend should show:

- Current workflow stage.
- Tool execution timeline.
- Approval prompts.
- Generated files.
- Verification results.
- Endpoint URL.
- Failure reason.
- Suggested fix.
- Retry and rollback buttons.
- Final deployment report.

For GPU inference deployments, the frontend should also show:

- Selected backend.
- GPU name, memory, CUDA, and driver status.
- Server status and endpoint URL.
- Health check result.
- Prediction test result.
- p50/p95 latency.
- GPU utilization evidence.
- Rollback action.

## Non-Functional Requirements

- No silent deploys: build, deploy, rollback, and credential-sensitive steps require approval.
- No unverified success: workflows need explicit success contract fields.
- No secret leakage: secrets stay in env vars, secret stores, or Kubernetes secrets.
- Reproducible outputs: generated files and commands are recorded.
- Auditable state: sessions, approvals, tool results, and artifacts persist.
- Safe experiment loop: fixed budget, metric comparison, rollback point.
- Clear target disambiguation: Lambda Labs GPU is not AWS Lambda.

## Success Metrics

- One Lambda Cloud GPU LitServe E2E demo passes from natural language.
- One general `deploy_gpu_inference` workflow can select backend, start service, verify endpoint, and report GPU evidence.
- 90%+ route accuracy on natural-language test matrix.
- 100% of production deployment plans include verification and rollback.
- Mean time to verified LitServe deploy under 10 minutes for fixture project.
- Failed workflow reports include actionable root cause and next command.
- No cross-user session access in API or WebSocket tests.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Tool sprawl without reliability | Use deterministic workflow templates and success contracts |
| Wrong Lambda target | Hard-code disambiguation and tests for Lambda Labs vs AWS Lambda |
| Wrong serving backend for model type | Use deterministic backend selection rules and model artifact inspection |
| Endpoint runs on CPU by accident | Capture GPU utilization evidence before marking GPU deployment successful |
| Unbounded self-improvement edits | Use fixed-budget experiments, narrow Hydra overrides, checkpoints |
| Credential leakage | Redaction, secret scanning, approval gates, no secret writes to repo |
| Deployment appears successful but endpoint fails | Mandatory health/predict smoke tests |
| Kubernetes complexity slows delivery | Start with KServe only, add TorchServe later if needed |
| External repo bloat | Borrow concepts from Hermes/autoresearch, do not vendor whole systems early |

## First Capstone Demo Definition

The first production-quality demo is:

> Auto-MLOps Agent takes an image classifier project and deploys it to Lambda Cloud GPU with LitServe.

Minimum success:

1. Detects runtime and GPU.
2. Creates LitServe app.
3. Creates Dockerfile.
4. Builds or validates image.
5. Runs LitServe server.
6. Calls `/health`.
7. Calls `/predict` with sample image.
8. Writes monitoring and rollback report.
9. Produces final artifact manifest.

This demo proves the agentic loop because it starts from natural language and ends with a verified running service.
