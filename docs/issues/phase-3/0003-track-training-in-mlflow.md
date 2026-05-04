# Issue 0003: Track Training In MLflow

## Title

Log params, metrics, artifacts, and model checkpoint to MLflow.

## Problem

The PRD requires `train_and_track` to produce an MLflow run and make training evidence auditable. Capturing metrics in memory or logs is not enough for the capstone path because downstream comparison and deployment need a durable run id, metric history, params, artifacts, and selected checkpoint reference.

Phase 3 should integrate MLflow tracking without hiding training failures or marking runs successful from declared evidence alone.

## Scope

- Initialize or validate the configured MLflow experiment for training.
- Start or attach to an MLflow run for the bounded training command.
- Log Hydra config values, effective overrides, dataset evidence, training budget, device, dependency/version summary, and entrypoint command as params or tags.
- Log captured metrics with step information when available.
- Log stdout/stderr summaries or log files as artifacts.
- Log checkpoint or model artifact paths produced by training.
- Record MLflow experiment id, run id, tracking URI, artifact URI, and run status as structured evidence.
- Add a `confirm_mlflow_run_exists`-style verification before `train_and_track` can satisfy its tracking contract.

## Out of scope

- Creating a remote MLflow tracking server.
- Model registry promotion beyond marking or logging a candidate artifact.
- Best-model comparison policy.
- HuggingFace, KServe, or Lambda deployment.
- Secret or credential management for remote tracking.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `metrics/`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/test_phase3_train_and_track.py`

## Tests To Write First

- `train_and_track` cannot satisfy the MLflow contract without an MLflow run id.
- A fixture bounded training run logs params, metrics, artifacts, and checkpoint reference to MLflow.
- Failed training records MLflow run status and logs failure evidence without marking the workflow succeeded.
- Missing or invalid tracking URI blocks with a clear next action when tracking is required.
- The artifact manifest and MLflow artifact URI refer to the same checkpoint or model artifact candidate.

## Acceptance Criteria

- MLflow experiment and run existence are verified.
- Params include effective Hydra overrides and bounded training controls.
- Metrics include the configured target metric.
- Artifacts include logs and a checkpoint or model artifact candidate.
- The final workflow output includes experiment id, run id, tracking URI, artifact URI, and run status.
- `train_and_track` cannot succeed when MLflow tracking is required but the run cannot be verified.

## Dependency/Blocker Notes

- Blocked by 0002.
- 0004 depends on verified MLflow run metadata for baseline comparison and artifact selection.
- Remote tracking credentials and production MLflow server setup remain deferred.
