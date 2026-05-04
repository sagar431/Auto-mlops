# Issue 0002: Run Training And Capture Metrics

## Title

Run bounded training and capture metrics, logs, duration, and artifacts.

## Problem

After a supported training project is detected, Auto-MLOps needs to run training as a controlled workflow step rather than a free-form shell command. The PRD's `train_and_track` and `train_until_better` workflows require fixed budgets, metric capture, duration tracking, and artifact reporting before the agent can compare or deploy a model.

The first implementation should run a bounded session-06-style train entrypoint with explicit controls so a failed, slow, or expensive training run cannot be mistaken for success.

## Scope

- Add `train_and_track` as a real **Workflow Template** with required inputs and a **Success Contract**.
- Project detected training metadata into runtime step arguments.
- Run the detected train entrypoint with bounded controls such as timeout, max epochs, device, data subset, and Hydra overrides.
- Capture stdout, stderr, log file paths, start time, finish time, duration, exit code, and failure reason.
- Parse or collect training metrics such as loss, accuracy, validation accuracy, or configured target metric.
- Record generated checkpoints, model artifacts, config snapshots, and logs in the **Artifact Manifest**.
- Return `blocked` when budget, entrypoint, data, dependency, or approval requirements are missing.
- Return `failed` when the bounded command runs but does not produce required metrics or artifacts.

## Out of scope

- MLflow logging beyond consuming or forwarding captured run evidence.
- Best-model selection against a baseline.
- Unbounded HPO.
- Editing model code or config files to improve accuracy.
- GPU/cloud provisioning.
- S3 DVC remote configuration.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/test_phase3_train_and_track.py`
- `tests/fixtures/`

## Tests To Write First

- Without detection evidence from 0001, `train_and_track` blocks before executing the train command.
- Without explicit bounded training controls, `train_and_track` blocks with a clear next action.
- A fixture train entrypoint that emits metrics and writes a checkpoint reaches contract-derived `succeeded`.
- A train command that exits non-zero returns `failed` and records logs, duration, command, and failure reason.
- A train command that exits zero but produces no configured metric returns `failed` or `blocked` instead of `succeeded`.
- Captured checkpoints, logs, and config snapshots appear as structured **Artifact Manifest** entries.

## Acceptance Criteria

- `train_and_track` exists as a registry-owned **Workflow Template**, not a prompt-authored skeleton.
- Training runs only with explicit timeout and budget controls.
- Metrics, logs, duration, command, exit code, and artifacts are captured as structured evidence.
- The workflow cannot succeed without a captured target metric.
- The workflow cannot succeed without a checkpoint or explicit no-artifact blocked reason.
- Failure output includes actionable next steps and paths to available logs.

## Dependency/Blocker Notes

- Blocked by 0001.
- 0003 consumes the captured params, metrics, logs, duration, and artifacts from this issue.
- 0004 consumes the captured metric and checkpoint evidence from this issue.
