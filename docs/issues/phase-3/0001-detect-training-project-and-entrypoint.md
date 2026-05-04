# Issue 0001: Detect Training Project And Entrypoint

## Title

Detect Hydra/PyTorch/TIMM training repos like `mlops_pratice/session-06`.

## Problem

Phase 3 needs training automation, but the agent must not guess how to train arbitrary projects. The first supported path should detect a narrow EMLO session-06-style image-classifier repo before any training command runs.

Detection must identify the project shape, training entrypoint, Hydra config root, DVC data signals, tests, framework dependencies, and checkpoint or artifact conventions. Unsupported or ambiguous projects should block with clear missing evidence instead of falling back to prompt-authored commands.

## Scope

- Add training-project detection for Hydra/PyTorch/TIMM image-classifier repos.
- Detect likely Hydra config directories and config names.
- Detect a train entrypoint such as a Python module or script that accepts Hydra overrides.
- Detect DVC signals such as `.dvc/`, `dvc.yaml`, `.dvc` files, or data paths referenced by config.
- Detect PyTorch and TIMM dependency signals from project files.
- Detect project tests and a minimal command that validates the training entrypoint without full training.
- Detect likely output directories for checkpoints, MLflow runs, logs, and model artifacts.
- Return structured evidence with confidence, detected paths, missing required inputs, and next actions.
- Add routing support so training requests can select `train_and_track` only when required training inputs can be resolved or requested.

## Out of scope

- Running training.
- Creating or rewriting the user's training code.
- Supporting arbitrary TensorFlow, sklearn, LLM, or non-image-classifier training projects.
- Configuring an S3 DVC remote.
- Selecting a best model artifact.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/`
- `tests/fixtures/`

## Tests To Write First

- A session-06-style fixture with Hydra configs, DVC data metadata, TIMM/PyTorch dependency signals, tests, and a train entrypoint is detected as supported.
- A project without a train entrypoint blocks before any training command is proposed.
- A project without Hydra config blocks or requests the missing config input instead of inventing overrides.
- A non-PyTorch or non-image-classifier project is rejected for Phase 3 with a clear unsupported-framework reason.
- Natural-language training requests select `train_and_track` only after required workflow inputs are known or listed as missing.

## Acceptance Criteria

- Detection returns structured **Observed Evidence** for project files that exist.
- The detected train entrypoint, Hydra config path, DVC/data evidence, test command, and artifact output candidates are recorded.
- Unsupported projects produce `blocked` with missing evidence and next action.
- No training command runs during detection.
- Detection results can be consumed by later `train_and_track` and `train_until_better` steps.

## Dependency/Blocker Notes

- No blockers.
- This issue must land before 0002, 0003, 0004, and 0005.
- `train_and_track` and `train_until_better` were intentionally excluded from the Phase 0 registry; this issue should prepare the narrow detection evidence needed before they become executable templates.
