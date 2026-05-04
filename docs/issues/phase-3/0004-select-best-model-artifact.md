# Issue 0004: Select Best Model Artifact

## Title

Compare latest run against baseline and choose a deployable artifact.

## Problem

Training is only useful to the capstone path when the agent can identify a deployable model artifact and explain why it was selected. The PRD requires best-run selection and `train_until_better` requires a keep/discard decision. Phase 3 needs deterministic comparison between a latest run and a baseline using a configured metric, direction, threshold, and artifact evidence.

Without this slice, downstream deployment workflows may pick the wrong checkpoint or deploy a worse model.

## Scope

- Add best-artifact selection for the supported session-06-style training path.
- Accept or derive a baseline from prior MLflow run metadata, a configured baseline metric, or an existing selected artifact.
- Compare latest run metric against baseline using explicit metric name, direction, threshold, and tie policy.
- Select the latest checkpoint only when it meets the configured improvement or threshold rule.
- Keep the previous baseline artifact when the latest run is worse or incomplete.
- Record model artifact path or URI, checksum when available, source run id, metric, comparison result, and decision.
- Add the first controlled `train_until_better` contract shape: baseline, one Hydra override set, fixed-budget attempt, compare, keep/discard, and stop reason.

## Out of scope

- Broad HPO search.
- Repeated multi-attempt optimization beyond the minimal controlled loop contract.
- Editing training code to improve results.
- Registering models in a production registry.
- Deploying the selected artifact.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `metrics/`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/test_phase3_train_until_better.py`

## Tests To Write First

- Given a latest run that beats the baseline, the latest checkpoint is selected and recorded as `selected`.
- Given a latest run that is worse than baseline, the baseline artifact remains selected and the latest change is discarded.
- Given missing metric direction or threshold, selection blocks with a clear next action.
- Given missing checkpoint evidence for the winning run, selection blocks instead of selecting a metric-only run.
- `train_until_better` records baseline, Hydra override set, metric, duration, decision, and stop condition for one bounded attempt.

## Acceptance Criteria

- Selection uses an explicit metric name, direction, threshold, and baseline.
- The selected artifact is recorded in the **Artifact Manifest** with `state: selected`.
- Worse changes are discarded with a structured reason.
- The chosen model artifact path or URI is available to `deploy_litserve_gpu` and future deployment workflows.
- `train_until_better` cannot run without fixed budget controls and a rollback point.
- Each improvement attempt records config, metric, duration, decision, and stop reason.

## Dependency/Blocker Notes

- Blocked by 0002 and 0003.
- 0005 depends on this issue before `build_capstone_pipeline` can treat training as an implemented sub-workflow.
- Full HPO, LR finder automation, and repeated search policies remain later-phase work unless explicitly scoped into this narrow loop.
