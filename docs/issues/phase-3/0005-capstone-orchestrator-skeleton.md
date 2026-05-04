# Issue 0005: Capstone Orchestrator Skeleton

## Title

Add `build_capstone_pipeline` as the Capstone Orchestrator skeleton.

## Problem

The product north star is automating the EMLO capstone end-to-end, but Auto-MLOps must reach that through verified PRD phases. Phase 3 should add a top-level Capstone Orchestrator without claiming the full capstone is implemented.

The Capstone Orchestrator should coordinate setup, data, training, deployment, and reporting stages, but at first it may invoke only implemented sub-workflows. Missing future capabilities must be recorded as blocked or deferred evidence so the workflow output is honest and useful.

## Scope

- Add top-level workflow id `build_capstone_pipeline`.
- Use "Capstone Orchestrator" as the product name for the coordinator.
- Add a deterministic **Workflow Template** skeleton with declared stages for setup, data, train, deploy, monitor, and report.
- Invoke only implemented sub-workflows at first, such as `setup_pipeline`, `train_and_track`, `train_until_better`, and Phase 2 LitServe deployment workflows when their inputs and contracts are satisfied.
- Record unsupported future stages as blocked or deferred evidence with capability name, reason, and later-phase pointer.
- Ensure sub-workflow success is contract-derived before the Capstone Orchestrator treats a stage as complete.
- Add routing for natural-language requests such as "Build full capstone pipeline" to `build_capstone_pipeline`.
- Produce a final orchestrator summary containing completed stages, blocked stages, deferred capabilities, artifact manifest entries, selected model artifact, endpoint evidence when available, and next actions.

## Out of scope

- Implementing missing sub-workflows just to make the orchestrator look complete.
- S3 DVC remote automation.
- KServe, Helm, ArgoCD, or production Kubernetes deployment.
- HuggingFace Spaces publishing.
- AWS Lambda serverless deployment.
- Stress tests.
- Frontend workflow timeline or endpoint cards.
- Final report or video generation.
- Editing `PRD.md`.

## Files Likely Touched

- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `summarization/`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`
- `tests/e2e/test_phase3_capstone_orchestrator.py`

## Tests To Write First

- "Build full capstone pipeline" selects `build_capstone_pipeline`.
- The Capstone Orchestrator does not call an unimplemented sub-workflow.
- A completed setup sub-workflow is recorded as complete only when its **Success Contract** succeeds.
- A completed training sub-workflow is recorded as complete only when `train_and_track` or `train_until_better` succeeds.
- Missing S3 DVC remote, KServe/Helm/ArgoCD, HuggingFace Spaces, AWS Lambda, stress tests, frontend, and final report/video are recorded as blocked or deferred evidence.
- The orchestrator can finish with a non-success status that still reports completed implemented stages and clear next actions.

## Acceptance Criteria

- `build_capstone_pipeline` exists as a top-level **Workflow Template**.
- The Capstone Orchestrator invokes only implemented sub-workflows.
- Sub-workflow completion is derived from each sub-workflow's **Success Contract**, not summary text.
- Deferred future capabilities are explicit in structured output.
- The final orchestrator output lists completed stages, blocked stages, deferred stages, selected artifacts, and next actions.
- The workflow cannot mark the full capstone succeeded while required later-phase capabilities are deferred.

## Dependency/Blocker Notes

- Blocked by 0001, 0002, 0003, and 0004.
- Depends on completed Phase 1 `setup_pipeline` and completed Phase 2 LitServe deployment foundations already on `main`.
- Future issues must replace deferred evidence with implemented sub-workflows one capability at a time.
