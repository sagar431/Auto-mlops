# Issue 0002: Add Phase 0 Deployment Workflow Templates

## Title

Add testable Phase 0 deployment templates to the Workflow Registry.

## Problem

Phase 0 routing and contract tests need registry-owned templates for the initial deployment paths. Without these templates, prompts can still invent or reorder deployment skeletons.

## Scope

- Add `deploy_litserve_gpu`, `deploy_gpu_inference`, `deploy_gradio_demo`, and `deploy_kserve_production` to the **Workflow Registry**.
- Declare ordered **Workflow Steps**, required **Workflow Inputs**, **Routing Aliases**, **Negative Routing Rules**, **Approval Gates**, and **Success Contracts** for each template.
- Model `deploy_gpu_inference` backend choices as declared **Workflow Branches**.
- Ensure `deploy_litserve_gpu` remains a first-class **LitServe GPU Workflow**.

## Out Of Scope

- Real GPU detection.
- Real LitServe server start.
- Real `/health` or `/predict` execution.
- Kubernetes execution.
- New deployment tools.
- Frontend endpoint cards.

## Files Likely Touched

- `workflow/` or registry module from issue 0001
- `tests/unit/test_workflow_registry.py`
- `tests/unit/` fixtures for templates

## Tests To Write First

- Registry contains exactly the required Phase 0 templates: `setup_pipeline`, `deploy_litserve_gpu`, `deploy_gpu_inference`, `deploy_gradio_demo`, and `deploy_kserve_production`.
- Registry excludes `rollback`, `monitor_and_alert`, `train_and_track`, and `train_until_better`.
- `deploy_litserve_gpu` declares observed evidence requirements for GPU detection, server start, `/health`, and `/predict`.
- `deploy_gpu_inference` declares backend branches rather than arbitrary tool chains.

## Acceptance Criteria

- All required Phase 0 templates are present and testable.
- No fake or placeholder templates are added.
- Deployment templates include contract checks and approval metadata as registry data.
- Lambda Labs GPU routing aliases and AWS Lambda negative rules are represented in template data.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -q
```

