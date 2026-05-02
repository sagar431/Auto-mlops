# Issue 0008: Add Phase 0 Routing And Contract Test Matrix

## Title

Add Prompt Contract Tests for Phase 0 routing and success-contract obligations.

## Problem

Phase 0 requires prompt tests for natural-language route and target selection, but those tests should assert registry-owned behavior rather than prompt wording. The test matrix should prove routing, rejected workflows, required inputs, and contract obligations.

## Scope

- Add **Prompt Contract Tests** for the Phase 0 natural-language prompts.
- Assert expected **Workflow Selection**, matched aliases, rejected workflows, missing inputs, and success contract obligations.
- Cover Lambda Labs GPU routing directly to `deploy_litserve_gpu`.
- Cover setup, Gradio demo, KServe production, and general GPU inference prompts.

## Out Of Scope

- Real deployment execution.
- Real GPU, Docker, Kubernetes, or HTTP checks.
- Broad LLM quality evaluation.
- Frontend tests.

## Files Likely Touched

- `tests/unit/test_phase0_prompt_contracts.py`
- `tests/unit/test_perception.py`
- `tests/unit/test_decision.py`
- `prompts/perception_prompt.txt`
- `prompts/decision_prompt.txt`
- `workflow/`

## Tests To Write First

- “Deploy this model on Lambda Labs GPU” selects `deploy_litserve_gpu` and rejects FastAPI Lambda.
- “Create a Gradio demo” selects `deploy_gradio_demo`.
- “Deploy to KServe with canary rollout” selects `deploy_kserve_production`.
- “Run this model and tell me if GPU is being used” selects `deploy_gpu_inference` with GPU evidence obligations.
- “Set up MLOps for this project” selects `setup_pipeline`.

## Acceptance Criteria

- Prompt contract tests assert registry behavior, not just prompt text.
- Each test checks selected workflow, rejected workflows when relevant, required inputs, and contract obligations.
- The Lambda Labs versus AWS Lambda ambiguity is locked by tests.
- Tests do not require real external services or credentials.

## Recommended Test Command

```bash
.venv/bin/python -m pytest tests/unit/test_phase0_prompt_contracts.py tests/unit/test_perception.py tests/unit/test_decision.py -q
```

