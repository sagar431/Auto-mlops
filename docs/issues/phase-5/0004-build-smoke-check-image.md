# Issue 0004: Build And Smoke-Check Image

## Title

Build and smoke-check the **Capstone Runtime Image** when Docker is available.

## Goal

Add local **Container Build Evidence** for Docker availability, image build attempts, build results, image id or tag, and bounded smoke checks. The workflow should support `container_local_ready` without Docker by recording deferred build evidence, while `container_capstone_complete` must require a successful image build.

## Dependencies

- Depends on issue 0003 because build and smoke evidence require a generated or validated Dockerfile/build spec.

## Expected Behavior

- Detect Docker availability and record version/context evidence when available.
- For `container_local_ready`:
  - If Docker is available, request approval and attempt image build.
  - If Docker is unavailable, record `image_build_deferred_reported` and allow local readiness to proceed only when build-spec validation passed.
- For `container_capstone_complete`:
  - Require Docker availability.
  - Require successful image build.
  - Record image tag plus digest or local image id.
  - Record the image as a selected or external artifact in the **Artifact Manifest**.
- Run a minimal bounded smoke check when feasible, such as import validation, CLI help, or entrypoint dry-run.
- Do not start full LitServe or KServe endpoints.
- Do not run full training, HPO, LR finder, stress tests, or load tests.
- Produce **Verification Results** for Docker availability, build attempt, build success/failure, smoke check, and deferred build state.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `docker_availability_reported`: **Observed Evidence**
- `image_build_attempt_reported`: **Observed Evidence** when Docker is available in `container_local_ready`
- `image_build_deferred_reported`: **Observed Evidence** or blocked/deferred evidence when Docker is unavailable in `container_local_ready`
- `docker_available`: **Observed Evidence** for `container_capstone_complete`
- `image_build_succeeded`: **Observed Evidence** for `container_capstone_complete`
- `container_smoke_check_passed`: **Observed Evidence** for `container_capstone_complete`
- `container_artifact_manifest_reported`: **Observed Evidence** for image id, tag, digest, or local image reference when available

Build failure must produce structured failed **Verification Result** and **Contract Failure** records.

## Approval Gates

- Docker availability detection is read-only and does not require an **Approval Gate**.
- Running `docker build` requires an **Approval Gate** with `builds_image`.
- Running a container smoke command that executes user project code, starts a process, or executes an entrypoint requires an **Approval Gate** with `executes_project_code`.
- This issue must not perform registry login or push.

## Tests To Add

- Tests that Docker unavailable records deferred build evidence for `container_local_ready`.
- Tests that Docker unavailable blocks `container_capstone_complete`.
- Tests that Docker build requires approval.
- Tests that successful build records image id or tag and **Artifact Manifest** entries.
- Tests that build failure produces structured failed **Verification Result** evidence.
- Tests that bounded smoke checks require approval when they execute project code.
- Tests that full serving endpoint checks are not required or run in Phase 5.

## Out Of Scope

- Registry target validation, registry authentication, registry push, GitHub Actions workflow generation, and final `container_ci_evidence.json` writing.
- Full LitServe/KServe endpoint startup, `/health`, `/predict`, Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, stress tests, load tests, full training in CI, frontend, final report, video generation, secret mutation, and unapproved registry push.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_execute_step.py -k container -q`
- `.venv/bin/python -m pytest tests/unit/test_agent_loop.py -k container -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase5_prepare_capstone_container_ci.py -q`
- `.venv/bin/python -m ruff check workflow agent action mcp_mlops_tools.py tests/unit/test_execute_step.py tests/unit/test_agent_loop.py tests/e2e/test_phase5_prepare_capstone_container_ci.py`
