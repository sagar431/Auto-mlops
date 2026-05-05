# Issue 0005: Configure And Validate Registry Target

## Title

Configure and validate **Container Registry Evidence** for the Phase 5 image target.

## Goal

Add registry target selection and validation without pushing images. GHCR should be the first-class default because it aligns with GitHub Actions and repository workflows. DockerHub and ECR may be accepted as explicit declared targets, but provider-specific deep automation remains deferred unless later issues require it.

## Dependencies

- Depends on issue 0004 because registry target evidence should reference a buildable image tag, local image id, or planned image reference.

## Expected Behavior

- Select GHCR as the default registry target when no explicit target is provided and repository context supports it.
- Accept DockerHub or ECR only as explicit user-provided registry targets.
- Validate registry target shape and image naming without storing secrets.
- Record redacted image references and registry URLs. Do not store raw tokens, passwords, AWS keys, GitHub tokens, `.env` contents, or unredacted secret material.
- Detect whether registry authentication capability appears available without performing login or push in this issue.
- For `container_local_ready`, allow absent registry target or credentials when missing registry capability is recorded as blocked/deferred.
- For `container_capstone_complete`, require registry target validation and authentication capability evidence before later push steps may run.
- Record **Container Registry Evidence** in structured evidence and **Artifact Manifest** entries where applicable.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `registry_target_validated`: **Observed Evidence** for `container_capstone_complete`
- `registry_auth_capability_verified`: **Observed Evidence** or structured blocked evidence for `container_capstone_complete`
- `secret_safety_validated`: **Observed Evidence**

Evidence should include:

- Registry provider
- Redacted registry URL or image reference
- Image name and tag decision
- Whether provider support is first-class or declared target support
- Authentication capability status without secret values
- Missing credential next actions when blocked/deferred

## Approval Gates

- Read-only registry target parsing and local config inspection do not require approval.
- Authenticated remote registry probes require an **Approval Gate** with **Remote Service Credential Use** when they use GitHub, GHCR, DockerHub, ECR, or other registry credentials.
- This issue must not run registry login or push.

## Tests To Add

- Tests that GHCR is selected as the default first-class registry target.
- Tests that DockerHub and ECR are accepted only when explicitly declared.
- Tests that registry URLs and image references are redacted where needed.
- Tests that raw credentials, tokens, AWS keys, and `.env` contents are not stored.
- Tests that `container_local_ready` defers missing registry capability.
- Tests that `container_capstone_complete` blocks when registry target validation or auth capability evidence is missing.
- Tests that registry push does not run in this issue.

## Out Of Scope

- Registry login, registry push, digest capture from push, GitHub Actions workflow generation, final `container_ci_evidence.json` writing, Kubernetes manifests, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment checks, stress tests, frontend, final report, video generation, secret creation/mutation, and unapproved registry push.
- DockerHub/ECR deep automation beyond declared target support.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -k registry -q`
- `.venv/bin/python -m pytest tests/unit/test_execute_step.py -k registry -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase5_prepare_capstone_container_ci.py -q`
- `.venv/bin/python -m ruff check workflow action mcp_mlops_tools.py tests/unit/test_workflow_registry.py tests/unit/test_execute_step.py tests/e2e/test_phase5_prepare_capstone_container_ci.py`
