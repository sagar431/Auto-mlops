# Issue 0006: Approval-Gated Registry Login And Push

## Title

Run approval-gated registry login and image push for `container_capstone_complete`.

## Goal

Add the risky registry transfer slice. The workflow must separate approval evidence from push execution evidence, prevent unapproved pushes, avoid secret storage, and record pushed image references and digests when available.

## Dependencies

- Depends on issue 0005 because registry login and push should run only after registry target validation and authentication capability checks are in place.

## Expected Behavior

- Require an **Approval Record** before any registry login or push step runs.
- Require the right **Risk Categories**:
  - **Remote Service Credential Use** for registry login or authenticated registry operations
  - `pushes_registry` for image push
- Keep `registry_push_approved` separate from `registry_push_succeeded`.
- For `container_local_ready`, registry push may remain deferred with next actions.
- For `container_capstone_complete`, require:
  - registry target validation
  - auth capability evidence
  - approval for push
  - observed successful push evidence
  - pushed image reference
  - digest when available
- Record failed auth or failed push as structured failed **Verification Results** and **Contract Failure** records.
- Do not write GitHub secrets, registry tokens, AWS keys, Docker credentials, or `.env` values.
- Redact registry URLs and command output that could expose credentials.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `registry_auth_capability_verified`: **Observed Evidence**
- `registry_push_approved`: **Observed Evidence** from an **Approval Record**
- `registry_push_succeeded`: **Observed Evidence** from registry command or API result
- `pushed_image_reference_reported`: **Observed Evidence**
- `secret_safety_validated`: **Observed Evidence**

Evidence should include:

- Approval record id or structured approval evidence
- Risk categories acknowledged
- Redacted login target
- Push command summary without secrets
- Pushed image reference
- Digest when available
- Structured failure reason and next action for denied approval, missing credentials, failed auth, or failed push

## Approval Gates

- Registry login requires an **Approval Gate** with **Remote Service Credential Use**.
- Registry push requires an **Approval Gate** with **Remote Service Credential Use** and `pushes_registry`.
- Denied approval must block the workflow before login or push executes.
- This issue must not mutate secrets or write credential material anywhere in the repository.

## Tests To Add

- Tests that registry login and push do not run without approval.
- Tests that denied approval blocks before registry commands execute.
- Tests that `registry_push_approved` can pass while `registry_push_succeeded` fails when push fails.
- Tests that a push attempt without approval is invalid and cannot satisfy the contract.
- Tests that pushed image references and digests are recorded when available.
- Tests that failed auth and failed push produce structured **Verification Results** and next actions.
- Tests that no secrets, registry tokens, AWS keys, Docker credentials, or `.env` values are stored.

## Out Of Scope

- Creating or mutating GitHub secrets, registry tokens, AWS credentials, `.env`, Kubernetes imagePullSecrets, or cloud infrastructure.
- GitHub Actions workflow generation, final `container_ci_evidence.json` writing, Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment checks, stress tests, frontend, final report, video generation, and unapproved registry push.
- DockerHub/ECR deep automation beyond declared target support unless explicitly required by a later issue.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_agent_loop.py -k approval -q`
- `.venv/bin/python -m pytest tests/unit/test_execute_step.py -k registry -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase5_prepare_capstone_container_ci.py -q`
- `.venv/bin/python -m ruff check workflow agent action mcp_mlops_tools.py tests/unit/test_agent_loop.py tests/unit/test_execute_step.py tests/e2e/test_phase5_prepare_capstone_container_ci.py`
