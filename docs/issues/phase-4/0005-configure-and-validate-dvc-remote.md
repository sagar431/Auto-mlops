# Issue 0005: Configure And Validate DVC Remote

## Title

Configure local or S3 DVC remotes with redacted validation evidence.

## Goal

Add **Capstone Data Remote** support to `prepare_capstone_data` so local remotes can satisfy **Local Data Readiness** and S3 remotes can satisfy the remote-validation portion of **Capstone Data Completeness** without leaking credentials.

## Expected Behavior

- Support local DVC remotes for development evidence.
- Support S3 DVC remotes for capstone-complete evidence.
- Require an **Approval Gate** before changing DVC remote configuration.
- Require `uses_cloud_credentials` risk when configuring or validating S3 remotes.
- Validate S3 remotes through **Credential Capability Evidence**, such as:
  - redacted AWS identity from `aws sts get-caller-identity`
  - reachable bucket or prefix
  - DVC remote resolution
  - harmless access probe where possible
- Redact S3 URLs in **Verification Results**, **Artifact Manifest** metadata, and data-stage evidence.
- Never write access keys, session tokens, raw `.env` contents, or secret material.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `s3_remote_validated`: **Observed Evidence** when `completion_mode` is `capstone_complete`.

It should also record local remote evidence for `local_ready` runs without pretending local evidence satisfies S3 completion.

Evidence should include:

- Remote type: `local`, `s3`, or `missing`.
- Redacted remote URL.
- Validation status.
- Redacted cloud identity metadata when available.
- Bucket or prefix reachability status.
- Blocked capability when credentials or approval are missing.
- **Approval Record** for remote configuration.

## Tests To Add

- Local remote configuration can satisfy local-ready remote evidence without S3 checks.
- `completion_mode=capstone_complete` blocks when no S3 remote is configured.
- S3 remote validation records redacted URL and credential capability evidence.
- Missing AWS credentials blocks with next actions and no secret leakage.
- `.env` files are not read into evidence or mutated to store credentials.
- Remote configuration requires approval before DVC config mutation.

## Out Of Scope

- DVC push or pull.
- Creating S3 buckets.
- Managing IAM users, policies, or AWS credentials.
- Full cloud provisioning.
- Secrets storage.
- CI/CD integration.

## Verification Commands

- `pytest tests/unit/test_agent_loop.py -k dvc_remote`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py -k remote`
- `ruff check mcp_mlops_tools.py agent workflow tests`

## Dependency / Blocker Notes

Blocked by [0004](./0004-dvc-track-capstone-data-package.md).
