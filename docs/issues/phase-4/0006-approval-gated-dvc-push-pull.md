# Issue 0006: Approval-Gated DVC Push/Pull

## Title

Gate DVC push and pull behind approvals and record transfer evidence.

## Goal

Add **Approval-Gated Data Transfer** steps to `prepare_capstone_data` so DVC push and pull can produce **Observed Evidence** when approved, while missing approval or credentials produce blocked evidence instead of unsafe automatic transfer.

## Expected Behavior

- Add registry-owned steps for `push_capstone_data` and `pull_capstone_data`.
- Require an **Approval Gate** before `dvc push`.
- Require an **Approval Gate** before `dvc pull`.
- Use risk categories:
  - `push_capstone_data`: `uses_cloud_credentials`
  - `pull_capstone_data`: `uses_cloud_credentials`, `writes_project_files`
- Record blocked evidence when approval is missing or denied.
- Record blocked evidence when credentials are missing.
- Record **Observed Evidence** after successful push or pull.
- Keep remote configuration and validation separate from transfer.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `s3_transfer_completed`: **Observed Evidence** when `completion_mode` is `capstone_complete`.

Evidence should include:

- Transfer direction: push or pull.
- Remote name.
- Redacted remote URL.
- Paths transferred when available.
- DVC command status.
- Timestamp or duration when available.
- **Approval Record**.
- Blocked capability when approval or credentials are missing.

## Tests To Add

- `dvc push` does not run without approval.
- `dvc pull` does not run without approval.
- Denied approval records block the transfer step and preserve prior completed evidence.
- Successful approved push produces `s3_transfer_completed` evidence.
- Missing credentials block transfer with next action.
- `completion_mode=local_ready` may succeed without S3 transfer but records deferred or blocked S3 transfer capability.
- `completion_mode=capstone_complete` blocks until approved S3 transfer evidence exists.

## Out Of Scope

- Automatic S3 transfer without approval.
- Creating buckets or credentials.
- Retrying large failed transfers with resume logic.
- Downloading datasets from external sources.
- CI/CD upload integration.

## Verification Commands

- `pytest tests/unit/test_agent_loop.py -k dvc_push`
- `pytest tests/unit/test_agent_loop.py -k dvc_pull`
- `pytest tests/e2e/test_phase4_prepare_capstone_data.py -k transfer`
- `ruff check mcp_mlops_tools.py agent workflow tests`

## Dependency / Blocker Notes

Blocked by [0005](./0005-configure-and-validate-dvc-remote.md).
