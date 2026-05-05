# Issue 0003: Generate Or Validate Runtime Image Spec

## Title

Generate or validate the default **Capstone Runtime Image** Dockerfile or build spec.

## Goal

Create the first container build-spec slice for Phase 5. The workflow should conservatively generate or validate one default **Capstone Runtime Image** for bounded tests, model artifact validation, and later deployment handoff, while avoiding broad framework-specific packaging or unsafe project mutations.

## Dependencies

- Depends on issue 0002 because build-spec generation should record resolved upstream evidence, local model artifact fallback, dependency context, and deferred upstream gaps when applicable.

## Expected Behavior

- Detect dependency context in priority order:
  1. `uv.lock` or `pyproject.toml`
  2. `requirements.txt`
  3. `setup.py`
- Detect an existing Dockerfile or container build spec and validate/reference it rather than overwriting it by default.
- Generate a default Dockerfile or build spec only after an **Approval Gate** when project files will be written.
- Require approval before overwriting an existing Dockerfile or `.dockerignore`.
- Default to a Python 3.10 or newer base image unless project evidence requires otherwise.
- Avoid heavyweight CUDA base images unless explicitly requested or observed as required.
- Record `intended_roles`, such as:
  - `ci`
  - `training_validation`
  - `inference_validation`
- Include bounded default commands for tests or model validation.
- Do not embed secrets, raw `.env` contents, absolute source dataset paths, registry tokens, AWS keys, or local-only configuration.
- Add evidence labels or comments only when useful and non-secret.
- Record generated or validated build-spec artifacts in the **Artifact Manifest**.

## Success Contract / Evidence Expectations

This issue should satisfy or fail:

- `container_build_spec_reported`: **Declared Evidence** for generated specs or **Observed Evidence** for validated existing specs
- `dependency_context_reported`: **Observed Evidence**
- `secret_safety_validated`: **Observed Evidence**
- `container_artifact_manifest_reported`: **Declared Evidence** or **Observed Evidence** for Dockerfile/build-spec artifacts

Evidence should include:

- Build spec path
- Whether the spec was generated, validated, adapted, or blocked
- Dependency files discovered
- Base image decision
- Intended roles
- Bounded commands
- Secret/path safety validation results
- Approval record for writes or overwrites when applicable

## Approval Gates

- Read-only inspection of dependency files, existing Dockerfiles, and project structure does not require approval.
- Writing a new Dockerfile, `.dockerignore`, or build-spec file requires an **Approval Gate** with `writes_project_files`.
- Overwriting or adapting an existing Dockerfile requires an **Approval Gate** with `writes_project_files`.
- This issue must not run Docker builds or execute container entrypoints.

## Tests To Add

- Tests for dependency detection priority: `uv.lock`/`pyproject.toml`, then `requirements.txt`, then `setup.py`.
- Tests that existing Dockerfiles are validated/referenced without overwrite by default.
- Tests that writes and overwrites are blocked without an **Approval Record**.
- Tests that generated specs include intended roles and bounded validation commands.
- Tests that generated specs avoid secrets, `.env` values, absolute source dataset paths, and unnecessary CUDA bases.
- Tests that build-spec artifacts are recorded in the **Artifact Manifest**.

## Out Of Scope

- Docker image build, image smoke checks, registry target validation, registry login or push, GitHub Actions workflow generation, and final `container_ci_evidence.json` writing.
- Perfect framework detection, broad framework-specific Dockerfile generation, multi-image packaging, separate training/inference images, slim production images, CUDA-optimized variants unless explicitly required for the default runtime image.
- Kubernetes, KServe, Helm, ArgoCD, GitOps, EKS provisioning, endpoint deployment checks, stress tests, frontend, final report, video generation, secret mutation, and unapproved registry push.
- Editing `PRD.md`.

## Verification Commands

- `.venv/bin/python -m pytest tests/unit/test_workflow_registry.py -k container -q`
- `.venv/bin/python -m pytest tests/unit/test_execute_step.py -k container -q`
- `.venv/bin/python -m pytest tests/e2e/test_phase5_prepare_capstone_container_ci.py -q`
- `.venv/bin/python -m ruff check workflow action mcp_mlops_tools.py tests/unit/test_workflow_registry.py tests/unit/test_execute_step.py tests/e2e/test_phase5_prepare_capstone_container_ci.py`
