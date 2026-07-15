# Auto-MLOps Project Handoff

- Last updated: 2026-07-15
- Repository: `sagar431/Auto-mlops`
- Default branch: `main`
- Last completed product milestone on `main`: `7437188a224c4979c0361144a8a886eb6719e402`

## How to resume in a fresh Codex session

After cloning or pulling the repository, use this prompt:

> Read `AGENTS.md`, `CLAUDE.md`, and `docs/PROJECT_HANDOFF.md` completely. Treat the code, tests, canonical MCP fixture, and current Git history as authoritative if this handoff is stale. Verify the branch and worktree, then continue from **Current next task**. Do not perform cloud operations, commit, push, open a PR, or merge unless I explicitly request it.

Before starting work:

```bash
git switch main
git pull --ff-only
git status -sb
uv sync --extra dev --locked
```

Do not copy secrets into this document. Keep API keys, cloud credentials, tokens,
datasets, checkpoints, local databases, and generated artifacts outside Git.

## Product goal

Auto-MLOps is a focused agent that turns natural-language requests into verified
MLOps workflows. It should execute and prove a small number of reliable paths,
not merely generate configuration files or expose a large unverified tool list.

The target capstone journey is:

```text
natural-language request
  -> inspect an ML project and data
  -> configure reproducible training
  -> version data with DVC
  -> track and compare experiments with MLflow
  -> select and package a model
  -> serve it through a verified API/container
  -> deploy through an approval-gated target
  -> test health, prediction, load, monitoring, and rollback
  -> report exact evidence and artifacts
```

The course-aligned destination includes Hydra, DVC with S3, MLflow and HPO or a
learning-rate finder, Docker and a registry, GitHub Actions, FastAPI, Kubernetes
and KServe, Helm, ArgoCD, Lambda, a Hugging Face demo, stress testing,
autoscaling, monitoring, and rollback. These should be added through bounded,
verified milestones rather than one large autonomous deployment.

`PRD.md` is the product source of truth and must not be edited during scoped
refactors unless the user explicitly requests a product change.

## Current architecture

The main execution path is:

```text
CLI / FastAPI
  -> agent loop (perception, decision, action, summarization)
  -> code-owned Workflow Registry
  -> action.execute_step approval and tool boundary
  -> modular MCP ToolRegistry
  -> domain handlers and explicit dependencies
  -> verified result/evidence returned to the agent
```

Important locations:

- Agent orchestration: `agent/`, `action/`, `decision/`, `summarization/`
- Workflow source of truth: `workflow/registry.py`
- Compatibility MCP facade: `mcp_mlops_tools.py`
- Modular MCP package: `mcp_servers/mlops/`
- Canonical MCP contract: `tests/fixtures/mcp_tool_contract.json`
- MCP architecture: `docs/architecture/mcp_server_foundation.md`
- Reproducible baseline: `docs/BASELINE.md`
- Golden slice contract: `examples/image_classification/docs/golden_slice.md`
- Course graph summary: `course_context/emlo_graphify/GRAPH_REPORT.md`
- Repository graph summary: `graphify-out/GRAPH_REPORT.md`

The Workflow Registry currently owns 11 workflows:

- `setup_pipeline`
- `detect_training_project`
- `train_and_track`
- `build_capstone_pipeline`
- `prepare_capstone_data`
- `prepare_capstone_container_ci`
- `deploy_litserve_preflight`
- `deploy_litserve_gpu`
- `deploy_gpu_inference`
- `deploy_gradio_demo`
- `deploy_kserve_production`

## Verified golden slice

The repository has one deliberately small, real end-to-end reference path:

```text
deterministic red/blue PNG data
  -> DVC prepare_golden_data
  -> checksum manifest
  -> DVC train_golden on CPU
  -> strict golden-image-classifier.v1 checkpoint
  -> verified local MLflow run and lineage
  -> FastAPI /health and /predict
  -> CPU Docker image and smoke test
```

Properties:

- Two ordered classes: `red`, `blue`
- CPU-only deterministic training
- 64 training and 16 validation images
- Strict checkpoint, preprocessing, class, and dataset-lineage validation
- Local DVC cache only; no remote is configured
- Local SQLite MLflow backend and local artifacts, both ignored by Git
- Generated data, checkpoints, caches, databases, and run artifacts are ignored
- No external dataset, cloud account, credential, LLM API, or GPU is required

Use the detailed contract rather than duplicating implementation assumptions:
`examples/image_classification/docs/golden_slice.md`.

## MCP modularization contracts

The MCP catalog currently contains exactly 98 tools with 98 unique names. The
canonical fixture records names, ordering, descriptions, Pydantic schemas, and
handler bindings. Refactors must preserve that fixture exactly unless a future
milestone explicitly changes the product contract.

`ToolSpec` and `ToolRegistry` are the single source for catalog construction,
validation, and dispatch. Do not introduce a second dispatch table.

The compatibility facade must continue to support established root imports and
proven root monkeypatch seams while domains are extracted incrementally.
Imports and registry construction must not perform filesystem mutation,
subprocess execution, network access, backend connections, or cloud actions.

Extracted domains:

- Hydra: four tools, explicit immutable filesystem boundary
- Basic MLflow: eight tools, lazy immutable SDK/client/filesystem boundary

Remaining tools are still adapted through the root facade. After the MLflow
extraction, `mcp_mlops_tools.py` is 14,320 lines and 86 ToolSpecs remain rooted
outside the Hydra and basic MLflow domains.

Known existing inventory limitations that should not be changed accidentally:

- `TriggerGitHubWorkflowInput` and `CheckWorkflowRunInput` are orphan schemas.
- `action.execute_step.AVAILABLE_TOOLS` contains 84 of the 98 MCP catalog tools.
- Some schemas intentionally map argument names for compatibility, including
  Hydra's `ml_model_config` to handler `model_config`.

## Completed rebuild milestones

| Milestone | Pull request | Merge commit | Result |
| --- | --- | --- | --- |
| Reproducible Phase 0 baseline | #46 | `dec4513` | Locked install, startup, core tests, and documentation established |
| Golden image-classification serving slice | #47 | `da99b44` | Real CPU training, checkpoint, FastAPI, Docker, health and prediction verification |
| Golden-slice CI quality gate | #48 | `676f769` | Hermetic PR/main quality job and dependent Docker smoke job |
| Deterministic DVC dataset lineage | #49 | `74ddcc2` | File-backed deterministic dataset, manifest, DVC pipeline, and checkpoint lineage |
| Local MLflow tracking for golden training | #50 | `142909a` | SQLite tracking, verified artifacts, metrics, parameters, and DVC hashes |
| Modular MCP registry foundation | #51 | `17db043` | Canonical ToolSpec registry and Hydra extraction |
| Hydra filesystem dependency boundary | #52 | `a840b8f` | Narrow immutable filesystem adapter with compatibility preserved |
| Basic MLflow MCP extraction | #53 | `7437188` | Eight tools moved behind a lazy immutable dependency boundary |

All listed milestones were merged normally after green GitHub checks. Their
remote feature branches were deleted.

## Current verification baseline

Use Python 3.10 and the locked `uv` environment.

Core checks:

```bash
uv sync --extra dev --locked
uv lock --check
uv run python cli.py --help
uv run python -c "import api_server; import mcp_mlops_tools"
uv run ruff check .
git diff --check
git diff --exit-code -- PRD.md
```

Current focused regression counts:

- MCP MLflow extraction and registry contracts: 41 passed
- Root-migrated MLflow checks: 6 passed
- Workflow Registry: 66 passed
- Agent loop: 167 passed
- Execute-step: 41 passed
- E2E: 20 passed
- Image-classification example: 155 passed

Full integration baseline:

- 122 passed
- 4 skipped
- 5 known failures

The five failures are all unchanged `check_data_quality` tests. The locked
Great Expectations package does not expose `great_expectations.from_pandas`.
Do not attribute these failures to an unrelated refactor, hide them, weaken the
tests, or expand a scoped milestone to fix them.

Expected non-failing warnings include MLflow's filesystem-backend deprecation
warning and PyTorch's CPU `pin_memory` warning.

The GitHub workflow `.github/workflows/quality-gate.yml` runs on PRs to `main`
and pushes to `main`. It has two required stages:

1. `Phase 0 and golden slice`
2. `Golden slice Docker smoke test`

Do not merge a milestone until both are green.

## Current next task

### Phase 3: extract the seven basic DVC MCP tools

Start from synchronized `main` on:

```text
refactor/extract-basic-dvc-tools
```

Tools in scope:

- `init_dvc_repo`
- `configure_dvc_remote`
- `add_data_to_dvc`
- `create_dvc_pipeline`
- `dvc_push`
- `dvc_pull`
- `dvc_reproduce`

Expected direction:

- Move unchanged input models to `mcp_servers/mlops/schemas/dvc.py`.
- Move the seven handlers and ToolSpecs to
  `mcp_servers/mlops/domains/dvc.py`.
- Introduce the smallest lazy, frozen dependency boundary justified by actual
  DVC operations: command execution, executable availability, filesystem/path
  behavior, and environment/config access only where currently used.
- Use scoped, automatically reset test overrides.
- Preserve historical root signatures, imports, return dictionaries, error
  messages, command arguments, working directories, approval behavior, and
  root monkeypatch seams such as command/tool checks where proven by tests.
- Ensure imports and registry construction execute no DVC command and mutate no
  repository, filesystem, remote, cache, or configuration.
- Keep all 98 tool contracts byte-for-byte equivalent to the canonical fixture.
- Do not extract capstone DVC orchestration tools in this phase.
- Do not configure a real remote, access S3, push or pull actual user data, or
  modify the golden DVC pipeline.
- Add recording/failing boundary tests and direct-versus-MCP equivalence tests.
- Update `docs/architecture/mcp_server_foundation.md`.
- Leave the work uncommitted and unpublished for review unless the user asks to
  publish it.

Minimum completion validation should follow the Phase 2 pattern: locked sync
and lock check, focused DVC and registry contracts, legacy DVC checks, workflow
registry, agent loop, execute-step, E2E, image-classification tests, integration
baseline comparison, Ruff, diff check, and an empty `PRD.md` diff.

## Roadmap after Phase 3

Continue behavior-preserving MCP decomposition in small reviewable milestones:

1. Docker and GitHub Actions tools
2. Data quality and monitoring tools
3. Training and capstone-data tools
4. LitServe and other serving targets
5. Kubernetes, KServe, Helm, and AWS tools

Then advance the capstone golden path rather than adding disconnected breadth:

1. Hydra-controlled HPO and a learning-rate finder linked to MLflow evidence
2. Approval-gated DVC S3 remote configuration and real push/pull verification
3. Registry publishing with immutable image/model versions
4. FastAPI middleware and selected Kubernetes/KServe deployment
5. Helm packaging and ArgoCD GitOps rollout
6. Lambda and Hugging Face demo endpoints
7. Load tests, concurrency, autoscaling, monitoring, rollback, and final report

At every stage, require evidence: exact artifacts, hashes, metrics, commands,
endpoints, logs, cleanup, and a clear statement of what was not executed.

## Cost and machine guidance

The current Phase 3 refactor and most upcoming MCP modularization work do not
need a GPU. Use a low-cost CPU machine for editing and local unit tests; let
GitHub Actions run the existing Linux Docker smoke test.

Reserve a paid GPU instance for milestones that actually verify CUDA, LitServe
GPU execution, GPU utilization, batching, or latency. Before stopping an
instance, confirm:

```bash
git status -sb
git branch --show-current
git log -1 --oneline
```

Commit and publish only reviewed work. Never depend on ignored local data,
models, MLflow databases, DVC caches, `.env` files, or credentials surviving a
machine shutdown.

## Handoff maintenance rule

Update this document after each merged milestone. Change only facts that were
verified from the repository or GitHub. At minimum update:

- last completed product milestone commit;
- completed-milestone table;
- extracted-domain inventory and facade size;
- current test baseline;
- **Current next task**;
- any new known blocker or intentional non-goal.

Keep this file compact and operational. Link to detailed contracts instead of
turning it into a transcript or duplicating entire implementation documents.
