# Phase 0 Baseline

Observed on 2026-07-15 in `/home/ubuntu/Auto-mlops`.

## Environment

- Operating environment: Linux
- Python: CPython 3.10.12 (`/usr/bin/python3`)
- uv: 0.11.28
- Project environment: `.venv`
- Installed project: `mlops-agent==1.0.0` from the repository working tree

## Installation

Install the locked runtime and development dependencies:

```bash
uv sync --extra dev --locked
```

The repository lockfile was stale at the start of verification: the first locked sync exited 1 because `uv.lock` did not contain all dependencies declared by `pyproject.toml`. Running `uv lock` refreshed the lockfile. A subsequent `uv sync --extra dev --locked` completed successfully and installed 247 packages.

## Working Commands

```bash
# CLI help
uv run python cli.py --help

# Core import smoke test
uv run python -c "import api_server; import mcp_mlops_tools"

# API server
uv run python api_server.py
# or
uv run uvicorn api_server:app --reload --port 8000

# MCP server
uv run python mcp_mlops_tools.py

# Focused and end-to-end tests
uv run pytest tests/unit/test_workflow_registry.py
uv run pytest tests/unit/test_agent_loop.py
uv run pytest tests/unit/test_execute_step.py
uv run pytest tests/e2e

# Lint
uv run ruff check .
```

The API startup command was also started through application startup and then stopped cleanly. The MCP server module passed the import smoke test; this baseline did not start that long-running server.

## Commands Executed and Results

| Command | Result | Observed detail |
| --- | --- | --- |
| `python --version` | PASS | Python 3.10.12 |
| `uv --version` | PASS | uv 0.11.28 after installing uv with `python -m pip install --user uv` |
| `uv sync --extra dev --locked` (initial) | FAIL | Lockfile required an update; no environment baseline was claimed from this attempt |
| `uv lock` | PASS | Refreshed `uv.lock` from `pyproject.toml`; warned that one non-selected resolution candidate, `numpy==2.4.0`, is yanked |
| `uv sync --extra dev --locked` (after lock refresh) | PASS | Installed the locked project and development environment |
| `uv sync --extra dev --locked` (final) | PASS | Rebuilt and reinstalled the local editable package without changing the lockfile |
| `uv run python cli.py --help` | PASS | Printed CLI commands and options, exit 0 |
| `uv run python -c "import api_server; import mcp_mlops_tools"` | PASS | Both core modules imported, exit 0 |
| `uv run python api_server.py` | PASS | Uvicorn started on port 8000, initialized the database, completed application startup, and shut down cleanly on interrupt |
| `uv run pytest tests/unit/test_workflow_registry.py` | PASS | 66 passed |
| `uv run pytest tests/unit/test_agent_loop.py` (initial) | FAIL | 166 passed, 1 failed because the test assumed the interpreter basename ended in `python`; uv used `python3` |
| `uv run pytest tests/unit/test_agent_loop.py` (after repair) | PASS | 167 passed, 3 MLflow future warnings |
| `uv run pytest tests/unit/test_execute_step.py` | PASS | 41 passed |
| `uv run pytest tests/e2e` (initial) | FAIL | 19 passed, 1 failed because a manifest-only test implicitly assumed DVC was not installed |
| `uv run pytest tests/e2e` (after repair) | PASS | 20 passed, 1 MLflow future warning |
| `uv run ruff check .` (initial) | PASS | All checks passed; Ruff reported that top-level `select` and `ignore` settings were deprecated |
| `uv run ruff check .` (after repair) | PASS | All checks passed without the configuration warning |

The Ruff configuration warning was repaired by moving `select` and `ignore` under `[tool.ruff.lint]`. The MLflow warnings report that its filesystem tracking backend is planned for deprecation. They do not fail the current baseline.

## Registry-Owned Workflows

The code-owned Workflow Registry currently exposes:

- `setup_pipeline` — Setup Pipeline
- `detect_training_project` — Detect Training Project
- `train_and_track` — Train And Track
- `build_capstone_pipeline` — Capstone Orchestrator
- `prepare_capstone_data` — Prepare Capstone Data
- `prepare_capstone_container_ci` — Prepare Capstone Container CI
- `deploy_litserve_preflight` — Deploy LitServe Preflight
- `deploy_litserve_gpu` — Deploy LitServe GPU
- `deploy_gpu_inference` — Deploy GPU Inference
- `deploy_gradio_demo` — Deploy Gradio Demo
- `deploy_kserve_production` — Deploy KServe Production

This list comes from `workflow.registry.get_workflow_registry()` rather than older tool-count or phase-status prose.
