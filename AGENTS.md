# Repository Guidelines

## Project Structure & Module Organization
- Core agent loop and orchestration live in `agent/`, with execution modules in `action/`, `decision/`, and `summarization/`.
- Platform features are split into domain packages such as `memory/`, `metrics/`, `monitoring/`, `observability/`, `resilience/`, `security/`, and `data_quality/`.
- API entry points are `cli.py`, `api_server.py`, and `mcp_mlops_tools.py`; database setup is in `db/` with `alembic.ini` migrations.
- Prompts and deployment assets live in `prompts/` and `templates/`, while `frontend/` holds UI work.
- Tests live in `tests/`; legacy root-level tests were moved to `tests/root_migrated/`.

## Build, Test, and Development Commands
- `uv sync` installs dependencies with uv; `pip install -e .` is the editable pip alternative.
- `cp .env.example .env` creates your local environment file; update API keys before running.
- `python cli.py --interactive` or `mlops-agent --interactive` starts the REPL; `python api_server.py` runs the API server locally.
- `uvicorn api_server:app --reload --port 8000` runs the server with hot reload.
- `python mcp_mlops_tools.py` starts the MCP tools server.
- `pytest` or `pytest tests/` runs the test suite; `python -m tests.root_migrated.test_mlops_tools --tool hydra` runs tool-category checks.
- `alembic upgrade head` applies database migrations when using Postgres.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, and `black` formatting with line length 100.
- Linting uses `ruff` (`ruff check .`) with `E`, `F`, `I`, and `UP` rules.
- Use `snake_case` for modules and functions, `PascalCase` for classes, and `test_*.py` for tests.

## Testing Guidelines
- Pytest is configured in `pytest.ini` with markers `asyncio`, `slow`, and `integration`.
- Example: `pytest -m "not slow"` skips slow tests.

## Commit & Pull Request Guidelines
- Commit messages follow Conventional Commits seen in history, e.g., `feat: ...`, `fix: ...`, `test: ...`.
- PRs should include a concise summary, the commands you ran (for example `pytest`), and screenshots for `frontend/` changes.

## Security & Configuration Tips
- Do not commit real secrets; keep them in `.env` and mirror keys from `.env.example`.
- When enabling Postgres, set `DATABASE_URL` to your local instance.

## Agent-Specific Instructions
- See `CLAUDE.md` for agent workflow and tool usage details.

## graphify

This project has a graphify knowledge graph at graphify-out/.
It also has compact EMLO course context at `course_context/emlo_graphify/`; use that instead of the original PDFs when course context is needed.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- Before answering course-alignment or product-roadmap questions, read course_context/emlo_graphify/GRAPH_REPORT.md
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
