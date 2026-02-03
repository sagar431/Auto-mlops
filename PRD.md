# Auto-MLOps Production PRD

## Overview
Transform Auto-MLOps from a working prototype into a production-grade MLOps automation platform. The agent uses natural language to create ML pipelines (Hydra, DVC, MLflow, Docker, GitHub Actions) and supports 5 deployment targets.

**Target:** Small team (<10 users), CLI-first, PostgreSQL backend, API key auth.

## Phase 1: Security Foundation

- [x] Create `security/__init__.py` module
- [x] Create `security/models.py` with User and APIKey SQLModel classes
- [x] Create `security/api_keys.py` with APIKeyManager (generate, verify, revoke)
- [x] Create `security/middleware.py` with FastAPI `get_current_user` dependency
- [x] Update `api_server.py` to add auth middleware to all endpoints (except /health)
- [x] Add admin endpoints: POST /admin/users, POST /admin/keys, DELETE /admin/keys/{id}
- [x] Update CORS in `api_server.py` from allow_origins=["*"] to configurable list
- [x] Add rate limiting with slowapi (100 req/min default)
- [x] Add CLI admin commands: `mlops-agent admin create-user`, `create-key`, `list-users`, `revoke-key`
- [x] Write tests for auth middleware and API key validation

## Phase 2: Database Persistence

- [x] Create `db/__init__.py` module
- [x] Create `db/models.py` with Session, Step, ExperimentState SQLModel tables
- [x] Create `db/session.py` with async PostgreSQL connection (asyncpg)
- [x] Create `db/repositories.py` with SessionRepository CRUD operations
- [x] Set up Alembic migrations in `db/migrations/`
- [x] Update `api_server.py` SessionManager to use database instead of in-memory dict
- [x] Update `agent/agentSession.py` to persist to database
- [x] Update `memory/memory_search.py` to query database instead of JSON files
- [x] Add DATABASE_URL to .env.example
- [x] Write tests for database repositories

## Phase 3: Observability & Reliability

- [x] Create `observability/__init__.py` module
- [x] Create `observability/logging.py` with structlog JSON configuration
- [x] Replace all print() statements with structured logger calls in agent/*.py
- [x] Replace all print() statements in action/execute_step.py and perception/perception.py
- [x] Create `observability/metrics.py` with Prometheus metrics endpoint
- [x] Add /metrics endpoint to api_server.py
- [x] Create `resilience/__init__.py` module
- [x] Create `resilience/circuit_breaker.py` with CircuitBreaker class
- [x] Create `resilience/retry.py` with exponential backoff decorator
- [x] Update `agent/agent_loop.py` StepExecutionTracker to use circuit breaker
- [x] Implement LLM fallback chain in `agent/model_manager.py` (Gemini → OpenAI → Gemini Flash)
- [x] Write tests for circuit breaker and retry logic

## Phase 4: Data Validation

- [x] Create `data_quality/__init__.py` module
- [x] Create `data_quality/validator.py` with Great Expectations integration
- [x] Add MCP tool `validate_dataset` to mcp_mlops_tools.py
- [x] Add MCP tool `create_expectation_suite` to mcp_mlops_tools.py
- [x] Add MCP tool `check_data_quality` to mcp_mlops_tools.py
- [x] Create Pydantic input models for new data quality tools
- [x] Update decision_prompt.txt to include data validation in pipeline planning
- [x] Write tests for data validation tools

## Phase 5: Model Monitoring

- [x] Create `monitoring/__init__.py` module
- [x] Create `monitoring/drift_detector.py` with Evidently integration
- [x] Create `monitoring/model_monitor.py` for performance tracking
- [x] Add MCP tool `detect_data_drift` to mcp_mlops_tools.py
- [x] Add MCP tool `monitor_model_performance` to mcp_mlops_tools.py
- [x] Add MCP tool `setup_alerting` to mcp_mlops_tools.py
- [x] Write tests for monitoring tools

## Phase 6: Example Projects

- [x] Create `examples/image_classification/` directory structure
- [x] Write `examples/image_classification/train.py` (PyTorch + CIFAR-10 + Hydra)
- [x] Write `examples/image_classification/model.py` (ResNet18)
- [x] Write `examples/image_classification/dataset.py` (CIFAR-10 loader)
- [x] Create Hydra configs for image_classification example
- [x] Create DVC pipeline for image_classification example
- [x] Write `examples/image_classification/README.md` with setup instructions

- [x] Create `examples/text_classification/` directory structure
- [x] Write `examples/text_classification/train.py` (HuggingFace + IMDB)
- [x] Write `examples/text_classification/model.py` (DistilBERT wrapper)
- [x] Create Hydra configs for text_classification example
- [x] Write `examples/text_classification/README.md`

- [x] Create `examples/tabular_regression/` directory structure
- [x] Write `examples/tabular_regression/train.py` (sklearn + California Housing)
- [x] Create Hydra configs for tabular_regression example
- [x] Write `examples/tabular_regression/README.md`

## Phase 7: Testing

- [x] Create `tests/` directory with pytest structure
- [x] Create `tests/conftest.py` with fixtures (test_project, async_client, db_session, mock_llm)
- [x] Migrate test_mlops_tools.py tests to `tests/integration/test_mcp_tools.py`
- [x] Write `tests/unit/test_agent_loop.py`
- [x] Write `tests/unit/test_perception.py`
- [x] Write `tests/unit/test_decision.py`
- [x] Write `tests/unit/test_execute_step.py`
- [x] Write `tests/integration/test_api_endpoints.py`
- [x] Write `tests/integration/test_websocket.py`
- [x] Write `tests/e2e/test_train_deploy_flow.py` (full pipeline test)
- [x] Write `tests/security/test_auth.py`
- [x] Write `tests/security/test_rate_limiting.py`
- [x] Create `tests/load/locustfile.py` for load testing
- [ ] Achieve >80% test coverage

## Phase 8: CLI & SDK

- [x] Add `init` command to cli.py (initialize ML project structure)
- [x] Add `deploy` command to cli.py (shortcut for deployment)
- [x] Add `monitor` command to cli.py (check model performance)
- [x] Add `validate` command to cli.py (validate dataset)
- [x] Create `sdk/__init__.py` module
- [x] Create `sdk/client.py` with MLOpsClient class (sync)
- [x] Create `sdk/async_client.py` with AsyncMLOpsClient class
- [x] Write SDK usage examples in README

## Phase 9: Documentation

- [x] Create `docs/` directory
- [x] Write `docs/getting-started.md`
- [x] Write `docs/api-reference.md` (all REST endpoints)
- [x] Write `docs/deployment-targets.md` (Gradio, LitServe, Lambda, TorchServe, KServe)
- [x] Write `docs/security.md` (API keys, RBAC, rate limiting)
- [x] Write `docs/examples/basic-pipeline.md`
- [x] Write `docs/examples/custom-deployment.md`
- [x] Write README.md with production setup instructions

## Phase 10: Final Polish

- [x] Update pyproject.toml with all new dependencies
- [x] Update .env.example with all required variables
- [ ] Run black and ruff on entire codebase
- [ ] Run bandit security scan and fix any issues
- [ ] Run full test suite and ensure all pass
- [ ] Run E2E test: train CIFAR-10 → deploy to Gradio
- [ ] Create demo GIF showing full workflow
- [ ] Tag v1.0.0 release
