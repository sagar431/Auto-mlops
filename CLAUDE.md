# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auto-MLOps is an AI-powered agent that automates ML pipeline setup using natural language. It uses a Perception → Decision → Action → Summarization loop with self-improvement capability when accuracy thresholds aren't met.

## Commands

### Installation
```bash
uv sync                    # Recommended: install with uv
pip install -e .           # Alternative: install with pip
cp .env.example .env       # Set up environment variables
```

### Testing
```bash
python test_mlops_tools.py                  # Run all MCP tool tests
python test_mlops_tools.py --tool hydra     # Test specific category: hydra, mlflow, dvc, docker, github, training, deployment
pytest                                       # Run pytest suite
```

### Running the MCP Server
```bash
python mcp_mlops_tools.py   # Runs the MCP server with 39 MLOps tools (28 core + 11 deployment)
```

### Linting
```bash
black .                     # Format code (line-length=100)
ruff check .                # Lint with ruff
```

## Architecture

### Agent Loop (`agent/agent_loop.py`)
The core orchestration follows this cycle:
1. **Perception** - Analyzes user query, extracts entities (project_path, accuracy_threshold), determines pipeline stage (setup/config/data/training/evaluation/improvement/deploy), routes to decision or summarize
2. **Decision** - Generates a DAG-based execution plan with tool calls and dependencies
3. **Action** - Executes MCP tools (Hydra, MLflow, DVC, Docker, GitHub Actions)
4. **Improvement Loop** - If accuracy < threshold and attempts remain, adjusts Hydra config and retrains
5. **Summarization** - Generates final report

### Context Manager (`agent/contextManager.py`)
Uses NetworkX DiGraph to track execution state:
- `MLOpsStepNode` - Individual execution step with status (pending/completed/failed)
- `ExperimentState` - Tracks accuracy, metrics, improvement attempts, pipeline stage
- Graph nodes represent steps with edges for dependencies

### MCP Tools (`mcp_mlops_tools.py`)
39 tools organized by category:
- **Hydra (4)**: analyze_project_config, create_hydra_config, update_hydra_config, validate_hydra_config
- **MLflow (8)**: init_mlflow_experiment, start_mlflow_run, log_mlflow_params/metrics/artifact, register_mlflow_model, get_best_mlflow_run, end_mlflow_run
- **DVC (7)**: init_dvc_repo, configure_dvc_remote, add_data_to_dvc, create_dvc_pipeline, dvc_push/pull/reproduce
- **Docker (4)**: create_ml_dockerfile, build_ml_docker_image, run_training_container, push_docker_image
- **GitHub Actions (2)**: create_github_workflow, add_workflow_step
- **Training Control (3)**: analyze_training_results, suggest_improvements, check_accuracy_threshold
- **Deployment (11)**: See Deployment Tools section below

Each tool uses Pydantic models for input validation (e.g., `CreateHydraConfigInput`).

### Deployment Tools (Phase 4)
11 new tools for multi-target deployment:

| Target | Tools | Use Case |
|--------|-------|----------|
| **LitServe** | create_litserve_api, configure_litserver | High-throughput inference, batching, GPU autoscaling |
| **Gradio** | create_gradio_interface, deploy_to_huggingface | Quick demos, prototypes, HF Spaces |
| **FastAPI+Lambda** | create_fastapi_app, create_lambda_dockerfile, generate_cdk_stack | Serverless, pay-per-use, AWS deployment |
| **TorchServe** | create_torchserve_handler, create_mar_archive, generate_torchserve_config | Enterprise production, model versioning |
| **KServe** | create_inference_service_yaml, generate_kserve_config | Kubernetes-native, auto-scaling |

### Deployment Templates (`templates/deployment/`)
```
templates/deployment/
├── litserve/          # LitAPI server templates
├── gradio/            # Gradio interface templates
├── fastapi_lambda/    # FastAPI + Lambda + CDK templates
├── torchserve/        # Handler + MAR packaging templates
└── kserve/            # InferenceService YAML templates
```

### Prompt Templates (`prompts/`)
LLM prompts that drive agent behavior:
- `perception_prompt.txt` - Entity extraction, stage detection, routing logic
- `decision_prompt.txt` - Execution plan generation with tool dependencies
- `improvement_prompt.txt` - Hyperparameter adjustment suggestions
- `summarizer_prompt.txt` - Final report generation
- `deployment_selector_prompt.txt` - Deployment target recommendation based on model/infra requirements

### Model Manager (`agent/model_manager.py`)
Handles LLM provider abstraction (OpenAI, Google). Use `get_model_manager()` to get singleton instance.
- `generate_text(prompt)` - Returns plain text response
- `generate_json(prompt)` - Returns parsed JSON dict

### Decision Module (`decision/decision.py`)
Generates graph-based execution plans:
- `Decision(prompt_path, tools_module)` - Initialize with decision prompt
- `build_decision_input(ctx, query, perception)` - Build input for decision
- Returns plan_graph with nodes (id, description, tool, args, depends_on)

### Action Module (`action/execute_step.py`)
Executes MCP tool calls:
- `execute_step(step_id, tool, args, ctx, tools_module)` - Returns `(success: bool, result: dict)`
- Automatically injects project_path from context

### Summarizer (`summarization/summarizer.py`)
Generates final reports:
- `Summarizer(prompt_path)` - Initialize with summarizer prompt
- `summarize(query, ctx, perception, session)` - Generate markdown summary

### Memory Search (`memory/memory_search.py`)
Searches past experiment sessions:
- `MemorySearch(base_dir)` - Load sessions from memory/session_logs
- `search_memory(query, top_k, threshold)` - Fuzzy search past sessions
- `search_by_metric(metric_name, min_value)` - Find best experiments

## Key Patterns

### Tool Execution Flow
Tools return `Dict[str, Any]` with `success: bool` field. On success, results are tracked in `ExperimentState` (artifacts, run_id, metrics).

### Self-Improvement Loop
When `accuracy < target_accuracy` and `improvement_attempt < max_improvement_attempts`:
1. Call `suggest_improvements` with current metrics/config
2. Apply config changes via `update_hydra_config`
3. Record attempt in `ExperimentState.improvement_history`
4. Repeat training

### Routing Logic
Perception routes to:
- `decision` - Need to plan/execute steps
- `summarize` - Goal achieved or accuracy threshold met
- `improve` - Training complete but below threshold
- `deploy` - Model ready, user requests deployment

### Deployment Flow
When `pipeline_stage == "deploy"`:
1. Perception detects deployment intent and target (litserve/gradio/lambda/torchserve/kserve)
2. Decision generates tool chain for chosen target
3. Action executes deployment tools (create templates, configs, Dockerfiles)
4. Summarization provides deployment instructions and next steps

## Running the Agent

### CLI Mode
```bash
# Single command
python cli.py "Set up MLOps pipeline for my cat-dog classifier"
python cli.py --project ./my_project --threshold 0.90 "Train until accuracy target"

# Interactive REPL
python cli.py --interactive
python cli.py -i --project ./my_project

# Using installed command
mlops-agent "Set up MLOps pipeline"
mlops-agent -i --project ./my_project
```

### API Server
```bash
# Start server
python api_server.py
# Or with uvicorn
uvicorn api_server:app --reload --port 8000

# Endpoints:
# POST /run           - Start agent with query
# GET  /status/{id}   - Get session status
# GET  /sessions      - List past sessions
# GET  /tools         - List available tools
# GET  /health        - Health check
# WS   /ws/{id}       - WebSocket for real-time events
```

### Python API
```python
from agent import AgentLoop
import asyncio

async def main():
    agent = AgentLoop()
    result = await agent.run(
        query="Set up MLOps pipeline for my cat-dog classifier",
        project_path="/path/to/project",
        accuracy_threshold=0.85
    )
    print(result)

asyncio.run(main())
```

## Environment Variables
```
GOOGLE_API_KEY                         # Google Gemini (default LLM)
OPENAI_API_KEY                         # OpenAI GPT-4 (alternative)
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DVC_REMOTE_URL  # DVC S3
DOCKER_REGISTRY, DOCKER_USERNAME       # Docker
DEFAULT_ACCURACY_THRESHOLD=0.85        # Training target
MAX_IMPROVEMENT_ATTEMPTS=3             # Retry limit

# Deployment (Phase 4)
HF_TOKEN                               # Hugging Face token for Spaces deployment
AWS_ACCOUNT_ID, AWS_REGION             # AWS Lambda/CDK deployment
```

## Development Status

**Phase 3 Complete** - Full pipeline integration:
- CLI with interactive REPL and single command mode
- FastAPI server with REST and WebSocket endpoints
- Rich console output with progress tracking
- Real-time event streaming for UI integration

**Phase 4 Complete** - Multi-Deployment Target Selection:
- [x] Templates: LitServe, Gradio, FastAPI+Lambda, TorchServe, KServe
- [x] 11 new deployment MCP tools
- [x] deployment_selector_prompt.txt for target recommendation
- [x] Perception updates for deployment intent detection
- [x] Decision updates with deployment tool chains
- [x] Agent loop updated with DEPLOY route and _run_deployment_loop()
- [x] All 11 deployment tools tested and verified working
- [ ] End-to-end testing: train → deploy flow

**Frontend & Monitoring**:
- React frontend with Dashboard component
- Landing page with feature overview
- Real-time pipeline monitoring via WebSocket
- Demo GIF and YouTube tutorial in README

### Deployment Targets

| Target | Best For | Key Features |
|--------|----------|--------------|
| **LitServe** | High-throughput (1000+ req/sec) | Batching, GPU autoscaling, streaming |
| **Gradio** | Demos & prototypes | Simple UI, instant sharing, HF Spaces |
| **FastAPI+Lambda** | Serverless, variable traffic | Pay-per-use, auto-scaling, CPU-only |
| **TorchServe** | Enterprise production | Model versioning, .mar packaging, hot-swap |
| **KServe** | Kubernetes-native | InferenceService, canary deployments |

### Example Deployment Flow
```
User: "Deploy my cat-dog classifier to AWS Lambda"

Perception → {pipeline_stage: "deploy", deployment_target: "fastapi_lambda"}
Decision  → [create_fastapi_app, create_lambda_dockerfile, generate_cdk_stack]
Action    → Creates app.py, Dockerfile, cdk.py
Summary   → "Run `cdk deploy` to deploy to AWS Lambda"
```
