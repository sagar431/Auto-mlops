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
python test_mlops_tools.py --tool hydra     # Test specific category: hydra, mlflow, dvc, docker, github, training
pytest                                       # Run pytest suite
```

### Running the MCP Server
```bash
python mcp_mlops_tools.py   # Runs the MCP server with 28 MLOps tools
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
28 tools organized by category:
- **Hydra (4)**: analyze_project_config, create_hydra_config, update_hydra_config, validate_hydra_config
- **MLflow (8)**: init_mlflow_experiment, start_mlflow_run, log_mlflow_params/metrics/artifact, register_mlflow_model, get_best_mlflow_run, end_mlflow_run
- **DVC (7)**: init_dvc_repo, configure_dvc_remote, add_data_to_dvc, create_dvc_pipeline, dvc_push/pull/reproduce
- **Docker (4)**: create_ml_dockerfile, build_ml_docker_image, run_training_container, push_docker_image
- **GitHub Actions (2)**: create_github_workflow, add_workflow_step
- **Training Control (3)**: analyze_training_results, suggest_improvements, check_accuracy_threshold

Each tool uses Pydantic models for input validation (e.g., `CreateHydraConfigInput`).

### Prompt Templates (`prompts/`)
LLM prompts that drive agent behavior:
- `perception_prompt.txt` - Entity extraction, stage detection, routing logic
- `decision_prompt.txt` - Execution plan generation with tool dependencies
- `improvement_prompt.txt` - Hyperparameter adjustment suggestions
- `summarizer_prompt.txt` - Final report generation

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

## Running the Agent

```python
from agent import AgentLoop
import asyncio

async def main():
    agent = AgentLoop()
    result = await agent.run(
        query="Set up MLOps pipeline for my cat-dog classifier with accuracy threshold 0.85",
        project_path="/path/to/project",
        accuracy_threshold=0.85
    )
    print(result)

asyncio.run(main())
```

Or use the convenience function:
```python
from agent.agent_loop import run_mlops_agent

result = await run_mlops_agent(
    query="Deploy my model with DVC and GitHub Actions",
    project_path="/path/to/project"
)
```

## Environment Variables
```
OPENAI_API_KEY, GOOGLE_API_KEY         # LLM providers
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, DVC_REMOTE_URL  # DVC S3
DOCKER_REGISTRY, DOCKER_USERNAME       # Docker
DEFAULT_ACCURACY_THRESHOLD=0.85        # Training target
MAX_IMPROVEMENT_ATTEMPTS=3             # Retry limit
```

## Development Status

**Phase 2 Complete** - Core agent architecture implemented:
- Agent loop with perception-decision-action cycle
- Graph-based execution tracking with NetworkX
- Self-improvement loop for training optimization
- Memory search for learning from past experiments
