# Getting Started with Auto-MLOps

This guide will help you get up and running with Auto-MLOps, an AI-powered ML pipeline automation tool.

## Prerequisites

- Python 3.10+
- PostgreSQL (for production)
- Docker (optional, for containerized deployments)

## Installation

### Using uv (Recommended)

```bash
git clone https://github.com/your-org/auto-mlops.git
cd auto-mlops
uv sync
```

### Using pip

```bash
git clone https://github.com/your-org/auto-mlops.git
cd auto-mlops
pip install -e .
```

## Configuration

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Set your API keys:

```bash
# Required: LLM Provider (at least one)
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key  # Optional fallback

# Database (for production)
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/mlops

# Optional
DEFAULT_ACCURACY_THRESHOLD=0.85
MAX_IMPROVEMENT_ATTEMPTS=3
```

3. Initialize the database (production):

```bash
alembic upgrade head
```

## Quick Start

### CLI Mode

```bash
# Initialize a new ML project
mlops-agent init ./my-project

# Single command
mlops-agent "Set up MLOps pipeline for my cat-dog classifier"

# With options
mlops-agent --project ./my_project --threshold 0.90 "Train until accuracy target"

# Interactive mode
mlops-agent --interactive
```

### API Server

```bash
# Start the server
python api_server.py

# Or with uvicorn (production)
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Python SDK

```python
from sdk import MLOpsClient

# Create client
client = MLOpsClient(api_key="your-api-key")

# Run a pipeline
result = client.run(
    "Set up MLOps pipeline for my project",
    project_path="/path/to/project",
    accuracy_threshold=0.85
)

print(f"Status: {result.status}")
print(f"Result: {result.result}")
```

## Example Workflow

### 1. Initialize Your Project

```bash
mlops-agent init ./my-ml-project --template pytorch
cd my-ml-project
```

### 2. Set Up the Pipeline

```bash
mlops-agent "Create Hydra config with learning_rate=0.001, batch_size=32"
mlops-agent "Initialize MLflow experiment named my-experiment"
mlops-agent "Set up DVC with S3 remote"
```

### 3. Train Your Model

```bash
mlops-agent "Train until accuracy reaches 85%"
```

### 4. Deploy

```bash
mlops-agent deploy gradio --model ./models/best_model.pt
```

## Next Steps

- [API Reference](./api-reference.md) - Full API documentation
- [Deployment Targets](./deployment-targets.md) - Available deployment options
- [Security](./security.md) - Authentication and authorization
- [Examples](./examples/basic-pipeline.md) - Step-by-step tutorials
