# Image Classification Example Walkthrough

This document provides a detailed walkthrough of using Auto-MLOps for an image classification task.

## Prerequisites

1. Auto-MLOps installed (`uv sync` or `pip install -e .` from project root)
2. API key configured (GOOGLE_API_KEY or OPENAI_API_KEY in .env)
3. Python 3.10+

## Step 1: Setup

First, run the setup script to create the project structure and synthetic data:

```bash
cd examples/image_classification
chmod +x setup_example.sh
./setup_example.sh
```

This creates:
- `project/data/train/{cat,dog}/` with synthetic images
- `project/models/` for saved checkpoints
- `project/configs/` for Hydra configs
- `project/logs/` for training logs

## Step 2: Start the Agent

Launch the Auto-MLOps agent in interactive mode:

```bash
mlops-agent -i --project ./project
```

Or use single-command mode:

```bash
mlops-agent --project ./project "Set up MLOps pipeline for cat-dog classification"
```

## Step 3: Pipeline Setup

The agent will:

1. **Analyze the project** - Detect training script, data structure, model architecture
2. **Create Hydra configs** - Generate configuration files for hyperparameters
3. **Initialize MLflow** - Set up experiment tracking
4. **Initialize DVC** - (Optional) Set up data version control

Example agent response:
```
✓ Analyzed project structure
✓ Created Hydra config at project/configs/config.yaml
✓ Initialized MLflow experiment 'image_classifier'
✓ Ready for training
```

## Step 4: Training

Ask the agent to train the model:

```
> Train the model until we reach 85% accuracy
```

The agent will:
1. Start an MLflow run
2. Execute the training script
3. Log metrics (loss, accuracy) at each epoch
4. Save model checkpoints
5. Check accuracy against threshold

## Step 5: Improvement Loop

If accuracy is below threshold, the agent automatically:

1. Analyzes training results
2. Suggests hyperparameter changes (learning rate, batch size, etc.)
3. Updates Hydra config
4. Retrains with new parameters
5. Repeats until threshold is met or max attempts reached

Example improvement cycle:
```
Attempt 1: accuracy = 0.72
  → Suggested: decrease learning_rate to 0.0001
  → Updated config
  → Retraining...

Attempt 2: accuracy = 0.81
  → Suggested: increase dropout to 0.6
  → Updated config
  → Retraining...

Attempt 3: accuracy = 0.87 ✓
  → Threshold met!
```

## Step 6: Deployment

Once training is complete, deploy the model:

### Option A: Gradio Demo
```
> Deploy to Gradio for a quick demo
```

Creates a web interface for testing the classifier.

### Option B: LitServe API
```
> Create a high-throughput API with LitServe
```

Creates a production-ready API with batching support.

### Option C: AWS Lambda
```
> Deploy to AWS Lambda
```

Creates serverless deployment with CDK infrastructure.

### Option D: TorchServe
```
> Deploy with TorchServe
```

Creates enterprise-grade model server with versioning.

### Option E: KServe
```
> Deploy to KServe
```

Creates Kubernetes-native inference service.

## Understanding the Agent Loop

The agent follows this cycle:

```
         ┌─────────────┐
         │  PERCEPTION │ ← Analyze user query
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  DECISION   │ ← Plan execution steps
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │   ACTION    │ ← Execute MCP tools
         └──────┬──────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼───┐             ┌─────▼─────┐
│IMPROVE│ ←──────────→│  DEPLOY   │
└───┬───┘             └─────┬─────┘
    │                       │
    └───────────┬───────────┘
                │
         ┌──────▼──────┐
         │ SUMMARIZE   │ ← Generate report
         └─────────────┘
```

## MCP Tools Used

| Tool | Purpose |
|------|---------|
| `analyze_project_config` | Understand project structure |
| `create_hydra_config` | Generate configuration files |
| `init_mlflow_experiment` | Set up experiment tracking |
| `start_mlflow_run` | Begin training run |
| `log_mlflow_metrics` | Record training metrics |
| `analyze_training_results` | Evaluate model performance |
| `suggest_improvements` | Recommend hyperparameter changes |
| `create_gradio_interface` | Generate demo UI |
| `create_litserve_api` | Generate API server |

## Tips

1. **Use synthetic data first** - The `--synthetic` flag creates demo data for testing
2. **Set realistic thresholds** - Start with 0.75 for quick testing
3. **Check MLflow UI** - Run `mlflow ui` to see experiment history
4. **Use interactive mode** - Easier to iterate and adjust queries

## Troubleshooting

### "No training data found"
Run the setup script or add images to `project/data/train/{class_name}/`

### "Model not found"
Train the model first before attempting deployment

### "API key not configured"
Set GOOGLE_API_KEY or OPENAI_API_KEY in your .env file
