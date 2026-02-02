# Tabular Regression Example

This example demonstrates how to use Auto-MLOps to set up, train, and deploy tabular regression models for house price prediction using natural language commands.

## Overview

This example shows the complete MLOps workflow:
1. **Setup** - Create modular Hydra configs, initialize MLflow experiment
2. **Data** - Load California Housing dataset, version control with DVC
3. **Training** - Train sklearn or PyTorch models with metric logging
4. **Evaluation** - Check accuracy against threshold (RMSE, R²)
5. **Improvement** - Auto-tune hyperparameters if needed
6. **Deployment** - Deploy to your chosen target (Gradio, LitServe, Lambda, etc.)

## Quick Start

### Prerequisites

```bash
# Install Auto-MLOps from the project root
cd /path/to/Auto-mlops
uv sync  # or pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (GOOGLE_API_KEY or OPENAI_API_KEY)
```

### Install Project Dependencies

```bash
# For sklearn models (root-level train.py)
pip install scikit-learn hydra-core omegaconf numpy

# For PyTorch models (project/train.py)
cd examples/tabular_regression/project
pip install -r requirements.txt
```

### Run Training

#### Sklearn Models (Recommended for quick experiments)

```bash
cd examples/tabular_regression

# Default training (Gradient Boosting)
python train.py

# Use Ridge regression
python train.py model=ridge

# Use Random Forest
python train.py model=random_forest

# Use experiment preset for higher accuracy
python train.py +experiment=high_accuracy
```

#### PyTorch Models (Deep learning)

```bash
cd examples/tabular_regression/project

# Default training (MLP model)
python train.py

# Use TabNet (attention-based)
python train.py model=tabnet

# Quick test with synthetic data
python train.py +experiment=quick_test
```

### Run with DVC Pipeline

```bash
cd examples/tabular_regression/project

# Initialize DVC (if not already done)
dvc init

# Run the full pipeline (prepare_data -> train -> evaluate)
dvc repro

# Run specific stages
dvc repro prepare_data
dvc repro train
dvc repro evaluate
```

### Run with Auto-MLOps Agent

```bash
# Single command mode
mlops-agent "Set up MLOps pipeline for the tabular regression project in examples/tabular_regression/project"

# Interactive mode
mlops-agent -i --project examples/tabular_regression/project
```

## Project Structure

```
examples/tabular_regression/
├── README.md                    # This file
├── train.py                     # Sklearn training script with Hydra
├── configs/                     # Sklearn Hydra configs
│   ├── config.yaml              # Main config
│   ├── model/                   # Model configs
│   │   ├── ridge.yaml           # Ridge regression
│   │   ├── random_forest.yaml   # Random Forest
│   │   └── gradient_boosting.yaml  # Gradient Boosting (default)
│   ├── training/                # Training configs
│   │   └── default.yaml         # Default settings
│   ├── data/                    # Data configs
│   │   └── california.yaml      # California Housing dataset
│   ├── paths/                   # Path configs
│   │   └── default.yaml         # Output paths
│   └── experiment/              # Experiment presets
│       ├── baseline.yaml        # Gradient Boosting baseline
│       ├── ridge_baseline.yaml  # Ridge baseline
│       ├── random_forest_baseline.yaml  # Random Forest baseline
│       └── high_accuracy.yaml   # Tuned high accuracy
├── project/                     # PyTorch ML project directory
│   ├── train.py                 # Training script with Hydra
│   ├── evaluate.py              # Evaluation script
│   ├── prepare_data.py          # Data preparation
│   ├── inference.py             # Inference utilities
│   ├── model.py                 # Model architectures (MLP, TabNet)
│   ├── dataset.py               # Dataset utilities
│   ├── requirements.txt         # Dependencies
│   ├── dvc.yaml                 # DVC pipeline
│   └── configs/                 # Hydra configuration
│       ├── config.yaml          # Main config
│       ├── model/               # Model configs (mlp, tabnet)
│       ├── training/            # Training configs
│       ├── data/                # Data configs
│       ├── paths/               # Path configs
│       └── experiment/          # Experiment presets
├── tests/                       # Test suite
│   ├── test_sklearn_training.py # Sklearn training tests
│   ├── test_sklearn_hydra_configs.py  # Sklearn config tests
│   ├── test_training.py         # PyTorch training tests
│   ├── test_model.py            # Model tests
│   ├── test_dataset.py          # Dataset tests
│   ├── test_inference.py        # Inference tests
│   ├── test_hydra_configs.py    # Config validation tests
│   └── test_dvc_pipeline.py     # DVC pipeline tests
├── agent_queries.md             # Sample queries for the agent
└── docs/
    └── walkthrough.md           # Detailed walkthrough
```

## Dataset

This example uses **California Housing**, a standard benchmark for regression:

- 20,640 samples with 8 features
- Target: Median house value ($100,000s)
- Features include location, demographics, and housing characteristics

### Features

| Feature | Description |
|---------|-------------|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |

## Model Architectures

### Sklearn Models (train.py)

| Model | Description | Expected Performance |
|-------|-------------|---------------------|
| **Gradient Boosting** (default) | Ensemble of decision trees with boosting | RMSE ~0.52, R² ~0.80 |
| **Random Forest** | Ensemble of decision trees with bagging | RMSE ~0.50, R² ~0.82 |
| **Ridge** | Linear regression with L2 regularization | RMSE ~0.73, R² ~0.59 |

### PyTorch Models (project/train.py)

| Model | Description | Expected Performance |
|-------|-------------|---------------------|
| **MLP** | Multi-layer perceptron with batch normalization | RMSE ~0.50-0.55, R² ~0.77 |
| **TabNet** | Attention-based tabular learning architecture | RMSE ~0.45-0.50, R² ~0.82 |

## Hydra Configuration

The project uses modular Hydra configs for flexible experiment management.

### Sklearn Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `model` | `gradient_boosting`, `random_forest`, `ridge` | Model type |
| `training` | `default` | Training settings (test_size, normalization) |
| `data` | `california` | Dataset configuration |
| `paths` | `default` | Output paths |

### PyTorch Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `model` | `mlp`, `tabnet` | Model architecture |
| `training` | `default`, `fast`, `long` | Training hyperparameters |
| `data` | `california`, `synthetic` | Dataset configuration |
| `paths` | `default` | Output paths |

### Experiment Presets

#### Sklearn

| Experiment | Description | Command |
|------------|-------------|---------|
| `baseline` | Gradient Boosting (default settings) | `python train.py +experiment=baseline` |
| `ridge_baseline` | Ridge regression baseline | `python train.py +experiment=ridge_baseline` |
| `random_forest_baseline` | Random Forest baseline | `python train.py +experiment=random_forest_baseline` |
| `high_accuracy` | Tuned Gradient Boosting (200 trees) | `python train.py +experiment=high_accuracy` |

#### PyTorch

| Experiment | Description | Command |
|------------|-------------|---------|
| `baseline` | MLP standard training | `python train.py +experiment=baseline` |
| `quick_test` | Fast test with synthetic data | `python train.py +experiment=quick_test` |
| `tabnet_baseline` | TabNet architecture | `python train.py +experiment=tabnet_baseline` |
| `high_accuracy` | Extended training | `python train.py +experiment=high_accuracy` |

### Override Examples

```bash
# Sklearn: Change model parameters
python train.py model.n_estimators=200
python train.py model.max_depth=5
python train.py model=ridge model.alpha=0.5

# PyTorch: Change training parameters
python train.py training.epochs=50 training.batch_size=64
python train.py training.learning_rate=0.0001
python train.py model=tabnet model.n_steps=5
```

## DVC Pipeline

The DVC pipeline automates the ML workflow:

```
prepare_data -> train -> evaluate
```

### Stages

1. **prepare_data**: Download and preprocess California Housing dataset
2. **train**: Train the model using Hydra config
3. **evaluate**: Compute regression metrics (RMSE, MAE, R²)

### Commands

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train

# Force re-run
dvc repro -f

# View pipeline DAG
dvc dag
```

## Running Tests

```bash
# From the tabular_regression directory
cd examples/tabular_regression

# Run all tests
pytest tests/ -v

# Run sklearn training tests
pytest tests/test_sklearn_training.py -v

# Run PyTorch tests
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ -v --cov=.
```

## Sample Agent Queries

Here are example queries you can use with the Auto-MLOps agent:

### Setup Pipeline
```
"Set up an MLOps pipeline for California Housing regression with RMSE threshold 0.5"
```

### Train Model
```
"Train a Gradient Boosting model and track metrics with MLflow"
```

### Deploy Model
```
"Deploy my regression model to Gradio for a quick demo"
"Deploy the model to AWS Lambda for serverless inference"
"Create a high-throughput API with LitServe"
```

See `agent_queries.md` for more examples.

## Deployment Options

After training, deploy to any of these targets:

| Target | Command | Use Case |
|--------|---------|----------|
| Gradio | `"Deploy to Gradio"` | Quick demo, prototyping |
| LitServe | `"Deploy with LitServe"` | High-throughput API |
| Lambda | `"Deploy to AWS Lambda"` | Serverless, pay-per-use |
| TorchServe | `"Deploy with TorchServe"` | Enterprise production |
| KServe | `"Deploy to KServe"` | Kubernetes-native |

## Metrics

The evaluation produces these metrics:

| Metric | Description | Better |
|--------|-------------|--------|
| **RMSE** | Root Mean Square Error | Lower |
| **MAE** | Mean Absolute Error | Lower |
| **R²** | Coefficient of Determination | Higher (max 1.0) |
| **MAPE** | Mean Absolute Percentage Error | Lower |

## Troubleshooting

### Data Loading Issues

If California Housing download fails:
```bash
# Manually test dataset loading
python -c "from sklearn.datasets import fetch_california_housing; fetch_california_housing()"
```

### CUDA Out of Memory (PyTorch)

Reduce batch size:
```bash
python train.py training.batch_size=16
```

### Hydra Config Errors

Validate configs:
```bash
# Sklearn
python train.py --cfg job

# PyTorch
python project/train.py --cfg job
```

### Missing Dependencies

```bash
# Sklearn dependencies
pip install scikit-learn hydra-core omegaconf

# PyTorch dependencies
pip install torch hydra-core omegaconf
```

## License

MIT License - See the main project LICENSE file.
