# Tabular Regression Example

This example demonstrates how to use Auto-MLOps to build a complete tabular regression pipeline using PyTorch neural networks and gradient boosting models on the California Housing dataset.

## Overview

**Task**: Predict median house values based on various features (location, demographics, housing characteristics)

**Dataset**: California Housing (20,640 samples, 8 features)

**Models**:
- **MLP** (Multi-Layer Perceptron): Deep neural network with configurable layers
- **TabNet**: Attention-based tabular learning architecture
- **XGBoostRegressor**: Gradient boosting (optional, requires xgboost)

**Expected Performance**:
- MLP: RMSE ~0.50-0.55
- TabNet: RMSE ~0.45-0.50
- XGBoost: RMSE ~0.45-0.50

## Quick Start

### 1. Install Dependencies

```bash
cd examples/tabular_regression/project
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python prepare_data.py --data-dir data --dataset california
```

### 3. Train Model

```bash
# Default (MLP model)
python train.py

# Use specific model
python train.py model=tabnet

# Quick test with synthetic data
python train.py +experiment=quick_test
```

### 4. Evaluate Model

```bash
python evaluate.py --model-path models/best_model.pt --data-dir data
```

### 5. Run Inference

```bash
python inference.py --model-path models/best_model.pt
```

## Project Structure

```
tabular_regression/
├── README.md                    # This file
├── agent_queries.md             # Sample Auto-MLOps queries
├── docs/
│   └── walkthrough.md           # Detailed tutorial
├── project/
│   ├── train.py                 # Training script (Hydra)
│   ├── evaluate.py              # Evaluation script
│   ├── prepare_data.py          # Data preparation
│   ├── inference.py             # Inference utilities
│   ├── model.py                 # Model architectures
│   ├── dataset.py               # Dataset utilities
│   ├── dvc.yaml                 # DVC pipeline
│   ├── requirements.txt         # Dependencies
│   └── configs/                 # Hydra configs
│       ├── config.yaml          # Main config
│       ├── model/               # Model configs
│       ├── training/            # Training configs
│       ├── data/                # Data configs
│       ├── paths/               # Path configs
│       └── experiment/          # Experiment presets
└── tests/                       # Test suite
    ├── conftest.py
    ├── test_model.py
    ├── test_dataset.py
    ├── test_training.py
    ├── test_inference.py
    ├── test_hydra_configs.py
    └── test_dvc_pipeline.py
```

## Using with Auto-MLOps Agent

```bash
# From project root
mlops-agent "Set up MLOps pipeline for the tabular regression project in examples/tabular_regression/project"

# Or interactive mode
mlops-agent -i --project examples/tabular_regression/project
```

See `agent_queries.md` for more example queries.

## Configuration

### Model Selection

```bash
# MLP (default)
python train.py model=mlp

# TabNet
python train.py model=tabnet
```

### Training Overrides

```bash
# Change epochs and batch size
python train.py training.epochs=50 training.batch_size=64

# Use different learning rate
python train.py training.learning_rate=0.0001
```

### Experiment Presets

```bash
# Quick test (synthetic data, few epochs)
python train.py +experiment=quick_test

# Baseline experiment
python train.py +experiment=baseline

# High accuracy (more epochs, lower LR)
python train.py +experiment=high_accuracy
```

## DVC Pipeline

Run the full pipeline:

```bash
dvc repro
```

Pipeline stages:
1. **prepare_data**: Download and preprocess California Housing dataset
2. **train**: Train the model with Hydra config
3. **evaluate**: Compute regression metrics (RMSE, MAE, R²)

## Metrics

The evaluation produces these metrics:
- **RMSE** (Root Mean Square Error): Lower is better
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **R²** (Coefficient of Determination): Higher is better (max 1.0)
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Deployment

After training, deploy your model:

```bash
# Deploy to LitServe (high throughput)
mlops-agent "Deploy tabular regression model to LitServe"

# Deploy to FastAPI + Lambda (serverless)
mlops-agent "Deploy to AWS Lambda"

# Create Gradio demo
mlops-agent "Create Gradio demo for the regression model"
```

## Feature Engineering

The California Housing dataset includes:
- `MedInc`: Median income in block group
- `HouseAge`: Median house age in block group
- `AveRooms`: Average number of rooms per household
- `AveBedrms`: Average number of bedrooms per household
- `Population`: Block group population
- `AveOccup`: Average number of household members
- `Latitude`: Block group latitude
- `Longitude`: Block group longitude

Target: `MedHouseVal` (median house value in $100,000s)

## License

This example is part of Auto-MLOps and is licensed under the same terms.
