# Image Classification Example

This example demonstrates how to use Auto-MLOps to set up, train, and deploy a CIFAR-10 image classification model using natural language commands.

## Overview

This example shows the complete MLOps workflow:
1. **Setup** - Create modular Hydra configs, initialize MLflow experiment
2. **Data** - Download CIFAR-10 and version control with DVC
3. **Training** - Train a CNN or ResNet18 model with metric logging
4. **Evaluation** - Check accuracy against threshold
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
cd examples/image_classification/project
pip install -r requirements.txt
```

### Run Training

```bash
# Default training (CIFAR-10 with CNN, 10 epochs)
cd project
python train.py

# Quick test (2 epochs, minimal data)
python train.py +experiment=quick_test

# High accuracy training (50 epochs)
python train.py +experiment=high_accuracy

# Use ResNet18 model
python train.py +experiment=resnet_baseline
```

### Run with DVC Pipeline

```bash
cd project

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
mlops-agent "Set up MLOps pipeline for CIFAR-10 classifier in examples/image_classification/project"

# Interactive mode
mlops-agent -i --project ./project
```

## Project Structure

```
examples/image_classification/
├── README.md                    # This file
├── project/                     # ML project directory
│   ├── train.py                # Training script with Hydra
│   ├── evaluate.py             # Evaluation script
│   ├── prepare_data.py         # Data preparation script
│   ├── inference.py            # Inference utilities
│   ├── model.py                # ResNet18 model definition
│   ├── dataset.py              # Dataset utilities
│   ├── requirements.txt        # Python dependencies
│   ├── dvc.yaml                # DVC pipeline definition
│   ├── configs/                # Hydra configuration
│   │   ├── config.yaml         # Main config with defaults
│   │   ├── model/              # Model configs
│   │   │   ├── cifar10_cnn.yaml
│   │   │   └── resnet18.yaml
│   │   ├── training/           # Training configs
│   │   │   ├── default.yaml
│   │   │   ├── fast.yaml
│   │   │   ├── long.yaml
│   │   │   └── sgd.yaml
│   │   ├── data/               # Data configs
│   │   │   ├── cifar10.yaml
│   │   │   └── cifar10_minimal.yaml
│   │   ├── paths/              # Path configs
│   │   │   └── default.yaml
│   │   └── experiment/         # Experiment presets
│   │       ├── baseline.yaml
│   │       ├── quick_test.yaml
│   │       ├── high_accuracy.yaml
│   │       └── resnet_baseline.yaml
│   ├── data/                   # Dataset directory (auto-downloaded)
│   └── models/                 # Saved model checkpoints
├── tests/                      # Test suite
│   ├── test_training.py        # Training tests
│   ├── test_model.py           # Model tests
│   ├── test_dataset.py         # Dataset tests
│   ├── test_inference.py       # Inference tests
│   ├── test_hydra_configs.py   # Config validation tests
│   └── test_dvc_pipeline.py    # DVC pipeline tests
├── agent_queries.md            # Sample queries for the agent
├── setup_example.sh            # Setup script
├── run_example.py              # Python script to run the example
└── docs/
    └── walkthrough.md          # Detailed walkthrough
```

## Dataset

This example uses **CIFAR-10**, a standard benchmark dataset containing 60,000 32x32 color images in 10 classes:

- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images
- 10,000 test images

The dataset is automatically downloaded on first run.

## Model Architectures

### CIFAR10CNN (Default)
A custom CNN optimized for CIFAR-10:
- 3 convolutional layers with batch normalization
- Max pooling after each conv layer
- 2 fully connected layers with dropout
- ~1.2M parameters

### ResNet18
A ResNet18 adapted for CIFAR-10:
- Modified first conv layer (3x3 instead of 7x7)
- Removed initial max pooling for small images
- Optional pretrained ImageNet weights
- ~11M parameters

## Hydra Configuration

The project uses modular Hydra configs for flexible experiment management:

### Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `model` | `cifar10_cnn`, `resnet18` | Model architecture |
| `training` | `default`, `fast`, `long`, `sgd` | Training hyperparameters |
| `data` | `cifar10`, `cifar10_minimal` | Dataset configuration |
| `paths` | `default` | Output paths |

### Experiment Presets

| Experiment | Description | Command |
|------------|-------------|---------|
| `baseline` | Standard training (10 epochs) | `python train.py +experiment=baseline` |
| `quick_test` | Fast testing (2 epochs) | `python train.py +experiment=quick_test` |
| `high_accuracy` | Extended training (50 epochs) | `python train.py +experiment=high_accuracy` |
| `resnet_baseline` | ResNet18 model | `python train.py +experiment=resnet_baseline` |

### Override Examples

```bash
# Change learning rate
python train.py training.learning_rate=0.01

# Use SGD optimizer with more epochs
python train.py training=sgd training.epochs=30

# Use ResNet18 with pretrained weights
python train.py model=resnet18 model.pretrained=true

# Combine multiple overrides
python train.py model=resnet18 training=long data=cifar10
```

## DVC Pipeline

The DVC pipeline automates the ML workflow:

```
prepare_data -> train -> evaluate
```

### Stages

1. **prepare_data**: Downloads CIFAR-10 and creates data info
2. **train**: Trains the model using Hydra config
3. **evaluate**: Evaluates the model and saves metrics

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
# From the image_classification directory
cd examples/image_classification

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ -v --cov=project
```

## Sample Agent Queries

Here are example queries you can use with the Auto-MLOps agent:

### Setup Pipeline
```
"Set up an MLOps pipeline for CIFAR-10 image classification with accuracy threshold 0.85"
```

### Train Model
```
"Train the CIFAR-10 classifier with ResNet18 and track metrics with MLflow"
```

### Deploy Model
```
"Deploy my image classifier to Gradio for a quick demo"
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

## Troubleshooting

### CIFAR-10 Download Issues

If the dataset download fails:
```bash
# Manually trigger download
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', download=True)"
```

### CUDA Out of Memory

Reduce batch size:
```bash
python train.py training.batch_size=16
```

### Hydra Config Errors

Validate configs:
```bash
python train.py --cfg job
```

## License

MIT License - See the main project LICENSE file.
