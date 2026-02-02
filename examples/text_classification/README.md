# Text Classification Example

This example demonstrates how to use Auto-MLOps to set up, train, and deploy a text classification model for sentiment analysis using natural language commands.

## Overview

This example shows the complete MLOps workflow:
1. **Setup** - Create modular Hydra configs, initialize MLflow experiment
2. **Data** - Download IMDB dataset or use synthetic data, version control with DVC
3. **Training** - Train a TextCNN, LSTM, or DistilBERT model with metric logging
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
cd examples/text_classification/project
pip install -r requirements.txt
```

### Run Training

```bash
# Default training (IMDB with TextCNN, 10 epochs)
cd project
python train.py

# Quick test with synthetic data (3 epochs)
python train.py +experiment=quick_test

# Use LSTM model
python train.py +experiment=lstm_baseline

# Use DistilBERT (highest accuracy, requires transformers library)
python train.py +experiment=distilbert_baseline

# Extended training for best accuracy
python train.py +experiment=high_accuracy
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
mlops-agent "Set up MLOps pipeline for sentiment analysis in examples/text_classification/project"

# Interactive mode
mlops-agent -i --project ./project
```

## Project Structure

```
examples/text_classification/
├── README.md                    # This file
├── project/                     # ML project directory
│   ├── train.py                # Training script with Hydra
│   ├── evaluate.py             # Evaluation script
│   ├── prepare_data.py         # Data preparation script
│   ├── inference.py            # Inference utilities
│   ├── model.py                # Model definitions (TextCNN, LSTM, DistilBERT)
│   ├── dataset.py              # Dataset utilities
│   ├── requirements.txt        # Python dependencies
│   ├── dvc.yaml                # DVC pipeline definition
│   ├── configs/                # Hydra configuration
│   │   ├── config.yaml         # Main config with defaults
│   │   ├── model/              # Model configs
│   │   │   ├── textcnn.yaml
│   │   │   ├── lstm.yaml
│   │   │   └── distilbert.yaml
│   │   ├── training/           # Training configs
│   │   │   ├── default.yaml
│   │   │   ├── fast.yaml
│   │   │   └── long.yaml
│   │   ├── data/               # Data configs
│   │   │   ├── imdb.yaml
│   │   │   └── synthetic.yaml
│   │   ├── paths/              # Path configs
│   │   │   └── default.yaml
│   │   └── experiment/         # Experiment presets
│   │       ├── baseline.yaml
│   │       ├── quick_test.yaml
│   │       ├── lstm_baseline.yaml
│   │       ├── distilbert_baseline.yaml
│   │       └── high_accuracy.yaml
│   ├── data/                   # Dataset directory
│   └── models/                 # Saved model checkpoints
├── tests/                      # Test suite
│   ├── test_training.py        # Training tests
│   ├── test_model.py           # Model tests
│   ├── test_dataset.py         # Dataset tests
│   ├── test_inference.py       # Inference tests
│   ├── test_hydra_configs.py   # Config validation tests
│   └── test_dvc_pipeline.py    # DVC pipeline tests
├── agent_queries.md            # Sample queries for the agent
└── docs/
    └── walkthrough.md          # Detailed walkthrough
```

## Dataset

This example uses **IMDB Movie Reviews**, a standard benchmark for binary sentiment classification:

- 25,000 training reviews (12,500 positive, 12,500 negative)
- 25,000 test reviews (12,500 positive, 12,500 negative)
- Binary classification: positive / negative sentiment

For quick testing without downloading, you can use synthetic data:
```bash
python train.py +experiment=quick_test
```

## Model Architectures

### TextCNN (Default)
A CNN architecture for text classification:
- Embedding layer
- Multiple parallel 1D convolutions (kernel sizes 3, 4, 5)
- Max pooling over time
- Fully connected layer with dropout

**Expected accuracy:** ~82-85% on IMDB

### LSTM
A bidirectional LSTM with attention:
- Embedding layer
- 2-layer bidirectional LSTM
- Attention mechanism
- Fully connected layer with dropout

**Expected accuracy:** ~84-87% on IMDB

### DistilBERT
A pretrained transformer model for high accuracy:
- Pretrained DistilBERT encoder (`distilbert-base-uncased`)
- [CLS] token pooling
- Classification head with dropout
- Optional encoder freezing for fine-tuning

**Expected accuracy:** ~88-92% on IMDB

**Note:** Requires the `transformers` library. Training is slower but achieves the best accuracy.

## Hydra Configuration

The project uses modular Hydra configs for flexible experiment management:

### Config Groups

| Group | Options | Description |
|-------|---------|-------------|
| `model` | `textcnn`, `lstm`, `distilbert` | Model architecture |
| `training` | `default`, `fast`, `long` | Training hyperparameters |
| `data` | `imdb`, `synthetic` | Dataset configuration |
| `paths` | `default` | Output paths |

### Experiment Presets

| Experiment | Description | Command |
|------------|-------------|---------|
| `baseline` | Standard training (10 epochs) | `python train.py +experiment=baseline` |
| `quick_test` | Fast testing with synthetic data | `python train.py +experiment=quick_test` |
| `lstm_baseline` | LSTM model training | `python train.py +experiment=lstm_baseline` |
| `distilbert_baseline` | DistilBERT (highest accuracy) | `python train.py +experiment=distilbert_baseline` |
| `high_accuracy` | Extended training (30 epochs) | `python train.py +experiment=high_accuracy` |

### Override Examples

```bash
# Change learning rate
python train.py training.learning_rate=0.01

# Use LSTM with more epochs
python train.py model=lstm training.epochs=20

# Use DistilBERT with frozen encoder (faster fine-tuning)
python train.py model=distilbert model.freeze_encoder=true

# Change embedding dimension (TextCNN/LSTM only)
python train.py model.embedding_dim=256

# Combine multiple overrides
python train.py model=lstm training=long data=imdb
```

## DVC Pipeline

The DVC pipeline automates the ML workflow:

```
prepare_data -> train -> evaluate
```

### Stages

1. **prepare_data**: Downloads/creates dataset and writes data info
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
# From the text_classification directory
cd examples/text_classification

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
"Set up an MLOps pipeline for IMDB sentiment classification with accuracy threshold 0.85"
```

### Train Model
```
"Train the sentiment classifier with LSTM and track metrics with MLflow"
```

### Deploy Model
```
"Deploy my sentiment classifier to Gradio for a quick demo"
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

### IMDB Download Issues

If the dataset download fails:
```bash
# Use synthetic data for testing
python train.py data=synthetic
```

### CUDA Out of Memory

Reduce batch size or max sequence length:
```bash
python train.py training.batch_size=16 data.max_length=128
```

### Hydra Config Errors

Validate configs:
```bash
python train.py --cfg job
```

## License

MIT License - See the main project LICENSE file.
