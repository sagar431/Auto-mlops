# Image Classification Example

This example demonstrates how to use Auto-MLOps to set up, train, and deploy an image classification model using natural language commands.

## Overview

This example shows the complete MLOps workflow:
1. **Setup** - Create Hydra configs, initialize MLflow experiment
2. **Data** - Version control with DVC
3. **Training** - Train a CNN model with metric logging
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

### Setup Example

```bash
cd examples/image_classification
./setup_example.sh
```

### Run with Auto-MLOps Agent

```bash
# Single command mode
mlops-agent "Set up MLOps pipeline for cat-dog image classifier in examples/image_classification/project"

# Interactive mode
mlops-agent -i --project ./project
```

## Project Structure

```
examples/image_classification/
├── README.md                    # This file
├── project/                     # ML project directory
│   ├── train.py                # Training script
│   ├── inference.py            # Inference utilities
│   ├── model.py                # CNN model definition
│   ├── dataset.py              # Dataset utilities
│   ├── requirements.txt        # Python dependencies
│   ├── data/                   # Dataset directory
│   ├── models/                 # Saved model checkpoints
│   ├── configs/                # Hydra configs (created by agent)
│   └── logs/                   # Training logs
├── agent_queries.md            # Sample queries for the agent
├── setup_example.sh            # Setup script
├── run_example.py              # Python script to run the example
└── docs/
    └── walkthrough.md          # Detailed walkthrough
```

## Sample Agent Queries

Here are some example queries you can use with the Auto-MLOps agent:

### Setup Pipeline
```
"Set up an MLOps pipeline for image classification with accuracy threshold 0.85"
```

### Train Model
```
"Train the image classifier and track metrics with MLflow"
```

### Deploy Model
```
"Deploy my image classifier to Gradio for a quick demo"
"Deploy the model to AWS Lambda for serverless inference"
"Create a high-throughput API with LitServe"
```

See `agent_queries.md` for more examples.

## Dataset

This example uses a simple cat-dog classification dataset. You can:

1. **Use the synthetic demo data** (default) - Small dataset for testing
2. **Download real data** - Use the setup script to fetch a real dataset
3. **Use your own data** - Place images in `project/data/train/{class_name}/`

## Model Architecture

The example uses a simple CNN architecture suitable for demonstration:

- 3 convolutional layers with ReLU activation
- Max pooling after each conv layer
- 2 fully connected layers
- Dropout for regularization

## Training Configuration

Default Hydra configuration (created by agent):

```yaml
model:
  num_classes: 2
  dropout: 0.5

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam

data:
  image_size: 224
  train_split: 0.8
```

## Deployment Options

After training, deploy to any of these targets:

| Target | Command | Use Case |
|--------|---------|----------|
| Gradio | `"Deploy to Gradio"` | Quick demo, prototyping |
| LitServe | `"Deploy with LitServe"` | High-throughput API |
| Lambda | `"Deploy to AWS Lambda"` | Serverless, pay-per-use |
| TorchServe | `"Deploy with TorchServe"` | Enterprise production |
| KServe | `"Deploy to KServe"` | Kubernetes-native |

## License

MIT License - See the main project LICENSE file.
