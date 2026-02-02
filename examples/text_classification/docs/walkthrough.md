# Text Classification Walkthrough

This document provides a detailed walkthrough of using Auto-MLOps to build a complete MLOps pipeline for text classification.

## Overview

We'll build a sentiment analysis system that can classify movie reviews as positive or negative. The workflow includes:

1. Setting up the project structure
2. Configuring the training pipeline
3. Training models
4. Evaluating performance
5. Deploying to production

## Prerequisites

Before starting, ensure you have:

- Python 3.10+
- Auto-MLOps installed (`pip install -e .` from project root)
- API key configured (Google Gemini or OpenAI)

## Step 1: Understand the Project Structure

The text classification example follows this structure:

```
project/
├── train.py          # Main training script
├── model.py          # TextCNN and LSTM models
├── dataset.py        # Data loading utilities
├── inference.py      # Inference wrapper
├── evaluate.py       # Model evaluation
├── prepare_data.py   # Data preparation
├── dvc.yaml          # DVC pipeline
└── configs/          # Hydra configuration
```

## Step 2: Configure the Pipeline

### Hydra Configuration

The project uses modular Hydra configs. The main `config.yaml` defines defaults:

```yaml
defaults:
  - model: textcnn
  - training: default
  - data: imdb
  - paths: default
  - _self_

seed: 42
```

### Model Configuration

Choose between TextCNN and LSTM:

**TextCNN** (`configs/model/textcnn.yaml`):
- Multiple parallel convolutions capture n-gram patterns
- Fast training and inference
- Good baseline performance

**LSTM** (`configs/model/lstm.yaml`):
- Bidirectional LSTM captures sequential patterns
- Attention mechanism focuses on important words
- Better for longer texts

### Training Configuration

Adjust hyperparameters in `configs/training/`:

- `default.yaml` - Balanced settings (10 epochs)
- `fast.yaml` - Quick experiments (3 epochs)
- `long.yaml` - Extended training (30 epochs)

## Step 3: Prepare Data

### Using IMDB Dataset

The IMDB dataset will be automatically downloaded on first run:

```bash
python prepare_data.py --data-dir data --output data/data_info.json
```

### Using Synthetic Data

For quick testing without downloading:

```bash
python prepare_data.py --data-dir data --output data/data_info.json --synthetic
```

## Step 4: Train the Model

### Basic Training

```bash
cd project
python train.py
```

This will:
1. Load or create the dataset
2. Build vocabulary from training data
3. Create the TextCNN model
4. Train for 10 epochs
5. Save best model to `models/`

### Using Experiment Presets

```bash
# Quick test
python train.py +experiment=quick_test

# LSTM model
python train.py +experiment=lstm_baseline

# Extended training
python train.py +experiment=high_accuracy
```

### Custom Configuration

Override specific settings:

```bash
python train.py model=lstm training.epochs=20 training.learning_rate=0.0005
```

## Step 5: Evaluate the Model

Run evaluation on the test set:

```bash
python evaluate.py --model-path models/best_model.pt --vocab-path models/vocab.json
```

This outputs:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix

## Step 6: Run Inference

Test predictions on new text:

```bash
python inference.py "This movie was fantastic! I loved every minute." --model models/best_model.pt --vocab models/vocab.json
```

## Step 7: Use DVC Pipeline

The DVC pipeline automates the entire workflow:

```bash
# Initialize DVC
dvc init

# Run full pipeline
dvc repro

# View pipeline
dvc dag
```

## Step 8: Use Auto-MLOps Agent

Let the agent handle the workflow:

```bash
# Interactive mode
mlops-agent -i --project ./project

# Example queries:
> Set up MLOps pipeline for sentiment analysis
> Train with LSTM model and track metrics with MLflow
> Deploy to Gradio for demo
```

## Model Details

### TextCNN Architecture

```
Input (batch_size, seq_len)
    ↓
Embedding (batch_size, seq_len, embedding_dim)
    ↓
Conv1D (kernel_size=3) → ReLU → MaxPool
Conv1D (kernel_size=4) → ReLU → MaxPool
Conv1D (kernel_size=5) → ReLU → MaxPool
    ↓
Concatenate
    ↓
Dropout
    ↓
Linear → Output (batch_size, num_classes)
```

### LSTM Architecture

```
Input (batch_size, seq_len)
    ↓
Embedding (batch_size, seq_len, embedding_dim)
    ↓
Bidirectional LSTM (2 layers)
    ↓
Attention
    ↓
Dropout
    ↓
Linear → Output (batch_size, num_classes)
```

## Performance Tips

### Vocabulary Size

- Larger vocab captures more words but increases model size
- Default 30,000 is good for most cases
- Reduce for smaller datasets

### Sequence Length

- Longer sequences capture more context but use more memory
- Default 256 is good for IMDB reviews
- Reduce for shorter texts

### Batch Size

- Larger batches are faster but use more memory
- Reduce if you encounter OOM errors
- 64 is a good default

## Troubleshooting

### OOM Errors

```bash
# Reduce batch size and sequence length
python train.py training.batch_size=16 data.max_length=128
```

### Slow Training

```bash
# Enable GPU
# Model will automatically use CUDA if available

# Use faster settings
python train.py +experiment=quick_test
```

### Poor Accuracy

```bash
# Try LSTM model
python train.py model=lstm

# Use extended training
python train.py +experiment=high_accuracy

# Increase embedding dimension
python train.py model.embedding_dim=256
```

## Next Steps

After training a good model:

1. **Deploy for demo**: Use Gradio for quick interactive testing
2. **Deploy to production**: Use LitServe, TorchServe, or Lambda
3. **Set up CI/CD**: Create GitHub Actions workflow
4. **Monitor performance**: Track metrics with MLflow
