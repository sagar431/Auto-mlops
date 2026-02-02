# Sample Agent Queries for Text Classification

This document contains example queries you can use with the Auto-MLOps agent to set up, train, and deploy the text classification model.

## Setup and Configuration

### Initialize MLOps Pipeline
```
Set up an MLOps pipeline for text classification in examples/text_classification/project
```

### Create Hydra Configuration
```
Create modular Hydra configs for sentiment analysis with TextCNN model
```

### Initialize Version Control
```
Initialize DVC and MLflow tracking for the text classification project
```

## Training

### Basic Training
```
Train a sentiment classifier on the IMDB dataset
```

### Training with Specific Model
```
Train an LSTM-based sentiment classifier with attention mechanism
```

### Training with Threshold
```
Train the sentiment model with accuracy threshold 0.88
```

### Quick Test
```
Run a quick training test with synthetic data to verify the pipeline works
```

### Extended Training
```
Train for longer to achieve higher accuracy on sentiment classification
```

## Hyperparameter Tuning

### Learning Rate
```
Try different learning rates for the sentiment classifier: 0.01, 0.001, 0.0001
```

### Model Architecture
```
Compare TextCNN vs LSTM performance on the IMDB dataset
```

### Embedding Dimension
```
Experiment with embedding dimension 256 instead of 128
```

## Evaluation

### Evaluate Model
```
Evaluate the trained sentiment model on the test set
```

### Get Detailed Metrics
```
Show per-class precision, recall, and F1 scores for the sentiment classifier
```

## Deployment

### Gradio Demo
```
Deploy my sentiment classifier to Gradio for an interactive demo
```

### AWS Lambda
```
Deploy the text classifier to AWS Lambda for serverless inference
```

### High-Throughput API
```
Create a LitServe API for the sentiment model that can handle high traffic
```

### TorchServe Production
```
Deploy the sentiment model to TorchServe with model versioning
```

### Kubernetes
```
Create KServe InferenceService for the text classification model
```

## Full Workflow Examples

### End-to-End Pipeline
```
Set up complete MLOps pipeline for IMDB sentiment classification:
1. Create Hydra configs
2. Initialize MLflow tracking
3. Train TextCNN model with accuracy threshold 0.85
4. Log metrics and save best model
5. Deploy to Gradio when accuracy target is met
```

### CI/CD Integration
```
Create a GitHub Actions workflow for the text classification project that:
- Runs tests on pull requests
- Trains model on merge to main
- Deploys to production when accuracy exceeds threshold
```

### Experiment Tracking
```
Set up MLflow experiment tracking for sentiment analysis and run multiple experiments:
- TextCNN with different filter sizes
- LSTM with varying hidden dimensions
- Compare results and select best model
```
