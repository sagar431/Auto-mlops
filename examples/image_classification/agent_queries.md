# Sample Agent Queries for Image Classification

This document contains example queries you can use with the Auto-MLOps agent to set up, train, and deploy your image classification model.

## Setup Queries

### Basic Pipeline Setup
```
Set up an MLOps pipeline for image classification in examples/image_classification/project
```

### Setup with Accuracy Threshold
```
Set up MLOps for my cat-dog classifier with 85% accuracy target
```

### Full Configuration
```
Create a complete MLOps setup with Hydra configs, MLflow tracking, and DVC versioning for image classification
```

## Data Management Queries

### Initialize DVC
```
Initialize DVC for version control of my image dataset
```

### Add Data to DVC
```
Add my training data to DVC and push to remote storage
```

## Training Queries

### Basic Training
```
Train my image classification model and track metrics
```

### Training with Target
```
Train the model until we reach 90% validation accuracy
```

### Training with Parameters
```
Train with batch size 64, learning rate 0.0001, for 20 epochs
```

## Evaluation Queries

### Check Results
```
Analyze the training results and show me the best run
```

### Compare Runs
```
Compare all MLflow runs and show me the best performing model
```

## Improvement Queries

### Suggest Improvements
```
My accuracy is 75%, suggest hyperparameter changes to improve it
```

### Auto-Improve
```
Keep training and tuning until we reach 85% accuracy
```

## Deployment Queries

### Gradio Demo
```
Deploy my image classifier to Gradio for a quick demo
```

### LitServe API
```
Create a high-throughput API for my model using LitServe
```

### AWS Lambda
```
Deploy the model to AWS Lambda for serverless inference
```

### TorchServe
```
Create a TorchServe deployment for production use
```

### KServe (Kubernetes)
```
Deploy to KServe for Kubernetes-native model serving
```

### Hugging Face Spaces
```
Deploy my classifier to Hugging Face Spaces
```

## Combined Workflows

### End-to-End Pipeline
```
Set up the full pipeline, train until 85% accuracy, then deploy to Gradio
```

### Quick Demo Flow
```
Create a quick demo with synthetic data and deploy to Gradio so I can test the model
```

### Production Flow
```
Set up production pipeline with MLflow, train model, and deploy to TorchServe with monitoring
```

## Monitoring Queries

### Setup Monitoring
```
Set up model performance monitoring with alerts
```

### Check for Drift
```
Check if there's data drift in my recent predictions
```
