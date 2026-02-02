# Sample Auto-MLOps Agent Queries

These are example natural language queries you can use with the Auto-MLOps agent for the tabular regression project.

## Pipeline Setup

```
Set up a complete MLOps pipeline for the tabular regression project
```

```
Initialize MLflow tracking for my regression experiments
```

```
Create a DVC pipeline for the California Housing regression project
```

## Training

```
Train an MLP model on the California Housing dataset with 100 epochs
```

```
Train a TabNet model with batch size 64 and learning rate 0.001
```

```
Run a quick training test with synthetic data
```

```
Train until RMSE is below 0.5
```

## Experiment Tracking

```
Start a new MLflow experiment called "housing-regression-v2"
```

```
Log the current model metrics to MLflow
```

```
Find the best run from the housing-regression experiment
```

```
Compare RMSE across my last 5 experiments
```

## Configuration

```
Create a Hydra config for TabNet with 3 attention steps
```

```
Update the training config to use AdamW optimizer with weight decay 0.01
```

```
Set up experiment configs for hyperparameter search
```

## Data Management

```
Initialize DVC and add the training data to version control
```

```
Configure a remote S3 bucket for DVC storage
```

```
Create a data preprocessing pipeline stage
```

## Model Evaluation

```
Evaluate the trained model and show all regression metrics
```

```
Generate predictions on the test set and compute R² score
```

```
Create a residual plot for model diagnostics
```

## Deployment

```
Deploy the regression model to LitServe for high-throughput inference
```

```
Create a Gradio demo for the housing price predictor
```

```
Package the model for AWS Lambda deployment
```

```
Create a TorchServe handler for the regression model
```

```
Generate Kubernetes deployment configs with KServe
```

## CI/CD

```
Create a GitHub Actions workflow for automated training
```

```
Add a workflow step to run model evaluation on PR
```

```
Set up automated model registration when tests pass
```

## Self-Improvement Loop

```
Train the model and improve until R² is above 0.85
```

```
If RMSE is above 0.55, suggest hyperparameter improvements
```

```
Run 3 improvement iterations targeting RMSE below 0.45
```

## Docker

```
Create a Dockerfile for training the regression model
```

```
Build and push the training container to Docker Hub
```

```
Run training in a Docker container with GPU support
```
