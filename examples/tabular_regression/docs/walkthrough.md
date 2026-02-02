# Tabular Regression Walkthrough

This guide walks you through building a complete tabular regression pipeline using Auto-MLOps.

## Prerequisites

- Python 3.9+
- Auto-MLOps installed (`pip install -e .` from repo root)
- (Optional) MLflow for experiment tracking
- (Optional) DVC for data versioning

## Step 1: Understand the Project Structure

The tabular regression example follows the standard Auto-MLOps project layout:

```
project/
├── train.py          # Hydra-powered training
├── evaluate.py       # Model evaluation
├── prepare_data.py   # Data download/preparation
├── inference.py      # Inference utilities
├── model.py          # Neural network architectures
├── dataset.py        # Data loading utilities
├── dvc.yaml          # Pipeline definition
└── configs/          # Hydra configuration hierarchy
```

## Step 2: Prepare the Data

First, download and prepare the California Housing dataset:

```bash
cd examples/tabular_regression/project
python prepare_data.py --data-dir data --dataset california
```

This creates:
- `data/train.csv` - Training data (80%)
- `data/test.csv` - Test data (20%)
- `data/data_info.json` - Dataset statistics
- `data/scaler.pkl` - Feature normalization parameters

### Dataset Overview

The California Housing dataset contains 20,640 samples with 8 features:

| Feature | Description | Range |
|---------|-------------|-------|
| MedInc | Median income | 0.5 - 15.0 |
| HouseAge | Median house age | 1 - 52 |
| AveRooms | Average rooms | 0.8 - 141.9 |
| AveBedrms | Average bedrooms | 0.3 - 34.1 |
| Population | Block population | 3 - 35,682 |
| AveOccup | Average occupancy | 0.7 - 1243.3 |
| Latitude | Location | 32.5 - 42.0 |
| Longitude | Location | -124.4 - -114.3 |

**Target**: Median house value (in $100,000s), range 0.15 - 5.0

## Step 3: Choose a Model

Three model architectures are available:

### MLP (Multi-Layer Perceptron)

A deep feedforward network with configurable hidden layers:

```yaml
# configs/model/mlp.yaml
name: mlp
input_dim: 8
hidden_dims: [128, 64, 32]
dropout: 0.2
activation: relu
```

Best for: General purpose, fast training, interpretable

### TabNet

Attention-based architecture designed for tabular data:

```yaml
# configs/model/tabnet.yaml
name: tabnet
input_dim: 8
n_steps: 3
n_d: 64
n_a: 64
gamma: 1.5
```

Best for: Feature selection, high accuracy, sparse attention

## Step 4: Configure Training

The training configuration controls hyperparameters:

```yaml
# configs/training/default.yaml
epochs: 100
batch_size: 128
learning_rate: 0.001
optimizer: adam
weight_decay: 0.0001
early_stopping:
  patience: 10
  min_delta: 0.0001
scheduler:
  name: reduce_on_plateau
  factor: 0.5
  patience: 5
```

## Step 5: Train the Model

### Basic Training

```bash
python train.py
```

### With Overrides

```bash
# Change model
python train.py model=tabnet

# Change hyperparameters
python train.py training.epochs=200 training.learning_rate=0.0005

# Use experiment preset
python train.py +experiment=high_accuracy
```

### Training Output

Training produces:
- `models/best_model.pt` - Best validation checkpoint
- `models/final_model.pt` - Final epoch checkpoint
- `models/model_config.json` - Model configuration
- Console output with progress bars

## Step 6: Evaluate the Model

```bash
python evaluate.py --model-path models/best_model.pt --data-dir data
```

This outputs `metrics.json`:

```json
{
  "rmse": 0.4823,
  "mae": 0.3245,
  "r2": 0.7892,
  "mape": 15.23,
  "samples": 4128
}
```

### Understanding Metrics

- **RMSE** (Root Mean Square Error): Standard deviation of prediction errors. Lower is better.
- **MAE** (Mean Absolute Error): Average absolute error. More robust to outliers.
- **R²** (R-squared): Proportion of variance explained. 1.0 is perfect.
- **MAPE** (Mean Absolute Percentage Error): Percentage error, useful for business context.

## Step 7: Run Inference

### Command Line

```bash
python inference.py --model-path models/best_model.pt
```

### Python API

```python
from inference import TabularRegressor

# Load model
model = TabularRegressor("models/best_model.pt", "data/scaler.pkl")

# Single prediction
features = {
    "MedInc": 8.3,
    "HouseAge": 41,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322,
    "AveOccup": 2.56,
    "Latitude": 37.88,
    "Longitude": -122.23
}
prediction = model.predict(features)
print(f"Predicted value: ${prediction * 100000:.2f}")

# Batch prediction
df = pd.read_csv("data/test.csv")
predictions = model.predict_batch(df)
```

## Step 8: Use DVC Pipeline

The `dvc.yaml` defines the complete pipeline:

```yaml
stages:
  prepare_data:
    cmd: python prepare_data.py --data-dir data --dataset california
    deps:
      - prepare_data.py
    outs:
      - data/train.csv
      - data/test.csv
      - data/data_info.json

  train:
    cmd: python train.py
    deps:
      - train.py
      - model.py
      - dataset.py
      - data/train.csv
      - configs/
    params:
      - configs/config.yaml:
          - model
          - training
    outs:
      - models/best_model.pt
      - models/model_config.json

  evaluate:
    cmd: python evaluate.py --model-path models/best_model.pt --data-dir data
    deps:
      - evaluate.py
      - models/best_model.pt
      - data/test.csv
    metrics:
      - metrics.json:
          cache: false
```

Run the full pipeline:

```bash
dvc repro
```

## Step 9: Track with MLflow

Enable MLflow tracking:

```bash
# Start MLflow server (optional)
mlflow ui --port 5000

# Train with tracking
python train.py mlflow.enabled=true mlflow.experiment_name="housing-regression"
```

## Step 10: Use Auto-MLOps Agent

Let the agent manage your pipeline:

```bash
# Interactive mode
mlops-agent -i --project .

# Single query
mlops-agent "Train TabNet model with early stopping and log to MLflow"
```

### Example Agent Interaction

```
You: Set up MLOps pipeline for this project

Agent: I'll set up a complete MLOps pipeline for your tabular regression project.

1. Initializing MLflow experiment "housing-regression"...
2. Creating Hydra configuration structure...
3. Setting up DVC pipeline...
4. Training baseline MLP model...

Training complete! Results:
- RMSE: 0.4823
- R²: 0.7892
- Model saved to models/best_model.pt
- Metrics logged to MLflow

Would you like me to:
- Try a different model (TabNet)?
- Run hyperparameter optimization?
- Deploy the model?
```

## Tips and Best Practices

### Feature Engineering

For better results, consider:
1. Log-transforming skewed features (Population, AveOccup)
2. Creating interaction features (Latitude × Longitude for location clusters)
3. Binning continuous features for nonlinear relationships

### Hyperparameter Tuning

Key hyperparameters to tune:
- Learning rate: Start with 0.001, try 0.0001 - 0.01
- Hidden dimensions: Try [64, 32], [128, 64, 32], [256, 128, 64]
- Dropout: 0.1 - 0.3 for regularization
- Batch size: 32, 64, 128, 256

### Avoiding Overfitting

1. Use early stopping (default patience: 10)
2. Apply dropout (default: 0.2)
3. Use weight decay (default: 0.0001)
4. Monitor train vs validation loss

## Troubleshooting

### High RMSE

- Increase model capacity (more hidden layers)
- Try TabNet for better feature learning
- Check for data quality issues
- Ensure features are normalized

### Slow Training

- Reduce batch size if GPU memory limited
- Use fewer hidden dimensions
- Enable mixed precision training

### Memory Issues

- Use smaller batch sizes
- Reduce hidden dimensions
- Process data in chunks

## Next Steps

1. **Experiment**: Try different model architectures
2. **Tune**: Use Optuna for hyperparameter optimization
3. **Deploy**: Use Auto-MLOps to deploy to your target platform
4. **Monitor**: Set up model monitoring in production
