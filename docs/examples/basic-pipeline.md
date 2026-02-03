# Basic Pipeline Tutorial

This tutorial walks you through setting up a complete MLOps pipeline for an image classification project.

## Prerequisites

- Auto-MLOps installed
- API key configured
- Python 3.10+

## Step 1: Initialize the Project

```bash
# Create a new project
mlops-agent init ./my-classifier --template pytorch

cd my-classifier
```

This creates:
```
my-classifier/
├── configs/
│   ├── config.yaml
│   ├── model/default.yaml
│   ├── training/default.yaml
│   └── data/default.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
├── train.py
├── requirements.txt
└── .gitignore
```

## Step 2: Configure Hydra

```bash
mlops-agent "Create Hydra config with:
- model: resnet18, pretrained=true, num_classes=10
- training: epochs=20, batch_size=64, lr=0.001, optimizer=adam
- data: dataset=cifar10, augmentation=true"
```

This generates:

```yaml
# configs/config.yaml
defaults:
  - model: resnet18
  - training: default
  - data: cifar10

project_name: my-classifier
seed: 42
```

## Step 3: Set Up MLflow

```bash
mlops-agent "Initialize MLflow experiment named 'cifar10-classifier' with tracking server at ./mlruns"
```

This:
- Creates `mlruns/` directory
- Sets up experiment tracking
- Configures artifact storage

## Step 4: Initialize DVC

```bash
mlops-agent "Set up DVC with local remote at ./dvc-storage"
```

This creates:
- `.dvc/` configuration
- `dvc.yaml` pipeline file
- Local storage for data versioning

## Step 5: Write Training Code

Edit `train.py`:

```python
#!/usr/bin/env python3
import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import datasets, transforms, models

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    # Set up MLflow
    mlflow.set_experiment(cfg.get("experiment_name", "default"))

    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": cfg.model.name,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
        })

        # Load data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        train_data = datasets.CIFAR10(
            "data/raw", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=cfg.training.batch_size, shuffle=True
        )

        # Create model
        model = models.resnet18(pretrained=cfg.model.pretrained)
        model.fc = nn.Linear(model.fc.in_features, cfg.model.num_classes)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg.training.epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = correct / total
            mlflow.log_metrics({
                "loss": total_loss / len(train_loader),
                "accuracy": accuracy
            }, step=epoch)

            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")

        # Save model
        torch.save(model.state_dict(), "models/model.pt")
        mlflow.log_artifact("models/model.pt")

        return accuracy

if __name__ == "__main__":
    train()
```

## Step 6: Create DVC Pipeline

```bash
mlops-agent "Create DVC pipeline with stages:
1. prepare: download and preprocess CIFAR10 data
2. train: run training with Hydra config
3. evaluate: compute test metrics"
```

This generates `dvc.yaml`:

```yaml
stages:
  prepare:
    cmd: python src/prepare_data.py
    deps:
      - src/prepare_data.py
    outs:
      - data/processed

  train:
    cmd: python train.py
    deps:
      - train.py
      - configs/
      - data/processed
    outs:
      - models/model.pt
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - src/evaluate.py
      - models/model.pt
    metrics:
      - evaluation.json:
          cache: false
```

## Step 7: Train the Model

```bash
# Run with default config
python train.py

# Or with overrides
python train.py training.epochs=50 training.learning_rate=0.0001
```

## Step 8: View Results

```bash
# Start MLflow UI
mlflow ui

# Open http://localhost:5000
```

## Step 9: Deploy to Gradio

```bash
mlops-agent deploy gradio --model ./models/model.pt
```

This creates `app.py`:

```python
import gradio as gr
import torch
from torchvision import transforms, models

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load("models/model.pt"))
model.eval()

# CIFAR-10 classes
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

def predict(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]

    return {CLASSES[i]: float(probs[i]) for i in range(10)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    title="CIFAR-10 Classifier"
)

demo.launch()
```

Run it:

```bash
python app.py
# Open http://localhost:7860
```

## Next Steps

- [Custom Deployment](./custom-deployment.md) - Deploy to other targets
- [API Reference](../api-reference.md) - Full API documentation
- [Security](../security.md) - Secure your deployment
