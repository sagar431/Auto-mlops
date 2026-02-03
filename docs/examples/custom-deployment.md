# Custom Deployment Guide

This guide covers advanced deployment scenarios and customization options.

## Deploying to AWS Lambda

### Prerequisites

- AWS CLI configured
- AWS CDK installed (`npm install -g aws-cdk`)
- Docker installed

### Step 1: Generate Lambda Deployment

```bash
mlops-agent "Deploy my model to AWS Lambda with:
- Memory: 1024MB
- Timeout: 30 seconds
- API Gateway with CORS"
```

### Step 2: Review Generated Files

```
deployment/
├── app.py           # FastAPI application
├── Dockerfile       # Lambda container image
├── cdk.py          # CDK infrastructure
└── requirements.txt
```

### Step 3: Customize the Handler

Edit `app.py`:

```python
from fastapi import FastAPI
from mangum import Mangum
import torch

app = FastAPI()

# Load model at cold start
model = None

def get_model():
    global model
    if model is None:
        model = torch.load("/opt/ml/model/model.pt")
        model.eval()
    return model

@app.post("/predict")
async def predict(data: dict):
    model = get_model()
    # Your inference logic here
    return {"prediction": result}

# Lambda handler
handler = Mangum(app)
```

### Step 4: Deploy

```bash
cd deployment/

# Bootstrap CDK (first time only)
cdk bootstrap

# Deploy
cdk deploy

# Output:
# ✅ LambdaStack
# Outputs:
# LambdaStack.ApiEndpoint = https://xxx.execute-api.us-east-1.amazonaws.com/prod/
```

## Deploying to TorchServe

### Step 1: Generate TorchServe Files

```bash
mlops-agent "Deploy to TorchServe with:
- Model name: my-classifier
- Version: 1.0
- Batch size: 32
- Response timeout: 60s"
```

### Step 2: Custom Handler

Edit `handler.py`:

```python
from ts.torch_handler.base_handler import BaseHandler
import torch
import json

class CustomHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")

        # Load your model
        self.model = torch.load(f"{model_dir}/model.pt")
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        # Transform input data
        inputs = []
        for row in data:
            input_data = row.get("data") or row.get("body")
            # Your preprocessing here
            inputs.append(preprocessed)
        return torch.stack(inputs)

    def inference(self, data):
        with torch.no_grad():
            return self.model(data)

    def postprocess(self, data):
        return data.tolist()
```

### Step 3: Create MAR Archive

```bash
torch-model-archiver \
  --model-name my-classifier \
  --version 1.0 \
  --model-file model.py \
  --serialized-file models/model.pt \
  --handler handler.py \
  --export-path model_store/
```

### Step 4: Start TorchServe

```bash
torchserve --start \
  --model-store model_store \
  --models my-classifier=my-classifier.mar \
  --ts-config config.properties
```

### Step 5: Test

```bash
curl -X POST http://localhost:8080/predictions/my-classifier \
  -H "Content-Type: application/json" \
  -d '{"data": [1,2,3,4,5]}'
```

## Deploying to KServe

### Prerequisites

- Kubernetes cluster
- KServe installed
- kubectl configured

### Step 1: Generate KServe Manifest

```bash
mlops-agent "Deploy to KServe with:
- Runtime: pytorch
- GPU: 1
- Min replicas: 1
- Max replicas: 5"
```

### Step 2: Custom Predictor

Create `predictor.py`:

```python
import kserve
from typing import Dict
import torch

class MyPredictor(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False

    def load(self):
        self.model = torch.load("/mnt/models/model.pt")
        self.model.eval()
        self.ready = True

    def predict(self, payload: Dict, headers: Dict = None) -> Dict:
        inputs = payload["instances"]
        tensor = torch.tensor(inputs)

        with torch.no_grad():
            outputs = self.model(tensor)

        return {"predictions": outputs.tolist()}

if __name__ == "__main__":
    model = MyPredictor("my-model")
    model.load()
    kserve.ModelServer().start([model])
```

### Step 3: Build and Push Image

```bash
docker build -t your-registry/my-model:v1 .
docker push your-registry/my-model:v1
```

### Step 4: Deploy to Kubernetes

```yaml
# inference_service.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-classifier
spec:
  predictor:
    containers:
      - name: kserve-container
        image: your-registry/my-model:v1
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 4Gi
          requests:
            memory: 2Gi
```

```bash
kubectl apply -f inference_service.yaml
```

### Step 5: Test

```bash
# Get the URL
kubectl get inferenceservice my-classifier

# Send request
curl -X POST https://<url>/v1/models/my-classifier:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [[1,2,3,4,5]]}'
```

## Multi-Model Serving

### Using TorchServe

```bash
# Register multiple models
torchserve --start \
  --model-store model_store \
  --models classifier=classifier.mar sentiment=sentiment.mar
```

### Using KServe

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: multi-model
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      storageUri: "s3://bucket/models/"
      resources:
        limits:
          memory: 8Gi
```

## A/B Testing with KServe

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-classifier
spec:
  predictor:
    canaryTrafficPercent: 10
    containers:
      - name: kserve-container
        image: your-registry/my-model:v2
```

This routes 10% of traffic to the canary (v2) and 90% to stable (v1).

## Monitoring Deployments

### Prometheus Metrics

All deployment targets expose Prometheus metrics:

```bash
# TorchServe
curl http://localhost:8082/metrics

# LitServe
curl http://localhost:8000/metrics

# KServe
# Metrics are scraped automatically by Prometheus
```

### Key Metrics to Monitor

- `request_latency_seconds` - Request latency histogram
- `requests_total` - Total requests counter
- `model_load_time_seconds` - Model loading time
- `gpu_memory_used_bytes` - GPU memory usage
