# Deployment Targets

Auto-MLOps supports 5 deployment targets for serving your trained models. Each target is optimized for different use cases.

## Overview

| Target | Best For | Throughput | Setup Complexity |
|--------|----------|------------|------------------|
| **Gradio** | Demos, prototypes | Low-Medium | Very Easy |
| **LitServe** | High-throughput inference | High | Easy |
| **AWS Lambda** | Serverless, pay-per-use | Medium | Medium |
| **TorchServe** | Enterprise production | High | Medium |
| **KServe** | Kubernetes-native | High | Complex |

## Gradio

Best for quick demos and prototypes. Instant sharing via Hugging Face Spaces.

### Features
- Interactive web UI with zero coding
- One-click deployment to Hugging Face Spaces
- Built-in examples and documentation
- Great for stakeholder demos

### Usage

```bash
mlops-agent deploy gradio --model ./models/best_model.pt
```

Or via natural language:

```bash
mlops-agent "Deploy my model to Gradio with image classification interface"
```

### Generated Files
- `app.py` - Gradio application
- `requirements.txt` - Dependencies

## LitServe

Best for high-throughput inference with batching and GPU autoscaling.

### Features
- Automatic request batching
- GPU autoscaling
- Streaming responses
- 1000+ requests/second

### Usage

```bash
mlops-agent deploy litserve --model ./models/best_model.pt
```

### Generated Files
- `server.py` - LitServe API server
- `Dockerfile` - Container image
- `requirements.txt` - Dependencies

### Configuration Options
```python
# configs/litserve.yaml
max_batch_size: 32
timeout: 30
workers: 4
accelerator: gpu
```

## AWS Lambda

Best for serverless, pay-per-use deployments with variable traffic.

### Features
- Pay only for what you use
- Automatic scaling to zero
- AWS CDK infrastructure as code
- API Gateway integration

### Usage

```bash
mlops-agent deploy lambda --model ./models/best_model.pt
```

### Generated Files
- `app.py` - FastAPI application
- `Dockerfile` - Lambda container image
- `cdk.py` - AWS CDK stack
- `requirements.txt` - Dependencies

### Deployment Steps
```bash
# Deploy to AWS
cd deployment/
cdk bootstrap
cdk deploy
```

### Requirements
- AWS CLI configured
- AWS CDK installed
- Docker for building images

## TorchServe

Best for enterprise production with model versioning and hot-swap.

### Features
- Model versioning
- A/B testing
- Metrics and logging
- Hot model updates

### Usage

```bash
mlops-agent deploy torchserve --model ./models/best_model.pt
```

### Generated Files
- `handler.py` - Custom handler
- `model.mar` - Model archive
- `config.properties` - TorchServe config

### Deployment Steps
```bash
# Create MAR archive
torch-model-archiver --model-name my_model \
  --version 1.0 \
  --handler handler.py \
  --serialized-file model.pt

# Start TorchServe
torchserve --start --model-store model_store --models my_model=my_model.mar
```

## KServe

Best for Kubernetes-native deployments with auto-scaling.

### Features
- Kubernetes-native
- Canary deployments
- Traffic splitting
- GPU support

### Usage

```bash
mlops-agent deploy kserve --model ./models/best_model.pt
```

### Generated Files
- `inference_service.yaml` - KServe manifest
- `Dockerfile` - Container image
- `predictor.py` - Prediction code

### Deployment Steps
```bash
# Apply to Kubernetes
kubectl apply -f inference_service.yaml

# Check status
kubectl get inferenceservice my-model
```

### Requirements
- Kubernetes cluster
- KServe installed
- kubectl configured

## Comparison Matrix

| Feature | Gradio | LitServe | Lambda | TorchServe | KServe |
|---------|--------|----------|--------|------------|--------|
| GPU Support | ✓ | ✓ | ✗ | ✓ | ✓ |
| Auto-scaling | ✗ | ✓ | ✓ | ✗ | ✓ |
| Batching | ✗ | ✓ | ✗ | ✓ | ✓ |
| Versioning | ✗ | ✗ | ✗ | ✓ | ✓ |
| Zero Cost Idle | ✗ | ✗ | ✓ | ✗ | ✗ |
| Setup Time | 1 min | 5 min | 15 min | 10 min | 30 min |

## Choosing the Right Target

### Use Gradio when:
- Building demos for stakeholders
- Rapid prototyping
- Need a web UI quickly

### Use LitServe when:
- High request volume
- Need GPU acceleration
- Latency-sensitive applications

### Use Lambda when:
- Variable/unpredictable traffic
- Cost optimization is priority
- CPU-only inference is acceptable

### Use TorchServe when:
- Enterprise production requirements
- Need model versioning
- A/B testing needed

### Use KServe when:
- Already using Kubernetes
- Need advanced traffic management
- Multi-model serving
