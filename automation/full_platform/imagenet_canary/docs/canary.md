# ImageNet Canary Deployment (Artifact-Only)

This package deploys a single ImageNet API via Argo Rollouts and shifts traffic between model revisions (30% → 100%).

## 1. Build the Image

```bash
cd automation/full_platform/imagenet_canary

docker build -t auto-mlops/imagenet-api:latest services/imagenet_api
```

Push to your registry and update the image in `k8s/rollout.yaml` if needed.

## 2. Install Controllers

```bash
# Argo Rollouts
kubectl create namespace argo-rollouts || true
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# NGINX Ingress (minikube)
minikube addons enable ingress
```

## 3. Deploy the Stack

```bash
kubectl apply -k k8s/
```

## 4. Canary Steps (3 ImageNet Models)

Use any three ImageNet classifiers from HuggingFace. Examples:

- `google/vit-base-patch16-224`
- `microsoft/resnet-50`
- `facebook/convnext-base-224`

### Step A: Model 1 (1 replica)

```bash
kubectl -n auto-mlops-canary scale rollout/imagenet-rollout --replicas=1
kubectl -n auto-mlops-canary set env rollout/imagenet-rollout HF_MODEL_ID=google/vit-base-patch16-224
```

### Step B: Model 2 (30% traffic)

```bash
kubectl -n auto-mlops-canary set env rollout/imagenet-rollout HF_MODEL_ID=microsoft/resnet-50
kubectl argo rollouts get rollout imagenet-rollout -n auto-mlops-canary
```

Wait until the rollout is at 30% canary. Capture Grafana screenshots.

### Step C: Model 2 (100% traffic)

The rollout will proceed to 100% after the pause duration. You can also promote:

```bash
kubectl argo rollouts promote imagenet-rollout -n auto-mlops-canary
```

### Step D: Model 2 (2 replicas)

```bash
kubectl -n auto-mlops-canary scale rollout/imagenet-rollout --replicas=2
```

### Step E: Model 3 (30% → 100%)

```bash
kubectl -n auto-mlops-canary set env rollout/imagenet-rollout HF_MODEL_ID=facebook/convnext-base-224
kubectl argo rollouts get rollout imagenet-rollout -n auto-mlops-canary
```

## 5. Grafana

Import `grafana/imagenet_dashboard.json` into Grafana. The dashboard uses:
- `inference_requests_total`
- `inference_latency_seconds_bucket`

## 6. Load Test

Use the shared load generator:

```bash
python ../scripts/send.py --url http://imagenet.local/api/predict --images /path/to/images
```

> Update `/etc/hosts` or use `minikube tunnel` to reach `imagenet.local`.
