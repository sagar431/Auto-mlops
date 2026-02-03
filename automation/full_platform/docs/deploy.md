# Deployment Guide (Artifacts Only)

This package generates code and manifests but does not run infrastructure. Follow these steps on your own machine/EC2.

## 1. Dataset Setup (Intel + Fruits)

```bash
cd automation/full_platform
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Kaggle credentials required in ~/.kaggle/kaggle.json
bash scripts/download_datasets.sh
python scripts/prepare_splits.py --dataset intel
python scripts/prepare_splits.py --dataset fruits
```

## 2. Training + Export

```bash
# Train a baseline TIMM model
python training/train_timm.py \
  --train-dir data/intel/train \
  --val-dir data/intel/val \
  --output-dir models

# Export TorchScript
python training/export_torchscript.py \
  --model resnet18 \
  --weights models/best_model.pt \
  --num-classes 6 \
  --output models/model.ts
```

> Update `--num-classes` based on your dataset class count.

## 3. Local Test with Docker Compose

```bash
docker compose up --build
```

- Model API: `http://localhost:8000/docs`
- Web UI: `http://localhost:8080`

## 3b. Push Images to ECR (Optional)

```bash
export AWS_REGION=us-east-1
export AWS_PROFILE=default
export TAG=latest
bash scripts/ecr_push.sh
```

## 4. Minikube Deployment (K8s Manifests)

```bash
minikube start
minikube addons enable ingress
kubectl apply -k k8s/
```

Copy your model artifacts into the PVC:

```bash
POD=$(kubectl -n auto-mlops get pod -l app=model-api -o jsonpath='{.items[0].metadata.name}')
kubectl -n auto-mlops cp models/model.ts "$POD":/app/models/model.ts
kubectl -n auto-mlops cp models/classes.txt "$POD":/app/models/classes.txt
```

Check status:

```bash
kubectl -n auto-mlops get pods,svc,ingress
```

If you want to access the Ingress on Minikube:

```bash
minikube tunnel
sudo sh -c 'echo \"127.0.0.1 auto-mlops.local\" >> /etc/hosts'
```

## 5. Helm Deployment

```bash
helm upgrade --install auto-mlops ./helm \
  --namespace auto-mlops --create-namespace
```

## 6. ArgoCD GitOps

1. Commit the repo with `automation/full_platform/helm`.
2. Update `argo/app.yaml` with your repo URL.
3. Apply it:

```bash
kubectl apply -f argo/app.yaml
```

## 7. Capture Required Outputs

Use the template in `reports/cluster_outputs.md` and run:

```bash
kubectl -n auto-mlops describe deploy/model-api
kubectl -n auto-mlops describe pod -l app=model-api
kubectl -n auto-mlops describe ingress auto-mlops
kubectl -n auto-mlops top pod
kubectl top node
kubectl get all -A -o yaml
```

Or generate the report automatically:

```bash
bash scripts/collect_k8s_outputs.sh
```
