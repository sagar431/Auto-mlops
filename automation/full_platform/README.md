# Full Platform Automation (Artifacts + Scripts)

This package generates all code, configs, and deployment artifacts for a full MLOps workflow using Intel Image Classification + Fruits/Veg datasets. It does **not** execute infrastructure steps. You run the scripts manually on your own AWS/EC2/Minikube environment.

## What You Get
- Dataset download + split scripts (Kaggle-based)
- DVC + S3 scaffolding
- Training (TIMM + HuggingFace) + HPO + LR finder scripts
- FastAPI model API with Redis caching
- Simple web UI (FastAPI + Jinja2)
- Dockerfiles + docker-compose
- ECR push helper (`scripts/ecr_push.sh`)
- Kubernetes manifests
- Helm chart
- ArgoCD Application YAML
- Load testing helper (`scripts/send.py`)
- Kubectl capture helper (`scripts/collect_k8s_outputs.sh`)
- Reporting template for kubectl outputs
- ImageNet canary pack (Argo Rollouts + Grafana dashboard)

## Quick Start (Artifacts Only)
1. Create and activate a Python env.
2. Install deps from `automation/full_platform/requirements.txt`.
3. Configure Kaggle API (`~/.kaggle/kaggle.json`).
4. Run `scripts/download_datasets.sh` to download the datasets.
5. Run `scripts/prepare_splits.py` to create train/val/test splits.
6. Run `training/train_timm.py` or `training/train_hf.py`.
7. Export model to TorchScript using `training/export_torchscript.py`.
8. Test locally with `docker-compose`.
9. Generate load with `scripts/send.py`.
10. Deploy to K8s with `k8s/` manifests or `helm/` chart.
11. Use `argo/app.yaml` for GitOps with ArgoCD.

## Folder Map
- `configs/` - dataset + training defaults
- `scripts/` - dataset, DVC, deploy, report helpers
- `training/` - training, HPO, LR finder, export
- `services/` - model API + web UI
- `models/` - TorchScript artifacts (`model.ts`, `classes.txt`)
- `docker-compose.yml` - local stack
- `k8s/` - Kubernetes manifests
- `helm/` - Helm chart
- `argo/` - ArgoCD Application YAML
- `docs/` - manual instructions
- `reports/` - output templates
- `imagenet_canary/` - Argo Rollouts canary deployment for 3 ImageNet models

## Notes
- Update `configs/datasets.yaml` to match your exact dataset sources.
- Configure S3 + DVC using `scripts/dvc_init.sh`.
- Update ArgoCD `repoURL` in `argo/app.yaml`.
