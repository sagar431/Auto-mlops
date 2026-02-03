# ImageNet Canary Pack

Artifact-only canary deployment for three HuggingFace ImageNet classifiers using Argo Rollouts.

## Contents
- `services/imagenet_api/` FastAPI inference server with Redis caching and Prometheus metrics
- `k8s/` Rollout, services, ingress, analysis template
- `argo/` ArgoCD application
- `grafana/` Dashboard JSON
- `docs/` Step-by-step canary instructions

## Start Here
Read `docs/canary.md`.
