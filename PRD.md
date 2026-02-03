# MLOps Deployment Agent — PRD

## Problem & Goals
The current agent can scaffold and execute many MLOps tasks but lacks end‑to‑end deployment governance, Kubernetes depth, and AWS automation. The goal is to make a reliable, human‑approved deployment agent that can build, deploy, and monitor ML services across Docker, EKS, and Lambda.

## Current Capabilities
The repo already provides a custom agent loop with planning/execution graphs, session persistence, and many deployment tool generators (LitServe, Gradio, TorchServe, KServe, FastAPI+Lambda). It also includes DVC tooling, MLflow integration, and Prometheus‑format metrics.

## Gap Analysis
Key gaps include:
1. Human‑in‑loop approval gates before execution.
2. Kubernetes primitives (Deployment, Service, Ingress, HPA, ConfigMap/Secret, Helm, canary, rollback).
3. AWS platform automation (EKS/ECR management, IAM policy tooling, cost estimates).
4. Docker hardening (multi‑stage builds, security scans, auto GPU detection).
5. Observability completeness (/ready endpoint, log sinks, alert policies as deployable CRDs).

## Roadmap
### Phase 1 — Safety & Reliability
1. Add approval checkpoints in the agent before any build/deploy step.
2. Add structured rollback hooks for each deploy target.
3. Expand state persistence to store plan checkpoints and recovery points.

### Phase 2 — Deployment Depth
1. Implement Kubernetes generators for Deployment, Service, Ingress, HPA, ConfigMap, and Secret.
2. Add EKS cluster discovery and kubeconfig setup.
3. Implement ECR repo creation/login and image tagging conventions.

### Phase 3 — Observability & Governance
1. Add /ready endpoint and standardized health contracts.
2. Add log sink configs (CloudWatch or ELK) per deploy target.
3. Add cost estimation and policy checks before approval.

## Non‑Functional Requirements
- No silent deploys: every deploy is gated by human approval.
- Deterministic output: tool outputs are reproducible and logged.
- Secrets never written to repo; use environment or secret stores.
- Fail‑safe behavior with clear rollback paths.

## Risks & Mitigations
- Credential leakage risk: enforce secret redaction and validation.
- Tool sprawl: maintain a minimal, stable tool surface.
- Runtime drift: enforce version pinning and manifest checks.

## Success Metrics
- Checklist coverage above 85%.
- Mean time to safe deploy under 10 minutes.
- Rollback execution under 2 minutes.
