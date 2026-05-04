# Phase 3 Implementation Issues

## Status

Planned. Phase 0, Phase 1, and Phase 2 are complete on `main`. Phase 3 turns the PRD's Train And Improve Loop into issue-sized work for `train_and_track`, `train_until_better`, and the first `build_capstone_pipeline` Capstone Orchestrator skeleton.

`PRD.md` remains the source of truth and is intentionally not edited by these issues.

## Goal

Phase 3 adds controlled training automation for Hydra/PyTorch/TIMM image-classifier projects shaped like the EMLO `mlops_pratice/session-06` reference: Hydra configs, DVC-tracked data, a train entrypoint, tests, MLflow tracking, and explicit checkpoint or model artifact selection.

The goal is not broad training support. The goal is one reliable path where Auto-MLOps can detect a supported training project, run bounded training, capture metrics and artifacts, log them to MLflow, select a deployable model artifact, and expose those capabilities to the Capstone Orchestrator without pretending deferred capstone capabilities are complete.

## Issues

1. [Detect Training Project And Entrypoint](./0001-detect-training-project-and-entrypoint.md)
2. [Run Training And Capture Metrics](./0002-run-training-and-capture-metrics.md)
3. [Track Training In MLflow](./0003-track-training-in-mlflow.md)
4. [Select Best Model Artifact](./0004-select-best-model-artifact.md)
5. [Capstone Orchestrator Skeleton](./0005-capstone-orchestrator-skeleton.md)

## Dependency Order

- 0001 has no blockers and establishes supported-project detection for session-06-style training repos.
- 0002 depends on 0001 because training must not run until the entrypoint, config system, data path, and budget controls are known.
- 0003 depends on 0002 because MLflow tracking needs captured params, metrics, logs, duration, and artifacts from a bounded training run.
- 0004 depends on 0002 and 0003 because model selection compares the latest run against a baseline using logged metrics and artifacts.
- 0005 depends on 0001 through 0004 because `build_capstone_pipeline` must invoke only implemented sub-workflows and record missing future capabilities as blocked or deferred evidence.

## Phase 3 Boundaries

Phase 3 includes:

- Detecting Hydra/PyTorch/TIMM image-classifier training projects shaped like `mlops_pratice/session-06`.
- Adding `train_and_track` as a real **Workflow Template** with a **Success Contract**.
- Adding `train_until_better` as a controlled improvement workflow with fixed budgets, Hydra overrides, metric comparison, and keep/discard decisions.
- Running bounded local training commands with explicit timeout, epoch, device, and dataset-size controls.
- Capturing params, metrics, logs, duration, checkpoints, and artifact paths as structured **Verification Results** and **Artifact Manifest** entries.
- Logging training params, metrics, artifacts, and selected checkpoints to MLflow.
- Selecting a deployable model artifact for downstream deployment workflows.
- Adding the `build_capstone_pipeline` Capstone Orchestrator skeleton as a top-level workflow id.
- Recording blocked or deferred evidence for unimplemented future capstone capabilities instead of inventing success.

Phase 3 excludes:

- Broad support for arbitrary ML frameworks beyond the narrow Hydra/PyTorch/TIMM image-classifier path.
- Unbounded hyperparameter search or autonomous project rewrites.
- Running expensive training without explicit budget controls and approval where needed.
- Editing `PRD.md`.
- Implementing frontend, cloud, Kubernetes, or report/video features.

## How Phase 3 Connects To The EMLO Capstone

The EMLO capstone requires an end-to-end MLOps pipeline: reproducible training, DVC-managed data, experiment tracking, model selection, deployment, monitoring, and a final report. Phase 3 covers the training and experiment-tracking core of that path.

The Capstone Orchestrator uses `build_capstone_pipeline` as the top-level coordinator. At first it should coordinate only implemented sub-workflows such as `setup_pipeline`, `train_and_track`, `train_until_better`, and the Phase 2 LitServe deployment path when their contracts are satisfied. Missing capstone pieces must appear as explicit blocked or deferred evidence in the workflow output.

## Deferred To Later Phases

- S3 DVC remote automation.
- KServe, Helm, and ArgoCD production deployment.
- HuggingFace Spaces packaging and publishing.
- AWS Lambda serverless deployment.
- Stress tests and load evidence.
- Frontend workflow timeline and endpoint cards.
- Final capstone report and video generation.
