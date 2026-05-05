# Phase 4 Implementation Issues

## Status

Planned. Phase 0, Phase 1, Phase 2, and Phase 3 are expected to be complete before this phase begins. Phase 4 turns the deferred capstone data stage into issue-sized work for `prepare_capstone_data`, a first-class **Workflow Template** for reproducible capstone data lineage.

`PRD.md` remains the source of truth and is intentionally not edited by these issues. This issue track narrows the next capstone slice around the EMLO requirement for DVC and S3 data versioning before Docker, CI/CD, frontend, report, demo deployment capabilities, or production deployment work.

## Goal

Phase 4 adds the **Capstone Data Automation Workflow** for two user-provided **Canonical Image Folder Datasets**. The workflow detects dataset layout, records deterministic split evidence, creates split manifests, DVC-tracks the **Capstone Data Package**, validates local or S3 DVC remotes, gates DVC push/pull behind approvals, and writes `.auto_mlops/capstone/data_stage_evidence.json` for `build_capstone_pipeline`.

The goal is not broad dataset automation. The goal is one reliable path where Auto-MLOps can prove exactly which data state is ready for training and capstone reporting without relying on prose, latest-run lookup, hidden session state, or unsafe cloud credential handling.

## Issues

1. [Register `prepare_capstone_data`](./0001-register-prepare-capstone-data.md)
2. [Detect Two Image-Folder Datasets](./0002-detect-two-image-folder-datasets.md)
3. [Generate Split Manifests](./0003-generate-split-manifests.md)
4. [Create And DVC-Track Capstone Data Package](./0004-dvc-track-capstone-data-package.md)
5. [Configure And Validate DVC Remote](./0005-configure-and-validate-dvc-remote.md)
6. [Approval-Gated DVC Push/Pull](./0006-approval-gated-dvc-push-pull.md)
7. [Data Stage Evidence And Orchestrator Handoff](./0007-data-stage-evidence-and-orchestrator-handoff.md)

## Dependency Order

- 0001 has no blockers and establishes the `prepare_capstone_data` **Workflow Template**, **Workflow Inputs**, routing, `completion_mode`, and contract branches.
- 0002 depends on 0001 because dataset detection must produce evidence against the registered workflow inputs.
- 0003 depends on 0002 because split manifests must be generated only for supported detected datasets.
- 0004 depends on 0003 because the **Capstone Data Package** should DVC-track split manifests and any materialized split folders.
- 0005 depends on 0004 because remote validation must point at the tracked capstone data package.
- 0006 depends on 0005 because DVC push/pull should run only after remote validation and approval checks are in place.
- 0007 depends on 0001 through 0006 because `data_stage_evidence.json` and `build_capstone_pipeline` must reflect completed, blocked, and deferred data-stage evidence from the implemented slices.

## Phase 4 Boundaries

Phase 4 includes:

- Adding `prepare_capstone_data` as a real **Workflow Template** with **Success Contract** branches for `local_ready` and `capstone_complete`.
- Requiring `dataset_1_path`, `dataset_2_path`, and `completion_mode` as declared **Workflow Inputs**.
- Supporting exactly two user-provided local or mounted **Canonical Image Folder Datasets** for capstone-complete success.
- Detecting existing train/test folders with class-labelled subdirectories.
- Creating deterministic **Split Manifests** with fixed seed, split ratio, and per-class counts.
- Materializing train/test folders only when downstream training requires physical split folders and only after an **Approval Gate**.
- DVC-tracking `data/capstone/<dataset_id>/` as the **Capstone Data Package**, not raw source datasets by default.
- Supporting local DVC remotes for **Local Data Readiness** and S3 DVC remotes for **Capstone Data Completeness**.
- Recording **Credential Capability Evidence** with redacted AWS identity, bucket, prefix, DVC remote resolution, and harmless access probes where possible.
- Requiring **Approval Gates** before DVC push or pull.
- Writing `.auto_mlops/capstone/data_stage_evidence.json`.
- Recording split manifests, DVC metadata, remote state, transfer state, and the data-stage evidence file in the **Artifact Manifest**.
- Updating `build_capstone_pipeline` to reference data-stage state without pretending deferred capstone capabilities are complete.

Phase 4 excludes:

- Dataset downloads from Kaggle, Hugging Face, URLs, or course pages.
- Arbitrary dataset formats beyond canonical image-folder datasets.
- Data cleaning, augmentation, deduplication, or quality scoring beyond layout and split counts.
- Training changes beyond optional contract-facing handoff evidence.
- HPO, learning-rate finder, and model selection changes.
- Docker, registry push, and CI/CD.
- Kubernetes, KServe, Helm, and ArgoCD.
- AWS Lambda serverless deployment.
- HuggingFace Spaces packaging or publishing.
- Stress tests.
- Frontend, report, or demo deployment capabilities.
- Final report or video generation.
- Automatic S3 transfer without approval.
- Secrets storage or `.env` mutation for AWS keys.
- Moving, deleting, or mutating source dataset folders without explicit approval.

## How Phase 4 Connects To The EMLO Capstone

The EMLO capstone requires two selected datasets, train/test splits, DVC versioning, and S3 storage before training, model optimization, containerization, deployment, stress testing, frontend, CI/CD, and final report evidence can claim end-to-end completion.

Phase 3 provides training and model artifact selection. Phase 4 provides the missing data lineage layer so later stages can point to the exact dataset state that produced a model. `build_capstone_pipeline` should use the durable data-stage evidence artifact as the handoff, not a latest-run query.

## Verification Commands

- `pytest tests/unit/test_workflow_registry.py`
- `pytest tests/unit/test_agent_loop.py`
- `pytest tests/e2e/`
- `ruff check .`
