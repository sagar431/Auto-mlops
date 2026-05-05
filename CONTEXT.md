# Auto-MLOps Product Context

Auto-MLOps is a production-like MLOps agent that turns natural-language requests into verified workflow outcomes. This context defines the product language used to distinguish deterministic workflow execution from open-ended agent planning.

## Language

**Workflow Registry**:
A code-owned catalog of supported workflows.
_Avoid_: tool list, prompt pattern, dynamic plan catalog

**Workflow Template**:
An ordered skeleton for one supported workflow.
_Avoid_: generated plan, ad hoc workflow, LLM-authored skeleton

**Workflow Step**:
One named action in a **Workflow Template**.
_Avoid_: arbitrary action, prompt step

**Optional Workflow Step**:
A declared **Workflow Step** that runs only when its registry-owned inclusion rule is satisfied.
_Avoid_: ad hoc step, LLM-added step

**Workflow Branch**:
A registry-owned alternative path within a **Workflow Template** selected by explicit rules.
_Avoid_: arbitrary tool chain, prompt-only path

**General GPU Inference Workflow**:
A **Workflow Template** that selects among declared serving backend branches for GPU inference requests.
_Avoid_: LitServe-only deployment, backend-specific workflow

**LitServe GPU Workflow**:
A **Workflow Template** for the stricter Lambda or local GPU LitServe deployment path.
_Avoid_: generic GPU deployment, AWS Lambda deployment

**Routing Alias**:
A user-facing phrase that may select a **Workflow Template**.
_Avoid_: prompt hint, keyword only

**Negative Routing Rule**:
A declared phrase or condition that must not select a **Workflow Template**.
_Avoid_: hidden prompt caveat, informal exception

**Workflow Selection**:
A structured routing decision that identifies a **Workflow Template** and explains the matched and rejected routing evidence.
_Avoid_: workflow id only, prompt choice

**Prompt Contract Test**:
A test that asserts natural-language routing produces the expected **Workflow Selection** and registry-owned obligations.
_Avoid_: prompt wording test, LLM-only test

**Fake Template**:
A named **Workflow Template** without testable steps and contract detail.
_Avoid_: placeholder workflow, reserved workflow name

**Tool Function**:
A concrete executable implementation that may be invoked by a **Workflow Step**.
_Avoid_: workflow step, contract check

**Workflow Input**:
A required or optional value declared by a **Workflow Template** before step arguments are expanded.
_Avoid_: implicit arg, prompt variable, guessed parameter

**Success Contract**:
The structured set of named checks and typed evidence fields that must be satisfied before a workflow can be marked successful.
_Avoid_: done criteria, completion note, summary claim

**Conditional Contract Check**:
A **Success Contract** check that is required only when its registry-owned condition applies.
_Avoid_: LLM-waived check, ad hoc exception

**Contract Failure**:
A structured record of missing evidence or failed checks for a **Success Contract**.
_Avoid_: generic error message, vague failure summary

**Approval Gate**:
A required human approval before a risky workflow step may run.
_Avoid_: warning, confirmation text

**Approval Record**:
An auditable record of an approval or denial for an **Approval Gate**.
_Avoid_: transient confirmation, chat acknowledgement

**Risk Category**:
A declared reason that a **Workflow Step** requires an **Approval Gate**.
_Avoid_: vague risk, generic approval

**Remote Service Credential Use**:
A **Risk Category** for workflow steps that use authenticated non-AWS remote services such as GitHub, GHCR, DockerHub, or other registry APIs without storing or exposing tokens.
_Avoid_: generic cloud credentials, plaintext token use, unaudited remote auth

**Verification Result**:
Evidence that a **Success Contract** check passed or failed.
_Avoid_: status text, success message

**Observed Evidence**:
Verification evidence captured from an executed check or inspected runtime state.
_Avoid_: planned command, generated instruction

**Declared Evidence**:
Verification evidence that records generated instructions, expected commands, or planned configuration without observing runtime behavior.
_Avoid_: observed result, executed check

**Artifact Manifest**:
A workflow-level record of files or external artifacts produced, validated, or selected by a workflow run.
_Avoid_: generated-files list, summary appendix

**Capstone Data Automation Workflow**:
A **Workflow Template** that detects dataset layout, validates DVC tracking, safely configures local or S3 remotes, and records dataset lineage evidence for the capstone path.
_Avoid_: generic data prep, ad hoc DVC setup, frontend/report/demo deployment capabilities

**Capstone Submission Path**:
The EMLO assignment path that proves an end-to-end MLOps pipeline across data versioning, training, containerization, deployment, demo, stress testing, frontend, CI/CD, and final evidence.
_Avoid_: demo only, deployment only, generic project checklist

**North-Star Capstone Workflow**:
The current concrete product journey used to harden Auto-MLOps around verified end-to-end evidence for the EMLO capstone path, without making capstone automation the whole product identity.
_Avoid_: whole product scope, generic MLOps support, capstone-only product

**Dataset Lineage Evidence**:
Observed or declared evidence that identifies which dataset path, DVC metadata, remote configuration, checksum, and pull or push state produced a training run.
_Avoid_: data note, dataset summary, data prep status

**Canonical Image Folder Dataset**:
A supported capstone dataset layout whose train and test image folders can be detected with class-labelled subdirectories.
_Avoid_: arbitrary image dataset, custom data loader support, downloaded dataset

**User-Provided Dataset Path**:
A local or mounted dataset path supplied by the user before **Capstone Data Automation Workflow** execution.
_Avoid_: dataset URL, auto-download source, inferred external dataset

**Deterministic Dataset Split**:
A reproducible train/test split created from a declared ratio and fixed seed while preserving class-labelled folder structure.
_Avoid_: random split, manual split note, implicit validation split

**Split Manifest**:
A generated artifact that records which source dataset files belong to each split without changing the source dataset layout.
_Avoid_: copied dataset, moved files, split note

**Capstone Data Package**:
The DVC-tracked Phase 4 output under `data/capstone/<dataset_id>/` that contains split manifests and any materialized train/test folders required by downstream training.
_Avoid_: raw dataset mirror, whole source dataset, arbitrary data folder

**Capstone Data Remote**:
A DVC remote used by the **Capstone Data Automation Workflow** to store the **Capstone Data Package**, with S3 required for capstone-complete evidence.
_Avoid_: generic remote, storage path, sync target

**Credential Capability Evidence**:
Redacted observed evidence that required cloud credentials can perform a specific operation without exposing secret values.
_Avoid_: secret dump, access key evidence, `.env` proof

**Approval-Gated Data Transfer**:
A DVC push or pull operation that may write to remote storage, download data, incur cost, or modify local files, and therefore requires an **Approval Gate**.
_Avoid_: automatic sync, background upload, safe validation

**Read-Only Data Inspection**:
Phase 4 dataset, DVC, or remote inspection that observes local state without writing files, changing DVC configuration, transferring data, or using cloud credentials.
_Avoid_: validation that mutates state, silent setup, implicit sync

**Partial Data Stage Completion**:
A blocked **Workflow Status** for the **Capstone Data Automation Workflow** that preserves completed local lineage evidence while identifying missing S3 transfer evidence.
_Avoid_: data success, soft success, warning-only completion

**Local Data Readiness**:
A Phase 4 workflow mode where local dataset paths, supported layouts, split evidence, DVC tracking, and durable data-stage evidence are sufficient for success.
_Avoid_: capstone completion, S3-ready, production data readiness

**Capstone Data Completeness**:
A Phase 4 workflow mode where **Local Data Readiness** plus S3 remote validation and approved S3 transfer evidence are required for success.
_Avoid_: local success, development-ready, partial capstone data

**Completion Mode**:
The stage-specific workflow input that selects which success contract branch a capstone workflow must satisfy.
_Avoid_: mode, target_completion, success type

**Data Completion Mode**:
The `prepare_capstone_data` **Completion Mode** value set, either `local_ready` or `capstone_complete`, that selects whether Phase 4 validates **Local Data Readiness** or **Capstone Data Completeness**.
_Avoid_: generic local readiness, container readiness

**Container CI Completion Mode**:
The `prepare_capstone_container_ci` **Completion Mode** value set, either `container_local_ready` or `container_capstone_complete`, that selects whether Phase 5 validates **Container Local Readiness** or **Container Capstone Completeness**.
_Avoid_: data readiness mode, generic capstone mode

**Container Build Evidence**:
Observed or declared evidence that records the generated or validated container build specification, dependency context, Docker availability, image build result, image tag, digest or local image id, and minimal smoke check when required by **Container CI Completion Mode**.
_Avoid_: Dockerfile note, deployment endpoint evidence, unverified image claim

**Capstone Runtime Image**:
The default Phase 5 container image specification for the **North-Star Capstone Workflow**, with recorded intended roles such as CI, training validation, inference validation, and later deployment handoff.
_Avoid_: separate training image, separate serving image, CUDA variant matrix, slim production image

**Container Registry Evidence**:
Observed or declared evidence that records the selected container registry target, redacted image reference, authentication capability, approval-gated registry login or push, pushed image reference, and digest when available.
_Avoid_: raw registry credentials, unredacted token, registry URL note, image build evidence

**Capstone CI Evidence**:
Observed or declared evidence that records generated GitHub Actions workflow files, workflow YAML validation, bounded repo-local commands, required capstone checks, secret-safe registry usage, selected image reference usage, and optional remote CI run status.
_Avoid_: CI note, generated YAML only, plaintext registry credentials, unbounded shell automation

**Data Stage Evidence Artifact**:
A durable capstone artifact at `.auto_mlops/capstone/data_stage_evidence.json` that records the exact dataset lineage state produced by **Capstone Data Automation Workflow**.
_Avoid_: latest run lookup, chat summary, hidden session state

**Capstone Container And CI Automation Workflow**:
A Phase 5 **Workflow Template** for the capstone path that produces Docker/container evidence, registry target evidence, approval-gated registry push evidence, GitHub Actions/CI evidence, and a durable container-and-CI handoff artifact.
_Avoid_: Kubernetes deployment, GitOps deployment, generic Docker setup, unverified CI note

**Container CI Stage Handoff**:
The orchestrator relationship where `prepare_capstone_container_ci` produces a **Container CI Evidence Artifact** and `build_capstone_pipeline` references that artifact without owning Dockerfile generation, image build, registry validation or push, or CI workflow generation.
_Avoid_: orchestrator-owned Docker setup, latest run lookup, hidden stage state

**Container CI Evidence Artifact**:
A durable capstone artifact at `.auto_mlops/capstone/container_ci_evidence.json` that records the container and CI/CD state produced by **Capstone Container And CI Automation Workflow**.
_Avoid_: CI summary, Docker note, registry push log only, hidden session state

**Container Local Readiness**:
A Phase 5 workflow mode where usable data-stage evidence, selected local model artifact evidence, generated container assets, and CI workflow validation are sufficient for local container/CI progress.
_Avoid_: capstone completion, pushed image readiness, Kubernetes readiness

**Container Capstone Completeness**:
A Phase 5 workflow mode where **Container Local Readiness** plus capstone-complete data evidence, MLflow-linked best artifact evidence, successful image build, validated registry target, approval-gated registry push evidence, and CI evidence are required for success.
_Avoid_: local container success, Dockerfile-only success, registry URL note

**Artifact State**:
A controlled state describing how an artifact relates to a workflow run.
_Avoid_: arbitrary artifact label, prose file status

**Deployment Report**:
A structured workflow output that records deployment evidence and contract status.
_Avoid_: markdown-only report, narrative summary

**Rollback Plan**:
A structured rollback command, script, or target reference produced by a deployment workflow.
_Avoid_: standalone rollback workflow, vague rollback note

**Workflow Runtime**:
The code-owned execution boundary that computes workflow status from **Verification Results**.
_Avoid_: summarizer, prompt, LLM judgment

**Workflow Status**:
A runtime-owned lifecycle state for a workflow run.
_Avoid_: arbitrary status string, prompt status

## Relationships

- A **Workflow Registry** contains one or more **Workflow Templates**.
- The Phase 0 **Workflow Registry** includes `setup_pipeline`, `deploy_litserve_gpu`, `deploy_gpu_inference`, `deploy_gradio_demo`, and `deploy_kserve_production`.
- The Phase 0 **Workflow Registry** excludes `rollback` and `monitor_and_alert` unless Phase 0 tests require them.
- The Phase 0 **Workflow Registry** excludes `train_and_track` and `train_until_better` until a Phase 0 test requires them.
- A **Workflow Template** contains one or more ordered **Workflow Steps**.
- A **Workflow Template** declares **Routing Aliases** and **Negative Routing Rules** when natural-language routing is required.
- A **Workflow Selection** includes the selected workflow id, confidence, matched aliases, rejected workflows, missing inputs, and selection reason.
- A **Prompt Contract Test** asserts expected **Workflow Selection**, required inputs, rejected workflows, and **Success Contract** obligations.
- A **Workflow Template** may contain **Optional Workflow Steps** only when their inclusion rules are declared in the registry.
- A **Workflow Template** may contain **Workflow Branches** only when valid branch names and selection rules are declared in the registry.
- `deploy_gpu_inference` is the **General GPU Inference Workflow** and uses declared backend **Workflow Branches**.
- `deploy_litserve_gpu` is the **LitServe GPU Workflow** and remains a first-class stricter path for Lambda or local GPU LitServe deployments.
- The prompt "Deploy this model on Lambda Labs GPU" selects `deploy_litserve_gpu` directly, not `deploy_gpu_inference` with a LitServe branch.
- `deploy_litserve_gpu` success requires **Observed Evidence** for GPU detection, server start, `/health`, and `/predict`, plus generated LitServe app evidence.
- If `deploy_litserve_gpu` cannot observe a GPU or observes no GPU, it ends `blocked` or `failed` with a clear reason and next action.
- A **Workflow Template** declares **Workflow Inputs** separately from **Workflow Step** arguments.
- A **Workflow Template** declares zero or more **Approval Gates** for risky **Workflow Steps**.
- An **Approval Gate** belongs to a **Workflow Step** and declares one or more **Risk Categories**.
- An **Approval Gate** must produce an **Approval Record** before the **Workflow Runtime** may execute its **Workflow Step**.
- A **Workflow Step** has stable workflow meaning and may bind to one or more allowed **Tool Functions**.
- A **Workflow Template** has exactly one **Success Contract**.
- A **Workflow Registry** must not include **Fake Templates**.
- A **Workflow Template** belongs in Phase 0 only when its routing and contract behavior can be tested.
- A **Success Contract** is satisfied by one or more **Verification Results**.
- A **Success Contract** check declares a required name, evidence type, source step, and pass condition.
- A **Conditional Contract Check** declares its condition in the registry and cannot be invented or waived by the LLM.
- A **Contract Failure** records missing evidence, failed checks, expected evidence type, source step, actual evidence when available, and next action.
- A **Verification Result** records whether its evidence is **Observed Evidence** or **Declared Evidence**.
- **Declared Evidence** cannot satisfy live-runtime checks such as health checks, prediction tests, GPU utilization, or latency.
- A **Success Contract** may require an **Artifact Manifest** when a **Workflow Template** creates, validates, or selects artifacts.
- An **Artifact Manifest** records artifact path or URI, artifact type, producing **Workflow Step**, checksum when available, and artifact state.
- A **Capstone Submission Path** requires dataset versioning with DVC and S3 before downstream deployment, demo, stress testing, frontend, CI/CD, and final report evidence can claim end-to-end completion.
- The **North-Star Capstone Workflow** is the first verified product journey for Auto-MLOps, but Auto-MLOps remains a broader MLOps agent product.
- `prepare_capstone_data` is the **Capstone Data Automation Workflow** for Phase 4.
- A **Capstone Data Automation Workflow** produces **Dataset Lineage Evidence** before downstream training, deployment, CI/CD, or reporting stages may claim reproducible capstone completion.
- A **Capstone Data Automation Workflow** operates only on **User-Provided Dataset Paths** in Phase 4 and does not download datasets.
- A **Capstone Data Automation Workflow** requires two selected **Canonical Image Folder Datasets** for the capstone path; unsupported dataset layouts block with explicit missing evidence.
- A **Capstone Data Automation Workflow** may create a **Deterministic Dataset Split** only after an **Approval Gate** records approval for project file writes.
- A **Deterministic Dataset Split** produces **Observed Evidence** for split ratio, seed, per-class counts, train path, and test path.
- A **Deterministic Dataset Split** produces a **Split Manifest** by default and copies files into train/test directories only when downstream training requires physical split folders.
- A **Capstone Data Automation Workflow** must not move original dataset files during Phase 4.
- A **Capstone Data Automation Workflow** DVC-tracks the **Capstone Data Package**, not the raw source dataset directories by default.
- A **Capstone Data Package** includes a DVC-tracked **Split Manifest** and includes DVC-tracked copied train/test directories only when physical split folders are required.
- A **Capstone Data Remote** may be local for development evidence, but S3 remote evidence is required before the **Capstone Submission Path** may claim data-stage completion.
- A **Capstone Data Remote** that uses S3 requires **Credential Capability Evidence** such as redacted AWS identity, reachable bucket or prefix, DVC remote resolution, or a harmless access probe.
- **Credential Capability Evidence** must not include access keys, session tokens, raw `.env` contents, or unredacted secret material.
- A **Capstone Data Automation Workflow** may configure and validate a **Capstone Data Remote** by default, but DVC push or pull is an **Approval-Gated Data Transfer**.
- An **Approval-Gated Data Transfer** produces **Observed Evidence** when it runs and blocked evidence when approval or credentials are missing.
- **Read-Only Data Inspection** does not require an **Approval Gate**.
- Phase 4 steps that write project files, change DVC state, use cloud credentials, or transfer data require an **Approval Gate** and produce an **Approval Record** before execution.
- A **Capstone Data Automation Workflow** supports **Local Data Readiness** and **Capstone Data Completeness** as distinct workflow modes.
- **Local Data Readiness** requires two dataset paths, supported layouts, split evidence, DVC tracking, DVC repo validation, a **Data Stage Evidence Artifact**, and dataset lineage artifacts.
- **Capstone Data Completeness** requires all **Local Data Readiness** evidence plus S3 remote validation and approved S3 transfer evidence.
- **Data Completion Mode** is either `local_ready` or `capstone_complete` and determines which Phase 4 contract branch is required for workflow success.
- A **Capstone Data Automation Workflow** returns **Partial Data Stage Completion** when local packaging and remote configuration evidence exist but capstone-required S3 transfer evidence is missing.
- **Partial Data Stage Completion** must appear in `build_capstone_pipeline` as completed data evidence plus a blocked capability, not as capstone success.
- A **Capstone Data Automation Workflow** produces a **Data Stage Evidence Artifact** that `build_capstone_pipeline` references through the **Artifact Manifest**.
- A **Data Stage Evidence Artifact** is the handoff from Phase 4 to downstream training, CI/CD, reporting, and deployment stages; downstream stages must not rely only on latest-run lookup for dataset lineage.
- Phase 4 exposes **Data Stage Evidence Artifact** evidence to `build_capstone_pipeline` but does not broaden `train_and_track` behavior unless a contract-facing handoff check requires it.
- A **Data Stage Evidence Artifact** includes schema version metadata, creation time, workflow status, dataset-level status, missing inputs, next actions, DVC remote state, blocked capabilities, verification results, and artifact manifest entries.
- A **Data Stage Evidence Artifact** requires exactly two dataset entries for capstone-complete success and may contain fewer entries only when the workflow status is blocked.
- A **Data Stage Evidence Artifact** must include itself in the **Artifact Manifest** and include split manifests or DVC files when generated.
- **Dataset Lineage Evidence** belongs in the **Artifact Manifest** when dataset files, DVC metadata, or DVC remote state are produced, validated, selected, or referenced.
- Phase 4 `prepare_capstone_data` excludes dataset downloads, arbitrary dataset formats, cleaning, augmentation, deduplication, quality scoring beyond layout and split counts, training behavior, HPO, LR finder, model selection, Docker, registry push, Kubernetes, KServe, Helm, ArgoCD, AWS Lambda, Hugging Face Spaces, stress testing, frontend, final report, video generation, automatic S3 transfer, secret storage, source dataset moves, source dataset deletion, and unapproved source dataset mutation.
- Phase 5 uses **Capstone Container And CI Automation Workflow** as the controlling roadmap term, overriding older PRD phase numbering without editing `PRD.md`.
- `prepare_capstone_container_ci` is the **Capstone Container And CI Automation Workflow** for Phase 5.
- `build_capstone_pipeline` uses **Container CI Stage Handoff** to surface completed, blocked, or deferred container/CI stage state and next actions, but it does not implement Dockerfile generation, image build, registry validation or push, or CI workflow generation.
- A **Container CI Stage Handoff** must reference the **Container CI Evidence Artifact** through the **Artifact Manifest** and must not rely on prose or hidden runtime state.
- Phase 5 produces a **Container CI Evidence Artifact** before Kubernetes or GitOps deployment evidence may claim reproducible capstone progress.
- The capstone evidence dependency chain is **Data Stage Evidence Artifact**, then training, MLflow, and best artifact evidence, then **Container CI Evidence Artifact**, then Kubernetes or GitOps deployment evidence.
- **Container Local Readiness** may succeed when the workflow observes or validates a usable **Data Stage Evidence Artifact**, a selected local model artifact or MLflow-backed best artifact, generated Dockerfile or container assets, and generated or validated GitHub Actions workflow evidence.
- **Container Capstone Completeness** requires **Container Local Readiness** plus capstone-complete data evidence, MLflow run evidence tied to the selected best artifact, training inputs that reference data-stage lineage, successful container image build evidence, validated registry target evidence, approval-gated registry push evidence, and CI evidence.
- **Container CI Completion Mode** is either `container_local_ready` or `container_capstone_complete` and determines which Phase 5 contract branch is required for workflow success.
- A **Container CI Evidence Artifact** records **Container CI Completion Mode** explicitly so `build_capstone_pipeline` can distinguish data-stage readiness from container/CI-stage readiness.
- A **Container CI Evidence Artifact** mirrors the Phase 4 evidence pattern and includes `schema_version`, `created_at`, `workflow_id`, `status`, `completion_mode`, `upstream_evidence`, `container`, `registry`, `ci`, `blocked_capabilities`, `deferred_capabilities`, `verification_results`, and `artifact_manifest`.
- A **Container CI Evidence Artifact** uses schema version `phase5.container_ci_evidence.v1` and workflow id `prepare_capstone_container_ci`.
- A **Container CI Evidence Artifact** is the durable handoff artifact for the container/CI stage; `build_capstone_pipeline` reads or references it rather than inferring success from prose.
- A **Container CI Evidence Artifact** must include itself, generated Dockerfile or container files, generated GitHub Actions workflow, image reference or digest when available, and upstream data or training evidence references in its **Artifact Manifest**.
- A **Container CI Evidence Artifact** includes a narrow `next_phase_readiness` section for Phase 6 Kubernetes and GitOps handoff.
- `next_phase_readiness` records image reference or digest readiness, registry push status, CI validation status, missing blockers for Kubernetes or GitOps, and deferred capabilities such as `kserve_deployment`, `helm_packaging`, `argocd_gitops`, and `eks_provisioning`.
- `next_phase_readiness` must not generate Kubernetes manifests or claim KServe, Helm, ArgoCD, GitOps, or EKS provisioning are complete.
- `next_phase_readiness` only records whether Phase 5 produced enough evidence for Phase 6 to start cleanly.
- If data and training evidence exist but the **Container CI Evidence Artifact** is missing, `build_capstone_pipeline` reports the container/CI stage as blocked with a next action to run `prepare_capstone_container_ci`.
- If upstream data or training evidence is missing and the **Container CI Evidence Artifact** is missing, `build_capstone_pipeline` reports the container/CI stage as deferred because earlier stages are not ready.
- If the **Container CI Evidence Artifact** exists with `container_local_ready`, `build_capstone_pipeline` reports container-local readiness as completed while preserving blocked or deferred capstone container completion gaps such as registry push or upstream evidence.
- If the **Container CI Evidence Artifact** exists with `container_capstone_complete` and required checks pass, `build_capstone_pipeline` reports the container/CI stage as completed for capstone container/CI readiness.
- `build_capstone_pipeline` must never mark Kubernetes or GitOps deployment complete from Phase 5 evidence.
- In `container_local_ready`, `prepare_capstone_container_ci` may generate and validate Docker or CI assets with a local model artifact while recording missing data-stage, MLflow, or best-artifact evidence in `upstream_evidence`.
- In `container_local_ready`, missing upstream capstone evidence must be explicitly marked deferred before the workflow may succeed locally.
- `build_capstone_pipeline` reports successful `container_local_ready` evidence as container-local readiness, not **Container Capstone Completeness**.
- In `container_capstone_complete`, `prepare_capstone_container_ci` blocks when the **Data Stage Evidence Artifact** is missing, not capstone-complete, or when MLflow-linked best artifact evidence is missing.
- `prepare_capstone_container_ci` must not generate complete evidence that hides upstream data-stage, training, MLflow, or best-artifact gaps.
- Phase 5 generates or validates one default **Capstone Runtime Image** first, not separate training and inference images.
- A **Capstone Runtime Image** records `intended_roles`, such as `ci`, `training_validation`, and `inference_validation`.
- A **Capstone Runtime Image** must be sufficient for bounded tests, model artifact validation, and later deployment handoff.
- If a project already has a Dockerfile, Phase 5 validates and references or adapts it rather than overwriting it by default.
- Phase 5 Dockerfile overwrites require an **Approval Gate** with `writes_project_files`.
- Phase 5 dependency detection prefers `uv.lock` or `pyproject.toml`, then `requirements.txt`, then `setup.py`.
- Phase 5 defaults to a Python 3.10 or newer base image unless project evidence requires otherwise.
- Phase 5 avoids heavyweight CUDA base images unless explicitly requested or observed as required.
- Phase 5 Dockerfile or build spec generation includes bounded default commands for tests or model validation.
- Phase 5 Dockerfile or build spec generation must not embed secrets, absolute source dataset paths, or local-only `.env` values.
- Phase 5 may add labels or comments tying generated container assets to Auto-MLOps evidence when useful.
- Phase 5 does not require perfect framework detection; the contract is a bounded **Capstone Runtime Image** that can be generated, validated, and handed off.
- Phase 5 defers separate training images, separate inference or serving images, CUDA-optimized image variants, slim production serving images, and multi-stage packaging optimization beyond what is needed for the default image.
- **Container Build Evidence** for `container_local_ready` requires generated or validated Dockerfile or container build spec evidence, identified dependency context such as `requirements.txt`, `pyproject.toml`, `setup.py`, or `uv.lock`, and Docker availability detection.
- In `container_local_ready`, image build is attempted and recorded when Docker is available; when Docker is unavailable, image build is blocked or deferred rather than failed if build spec validation passes.
- In `container_local_ready`, registry push evidence may be deferred when the artifact records the deferred capability and next action.
- **Container Build Evidence** for `container_capstone_complete` requires Docker availability, successful image build, image tag plus digest or local image id, container artifact entries in the **Artifact Manifest**, and a minimal container smoke check when feasible.
- A Phase 5 minimal container smoke check may validate imports, CLI help, or an entrypoint dry-run; it does not require a full serving endpoint.
- **Container Registry Evidence** supports generic registry target recording, with GHCR as the default first-class Phase 5 target because it aligns with GitHub Actions and repository workflows.
- DockerHub and ECR may be accepted as explicit user-provided registry targets in Phase 5, but provider-specific automation beyond target recording, validation, and approval-gated push may be deferred unless required by tests.
- In `container_local_ready`, registry target and credentials may be absent when missing registry capability is recorded as blocked or deferred.
- In `container_capstone_complete`, registry target validation, authentication capability evidence, approval-gated registry login or push evidence, pushed image reference, digest when available, and CI workflow evidence are required for workflow success.
- **Container Registry Evidence** must not store raw credentials, access tokens, or unredacted secret material.
- `registry_push_approved` means an **Approval Record** exists for the registry push step with the required **Risk Categories**.
- `registry_push_succeeded` means observed registry command or API evidence shows the image push succeeded.
- **Container Registry Evidence** records registry push approval separately from registry push execution because approval can exist without a successful push, and a push attempt without approval is invalid.
- A **Container CI Evidence Artifact** records registry approval details under structured approval evidence, such as `registry.approvals` or top-level `approval_records`.
- **Capstone CI Evidence** for `container_local_ready` requires generated `.github/workflows/capstone-ci.yml` or equivalent, parseable workflow YAML, bounded repo-local referenced commands, and checks for tests, data-stage evidence validation, training, MLflow or best artifact validation, and optional image build.
- **Capstone CI Evidence** for `container_local_ready` does not require an observed remote GitHub Actions run.
- **Capstone CI Evidence** for `container_capstone_complete` requires all local CI evidence, GitHub secrets usage for registry login or push, no plaintext registry credentials, and references to the selected image tag or registry target.
- Phase 5 CI runs bounded checks by default: unit or selected fast tests, **Data Stage Evidence Artifact** schema validation, MLflow or best artifact evidence validation, optional tiny fixture or smoke training only when an explicit bounded command is configured, optional image build, and optional registry push only under approved secret-backed conditions.
- Phase 5 CI must not run full training by default.
- Phase 5 CI may include a disabled or manual full-training job placeholder, but full training in CI is deferred until a later issue defines fixed budgets and success contract requirements.
- Phase 5 inspects existing `.github/workflows/*` files as read-only evidence and does not mutate existing workflows by default.
- Phase 5 generates a dedicated `.github/workflows/capstone-ci.yml` file unless the user explicitly asks to modify an existing workflow.
- If `.github/workflows/capstone-ci.yml` already exists, Phase 5 validates it and may adapt it only after an **Approval Gate** with `writes_project_files`.
- Phase 5 never deletes or rewrites unrelated GitHub Actions workflow files.
- Phase 5 records relevant existing workflow files as validated artifacts in the **Artifact Manifest** when they influence **Capstone CI Evidence**.
- Phase 5 blocks with next actions when existing workflows conflict with capstone CI requirements instead of silently merging unrelated CI behavior.
- Remote GitHub Actions run evidence should be included in **Capstone CI Evidence** when available through GitHub CLI or API, but is not mandatory in Phase 5 when network or authentication is unavailable unless a later issue explicitly requires remote CI verification.
- Missing remote CI run evidence is recorded as blocked or deferred and must not be treated as succeeded.
- `act` simulation is optional and deferred in Phase 5.
- Phase 5 read-only inspection does not require an **Approval Gate** when it reads project files, detects dependency files, detects Docker availability, inspects existing Dockerfile or workflow YAML, validates YAML syntax, or inspects existing local image metadata without mutating state.
- Phase 5 file generation or overwrite steps require an **Approval Gate** with `writes_project_files`.
- Phase 5 image build steps require an **Approval Gate** with `builds_image`.
- Phase 5 container smoke commands require an **Approval Gate** with `executes_project_code` when they run user project code, start a process, or execute an entrypoint.
- Phase 5 registry login and push require **Approval Gates** with **Remote Service Credential Use**, and push also requires `pushes_registry`.
- Phase 5 authenticated GitHub API or CLI inspection for remote CI evidence requires **Remote Service Credential Use**.
- Phase 5 must not claim **Container Capstone Completeness** from only a model file, Dockerfile, registry URL, CI YAML, or prose summary.
- Phase 5 includes Docker/container evidence, registry target evidence, approval-gated registry push evidence, GitHub Actions/CI evidence, and `.auto_mlops/capstone/container_ci_evidence.json`.
- Phase 5 proves a container can be built, validated, pushed when approved, and handed off; deployment endpoint evidence belongs to Phase 2 LitServe serving or Phase 6 Kubernetes/GitOps deployment.
- Phase 5 excludes Kubernetes, KServe, Helm, ArgoCD, and broader GitOps deployment work; those capabilities belong in Phase 6 or later.
- Phase 5 is planned as seven vertical contract milestones: register `prepare_capstone_container_ci`, resolve upstream evidence, generate or validate the default **Capstone Runtime Image** build spec, build and smoke-check the image when Docker is available, configure and validate the registry target, approval-gate registry login or push, and generate or validate **Capstone CI Evidence** while writing the **Container CI Evidence Artifact** with orchestrator handoff.
- Phase 5 keeps the durable orchestrator handoff last so evidence accumulates through focused contract milestones before `build_capstone_pipeline` references the container/CI stage.
- Phase 5 issue 1 registers `prepare_capstone_container_ci`, **Workflow Inputs**, routing aliases, **Container CI Completion Mode** values, local and capstone-complete contract branches, and blocked behavior.
- Phase 5 issue 2 resolves upstream evidence from **Data Stage Evidence Artifact**, training, MLflow, best artifact evidence, local model artifact fallback for `container_local_ready`, and blocked or deferred upstream evidence behavior.
- Phase 5 issue 3 generates or validates the default **Capstone Runtime Image** build spec, dependency context, `intended_roles`, and approval-gated writes or overwrites.
- Phase 5 issue 4 records Docker availability, build result, image id or tag, bounded smoke check, and blocked or deferred local-ready behavior when Docker is unavailable.
- Phase 5 issue 5 configures and validates **Container Registry Evidence**, with GHCR as the first-class default, DockerHub or ECR as explicit declared targets, redacted registry evidence, and authentication capability checks without storing secrets.
- Phase 5 issue 6 records approval-gated registry login or push evidence, approval records, pushed image reference, digest when available, and structured failed auth or push evidence.
- Phase 5 issue 7 generates or validates **Capstone CI Evidence**, writes the **Container CI Evidence Artifact**, records it in the **Artifact Manifest**, and updates `build_capstone_pipeline` to reference container/CI stage evidence.
- Common **Success Contract** checks for `prepare_capstone_container_ci` are `upstream_evidence_resolved`, `container_build_spec_reported`, `dependency_context_reported`, `container_ci_evidence_artifact_reported`, `container_artifact_manifest_reported`, `capstone_ci_workflow_reported`, `capstone_ci_workflow_validated`, and `secret_safety_validated`.
- **Success Contract** checks for `container_local_ready` include `local_model_artifact_resolved` or `mlflow_best_artifact_resolved`, `docker_availability_reported`, conditional `image_build_attempt_reported` when Docker is available, and conditional `image_build_deferred_reported` when Docker is unavailable.
- **Success Contract** checks for `container_capstone_complete` include `data_stage_capstone_complete_verified`, `mlflow_best_artifact_verified`, `training_lineage_verified`, `docker_available`, `image_build_succeeded`, `container_smoke_check_passed`, `registry_target_validated`, `registry_auth_capability_verified`, `registry_push_approved`, `registry_push_succeeded`, `pushed_image_reference_reported`, and `capstone_ci_registry_usage_validated`.
- Phase 5 avoids vague **Success Contract** checks such as `docker_ready` because they hide whether evidence came from build spec validation, dependency context, local image build, smoke check, registry authentication, registry push, or CI usage.
- An **Artifact State** is one of `generated`, `validated`, `selected`, or `external`.
- A **Deployment Report** records structured evidence such as target, selected backend, endpoint URL, server command, health result, prediction result, latency summary, GPU evidence, artifacts, approvals, rollback plan, and contract status.
- Deployment **Success Contracts** require a **Rollback Plan** even when rollback execution is outside the current workflow scope.
- The **Workflow Runtime** executes a selected **Workflow Template** and marks success only when its **Success Contract** is satisfied.
- The **Workflow Runtime** is the only authority that may set **Workflow Status**.
- A **Workflow Status** of `blocked` means a run is missing required input or waiting at an **Approval Gate**.
- A **Workflow Step** waiting for an **Approval Record** keeps the workflow `blocked`.
- Low-confidence or conflicting **Workflow Selection** produces a `blocked` **Workflow Status** instead of a generic fallback workflow.
- Missing environment prerequisites such as Docker, GPU, Kubernetes, ports, or credentials produce `blocked` or `failed` status when required observed checks cannot run.
- Missing required **Workflow Inputs** block a workflow before **Workflow Step** execution starts.
- A **Workflow Status** of `succeeded` means all required **Success Contract** checks passed.

## Example Dialogue

> **Dev:** "Can the model generate an extra deployment step if it thinks the workflow needs one?"
> **Domain expert:** "No. It may select a **Workflow Template** from the **Workflow Registry** and fill allowed arguments, but any variation must already be declared as optional steps or branches in the registry."

## Flagged Ambiguities

- "workflow" can mean either a product-supported **Workflow Template** or an LLM-generated execution plan. Resolved: Phase 0 uses **Workflow Template** for the registry-owned skeleton and does not allow the LLM to invent skeleton steps at runtime.
- "success" can mean either a final response claim or verified product completion. Resolved: workflow success requires a satisfied **Success Contract** backed by **Verification Results**.
- "validation" can mean either explanatory prompt review or executable contract enforcement. Resolved: **Success Contract** validation belongs to the **Workflow Runtime**.
- "contract" can mean either prose expectations or executable requirements. Resolved: a **Success Contract** is structured as named checks with typed evidence fields, not free-form text.
- "status" can mean free-form progress text or a controlled lifecycle state. Resolved: **Workflow Status** is a small runtime-owned enum, not prompt-authored text.
- "argument" can mean a user-provided workflow value or a concrete tool-call parameter. Resolved: **Workflow Inputs** are declared separately from **Workflow Step** arguments.
- "step" can mean a workflow-level action or a concrete function call. Resolved: **Workflow Step** is stable product language; **Tool Function** is implementation language.
- "optional" can mean either deterministic registry variation or runtime invention. Resolved: an **Optional Workflow Step** must have a registry-owned inclusion rule.
- "branch" can mean either a supported product alternative or a model-invented path. Resolved: a **Workflow Branch** is registry-owned and selected by explicit rules.
- "placeholder" can mean either planned product scope or an implemented registry entry. Resolved: Phase 0 allows PRD language for future workflows, but no **Fake Templates** in the **Workflow Registry**.
- "Lambda" can mean Lambda Labs GPU infrastructure or AWS Lambda serverless. Resolved: this distinction belongs in **Routing Aliases** and **Negative Routing Rules**, not only prompts.
- "GPU deployment" can mean general backend selection or the specific LitServe capstone path. Resolved: **General GPU Inference Workflow** and **LitServe GPU Workflow** are separate templates.
- "selection" can mean either an opaque id or a debuggable routing record. Resolved: **Workflow Selection** is structured and records matched and rejected evidence.
- "fallback" can mean either safe clarification or arbitrary generic setup. Resolved: ambiguous **Workflow Selection** blocks for clarification instead of starting a generic workflow.
- "approval" can mean either a generic confirmation or a precise risk acknowledgement. Resolved: an **Approval Gate** declares **Risk Categories** such as `writes_project_files`, `installs_packages`, `starts_server`, `builds_image`, `pushes_registry`, `uses_cloud_credentials`, or `exposes_port`.
- "approved" can mean either a chat response or an auditable workflow event. Resolved: approval requires an **Approval Record**.
- "artifact list" can mean either a prose summary or structured output evidence. Resolved: generated, validated, and selected artifacts belong in an **Artifact Manifest**.
- "data automation" can mean generic preprocessing, frontend/report/demo deployment capabilities, or capstone reproducibility work. Resolved: Phase 4 uses **Capstone Data Automation Workflow** for DVC/S3-backed dataset lineage before Docker, CI/CD, frontend, report, or demo deployment capabilities.
- "prepare reproducible data" can imply a broad reusable workflow. Resolved: Phase 4 uses `prepare_capstone_data` to keep scope tied to the EMLO capstone path.
- "support two datasets" can mean broad support for any dataset pair or a contract requiring two supported datasets. Resolved: Phase 4 requires two **Canonical Image Folder Datasets** and blocks on unsupported layouts.
- "dataset source" can mean a local path, mounted path, URL, Kaggle source, or Hugging Face source. Resolved: Phase 4 requires **User-Provided Dataset Paths** and defers dataset download automation.
- "split" can mean a discovered existing train/test layout or a file-writing transformation. Resolved: creating a **Deterministic Dataset Split** requires an **Approval Gate** and observed split evidence.
- "create split" can mean copying, moving, or manifesting files. Resolved: Phase 4 defaults to a **Split Manifest**, may copy when required, and never moves source dataset files.
- "DVC-track the data" can mean tracking raw source datasets, generated split metadata, or copied training folders. Resolved: Phase 4 tracks the **Capstone Data Package** and records raw source paths as external artifacts by default.
- "DVC remote" can mean local development storage or capstone-required S3 storage. Resolved: **Capstone Data Remote** supports both, but only S3 satisfies capstone-complete data evidence.
- "credential evidence" can mean unsafe secret inspection or safe capability checks. Resolved: Phase 4 uses **Credential Capability Evidence** and records only redacted cloud identity, bucket, prefix, and access-probe results.
- "validate remote" can mean harmless configuration checks or data transfer. Resolved: DVC push and pull are **Approval-Gated Data Transfers** and are not part of default remote validation.
- "approval for data automation" can mean approving every inspection or only risky actions. Resolved: **Read-Only Data Inspection** does not require approval, while mutations, DVC state changes, cloud credential use, and data transfers do.
- "partially complete data stage" can mean success with warnings or blocked progress with useful evidence. Resolved: Phase 4 uses **Partial Data Stage Completion** and remains blocked until capstone-required S3 transfer evidence exists.
- "Phase 4 success" can mean local development readiness or EMLO capstone completion. Resolved: Phase 4 separates **Local Data Readiness** from **Capstone Data Completeness**.
- "mode" can mean layout mode, execution mode, or success mode. Resolved: capstone workflows use **Completion Mode** with stage-specific values.
- "data-stage handoff" can mean reading agent memory or a durable artifact. Resolved: Phase 4 writes a **Data Stage Evidence Artifact** and the orchestrator references it explicitly.
- "Phase 5" can mean the older PRD Kubernetes production path or the newer capstone dependency sequence. Resolved: Phase 5 is **Capstone Container And CI Automation Workflow**; Kubernetes, KServe, Helm, ArgoCD, and GitOps move to Phase 6 or later.
- "container/CI handoff" can mean a build log, CI summary, registry URL, or durable artifact. Resolved: Phase 5 writes a **Container CI Evidence Artifact** and the orchestrator references it explicitly.
- "build capstone pipeline" can mean implementing every capstone stage or reporting stage state. Resolved: `build_capstone_pipeline` reports **Container CI Stage Handoff** state, while `prepare_capstone_container_ci` owns Phase 5 implementation.
- "missing container/CI evidence" can mean the stage failed, is blocked, or is deferred behind earlier stages. Resolved: `build_capstone_pipeline` blocks when upstream data and training are ready but container/CI evidence is missing, and defers when upstream stages are not ready.
- "next phase readiness" can mean generating Kubernetes deployment assets or only reporting handoff readiness. Resolved: Phase 5 `next_phase_readiness` only reports whether container, registry, and CI evidence are ready for Phase 6 and records Kubernetes/GitOps capabilities as deferred.
- "capstone" can mean the whole Auto-MLOps product identity or the first verified end-to-end product journey. Resolved: use **North-Star Capstone Workflow** for the current concrete capstone path while keeping Auto-MLOps broader than capstone automation.
- "upstream evidence" can mean a hard prerequisite for all container work or a deferred capstone gap during local asset generation. Resolved: `container_local_ready` may defer missing upstream capstone evidence, while `container_capstone_complete` blocks on missing capstone-complete data and MLflow-linked best artifact evidence.
- "training evidence" can mean an observed MLflow run, a selected best model artifact, or just an existing model file. Resolved: **Container Local Readiness** may use observed local model artifact evidence, while **Container Capstone Completeness** requires MLflow-linked best artifact evidence tied back to data-stage lineage.
- "training/inference container" can mean one runtime image or multiple specialized images. Resolved: Phase 5 starts with one **Capstone Runtime Image** and defers specialized image variants.
- "Dockerfile generation" can mean overwriting project packaging or conservatively adding a generated build spec. Resolved: Phase 5 validates existing Dockerfiles first, requires approval for overwrites, and avoids secrets, absolute source dataset paths, local-only `.env` values, and unnecessary CUDA bases.
- "container ready" can mean local generated assets or capstone-complete registry evidence. Resolved: Phase 5 separates **Container Local Readiness** from **Container Capstone Completeness**.
- "completion_mode" can mean data-stage values or container/CI-stage values. Resolved: the workflow input name is reused, but Phase 5 uses **Container CI Completion Mode** values `container_local_ready` and `container_capstone_complete`.
- "Docker evidence" can mean a generated Dockerfile, a validated build spec, a local image build, a pushed image, or a deployed endpoint. Resolved: Phase 5 uses **Container Build Evidence** and does not require serving endpoint evidence.
- "registry evidence" can mean a configured target, successful login, pushed image, or provider-specific registry integration. Resolved: Phase 5 uses **Container Registry Evidence**, defaults to GHCR, and treats DockerHub or ECR as explicit targets with provider-specific depth deferred unless tests require it.
- "push approved" can mean the user approved the risky action or that the image push actually completed. Resolved: Phase 5 separates `registry_push_approved` from `registry_push_succeeded`.
- "CI evidence" can mean generated workflow YAML, local validation, `act` simulation, or observed GitHub Actions run status. Resolved: Phase 5 uses **Capstone CI Evidence** with required local validation, optional remote run evidence, and deferred `act` simulation.
- "CI training" can mean evidence validation, bounded smoke training, or full training. Resolved: Phase 5 CI runs bounded evidence checks by default and defers full training in CI until fixed budgets are defined.
- "CI workflow generation" can mean creating a dedicated capstone workflow or mutating existing CI. Resolved: Phase 5 generates `.github/workflows/capstone-ci.yml` by default, inspects existing workflows read-only, and never rewrites unrelated workflows.
- "docker ready" can hide build-spec, dependency, local-image, smoke-check, registry, and CI evidence layers. Resolved: Phase 5 uses explicit **Success Contract** checks rather than vague readiness checks.
- "credential use" can mean AWS credential capability checks or authenticated GitHub/registry interactions. Resolved: Phase 5 uses **Remote Service Credential Use** for non-AWS remote service authentication.
- "capstone data automation" can expand into acquisition, cleaning, training, deployment, or reporting. Resolved: Phase 4 is limited to reproducible data lineage through `prepare_capstone_data`.
- "artifact status" can mean arbitrary prose or a controlled state. Resolved: **Artifact State** uses a small enum.
- "deployment report" can mean either structured evidence or rendered markdown. Resolved: **Deployment Report** is structured first; markdown is only a rendering.
- "rollback" can mean either rollback readiness or rollback execution. Resolved: Phase 0 deployment workflows require a **Rollback Plan**, while a standalone rollback workflow is included only when tests require it.
- "evidence" can mean observed runtime facts or declared plans. Resolved: **Observed Evidence** and **Declared Evidence** are distinct, and declared evidence cannot satisfy live-runtime checks.
- "finished" can mean either useful partial output or verified success. Resolved: a workflow may finish `blocked` or `failed` with declared next steps, but cannot finish `succeeded` when required observed checks did not run.
- "conditional" can mean a deterministic environment condition or a model-created exception. Resolved: **Conditional Contract Checks** are registry-owned and narrowly scoped.
- "failure" can mean either an opaque error or a debuggable contract result. Resolved: **Contract Failure** is structured as missing evidence and failed checks.
- "prompt test" can mean either checking prompt text or checking product routing behavior. Resolved: **Prompt Contract Tests** assert registry contract behavior from natural-language input.
