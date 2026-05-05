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
The `prepare_capstone_data` input that selects whether Phase 4 validates **Local Data Readiness** or **Capstone Data Completeness**.
_Avoid_: mode, target_completion, success type

**Data Stage Evidence Artifact**:
A durable capstone artifact at `.auto_mlops/capstone/data_stage_evidence.json` that records the exact dataset lineage state produced by **Capstone Data Automation Workflow**.
_Avoid_: latest run lookup, chat summary, hidden session state

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
- **Completion Mode** is either `local_ready` or `capstone_complete` and determines which Phase 4 contract branch is required for workflow success.
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
- "mode" can mean layout mode, execution mode, or success mode. Resolved: Phase 4 uses **Completion Mode** with `local_ready` and `capstone_complete`.
- "data-stage handoff" can mean reading agent memory or a durable artifact. Resolved: Phase 4 writes a **Data Stage Evidence Artifact** and the orchestrator references it explicitly.
- "capstone data automation" can expand into acquisition, cleaning, training, deployment, or reporting. Resolved: Phase 4 is limited to reproducible data lineage through `prepare_capstone_data`.
- "artifact status" can mean arbitrary prose or a controlled state. Resolved: **Artifact State** uses a small enum.
- "deployment report" can mean either structured evidence or rendered markdown. Resolved: **Deployment Report** is structured first; markdown is only a rendering.
- "rollback" can mean either rollback readiness or rollback execution. Resolved: Phase 0 deployment workflows require a **Rollback Plan**, while a standalone rollback workflow is included only when tests require it.
- "evidence" can mean observed runtime facts or declared plans. Resolved: **Observed Evidence** and **Declared Evidence** are distinct, and declared evidence cannot satisfy live-runtime checks.
- "finished" can mean either useful partial output or verified success. Resolved: a workflow may finish `blocked` or `failed` with declared next steps, but cannot finish `succeeded` when required observed checks did not run.
- "conditional" can mean a deterministic environment condition or a model-created exception. Resolved: **Conditional Contract Checks** are registry-owned and narrowly scoped.
- "failure" can mean either an opaque error or a debuggable contract result. Resolved: **Contract Failure** is structured as missing evidence and failed checks.
- "prompt test" can mean either checking prompt text or checking product routing behavior. Resolved: **Prompt Contract Tests** assert registry contract behavior from natural-language input.
