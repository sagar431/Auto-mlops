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
- "artifact status" can mean arbitrary prose or a controlled state. Resolved: **Artifact State** uses a small enum.
- "deployment report" can mean either structured evidence or rendered markdown. Resolved: **Deployment Report** is structured first; markdown is only a rendering.
- "rollback" can mean either rollback readiness or rollback execution. Resolved: Phase 0 deployment workflows require a **Rollback Plan**, while a standalone rollback workflow is included only when tests require it.
- "evidence" can mean observed runtime facts or declared plans. Resolved: **Observed Evidence** and **Declared Evidence** are distinct, and declared evidence cannot satisfy live-runtime checks.
- "finished" can mean either useful partial output or verified success. Resolved: a workflow may finish `blocked` or `failed` with declared next steps, but cannot finish `succeeded` when required observed checks did not run.
- "conditional" can mean a deterministic environment condition or a model-created exception. Resolved: **Conditional Contract Checks** are registry-owned and narrowly scoped.
- "failure" can mean either an opaque error or a debuggable contract result. Resolved: **Contract Failure** is structured as missing evidence and failed checks.
- "prompt test" can mean either checking prompt text or checking product routing behavior. Resolved: **Prompt Contract Tests** assert registry contract behavior from natural-language input.
