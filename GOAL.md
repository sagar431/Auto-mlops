<goal>
Implement Auto-MLOps Phase 3 for the Train And Improve Loop and the first Capstone Orchestrator skeleton.

Deliver registry-owned, contract-validated support for the narrow EMLO session-06-style Hydra/PyTorch/TIMM image-classifier training path:

- detect supported training projects before running training
- add bounded `train_and_track`
- log training evidence to MLflow
- select or keep the best model artifact using explicit comparison rules
- add the first `train_until_better` controlled one-attempt improvement workflow
- add `build_capstone_pipeline` as the Capstone Orchestrator skeleton that invokes only implemented sub-workflows and records later-phase capabilities as blocked or deferred evidence
</goal>

<context>
Start in `/home/ubuntu/Auto-mlops`.

Read these files first:

- `AGENTS.md`
- `CLAUDE.md`
- `CONTEXT.md`
- `PRD.md`
- `docs/issues/phase-3/README.md`
- `docs/issues/phase-3/0001-detect-training-project-and-entrypoint.md`
- `docs/issues/phase-3/0002-run-training-and-capture-metrics.md`
- `docs/issues/phase-3/0003-track-training-in-mlflow.md`
- `docs/issues/phase-3/0004-select-best-model-artifact.md`
- `docs/issues/phase-3/0005-capstone-orchestrator-skeleton.md`
- `docs/adr/0001-workflow-registry-source-of-truth.md`
- `docs/adr/0002-workflow-runtime-executes-registry-templates-for-phase-1-setup-pipeline.md`
- `docs/adr/0003-registry-executed-workflows-skip-post-step-llm-perception-by-default.md`
- `workflow/registry.py`
- `agent/agent_loop.py`
- `action/execute_step.py`
- `mcp_mlops_tools.py`
- `tests/unit/test_workflow_registry.py`
- `tests/unit/test_agent_loop.py`

Use graph context as required by `AGENTS.md`:

- read `graphify-out/GRAPH_REPORT.md` before architecture/codebase decisions
- read `course_context/emlo_graphify/GRAPH_REPORT.md` if course alignment is needed
- after modifying code files, run `graphify update .` if the command is available

Discovery commands that may help:

- `rg "WorkflowTemplate|Workflow Template|SuccessContract|Success Contract|ArtifactManifest|Artifact Manifest" workflow agent action tests mcp_mlops_tools.py`
- `rg "setup_pipeline|deploy_litserve|train_and_track|train_until_better|build_capstone_pipeline" .`
- `rg "mlflow|checkpoint|metric|artifact|hydra|dvc|timm|torch" mcp_mlops_tools.py workflow agent action tests`
</context>

<constraints>
Preserve the product language in `CONTEXT.md`.

Use the Workflow Registry as the source of truth. Do not implement prompt-authored fake workflows. The LLM may select workflow ids and fill allowed inputs, but workflow steps, branches, inputs, approval gates, success contracts, and artifact requirements must be code-owned and testable.

Keep Phase 3 narrow:

- support Hydra/PyTorch/TIMM image-classifier projects shaped like EMLO `mlops_pratice/session-06`
- do not support arbitrary TensorFlow, sklearn, LLM, tabular, or non-image-classifier training projects
- do not run training during detection
- do not run training without explicit timeout, max epoch, device, dataset-size or subset, and Hydra override controls
- do not add unbounded hyperparameter search
- do not rewrite the user's training code to chase accuracy
- do not configure remote MLflow servers or remote DVC/S3 credentials
- do not implement S3 DVC remote automation, KServe/Helm/ArgoCD, HuggingFace Spaces, AWS Lambda serverless deployment, stress tests, frontend timeline/cards, final report, or video generation
- do not edit `PRD.md`

Workflow success must be derived from structured `VerificationResult`, `ArtifactManifest`, and contract validation evidence, not prose summaries.

Use `blocked` when required inputs, budget controls, entrypoint/config/data evidence, approval, or tracking configuration are missing. Use `failed` when a bounded command runs and fails or does not produce required metrics/artifacts.

Deferred Capstone Orchestrator stages must be explicit structured output with capability name, reason, and later-phase pointer.
</constraints>

<done_when>
Phase 3 is complete only when all of these are true:

1. `detect_training_project` or equivalent detection support returns structured observed evidence for a session-06-style fixture, including train entrypoint, Hydra config path/name, DVC or data evidence, PyTorch/TIMM dependency signals, test command, output directories, checkpoint/artifact candidates, confidence, missing inputs, and next actions.
2. Unsupported or ambiguous training projects block with missing evidence and next actions, and no training command runs during detection.
3. Natural-language training requests select `train_and_track` only when required workflow inputs are resolved or listed as missing.
4. `train_and_track` exists as a real registry-owned Workflow Template with declared required inputs, ordered Workflow Steps, approval or budget gates where needed, artifact requirements, and a Success Contract.
5. `train_and_track` blocks before executing when detection evidence or explicit bounded training controls are missing.
6. A bounded fixture training command that emits a target metric and writes a checkpoint reaches contract-derived `succeeded`.
7. A non-zero training command records command, exit code, stdout/stderr or logs, duration, failure reason, and returns `failed`.
8. A zero-exit training command with no configured target metric or no checkpoint/model artifact cannot return `succeeded`.
9. Captured params, metrics, stdout/stderr summaries or log files, duration, effective Hydra overrides, config snapshots, checkpoint/model artifacts, and artifact paths are represented as structured Verification Results and Artifact Manifest entries.
10. MLflow tracking records or verifies experiment id, run id, tracking URI, artifact URI, run status, params, metrics, logs, and checkpoint/model artifact reference when tracking is required.
11. `train_and_track` cannot satisfy its tracking contract when MLflow run existence cannot be verified.
12. Best-artifact selection uses explicit metric name, direction, threshold, tie policy, baseline, and checkpoint/model artifact evidence.
13. Winning latest runs produce an Artifact Manifest entry with `state: selected`, source run id, metric, comparison result, path or URI, and checksum when available.
14. Worse or incomplete latest runs keep the baseline artifact and record a structured discard or keep-baseline reason.
15. `train_until_better` exists as a controlled minimal workflow shape: baseline, one Hydra override set, fixed-budget attempt, compare, keep/discard, rollback point, duration, decision, and stop reason.
16. `train_until_better` blocks without fixed budget controls and a rollback point.
17. `build_capstone_pipeline` exists as a top-level Workflow Template named Capstone Orchestrator, with declared setup, data, train, deploy, monitor, and report stages.
18. `build_capstone_pipeline` invokes only implemented sub-workflows such as `setup_pipeline`, `train_and_track`, `train_until_better`, and Phase 2 LitServe workflows when their inputs and contracts are satisfied.
19. `build_capstone_pipeline` records missing S3 DVC remote automation, KServe/Helm/ArgoCD, HuggingFace Spaces, AWS Lambda, stress tests, frontend, final report, and video as blocked or deferred evidence instead of marking full capstone success.
20. The final Capstone Orchestrator output lists completed stages, blocked stages, deferred stages, selected model artifact when available, endpoint evidence when available, artifact manifest entries, and next actions.
21. Focused tests for Phase 3 pass:
    - `pytest tests/unit/test_workflow_registry.py tests/unit/test_agent_loop.py tests/unit/test_execute_step.py`
    - `pytest tests/e2e/test_phase3_train_and_track.py tests/e2e/test_phase3_train_until_better.py tests/e2e/test_phase3_capstone_orchestrator.py` if those files exist, or the equivalent new e2e files added for Phase 3.
22. The broader relevant test suite passes or any remaining failures are documented as pre-existing or environment-blocked:
    - `pytest tests/unit tests/e2e`
    - `ruff check .` if ruff is installed in the project environment.
</done_when>

<workflow>
1. Inspect current git status and preserve unrelated changes.
2. Read the context files and graph reports listed above.
3. Map existing registry, runtime, tool, and test patterns before editing.
4. Implement Phase 3 test-first where practical:
   - detection fixture and detection/routing tests
   - `train_and_track` registry and runtime tests
   - MLflow tracking contract tests
   - best-artifact and `train_until_better` tests
   - Capstone Orchestrator routing and deferred-stage tests
5. Implement detection for the narrow supported training project shape.
6. Add or extend tool/runtime functions for bounded training execution and evidence capture.
7. Add MLflow logging and verification in a way that does not hide failed training.
8. Add deterministic best-artifact comparison and selection.
9. Add `train_until_better` as the minimal controlled one-attempt loop.
10. Add `build_capstone_pipeline` as a registry-owned orchestrator skeleton that coordinates implemented workflows and records deferred later-phase capabilities.
11. Run focused tests after each major slice.
12. Run broader tests and lint when the focused suite passes.
13. Update `graphify` if code files changed and the command is available.
14. Final review: check for accidental `PRD.md` edits, broad framework support, fake template shortcuts, prompt-authored success, and unbounded training behavior.
</workflow>

<verification_loop>
Use focused checks first:

- `pytest tests/unit/test_workflow_registry.py`
- `pytest tests/unit/test_agent_loop.py`
- `pytest tests/unit/test_execute_step.py`
- `pytest tests/e2e/test_phase3_train_and_track.py`
- `pytest tests/e2e/test_phase3_train_until_better.py`
- `pytest tests/e2e/test_phase3_capstone_orchestrator.py`

If a named Phase 3 e2e file does not exist yet, create the appropriate test file or run the equivalent new test file.

Then run:

- `pytest tests/unit tests/e2e`
- `ruff check .` if available
- `graphify update .` after code changes if available

If a command cannot run because dependencies are missing or the environment lacks a needed binary, record the exact command, the blocking error, and the smallest next action. Do not claim success from unrun checks.
</verification_loop>

<execution_rules>
- Check git status before edits.
- Preserve unrelated user changes.
- Prefer `rg` over `grep` when available.
- Use the runtime's patch/edit tool for manual edits when available.
- Read context files before implementation.
- Batch independent file reads in parallel when the runtime supports it.
- Run focused tests before broad tests.
- Do not paper over failures.
- Do not widen scope.
- Keep the final answer concise.
</execution_rules>

<output_contract>
When complete, provide a concise final report containing:

- changed files grouped by purpose
- tests and verification commands run, with pass/fail/block status
- any environment blockers or pre-existing failures
- confirmation that `PRD.md` was not edited
- confirmation that Phase 3 deferred capabilities remain blocked or deferred, not fake-succeeded

Do not mark the goal complete unless the `done_when` contract is satisfied or remaining blockers are explicit and outside the repository implementation scope.
</output_contract>
