# Phase 1 Implementation Issues

These local issue files refine the PRD's "Verified Setup Pipeline" wording into the runtime integration work needed to make `setup_pipeline` execute from the Phase 0 **Workflow Registry**. `PRD.md` is intentionally not edited yet.

Phase 1 is limited to the first usable local `setup_pipeline` workflow. Deployment, training, cloud, frontend, persistent database migrations, and multi-workflow orchestration remain out of scope.

## Issues

1. [Select Workflow Before Perception Or Decision](./0001-select-workflow-before-perception-or-decision.md)
2. [Project setup_pipeline Template Into Executable Runtime Steps](./0002-project-setup-pipeline-template-into-executable-runtime-steps.md)
3. [Block Missing Required Workflow Inputs With Clarifying Questions](./0003-block-missing-required-workflow-inputs-with-clarifying-questions.md)
4. [Enforce setup_pipeline Approval Gates Before execute_step](./0004-enforce-setup-pipeline-approval-gates-before-execute-step.md)
5. [Capture Verification Results And Artifact Manifest For setup_pipeline](./0005-capture-verification-results-and-artifact-manifest-for-setup-pipeline.md)
6. [Derive Final Workflow Status From SuccessContract](./0006-derive-final-workflow-status-from-success-contract.md)
7. [Add Local setup_pipeline End-to-End Test Fixture](./0007-add-local-setup-pipeline-end-to-end-test-fixture.md)

## Dependency Order

- 0001 has no blockers.
- 0002 depends on 0001.
- 0003 depends on 0001 and 0002.
- 0004 depends on 0002.
- 0005 depends on 0002 and 0004.
- 0006 depends on 0005.
- 0007 depends on 0001 through 0006.
