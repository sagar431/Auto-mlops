# Phase 0 Implementation Issues

These local issue files break PRD Phase 0 into focused implementation slices. They are ordered by dependency and intentionally exclude real GPU, LitServe, Kubernetes, frontend, cloud, and training-loop execution.

## Issues

1. [Add Workflow Registry Core And Setup Pipeline Template](./0001-workflow-registry-core-and-setup-template.md)
2. [Add Phase 0 Deployment Workflow Templates](./0002-add-phase-0-deployment-workflow-templates.md)
3. [Add Structured Workflow Selection Decision Object](./0003-workflow-selection-decision-object.md)
4. [Add Success Contract Validator And Runtime-Owned Status](./0004-success-contract-validator-and-runtime-status.md)
5. [Add Artifact Manifest Schema And Contract Checks](./0005-artifact-manifest-schema-and-contract-checks.md)
6. [Add Approval Gate Metadata And Blocking Semantics](./0006-approval-gate-metadata-and-blocking.md)
7. [Add Deployment Report Schema And Rollback Readiness Contract](./0007-deployment-report-schema-and-rollback-readiness.md)
8. [Add Phase 0 Routing And Contract Test Matrix](./0008-phase-0-routing-and-contract-test-matrix.md)

## Dependency Order

- 0001 has no blockers.
- 0002 depends on 0001.
- 0003 depends on 0001 and 0002.
- 0004 depends on 0001 and 0002.
- 0005 depends on 0004.
- 0006 depends on 0001 and 0004.
- 0007 depends on 0004 and 0005.
- 0008 depends on 0002, 0003, and 0004.
