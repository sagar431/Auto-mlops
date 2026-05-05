# Phase 5 Follows Capstone Container And CI Evidence Before Kubernetes

Auto-MLOps will use the newer capstone dependency sequence from `CONTEXT.md` and the Phase 4 issue docs as the implementation roadmap for Phase 5. Phase 5 is `prepare_capstone_container_ci`, the **Capstone Container And CI Automation Workflow**. Kubernetes and GitOps work, including KServe, Helm, ArgoCD, and related deployment automation, moves to Phase 6 or later.

This decision overrides the older `PRD.md` phase numbering for implementation order without editing `PRD.md` in this planning session. `PRD.md` remains historical product context.

The trade-off is that Kubernetes production work waits, but Phase 5 first produces the reproducible container, registry, and CI evidence that Kubernetes and GitOps deployment evidence should depend on. This avoids implementing KServe, Helm, or ArgoCD before there is a durable `.auto_mlops/capstone/container_ci_evidence.json` handoff artifact, registry evidence, and CI evidence.

Consequences:

- Phase 5 issue docs use the container/CI scope.
- Phase 6 or later owns Kubernetes, KServe, Helm, ArgoCD, and GitOps.
- Future contributors should not treat the older `PRD.md` phase numbering as implementation order.
- `PRD.md` should remain unedited for this planning decision.
