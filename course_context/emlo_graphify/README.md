# EMLO Course Graph Context

This folder contains compact Graphify artifacts extracted from the EMLO course materials.

Use these files as the course context for architecture, product, prompt, and workflow decisions:

- `GRAPH_REPORT.md` — human-readable summary of communities, god nodes, hyperedges, gaps, and suggested questions.
- `graph.json` — machine-readable knowledge graph for Graphify queries and graph-aware tools.

The original PDFs, notebooks, and downloaded course materials are intentionally not committed to this repository. Keep this folder small and use it as context instead of uploading or re-reading the full course corpus.

Recommended use in future AI sessions:

1. Read `course_context/emlo_graphify/GRAPH_REPORT.md`.
2. Use `course_context/emlo_graphify/graph.json` for Graphify query/path/explain workflows.
3. Align Auto-MLOps implementation with the course graph workflows:
   - reproducible training: Hydra, DVC, MLflow, HPO/LR finder
   - cloud deployment foundations: Docker, registry, AWS/EKS
   - LitServe GPU serving lifecycle
   - Gradio demo deployment
   - TorchServe MAR production serving
   - KServe/Knative canary and rollback
   - monitoring, alerts, and rollback reports
