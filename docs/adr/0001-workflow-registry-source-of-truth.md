# Workflow Registry Is The Source Of Truth For Supported Workflow Skeletons

Auto-MLOps will use the **Workflow Registry** as the source of truth for supported workflow skeletons. This prioritizes reliability, verifiability, and **Success Contract** enforcement over open-ended agent flexibility: the LLM may select a workflow and fill allowed arguments, but it may not invent, remove, or reorder required steps.

The trade-off is less LLM flexibility, but fewer unverified or random tool chains. We rejected the alternative of letting prompts or the LLM dynamically generate full `plan_graph` skeletons for registered workflows.
