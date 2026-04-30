# Graph Report - /home/ubuntu/emlo_course_materials  (2026-04-29)

## Corpus Check
- Corpus is ~3,239 words - fits in a single context window. You may not need a graph.

## Summary
- 131 nodes · 146 edges · 11 communities detected
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 18 edges (avg confidence: 0.76)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_MLOps Agent Architecture|MLOps Agent Architecture]]
- [[_COMMUNITY_Capstone TorchServe Pipeline|Capstone TorchServe Pipeline]]
- [[_COMMUNITY_MLOps Foundations|MLOps Foundations]]
- [[_COMMUNITY_Kubernetes KServe Canary|Kubernetes KServe Canary]]
- [[_COMMUNITY_LitServe FastAPI Serving|LitServe FastAPI Serving]]
- [[_COMMUNITY_Training Experiment Tracking|Training Experiment Tracking]]
- [[_COMMUNITY_Model Optimization LLMs|Model Optimization LLMs]]
- [[_COMMUNITY_EKS Autoscaling IAM|EKS Autoscaling IAM]]
- [[_COMMUNITY_Gradio Demo Deployment|Gradio Demo Deployment]]
- [[_COMMUNITY_AWS Lambda Serverless|AWS Lambda Serverless]]
- [[_COMMUNITY_Kubernetes Storage|Kubernetes Storage]]

## God Nodes (most connected - your core abstractions)
1. `LangGraph MLOps Deployment Agent` - 9 edges
2. `EMLO MLOps Course Syllabus` - 9 edges
3. `Amazon EKS` - 9 edges
4. `MLOps Agent Feature Checklist` - 8 edges
5. `TorchServe` - 7 edges
6. `Containerization` - 6 edges
7. `Minikube` - 6 edges
8. `torch.compile` - 6 edges
9. `Hydra Configuration Management` - 5 edges
10. `Data Version Control (DVC)` - 5 edges

## Surprising Connections (you probably didn't know these)
- `Data Version Control (DVC)` --semantically_similar_to--> `Experiment Tracking`  [INFERRED] [semantically similar]
  Session 06 - Data Version Control.pdf → Session 07 - Experiment Tracking & Hyperparameter Optimization.pdf
- `Containerization` --conceptually_related_to--> `Docker Tools`  [INFERRED]
  Session 02 - Docker - I.pdf → MLOps_Agent_Feature_Blueprint.md
- `ECR, ECS, and SageMaker` --conceptually_related_to--> `AWS Tools`  [INFERRED]
  Session 08 - AWS Crash Course.pdf → MLOps_Agent_Feature_Blueprint.md
- `Course Video Links Readme` --references--> `Session 01 - Introduction to MLOps`  [INFERRED]
  Readme.md → Session 01 - Introduction to MLOps.pdf
- `Dynamic Batching` --semantically_similar_to--> `Request Batching`  [INFERRED] [semantically similar]
  Session 12 - Deployment w_ TorchServe.pdf → Session 09 - Deployment w_ LitServe.pdf

## Hyperedges (group relationships)
- **MLOps Agent Deployment Stack** — mlops_agent_feature_blueprint_langgraph_agent, mlops_agent_feature_blueprint_docker_tools, mlops_agent_feature_blueprint_kubernetes_tools, mlops_agent_feature_blueprint_aws_tools, mlops_agent_feature_blueprint_serving_tools [EXTRACTED 1.00]
- **Reproducible Training Workflow** — session_04_pytorch_lightning_i_pytorch_lightning, session_05_pytorch_lightning_ii_hydra, session_06_data_version_control_dvc, session_07_experiment_tracking_hpo_experiment_tracking [INFERRED 0.85]
- **Cloud ML Deployment Foundations** — session_02_docker_i_containerization, session_03_docker_ii_docker_compose, session_08_aws_crash_course_ec2, session_08_aws_crash_course_ecr_ecs_sagemaker [INFERRED 0.80]
- **LitServe Serving Lifecycle** — session09_litapi, session09_batching, session09_streaming, session09_gpu_autoscaling [EXTRACTED 1.00]
- **KServe Knative Canary Stack** — session16_kserve, session17_knative, session17_revision_management, session17_canary_deployment, session17_scale_to_zero [EXTRACTED 1.00]
- **Capstone MLOps Architecture** — capstone_dvc_s3, capstone_hpo_lr_finder, capstone_docker_registry, capstone_cicd, capstone_mlops_pipeline [EXTRACTED 1.00]

## Communities

### Community 0 - "MLOps Agent Architecture"
Cohesion: 0.15
Nodes (19): MLOps Agent Implementation Constraints, Claude Implementation Prompt, Agent State Management, AWS Tools, Docker Tools, MLOps Agent Feature Blueprint, Git and CI/CD Tools, Human-in-Loop Approval Gate (+11 more)

### Community 1 - "Capstone TorchServe Pipeline"
Cohesion: 0.12
Nodes (17): CI/CD for ML Deployment, Docker Image Registry Deployment, DVC and S3 Data Versioning, Hyperparameter Optimization and Learning Rate Finder, End-to-End MLOps Pipeline, TIMM or Hugging Face Model Choice, TIMM Image Classifier, MAR Model Archive (+9 more)

### Community 2 - "MLOps Foundations"
Cohesion: 0.15
Nodes (15): Course Video Links Readme, EMLO MLOps Course Syllabus, Session 01 - Introduction to MLOps, MLOps Lifecycle, Docker Compose, Session 03 - Docker II, Multi-Container Docker Applications, Session 06 - Data Version Control (+7 more)

### Community 3 - "Kubernetes KServe Canary"
Cohesion: 0.16
Nodes (15): Container Runtime, kubectl, Kubernetes, Kubernetes Dashboard, Managed Kubernetes Services, Minikube, Istio, KServe (+7 more)

### Community 4 - "LitServe FastAPI Serving"
Cohesion: 0.14
Nodes (14): Request Batching, FastAPI, GPU Autoscaling, LitAPI, LitServe, Streaming Responses, vLLM, Dynamic Batching (+6 more)

### Community 5 - "Training Experiment Tracking"
Cohesion: 0.18
Nodes (13): Session 04 - PyTorch Lightning I, Hugging Face, PyTorch Lightning, Training Loop Abstractions, Configuration Composition, Session 05 - PyTorch Lightning II, Hydra Configuration Management, OmegaConf (+5 more)

### Community 6 - "Model Optimization LLMs"
Cohesion: 0.15
Nodes (13): AOTAutograd, CPU-GPU Orchestration Bottleneck, PrimTorch, torch.compile, TorchDynamo, TorchInductor, OpenAI Triton, Hardware-Specific CPU Optimization (+5 more)

### Community 7 - "EKS Autoscaling IAM"
Cohesion: 0.17
Nodes (12): Application Load Balancer, Cluster Autoscaler, Amazon EKS, Horizontal Pod Autoscaler, AWS IAM, Managed Kubernetes Control Plane, Amazon VPC Networking, Worker Nodes (+4 more)

### Community 8 - "Gradio Demo Deployment"
Cohesion: 0.29
Nodes (7): Gradio, gradio_client, gr.Interface, Hugging Face Sharing, Streamlit, Gradio on AWS Lambda, TorchScript Model

### Community 9 - "AWS Lambda Serverless"
Cohesion: 0.5
Nodes (4): AWS Lambda, Event-Driven Execution, Firecracker MicroVM, Stateless Functions

### Community 10 - "Kubernetes Storage"
Cohesion: 1.0
Nodes (2): Kubernetes Volume, EBS on EKS

## Knowledge Gaps
- **58 isolated node(s):** `MLOps Agent Feature Blueprint`, `Agent State Management`, `Git and CI/CD Tools`, `Quick Audit Prompt`, `Implementation Audit Report Format` (+53 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Kubernetes Storage`** (2 nodes): `Kubernetes Volume`, `EBS on EKS`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EMLO MLOps Course Syllabus` connect `MLOps Foundations` to `MLOps Agent Architecture`, `Training Experiment Tracking`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `TorchServe` connect `Capstone TorchServe Pipeline` to `LitServe FastAPI Serving`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `Session 08 - AWS Crash Course` connect `MLOps Foundations` to `MLOps Agent Architecture`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **What connects `MLOps Agent Feature Blueprint`, `Agent State Management`, `Git and CI/CD Tools` to the rest of the system?**
  _58 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Capstone TorchServe Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Should `LitServe FastAPI Serving` be split into smaller, more focused modules?**
  _Cohesion score 0.14 - nodes in this community are weakly interconnected._