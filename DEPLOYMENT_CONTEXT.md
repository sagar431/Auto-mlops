# MLOps Agent - Deployment Feature Context

## 🎯 Overview

This document provides context for implementing the **Multi-Deployment Target Selection** feature in the MLOps Agent. The agent should allow users to choose where they want to deploy their trained models.

---

## 📁 Project Structure (Existing)

```
/Users/sagar/Desktop/Agent_codefile/mlops_agent/
├── agent/                    # Core agent loop (Phase 2 - exists)
│   ├── __init__.py
│   ├── agent_loop.py        # Main P→D→A loop
│   ├── agentSession.py      # Session management
│   ├── contextManager.py    # Graph-based execution tracking
│   └── model_manager.py     # LLM provider management
├── perception/              # Query analysis
├── decision/                # Planning
├── action/                  # Step execution
├── summarization/           # Result summaries
├── memory/                  # Session & experiment logs
├── mcp_servers/             # MCP tool implementations
│   └── mcp_configs/
│       ├── hydra_ops.py
│       ├── mlflow_ops.py
│       ├── dvc_ops.py
│       └── github_ops.py
├── prompts/                 # LLM prompts
│   ├── perception_prompt.txt
│   ├── decision_prompt.txt
│   ├── improvement_prompt.txt
│   └── summarizer_prompt.txt
├── templates/               # Config templates
│   ├── hydra/
│   ├── dvc/
│   ├── docker/
│   └── github/
├── config/
│   ├── mlops_defaults.yaml
│   ├── models.json
│   └── profiles.yaml
├── mcp_mlops_tools.py       # Main MCP server (28 tools)
└── requirements.txt
```

---

## 🚀 New Feature: Deployment Target Selection

### User Story
> "As a user, after training my model, I want to choose where to deploy it: LitServe, Gradio, FastAPI+Lambda, TorchServe, or KServe"

### 5 Deployment Targets

| Target | Use Case | Key Characteristics |
|--------|----------|---------------------|
| **LitServe** | High-throughput inference | Built on FastAPI, batching, GPU autoscaling, streaming |
| **Gradio** | Quick prototypes & demos | Simple UI, `share=True` for instant URL, HF Spaces |
| **FastAPI + Lambda** | Serverless deployment | Pay-per-use, auto-scaling, CPU-only, cold starts |
| **TorchServe** | Enterprise production | Model versioning, .mar packaging, hot-swapping |
| **KServe** | Kubernetes-native | InferenceService, auto-scaling, canary deployments |

---

## 🔧 Implementation Requirements

### 1. New MCP Tools to Add (in mcp_mlops_tools.py)

Add a new category: **Deployment Tools**

```python
# ========== DEPLOYMENT TOOLS ==========

# LitServe Tools
- create_litserve_api(project_path, model_path, model_type) -> Creates LitAPI class
- configure_litserver(project_path, max_batch_size, workers_per_device, accelerator) -> Server config

# Gradio Tools  
- create_gradio_interface(project_path, model_path, interface_type) -> Creates gr.Interface
- deploy_to_huggingface(project_path, space_name, hf_token) -> Deploys to HF Spaces

# FastAPI + Lambda Tools
- create_fastapi_app(project_path, model_path, endpoint_type) -> Creates FastAPI app
- create_lambda_dockerfile(project_path, runtime) -> Dockerfile with Lambda adapter
- generate_cdk_stack(project_path, stack_name, memory_size) -> AWS CDK deployment

# TorchServe Tools
- create_torchserve_handler(project_path, model_path, handler_type) -> Custom handler
- create_mar_archive(project_path, model_name, version) -> Package into .mar
- generate_torchserve_config(project_path, inference_port, management_port) -> config.properties

# KServe Tools
- create_inference_service_yaml(project_path, model_name, runtime) -> InferenceService YAML
- generate_kserve_config(project_path, min_replicas, max_replicas) -> Scaling config
```

### 2. New Templates to Create

```
templates/
├── deployment/
│   ├── litserve/
│   │   ├── server.py.template
│   │   └── requirements.txt.template
│   ├── gradio/
│   │   ├── app.py.template
│   │   └── requirements.txt.template
│   ├── fastapi_lambda/
│   │   ├── app.py.template
│   │   ├── Dockerfile.template
│   │   └── cdk.py.template
│   ├── torchserve/
│   │   ├── handler.py.template
│   │   ├── config.properties.template
│   │   └── requirements.txt.template
│   └── kserve/
│       ├── inference_service.yaml.template
│       └── config.yaml.template
```

### 3. New Prompt to Create

Create `prompts/deployment_selector_prompt.txt`:

```
# MLOps Deployment Selector Prompt

You are the DEPLOYMENT SELECTOR module of an MLOps Agent.

## Your Role
Analyze the user's deployment requirements and recommend the optimal deployment target.

## Input Context
- Model Type: {model_type} (e.g., image_classifier, llm, diffusion, etc.)
- Model Framework: {framework} (pytorch, tensorflow, onnx)
- Expected Traffic: {traffic} (low, medium, high)
- Infrastructure: {infrastructure} (local, aws, kubernetes, huggingface)
- GPU Required: {gpu_required} (true/false)
- Budget: {budget} (free, pay-per-use, dedicated)

## Deployment Options

### 1. LitServe
- Best for: High-throughput inference (1000+ req/sec)
- Features: Batching, GPU autoscaling, streaming
- Requires: GPU recommended, server infrastructure
- NOT for: LLMs (use vLLM instead)

### 2. Gradio
- Best for: Quick prototypes, demos, internal tools
- Features: Simple UI, instant sharing, HF Spaces integration
- Requires: Minimal setup
- NOT for: Production high-traffic systems

### 3. FastAPI + Lambda
- Best for: Serverless, cost-effective, variable traffic
- Features: Pay-per-use, auto-scaling, no server management
- Requires: AWS account, model must be CPU-compatible or ONNX
- NOT for: GPU inference, real-time low-latency

### 4. TorchServe
- Best for: Enterprise production, model versioning
- Features: Multi-model, dynamic batching, hot-swapping, metrics
- Requires: Server infrastructure, .mar packaging knowledge
- NOT for: Quick prototypes

### 5. KServe
- Best for: Kubernetes-native deployments, auto-scaling
- Features: InferenceService, canary deployments, multi-framework
- Requires: Kubernetes cluster, K8s expertise
- NOT for: Simple deployments without K8s

## Output JSON
{
  "recommended_target": "litserve|gradio|fastapi_lambda|torchserve|kserve",
  "reasoning": "Why this target is best for the user's requirements",
  "alternatives": ["list of other suitable options"],
  "warnings": ["any concerns or limitations"],
  "next_steps": ["step1", "step2", "step3"]
}
```

### 4. Config Updates

Add to `config/profiles.yaml`:

```yaml
deployment:
  targets:
    litserve:
      default_batch_size: 64
      default_workers: 4
      accelerator: "auto"  # cpu, gpu, auto
    
    gradio:
      share: false
      server_port: 7860
      
    fastapi_lambda:
      memory_size: 1024
      timeout: 30
      runtime: "python3.11"
      
    torchserve:
      inference_port: 8080
      management_port: 8081
      metrics_port: 8082
      
    kserve:
      min_replicas: 1
      max_replicas: 5
      target_utilization: 80
```

---

## 📝 Template Examples

### LitServe Template (templates/deployment/litserve/server.py.template)

```python
"""LitServe Server for ${model_name}"""
import torch
import litserve as ls
from PIL import Image
import io
import base64

class ${class_name}API(ls.LitAPI):
    def setup(self, device):
        """Initialize model and components"""
        self.device = device
        # Load model
        self.model = torch.jit.load("${model_path}")
        self.model = self.model.to(device)
        self.model.eval()
        
        # Setup transforms
        ${transforms_code}
        
        # Load labels if needed
        ${labels_code}
    
    def decode_request(self, request):
        """Convert request to model input"""
        ${decode_code}
    
    def predict(self, x):
        """Run inference"""
        with torch.no_grad():
            return self.model(x)
    
    def encode_response(self, output):
        """Convert output to response"""
        ${encode_code}

if __name__ == "__main__":
    api = ${class_name}API()
    server = ls.LitServer(
        api,
        accelerator="${accelerator}",
        max_batch_size=${batch_size},
        batch_timeout=${batch_timeout},
        workers_per_device=${workers}
    )
    server.run(port=${port})
```

### Gradio Template (templates/deployment/gradio/app.py.template)

```python
"""Gradio Interface for ${model_name}"""
import gradio as gr
import torch
from PIL import Image

class ${class_name}:
    def __init__(self, model_path="${model_path}"):
        self.device = torch.device("${device}")
        self.model = torch.jit.load(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        ${setup_code}
    
    @torch.no_grad()
    def predict(self, ${input_params}):
        ${predict_code}

# Create instance
classifier = ${class_name}()

# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=${inputs},
    outputs=${outputs},
    title="${title}",
    description="${description}",
    examples=${examples}
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=${port}, share=${share})
```

### TorchServe Handler Template (templates/deployment/torchserve/handler.py.template)

```python
"""TorchServe Handler for ${model_name}"""
import torch
import logging
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class ${class_name}Handler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, context):
        """Initialize model. Called once when model is loaded."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(f"{model_dir}/${model_file}")
        self.model.to(self.device)
        self.model.eval()
        
        ${init_code}
        
        self.initialized = True
        logger.info("Model initialized successfully")
    
    def preprocess(self, data):
        """Preprocess input data"""
        ${preprocess_code}
    
    def inference(self, data):
        """Run inference"""
        with torch.no_grad():
            return self.model(data)
    
    def postprocess(self, inference_output):
        """Postprocess model output"""
        ${postprocess_code}
```

### FastAPI + Lambda Template (templates/deployment/fastapi_lambda/app.py.template)

```python
"""FastAPI Application for ${model_name} - Lambda Ready"""
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="${title}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
device = torch.device("cpu")  # Lambda is CPU-only
model = torch.jit.load("${model_path}")
model.to(device)
model.eval()

${setup_code}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(${endpoint_params}):
    ${predict_code}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=${port})
```

### Lambda Dockerfile Template (templates/deployment/fastapi_lambda/Dockerfile.template)

```dockerfile
FROM public.ecr.aws/docker/library/python:${python_version}-slim

# Copy Lambda adapter
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=${port}
WORKDIR /var/task

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py ./
COPY ${model_file} ./

CMD exec uvicorn --host 0.0.0.0 --port $PORT app:app
```

### KServe InferenceService Template (templates/deployment/kserve/inference_service.yaml.template)

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: ${service_name}
  namespace: ${namespace}
spec:
  predictor:
    serviceAccountName: ${service_account}
    ${runtime}:
      storageUri: ${storage_uri}
      resources:
        limits:
          cpu: "${cpu_limit}"
          memory: "${memory_limit}"
          ${gpu_config}
        requests:
          cpu: "${cpu_request}"
          memory: "${memory_request}"
    minReplicas: ${min_replicas}
    maxReplicas: ${max_replicas}
```

---

## 🔄 Integration with Existing Agent

### Decision Module Update

The Decision module should recognize deployment intents and route to deployment tools:

```python
# In decision logic
if perception_output["pipeline_stage"] == "deploy":
    deployment_target = perception_output.get("deployment_target")
    
    if deployment_target == "litserve":
        steps = [
            {"tool": "create_litserve_api", "args": {...}},
            {"tool": "configure_litserver", "args": {...}},
        ]
    elif deployment_target == "gradio":
        steps = [
            {"tool": "create_gradio_interface", "args": {...}},
        ]
    # ... etc
```

### Perception Module Update

Add deployment detection to perception:

```python
# Deployment intent patterns
deployment_patterns = {
    "litserve": ["high throughput", "batching", "gpu scaling", "litserve"],
    "gradio": ["demo", "prototype", "quick", "ui", "interface", "gradio"],
    "lambda": ["serverless", "lambda", "aws", "pay per use", "cost effective"],
    "torchserve": ["production", "versioning", "enterprise", "torchserve"],
    "kserve": ["kubernetes", "k8s", "kserve", "auto scaling", "canary"],
}
```

---

## 📋 Implementation Checklist

### Phase 1: Templates & Config
- [x] Create `templates/deployment/` directory structure
- [x] Add LitServe templates (server.py, requirements.txt)
- [x] Add Gradio templates (app.py, requirements.txt)
- [x] Add FastAPI + Lambda templates (app.py, Dockerfile, cdk_stack.py, requirements.txt)
- [x] Add TorchServe templates (handler.py, config.properties, create_mar.sh, requirements.txt)
- [x] Add KServe templates (inference_service.yaml, config.yaml, kustomization.yaml)
- [x] Update `config/profiles.yaml` with deployment settings

### Phase 2: MCP Tools
- [x] Add `create_litserve_api` tool
- [x] Add `configure_litserver` tool
- [x] Add `create_gradio_interface` tool
- [x] Add `deploy_to_huggingface` tool
- [x] Add `create_fastapi_app` tool
- [x] Add `create_lambda_dockerfile` tool
- [x] Add `generate_cdk_stack` tool
- [x] Add `create_torchserve_handler` tool
- [x] Add `create_mar_archive` tool
- [x] Add `generate_torchserve_config` tool
- [x] Add `create_inference_service_yaml` tool
- [x] Add `generate_kserve_config` tool (bonus)

### Phase 3: Prompts & Integration
- [x] Create `prompts/deployment_selector_prompt.txt`
- [x] Update perception prompt for deployment detection
- [x] Update decision prompt with deployment tools
- [x] Add deployment stage to agent loop (agent_loop.py updated)

### Phase 4: Testing
- [ ] Test each deployment template generation
- [ ] Test deployment tool chain
- [ ] End-to-end test: train → deploy flow

---

## 🎯 Example User Flow

```
User: "Deploy my cat-dog classifier to AWS Lambda for serverless inference"

Agent (Perception):
{
  "pipeline_stage": "deploy",
  "deployment_target": "fastapi_lambda",
  "model_type": "image_classifier",
  "entities": ["cat-dog classifier", "AWS Lambda", "serverless"]
}

Agent (Decision):
{
  "steps": [
    {"tool": "create_fastapi_app", "args": {"model_path": "model.pt", "endpoint_type": "image"}},
    {"tool": "create_lambda_dockerfile", "args": {"runtime": "python3.11"}},
    {"tool": "generate_cdk_stack", "args": {"stack_name": "CatDogClassifier"}}
  ]
}

Agent (Action):
- Creates app.py with FastAPI endpoint
- Creates Dockerfile with Lambda adapter
- Creates CDK stack for deployment

Agent (Summary):
"✅ Created serverless deployment package:
- app.py: FastAPI endpoint for image classification
- Dockerfile: Lambda-ready container
- cdk.py: AWS CDK stack for deployment

Next steps:
1. Run `cdk deploy` to deploy to AWS Lambda
2. Test endpoint at the generated URL"
```

---

## 📚 Reference Links (from EMLO4)

- LitServe Docs: https://lightning.ai/docs/litserve/home
- Gradio Docs: https://www.gradio.app/docs
- TorchServe: https://pytorch.org/serve/
- KServe: https://kserve.github.io/website/
- AWS Lambda Web Adapter: https://github.com/awslabs/aws-lambda-web-adapter
- FastAPI: https://fastapi.tiangolo.com/

---

## 🔑 Key Design Principles

1. **User Choice**: Always let user choose deployment target (don't force)
2. **Smart Defaults**: Provide sensible defaults based on model type
3. **Template-Based**: Use templates for consistency and maintainability
4. **Incremental**: Each tool should do one thing well
5. **Testable**: Generate testable deployment code
6. **Documentation**: Generated code should be well-commented

---

*This context document is for Claude Code to understand the deployment feature implementation for the MLOps Agent project.*
