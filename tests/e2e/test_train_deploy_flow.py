#!/usr/bin/env python3
"""
End-to-end tests for the full train -> deploy pipeline flow.

Tests the complete MLOps agent workflow:
1. Initial perception and setup
2. Decision planning with tool execution
3. Training with metrics tracking
4. Improvement loop when below accuracy threshold
5. Deployment to various targets (Gradio, LitServe, etc.)
6. Final summarization

Run with: pytest tests/e2e/test_train_deploy_flow.py -v
"""

from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from agent.agent_loop import AgentLoop, StepType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_prompts_dir(tmp_path):
    """Create temporary prompts directory with test prompts."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "perception_prompt.txt").write_text(
        "Analyze: {query}\nStage: {stage}\nExperiment: {experiment_state}"
    )
    (prompts_dir / "decision_prompt.txt").write_text("Plan: {query}\nPerception: {perception}")
    (prompts_dir / "summarizer_prompt.txt").write_text("Summarize: {query}")
    (prompts_dir / "improvement_prompt.txt").write_text(
        "Improve: target={target_accuracy} current={current_accuracy} "
        "loss={current_loss} gap={gap} config={current_config} "
        "history={training_history} attempt={attempt} max={max_attempts} "
        "previous={previous_improvements}"
    )

    return str(prompts_dir)


@pytest.fixture
def test_project(tmp_path):
    """Create a temporary ML project structure for testing."""
    project_dir = tmp_path / "test_ml_project"
    project_dir.mkdir()

    # Create basic project structure
    (project_dir / "config").mkdir()
    (project_dir / "configs").mkdir()
    (project_dir / "data").mkdir()
    (project_dir / "src").mkdir()
    (project_dir / "models").mkdir()

    # Create a basic config file
    config_content = """
defaults:
  - _self_

experiment:
  name: test_experiment
  seed: 42

model:
  type: resnet18
  pretrained: true

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
"""
    (project_dir / "configs" / "config.yaml").write_text(config_content)

    # Create a basic training script
    train_script = '''#!/usr/bin/env python3
"""Training script for testing."""

def train():
    print("Training model...")
    return {"accuracy": 0.85, "loss": 0.15}

if __name__ == "__main__":
    train()
'''
    (project_dir / "src" / "train.py").write_text(train_script)

    # Create sample data
    (project_dir / "data" / "train.csv").write_text("id,label\n1,cat\n2,dog\n")

    # Create requirements.txt
    (project_dir / "requirements.txt").write_text(
        "torch>=2.0.0\nhydra-core>=1.3.0\nmlflow>=2.0.0\n"
    )

    # Create a model checkpoint placeholder
    (project_dir / "models" / "model.pth").write_bytes(b"mock_model_weights")

    return str(project_dir)


@pytest.fixture
def mock_tools_module():
    """Create a mock tools module with all required MLOps tools."""
    module = ModuleType("mock_mcp_tools")

    # Hydra tools
    def analyze_project_config(project_path: str) -> dict:
        return {
            "success": True,
            "has_train_script": True,
            "has_config": True,
            "framework": "pytorch",
        }

    def create_hydra_config(project_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "config_path": f"{project_path}/configs/config.yaml",
            "created_files": ["config.yaml", "model.yaml", "training.yaml"],
        }

    def update_hydra_config(project_path: str, config_path: str, updates: dict) -> dict:
        return {
            "success": True,
            "config_path": config_path,
            "updates_applied": updates,
        }

    def validate_hydra_config(project_path: str, config_path: str) -> dict:
        return {"success": True, "valid": True, "errors": []}

    # MLflow tools
    def init_mlflow_experiment(experiment_name: str, **kwargs) -> dict:
        return {
            "success": True,
            "experiment_id": "exp-123",
            "experiment_name": experiment_name,
        }

    def start_mlflow_run(experiment_name: str, run_name: str = None, **kwargs) -> dict:
        return {
            "success": True,
            "run_id": "run-456",
            "run_name": run_name or "training_run",
        }

    def log_mlflow_params(params: dict) -> dict:
        return {"success": True, "logged_params": params}

    def log_mlflow_metrics(metrics: dict, step: int = None) -> dict:
        return {"success": True, "logged_metrics": metrics, "step": step}

    def log_mlflow_artifact(artifact_path: str, artifact_type: str = None) -> dict:
        return {"success": True, "artifact_path": artifact_path}

    def end_mlflow_run(status: str = "FINISHED") -> dict:
        return {"success": True, "status": status}

    def get_best_mlflow_run(experiment_name: str, metric_name: str) -> dict:
        return {
            "success": True,
            "run_id": "run-456",
            "metrics": {"accuracy": 0.85, "loss": 0.15},
        }

    def register_mlflow_model(model_name: str, run_id: str, **kwargs) -> dict:
        return {
            "success": True,
            "model_name": model_name,
            "model_version": "1",
        }

    # Training tools
    def analyze_training_results(results_path: str = None, **kwargs) -> dict:
        return {
            "success": True,
            "current_value": 0.82,
            "metrics": {"accuracy": 0.82, "loss": 0.25, "f1_score": 0.80},
        }

    def check_accuracy_threshold(current_accuracy: float, target_accuracy: float) -> dict:
        return {
            "success": True,
            "threshold_met": current_accuracy >= target_accuracy,
            "current_accuracy": current_accuracy,
            "target_accuracy": target_accuracy,
            "gap": target_accuracy - current_accuracy,
        }

    def suggest_improvements(
        current_metrics: dict,
        current_config: dict,
        target_accuracy: float,
        attempt_number: int,
    ) -> dict:
        return {
            "success": True,
            "gap": target_accuracy - current_metrics.get("accuracy", 0),
            "config_changes": {"learning_rate": 0.0005, "epochs": 15},
            "reasoning": "Reducing learning rate and increasing epochs for better convergence",
        }

    # Deployment tools
    def create_gradio_interface(
        project_path: str, model_path: str, interface_type: str = "image", **kwargs
    ) -> dict:
        return {
            "success": True,
            "interface_path": f"{project_path}/app.py",
            "interface_type": interface_type,
        }

    def deploy_to_huggingface(
        project_path: str, repo_name: str, space_name: str = None, **kwargs
    ) -> dict:
        return {
            "success": True,
            "space_url": f"https://huggingface.co/spaces/{repo_name}",
            "space_name": space_name or repo_name,
        }

    def create_litserve_api(
        project_path: str, model_path: str, api_name: str = "serve", **kwargs
    ) -> dict:
        return {
            "success": True,
            "api_path": f"{project_path}/{api_name}.py",
            "config_path": f"{project_path}/litserve_config.yaml",
        }

    def configure_litserver(project_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "config_path": f"{project_path}/litserve_config.yaml",
        }

    def create_fastapi_app(project_path: str, model_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "app_path": f"{project_path}/app.py",
            "requirements_path": f"{project_path}/requirements.txt",
        }

    def create_lambda_dockerfile(project_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "dockerfile_path": f"{project_path}/Dockerfile.lambda",
        }

    def generate_cdk_stack(project_path: str, stack_name: str, **kwargs) -> dict:
        return {
            "success": True,
            "stack_path": f"{project_path}/cdk_stack.py",
            "stack_name": stack_name,
        }

    def create_torchserve_handler(project_path: str, model_name: str, **kwargs) -> dict:
        return {
            "success": True,
            "handler_path": f"{project_path}/handler.py",
        }

    def create_mar_archive(project_path: str, model_name: str, **kwargs) -> dict:
        return {
            "success": True,
            "mar_path": f"{project_path}/{model_name}.mar",
        }

    def generate_torchserve_config(project_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "config_path": f"{project_path}/config.properties",
        }

    def create_inference_service_yaml(project_path: str, service_name: str, **kwargs) -> dict:
        return {
            "success": True,
            "yaml_path": f"{project_path}/inference-service.yaml",
        }

    def generate_kserve_config(project_path: str, **kwargs) -> dict:
        return {
            "success": True,
            "config_path": f"{project_path}/kserve-config.yaml",
        }

    # Assign all functions to the module
    module.analyze_project_config = analyze_project_config
    module.create_hydra_config = create_hydra_config
    module.update_hydra_config = update_hydra_config
    module.validate_hydra_config = validate_hydra_config
    module.init_mlflow_experiment = init_mlflow_experiment
    module.start_mlflow_run = start_mlflow_run
    module.log_mlflow_params = log_mlflow_params
    module.log_mlflow_metrics = log_mlflow_metrics
    module.log_mlflow_artifact = log_mlflow_artifact
    module.end_mlflow_run = end_mlflow_run
    module.get_best_mlflow_run = get_best_mlflow_run
    module.register_mlflow_model = register_mlflow_model
    module.analyze_training_results = analyze_training_results
    module.check_accuracy_threshold = check_accuracy_threshold
    module.suggest_improvements = suggest_improvements
    module.create_gradio_interface = create_gradio_interface
    module.deploy_to_huggingface = deploy_to_huggingface
    module.create_litserve_api = create_litserve_api
    module.configure_litserver = configure_litserver
    module.create_fastapi_app = create_fastapi_app
    module.create_lambda_dockerfile = create_lambda_dockerfile
    module.generate_cdk_stack = generate_cdk_stack
    module.create_torchserve_handler = create_torchserve_handler
    module.create_mar_archive = create_mar_archive
    module.generate_torchserve_config = generate_torchserve_config
    module.create_inference_service_yaml = create_inference_service_yaml
    module.generate_kserve_config = generate_kserve_config

    return module


# ============================================================================
# Helper Functions
# ============================================================================


def create_perception_response(
    route: str = "decision",
    pipeline_stage: str = "setup",
    goal_achieved: bool = False,
    entities: dict = None,
    deployment_target: str = None,
) -> dict:
    """Create a mock perception response."""
    base_entities = entities or {}
    if deployment_target:
        base_entities["deployment_target"] = deployment_target

    return {
        "entities": base_entities,
        "pipeline_stage": pipeline_stage,
        "required_tools": ["analyze_project_config", "create_hydra_config"],
        "result_requirement": "Set up MLOps pipeline",
        "original_goal_achieved": goal_achieved,
        "local_goal_achieved": goal_achieved,
        "missing_requirements": [],
        "route": route,
        "confidence": 0.9,
        "reasoning": f"Current stage: {pipeline_stage}",
    }


def create_decision_response(nodes: list, next_step_id: str = "0") -> dict:
    """Create a mock decision response with a plan graph."""
    return {
        "strategy": "sequential",
        "reasoning": "Execute steps in order",
        "plan_graph": {"nodes": nodes},
        "next_step_id": next_step_id,
        "code_variants": {},
    }


# ============================================================================
# End-to-End Test: Complete Train -> Deploy Flow with Gradio
# ============================================================================


class TestTrainDeployFlowGradio:
    """End-to-end tests for complete train -> deploy flow with Gradio target."""

    @pytest.fixture
    def events(self):
        """Capture events emitted during agent execution."""
        return []

    @pytest.fixture
    def event_handler(self, events):
        """Create an event handler that captures all events."""

        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_full_pipeline_train_and_deploy_gradio(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """
        Test the complete pipeline from training to Gradio deployment.

        Flow:
        1. Perception analyzes query and routes to decision
        2. Decision creates setup plan (analyze, hydra config, mlflow init)
        3. Execute setup steps
        4. Perception detects training complete, routes to deploy
        5. Decision creates Gradio deployment plan
        6. Execute deployment steps
        7. Final summary generated
        """
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        # Track call sequence for dynamic responses
        perception_call_count = [0]
        decision_call_count = [0]

        async def mock_perception_run(perception_input, session=None):
            perception_call_count[0] += 1
            call_num = perception_call_count[0]

            # Initial perception - route to decision for setup
            if call_num == 1:
                return create_perception_response(
                    route="decision",
                    pipeline_stage="setup",
                    entities={"project_path": test_project},
                )

            # Perception after step 0 - continue execution
            if call_num == 2:
                return create_perception_response(
                    route="decision",
                    pipeline_stage="config",
                )

            # Perception after step 1 - continue execution
            if call_num == 3:
                return create_perception_response(
                    route="decision",
                    pipeline_stage="config",
                )

            # Perception after step 2 (all setup done) - route to deploy
            if call_num == 4:
                return create_perception_response(
                    route="deploy",
                    pipeline_stage="deploy",
                    goal_achieved=False,
                    entities={"deployment_target": "gradio"},
                    deployment_target="gradio",
                )

            # Any remaining calls - return summarize signal
            return create_perception_response(
                route="summarize",
                pipeline_stage="deploy",
                goal_achieved=True,
            )

        async def mock_decision_run(decision_input, session=None):
            decision_call_count[0] += 1
            call_num = decision_call_count[0]

            # Decision - create setup plan
            if call_num == 1:
                return create_decision_response(
                    [
                        {
                            "id": "0",
                            "description": "Analyze project configuration",
                            "tool": "analyze_project_config",
                            "args": {"project_path": test_project},
                            "depends_on": [],
                        },
                        {
                            "id": "1",
                            "description": "Create Hydra configuration",
                            "tool": "create_hydra_config",
                            "args": {"project_path": test_project},
                            "depends_on": ["0"],
                        },
                        {
                            "id": "2",
                            "description": "Initialize MLflow experiment",
                            "tool": "init_mlflow_experiment",
                            "args": {"experiment_name": "test_experiment"},
                            "depends_on": ["1"],
                        },
                    ]
                )

            # Decision - create deployment plan
            if call_num == 2:
                return create_decision_response(
                    [
                        {
                            "id": "0",
                            "description": "Create Gradio interface",
                            "tool": "create_gradio_interface",
                            "args": {
                                "project_path": test_project,
                                "model_path": f"{test_project}/models/model.pth",
                                "interface_type": "image",
                            },
                            "depends_on": [],
                        },
                        {
                            "id": "1",
                            "description": "Deploy to Hugging Face",
                            "tool": "deploy_to_huggingface",
                            "args": {
                                "project_path": test_project,
                                "repo_name": "test-classifier",
                            },
                            "depends_on": ["0"],
                        },
                    ]
                )

            return create_decision_response([])

        # Mock perception, decision, and summarizer
        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": """# MLOps Pipeline Summary

## Setup Complete
- Project analyzed successfully
- Hydra configuration created
- MLflow experiment initialized

## Deployment
- Gradio interface created at app.py
- Deployed to Hugging Face Spaces: https://huggingface.co/spaces/test-classifier

## Next Steps
1. Run `python app.py` to test locally
2. Visit HuggingFace Spaces to see live demo
"""}

            result = await agent.run(
                query="Set up MLOps pipeline and deploy to Gradio",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Verify the result
        assert "MLOps Pipeline Summary" in result
        assert "Gradio" in result or "gradio" in result.lower()

        # Verify agent status
        assert agent.status == "success"

        # Verify events were emitted
        event_types = [e["type"] for e in events]
        assert "status" in event_types
        assert "phase" in event_types
        assert "perception" in event_types
        assert "plan" in event_types
        assert "step_start" in event_types
        assert "step_complete" in event_types
        assert "deployment_start" in event_types
        assert "deployment_complete" in event_types

        # Verify deployment events contain correct target
        deployment_start_events = [e for e in events if e["type"] == "deployment_start"]
        assert len(deployment_start_events) > 0
        assert deployment_start_events[0]["data"]["target"] == "gradio"


# ============================================================================
# End-to-End Test: Train with Improvement Loop
# ============================================================================


class TestTrainWithImprovementLoop:
    """End-to-end tests for training with improvement loop when below threshold."""

    @pytest.fixture
    def events(self):
        """Capture events emitted during agent execution."""
        return []

    @pytest.fixture
    def event_handler(self, events):
        """Create an event handler that captures all events."""

        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_improvement_loop_reaches_threshold(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """
        Test the improvement loop when initial training is below threshold.

        Flow:
        1. Initial setup and training
        2. Training result below threshold (0.75 < 0.85)
        3. Improvement loop triggered
        4. Suggest improvements and apply config changes
        5. Accuracy improves to meet threshold
        6. Summary generated
        """
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        perception_call_count = [0]

        async def mock_perception_run(perception_input, session=None):
            perception_call_count[0] += 1
            call_num = perception_call_count[0]

            # Initial perception - route to decision
            if call_num == 1:
                return create_perception_response(
                    route="decision",
                    pipeline_stage="training",
                    entities={"project_path": test_project},
                )

            # Perception after step 0
            if call_num == 2:
                # Set accuracy below threshold to trigger improvement
                agent.ctx.experiment_state.current_accuracy = 0.75
                agent.ctx.experiment_state.stage = "evaluation"
                return create_perception_response(
                    route="decision",
                    pipeline_stage="training",
                )

            # Perception after step 1 - trigger improvement loop
            if call_num == 3:
                return create_perception_response(
                    route="improve",
                    pipeline_stage="evaluation",
                )

            # After improvement
            return create_perception_response(
                route="summarize",
                pipeline_stage="evaluation",
                goal_achieved=True,
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Start MLflow run",
                        "tool": "start_mlflow_run",
                        "args": {
                            "experiment_name": "test_experiment",
                            "run_name": "training_run",
                        },
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Analyze training results",
                        "tool": "analyze_training_results",
                        "args": {"results_path": f"{test_project}/results"},
                        "depends_on": ["0"],
                    },
                ]
            )

        async def mock_improvement_suggestion(prompt, **kwargs):
            return {
                "should_retry": True,
                "improvement": {
                    "changes": {"learning_rate": 0.0005, "epochs": 15},
                    "reasoning": "Reduce LR and increase epochs",
                },
                "expected_improvement": {"accuracy_gain": 0.15},
                "hydra_overrides": ["training.learning_rate=0.0005", "training.epochs=15"],
            }

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(
                agent.model_manager, "generate_json", side_effect=mock_improvement_suggestion
            ),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": """# Training Summary

## Improvement Loop
- Initial accuracy: 0.75
- Final accuracy: 0.90 (threshold: 0.85)
- Improvements applied: 1

## Result
Target accuracy threshold met successfully!
"""}

            await agent.run(
                query="Train model until accuracy reaches 0.85",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Verify improvement events were emitted
        event_types = [e["type"] for e in events]
        assert "phase" in event_types

        # Check that improvement phase was triggered
        phase_events = [e for e in events if e["type"] == "phase"]
        phase_messages = [e["data"].get("phase") for e in phase_events]
        assert "improvement" in phase_messages or agent.ctx.experiment_state.improvement_attempt > 0


# ============================================================================
# End-to-End Test: Deploy to Multiple Targets
# ============================================================================


class TestDeployToMultipleTargets:
    """End-to-end tests for deployment to different targets."""

    @pytest.fixture
    def events(self):
        """Capture events emitted during agent execution."""
        return []

    @pytest.fixture
    def event_handler(self, events):
        """Create an event handler that captures all events."""

        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_deploy_to_litserve(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test deployment to LitServe target."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(
                route="deploy",
                pipeline_stage="deploy",
                entities={"deployment_target": "litserve"},
                deployment_target="litserve",
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Create LitServe API",
                        "tool": "create_litserve_api",
                        "args": {
                            "project_path": test_project,
                            "model_path": f"{test_project}/models/model.pth",
                        },
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Configure LitServer",
                        "tool": "configure_litserver",
                        "args": {"project_path": test_project},
                        "depends_on": ["0"],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {
                "summary_markdown": "LitServe deployment complete. Run with `python serve.py`"
            }

            await agent.run(
                query="Deploy model to LitServe",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        assert agent.status == "success"

        # Verify deployment events
        deployment_events = [e for e in events if e["type"] == "deployment_start"]
        assert len(deployment_events) > 0
        assert deployment_events[0]["data"]["target"] == "litserve"

    @pytest.mark.asyncio
    async def test_deploy_to_fastapi_lambda(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test deployment to FastAPI + Lambda target."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(
                route="deploy",
                pipeline_stage="deploy",
                entities={"deployment_target": "fastapi_lambda"},
                deployment_target="fastapi_lambda",
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Create FastAPI app",
                        "tool": "create_fastapi_app",
                        "args": {
                            "project_path": test_project,
                            "model_path": f"{test_project}/models/model.pth",
                        },
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Create Lambda Dockerfile",
                        "tool": "create_lambda_dockerfile",
                        "args": {"project_path": test_project},
                        "depends_on": ["0"],
                    },
                    {
                        "id": "2",
                        "description": "Generate CDK stack",
                        "tool": "generate_cdk_stack",
                        "args": {
                            "project_path": test_project,
                            "stack_name": "ml-inference",
                        },
                        "depends_on": ["1"],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {
                "summary_markdown": "FastAPI + Lambda deployment ready. Run `cdk deploy`"
            }

            await agent.run(
                query="Deploy model to AWS Lambda",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        assert agent.status == "success"

        # Verify correct deployment target
        deployment_events = [e for e in events if e["type"] == "deployment_start"]
        assert len(deployment_events) > 0
        assert deployment_events[0]["data"]["target"] == "fastapi_lambda"

    @pytest.mark.asyncio
    async def test_deploy_to_torchserve(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test deployment to TorchServe target."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(
                route="deploy",
                pipeline_stage="deploy",
                entities={"deployment_target": "torchserve"},
                deployment_target="torchserve",
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Create TorchServe handler",
                        "tool": "create_torchserve_handler",
                        "args": {
                            "project_path": test_project,
                            "model_name": "classifier",
                        },
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Create MAR archive",
                        "tool": "create_mar_archive",
                        "args": {
                            "project_path": test_project,
                            "model_name": "classifier",
                        },
                        "depends_on": ["0"],
                    },
                    {
                        "id": "2",
                        "description": "Generate TorchServe config",
                        "tool": "generate_torchserve_config",
                        "args": {"project_path": test_project},
                        "depends_on": ["1"],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {
                "summary_markdown": "TorchServe deployment ready. Run `torchserve --start`"
            }

            await agent.run(
                query="Deploy model to TorchServe",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        assert agent.status == "success"

        deployment_events = [e for e in events if e["type"] == "deployment_start"]
        assert len(deployment_events) > 0
        assert deployment_events[0]["data"]["target"] == "torchserve"

    @pytest.mark.asyncio
    async def test_deploy_to_kserve(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test deployment to KServe target."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(
                route="deploy",
                pipeline_stage="deploy",
                entities={"deployment_target": "kserve"},
                deployment_target="kserve",
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Create InferenceService YAML",
                        "tool": "create_inference_service_yaml",
                        "args": {
                            "project_path": test_project,
                            "service_name": "classifier-service",
                        },
                        "depends_on": [],
                    },
                    {
                        "id": "1",
                        "description": "Generate KServe config",
                        "tool": "generate_kserve_config",
                        "args": {"project_path": test_project},
                        "depends_on": ["0"],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {
                "summary_markdown": "KServe deployment ready. Apply with `kubectl apply -f`"
            }

            await agent.run(
                query="Deploy model to KServe",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        assert agent.status == "success"

        deployment_events = [e for e in events if e["type"] == "deployment_start"]
        assert len(deployment_events) > 0
        assert deployment_events[0]["data"]["target"] == "kserve"


# ============================================================================
# End-to-End Test: Full Pipeline with Training, Improvement, and Deployment
# ============================================================================


class TestFullPipelineWithImprovement:
    """Test the complete flow: setup -> train -> improve -> deploy."""

    @pytest.fixture
    def events(self):
        """Capture events emitted during agent execution."""
        return []

    @pytest.fixture
    def event_handler(self, events):
        """Create an event handler that captures all events."""

        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_all_phases(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """
        Test the complete pipeline with all phases:
        1. Setup: analyze project, create configs
        2. Training: run training
        3. Summary: generate final summary
        """
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        perception_call_count = [0]
        decision_call_count = [0]

        async def mock_perception_run(perception_input, session=None):
            perception_call_count[0] += 1
            call_num = perception_call_count[0]

            # Phase 1: Setup
            if call_num == 1:
                return create_perception_response(
                    route="decision",
                    pipeline_stage="setup",
                    entities={"project_path": test_project},
                )

            # After setup - summarize
            if call_num == 2:
                return create_perception_response(
                    route="summarize",
                    pipeline_stage="complete",
                    goal_achieved=True,
                )

            return create_perception_response(
                route="summarize",
                pipeline_stage="complete",
                goal_achieved=True,
            )

        async def mock_decision_run(decision_input, session=None):
            decision_call_count[0] += 1
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Analyze project",
                        "tool": "analyze_project_config",
                        "args": {"project_path": test_project},
                        "depends_on": [],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": """# Complete Pipeline Summary

## Setup Phase
- Project analyzed
- Configurations created

## Status
Pipeline completed successfully!
"""}

            await agent.run(
                query="Set up MLOps pipeline, train model, and deploy when ready",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Verify all major phases occurred
        event_types = [e["type"] for e in events]
        phase_events = [e for e in events if e["type"] == "phase"]
        phases = [e["data"].get("phase") for e in phase_events]

        # Should have perception, decision, execution phases
        assert "perception" in event_types
        assert "perception" in phases or "decision" in phases
        assert "execution" in phases

        # Verify agent completed
        assert agent.status in ("success", "partial")


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling during the train -> deploy flow."""

    @pytest.fixture
    def events(self):
        """Capture events emitted during agent execution."""
        return []

    @pytest.fixture
    def event_handler(self, events):
        """Create an event handler that captures all events."""

        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_handles_step_failure_gracefully(
        self, mock_prompts_dir, test_project, events, event_handler
    ):
        """Test that the agent handles step failures gracefully."""
        # Create a tools module with a failing tool
        module = ModuleType("failing_tools")

        def failing_tool(project_path: str) -> dict:
            return {
                "success": False,
                "error": "Tool execution failed intentionally",
            }

        module.failing_tool = failing_tool

        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=module,
            on_event=event_handler,
        )

        perception_call_count = [0]

        async def mock_perception_run(perception_input, session=None):
            perception_call_count[0] += 1
            if perception_call_count[0] == 1:
                return create_perception_response(route="decision", pipeline_stage="setup")
            return create_perception_response(route="summarize", goal_achieved=False)

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Run failing tool",
                        "tool": "failing_tool",
                        "args": {"project_path": test_project},
                        "depends_on": [],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": "Pipeline failed due to tool errors."}

            await agent.run(
                query="Run failing pipeline",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Should have step_failed events
        event_types = [e["type"] for e in events]
        assert "step_failed" in event_types

    @pytest.mark.asyncio
    async def test_handles_empty_decision_plan(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test handling when decision returns empty plan."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(route="decision", pipeline_stage="setup")

        async def mock_decision_run(decision_input, session=None):
            # Return empty plan
            return {"strategy": "sequential", "plan_graph": {"nodes": []}}

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": "No steps to execute."}

            await agent.run(
                query="Test empty plan",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Should have error event for empty plan
        event_types = [e["type"] for e in events]
        assert "error" in event_types


# ============================================================================
# Context State Tracking Tests
# ============================================================================


class TestContextStateTracking:
    """Test that context and experiment state are properly tracked."""

    @pytest.fixture
    def events(self):
        return []

    @pytest.fixture
    def event_handler(self, events):
        async def handler(event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        return handler

    @pytest.mark.asyncio
    async def test_experiment_state_updated_correctly(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test that experiment state is properly updated during execution."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        async def mock_perception_run(perception_input, session=None):
            return create_perception_response(
                route="summarize",
                pipeline_stage="complete",
                goal_achieved=True,
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": "Complete"}

            await agent.run(
                query="Test state tracking",
                project_path=test_project,
                accuracy_threshold=0.90,
            )

        # Verify context was initialized
        assert agent.ctx is not None
        assert agent.ctx.project_path == test_project
        assert agent.ctx.experiment_state.target_accuracy == 0.90

    @pytest.mark.asyncio
    async def test_step_tracking_in_context(
        self, mock_prompts_dir, test_project, mock_tools_module, events, event_handler
    ):
        """Test that steps are properly tracked in context graph."""
        agent = AgentLoop(
            prompts_dir=mock_prompts_dir,
            tools_module=mock_tools_module,
            on_event=event_handler,
        )

        perception_call_count = [0]

        async def mock_perception_run(perception_input, session=None):
            perception_call_count[0] += 1
            if perception_call_count[0] == 1:
                return create_perception_response(route="decision", pipeline_stage="setup")
            return create_perception_response(
                route="summarize", pipeline_stage="complete", goal_achieved=True
            )

        async def mock_decision_run(decision_input, session=None):
            return create_decision_response(
                [
                    {
                        "id": "0",
                        "description": "Test step",
                        "tool": "analyze_project_config",
                        "args": {"project_path": test_project},
                        "depends_on": [],
                    },
                ]
            )

        with (
            patch.object(agent.perception, "run", side_effect=mock_perception_run),
            patch.object(agent.decision, "run", side_effect=mock_decision_run),
            patch.object(agent.summarizer, "summarize", new_callable=AsyncMock) as mock_sum,
        ):
            mock_sum.return_value = {"summary_markdown": "Complete"}

            await agent.run(
                query="Test step tracking",
                project_path=test_project,
                accuracy_threshold=0.85,
            )

        # Verify steps were added to graph
        assert agent.ctx.graph is not None
        # ROOT step should always be present
        assert StepType.ROOT in agent.ctx.graph.nodes
