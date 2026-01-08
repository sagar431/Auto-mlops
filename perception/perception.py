"""
Perception Module for MLOps Agent.
Analyzes ML pipeline context and routes to appropriate action.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

from agent.model_manager import get_model_manager


class Perception:
    """
    Perception module for understanding ML pipeline context.
    Extracts entities, determines pipeline stage, and routes to next action.
    """
    
    def __init__(self, prompt_path: str):
        """
        Initialize Perception with prompt template.
        
        Args:
            prompt_path: Path to perception_prompt.txt
        """
        self.prompt_template = self._load_prompt(prompt_path)
        self.model_manager = get_model_manager()
    
    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        try:
            return Path(path).read_text()
        except Exception as e:
            print(f"Warning: Could not load perception prompt: {e}")
            return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """Analyze the MLOps context and provide JSON output with:
- entities: extracted entities (project_path, experiment_name, accuracy_threshold)
- pipeline_stage: setup|config|data|training|evaluation|improvement|deploy
- required_tools: list of tools needed
- route: decision|summarize|improve
- original_goal_achieved: boolean
"""
    
    async def run(
        self,
        perception_input: Dict[str, Any],
        session: Any = None
    ) -> Dict[str, Any]:
        """
        Run perception analysis on input context.
        
        Args:
            perception_input: Input context dictionary
            session: Optional session for logging
        
        Returns:
            Perception output with entities, stage, route, etc.
        """
        # Format prompt with input
        prompt = self.prompt_template.format(
            query=perception_input.get("query", ""),
            stage=perception_input.get("stage", "setup"),
            experiment_state=json.dumps(perception_input.get("experiment_state", {}), indent=2),
            completed_steps=json.dumps(perception_input.get("completed_steps", []), indent=2),
            failed_steps=json.dumps(perception_input.get("failed_steps", []), indent=2),
            tools=json.dumps(perception_input.get("tools", []), indent=2),
            memory=json.dumps(perception_input.get("memory", []), indent=2),
            previous_steps=json.dumps(perception_input.get("completed_steps", []), indent=2)
        )
        
        try:
            # Get LLM response
            result = await self.model_manager.generate_json(prompt)
            
            # Validate and normalize output
            result = self._normalize_output(result)
            
            # Log to session if available
            if session:
                session.add_message(
                    role="assistant",
                    content=f"Perception: {result.get('route', 'unknown')} - {result.get('reasoning', '')}",
                    metadata={"module": "perception", "stage": result.get("pipeline_stage")}
                )
            
            return result
            
        except Exception as e:
            print(f"Perception error: {e}")
            return self._get_fallback_output(perception_input)
    
    def _normalize_output(self, output: Dict) -> Dict:
        """Normalize and validate perception output."""
        # Ensure required fields exist
        defaults = {
            "entities": {},
            "pipeline_stage": "setup",
            "required_tools": [],
            "result_requirement": "",
            "original_goal_achieved": False,
            "local_goal_achieved": False,
            "missing_requirements": [],
            "route": "decision",
            "confidence": 0.5,
            "reasoning": ""
        }
        
        for key, default in defaults.items():
            if key not in output:
                output[key] = default
        
        # Validate route
        valid_routes = ["decision", "summarize", "improve"]
        if output["route"] not in valid_routes:
            output["route"] = "decision"
        
        # Validate stage
        valid_stages = ["setup", "config", "data", "training", "evaluation", "improvement", "deploy"]
        if output["pipeline_stage"] not in valid_stages:
            output["pipeline_stage"] = "setup"
        
        return output
    
    def _get_fallback_output(self, perception_input: Dict) -> Dict:
        """Get fallback output when LLM fails."""
        return {
            "entities": {},
            "pipeline_stage": perception_input.get("stage", "setup"),
            "required_tools": [],
            "result_requirement": "Continue with next step",
            "original_goal_achieved": False,
            "local_goal_achieved": False,
            "missing_requirements": [],
            "route": "decision",
            "confidence": 0.3,
            "reasoning": "Fallback due to LLM error"
        }


def build_perception_input(
    query: str,
    memory: list,
    ctx: Any,
    snapshot_type: str = "initial"
) -> Dict[str, Any]:
    """
    Build input for perception module.
    
    Args:
        query: User query
        memory: Memory search results
        ctx: Context manager
        snapshot_type: "initial" or "step_result"
    
    Returns:
        Dictionary with all perception inputs
    """
    # Get available tools
    tools = [
        "analyze_project_config", "create_hydra_config", "update_hydra_config", "validate_hydra_config",
        "init_mlflow_experiment", "start_mlflow_run", "log_mlflow_params", "log_mlflow_metrics",
        "log_mlflow_artifact", "register_mlflow_model", "get_best_mlflow_run", "end_mlflow_run",
        "init_dvc_repo", "configure_dvc_remote", "add_data_to_dvc", "create_dvc_pipeline",
        "dvc_push", "dvc_pull", "dvc_reproduce",
        "create_ml_dockerfile", "build_ml_docker_image", "run_training_container", "push_docker_image",
        "create_github_workflow", "add_workflow_step",
        "analyze_training_results", "suggest_improvements", "check_accuracy_threshold"
    ]
    
    return {
        "query": query,
        "stage": ctx.experiment_state.stage,
        "experiment_state": ctx.experiment_state.to_dict(),
        "completed_steps": ctx.get_completed_steps(),
        "failed_steps": ctx.get_failed_steps(),
        "tools": tools,
        "memory": memory[:5] if memory else [],  # Limit memory items
        "snapshot_type": snapshot_type,
        "project_path": ctx.project_path
    }
