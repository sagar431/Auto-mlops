"""
Decision Module for MLOps Agent.
Generates graph-based execution plans with tool calls for ML pipeline automation.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from agent.model_manager import get_model_manager


class Decision:
    """
    Decision module for generating MLOps execution plans.
    Creates a DAG of tool calls with dependencies.
    """

    def __init__(self, prompt_path: str, tools_module: Any = None):
        """
        Initialize Decision with prompt template.

        Args:
            prompt_path: Path to decision_prompt.txt
            tools_module: Optional module containing MCP tools
        """
        self.prompt_template = self._load_prompt(prompt_path)
        self.tools_module = tools_module
        self.model_manager = get_model_manager()

    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        try:
            return Path(path).read_text()
        except Exception as e:
            print(f"Warning: Could not load decision prompt: {e}")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """Generate an execution plan for the MLOps task.
Output JSON with:
- plan_graph: {nodes: [{id, description, tool, args, depends_on}]}
- next_step_id: first step to execute
- code_variants: optional alternative implementations
"""

    async def run(
        self,
        decision_input: Dict[str, Any],
        session: Any = None
    ) -> Dict[str, Any]:
        """
        Run decision to generate execution plan.

        Args:
            decision_input: Input context from build_decision_input
            session: Optional session for logging

        Returns:
            Plan with nodes, next_step_id, and code_variants
        """
        # Format prompt with input context
        prompt = self._format_prompt(decision_input)

        try:
            # Get LLM response
            result = await self.model_manager.generate_json(prompt)

            # Validate and normalize output
            result = self._normalize_output(result)

            # Log to session if available
            if session:
                session.add_message(
                    role="assistant",
                    content=f"Decision: Generated plan with {len(result.get('plan_graph', {}).get('nodes', []))} steps",
                    metadata={"module": "decision", "next_step": result.get("next_step_id")}
                )

            return result

        except Exception as e:
            print(f"Decision error: {e}")
            return self._get_fallback_output(decision_input)

    def _format_prompt(self, decision_input: Dict) -> str:
        """Format the decision prompt with input context."""
        try:
            # Try to use template variables
            formatted = self.prompt_template.format(
                query=decision_input.get("original_query", ""),
                perception=json.dumps(decision_input.get("perception", {}), indent=2),
                state=json.dumps(decision_input.get("state", {}), indent=2),
                completed_steps=json.dumps(decision_input.get("completed_steps", []), indent=2),
                failed_steps=json.dumps(decision_input.get("failed_steps", []), indent=2),
                experiment_state=json.dumps(decision_input.get("experiment_state", {}), indent=2)
            )
            return formatted
        except KeyError:
            # If template formatting fails, append as JSON
            return f"{self.prompt_template}\n\nInput Context:\n```json\n{json.dumps(decision_input, indent=2)}\n```"

    def _normalize_output(self, output: Dict) -> Dict:
        """Normalize and validate decision output."""
        defaults = {
            "strategy": "sequential",
            "reasoning": "",
            "plan_graph": {"nodes": []},
            "next_step_id": "0",
            "code_variants": {}
        }

        for key, default in defaults.items():
            if key not in output:
                output[key] = default

        # Validate plan_graph structure
        if "nodes" not in output["plan_graph"]:
            output["plan_graph"]["nodes"] = []

        # Ensure nodes have required fields
        for node in output["plan_graph"]["nodes"]:
            node.setdefault("id", "0")
            node.setdefault("description", "")
            node.setdefault("tool", None)
            node.setdefault("args", {})
            node.setdefault("depends_on", [])

        # Set next_step_id to first node if not specified
        if not output["next_step_id"] and output["plan_graph"]["nodes"]:
            output["next_step_id"] = output["plan_graph"]["nodes"][0]["id"]

        return output

    def _get_fallback_output(self, decision_input: Dict) -> Dict:
        """Get fallback output when LLM fails."""
        perception = decision_input.get("perception", {})
        entities = perception.get("entities", {})
        project_path = entities.get("project_path", ".")

        # Generate a simple fallback plan based on perception
        stage = perception.get("pipeline_stage", "setup")

        fallback_plans = {
            "setup": [
                {"id": "0", "description": "Analyze project structure", "tool": "analyze_project_config",
                 "args": {"project_path": project_path}, "depends_on": []},
            ],
            "config": [
                {"id": "0", "description": "Create Hydra configuration", "tool": "create_hydra_config",
                 "args": {"project_path": project_path}, "depends_on": []},
            ],
            "training": [
                {"id": "0", "description": "Initialize MLflow experiment", "tool": "init_mlflow_experiment",
                 "args": {"experiment_name": entities.get("experiment_name", "default_experiment")}, "depends_on": []},
            ]
        }

        nodes = fallback_plans.get(stage, fallback_plans["setup"])

        return {
            "strategy": "sequential",
            "reasoning": "Fallback plan due to LLM error",
            "plan_graph": {"nodes": nodes},
            "next_step_id": "0",
            "code_variants": {}
        }


def build_decision_input(
    ctx: Any,
    query: str,
    perception: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build input for decision module.

    Args:
        ctx: Context manager with execution state
        query: Original user query
        perception: Perception output

    Returns:
        Dictionary with all decision inputs
    """
    return {
        "current_time": datetime.utcnow().isoformat(),
        "run_id": f"{ctx.session_id}-D",
        "original_query": query,
        "perception": perception,
        "state": {
            "pipeline_stage": ctx.experiment_state.stage,
            "project_path": ctx.project_path
        },
        "completed_steps": ctx.get_completed_steps(),
        "failed_steps": ctx.get_failed_steps(),
        "experiment_state": ctx.experiment_state.to_dict(),
        "globals_schema": {
            k: {
                "type": type(v).__name__,
                "preview": str(v)[:500] + ("..." if len(str(v)) > 500 else "")
            } for k, v in ctx.globals.items()
        }
    }
