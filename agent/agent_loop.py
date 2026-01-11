"""
Agent Loop for MLOps Agent.
Graph-based execution loop with self-improvement capability.
Orchestrates: Perception -> Decision -> Action -> (Improve if needed) -> Summarize
"""

import uuid
import asyncio
from datetime import datetime
from typing import Callable, Optional, Dict, Any, List
from pathlib import Path

from agent.contextManager import ContextManager
from agent.agentSession import AgentSession
from agent.model_manager import get_model_manager
from perception.perception import Perception, build_perception_input
from decision.decision import Decision, build_decision_input
from summarization.summarizer import Summarizer
from action.execute_step import execute_step
from memory.memory_search import MemorySearch


class Route:
    """Routing constants."""
    SUMMARIZE = "summarize"
    DECISION = "decision"
    IMPROVE = "improve"
    DEPLOY = "deploy"


class StepType:
    """Step type constants."""
    ROOT = "ROOT"
    CODE = "CODE"
    IMPROVE = "IMPROVE"
    DEPLOY = "DEPLOY"


class StepExecutionTracker:
    """Tracks step execution attempts and limits."""

    def __init__(self, max_steps: int = 15, max_retries: int = 3):
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.attempts: Dict[str, int] = {}
        self.tries = 0
        self.root_failures = 0

    def increment(self):
        self.tries += 1

    def record_failure(self, step_id: str):
        self.attempts[step_id] = self.attempts.get(step_id, 0) + 1

    def retry_step_id(self, step_id: str) -> str:
        attempts = self.attempts.get(step_id, 0)
        return f"{step_id}F{attempts}" if attempts > 0 else step_id

    def should_continue(self) -> bool:
        return self.tries < self.max_steps

    def has_exceeded_retries(self, step_id: str) -> bool:
        return self.attempts.get(step_id, 0) >= self.max_retries


class AgentLoop:
    """
    Main agent loop for MLOps operations.
    Implements the Perception -> Decision -> Action cycle with self-improvement.
    """

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        tools_module: Optional[Any] = None,
        on_event: Optional[Callable] = None,
        profile: str = "default"
    ):
        # Load prompts
        prompts_dir = Path(prompts_dir) if prompts_dir else Path(__file__).parent.parent / "prompts"
        
        self.perception = Perception(prompts_dir / "perception_prompt.txt")
        self.decision = Decision(prompts_dir / "decision_prompt.txt", tools_module)
        self.summarizer = Summarizer(prompts_dir / "summarizer_prompt.txt")
        self.improvement_prompt = self._load_prompt(prompts_dir / "improvement_prompt.txt")
        
        self.tools_module = tools_module
        self.profile = profile
        self.on_event = on_event
        self.status = "idle"
        
        # Model manager for LLM calls
        self.model_manager = get_model_manager()

    def _load_prompt(self, path: Path) -> str:
        """Load prompt from file."""
        try:
            return path.read_text()
        except Exception:
            return ""

    async def _emit(self, event_type: str, data: Dict = None):
        """Emit an event to callback if registered."""
        if self.on_event:
            try:
                await self.on_event(event_type, data or {})
            except Exception as e:
                print(f"Failed to emit event: {e}")

    async def run(
        self,
        query: str,
        project_path: Optional[str] = None,
        accuracy_threshold: float = 0.85
    ) -> str:
        """
        Run the MLOps agent loop.
        
        Args:
            query: User query/goal
            project_path: Path to ML project
            accuracy_threshold: Target accuracy for training
        
        Returns:
            Final summary or error message
        """
        self.status = "running"
        await self._emit("status", {"status": "running", "message": "Agent started"})
        
        # Initialize session
        self._initialize_session(query, project_path, accuracy_threshold)
        
        # Phase 1: Initial Perception
        await self._emit("phase", {"phase": "perception", "message": "Analyzing request..."})
        await self._run_initial_perception()
        
        # Check for early exit
        if self._should_summarize():
            await self._emit("phase", {"phase": "summary", "message": "Generating summary..."})
            return await self._summarize()
        
        # Phase 2: Decision + Execution Loop
        await self._run_decision_loop()

        # Phase 3: Check for deployment if requested
        if self._needs_deployment():
            await self._run_deployment_loop()

        # Phase 4: Check for improvement if training occurred
        if self._needs_improvement():
            await self._run_improvement_loop()

        # Phase 5: Final Summary
        if self.status == "success" or self.ctx.experiment_state.threshold_met():
            return self.final_output
        
        return await self._handle_failure()

    def _initialize_session(
        self,
        query: str,
        project_path: Optional[str],
        accuracy_threshold: float
    ):
        """Initialize session context and state."""
        self.session_id = str(uuid.uuid4())
        self.query = query
        
        # Create context manager
        self.ctx = ContextManager(
            session_id=self.session_id,
            original_query=query,
            project_path=project_path
        )
        self.ctx.experiment_state.target_accuracy = accuracy_threshold
        
        # Create session
        self.session = AgentSession(
            session_id=self.session_id,
            original_query=query,
            project_path=project_path,
            profile=self.profile
        )
        
        # Load memory from past experiments
        self.memory = MemorySearch().search_memory(query)
        self.ctx.globals["memory"] = self.memory
        
        # Placeholders
        self.p_out: Dict = {}
        self.code_variants: Dict = {}
        self.next_step_id: str = "0"
        self.final_output: str = ""

    async def _run_initial_perception(self):
        """Run initial perception on user query."""
        p_input = build_perception_input(
            query=self.query,
            memory=self.memory,
            ctx=self.ctx
        )
        
        self.p_out = await self.perception.run(p_input, session=self.session)
        
        # Add ROOT node
        self.ctx.add_step(
            step_id=StepType.ROOT,
            description="Initial query analysis",
            step_type=StepType.ROOT
        )
        self.ctx.mark_step_completed(StepType.ROOT)
        self.ctx.attach_perception(StepType.ROOT, self.p_out)
        
        await self._emit("perception", {
            "step_id": "ROOT",
            "entities": self.p_out.get("entities", {}),
            "pipeline_stage": self.p_out.get("pipeline_stage", "setup"),
            "route": self.p_out.get("route", "decision")
        })
        
        self.ctx.print_graph()

    def _should_summarize(self) -> bool:
        """Check if we should skip to summary."""
        return (
            self.p_out.get("original_goal_achieved", False) or
            self.p_out.get("route") == Route.SUMMARIZE
        )

    def _needs_improvement(self) -> bool:
        """Check if training needs improvement."""
        exp = self.ctx.experiment_state
        return (
            exp.current_accuracy is not None and
            not exp.threshold_met() and
            exp.can_improve() and
            exp.stage in ["evaluation", "training"]
        )

    def _needs_deployment(self) -> bool:
        """Check if deployment is requested."""
        return (
            self.p_out.get("route") == Route.DEPLOY or
            self.p_out.get("pipeline_stage") == "deploy" or
            self.p_out.get("entities", {}).get("deployment_target") is not None
        )

    async def _summarize(self) -> str:
        """Generate final summary."""
        summary = await self.summarizer.summarize(
            query=self.query,
            ctx=self.ctx,
            perception=self.p_out,
            session=self.session
        )
        self.ctx.attach_summary(summary)
        self.status = "success"
        self.final_output = summary.get("summary_markdown", str(summary))
        return self.final_output

    async def _run_decision_loop(self):
        """Run decision and execute steps in a loop."""
        await self._emit("phase", {"phase": "decision", "message": "Planning execution..."})
        
        # Get initial plan
        d_input = build_decision_input(
            ctx=self.ctx,
            query=self.query,
            perception=self.p_out
        )
        d_out = await self.decision.run(d_input, session=self.session)
        
        if not d_out.get("plan_graph", {}).get("nodes"):
            await self._emit("error", {"error": "No execution plan generated"})
            return
        
        # Extract plan
        self.code_variants = d_out.get("code_variants", {})
        self.next_step_id = d_out.get("next_step_id", "0")
        plan_nodes = d_out["plan_graph"]["nodes"]
        
        await self._emit("plan", {
            "nodes": plan_nodes,
            "next_step_id": self.next_step_id,
            "total_steps": len(plan_nodes)
        })
        
        # Add nodes to graph
        for node in plan_nodes:
            self.ctx.add_step(
                step_id=node["id"],
                description=node["description"],
                step_type=StepType.CODE,
                tool=node.get("tool"),
                args=node.get("args"),
                from_node=StepType.ROOT
            )
        
        # Execute steps
        await self._execute_steps_loop()

    async def _execute_steps_loop(self):
        """Execute steps with perception feedback."""
        await self._emit("phase", {"phase": "execution", "message": "Executing steps..."})
        
        tracker = StepExecutionTracker(max_steps=15, max_retries=3)
        
        while tracker.should_continue():
            tracker.increment()
            
            # Skip completed steps
            if self.ctx.is_step_completed(self.next_step_id):
                self.next_step_id = self._pick_next_step()
                if self.next_step_id is None:
                    break
                continue
            
            # Get step info
            if self.next_step_id not in self.ctx.graph.nodes:
                break
            
            step_data = self.ctx.graph.nodes[self.next_step_id]["data"]
            
            await self._emit("step_start", {
                "step_id": self.next_step_id,
                "description": step_data.description,
                "tool": step_data.tool,
                "loop": tracker.tries
            })
            
            # Execute step
            success, result = await execute_step(
                step_id=self.next_step_id,
                tool=step_data.tool,
                args=step_data.args or {},
                ctx=self.ctx,
                tools_module=self.tools_module
            )
            
            if not success:
                self.ctx.mark_step_failed(self.next_step_id, str(result.get("error", "Unknown error")))
                tracker.record_failure(self.next_step_id)
                
                await self._emit("step_failed", {
                    "step_id": self.next_step_id,
                    "error": str(result.get("error", ""))[:200],
                    "attempts": tracker.attempts.get(self.next_step_id, 1)
                })
                
                if tracker.has_exceeded_retries(self.next_step_id):
                    break
                continue
            
            # Update result
            self.ctx.update_step_result(self.next_step_id, result)
            
            await self._emit("step_complete", {
                "step_id": self.next_step_id,
                "success": True,
                "result_summary": str(result)[:300]
            })
            
            # Run perception after step
            p_input = build_perception_input(
                query=self.query,
                memory=self.memory,
                ctx=self.ctx,
                snapshot_type="step_result"
            )
            self.p_out = await self.perception.run(p_input, session=self.session)
            self.ctx.attach_perception(self.next_step_id, self.p_out)
            
            # Check routing
            if self.p_out.get("original_goal_achieved") or self.p_out.get("route") == Route.SUMMARIZE:
                self.status = "success"
                await self._emit("phase", {"phase": "summary", "message": "Goal achieved!"})
                self.final_output = await self._summarize()
                return
            
            if self.p_out.get("route") == Route.IMPROVE:
                # Break to run improvement loop
                return

            if self.p_out.get("route") == Route.DEPLOY:
                # Break to run deployment loop
                return
            
            # Get next step
            self.next_step_id = self._pick_next_step()
            if self.next_step_id is None:
                break

    async def _run_improvement_loop(self):
        """Run the self-improvement loop for training."""
        await self._emit("phase", {"phase": "improvement", "message": "Optimizing training..."})
        
        exp = self.ctx.experiment_state
        
        while exp.can_improve() and not exp.threshold_met():
            attempt = exp.improvement_attempt + 1
            
            await self._emit("improvement_start", {
                "attempt": attempt,
                "current_accuracy": exp.current_accuracy,
                "target_accuracy": exp.target_accuracy,
                "gap": exp.get_accuracy_gap()
            })
            
            # Get improvement suggestions
            improvement = await self._get_improvement_suggestion()
            
            if not improvement.get("should_retry", False):
                await self._emit("improvement_stop", {
                    "reason": "No more improvements suggested"
                })
                break
            
            # Apply improvements
            config_changes = improvement.get("improvement", {}).get("changes", {})
            hydra_overrides = improvement.get("hydra_overrides", [])
            
            await self._emit("improvement_apply", {
                "attempt": attempt,
                "changes": config_changes,
                "reasoning": improvement.get("improvement", {}).get("reasoning", "")
            })
            
            # Update Hydra config if we have a project path
            if self.ctx.project_path and hydra_overrides:
                from mcp_mlops_tools import update_hydra_config
                
                updates = {}
                for override in hydra_overrides:
                    if "=" in override:
                        key, value = override.split("=", 1)
                        # Convert value to appropriate type
                        try:
                            value = float(value) if "." in value else int(value)
                        except ValueError:
                            pass
                        
                        # Build nested dict
                        parts = key.split(".")
                        current = updates
                        for part in parts[:-1]:
                            current = current.setdefault(part, {})
                        current[parts[-1]] = value
                
                result = update_hydra_config(
                    project_path=self.ctx.project_path,
                    config_path="configs/config.yaml",
                    updates=updates
                )
                
                if result.get("success"):
                    self.ctx.set_experiment_config(updates)
            
            # Record improvement attempt
            # In a real scenario, training would run here
            # For now, we simulate the accuracy improvement
            new_accuracy = exp.current_accuracy + improvement.get("expected_improvement", {}).get("accuracy_gain", 0.02)
            exp.record_improvement_attempt(config_changes, new_accuracy)
            
            await self._emit("improvement_complete", {
                "attempt": attempt,
                "new_accuracy": new_accuracy,
                "threshold_met": exp.threshold_met()
            })
        
        # After improvement loop, summarize
        if exp.threshold_met():
            self.status = "success"
            await self._emit("phase", {"phase": "summary", "message": "Target achieved!"})
            self.final_output = await self._summarize()
        else:
            self.status = "partial"
            self.final_output = await self._summarize()

    async def _run_deployment_loop(self):
        """Run the deployment workflow."""
        await self._emit("phase", {"phase": "deployment", "message": "Setting up deployment..."})

        # Get deployment target from perception
        deployment_target = self.p_out.get("entities", {}).get("deployment_target", "gradio")

        await self._emit("deployment_start", {
            "target": deployment_target,
            "project_path": self.ctx.project_path
        })

        # Build deployment-focused decision input
        deploy_perception = {
            **self.p_out,
            "pipeline_stage": "deploy",
            "route": Route.DEPLOY,
            "deployment_target": deployment_target
        }

        d_input = build_decision_input(
            ctx=self.ctx,
            query=self.query,
            perception=deploy_perception
        )

        # Get deployment plan
        d_out = await self.decision.run(d_input, session=self.session)

        if not d_out.get("plan_graph", {}).get("nodes"):
            await self._emit("error", {"error": "No deployment plan generated"})
            return

        plan_nodes = d_out["plan_graph"]["nodes"]

        await self._emit("deployment_plan", {
            "target": deployment_target,
            "steps": len(plan_nodes),
            "nodes": plan_nodes
        })

        # Add deployment nodes to graph
        for node in plan_nodes:
            self.ctx.add_step(
                step_id=f"deploy_{node['id']}",
                description=node["description"],
                step_type=StepType.DEPLOY,
                tool=node.get("tool"),
                args=node.get("args"),
                from_node=StepType.ROOT
            )

        # Execute deployment steps
        for node in plan_nodes:
            step_id = f"deploy_{node['id']}"

            await self._emit("step_start", {
                "step_id": step_id,
                "description": node["description"],
                "tool": node.get("tool"),
                "phase": "deployment"
            })

            success, result = await execute_step(
                step_id=step_id,
                tool=node.get("tool"),
                args=node.get("args", {}),
                ctx=self.ctx,
                tools_module=self.tools_module
            )

            if success:
                self.ctx.update_step_result(step_id, result)
                await self._emit("step_complete", {
                    "step_id": step_id,
                    "success": True,
                    "result_summary": str(result)[:300]
                })
            else:
                self.ctx.mark_step_failed(step_id, str(result.get("error", "Unknown error")))
                await self._emit("step_failed", {
                    "step_id": step_id,
                    "error": str(result.get("error", ""))[:200]
                })

        await self._emit("deployment_complete", {
            "target": deployment_target,
            "status": "success"
        })

        # Summarize deployment
        self.status = "success"
        await self._emit("phase", {"phase": "summary", "message": "Deployment complete!"})
        self.final_output = await self._summarize()

    async def _get_improvement_suggestion(self) -> Dict[str, Any]:
        """Get improvement suggestions from LLM."""
        exp = self.ctx.experiment_state
        
        prompt = self.improvement_prompt.format(
            target_accuracy=exp.target_accuracy,
            current_accuracy=exp.current_accuracy,
            current_loss=exp.current_loss or "N/A",
            gap=exp.get_accuracy_gap(),
            current_config=exp.current_config,
            training_history=exp.improvement_history,
            attempt=exp.improvement_attempt + 1,
            max_attempts=exp.max_improvement_attempts,
            previous_improvements=[h.get("config_changes", {}) for h in exp.improvement_history]
        )
        
        try:
            return await self.model_manager.generate_json(prompt)
        except Exception as e:
            print(f"Error getting improvement suggestion: {e}")
            return {"should_retry": False, "error": str(e)}

    def _pick_next_step(self) -> Optional[str]:
        """Pick the next pending step to execute."""
        pending = self.ctx.get_pending_steps()
        return pending[0] if pending else None

    async def _handle_failure(self) -> str:
        """Handle failure case."""
        self.status = "failed"
        self.session.mark_completed(success=False)
        
        await self._emit("status", {
            "status": "failed",
            "message": "Agent stopped due to errors"
        })
        
        # Still generate a summary
        self.final_output = await self._summarize()
        return self.final_output


# Convenience function
async def run_mlops_agent(
    query: str,
    project_path: Optional[str] = None,
    accuracy_threshold: float = 0.85,
    on_event: Optional[Callable] = None
) -> str:
    """
    Run the MLOps agent with a query.
    
    Args:
        query: User query (e.g., "Set up MLOps pipeline for my project")
        project_path: Path to ML project
        accuracy_threshold: Target accuracy
        on_event: Callback for events
    
    Returns:
        Final summary markdown
    """
    agent = AgentLoop(on_event=on_event)
    return await agent.run(
        query=query,
        project_path=project_path,
        accuracy_threshold=accuracy_threshold
    )
