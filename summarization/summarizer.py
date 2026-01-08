"""
Summarizer Module for MLOps Agent.
Generates final summaries from ML pipeline execution results.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from agent.model_manager import get_model_manager


class Summarizer:
    """
    Summarizer module for generating MLOps execution summaries.
    Creates human-readable reports of pipeline operations.
    """

    def __init__(self, prompt_path: str):
        """
        Initialize Summarizer with prompt template.

        Args:
            prompt_path: Path to summarizer_prompt.txt
        """
        self.prompt_template = self._load_prompt(prompt_path)
        self.model_manager = get_model_manager()

    def _load_prompt(self, path: str) -> str:
        """Load prompt template from file."""
        try:
            return Path(path).read_text()
        except Exception as e:
            print(f"Warning: Could not load summarizer prompt: {e}")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """Generate a summary of the MLOps pipeline execution.
Include:
- What was accomplished
- Key metrics and results
- Any errors or issues
- Next recommended steps
Output as markdown."""

    async def run(
        self,
        s_input: Dict[str, Any],
        session: Any = None
    ) -> Dict[str, Any]:
        """
        Run summarization on execution results.

        Args:
            s_input: Input context with query, results, etc.
            session: Optional session for logging

        Returns:
            Summary output with markdown and metadata
        """
        prompt = self._format_prompt(s_input)

        try:
            # Get LLM response (as text, not JSON)
            response = await self.model_manager.generate_text(prompt)

            result = {
                "summary_markdown": response,
                "goal_achieved": s_input.get("goal_achieved", False),
                "confidence": 0.9,
                "timestamp": datetime.utcnow().isoformat()
            }

            if session:
                session.add_message(
                    role="assistant",
                    content=f"Summary generated",
                    metadata={"module": "summarizer"}
                )

            return result

        except Exception as e:
            print(f"Summarizer error: {e}")
            return self._get_fallback_summary(s_input)

    def _format_prompt(self, s_input: Dict) -> str:
        """Format the summarizer prompt with context."""
        context = json.dumps(s_input, indent=2, default=str)
        return f"{self.prompt_template}\n\nExecution Context:\n```json\n{context}\n```"

    def _get_fallback_summary(self, s_input: Dict) -> Dict:
        """Generate fallback summary when LLM fails."""
        query = s_input.get("original_query", "MLOps operation")
        exp_state = s_input.get("experiment_state", {})

        summary = f"""## MLOps Execution Summary

**Query**: {query}

### Status
- Pipeline Stage: {exp_state.get('stage', 'unknown')}
- Accuracy: {exp_state.get('current_accuracy', 'N/A')} / {exp_state.get('target_accuracy', 'N/A')}
- Improvement Attempts: {exp_state.get('improvement_attempt', 0)}

### Artifacts Created
{self._format_artifacts(exp_state.get('artifacts_created', []))}

### Notes
Summary generation encountered an error. Please review the execution logs for details.
"""
        return {
            "summary_markdown": summary,
            "goal_achieved": False,
            "confidence": 0.5,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _format_artifacts(self, artifacts: list) -> str:
        """Format artifacts list for summary."""
        if not artifacts:
            return "- None"
        return "\n".join(f"- `{a}`" for a in artifacts[:10])

    async def summarize(
        self,
        query: str,
        ctx: Any,
        perception: Dict[str, Any],
        session: Any = None
    ) -> Dict[str, Any]:
        """
        Generate final summary for MLOps operation.

        Args:
            query: Original user query
            ctx: Context manager with execution state
            perception: Latest perception output
            session: Session for logging

        Returns:
            Summary dict with markdown and metadata
        """
        # Mark remaining pending steps as skipped
        for node_id in ctx.graph.nodes:
            node = ctx.graph.nodes[node_id]["data"]
            if node.status == "pending":
                node.status = "skipped"

        # Build summary input
        s_input = {
            "original_query": query,
            "experiment_state": ctx.experiment_state.to_dict(),
            "completed_steps": ctx.get_completed_steps(),
            "failed_steps": ctx.get_failed_steps(),
            "artifacts_created": ctx.experiment_state.artifacts_created,
            "perception": perception,
            "goal_achieved": perception.get("original_goal_achieved", False) or
                           ctx.experiment_state.threshold_met(),
            "globals": {
                k: str(v)[:200] for k, v in ctx.globals.items()
                if k not in ["memory"]
            }
        }

        # Run summarization
        summary = await self.run(s_input, session=session)

        # Attach to context
        ctx.attach_summary(summary)

        # Save session logs
        self._save_session(ctx, session, summary)

        # Print summary
        print("\n" + "=" * 60)
        print("MLOps Pipeline Summary")
        print("=" * 60)
        print(summary.get("summary_markdown", "No summary available"))
        print("=" * 60 + "\n")

        return summary

    def _save_session(self, ctx: Any, session: Any, summary: Dict):
        """Save session logs to disk."""
        try:
            # Create logs directory
            logs_dir = Path("memory/session_logs")
            date_dir = logs_dir / datetime.utcnow().strftime("%Y/%m/%d")
            date_dir.mkdir(parents=True, exist_ok=True)

            # Save session data
            session_file = date_dir / f"{ctx.session_id}.json"
            session_data = {
                "session_id": ctx.session_id,
                "original_query": ctx.original_query,
                "project_path": ctx.project_path,
                "experiment_state": ctx.experiment_state.to_dict(),
                "context_snapshot": ctx.get_context_snapshot(),
                "final_summary": summary.get("summary_markdown", ""),
                "goal_achieved": summary.get("goal_achieved", False),
                "status": "success" if summary.get("goal_achieved") else "partial",
                "timestamp": datetime.utcnow().isoformat()
            }

            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, default=str)

            print(f"  Session saved to: {session_file}")

        except Exception as e:
            print(f"  Warning: Could not save session: {e}")
