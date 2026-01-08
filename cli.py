#!/usr/bin/env python3
"""
MLOps Agent CLI - AI-powered ML Pipeline Automation

Usage:
    python agent.py "Set up MLOps pipeline for my project"
    python agent.py --project /path/to/project --threshold 0.85
    python agent.py --interactive
    python agent.py --help
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Rich console for beautiful output
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint

# Agent imports
from agent.agent_loop import AgentLoop, run_mlops_agent
from agent.contextManager import ContextManager

console = Console()

# Status icons
ICONS = {
    "running": "🔄",
    "success": "✅",
    "failed": "❌",
    "pending": "⏳",
    "perception": "🔍",
    "decision": "🧠",
    "execution": "⚡",
    "improvement": "📈",
    "summary": "📋",
    "step_start": "▶️",
    "step_complete": "✓",
    "step_failed": "✗",
}


class AgentEventHandler:
    """Handles agent events and displays progress."""

    def __init__(self):
        self.current_phase = "idle"
        self.steps_completed = 0
        self.steps_total = 0
        self.current_step = None
        self.errors = []
        self.start_time = None

    async def handle_event(self, event_type: str, data: Dict[str, Any]):
        """Handle events from the agent loop."""

        if event_type == "status":
            status = data.get("status", "unknown")
            message = data.get("message", "")
            if status == "running":
                self.start_time = datetime.now()
                console.print(f"\n{ICONS['running']} [bold cyan]Agent Started[/bold cyan]: {message}")
            elif status == "failed":
                console.print(f"\n{ICONS['failed']} [bold red]Agent Failed[/bold red]: {message}")

        elif event_type == "phase":
            phase = data.get("phase", "unknown")
            message = data.get("message", "")
            icon = ICONS.get(phase, "📌")
            self.current_phase = phase
            console.print(f"\n{icon} [bold yellow]{phase.upper()}[/bold yellow]: {message}")

        elif event_type == "perception":
            entities = data.get("entities", {})
            stage = data.get("pipeline_stage", "setup")
            route = data.get("route", "decision")
            console.print(f"   Stage: [cyan]{stage}[/cyan] → Route: [cyan]{route}[/cyan]")
            if entities:
                console.print(f"   Entities: {entities}")

        elif event_type == "plan":
            nodes = data.get("nodes", [])
            self.steps_total = data.get("total_steps", len(nodes))
            console.print(f"\n📋 [bold]Execution Plan[/bold] ({self.steps_total} steps):")
            for node in nodes[:5]:  # Show first 5 steps
                console.print(f"   [{node['id']}] {node['description'][:50]}...")
            if len(nodes) > 5:
                console.print(f"   ... and {len(nodes) - 5} more steps")

        elif event_type == "step_start":
            step_id = data.get("step_id", "?")
            description = data.get("description", "")[:50]
            tool = data.get("tool", "")
            self.current_step = step_id
            console.print(f"\n{ICONS['step_start']} [bold]Step {step_id}[/bold]: {description}")
            if tool:
                console.print(f"   Tool: [green]{tool}[/green]")

        elif event_type == "step_complete":
            step_id = data.get("step_id", "?")
            self.steps_completed += 1
            console.print(f"   {ICONS['step_complete']} [green]Completed[/green] ({self.steps_completed}/{self.steps_total})")

        elif event_type == "step_failed":
            step_id = data.get("step_id", "?")
            error = data.get("error", "Unknown error")[:100]
            attempts = data.get("attempts", 1)
            self.errors.append(error)
            console.print(f"   {ICONS['step_failed']} [red]Failed[/red] (attempt {attempts}): {error}")

        elif event_type == "improvement_start":
            attempt = data.get("attempt", 1)
            current = data.get("current_accuracy", 0)
            target = data.get("target_accuracy", 0.85)
            gap = data.get("gap", 0)
            console.print(f"\n{ICONS['improvement']} [bold yellow]Improvement Attempt {attempt}[/bold yellow]")
            console.print(f"   Current: [cyan]{current:.2%}[/cyan] → Target: [cyan]{target:.2%}[/cyan] (gap: {gap:.2%})")

        elif event_type == "improvement_apply":
            changes = data.get("changes", {})
            reasoning = data.get("reasoning", "")[:100]
            console.print(f"   Applying: {list(changes.keys())}")
            if reasoning:
                console.print(f"   Reason: {reasoning}")

        elif event_type == "improvement_complete":
            new_accuracy = data.get("new_accuracy", 0)
            threshold_met = data.get("threshold_met", False)
            status_icon = ICONS['success'] if threshold_met else ICONS['pending']
            console.print(f"   {status_icon} New accuracy: [bold]{new_accuracy:.2%}[/bold]")

        elif event_type == "error":
            error = data.get("error", "Unknown error")
            console.print(f"\n{ICONS['failed']} [bold red]Error[/bold red]: {error}")

    def print_summary(self, result: str, success: bool):
        """Print final summary."""
        duration = ""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            duration = f" in {elapsed:.1f}s"

        status = "[bold green]SUCCESS[/bold green]" if success else "[bold red]FAILED[/bold red]"

        console.print(f"\n{'='*60}")
        console.print(f"{ICONS['summary']} [bold]Final Summary[/bold] - {status}{duration}")
        console.print(f"{'='*60}\n")

        # Render result as markdown
        console.print(Markdown(result))

        if self.errors:
            console.print(f"\n[yellow]Warnings/Errors ({len(self.errors)}):[/yellow]")
            for err in self.errors[:5]:
                console.print(f"  • {err[:80]}")


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                     🤖 MLOps Agent                            ║
║         AI-Powered ML Pipeline Automation                     ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner, style="cyan"))


def print_help():
    """Print help information."""
    help_text = """
[bold]Available Commands:[/bold]

  [cyan]Natural Language Queries:[/cyan]
    "Set up MLOps pipeline for my project"
    "Add Hydra config with learning_rate=0.001"
    "Initialize MLflow experiment named cat-dog-classifier"
    "Create DVC pipeline with S3 remote"
    "Build Docker image for training"
    "Create GitHub Actions CI/CD workflow"
    "Train until accuracy reaches 85%"

  [cyan]Special Commands:[/cyan]
    help     - Show this help
    status   - Show current experiment status
    history  - Show past sessions
    exit     - Exit the agent

  [cyan]Examples:[/cyan]
    > Set up complete MLOps pipeline for /path/to/my/project with 85% accuracy target
    > Add data versioning with DVC and push to S3
    > Create training workflow that triggers on push to main
    """
    console.print(Panel(help_text, title="Help", border_style="green"))


async def run_agent_with_events(
    query: str,
    project_path: Optional[str] = None,
    accuracy_threshold: float = 0.85
) -> str:
    """Run agent with event handling."""
    handler = AgentEventHandler()

    agent = AgentLoop(on_event=handler.handle_event)

    try:
        result = await agent.run(
            query=query,
            project_path=project_path,
            accuracy_threshold=accuracy_threshold
        )

        success = agent.status == "success"
        handler.print_summary(result, success)

        return result

    except Exception as e:
        console.print(f"\n{ICONS['failed']} [bold red]Agent Error[/bold red]: {str(e)}")
        raise


async def interactive_mode(project_path: Optional[str] = None, accuracy_threshold: float = 0.85):
    """Run agent in interactive REPL mode."""
    print_banner()

    console.print("[dim]Type 'help' for available commands, 'exit' to quit[/dim]\n")

    if project_path:
        console.print(f"[dim]Project: {project_path}[/dim]")
    console.print(f"[dim]Accuracy threshold: {accuracy_threshold:.0%}[/dim]\n")

    while True:
        try:
            # Get user input
            query = console.input("[bold cyan]mlops>[/bold cyan] ").strip()

            if not query:
                continue

            # Handle special commands
            if query.lower() == "exit":
                console.print("\n[dim]Goodbye! 👋[/dim]")
                break

            elif query.lower() == "help":
                print_help()
                continue

            elif query.lower() == "status":
                console.print("[dim]Status: Idle (no active experiment)[/dim]")
                continue

            elif query.lower() == "history":
                from memory.memory_search import MemorySearch
                ms = MemorySearch()
                if ms.index_data:
                    table = Table(title="Recent Sessions")
                    table.add_column("Session ID", style="cyan")
                    table.add_column("Query", style="white")
                    table.add_column("Status", style="green")
                    for session in ms.index_data[-5:]:
                        table.add_row(
                            session["session_id"][:8],
                            session["original_query"][:40] + "...",
                            session.get("status", "unknown")
                        )
                    console.print(table)
                else:
                    console.print("[dim]No sessions found[/dim]")
                continue

            # Run agent with query
            await run_agent_with_events(
                query=query,
                project_path=project_path,
                accuracy_threshold=accuracy_threshold
            )

            console.print()  # Add spacing

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


async def single_command_mode(
    query: str,
    project_path: Optional[str] = None,
    accuracy_threshold: float = 0.85
):
    """Run agent with a single command."""
    print_banner()

    console.print(f"[bold]Query:[/bold] {query}")
    if project_path:
        console.print(f"[bold]Project:[/bold] {project_path}")
    console.print(f"[bold]Accuracy Target:[/bold] {accuracy_threshold:.0%}")

    await run_agent_with_events(
        query=query,
        project_path=project_path,
        accuracy_threshold=accuracy_threshold
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MLOps Agent - AI-powered ML Pipeline Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py "Set up MLOps pipeline for my cat-dog classifier"
  python agent.py --project ./my_project --threshold 0.90 "Train until accuracy target"
  python agent.py --interactive --project ./my_project
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query for the agent"
    )

    parser.add_argument(
        "-p", "--project",
        type=str,
        default=None,
        help="Path to ML project directory"
    )

    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.85,
        help="Accuracy threshold (default: 0.85)"
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive REPL mode"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="MLOps Agent v1.0.0"
    )

    args = parser.parse_args()

    # Validate project path if provided
    if args.project:
        project_path = Path(args.project).resolve()
        if not project_path.exists():
            console.print(f"[red]Error: Project path does not exist: {project_path}[/red]")
            sys.exit(1)
        args.project = str(project_path)

    # Run in appropriate mode
    try:
        if args.interactive:
            asyncio.run(interactive_mode(args.project, args.threshold))
        elif args.query:
            asyncio.run(single_command_mode(args.query, args.project, args.threshold))
        else:
            # No query provided, enter interactive mode
            asyncio.run(interactive_mode(args.project, args.threshold))

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
