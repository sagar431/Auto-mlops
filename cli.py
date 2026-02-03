#!/usr/bin/env python3
"""
MLOps Agent CLI - AI-powered ML Pipeline Automation

Usage:
    python cli.py "Set up MLOps pipeline for my project"
    python cli.py --project /path/to/project --threshold 0.85
    python cli.py --interactive
    python cli.py admin create-user --username admin --email admin@example.com --password secret
    python cli.py admin create-key --name "My API Key"
    python cli.py admin list-users
    python cli.py admin revoke-key --key-id <key_id>
    python cli.py --help
"""

import argparse
import asyncio
import getpass
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

# Rich console for beautiful output
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

# Agent imports
from agent.agent_loop import AgentLoop

console = Console()

# Default API server URL
DEFAULT_API_URL = os.environ.get("MLOPS_API_URL", "http://localhost:8000")

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

    async def handle_event(self, event_type: str, data: dict[str, Any]):
        """Handle events from the agent loop."""

        if event_type == "status":
            status = data.get("status", "unknown")
            message = data.get("message", "")
            if status == "running":
                self.start_time = datetime.now()
                console.print(
                    f"\n{ICONS['running']} [bold cyan]Agent Started[/bold cyan]: {message}"
                )
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
            console.print(
                f"   {ICONS['step_complete']} [green]Completed[/green] ({self.steps_completed}/{self.steps_total})"
            )

        elif event_type == "step_failed":
            step_id = data.get("step_id", "?")
            error = data.get("error", "Unknown error")[:100]
            attempts = data.get("attempts", 1)
            self.errors.append(error)
            console.print(
                f"   {ICONS['step_failed']} [red]Failed[/red] (attempt {attempts}): {error}"
            )

        elif event_type == "improvement_start":
            attempt = data.get("attempt", 1)
            current = data.get("current_accuracy", 0)
            target = data.get("target_accuracy", 0.85)
            gap = data.get("gap", 0)
            console.print(
                f"\n{ICONS['improvement']} [bold yellow]Improvement Attempt {attempt}[/bold yellow]"
            )
            console.print(
                f"   Current: [cyan]{current:.2%}[/cyan] → Target: [cyan]{target:.2%}[/cyan] (gap: {gap:.2%})"
            )

        elif event_type == "improvement_apply":
            changes = data.get("changes", {})
            reasoning = data.get("reasoning", "")[:100]
            console.print(f"   Applying: {list(changes.keys())}")
            if reasoning:
                console.print(f"   Reason: {reasoning}")

        elif event_type == "improvement_complete":
            new_accuracy = data.get("new_accuracy", 0)
            threshold_met = data.get("threshold_met", False)
            status_icon = ICONS["success"] if threshold_met else ICONS["pending"]
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
    query: str, project_path: str | None = None, accuracy_threshold: float = 0.85
) -> str:
    """Run agent with event handling."""
    handler = AgentEventHandler()

    agent = AgentLoop(on_event=handler.handle_event)

    try:
        result = await agent.run(
            query=query, project_path=project_path, accuracy_threshold=accuracy_threshold
        )

        success = agent.status == "success"
        handler.print_summary(result, success)

        return result

    except Exception as e:
        console.print(f"\n{ICONS['failed']} [bold red]Agent Error[/bold red]: {str(e)}")
        raise


async def interactive_mode(project_path: str | None = None, accuracy_threshold: float = 0.85):
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
                            session.get("status", "unknown"),
                        )
                    console.print(table)
                else:
                    console.print("[dim]No sessions found[/dim]")
                continue

            # Run agent with query
            await run_agent_with_events(
                query=query, project_path=project_path, accuracy_threshold=accuracy_threshold
            )

            console.print()  # Add spacing

        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")


async def single_command_mode(
    query: str, project_path: str | None = None, accuracy_threshold: float = 0.85
):
    """Run agent with a single command."""
    print_banner()

    console.print(f"[bold]Query:[/bold] {query}")
    if project_path:
        console.print(f"[bold]Project:[/bold] {project_path}")
    console.print(f"[bold]Accuracy Target:[/bold] {accuracy_threshold:.0%}")

    await run_agent_with_events(
        query=query, project_path=project_path, accuracy_threshold=accuracy_threshold
    )


# ============================================================================
# Admin CLI Commands
# ============================================================================


def get_admin_headers(api_key: str | None = None) -> dict[str, str]:
    """Get headers for admin API requests."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    return headers


def admin_create_user(args: argparse.Namespace) -> int:
    """Create a new user via the API."""
    api_url = args.api_url
    api_key = args.api_key

    # Get password interactively if not provided
    password = args.password
    if not password:
        password = getpass.getpass("Password: ")
        if not password:
            console.print("[red]Error: Password is required[/red]")
            return 1

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/admin/users",
                headers=get_admin_headers(api_key),
                json={
                    "username": args.username,
                    "email": args.email,
                    "password": password,
                    "is_admin": args.admin,
                },
            )

            if response.status_code == 200:
                data = response.json()
                console.print("[green]User created successfully![/green]")
                table = Table(title="User Details")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("ID", data["id"])
                table.add_row("Username", data["username"])
                table.add_row("Email", data["email"])
                table.add_row("Is Admin", str(data["is_admin"]))
                table.add_row("Created At", data["created_at"])
                console.print(table)
                return 0
            else:
                error_detail = response.json().get("detail", response.text)
                console.print(f"[red]Error: {error_detail}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Error: Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running (python api_server.py)[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def admin_create_key(args: argparse.Namespace) -> int:
    """Create a new API key via the API."""
    api_url = args.api_url
    api_key = args.api_key

    try:
        request_data = {"name": args.name}
        if args.user_id:
            request_data["user_id"] = args.user_id
        if args.expires_in_days:
            request_data["expires_in_days"] = args.expires_in_days
        if args.scopes:
            request_data["scopes"] = args.scopes.split(",")

        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{api_url}/admin/keys",
                headers=get_admin_headers(api_key),
                json=request_data,
            )

            if response.status_code == 200:
                data = response.json()
                console.print("[green]API key created successfully![/green]")
                console.print()
                console.print(
                    Panel(
                        f"[bold yellow]{data['raw_key']}[/bold yellow]",
                        title="[red]IMPORTANT: Save this API key now![/red]",
                        subtitle="[dim]This is the only time it will be shown[/dim]",
                    )
                )
                console.print()
                table = Table(title="API Key Details")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")
                table.add_row("Key ID", data["key_id"])
                table.add_row("Name", data["name"])
                table.add_row("User ID", data.get("user_id") or "N/A")
                table.add_row("Created At", data["created_at"])
                table.add_row("Expires At", data.get("expires_at") or "Never")
                console.print(table)
                return 0
            else:
                error_detail = response.json().get("detail", response.text)
                console.print(f"[red]Error: {error_detail}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Error: Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running (python api_server.py)[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def admin_list_users(args: argparse.Namespace) -> int:
    """List all users via the API."""
    api_url = args.api_url
    api_key = args.api_key

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{api_url}/admin/users",
                headers=get_admin_headers(api_key),
            )

            if response.status_code == 200:
                users = response.json()
                if not users:
                    console.print("[dim]No users found[/dim]")
                    return 0

                table = Table(title=f"Users ({len(users)} total)")
                table.add_column("ID", style="cyan")
                table.add_column("Username", style="white")
                table.add_column("Email", style="white")
                table.add_column("Admin", style="yellow")
                table.add_column("Active", style="green")
                table.add_column("Created At", style="dim")

                for user in users:
                    table.add_row(
                        user["id"],
                        user["username"],
                        user["email"],
                        "Yes" if user["is_admin"] else "No",
                        "Yes" if user["is_active"] else "No",
                        user["created_at"][:19],  # Trim to datetime
                    )

                console.print(table)
                return 0
            else:
                error_detail = response.json().get("detail", response.text)
                console.print(f"[red]Error: {error_detail}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Error: Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running (python api_server.py)[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def admin_list_keys(args: argparse.Namespace) -> int:
    """List all API keys via the API."""
    api_url = args.api_url
    api_key = args.api_key

    try:
        params = {}
        if args.user_id:
            params["user_id"] = args.user_id
        if args.include_revoked:
            params["include_revoked"] = "true"

        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                f"{api_url}/admin/keys",
                headers=get_admin_headers(api_key),
                params=params,
            )

            if response.status_code == 200:
                keys = response.json()
                if not keys:
                    console.print("[dim]No API keys found[/dim]")
                    return 0

                table = Table(title=f"API Keys ({len(keys)} total)")
                table.add_column("Key ID", style="cyan")
                table.add_column("Name", style="white")
                table.add_column("Prefix", style="yellow")
                table.add_column("User ID", style="white")
                table.add_column("Active", style="green")
                table.add_column("Expires At", style="dim")
                table.add_column("Last Used", style="dim")

                for key in keys:
                    table.add_row(
                        key["key_id"][:16] + "...",
                        key["name"],
                        key["key_prefix"],
                        key.get("user_id") or "N/A",
                        "Yes" if key["is_active"] else "[red]No[/red]",
                        key.get("expires_at", "Never")[:19] if key.get("expires_at") else "Never",
                        (
                            key.get("last_used_at", "Never")[:19]
                            if key.get("last_used_at")
                            else "Never"
                        ),
                    )

                console.print(table)
                return 0
            else:
                error_detail = response.json().get("detail", response.text)
                console.print(f"[red]Error: {error_detail}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Error: Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running (python api_server.py)[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def admin_revoke_key(args: argparse.Namespace) -> int:
    """Revoke an API key via the API."""
    api_url = args.api_url
    api_key = args.api_key
    key_id = args.key_id

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.delete(
                f"{api_url}/admin/keys/{key_id}",
                headers=get_admin_headers(api_key),
            )

            if response.status_code == 200:
                data = response.json()
                console.print(f"[green]{data['message']}[/green]")
                return 0
            else:
                error_detail = response.json().get("detail", response.text)
                console.print(f"[red]Error: {error_detail}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Error: Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running (python api_server.py)[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


# ============================================================================
# Init, Deploy, Monitor, Validate Commands
# ============================================================================


def cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new ML project structure."""
    project_path = Path(args.path).resolve()
    # template = args.template  # TODO: Use template for different project structures

    console.print(f"[cyan]Initializing ML project at:[/cyan] {project_path}")

    # Create directory structure
    dirs = [
        "configs",
        "configs/model",
        "configs/training",
        "configs/data",
        "data/raw",
        "data/processed",
        "models",
        "src",
        "notebooks",
        "tests",
    ]

    try:
        project_path.mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (project_path / d).mkdir(parents=True, exist_ok=True)
            console.print(f"  [green]✓[/green] Created {d}/")

        # Create basic config.yaml
        config_content = """# Hydra configuration
defaults:
  - model: default
  - training: default
  - data: default

project_name: my_ml_project
seed: 42
"""
        (project_path / "configs" / "config.yaml").write_text(config_content)
        console.print("  [green]✓[/green] Created configs/config.yaml")

        # Create model config
        model_config = """# Model configuration
name: resnet18
pretrained: true
num_classes: 10
"""
        (project_path / "configs" / "model" / "default.yaml").write_text(model_config)

        # Create training config
        training_config = """# Training configuration
epochs: 10
batch_size: 32
learning_rate: 0.001
optimizer: adam
"""
        (project_path / "configs" / "training" / "default.yaml").write_text(training_config)

        # Create data config
        data_config = """# Data configuration
dataset: custom
train_split: 0.8
val_split: 0.1
test_split: 0.1
"""
        (project_path / "configs" / "data" / "default.yaml").write_text(data_config)

        # Create train.py template
        train_template = '''#!/usr/bin/env python3
"""Training script with Hydra configuration."""
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    """Train the model."""
    print(f"Training {cfg.project_name}")
    print(f"Model: {cfg.model.name}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    # TODO: Implement training logic
    accuracy = 0.0
    return accuracy

if __name__ == "__main__":
    train()
'''
        (project_path / "train.py").write_text(train_template)
        console.print("  [green]✓[/green] Created train.py")

        # Create .gitignore
        gitignore = """# Python
__pycache__/
*.py[cod]
.env
venv/
.venv/

# ML artifacts
*.pt
*.pth
*.onnx
mlruns/
outputs/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# IDE
.idea/
.vscode/
"""
        (project_path / ".gitignore").write_text(gitignore)
        console.print("  [green]✓[/green] Created .gitignore")

        # Create requirements.txt
        requirements = """torch>=2.0.0
torchvision>=0.15.0
hydra-core>=1.3.0
omegaconf>=2.3.0
mlflow>=2.10.0
dvc>=3.0.0
"""
        (project_path / "requirements.txt").write_text(requirements)
        console.print("  [green]✓[/green] Created requirements.txt")

        console.print("\n[green]Project initialized successfully![/green]")
        console.print("\nNext steps:")
        console.print(f"  1. cd {project_path}")
        console.print("  2. pip install -r requirements.txt")
        console.print("  3. mlops-agent 'Set up MLOps pipeline'")
        return 0

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy a trained model to a target platform."""
    target = args.target
    model_path = args.model
    project_path = args.project or "."

    console.print(f"[cyan]Deploying model to:[/cyan] {target}")

    # Build the deployment query
    target_descriptions = {
        "gradio": "Deploy to Gradio for interactive demo",
        "litserve": "Deploy to LitServe for high-throughput inference",
        "lambda": "Deploy to AWS Lambda for serverless inference",
        "torchserve": "Deploy to TorchServe for production serving",
        "kserve": "Deploy to KServe on Kubernetes",
    }

    if target not in target_descriptions:
        console.print(f"[red]Unknown target: {target}[/red]")
        console.print(f"Available targets: {', '.join(target_descriptions.keys())}")
        return 1

    query = f"Deploy my model to {target}"
    if model_path:
        query += f" using model at {model_path}"

    console.print(f"[dim]Running: {query}[/dim]\n")

    try:
        asyncio.run(run_agent_with_events(query=query, project_path=project_path))
        return 0
    except Exception as e:
        console.print(f"[red]Deployment failed: {str(e)}[/red]")
        return 1


def cmd_monitor(args: argparse.Namespace) -> int:
    """Monitor model performance and detect drift."""
    model_name = args.model
    api_url = args.api_url

    console.print(f"[cyan]Monitoring model:[/cyan] {model_name or 'all models'}")

    try:
        with httpx.Client(timeout=30.0) as client:
            # Get metrics from the API
            response = client.get(f"{api_url}/metrics")
            if response.status_code == 200:
                data = response.json()

                # Display system metrics
                table = Table(title="System Metrics")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="white")

                system = data.get("system", {})
                table.add_row("CPU Usage", f"{system.get('cpu_percent', 0):.1f}%")
                table.add_row("Memory Usage", f"{system.get('memory_percent', 0):.1f}%")
                table.add_row("Disk Usage", f"{system.get('disk_percent', 0):.1f}%")
                console.print(table)

                # Display agent metrics
                agent = data.get("agent", {})
                table2 = Table(title="Agent Metrics")
                table2.add_column("Metric", style="cyan")
                table2.add_column("Value", style="white")
                table2.add_row("Total Sessions", str(agent.get("total_sessions", 0)))
                table2.add_row("Success Rate", f"{agent.get('success_rate', 0):.1%}")
                table2.add_row("Avg Execution Time", f"{agent.get('avg_execution_time', 0):.1f}s")
                console.print(table2)

                return 0
            else:
                console.print(f"[red]Failed to get metrics: {response.status_code}[/red]")
                return 1

    except httpx.ConnectError:
        console.print(f"[red]Could not connect to API server at {api_url}[/red]")
        console.print("[dim]Make sure the API server is running[/dim]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a dataset using Great Expectations."""
    data_path = Path(args.data).resolve()
    suite_name = args.suite

    console.print(f"[cyan]Validating dataset:[/cyan] {data_path}")

    if not data_path.exists():
        console.print(f"[red]Error: Data path does not exist: {data_path}[/red]")
        return 1

    query = f"Validate the dataset at {data_path}"
    if suite_name:
        query += f" using expectation suite '{suite_name}'"

    try:
        asyncio.run(run_agent_with_events(query=query, project_path=str(data_path.parent)))
        return 0
    except Exception as e:
        console.print(f"[red]Validation failed: {str(e)}[/red]")
        return 1


def setup_admin_parser(subparsers: argparse._SubParsersAction) -> None:
    """Set up the admin subcommand parser."""
    admin_parser = subparsers.add_parser(
        "admin",
        help="Admin commands for user and API key management",
        description="Admin commands for managing users and API keys via the API server.",
    )

    # Common admin arguments
    admin_parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})",
    )
    admin_parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MLOPS_API_KEY"),
        help="API key for authentication (or set MLOPS_API_KEY env var)",
    )

    admin_subparsers = admin_parser.add_subparsers(dest="admin_command", help="Admin commands")

    # create-user command
    create_user_parser = admin_subparsers.add_parser(
        "create-user",
        help="Create a new user",
        description="Create a new user account.",
    )
    create_user_parser.add_argument(
        "--username", "-u", type=str, required=True, help="Username for the new user"
    )
    create_user_parser.add_argument(
        "--email", "-e", type=str, required=True, help="Email address for the new user"
    )
    create_user_parser.add_argument(
        "--password",
        "-p",
        type=str,
        default=None,
        help="Password (will prompt if not provided)",
    )
    create_user_parser.add_argument(
        "--admin", "-a", action="store_true", help="Create user with admin privileges"
    )
    create_user_parser.set_defaults(func=admin_create_user)

    # create-key command
    create_key_parser = admin_subparsers.add_parser(
        "create-key",
        help="Create a new API key",
        description="Create a new API key for authentication.",
    )
    create_key_parser.add_argument(
        "--name", "-n", type=str, required=True, help="Name for the API key"
    )
    create_key_parser.add_argument(
        "--user-id", "-u", type=str, default=None, help="User ID to associate with the key"
    )
    create_key_parser.add_argument(
        "--expires-in-days",
        "-e",
        type=int,
        default=None,
        help="Number of days until the key expires",
    )
    create_key_parser.add_argument(
        "--scopes",
        "-s",
        type=str,
        default=None,
        help="Comma-separated list of scopes (e.g., 'read,write')",
    )
    create_key_parser.set_defaults(func=admin_create_key)

    # list-users command
    list_users_parser = admin_subparsers.add_parser(
        "list-users",
        help="List all users",
        description="List all registered users.",
    )
    list_users_parser.set_defaults(func=admin_list_users)

    # list-keys command
    list_keys_parser = admin_subparsers.add_parser(
        "list-keys",
        help="List all API keys",
        description="List all API keys.",
    )
    list_keys_parser.add_argument(
        "--user-id", "-u", type=str, default=None, help="Filter by user ID"
    )
    list_keys_parser.add_argument(
        "--include-revoked", "-r", action="store_true", help="Include revoked keys"
    )
    list_keys_parser.set_defaults(func=admin_list_keys)

    # revoke-key command
    revoke_key_parser = admin_subparsers.add_parser(
        "revoke-key",
        help="Revoke an API key",
        description="Revoke an API key by its ID.",
    )
    revoke_key_parser.add_argument(
        "--key-id", "-k", type=str, required=True, help="ID of the API key to revoke"
    )
    revoke_key_parser.set_defaults(func=admin_revoke_key)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MLOps Agent - AI-powered ML Pipeline Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "Set up MLOps pipeline for my cat-dog classifier"
  python cli.py --project ./my_project --threshold 0.90 "Train until accuracy target"
  python cli.py --interactive --project ./my_project

Admin Commands:
  python cli.py admin create-user --username admin --email admin@example.com
  python cli.py admin create-key --name "My API Key"
  python cli.py admin list-users
  python cli.py admin list-keys
  python cli.py admin revoke-key --key-id <key_id>
        """,
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Set up admin subcommand
    setup_admin_parser(subparsers)

    # Init command
    init_parser = subparsers.add_parser(
        "init", help="Initialize a new ML project structure"
    )
    init_parser.add_argument(
        "path", nargs="?", default=".", help="Path for the new project (default: current directory)"
    )
    init_parser.add_argument(
        "--template", "-t", type=str, default="basic",
        choices=["basic", "pytorch", "sklearn", "huggingface"],
        help="Project template to use (default: basic)"
    )
    init_parser.set_defaults(func=cmd_init)

    # Deploy command
    deploy_parser = subparsers.add_parser(
        "deploy", help="Deploy a trained model to a target platform"
    )
    deploy_parser.add_argument(
        "target", choices=["gradio", "litserve", "lambda", "torchserve", "kserve"],
        help="Deployment target platform"
    )
    deploy_parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Path to the trained model"
    )
    deploy_parser.add_argument(
        "--project", "-p", type=str, default=None,
        help="Project directory"
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    # Monitor command
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor model performance and system metrics"
    )
    monitor_parser.add_argument(
        "--model", "-m", type=str, default=None,
        help="Specific model to monitor"
    )
    monitor_parser.add_argument(
        "--api-url", type=str, default=DEFAULT_API_URL,
        help=f"API server URL (default: {DEFAULT_API_URL})"
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a dataset using Great Expectations"
    )
    validate_parser.add_argument(
        "data", help="Path to the dataset to validate"
    )
    validate_parser.add_argument(
        "--suite", "-s", type=str, default=None,
        help="Name of the expectation suite to use"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Agent arguments (for non-admin usage)
    parser.add_argument("query", nargs="?", help="Natural language query for the agent")

    parser.add_argument(
        "-p", "--project", type=str, default=None, help="Path to ML project directory"
    )

    parser.add_argument(
        "-t", "--threshold", type=float, default=0.85, help="Accuracy threshold (default: 0.85)"
    )

    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Run in interactive REPL mode"
    )

    parser.add_argument("--version", action="version", version="MLOps Agent v1.0.0")

    args = parser.parse_args()

    # Handle subcommands (admin, init, deploy, monitor, validate)
    if args.command in ("admin", "init", "deploy", "monitor", "validate"):
        if not hasattr(args, "func") or args.func is None:
            # No subcommand provided, show help
            parser.parse_args([args.command, "--help"])
            return
        sys.exit(args.func(args))

    # Validate project path if provided
    if hasattr(args, "project") and args.project:
        project_path = Path(args.project).resolve()
        if not project_path.exists():
            console.print(f"[red]Error: Project path does not exist: {project_path}[/red]")
            sys.exit(1)
        args.project = str(project_path)

    # Run in appropriate mode
    try:
        if hasattr(args, "interactive") and args.interactive:
            asyncio.run(interactive_mode(args.project, args.threshold))
        elif hasattr(args, "query") and args.query:
            asyncio.run(single_command_mode(args.query, args.project, args.threshold))
        else:
            # No query provided, enter interactive mode
            project = getattr(args, "project", None)
            threshold = getattr(args, "threshold", 0.85)
            asyncio.run(interactive_mode(project, threshold))

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Fatal Error:[/bold red] {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
