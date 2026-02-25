#!/usr/bin/env python3
"""
End-to-end demo: Train Fashion MNIST → Deploy via Auto-MLOps MCP tools.

This script demonstrates the full pipeline WITHOUT needing LLM API keys.
It calls the same MCP tools that the agent uses, but orchestrates them directly.

Usage:
    python examples/fashion_mnist/run_e2e_demo.py

What it does:
    1. Analyzes the project structure (analyze_project_config)
    2. Validates Hydra configs (validate_hydra_config)
    3. Trains a Fashion MNIST CNN (real PyTorch training)
    4. Creates a Dockerfile for the trained model (create_ml_dockerfile)
    5. Creates a Gradio demo app (create_gradio_interface)
    6. Creates a FastAPI serving app (create_fastapi_app)
    7. Creates a LitServe high-perf API (create_litserve_api)
    8. Prints deployment instructions
"""

import sys
import time
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

PROJECT_DIR = Path(__file__).resolve().parent / "project"


def step(num: int, title: str):
    """Print a step header."""
    console.print(f"\n[bold cyan]{'='*60}[/]")
    console.print(f"[bold cyan]  Step {num}: {title}[/]")
    console.print(f"[bold cyan]{'='*60}[/]\n")


def check_result(result: dict, tool_name: str):
    """Check tool result and report."""
    if result.get("success"):
        console.print(f"  [green]OK[/] {tool_name}")
    else:
        console.print(f"  [red]FAIL[/] {tool_name}: {result.get('error', 'unknown')}")
    return result.get("success", False)


def main():
    console.print(Panel.fit(
        "[bold white]Auto-MLOps End-to-End Demo[/]\n"
        "[dim]Fashion MNIST: Train → Deploy[/]",
        border_style="bright_blue",
    ))

    # ----------------------------------------------------------------
    # Step 1: Analyze project
    # ----------------------------------------------------------------
    step(1, "Analyze Project Structure")
    from mcp_mlops_tools import analyze_project_config

    analysis = analyze_project_config(str(PROJECT_DIR))
    check_result(analysis, "analyze_project_config")

    table = Table(title="Project Analysis")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="green")
    for key in ["has_hydra", "has_config_yaml", "has_requirements", "has_train_script"]:
        table.add_row(key, "Yes" if analysis.get(key) else "[red]No[/]")
    console.print(table)

    if analysis.get("recommendations"):
        for rec in analysis["recommendations"]:
            console.print(f"  [yellow]Recommendation:[/] {rec}")

    # ----------------------------------------------------------------
    # Step 2: Validate Hydra config
    # ----------------------------------------------------------------
    step(2, "Validate Hydra Configuration")
    from mcp_mlops_tools import validate_hydra_config

    validation = validate_hydra_config(str(PROJECT_DIR))
    check_result(validation, "validate_hydra_config")

    if validation.get("warnings"):
        for w in validation["warnings"]:
            console.print(f"  [yellow]Warning:[/] {w}")
    if validation.get("issues"):
        for issue in validation["issues"]:
            console.print(f"  [red]Issue:[/] {issue}")

    # ----------------------------------------------------------------
    # Step 3: Train the model
    # ----------------------------------------------------------------
    step(3, "Train Fashion MNIST Model (real PyTorch training)")

    # Import training from the example project
    sys.path.insert(0, str(PROJECT_DIR))
    from train import train

    console.print("  [dim]Training with 3 epochs (CPU)...[/]")
    start = time.time()
    results = train(
        data_dir=str(PROJECT_DIR / "data"),
        output_dir=str(PROJECT_DIR / "models"),
        epochs=3,
        batch_size=128,
        learning_rate=0.001,
        seed=42,
    )
    elapsed = time.time() - start

    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Best Accuracy", f"{results['best_accuracy']:.4f}")
    table.add_row("Final Accuracy", f"{results['final_accuracy']:.4f}")
    table.add_row("Epochs", str(results["epochs_trained"]))
    table.add_row("Time", f"{elapsed:.1f}s")
    table.add_row("Model Path", results["model_path"])
    console.print(table)

    model_path = results["model_path"]

    # ----------------------------------------------------------------
    # Step 4: Create Dockerfile
    # ----------------------------------------------------------------
    step(4, "Create ML Dockerfile")
    from mcp_mlops_tools import create_ml_dockerfile

    dockerfile = create_ml_dockerfile(
        project_path=str(PROJECT_DIR),
        base_image="python:3.11-slim",
        requirements_file="requirements.txt",
        entry_point="train.py",
        expose_port=8000,
    )
    check_result(dockerfile, "create_ml_dockerfile")
    if dockerfile.get("success"):
        console.print(f"  [dim]Created: {dockerfile.get('dockerfile_path')}[/]")

    # ----------------------------------------------------------------
    # Step 5: Create Gradio demo app
    # ----------------------------------------------------------------
    step(5, "Create Gradio Deployment")
    from mcp_mlops_tools import create_gradio_interface

    gradio_result = create_gradio_interface(
        project_path=str(PROJECT_DIR),
        model_path=model_path,
        model_name="FashionCNN",
        interface_type="image_classifier",
        title="Fashion MNIST Classifier",
        description="Upload a 28x28 grayscale image of clothing to classify it.",
    )
    check_result(gradio_result, "create_gradio_interface")
    if gradio_result.get("success"):
        console.print(f"  [dim]Created: {gradio_result.get('app_path')}[/]")

    # ----------------------------------------------------------------
    # Step 6: Create FastAPI serving app
    # ----------------------------------------------------------------
    step(6, "Create FastAPI Deployment")
    from mcp_mlops_tools import create_fastapi_app

    fastapi_result = create_fastapi_app(
        project_path=str(PROJECT_DIR),
        model_path=model_path,
        model_name="FashionCNN",
        endpoint_type="image",
        title="Fashion MNIST Inference API",
    )
    check_result(fastapi_result, "create_fastapi_app")
    if fastapi_result.get("success"):
        console.print(f"  [dim]Created: {fastapi_result.get('app_path')}[/]")

    # ----------------------------------------------------------------
    # Step 7: Create LitServe API
    # ----------------------------------------------------------------
    step(7, "Create LitServe High-Performance API")
    from mcp_mlops_tools import create_litserve_api

    litserve_result = create_litserve_api(
        project_path=str(PROJECT_DIR),
        model_path=model_path,
        model_name="FashionCNN",
        model_type="image_classifier",
        class_labels=results["class_names"],
    )
    check_result(litserve_result, "create_litserve_api")
    if litserve_result.get("success"):
        console.print(f"  [dim]Created: {litserve_result.get('server_path')}[/]")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    console.print(f"\n[bold green]{'='*60}[/]")
    console.print(f"[bold green]  Demo Complete![/]")
    console.print(f"[bold green]{'='*60}[/]\n")

    # List generated artifacts
    table = Table(title="Generated Artifacts")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="dim")
    table.add_column("Deploy With", style="yellow")

    table.add_row("Trained Model", str(PROJECT_DIR / "models" / "best_model.pt"), "--")
    table.add_row("Dockerfile", str(PROJECT_DIR / "Dockerfile"), "docker build & run")
    table.add_row("Gradio App", str(PROJECT_DIR / "deployment" / "gradio" / "app.py"), "python app.py")
    table.add_row("FastAPI App", str(PROJECT_DIR / "deployment" / "fastapi_lambda" / "app.py"), "uvicorn app:app")
    table.add_row("LitServe API", str(PROJECT_DIR / "deployment" / "litserve" / "server.py"), "python server.py")
    console.print(table)

    console.print("\n[bold]Next steps:[/]")
    console.print("  1. [cyan]docker build -t fashion-mnist .[/]  (containerize)")
    console.print("  2. [cyan]cd deployment/gradio && python app.py[/]  (quick demo)")
    console.print("  3. [cyan]cd deployment/fastapi_lambda && uvicorn app:app[/]  (production API)")
    console.print("  4. Use the full agent: [cyan]python cli.py 'Deploy fashion mnist to AWS Lambda'[/]")
    console.print()


if __name__ == "__main__":
    main()
