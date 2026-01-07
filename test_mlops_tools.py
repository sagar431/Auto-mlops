#!/usr/bin/env python3
"""
Test Suite for MLOps MCP Tools

Usage:
    python test_mlops_tools.py                    # Run all tests
    python test_mlops_tools.py --tool hydra       # Test Hydra tools only
    python test_mlops_tools.py --tool mlflow      # Test MLflow tools only
    python test_mlops_tools.py --tool dvc         # Test DVC tools only
    python test_mlops_tools.py --tool docker      # Test Docker tools only
    python test_mlops_tools.py --tool github      # Test GitHub Actions tools only
    python test_mlops_tools.py --tool training    # Test Training Control tools only
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path
from datetime import datetime

# Import tool functions from mcp_mlops_tools
from mcp_mlops_tools import (
    # Hydra tools
    analyze_project_config,
    create_hydra_config,
    update_hydra_config,
    validate_hydra_config,
    # MLflow tools
    init_mlflow_experiment,
    start_mlflow_run,
    log_mlflow_params,
    log_mlflow_metrics,
    end_mlflow_run,
    get_best_mlflow_run,
    # DVC tools
    init_dvc_repo,
    configure_dvc_remote,
    add_data_to_dvc,
    create_dvc_pipeline,
    # Docker tools
    create_ml_dockerfile,
    build_ml_docker_image,
    # GitHub Actions tools
    create_github_workflow,
    add_workflow_step,
    # Training control tools
    suggest_improvements,
    check_tool_installed,
)


class Colors:
    """Terminal colors for output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.END}\n")


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    print(f"  {status} {name}")
    if details and not passed:
        print(f"       {Colors.YELLOW}{details}{Colors.END}")


class TestMLOpsTools:
    """Test suite for MLOps MCP tools."""
    
    def __init__(self):
        self.test_dir = None
        self.results = {"passed": 0, "failed": 0, "skipped": 0}
        
    def setup(self):
        """Create temporary test directory with sample project structure."""
        self.test_dir = tempfile.mkdtemp(prefix="mlops_test_")
        print(f"{Colors.BLUE}Test directory: {self.test_dir}{Colors.END}")
        
        # Create sample project structure
        project_path = Path(self.test_dir) / "sample_project"
        project_path.mkdir()
        
        # Create requirements.txt
        requirements = """torch>=2.0.0
torchvision>=0.15.0
hydra-core>=1.3.0
mlflow>=2.0.0
"""
        (project_path / "requirements.txt").write_text(requirements)
        
        # Create train.py
        train_script = '''#!/usr/bin/env python3
"""Sample training script."""
print("Training...")
'''
        (project_path / "train.py").write_text(train_script)
        
        # Create data directory
        (project_path / "data").mkdir()
        (project_path / "data" / "sample.csv").write_text("id,label\n1,cat\n2,dog\n")
        
        self.project_path = str(project_path)
        return project_path
    
    def teardown(self):
        """Clean up test directory."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"\n{Colors.BLUE}Cleaned up test directory{Colors.END}")
    
    def record_result(self, passed: bool):
        if passed:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
    
    def skip_test(self, reason: str):
        self.results["skipped"] += 1
        print(f"  {Colors.YELLOW}⊘ SKIP{Colors.END} {reason}")
    
    def test_hydra_tools(self):
        """Test Hydra configuration tools."""
        print_header("Testing Hydra Configuration Tools")
        
        # Test 1: Analyze project config
        print(f"\n{Colors.BOLD}Test: analyze_project_config{Colors.END}")
        result = analyze_project_config(self.project_path)
        passed = result.get("success", False)
        print_test("Analyze project structure", passed)
        if passed:
            print(f"       Has train.py: {result.get('has_train_script')}")
        self.record_result(passed)
        
        # Test 2: Create Hydra config
        print(f"\n{Colors.BOLD}Test: create_hydra_config{Colors.END}")
        result = create_hydra_config(
            self.project_path,
            config_name="config",
            model_config={"name": "resnet18", "pretrained": True},
            training_config={"epochs": 10, "learning_rate": 0.001}
        )
        passed = result.get("success", False)
        print_test("Create Hydra config structure", passed)
        if passed:
            print(f"       Created files: {len(result.get('created_files', []))}")
        self.record_result(passed)
        
        # Test 3: Validate config
        print(f"\n{Colors.BOLD}Test: validate_hydra_config{Colors.END}")
        result = validate_hydra_config(self.project_path, "configs/config.yaml")
        passed = result.get("success", False) or result.get("valid", False)
        print_test("Validate Hydra config", passed)
        self.record_result(passed)
        
        # Test 4: Update config
        print(f"\n{Colors.BOLD}Test: update_hydra_config{Colors.END}")
        result = update_hydra_config(
            self.project_path,
            "configs/config.yaml",
            {"seed": 123}
        )
        passed = result.get("success", False)
        print_test("Update Hydra config", passed)
        self.record_result(passed)
    
    def test_mlflow_tools(self):
        """Test MLflow experiment tracking tools."""
        print_header("Testing MLflow Experiment Tracking Tools")
        
        try:
            import mlflow
            mlflow_available = True
        except ImportError:
            mlflow_available = False
            print(f"{Colors.YELLOW}MLflow not installed. Skipping.{Colors.END}")
        
        if not mlflow_available:
            self.skip_test("MLflow not installed")
            return
        
        # Test 1: Initialize experiment
        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=os.path.join(self.test_dir, "mlruns")
        )
        passed = result.get("success", False)
        print_test("Initialize MLflow experiment", passed)
        self.record_result(passed)
        
        # Test 2: Start run
        result = start_mlflow_run(experiment_name=experiment_name, run_name="test_run")
        passed = result.get("success", False)
        print_test("Start MLflow run", passed)
        self.record_result(passed)
        
        # Test 3: Log params
        result = log_mlflow_params(params={"learning_rate": 0.001, "epochs": 10})
        passed = result.get("success", False)
        print_test("Log parameters", passed)
        self.record_result(passed)
        
        # Test 4: Log metrics
        result = log_mlflow_metrics(metrics={"accuracy": 0.85, "loss": 0.15}, step=1)
        passed = result.get("success", False)
        print_test("Log metrics", passed)
        self.record_result(passed)
        
        # Test 5: End run
        result = end_mlflow_run(status="FINISHED")
        passed = result.get("success", False)
        print_test("End MLflow run", passed)
        self.record_result(passed)
        
        # Test 6: Get best run
        result = get_best_mlflow_run(experiment_name=experiment_name, metric_name="accuracy")
        passed = result.get("success", False)
        print_test("Get best run", passed)
        if passed:
            print(f"       Best accuracy: {result.get('best_metric', {}).get('accuracy')}")
        self.record_result(passed)
    
    def test_dvc_tools(self):
        """Test DVC data versioning tools."""
        print_header("Testing DVC Data Versioning Tools")
        
        if not check_tool_installed("dvc"):
            print(f"{Colors.YELLOW}DVC not installed. Skipping.{Colors.END}")
            self.skip_test("DVC not installed")
            return
        
        # Test 1: Initialize DVC
        result = init_dvc_repo(self.project_path, no_scm=True)
        passed = result.get("success", False)
        print_test("Initialize DVC repo", passed)
        self.record_result(passed)
        
        if not passed:
            return
        
        # Test 2: Configure remote
        result = configure_dvc_remote(
            self.project_path,
            remote_name="local_test",
            remote_url=os.path.join(self.test_dir, "dvc_remote"),
            default=True
        )
        passed = result.get("success", False)
        print_test("Configure DVC remote", passed)
        self.record_result(passed)
        
        # Test 3: Add data to DVC
        result = add_data_to_dvc(self.project_path, "data/sample.csv")
        passed = result.get("success", False)
        print_test("Add data to DVC", passed)
        if passed:
            print(f"       DVC file: {result.get('dvc_file')}")
        self.record_result(passed)
        
        # Test 4: Create pipeline
        stages = [
            {"name": "prepare", "cmd": "python prepare.py", "deps": ["data/"], "outs": ["processed/"]},
            {"name": "train", "cmd": "python train.py", "deps": ["processed/"], "outs": ["models/"]}
        ]
        result = create_dvc_pipeline(self.project_path, stages)
        passed = result.get("success", False)
        print_test("Create DVC pipeline", passed)
        if passed:
            print(f"       Stages: {result.get('stages')}")
        self.record_result(passed)
    
    def test_docker_tools(self):
        """Test Docker tools."""
        print_header("Testing Docker Tools")
        
        # Test 1: Create Dockerfile (doesn't require Docker)
        result = create_ml_dockerfile(
            self.project_path,
            base_image="python:3.11-slim",
            entry_point="train.py",
            expose_port=8000
        )
        passed = result.get("success", False)
        print_test("Create ML Dockerfile", passed)
        if passed:
            print(f"       Dockerfile: {result.get('dockerfile_path')}")
        self.record_result(passed)
        
        # Test 2: Build image (requires Docker)
        if not check_tool_installed("docker"):
            self.skip_test("Docker not installed - skipping build")
        else:
            result = build_ml_docker_image(self.project_path, image_name="mlops-test", tag="latest")
            passed = result.get("success", False)
            print_test("Build Docker image", passed, result.get("error", ""))
            self.record_result(passed)
    
    def test_github_tools(self):
        """Test GitHub Actions tools."""
        print_header("Testing GitHub Actions Tools")
        
        # Test 1: Create workflow
        result = create_github_workflow(
            self.project_path,
            workflow_name="ml-pipeline",
            trigger_on=["push", "workflow_dispatch"],
            use_dvc=True,
            use_mlflow=True,
            accuracy_threshold=0.85
        )
        passed = result.get("success", False)
        print_test("Create GitHub workflow", passed)
        if passed:
            print(f"       Workflow: {result.get('workflow_path')}")
        self.record_result(passed)
        
        # Test 2: Add workflow step
        result = add_workflow_step(
            self.project_path,
            workflow_file=".github/workflows/ml-pipeline.yml",
            job_name="train",
            step={"name": "Custom step", "run": "echo 'Hello'"}
        )
        passed = result.get("success", False)
        print_test("Add workflow step", passed)
        self.record_result(passed)
    
    def test_training_control_tools(self):
        """Test training control tools."""
        print_header("Testing Training Control Tools")
        
        # Test 1: Suggest improvements (attempt 1)
        print(f"\n{Colors.BOLD}Test: suggest_improvements (attempt 1){Colors.END}")
        result = suggest_improvements(
            current_metrics={"accuracy": 0.72, "loss": 0.45},
            current_config={"learning_rate": 0.01, "epochs": 10, "batch_size": 32},
            target_accuracy=0.85,
            attempt_number=1
        )
        passed = result.get("success", False)
        print_test("Suggest improvements", passed)
        if passed:
            print(f"       Gap: {result.get('gap'):.2%}")
            print(f"       Suggestions: {list(result.get('config_changes', {}).keys())}")
            for reason in result.get("reasoning", [])[:2]:
                print(f"       → {reason}")
        self.record_result(passed)
        
        # Test 2: Suggest improvements (attempt 2 - should add regularization)
        print(f"\n{Colors.BOLD}Test: suggest_improvements (attempt 2){Colors.END}")
        result = suggest_improvements(
            current_metrics={"accuracy": 0.78, "loss": 0.35},
            current_config={"learning_rate": 0.005, "epochs": 15, "batch_size": 32},
            target_accuracy=0.85,
            attempt_number=2
        )
        passed = result.get("success", False)
        print_test("Suggest improvements (retry)", passed)
        if passed:
            print(f"       Gap: {result.get('gap'):.2%}")
            print(f"       New changes: {list(result.get('config_changes', {}).keys())}")
        self.record_result(passed)
    
    def run_all(self, tool_filter: str = None):
        """Run all tests or filtered tests."""
        self.setup()
        
        try:
            if tool_filter is None or tool_filter == "hydra":
                self.test_hydra_tools()
            
            if tool_filter is None or tool_filter == "mlflow":
                self.test_mlflow_tools()
            
            if tool_filter is None or tool_filter == "dvc":
                self.test_dvc_tools()
            
            if tool_filter is None or tool_filter == "docker":
                self.test_docker_tools()
            
            if tool_filter is None or tool_filter == "github":
                self.test_github_tools()
            
            if tool_filter is None or tool_filter == "training":
                self.test_training_control_tools()
            
        finally:
            self.teardown()
        
        # Print summary
        print_header("Test Summary")
        total = self.results["passed"] + self.results["failed"]
        print(f"  {Colors.GREEN}Passed: {self.results['passed']}{Colors.END}")
        print(f"  {Colors.RED}Failed: {self.results['failed']}{Colors.END}")
        print(f"  {Colors.YELLOW}Skipped: {self.results['skipped']}{Colors.END}")
        print(f"  Total: {total}")
        
        if total > 0:
            pass_rate = (self.results["passed"] / total) * 100
            color = Colors.GREEN if pass_rate >= 80 else Colors.YELLOW if pass_rate >= 50 else Colors.RED
            print(f"\n  {color}Pass Rate: {pass_rate:.1f}%{Colors.END}")
        
        return self.results["failed"] == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MLOps MCP Tools")
    parser.add_argument(
        "--tool",
        choices=["hydra", "mlflow", "dvc", "docker", "github", "training"],
        help="Test only specific tool category"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            MLOps MCP Tools - Test Suite                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    tester = TestMLOpsTools()
    success = tester.run_all(tool_filter=args.tool)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
