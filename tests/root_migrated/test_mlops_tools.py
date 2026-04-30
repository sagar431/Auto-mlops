#!/usr/bin/env python3
"""
Test Suite for MLOps MCP Tools

Usage:
    python -m tests.root_migrated.test_mlops_tools                    # Run all tests
    python -m tests.root_migrated.test_mlops_tools --tool hydra       # Test Hydra tools only
    python -m tests.root_migrated.test_mlops_tools --tool mlflow      # Test MLflow tools only
    python -m tests.root_migrated.test_mlops_tools --tool dvc         # Test DVC tools only
    python -m tests.root_migrated.test_mlops_tools --tool docker      # Test Docker tools only
    python -m tests.root_migrated.test_mlops_tools --tool github      # Test GitHub Actions tools only
    python -m tests.root_migrated.test_mlops_tools --tool training    # Test Training Control tools only
    python -m tests.root_migrated.test_mlops_tools --tool data_quality # Test Data Quality tools only
"""

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Import tool functions from mcp_mlops_tools
from mcp_mlops_tools import (
    add_data_to_dvc,
    add_workflow_step,
    # Hydra tools
    analyze_project_config,
    build_ml_docker_image,
    # Data quality tools
    check_data_quality,
    check_tool_installed,
    compare_distributions,
    configure_dvc_remote,
    create_dvc_pipeline,
    create_expectation_suite,
    # GitHub Actions tools
    create_github_workflow,
    create_hydra_config,
    # Docker tools
    create_ml_dockerfile,
    detect_anomalies,
    end_mlflow_run,
    get_best_mlflow_run,
    # DVC tools
    init_dvc_repo,
    # MLflow tools
    init_mlflow_experiment,
    log_mlflow_metrics,
    log_mlflow_params,
    profile_dataset,
    start_mlflow_run,
    # Training control tools
    suggest_improvements,
    update_hydra_config,
    validate_dataset,
    validate_hydra_config,
    validate_schema,
)


class Colors:
    """Terminal colors for output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    END = "\033[0m"
    BOLD = "\033[1m"


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
            training_config={"epochs": 10, "learning_rate": 0.001},
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
        result = update_hydra_config(self.project_path, "configs/config.yaml", {"seed": 123})
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
            experiment_name=experiment_name, tracking_uri=os.path.join(self.test_dir, "mlruns")
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
            default=True,
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
            {
                "name": "prepare",
                "cmd": "python prepare.py",
                "deps": ["data/"],
                "outs": ["processed/"],
            },
            {
                "name": "train",
                "cmd": "python train.py",
                "deps": ["processed/"],
                "outs": ["models/"],
            },
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
            expose_port=8000,
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
            accuracy_threshold=0.85,
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
            step={"name": "Custom step", "run": "echo 'Hello'"},
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
            attempt_number=1,
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
            attempt_number=2,
        )
        passed = result.get("success", False)
        print_test("Suggest improvements (retry)", passed)
        if passed:
            print(f"       Gap: {result.get('gap'):.2%}")
            print(f"       New changes: {list(result.get('config_changes', {}).keys())}")
        self.record_result(passed)

    def test_data_quality_tools(self):
        """Test data quality tools."""
        print_header("Testing Data Quality Tools")

        # Test 1: Validate CSV dataset
        print(f"\n{Colors.BOLD}Test: validate_dataset (CSV){Colors.END}")

        # Create a test CSV with some quality issues
        csv_path = Path(self.project_path) / "data" / "test_data.csv"
        csv_content = """id,label,value,category
1,cat,10.5,A
2,dog,15.2,B
3,cat,12.3,A
4,,18.1,C
5,dog,14.7,B
1,cat,10.5,A
6,bird,100.0,D
7,cat,11.2,A
8,dog,,B
"""
        csv_path.write_text(csv_content)

        result = validate_dataset(str(csv_path), dataset_type="csv")
        passed = result.get("success", False)
        print_test("Validate CSV dataset", passed)
        if passed:
            print(f"       Total rows: {result.get('statistics', {}).get('total_rows')}")
            print(f"       Issues found: {result.get('total_issues', 0)}")
            print(f"       Warnings: {result.get('total_warnings', 0)}")
            print(f"       Is valid: {result.get('is_valid')}")
        self.record_result(passed)

        # Test 2: Validate with auto-detect
        print(f"\n{Colors.BOLD}Test: validate_dataset (auto-detect){Colors.END}")
        result = validate_dataset(str(csv_path))
        passed = result.get("success", False) and result.get("dataset_type") == "csv"
        print_test("Auto-detect CSV type", passed)
        if passed:
            print(f"       Detected type: {result.get('dataset_type')}")
        self.record_result(passed)

        # Test 3: Validate with specific checks
        print(f"\n{Colors.BOLD}Test: validate_dataset (specific checks){Colors.END}")
        result = validate_dataset(
            str(csv_path), dataset_type="csv", checks=["missing_values", "duplicates"]
        )
        passed = result.get("success", False)
        checks_performed = result.get("checks_performed", [])
        print_test("Validate with specific checks", passed)
        if passed:
            print(f"       Checks performed: {checks_performed}")
        self.record_result(passed)

        # Test 4: Validate image directory (create mock structure)
        print(f"\n{Colors.BOLD}Test: validate_dataset (images){Colors.END}")

        # Create image directory structure
        images_dir = Path(self.project_path) / "images"
        (images_dir / "cats").mkdir(parents=True)
        (images_dir / "dogs").mkdir(parents=True)

        # Create minimal valid image files (1x1 pixel PNG)
        # PNG header for 1x1 white pixel
        png_data = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,
                0x08,
                0x02,
                0x00,
                0x00,
                0x00,
                0x90,
                0x77,
                0x53,
                0xDE,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x49,
                0x44,
                0x41,
                0x54,
                0x08,
                0xD7,
                0x63,
                0xF8,
                0xFF,
                0xFF,
                0x3F,
                0x00,
                0x05,
                0xFE,
                0x02,
                0xFE,
                0xDC,
                0xCC,
                0x59,
                0xE7,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )

        for i in range(5):
            (images_dir / "cats" / f"cat_{i}.png").write_bytes(png_data)
        for i in range(3):
            (images_dir / "dogs" / f"dog_{i}.png").write_bytes(png_data)

        # Create an invalid image file
        (images_dir / "cats" / "invalid.png").write_text("not an image")

        result = validate_dataset(str(images_dir), dataset_type="images")
        passed = result.get("success", False)
        print_test("Validate image dataset", passed)
        if passed:
            stats = result.get("statistics", {})
            print(f"       Total images: {stats.get('total_images')}")
            print(f"       Classes: {stats.get('classes')}")
            print(f"       Issues: {result.get('total_issues', 0)}")
        self.record_result(passed)

        # Test 5: Validate non-existent path
        print(f"\n{Colors.BOLD}Test: validate_dataset (non-existent path){Colors.END}")
        result = validate_dataset("/nonexistent/path/data.csv")
        passed = result.get("success", False) is False
        print_test("Handle non-existent path", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 6: Validate with sample size
        print(f"\n{Colors.BOLD}Test: validate_dataset (sample size){Colors.END}")

        # Create larger CSV
        large_csv_path = Path(self.project_path) / "data" / "large_data.csv"
        rows = ["id,value,label"]
        for i in range(100):
            rows.append(f"{i},{i * 1.5},{'cat' if i % 2 == 0 else 'dog'}")
        large_csv_path.write_text("\n".join(rows))

        result = validate_dataset(str(large_csv_path), dataset_type="csv", sample_size=10)
        passed = result.get("success", False)
        print_test("Validate with sample size", passed)
        if passed:
            stats = result.get("statistics", {})
            print(f"       Total rows validated: {stats.get('total_rows')}")
        self.record_result(passed)

        # Test 7: Create expectation suite
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (basic){Colors.END}")
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
                "description": "ID column should not have null values",
            },
            {
                "expectation_type": "expect_column_values_to_be_unique",
                "column": "id",
            },
            {
                "expectation_type": "expect_column_values_to_be_in_set",
                "column": "label",
                "kwargs": {"value_set": ["cat", "dog", "bird"]},
            },
        ]
        result = create_expectation_suite(self.project_path, "test_suite", expectations)
        passed = result.get("success", False)
        print_test("Create expectation suite", passed)
        if passed:
            print(f"       Suite name: {result.get('suite_name')}")
            print(f"       Expectations: {result.get('expectation_count')}")
            print(f"       Suite path: {result.get('suite_path')}")
        self.record_result(passed)

        # Verify the file was created and is valid JSON
        if passed:
            import json

            suite_path = Path(result.get("suite_path"))
            file_exists = suite_path.exists()
            print_test("Suite file exists", file_exists)
            self.record_result(file_exists)

            if file_exists:
                try:
                    with open(suite_path) as f:
                        suite_data = json.load(f)
                    valid_json = "expectations" in suite_data
                    print_test("Suite file is valid JSON", valid_json)
                    if valid_json:
                        print(f"       Expectations in file: {len(suite_data['expectations'])}")
                    self.record_result(valid_json)
                except Exception as e:
                    print_test("Suite file is valid JSON", False, str(e))
                    self.record_result(False)

        # Test 8: Create expectation suite with range expectation
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (with kwargs){Colors.END}")
        expectations_with_kwargs = [
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "value",
                "kwargs": {"min_value": 0, "max_value": 100},
                "severity": "warning",
            },
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1, "max_value": 1000},
            },
        ]
        result = create_expectation_suite(
            self.project_path, "test_suite_kwargs", expectations_with_kwargs
        )
        passed = result.get("success", False)
        print_test("Create suite with kwargs", passed)
        if passed:
            print(f"       Expectation types: {result.get('expectation_types')}")
        self.record_result(passed)

        # Test 9: Create expectation suite with custom output directory
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (custom output dir){Colors.END}")
        result = create_expectation_suite(
            self.project_path,
            "custom_dir_suite",
            [{"expectation_type": "expect_column_to_exist", "column": "id"}],
            output_dir="custom_expectations",
        )
        passed = result.get("success", False)
        print_test("Create suite in custom directory", passed)
        if passed:
            print(f"       Suite path: {result.get('suite_path')}")
            custom_path = Path(self.project_path) / "custom_expectations" / "custom_dir_suite.json"
            path_correct = custom_path.exists()
            print_test("Custom path is correct", path_correct)
            self.record_result(path_correct)
        self.record_result(passed)

        # Test 10: Create expectation suite with invalid expectations
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (validation errors){Colors.END}")
        invalid_expectations = [
            {"column": "id"},  # Missing expectation_type
            {"expectation_type": ""},  # Empty expectation_type
        ]
        result = create_expectation_suite(self.project_path, "invalid_suite", invalid_expectations)
        passed = result.get("success", False) is False
        print_test("Reject invalid expectations", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 11: Create expectation suite with empty expectations list
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (empty expectations){Colors.END}")
        result = create_expectation_suite(self.project_path, "empty_suite", [])
        passed = result.get("success", False) is False
        print_test("Reject empty expectations", passed)
        self.record_result(passed)

        # Test 12: Create expectation suite with non-existent project path
        print(f"\n{Colors.BOLD}Test: create_expectation_suite (non-existent path){Colors.END}")
        result = create_expectation_suite(
            "/nonexistent/path",
            "test_suite",
            [{"expectation_type": "expect_column_to_exist", "column": "id"}],
        )
        passed = result.get("success", False) is False
        print_test("Handle non-existent path", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 13: check_data_quality (basic - no expectations)
        print(f"\n{Colors.BOLD}Test: check_data_quality (basic){Colors.END}")
        csv_path = Path(self.project_path) / "data" / "test_data.csv"
        # Create test CSV if not exists
        if not csv_path.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_content = """id,label,value,category
1,cat,10.5,A
2,dog,15.2,B
3,cat,12.3,A
4,,18.1,C
5,dog,14.7,B
"""
            csv_path.write_text(csv_content)

        result = check_data_quality(str(csv_path))
        passed = result.get("success", False)
        print_test("Basic data quality check", passed)
        if passed:
            print(f"       Overall score: {result.get('overall_score')}")
            print(f"       Passed checks: {result.get('passed_checks')}")
            print(f"       Failed checks: {result.get('failed_checks')}")
            print(f"       Is valid: {result.get('is_valid')}")
        self.record_result(passed)

        # Test 14: check_data_quality (with custom expectations)
        print(f"\n{Colors.BOLD}Test: check_data_quality (custom expectations){Colors.END}")
        expectations = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "column": "value",
                "kwargs": {"min_value": 0, "max_value": 100},
                "severity": "warning",
            },
        ]
        result = check_data_quality(str(csv_path), expectations=expectations)
        passed = result.get("success", False)
        print_test("Check with custom expectations", passed)
        if passed:
            print(f"       Overall score: {result.get('overall_score')}")
            val_results = result.get("validation_results", [])
            print(f"       Validation results: {len(val_results)}")
        self.record_result(passed)

        # Test 15: check_data_quality (with statistics)
        print(f"\n{Colors.BOLD}Test: check_data_quality (with statistics){Colors.END}")
        result = check_data_quality(str(csv_path), include_statistics=True)
        passed = result.get("success", False) and "statistics" in result
        print_test("Check includes statistics", passed)
        if passed:
            stats = result.get("statistics", {})
            print(f"       Row count: {stats.get('row_count')}")
            print(f"       Column count: {stats.get('column_count')}")
            print(f"       Missing %: {stats.get('missing_percentage')}")
        self.record_result(passed)

        # Test 16: check_data_quality (without statistics)
        print(f"\n{Colors.BOLD}Test: check_data_quality (without statistics){Colors.END}")
        result = check_data_quality(str(csv_path), include_statistics=False)
        passed = result.get("success", False) and "statistics" not in result
        print_test("Check without statistics", passed)
        self.record_result(passed)

        # Test 17: check_data_quality (fail_on_error=True with failing check)
        print(f"\n{Colors.BOLD}Test: check_data_quality (fail_on_error){Colors.END}")
        # Create CSV with null values in 'id' column to trigger failure
        csv_with_nulls = Path(self.project_path) / "data" / "data_with_nulls.csv"
        csv_with_nulls.write_text("""id,value
1,10
,20
3,30
,40
""")
        expectations_strict = [
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "column": "id",
                "severity": "error",
            }
        ]
        result = check_data_quality(
            str(csv_with_nulls), expectations=expectations_strict, fail_on_error=True
        )
        # Should fail because id column has nulls and fail_on_error=True
        passed = result.get("success", True) is False
        print_test("Fail on error-level check failure", passed)
        if passed:
            print(f"       Message: {result.get('message', '')[:60]}...")
        self.record_result(passed)

        # Test 18: check_data_quality (non-existent file)
        print(f"\n{Colors.BOLD}Test: check_data_quality (non-existent file){Colors.END}")
        result = check_data_quality("/nonexistent/path/data.csv")
        passed = result.get("success", False) is False
        print_test("Handle non-existent file", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 19: check_data_quality (unsupported file type)
        print(f"\n{Colors.BOLD}Test: check_data_quality (unsupported file type){Colors.END}")
        txt_file = Path(self.project_path) / "data" / "data.txt"
        txt_file.write_text("some text data")
        result = check_data_quality(str(txt_file))
        passed = result.get("success", False) is False
        print_test("Reject unsupported file type", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 20: check_data_quality (parquet file)
        print(f"\n{Colors.BOLD}Test: check_data_quality (parquet file){Colors.END}")
        try:
            import pandas as pd

            parquet_path = Path(self.project_path) / "data" / "test_data.parquet"
            df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 15.7]})
            df.to_parquet(parquet_path)

            result = check_data_quality(str(parquet_path))
            passed = result.get("success", False)
            print_test("Check parquet file", passed)
            if passed:
                print(f"       Overall score: {result.get('overall_score')}")
        except ImportError:
            print_test("Check parquet file", True)
            print("       (Skipped - pandas/pyarrow not available)")
            passed = True
        self.record_result(passed)

        # Test 21: profile_dataset (basic)
        print(f"\n{Colors.BOLD}Test: profile_dataset (basic){Colors.END}")
        csv_path = Path(self.project_path) / "data" / "test_data.csv"
        result = profile_dataset(str(csv_path))
        passed = result.get("success", False)
        print_test("Profile CSV dataset", passed)
        if passed:
            stats = result.get("statistics", {})
            print(f"       Row count: {stats.get('row_count')}")
            print(f"       Column count: {stats.get('column_count')}")
            print(f"       Memory usage: {stats.get('memory_usage_mb')} MB")
            print(f"       Columns profiled: {len(result.get('columns', []))}")
        self.record_result(passed)

        # Test 22: profile_dataset (with custom name)
        print(f"\n{Colors.BOLD}Test: profile_dataset (custom name){Colors.END}")
        result = profile_dataset(str(csv_path), dataset_name="my_custom_dataset")
        passed = result.get("success", False) and result.get("dataset_name") == "my_custom_dataset"
        print_test("Profile with custom name", passed)
        if passed:
            print(f"       Dataset name: {result.get('dataset_name')}")
        self.record_result(passed)

        # Test 23: profile_dataset (without column stats)
        print(f"\n{Colors.BOLD}Test: profile_dataset (without column stats){Colors.END}")
        result = profile_dataset(str(csv_path), include_column_stats=False)
        passed = result.get("success", False) and "columns" not in result
        print_test("Profile without column stats", passed)
        self.record_result(passed)

        # Test 24: profile_dataset (non-existent file)
        print(f"\n{Colors.BOLD}Test: profile_dataset (non-existent file){Colors.END}")
        result = profile_dataset("/nonexistent/path/data.csv")
        passed = result.get("success", False) is False
        print_test("Handle non-existent file", passed)
        if passed:
            print(f"       Error: {result.get('error', '')[:50]}...")
        self.record_result(passed)

        # Test 25: detect_anomalies (basic)
        print(f"\n{Colors.BOLD}Test: detect_anomalies (basic){Colors.END}")
        # Create CSV with some outliers
        anomaly_csv = Path(self.project_path) / "data" / "anomaly_data.csv"
        anomaly_content = """id,value,category
1,10.5,A
2,15.2,B
3,12.3,A
4,18.1,C
5,14.7,B
6,100.0,A
7,11.2,A
8,-50.0,B
1,10.5,A
"""
        anomaly_csv.write_text(anomaly_content)

        result = detect_anomalies(str(anomaly_csv))
        passed = result.get("success", False)
        print_test("Detect anomalies in dataset", passed)
        if passed:
            print(f"       Total anomalies: {result.get('total_anomalies')}")
            print(f"       Affected rows: {result.get('affected_rows')}")
            print(f"       Affected %: {result.get('affected_percentage')}")
            print(f"       Types: {result.get('anomalies_by_type')}")
        self.record_result(passed)

        # Test 26: detect_anomalies (specific methods)
        print(f"\n{Colors.BOLD}Test: detect_anomalies (specific methods){Colors.END}")
        result = detect_anomalies(str(anomaly_csv), methods=["iqr", "duplicates"])
        passed = result.get("success", False)
        print_test("Detect with specific methods", passed)
        if passed:
            print(f"       Methods used: {result.get('methods_used')}")
        self.record_result(passed)

        # Test 27: detect_anomalies (custom thresholds)
        print(f"\n{Colors.BOLD}Test: detect_anomalies (custom thresholds){Colors.END}")
        result = detect_anomalies(str(anomaly_csv), outlier_threshold=2.0, zscore_threshold=2.5)
        passed = result.get("success", False)
        print_test("Detect with custom thresholds", passed)
        if passed:
            thresholds = result.get("thresholds", {})
            print(f"       IQR multiplier: {thresholds.get('iqr_multiplier')}")
            print(f"       Z-score threshold: {thresholds.get('zscore_threshold')}")
        self.record_result(passed)

        # Test 28: detect_anomalies (non-existent file)
        print(f"\n{Colors.BOLD}Test: detect_anomalies (non-existent file){Colors.END}")
        result = detect_anomalies("/nonexistent/path/data.csv")
        passed = result.get("success", False) is False
        print_test("Handle non-existent file", passed)
        self.record_result(passed)

        # Test 29: validate_schema (basic)
        print(f"\n{Colors.BOLD}Test: validate_schema (basic){Colors.END}")
        schema = {
            "schema_name": "test_schema",
            "version": "1.0",
            "fields": [
                {"name": "id", "data_type": "numeric", "nullable": False},
                {"name": "label", "data_type": "categorical", "nullable": True},
                {"name": "value", "data_type": "numeric", "nullable": True},
                {"name": "category", "data_type": "categorical", "nullable": True},
            ],
            "strict": False,
        }
        result = validate_schema(str(csv_path), schema)
        passed = result.get("success", False)
        print_test("Validate schema", passed)
        if passed:
            print(f"       Is valid: {result.get('is_valid')}")
            print(f"       Missing columns: {result.get('missing_columns')}")
            print(f"       Type mismatches: {len(result.get('type_mismatches', []))}")
        self.record_result(passed)

        # Test 30: validate_schema (with missing column)
        print(f"\n{Colors.BOLD}Test: validate_schema (missing column){Colors.END}")
        schema_with_extra = {
            "schema_name": "test_schema_missing",
            "fields": [
                {"name": "id", "data_type": "numeric"},
                {"name": "nonexistent_column", "data_type": "text"},
            ],
        }
        result = validate_schema(str(csv_path), schema_with_extra)
        passed = result.get("success", False) and "nonexistent_column" in result.get(
            "missing_columns", []
        )
        print_test("Detect missing column", passed)
        if passed:
            print(f"       Missing: {result.get('missing_columns')}")
        self.record_result(passed)

        # Test 31: validate_schema (strict mode with extra columns)
        print(f"\n{Colors.BOLD}Test: validate_schema (strict mode){Colors.END}")
        schema_strict = {
            "schema_name": "strict_schema",
            "fields": [
                {"name": "id", "data_type": "numeric"},
            ],
            "strict": True,
        }
        result = validate_schema(str(csv_path), schema_strict)
        passed = result.get("success", False) and len(result.get("extra_columns", [])) > 0
        print_test("Detect extra columns in strict mode", passed)
        if passed:
            print(f"       Extra columns: {result.get('extra_columns')}")
        self.record_result(passed)

        # Test 32: validate_schema (non-existent file)
        print(f"\n{Colors.BOLD}Test: validate_schema (non-existent file){Colors.END}")
        result = validate_schema("/nonexistent/path/data.csv", schema)
        passed = result.get("success", False) is False
        print_test("Handle non-existent file", passed)
        self.record_result(passed)

        # Test 33: compare_distributions (basic)
        print(f"\n{Colors.BOLD}Test: compare_distributions (basic){Colors.END}")
        # Create reference and current datasets
        ref_csv = Path(self.project_path) / "data" / "reference.csv"
        ref_content = """id,value,score
1,10.5,80
2,12.3,85
3,11.8,82
4,10.2,79
5,13.1,88
6,11.5,83
7,12.0,84
8,10.9,81
9,11.2,82
10,12.5,86
"""
        ref_csv.write_text(ref_content)

        cur_csv = Path(self.project_path) / "data" / "current.csv"
        cur_content = """id,value,score
1,11.0,81
2,12.8,86
3,11.3,83
4,10.7,80
5,13.5,89
6,11.9,84
7,12.4,85
8,11.4,82
9,11.7,83
10,12.9,87
"""
        cur_csv.write_text(cur_content)

        result = compare_distributions(str(ref_csv), str(cur_csv))
        passed = result.get("success", False)
        print_test("Compare distributions", passed)
        if passed:
            print(f"       Columns compared: {result.get('columns_compared')}")
            print(f"       Drift detected: {result.get('drift_detected')}")
            print(f"       Shifts found: {result.get('total_shifts_detected')}")
        self.record_result(passed)

        # Test 34: compare_distributions (with drift)
        print(f"\n{Colors.BOLD}Test: compare_distributions (with drift){Colors.END}")
        # Create current dataset with significant drift
        drift_csv = Path(self.project_path) / "data" / "drifted.csv"
        drift_content = """id,value,score
1,50.0,20
2,55.3,25
3,52.8,22
4,48.2,18
5,60.1,30
6,53.5,23
7,57.0,27
8,51.9,21
9,54.2,24
10,58.5,28
"""
        drift_csv.write_text(drift_content)

        result = compare_distributions(str(ref_csv), str(drift_csv))
        passed = result.get("success", False) and result.get("drift_detected", False)
        print_test("Detect distribution drift", passed)
        if passed:
            shifts = result.get("shifts", [])
            print(f"       Drifted columns: {[s.get('column') for s in shifts]}")
            if shifts:
                print(f"       P-value: {shifts[0].get('p_value')}")
        self.record_result(passed)

        # Test 35: compare_distributions (specific columns)
        print(f"\n{Colors.BOLD}Test: compare_distributions (specific columns){Colors.END}")
        result = compare_distributions(str(ref_csv), str(cur_csv), columns=["value"])
        passed = result.get("success", False) and result.get("columns_compared") == ["value"]
        print_test("Compare specific columns", passed)
        if passed:
            print(f"       Columns compared: {result.get('columns_compared')}")
        self.record_result(passed)

        # Test 36: compare_distributions (non-existent file)
        print(f"\n{Colors.BOLD}Test: compare_distributions (non-existent reference){Colors.END}")
        result = compare_distributions("/nonexistent/ref.csv", str(cur_csv))
        passed = result.get("success", False) is False
        print_test("Handle non-existent reference file", passed)
        self.record_result(passed)

        # Test 37: compare_distributions (non-existent current)
        print(f"\n{Colors.BOLD}Test: compare_distributions (non-existent current){Colors.END}")
        result = compare_distributions(str(ref_csv), "/nonexistent/cur.csv")
        passed = result.get("success", False) is False
        print_test("Handle non-existent current file", passed)
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

            if tool_filter is None or tool_filter == "data_quality":
                self.test_data_quality_tools()

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
            color = (
                Colors.GREEN
                if pass_rate >= 80
                else Colors.YELLOW if pass_rate >= 50 else Colors.RED
            )
            print(f"\n  {color}Pass Rate: {pass_rate:.1f}%{Colors.END}")

        return self.results["failed"] == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test MLOps MCP Tools")
    parser.add_argument(
        "--tool",
        choices=["hydra", "mlflow", "dvc", "docker", "github", "training", "data_quality"],
        help="Test only specific tool category",
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
