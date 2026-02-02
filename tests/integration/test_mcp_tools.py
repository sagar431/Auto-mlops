#!/usr/bin/env python3
"""
Integration tests for MLOps MCP Tools using pytest.

Tests the MCP tools for:
- Hydra configuration management
- MLflow experiment tracking
- DVC data versioning
- Docker containerization
- GitHub Actions CI/CD
- Training control
- Data quality validation

Usage:
    pytest tests/integration/test_mcp_tools.py -v
    pytest tests/integration/test_mcp_tools.py -v -k "hydra"
    pytest tests/integration/test_mcp_tools.py -v -k "mlflow"
    pytest tests/integration/test_mcp_tools.py -v -k "dvc"
    pytest tests/integration/test_mcp_tools.py -v -k "docker"
    pytest tests/integration/test_mcp_tools.py -v -k "github"
    pytest tests/integration/test_mcp_tools.py -v -k "training"
    pytest tests/integration/test_mcp_tools.py -v -k "data_quality"
"""

from datetime import datetime
from pathlib import Path

import pytest

from mcp_mlops_tools import (
    add_data_to_dvc,
    add_workflow_step,
    analyze_project_config,
    build_ml_docker_image,
    check_data_quality,
    check_tool_installed,
    compare_distributions,
    configure_dvc_remote,
    create_dvc_pipeline,
    create_expectation_suite,
    create_github_workflow,
    create_hydra_config,
    create_ml_dockerfile,
    detect_anomalies,
    end_mlflow_run,
    get_best_mlflow_run,
    init_dvc_repo,
    init_mlflow_experiment,
    log_mlflow_metrics,
    log_mlflow_params,
    profile_dataset,
    start_mlflow_run,
    suggest_improvements,
    update_hydra_config,
    validate_dataset,
    validate_hydra_config,
    validate_schema,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_project(tmp_path):
    """Create a temporary sample project structure for testing."""
    project_path = tmp_path / "sample_project"
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

    return str(project_path)


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Create a temporary MLflow tracking directory."""
    mlruns = tmp_path / "mlruns"
    mlruns.mkdir()
    return str(mlruns)


@pytest.fixture
def dvc_remote_path(tmp_path):
    """Create a temporary DVC remote directory."""
    remote = tmp_path / "dvc_remote"
    remote.mkdir()
    return str(remote)


@pytest.fixture
def test_csv_data(sample_project):
    """Create test CSV data with some quality issues."""
    csv_path = Path(sample_project) / "data" / "test_data.csv"
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
    return str(csv_path)


@pytest.fixture
def test_images_dir(sample_project):
    """Create test image directory structure."""
    images_dir = Path(sample_project) / "images"
    (images_dir / "cats").mkdir(parents=True)
    (images_dir / "dogs").mkdir(parents=True)

    # Minimal valid 1x1 pixel PNG data
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

    return str(images_dir)


# ============================================================================
# Hydra Configuration Tools Tests
# ============================================================================


class TestHydraTools:
    """Test Hydra configuration tools."""

    def test_analyze_project_config(self, sample_project):
        """Test analyzing project configuration."""
        result = analyze_project_config(sample_project)
        assert result.get("success") is True
        assert "has_train_script" in result

    def test_create_hydra_config(self, sample_project):
        """Test creating Hydra config structure."""
        result = create_hydra_config(
            sample_project,
            config_name="config",
            model_config={"name": "resnet18", "pretrained": True},
            training_config={"epochs": 10, "learning_rate": 0.001},
        )
        assert result.get("success") is True
        assert len(result.get("created_files", [])) > 0

    def test_validate_hydra_config(self, sample_project):
        """Test validating Hydra config after creation."""
        # First create a config
        create_hydra_config(
            sample_project,
            config_name="config",
            model_config={"name": "resnet18"},
        )
        # Then validate it
        result = validate_hydra_config(sample_project, "configs/config.yaml")
        assert result.get("success") is True or result.get("valid") is True

    def test_update_hydra_config(self, sample_project):
        """Test updating Hydra config."""
        # First create a config
        create_hydra_config(sample_project, config_name="config")
        # Then update it
        result = update_hydra_config(sample_project, "configs/config.yaml", {"seed": 123})
        assert result.get("success") is True


# ============================================================================
# MLflow Experiment Tracking Tools Tests
# ============================================================================


class TestMLflowTools:
    """Test MLflow experiment tracking tools."""

    @pytest.fixture(autouse=True)
    def check_mlflow(self):
        """Skip tests if MLflow is not available."""
        try:
            import mlflow  # noqa: F401

            self.mlflow_available = True
        except ImportError:
            self.mlflow_available = False

    @pytest.fixture
    def mlflow_session(self, tmp_path):
        """Create an isolated MLflow session for testing."""
        try:
            import mlflow

            tracking_uri = str(tmp_path / "mlruns")
            mlflow.set_tracking_uri(tracking_uri)
            yield {"tracking_uri": tracking_uri, "mlflow": mlflow}
            # Clean up any active runs
            if mlflow.active_run():
                mlflow.end_run()
        except ImportError:
            pytest.skip("MLflow not installed")

    def test_init_mlflow_experiment(self, mlflow_session):
        """Test initializing MLflow experiment."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        result = init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        assert result.get("success") is True

    def test_start_mlflow_run(self, mlflow_session):
        """Test starting MLflow run."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        result = start_mlflow_run(experiment_name=experiment_name, run_name="test_run")
        assert result.get("success") is True

    def test_log_mlflow_params(self, mlflow_session):
        """Test logging MLflow parameters."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        mlflow = mlflow_session["mlflow"]
        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        # Start a run using mlflow directly to ensure it's active
        mlflow.start_run(run_name="test_run")
        result = log_mlflow_params(params={"learning_rate": 0.001, "epochs": 10})
        assert result.get("success") is True

    def test_log_mlflow_metrics(self, mlflow_session):
        """Test logging MLflow metrics."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        mlflow = mlflow_session["mlflow"]
        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        # Start a run using mlflow directly to ensure it's active
        mlflow.start_run(run_name="test_run")
        result = log_mlflow_metrics(metrics={"accuracy": 0.85, "loss": 0.15}, step=1)
        assert result.get("success") is True

    def test_end_mlflow_run(self, mlflow_session):
        """Test ending MLflow run."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        mlflow = mlflow_session["mlflow"]
        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        # Start a run using mlflow directly to ensure it's active
        mlflow.start_run(run_name="test_run")
        result = end_mlflow_run(status="FINISHED")
        assert result.get("success") is True

    def test_get_best_mlflow_run(self, mlflow_session):
        """Test getting best MLflow run."""
        if not self.mlflow_available:
            pytest.skip("MLflow not installed")

        mlflow = mlflow_session["mlflow"]
        experiment_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        init_mlflow_experiment(
            experiment_name=experiment_name,
            tracking_uri=mlflow_session["tracking_uri"],
        )
        # Start a run using mlflow directly
        mlflow.start_run(run_name="test_run")
        mlflow.log_metric("accuracy", 0.85)
        mlflow.end_run()

        result = get_best_mlflow_run(experiment_name=experiment_name, metric_name="accuracy")
        assert result.get("success") is True


# ============================================================================
# DVC Data Versioning Tools Tests
# ============================================================================


class TestDVCTools:
    """Test DVC data versioning tools."""

    @pytest.fixture(autouse=True)
    def check_dvc(self):
        """Skip tests if DVC is not available or has dependency issues."""
        self.dvc_available = check_tool_installed("dvc")
        if self.dvc_available:
            # Check for dependency issues by trying to run dvc version
            import subprocess

            result = subprocess.run(["dvc", "version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                self.dvc_available = False
                self.dvc_error = result.stderr

    def test_init_dvc_repo(self, sample_project):
        """Test initializing DVC repository."""
        if not self.dvc_available:
            pytest.skip("DVC not installed or has dependency issues")

        result = init_dvc_repo(sample_project, no_scm=True)
        # Skip if DVC has dependency issues
        if not result.get("success") and "cannot import name" in result.get("stderr", ""):
            pytest.skip("DVC has dependency issues")
        assert result.get("success") is True

    def test_configure_dvc_remote(self, sample_project, dvc_remote_path):
        """Test configuring DVC remote."""
        if not self.dvc_available:
            pytest.skip("DVC not installed or has dependency issues")

        init_result = init_dvc_repo(sample_project, no_scm=True)
        if not init_result.get("success"):
            pytest.skip("DVC initialization failed")

        result = configure_dvc_remote(
            sample_project,
            remote_name="local_test",
            remote_url=dvc_remote_path,
            default=True,
        )
        assert result.get("success") is True

    def test_add_data_to_dvc(self, sample_project, dvc_remote_path):
        """Test adding data to DVC."""
        if not self.dvc_available:
            pytest.skip("DVC not installed or has dependency issues")

        init_result = init_dvc_repo(sample_project, no_scm=True)
        if not init_result.get("success"):
            pytest.skip("DVC initialization failed")

        configure_dvc_remote(
            sample_project,
            remote_name="local_test",
            remote_url=dvc_remote_path,
            default=True,
        )
        result = add_data_to_dvc(sample_project, "data/sample.csv")
        # Skip if DVC has dependency issues
        if not result.get("success") and "cannot import name" in result.get("stderr", ""):
            pytest.skip("DVC has dependency issues")
        assert result.get("success") is True
        assert result.get("dvc_file") is not None

    def test_create_dvc_pipeline(self, sample_project, dvc_remote_path):
        """Test creating DVC pipeline."""
        if not self.dvc_available:
            pytest.skip("DVC not installed or has dependency issues")

        init_result = init_dvc_repo(sample_project, no_scm=True)
        if not init_result.get("success"):
            pytest.skip("DVC initialization failed")

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
        result = create_dvc_pipeline(sample_project, stages)
        assert result.get("success") is True
        assert result.get("stages") is not None


# ============================================================================
# Docker Tools Tests
# ============================================================================


class TestDockerTools:
    """Test Docker tools."""

    def test_create_ml_dockerfile(self, sample_project):
        """Test creating ML Dockerfile."""
        result = create_ml_dockerfile(
            sample_project,
            base_image="python:3.11-slim",
            entry_point="train.py",
            expose_port=8000,
        )
        assert result.get("success") is True
        assert result.get("dockerfile_path") is not None

    def test_build_ml_docker_image(self, sample_project):
        """Test building Docker image."""
        if not check_tool_installed("docker"):
            pytest.skip("Docker not installed")

        # First create the Dockerfile
        create_ml_dockerfile(
            sample_project,
            base_image="python:3.11-slim",
            entry_point="train.py",
        )

        result = build_ml_docker_image(sample_project, image_name="mlops-test", tag="latest")
        # Skip if Docker daemon is not accessible (permission denied, not running, etc.)
        if not result.get("success"):
            error_msg = result.get("stderr", "") or result.get("error", "")
            if "permission denied" in error_msg.lower() or "cannot connect" in error_msg.lower():
                pytest.skip("Docker daemon not accessible (permission denied or not running)")
        assert result.get("success") is True


# ============================================================================
# GitHub Actions Tools Tests
# ============================================================================


class TestGitHubActionsTools:
    """Test GitHub Actions tools."""

    def test_create_github_workflow(self, sample_project):
        """Test creating GitHub workflow."""
        result = create_github_workflow(
            sample_project,
            workflow_name="ml-pipeline",
            trigger_on=["push", "workflow_dispatch"],
            use_dvc=True,
            use_mlflow=True,
            accuracy_threshold=0.85,
        )
        assert result.get("success") is True
        assert result.get("workflow_path") is not None

    def test_add_workflow_step(self, sample_project):
        """Test adding workflow step."""
        # First create the workflow
        create_github_workflow(
            sample_project,
            workflow_name="ml-pipeline",
            trigger_on=["push"],
        )

        result = add_workflow_step(
            sample_project,
            workflow_file=".github/workflows/ml-pipeline.yml",
            job_name="train",
            step={"name": "Custom step", "run": "echo 'Hello'"},
        )
        assert result.get("success") is True


# ============================================================================
# Training Control Tools Tests
# ============================================================================


class TestTrainingControlTools:
    """Test training control tools."""

    def test_suggest_improvements_attempt_1(self):
        """Test suggesting improvements on first attempt."""
        result = suggest_improvements(
            current_metrics={"accuracy": 0.72, "loss": 0.45},
            current_config={"learning_rate": 0.01, "epochs": 10, "batch_size": 32},
            target_accuracy=0.85,
            attempt_number=1,
        )
        assert result.get("success") is True
        assert "gap" in result
        assert "config_changes" in result
        assert "reasoning" in result

    def test_suggest_improvements_attempt_2(self):
        """Test suggesting improvements on second attempt with regularization."""
        result = suggest_improvements(
            current_metrics={"accuracy": 0.78, "loss": 0.35},
            current_config={"learning_rate": 0.005, "epochs": 15, "batch_size": 32},
            target_accuracy=0.85,
            attempt_number=2,
        )
        assert result.get("success") is True
        assert "gap" in result
        assert "config_changes" in result


# ============================================================================
# Data Quality Tools Tests
# ============================================================================


class TestDataQualityTools:
    """Test data quality tools."""

    def test_validate_dataset_csv(self, test_csv_data):
        """Test validating CSV dataset."""
        result = validate_dataset(test_csv_data, dataset_type="csv")
        assert result.get("success") is True
        assert "statistics" in result
        assert "is_valid" in result

    def test_validate_dataset_auto_detect(self, test_csv_data):
        """Test auto-detecting dataset type."""
        result = validate_dataset(test_csv_data)
        assert result.get("success") is True
        assert result.get("dataset_type") == "csv"

    def test_validate_dataset_specific_checks(self, test_csv_data):
        """Test validating with specific checks."""
        result = validate_dataset(
            test_csv_data, dataset_type="csv", checks=["missing_values", "duplicates"]
        )
        assert result.get("success") is True
        assert "checks_performed" in result

    def test_validate_dataset_images(self, test_images_dir):
        """Test validating image dataset."""
        result = validate_dataset(test_images_dir, dataset_type="images")
        assert result.get("success") is True
        assert "statistics" in result

    def test_validate_dataset_nonexistent_path(self):
        """Test handling non-existent path."""
        result = validate_dataset("/nonexistent/path/data.csv")
        assert result.get("success") is False
        assert "error" in result

    def test_validate_dataset_sample_size(self, sample_project):
        """Test validating with sample size."""
        # Create larger CSV
        large_csv_path = Path(sample_project) / "data" / "large_data.csv"
        rows = ["id,value,label"]
        for i in range(100):
            rows.append(f"{i},{i * 1.5},{'cat' if i % 2 == 0 else 'dog'}")
        large_csv_path.write_text("\n".join(rows))

        result = validate_dataset(str(large_csv_path), dataset_type="csv", sample_size=10)
        assert result.get("success") is True


class TestExpectationSuite:
    """Test expectation suite creation."""

    def test_create_expectation_suite_basic(self, sample_project):
        """Test creating basic expectation suite."""
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
        result = create_expectation_suite(sample_project, "test_suite", expectations)
        assert result.get("success") is True
        assert result.get("suite_name") == "test_suite"
        assert result.get("expectation_count") == 3

        # Verify file was created and is valid JSON
        import json

        suite_path = Path(result.get("suite_path"))
        assert suite_path.exists()
        with open(suite_path) as f:
            suite_data = json.load(f)
        assert "expectations" in suite_data

    def test_create_expectation_suite_with_kwargs(self, sample_project):
        """Test creating expectation suite with kwargs."""
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
            sample_project, "test_suite_kwargs", expectations_with_kwargs
        )
        assert result.get("success") is True
        assert "expectation_types" in result

    def test_create_expectation_suite_custom_output_dir(self, sample_project):
        """Test creating expectation suite in custom directory."""
        result = create_expectation_suite(
            sample_project,
            "custom_dir_suite",
            [{"expectation_type": "expect_column_to_exist", "column": "id"}],
            output_dir="custom_expectations",
        )
        assert result.get("success") is True
        custom_path = Path(sample_project) / "custom_expectations" / "custom_dir_suite.json"
        assert custom_path.exists()

    def test_create_expectation_suite_invalid_expectations(self, sample_project):
        """Test rejecting invalid expectations."""
        invalid_expectations = [
            {"column": "id"},  # Missing expectation_type
            {"expectation_type": ""},  # Empty expectation_type
        ]
        result = create_expectation_suite(sample_project, "invalid_suite", invalid_expectations)
        assert result.get("success") is False
        assert "error" in result

    def test_create_expectation_suite_empty(self, sample_project):
        """Test rejecting empty expectations."""
        result = create_expectation_suite(sample_project, "empty_suite", [])
        assert result.get("success") is False

    def test_create_expectation_suite_nonexistent_path(self):
        """Test handling non-existent path."""
        result = create_expectation_suite(
            "/nonexistent/path",
            "test_suite",
            [{"expectation_type": "expect_column_to_exist", "column": "id"}],
        )
        assert result.get("success") is False
        assert "error" in result


class TestCheckDataQuality:
    """Test data quality check functions."""

    def test_check_data_quality_basic(self, test_csv_data):
        """Test basic data quality check."""
        result = check_data_quality(test_csv_data)
        assert result.get("success") is True
        assert "overall_score" in result
        assert "passed_checks" in result
        assert "failed_checks" in result
        assert "is_valid" in result

    def test_check_data_quality_custom_expectations(self, test_csv_data):
        """Test data quality with custom expectations."""
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
        result = check_data_quality(test_csv_data, expectations=expectations)
        assert result.get("success") is True
        assert "validation_results" in result

    def test_check_data_quality_with_statistics(self, test_csv_data):
        """Test data quality with statistics included."""
        result = check_data_quality(test_csv_data, include_statistics=True)
        assert result.get("success") is True
        assert "statistics" in result

    def test_check_data_quality_without_statistics(self, test_csv_data):
        """Test data quality without statistics."""
        result = check_data_quality(test_csv_data, include_statistics=False)
        assert result.get("success") is True
        assert "statistics" not in result

    def test_check_data_quality_fail_on_error(self, sample_project):
        """Test fail_on_error behavior with failing check."""
        csv_with_nulls = Path(sample_project) / "data" / "data_with_nulls.csv"
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
        assert result.get("success") is False

    def test_check_data_quality_nonexistent_file(self):
        """Test handling non-existent file."""
        result = check_data_quality("/nonexistent/path/data.csv")
        assert result.get("success") is False
        assert "error" in result

    def test_check_data_quality_unsupported_file_type(self, sample_project):
        """Test rejecting unsupported file type."""
        txt_file = Path(sample_project) / "data" / "data.txt"
        txt_file.write_text("some text data")
        result = check_data_quality(str(txt_file))
        assert result.get("success") is False
        assert "error" in result

    def test_check_data_quality_parquet(self, sample_project):
        """Test data quality check on parquet file."""
        try:
            import pandas as pd

            parquet_path = Path(sample_project) / "data" / "test_data.parquet"
            df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 15.7]})
            df.to_parquet(parquet_path)

            result = check_data_quality(str(parquet_path))
            assert result.get("success") is True
        except ImportError:
            pytest.skip("pandas/pyarrow not available")


class TestProfileDataset:
    """Test dataset profiling functions."""

    def test_profile_dataset_basic(self, test_csv_data):
        """Test basic dataset profiling."""
        result = profile_dataset(test_csv_data)
        assert result.get("success") is True
        assert "statistics" in result
        assert "columns" in result

    def test_profile_dataset_custom_name(self, test_csv_data):
        """Test profiling with custom dataset name."""
        result = profile_dataset(test_csv_data, dataset_name="my_custom_dataset")
        assert result.get("success") is True
        assert result.get("dataset_name") == "my_custom_dataset"

    def test_profile_dataset_without_column_stats(self, test_csv_data):
        """Test profiling without column statistics."""
        result = profile_dataset(test_csv_data, include_column_stats=False)
        assert result.get("success") is True
        assert "columns" not in result

    def test_profile_dataset_nonexistent_file(self):
        """Test handling non-existent file."""
        result = profile_dataset("/nonexistent/path/data.csv")
        assert result.get("success") is False
        assert "error" in result


class TestDetectAnomalies:
    """Test anomaly detection functions."""

    @pytest.fixture
    def anomaly_csv(self, sample_project):
        """Create CSV with outliers for anomaly detection."""
        anomaly_csv = Path(sample_project) / "data" / "anomaly_data.csv"
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
        return str(anomaly_csv)

    def test_detect_anomalies_basic(self, anomaly_csv):
        """Test basic anomaly detection."""
        result = detect_anomalies(anomaly_csv)
        assert result.get("success") is True
        assert "total_anomalies" in result
        assert "affected_rows" in result
        assert "affected_percentage" in result
        assert "anomalies_by_type" in result

    def test_detect_anomalies_specific_methods(self, anomaly_csv):
        """Test anomaly detection with specific methods."""
        result = detect_anomalies(anomaly_csv, methods=["iqr", "duplicates"])
        assert result.get("success") is True
        assert "methods_used" in result

    def test_detect_anomalies_custom_thresholds(self, anomaly_csv):
        """Test anomaly detection with custom thresholds."""
        result = detect_anomalies(anomaly_csv, outlier_threshold=2.0, zscore_threshold=2.5)
        assert result.get("success") is True
        assert "thresholds" in result

    def test_detect_anomalies_nonexistent_file(self):
        """Test handling non-existent file."""
        result = detect_anomalies("/nonexistent/path/data.csv")
        assert result.get("success") is False


class TestValidateSchema:
    """Test schema validation functions."""

    def test_validate_schema_basic(self, test_csv_data):
        """Test basic schema validation."""
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
        result = validate_schema(test_csv_data, schema)
        assert result.get("success") is True
        assert "is_valid" in result

    def test_validate_schema_missing_column(self, test_csv_data):
        """Test detecting missing column."""
        schema_with_extra = {
            "schema_name": "test_schema_missing",
            "fields": [
                {"name": "id", "data_type": "numeric"},
                {"name": "nonexistent_column", "data_type": "text"},
            ],
        }
        result = validate_schema(test_csv_data, schema_with_extra)
        assert result.get("success") is True
        assert "nonexistent_column" in result.get("missing_columns", [])

    def test_validate_schema_strict_mode(self, test_csv_data):
        """Test detecting extra columns in strict mode."""
        schema_strict = {
            "schema_name": "strict_schema",
            "fields": [
                {"name": "id", "data_type": "numeric"},
            ],
            "strict": True,
        }
        result = validate_schema(test_csv_data, schema_strict)
        assert result.get("success") is True
        assert len(result.get("extra_columns", [])) > 0

    def test_validate_schema_nonexistent_file(self):
        """Test handling non-existent file."""
        schema = {"schema_name": "test", "fields": []}
        result = validate_schema("/nonexistent/path/data.csv", schema)
        assert result.get("success") is False


class TestCompareDistributions:
    """Test distribution comparison functions."""

    @pytest.fixture
    def reference_csv(self, sample_project):
        """Create reference CSV for distribution comparison."""
        ref_csv = Path(sample_project) / "data" / "reference.csv"
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
        return str(ref_csv)

    @pytest.fixture
    def current_csv(self, sample_project):
        """Create current CSV for distribution comparison."""
        cur_csv = Path(sample_project) / "data" / "current.csv"
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
        return str(cur_csv)

    @pytest.fixture
    def drifted_csv(self, sample_project):
        """Create CSV with significant drift."""
        drift_csv = Path(sample_project) / "data" / "drifted.csv"
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
        return str(drift_csv)

    def test_compare_distributions_basic(self, reference_csv, current_csv):
        """Test basic distribution comparison."""
        result = compare_distributions(reference_csv, current_csv)
        assert result.get("success") is True
        assert "columns_compared" in result
        assert "drift_detected" in result

    def test_compare_distributions_with_drift(self, reference_csv, drifted_csv):
        """Test detecting distribution drift."""
        result = compare_distributions(reference_csv, drifted_csv)
        assert result.get("success") is True
        assert result.get("drift_detected") is True

    def test_compare_distributions_specific_columns(self, reference_csv, current_csv):
        """Test comparing specific columns."""
        result = compare_distributions(reference_csv, current_csv, columns=["value"])
        assert result.get("success") is True
        assert result.get("columns_compared") == ["value"]

    def test_compare_distributions_nonexistent_reference(self, current_csv):
        """Test handling non-existent reference file."""
        result = compare_distributions("/nonexistent/ref.csv", current_csv)
        assert result.get("success") is False

    def test_compare_distributions_nonexistent_current(self, reference_csv):
        """Test handling non-existent current file."""
        result = compare_distributions(reference_csv, "/nonexistent/cur.csv")
        assert result.get("success") is False
