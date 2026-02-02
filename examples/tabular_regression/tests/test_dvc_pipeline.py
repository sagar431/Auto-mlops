"""Tests for DVC pipeline configuration."""

from pathlib import Path

import yaml

# Project directory
project_dir = Path(__file__).parent.parent / "project"

DVC_FILE = project_dir / "dvc.yaml"


class TestDVCPipeline:
    """Tests for DVC pipeline configuration."""

    def test_dvc_file_exists(self):
        """Test that dvc.yaml exists."""
        assert DVC_FILE.exists()

    def test_dvc_file_loads(self):
        """Test that dvc.yaml is valid YAML."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "stages" in config

    def test_prepare_data_stage(self):
        """Test prepare_data stage configuration."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)

        stages = config["stages"]
        assert "prepare_data" in stages

        stage = stages["prepare_data"]
        assert "cmd" in stage
        assert "prepare_data.py" in stage["cmd"]
        assert "deps" in stage
        assert "outs" in stage

    def test_train_stage(self):
        """Test train stage configuration."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)

        stages = config["stages"]
        assert "train" in stages

        stage = stages["train"]
        assert "cmd" in stage
        assert "train.py" in stage["cmd"]
        assert "deps" in stage
        assert "params" in stage
        assert "outs" in stage

    def test_evaluate_stage(self):
        """Test evaluate stage configuration."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)

        stages = config["stages"]
        assert "evaluate" in stages

        stage = stages["evaluate"]
        assert "cmd" in stage
        assert "evaluate.py" in stage["cmd"]
        assert "deps" in stage
        assert "metrics" in stage

    def test_stage_dependencies_exist(self):
        """Test that all dependency files exist."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)

        # Check prepare_data deps
        for dep in config["stages"]["prepare_data"]["deps"]:
            assert (project_dir / dep).exists(), f"Missing dependency: {dep}"

        # Check train deps (only code files, not data)
        code_deps = [
            "train.py",
            "model.py",
            "dataset.py",
        ]
        for dep in code_deps:
            assert (project_dir / dep).exists(), f"Missing dependency: {dep}"

        # Check evaluate deps (only code files)
        assert (project_dir / "evaluate.py").exists()

    def test_metrics_file_configured(self):
        """Test that metrics file is configured correctly."""
        with open(DVC_FILE) as f:
            config = yaml.safe_load(f)

        metrics = config["stages"]["evaluate"]["metrics"]
        assert len(metrics) > 0
        assert "metrics.json" in metrics[0]
