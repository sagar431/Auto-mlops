"""Tests for DVC pipeline configuration."""

from pathlib import Path

import pytest
import yaml


class TestDVCPipeline:
    """Tests for DVC pipeline configuration."""

    @pytest.fixture
    def project_dir(self):
        """Return the project directory path."""
        return Path(__file__).parent.parent / "project"

    def test_dvc_yaml_exists(self, project_dir):
        """Test that dvc.yaml exists."""
        assert (project_dir / "dvc.yaml").exists()

    def test_dvc_yaml_valid(self, project_dir):
        """Test that dvc.yaml is valid YAML."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        assert "stages" in config

    def test_dvc_has_prepare_data_stage(self, project_dir):
        """Test that prepare_data stage is defined."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        assert "prepare_data" in config["stages"]
        stage = config["stages"]["prepare_data"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "outs" in stage

    def test_dvc_has_train_stage(self, project_dir):
        """Test that train stage is defined."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        assert "train" in config["stages"]
        stage = config["stages"]["train"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "outs" in stage
        assert "params" in stage

    def test_dvc_has_evaluate_stage(self, project_dir):
        """Test that evaluate stage is defined."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        assert "evaluate" in config["stages"]
        stage = config["stages"]["evaluate"]
        assert "cmd" in stage
        assert "deps" in stage
        assert "metrics" in stage

    def test_train_depends_on_prepare_data_outputs(self, project_dir):
        """Test that train stage depends on prepare_data outputs."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        train_deps = config["stages"]["train"]["deps"]
        assert any("data/train" in dep for dep in train_deps)

    def test_evaluate_depends_on_train_outputs(self, project_dir):
        """Test that evaluate stage depends on train outputs."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        evaluate_deps = config["stages"]["evaluate"]["deps"]
        assert "models/best_model.pt" in evaluate_deps

    def test_train_outputs_model_files(self, project_dir):
        """Test that train stage outputs expected model files."""
        with open(project_dir / "dvc.yaml") as f:
            config = yaml.safe_load(f)
        train_outs = config["stages"]["train"]["outs"]
        assert "models/best_model.pt" in train_outs
        assert "models/final_model.pt" in train_outs
        assert "models/vocab.json" in train_outs

    def test_dvcignore_exists(self, project_dir):
        """Test that .dvcignore exists."""
        assert (project_dir / ".dvcignore").exists()
