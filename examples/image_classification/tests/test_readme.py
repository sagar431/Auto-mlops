"""Tests to validate that README.md accurately reflects the project structure."""

from pathlib import Path

import pytest


@pytest.fixture
def example_dir():
    """Return the image_classification example directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def project_dir(example_dir):
    """Return the project directory."""
    return example_dir / "project"


class TestProjectStructure:
    """Tests that verify the project structure matches README documentation."""

    def test_readme_exists(self, example_dir):
        """Test that README.md exists."""
        readme = example_dir / "README.md"
        assert readme.exists(), "README.md should exist"

    def test_project_directory_exists(self, project_dir):
        """Test that project directory exists."""
        assert project_dir.exists(), "project/ directory should exist"

    def test_main_scripts_exist(self, project_dir):
        """Test that main Python scripts exist."""
        scripts = [
            "train.py",
            "evaluate.py",
            "prepare_data.py",
            "inference.py",
            "model.py",
            "dataset.py",
        ]
        for script in scripts:
            assert (project_dir / script).exists(), f"{script} should exist"

    def test_requirements_exists(self, project_dir):
        """Test that requirements.txt exists."""
        assert (project_dir / "requirements.txt").exists(), "requirements.txt should exist"

    def test_dvc_yaml_exists(self, project_dir):
        """Test that dvc.yaml exists."""
        assert (project_dir / "dvc.yaml").exists(), "dvc.yaml should exist"


class TestHydraConfigs:
    """Tests that verify Hydra config structure matches README documentation."""

    def test_configs_directory_exists(self, project_dir):
        """Test that configs directory exists."""
        assert (project_dir / "configs").exists(), "configs/ directory should exist"

    def test_main_config_exists(self, project_dir):
        """Test that main config.yaml exists."""
        assert (
            project_dir / "configs" / "config.yaml"
        ).exists(), "configs/config.yaml should exist"

    def test_model_configs_exist(self, project_dir):
        """Test that model config files exist."""
        model_dir = project_dir / "configs" / "model"
        assert model_dir.exists(), "configs/model/ directory should exist"
        assert (model_dir / "cifar10_cnn.yaml").exists(), "cifar10_cnn.yaml should exist"
        assert (model_dir / "resnet18.yaml").exists(), "resnet18.yaml should exist"

    def test_training_configs_exist(self, project_dir):
        """Test that training config files exist."""
        training_dir = project_dir / "configs" / "training"
        assert training_dir.exists(), "configs/training/ directory should exist"
        configs = ["default.yaml", "fast.yaml", "long.yaml", "sgd.yaml"]
        for config in configs:
            assert (training_dir / config).exists(), f"training/{config} should exist"

    def test_data_configs_exist(self, project_dir):
        """Test that data config files exist."""
        data_dir = project_dir / "configs" / "data"
        assert data_dir.exists(), "configs/data/ directory should exist"
        assert (data_dir / "cifar10.yaml").exists(), "cifar10.yaml should exist"
        assert (data_dir / "cifar10_minimal.yaml").exists(), "cifar10_minimal.yaml should exist"

    def test_paths_configs_exist(self, project_dir):
        """Test that paths config files exist."""
        paths_dir = project_dir / "configs" / "paths"
        assert paths_dir.exists(), "configs/paths/ directory should exist"
        assert (paths_dir / "default.yaml").exists(), "paths/default.yaml should exist"

    def test_experiment_configs_exist(self, project_dir):
        """Test that experiment config files exist."""
        exp_dir = project_dir / "configs" / "experiment"
        assert exp_dir.exists(), "configs/experiment/ directory should exist"
        experiments = [
            "baseline.yaml",
            "quick_test.yaml",
            "high_accuracy.yaml",
            "resnet_baseline.yaml",
        ]
        for exp in experiments:
            assert (exp_dir / exp).exists(), f"experiment/{exp} should exist"


class TestTestsDirectory:
    """Tests that verify tests directory structure."""

    def test_tests_directory_exists(self, example_dir):
        """Test that tests directory exists."""
        assert (example_dir / "tests").exists(), "tests/ directory should exist"

    def test_test_files_exist(self, example_dir):
        """Test that documented test files exist."""
        tests_dir = example_dir / "tests"
        test_files = [
            "test_training.py",
            "test_model.py",
            "test_dataset.py",
            "test_inference.py",
            "test_hydra_configs.py",
            "test_dvc_pipeline.py",
        ]
        for test_file in test_files:
            assert (tests_dir / test_file).exists(), f"tests/{test_file} should exist"


class TestSupportingFiles:
    """Tests that verify supporting files exist."""

    def test_setup_script_exists(self, example_dir):
        """Test that setup script exists."""
        assert (example_dir / "setup_example.sh").exists(), "setup_example.sh should exist"

    def test_agent_queries_exists(self, example_dir):
        """Test that agent_queries.md exists."""
        assert (example_dir / "agent_queries.md").exists(), "agent_queries.md should exist"

    def test_run_example_exists(self, example_dir):
        """Test that run_example.py exists."""
        assert (example_dir / "run_example.py").exists(), "run_example.py should exist"

    def test_docs_directory_exists(self, example_dir):
        """Test that docs directory exists."""
        assert (example_dir / "docs").exists(), "docs/ directory should exist"
        assert (
            example_dir / "docs" / "walkthrough.md"
        ).exists(), "docs/walkthrough.md should exist"


class TestReadmeContent:
    """Tests that verify README content is accurate."""

    def test_readme_mentions_cifar10(self, example_dir):
        """Test that README mentions CIFAR-10 dataset."""
        readme = example_dir / "README.md"
        content = readme.read_text()
        assert "CIFAR-10" in content, "README should mention CIFAR-10 dataset"

    def test_readme_mentions_hydra(self, example_dir):
        """Test that README mentions Hydra configuration."""
        readme = example_dir / "README.md"
        content = readme.read_text()
        assert "Hydra" in content, "README should mention Hydra configuration"

    def test_readme_mentions_dvc(self, example_dir):
        """Test that README mentions DVC."""
        readme = example_dir / "README.md"
        content = readme.read_text()
        assert "DVC" in content, "README should mention DVC"

    def test_readme_mentions_experiments(self, example_dir):
        """Test that README mentions experiment presets."""
        readme = example_dir / "README.md"
        content = readme.read_text()
        experiments = ["baseline", "quick_test", "high_accuracy", "resnet_baseline"]
        for exp in experiments:
            assert exp in content, f"README should mention {exp} experiment"

    def test_readme_mentions_models(self, example_dir):
        """Test that README mentions model options."""
        readme = example_dir / "README.md"
        content = readme.read_text()
        assert "CIFAR10CNN" in content, "README should mention CIFAR10CNN model"
        assert "ResNet18" in content, "README should mention ResNet18 model"
