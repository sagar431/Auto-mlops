"""Tests for Hydra configuration files."""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))

# Config directory
CONFIG_DIR = project_dir / "configs"


class TestConfigStructure:
    """Tests for config directory structure."""

    def test_config_dir_exists(self):
        """Test that config directory exists."""
        assert CONFIG_DIR.exists()

    def test_main_config_exists(self):
        """Test that main config.yaml exists."""
        assert (CONFIG_DIR / "config.yaml").exists()

    def test_model_configs_exist(self):
        """Test that model configs exist."""
        model_dir = CONFIG_DIR / "model"
        assert model_dir.exists()
        assert (model_dir / "mlp.yaml").exists()
        assert (model_dir / "tabnet.yaml").exists()

    def test_training_configs_exist(self):
        """Test that training configs exist."""
        training_dir = CONFIG_DIR / "training"
        assert training_dir.exists()
        assert (training_dir / "default.yaml").exists()
        assert (training_dir / "fast.yaml").exists()
        assert (training_dir / "long.yaml").exists()

    def test_data_configs_exist(self):
        """Test that data configs exist."""
        data_dir = CONFIG_DIR / "data"
        assert data_dir.exists()
        assert (data_dir / "california.yaml").exists()
        assert (data_dir / "synthetic.yaml").exists()

    def test_paths_configs_exist(self):
        """Test that paths configs exist."""
        paths_dir = CONFIG_DIR / "paths"
        assert paths_dir.exists()
        assert (paths_dir / "default.yaml").exists()

    def test_experiment_configs_exist(self):
        """Test that experiment configs exist."""
        exp_dir = CONFIG_DIR / "experiment"
        assert exp_dir.exists()
        assert (exp_dir / "baseline.yaml").exists()
        assert (exp_dir / "quick_test.yaml").exists()


class TestConfigContent:
    """Tests for config content validity."""

    def test_main_config_loads(self):
        """Test that main config loads without error."""
        config = OmegaConf.load(CONFIG_DIR / "config.yaml")
        assert "defaults" in config
        assert "seed" in config

    def test_mlp_config_loads(self):
        """Test MLP config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "mlp.yaml")
        assert config.name == "mlp"
        assert "hidden_dims" in config
        assert "dropout" in config

    def test_tabnet_config_loads(self):
        """Test TabNet config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "tabnet.yaml")
        assert config.name == "tabnet"
        assert "n_steps" in config
        assert "n_d" in config
        assert "n_a" in config

    def test_training_config_loads(self):
        """Test training config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "training" / "default.yaml")
        assert "epochs" in config
        assert "batch_size" in config
        assert "learning_rate" in config
        assert "optimizer" in config

    def test_data_config_loads(self):
        """Test data config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "data" / "california.yaml")
        assert "data_dir" in config
        assert "dataset" in config

    def test_experiment_config_loads(self):
        """Test experiment config loads."""
        config = OmegaConf.load(CONFIG_DIR / "experiment" / "baseline.yaml")
        assert "defaults" in config
        assert "seed" in config


class TestConfigValues:
    """Tests for config value validity."""

    def test_epochs_positive(self):
        """Test that epochs are positive."""
        for name in ["default", "fast", "long"]:
            config = OmegaConf.load(CONFIG_DIR / "training" / f"{name}.yaml")
            assert config.epochs > 0

    def test_batch_size_positive(self):
        """Test that batch sizes are positive."""
        for name in ["default", "fast", "long"]:
            config = OmegaConf.load(CONFIG_DIR / "training" / f"{name}.yaml")
            assert config.batch_size > 0

    def test_learning_rate_valid(self):
        """Test that learning rates are valid."""
        for name in ["default", "fast", "long"]:
            config = OmegaConf.load(CONFIG_DIR / "training" / f"{name}.yaml")
            assert 0 < config.learning_rate < 1

    def test_dropout_valid(self):
        """Test that dropout is in valid range."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "mlp.yaml")
        assert 0 <= config.dropout < 1

    def test_hidden_dims_valid(self):
        """Test that hidden dims are valid."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "mlp.yaml")
        assert len(config.hidden_dims) > 0
        assert all(d > 0 for d in config.hidden_dims)
