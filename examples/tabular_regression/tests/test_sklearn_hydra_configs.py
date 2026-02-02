"""Tests for sklearn training Hydra configuration files."""

from pathlib import Path

from omegaconf import OmegaConf

# Config directory for sklearn training (at root of tabular_regression)
CONFIG_DIR = Path(__file__).parent.parent / "configs"


class TestSklearnConfigStructure:
    """Tests for sklearn config directory structure."""

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
        assert (model_dir / "ridge.yaml").exists()
        assert (model_dir / "random_forest.yaml").exists()
        assert (model_dir / "gradient_boosting.yaml").exists()

    def test_training_configs_exist(self):
        """Test that training configs exist."""
        training_dir = CONFIG_DIR / "training"
        assert training_dir.exists()
        assert (training_dir / "default.yaml").exists()

    def test_data_configs_exist(self):
        """Test that data configs exist."""
        data_dir = CONFIG_DIR / "data"
        assert data_dir.exists()
        assert (data_dir / "california.yaml").exists()

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
        assert (exp_dir / "ridge_baseline.yaml").exists()
        assert (exp_dir / "random_forest_baseline.yaml").exists()
        assert (exp_dir / "high_accuracy.yaml").exists()


class TestSklearnConfigContent:
    """Tests for sklearn config content validity."""

    def test_main_config_loads(self):
        """Test that main config loads without error."""
        config = OmegaConf.load(CONFIG_DIR / "config.yaml")
        assert "defaults" in config
        assert "seed" in config

    def test_ridge_config_loads(self):
        """Test Ridge config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "ridge.yaml")
        assert config.name == "ridge"
        assert "alpha" in config

    def test_random_forest_config_loads(self):
        """Test RandomForest config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "random_forest.yaml")
        assert config.name == "random_forest"
        assert "n_estimators" in config
        assert "max_depth" in config

    def test_gradient_boosting_config_loads(self):
        """Test GradientBoosting config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "gradient_boosting.yaml")
        assert config.name == "gradient_boosting"
        assert "n_estimators" in config
        assert "max_depth" in config
        assert "learning_rate" in config

    def test_training_config_loads(self):
        """Test training config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "training" / "default.yaml")
        assert "test_size" in config
        assert "normalize" in config

    def test_data_config_loads(self):
        """Test data config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "data" / "california.yaml")
        assert "dataset" in config

    def test_paths_config_loads(self):
        """Test paths config loads and has required fields."""
        config = OmegaConf.load(CONFIG_DIR / "paths" / "default.yaml")
        assert "output_dir" in config

    def test_experiment_config_baseline_loads(self):
        """Test baseline experiment config loads."""
        config = OmegaConf.load(CONFIG_DIR / "experiment" / "baseline.yaml")
        assert "defaults" in config
        assert "seed" in config

    def test_experiment_config_high_accuracy_loads(self):
        """Test high_accuracy experiment config loads."""
        config = OmegaConf.load(CONFIG_DIR / "experiment" / "high_accuracy.yaml")
        assert "defaults" in config
        assert "model" in config
        # High accuracy should have tuned model parameters
        assert "n_estimators" in config.model


class TestSklearnConfigValues:
    """Tests for sklearn config value validity."""

    def test_alpha_positive(self):
        """Test that Ridge alpha is positive."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "ridge.yaml")
        assert config.alpha > 0

    def test_n_estimators_positive(self):
        """Test that n_estimators are positive for ensemble models."""
        for name in ["random_forest", "gradient_boosting"]:
            config = OmegaConf.load(CONFIG_DIR / "model" / f"{name}.yaml")
            assert config.n_estimators > 0

    def test_learning_rate_valid(self):
        """Test that learning rate is in valid range."""
        config = OmegaConf.load(CONFIG_DIR / "model" / "gradient_boosting.yaml")
        assert 0 < config.learning_rate <= 1

    def test_test_size_valid(self):
        """Test that test_size is in valid range."""
        config = OmegaConf.load(CONFIG_DIR / "training" / "default.yaml")
        assert 0 < config.test_size < 1

    def test_seed_is_integer(self):
        """Test that seed is an integer."""
        config = OmegaConf.load(CONFIG_DIR / "config.yaml")
        assert isinstance(config.seed, int)
