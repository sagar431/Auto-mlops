"""Tests for Hydra configuration files."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Config directory path
CONFIG_DIR = str(Path(__file__).parent.parent / "project" / "configs")


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra's global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


class TestMainConfig:
    """Tests for the main config.yaml."""

    def test_load_default_config(self):
        """Test loading the default configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config")

            # Check that defaults are loaded
            assert "model" in cfg
            assert "training" in cfg
            assert "data" in cfg
            assert "paths" in cfg
            assert "seed" in cfg

    def test_default_seed(self):
        """Test default seed value."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config")
            assert cfg.seed == 42


class TestModelConfigs:
    """Tests for model configuration files."""

    def test_cifar10_cnn_config(self):
        """Test CIFAR10 CNN model configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["model=cifar10_cnn"])

            assert cfg.model.name == "cifar10_cnn"
            assert cfg.model.num_classes == 10
            assert cfg.model.dropout == 0.5

    def test_resnet18_config(self):
        """Test ResNet18 model configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["model=resnet18"])

            assert cfg.model.name == "resnet18"
            assert cfg.model.num_classes == 10
            assert not cfg.model.pretrained


class TestTrainingConfigs:
    """Tests for training configuration files."""

    def test_default_training_config(self):
        """Test default training configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["training=default"])

            assert cfg.training.epochs == 10
            assert cfg.training.batch_size == 128
            assert cfg.training.learning_rate == 0.001
            assert cfg.training.optimizer == "adam"

    def test_fast_training_config(self):
        """Test fast training configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["training=fast"])

            assert cfg.training.epochs == 3
            assert cfg.training.batch_size == 256
            assert cfg.training.learning_rate == 0.01
            assert cfg.training.optimizer == "adam"

    def test_long_training_config(self):
        """Test long training configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["training=long"])

            assert cfg.training.epochs == 50
            assert cfg.training.batch_size == 64
            assert cfg.training.learning_rate == 0.001
            assert cfg.training.weight_decay == 0.0001
            assert cfg.training.scheduler.name == "cosine"

    def test_sgd_training_config(self):
        """Test SGD training configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["training=sgd"])

            assert cfg.training.epochs == 30
            assert cfg.training.batch_size == 128
            assert cfg.training.learning_rate == 0.1
            assert cfg.training.optimizer == "sgd"
            assert cfg.training.momentum == 0.9
            assert cfg.training.weight_decay == 0.0005


class TestDataConfigs:
    """Tests for data configuration files."""

    def test_cifar10_data_config(self):
        """Test CIFAR-10 data configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["data=cifar10"])

            assert cfg.data.dataset == "cifar10"
            assert cfg.data.image_size == 32
            assert cfg.data.num_workers == 4
            assert cfg.data.augmentation.horizontal_flip
            assert cfg.data.augmentation.random_crop
            assert cfg.data.augmentation.crop_padding == 4

    def test_cifar10_minimal_data_config(self):
        """Test CIFAR-10 minimal data configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["data=cifar10_minimal"])

            assert cfg.data.dataset == "cifar10"
            assert cfg.data.image_size == 32
            assert cfg.data.num_workers == 2
            assert cfg.data.augmentation.horizontal_flip
            assert not cfg.data.augmentation.random_crop

    def test_cifar10_normalization_values(self):
        """Test CIFAR-10 normalization values are correct."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["data=cifar10"])

            # Check normalization values match CIFAR-10 statistics
            assert cfg.data.normalize.mean == [0.4914, 0.4822, 0.4465]
            assert cfg.data.normalize.std == [0.2470, 0.2435, 0.2616]


class TestPathsConfigs:
    """Tests for paths configuration files."""

    def test_default_paths_config(self):
        """Test default paths configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["paths=default"])

            assert "output_dir" in cfg.paths
            assert "log_dir" in cfg.paths
            assert "checkpoint_dir" in cfg.paths


class TestExperimentConfigs:
    """Tests for experiment configuration files."""

    def test_baseline_experiment(self):
        """Test baseline experiment configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["+experiment=baseline"])

            assert cfg.experiment_name == "baseline"
            assert cfg.model.name == "cifar10_cnn"
            assert cfg.training.epochs == 10
            assert cfg.data.dataset == "cifar10"

    def test_quick_test_experiment(self):
        """Test quick_test experiment configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["+experiment=quick_test"])

            assert cfg.experiment_name == "quick_test"
            assert cfg.model.name == "cifar10_cnn"
            assert cfg.training.epochs == 3
            assert cfg.data.num_workers == 2

    def test_high_accuracy_experiment(self):
        """Test high_accuracy experiment configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["+experiment=high_accuracy"])

            assert cfg.experiment_name == "high_accuracy"
            assert cfg.model.name == "cifar10_cnn"
            assert cfg.model.dropout == 0.3  # Overridden in experiment
            assert cfg.training.epochs == 50
            assert cfg.training.scheduler.name == "cosine"

    def test_resnet_baseline_experiment(self):
        """Test resnet_baseline experiment configuration."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["+experiment=resnet_baseline"])

            assert cfg.experiment_name == "resnet_baseline"
            assert cfg.model.name == "resnet18"
            assert cfg.training.optimizer == "sgd"
            assert cfg.training.momentum == 0.9


class TestConfigOverrides:
    """Tests for configuration overrides."""

    def test_override_model(self):
        """Test overriding model config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["model=resnet18"])

            assert cfg.model.name == "resnet18"

    def test_override_training(self):
        """Test overriding training config."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config", overrides=["training=fast"])

            assert cfg.training.epochs == 3

    def test_override_individual_values(self):
        """Test overriding individual config values."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "training.epochs=20",
                    "training.learning_rate=0.01",
                    "model.dropout=0.3",
                ],
            )

            assert cfg.training.epochs == 20
            assert cfg.training.learning_rate == 0.01
            assert cfg.model.dropout == 0.3

    def test_combine_model_and_training(self):
        """Test combining different model and training configs."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="config",
                overrides=["model=resnet18", "training=long"],
            )

            assert cfg.model.name == "resnet18"
            assert cfg.training.epochs == 50


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_all_configs_are_valid_yaml(self):
        """Test that all config files are valid YAML and loadable by Hydra."""
        config_path = Path(CONFIG_DIR)

        # Test main config
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config")
            assert cfg is not None

        GlobalHydra.instance().clear()

        # Test all model configs
        for model_config in (config_path / "model").glob("*.yaml"):
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"model={model_config.stem}"],
                )
                assert cfg.model is not None
            GlobalHydra.instance().clear()

        # Test all training configs
        for training_config in (config_path / "training").glob("*.yaml"):
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"training={training_config.stem}"],
                )
                assert cfg.training is not None
            GlobalHydra.instance().clear()

        # Test all data configs
        for data_config in (config_path / "data").glob("*.yaml"):
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"data={data_config.stem}"],
                )
                assert cfg.data is not None
            GlobalHydra.instance().clear()

        # Test all experiment configs
        for exp_config in (config_path / "experiment").glob("*.yaml"):
            with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[f"+experiment={exp_config.stem}"],
                )
                assert cfg is not None
            GlobalHydra.instance().clear()

    def test_config_types(self):
        """Test that config values have correct types."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(config_name="config")

            # Check integer types
            assert isinstance(cfg.seed, int)
            assert isinstance(cfg.model.num_classes, int)
            assert isinstance(cfg.training.epochs, int)
            assert isinstance(cfg.training.batch_size, int)
            assert isinstance(cfg.data.image_size, int)
            assert isinstance(cfg.data.num_workers, int)

            # Check float types
            assert isinstance(cfg.model.dropout, float)
            assert isinstance(cfg.training.learning_rate, float)

            # Check string types
            assert isinstance(cfg.training.optimizer, str)

            # Check list types (OmegaConf returns ListConfig, so convert to list)
            assert len(OmegaConf.to_container(cfg.data.normalize.mean)) == 3
            assert len(OmegaConf.to_container(cfg.data.normalize.std)) == 3
