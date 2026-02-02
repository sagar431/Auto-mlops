"""Tests for Hydra configuration files."""

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "project"))


class TestConfigFiles:
    """Tests for configuration file validity."""

    @pytest.fixture
    def config_dir(self):
        """Return the config directory path."""
        return Path(__file__).parent.parent / "project" / "configs"

    def test_main_config_exists(self, config_dir):
        """Test that main config file exists."""
        assert (config_dir / "config.yaml").exists()

    def test_main_config_valid(self, config_dir):
        """Test that main config file is valid YAML."""
        config = OmegaConf.load(config_dir / "config.yaml")
        assert "defaults" in config
        assert "seed" in config

    def test_model_configs_exist(self, config_dir):
        """Test that model config files exist."""
        model_dir = config_dir / "model"
        assert model_dir.exists()
        assert (model_dir / "textcnn.yaml").exists()
        assert (model_dir / "lstm.yaml").exists()
        assert (model_dir / "distilbert.yaml").exists()

    def test_textcnn_config_valid(self, config_dir):
        """Test TextCNN config is valid."""
        config = OmegaConf.load(config_dir / "model" / "textcnn.yaml")
        assert config.name == "textcnn"
        assert "num_classes" in config
        assert "embedding_dim" in config
        assert "dropout" in config
        assert "num_filters" in config
        assert "kernel_sizes" in config

    def test_lstm_config_valid(self, config_dir):
        """Test LSTM config is valid."""
        config = OmegaConf.load(config_dir / "model" / "lstm.yaml")
        assert config.name == "lstm"
        assert "num_classes" in config
        assert "embedding_dim" in config
        assert "dropout" in config
        assert "hidden_dim" in config
        assert "num_layers" in config
        assert "bidirectional" in config

    def test_distilbert_config_valid(self, config_dir):
        """Test DistilBERT config is valid."""
        config = OmegaConf.load(config_dir / "model" / "distilbert.yaml")
        assert config.name == "distilbert"
        assert "num_classes" in config
        assert "embedding_dim" in config
        assert "dropout" in config
        assert "pretrained_model_name" in config
        assert "freeze_encoder" in config

    def test_training_configs_exist(self, config_dir):
        """Test that training config files exist."""
        training_dir = config_dir / "training"
        assert training_dir.exists()
        assert (training_dir / "default.yaml").exists()
        assert (training_dir / "fast.yaml").exists()
        assert (training_dir / "long.yaml").exists()

    def test_training_config_valid(self, config_dir):
        """Test training config is valid."""
        config = OmegaConf.load(config_dir / "training" / "default.yaml")
        assert "epochs" in config
        assert "batch_size" in config
        assert "learning_rate" in config
        assert "optimizer" in config

    def test_data_configs_exist(self, config_dir):
        """Test that data config files exist."""
        data_dir = config_dir / "data"
        assert data_dir.exists()
        assert (data_dir / "imdb.yaml").exists()
        assert (data_dir / "synthetic.yaml").exists()

    def test_data_config_valid(self, config_dir):
        """Test data config is valid."""
        config = OmegaConf.load(config_dir / "data" / "imdb.yaml")
        assert "dataset" in config
        assert "vocab_size" in config
        assert "max_length" in config
        assert "num_workers" in config

    def test_paths_config_exists(self, config_dir):
        """Test that paths config file exists."""
        paths_dir = config_dir / "paths"
        assert paths_dir.exists()
        assert (paths_dir / "default.yaml").exists()

    def test_paths_config_valid(self, config_dir):
        """Test paths config is valid."""
        config = OmegaConf.load(config_dir / "paths" / "default.yaml")
        assert "output_dir" in config
        assert "log_dir" in config

    def test_experiment_configs_exist(self, config_dir):
        """Test that experiment config files exist."""
        experiment_dir = config_dir / "experiment"
        assert experiment_dir.exists()
        assert (experiment_dir / "baseline.yaml").exists()
        assert (experiment_dir / "quick_test.yaml").exists()
        assert (experiment_dir / "lstm_baseline.yaml").exists()
        assert (experiment_dir / "high_accuracy.yaml").exists()
        assert (experiment_dir / "distilbert_baseline.yaml").exists()

    def test_experiment_config_structure(self, config_dir):
        """Test experiment configs have correct structure."""
        config = OmegaConf.load(config_dir / "experiment" / "baseline.yaml")
        assert "defaults" in config

    def test_fast_training_has_fewer_epochs(self, config_dir):
        """Test that fast training config has fewer epochs."""
        default = OmegaConf.load(config_dir / "training" / "default.yaml")
        fast = OmegaConf.load(config_dir / "training" / "fast.yaml")
        assert fast.epochs < default.epochs

    def test_long_training_has_more_epochs(self, config_dir):
        """Test that long training config has more epochs."""
        default = OmegaConf.load(config_dir / "training" / "default.yaml")
        long = OmegaConf.load(config_dir / "training" / "long.yaml")
        assert long.epochs > default.epochs
