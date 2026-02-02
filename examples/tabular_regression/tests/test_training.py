"""Tests for training functionality."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project directory to path
project_dir = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_dir))

from dataset import TabularDataset  # noqa: E402
from model import MLP  # noqa: E402
from train import train_epoch, validate  # noqa: E402


@pytest.fixture
def simple_model():
    """Create a simple MLP model for testing."""
    return MLP(input_dim=8, hidden_dims=[16, 8])


@pytest.fixture
def train_loader(sample_features, sample_targets):
    """Create a training dataloader."""
    dataset = TabularDataset(sample_features[:80], sample_targets[:80])
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def val_loader(sample_features, sample_targets):
    """Create a validation dataloader."""
    dataset = TabularDataset(sample_features[80:], sample_targets[80:])
    return DataLoader(dataset, batch_size=16, shuffle=False)


class TestTrainEpoch:
    """Tests for train_epoch function."""

    def test_train_epoch(self, simple_model, train_loader):
        """Test single training epoch."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        loss, rmse = train_epoch(simple_model, train_loader, optimizer, criterion, "cpu")

        assert isinstance(loss, float)
        assert isinstance(rmse, float)
        assert loss >= 0
        assert rmse >= 0

    def test_model_updates(self, simple_model, train_loader):
        """Test that model weights are updated during training."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Get initial weights
        initial_weights = simple_model.network[0].weight.clone()

        train_epoch(simple_model, train_loader, optimizer, criterion, "cpu")

        # Check weights changed
        assert not torch.allclose(initial_weights, simple_model.network[0].weight)


class TestValidate:
    """Tests for validate function."""

    def test_validate(self, simple_model, val_loader):
        """Test validation."""
        criterion = nn.MSELoss()

        loss, rmse = validate(simple_model, val_loader, criterion, "cpu")

        assert isinstance(loss, float)
        assert isinstance(rmse, float)
        assert loss >= 0
        assert rmse >= 0

    def test_validate_no_gradients(self, simple_model, val_loader):
        """Test that validation doesn't compute gradients."""
        criterion = nn.MSELoss()

        # Enable gradient tracking
        simple_model.train()
        for param in simple_model.parameters():
            param.requires_grad = True

        validate(simple_model, val_loader, criterion, "cpu")

        # Model should be in eval mode after validation
        assert not simple_model.training
