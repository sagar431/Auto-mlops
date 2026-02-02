"""Model architectures for tabular regression."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multi-Layer Perceptron for tabular regression.

    A feedforward neural network with configurable hidden layers,
    dropout, and batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.2,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x).squeeze(-1)


class GLUBlock(nn.Module):
    """Gated Linear Unit block used in TabNet."""

    def __init__(self, input_dim: int, output_dim: int, virtual_batch_size: int = None):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2, bias=False)
        self.bn = nn.BatchNorm1d(output_dim * 2, virtual_batch_size or output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class TabNet(nn.Module):
    """TabNet architecture for tabular data.

    A simplified implementation of TabNet with attention-based
    feature selection and sparse learning.

    Reference: https://arxiv.org/abs/1908.07442
    """

    def __init__(
        self,
        input_dim: int,
        n_steps: int = 3,
        n_d: int = 64,
        n_a: int = 64,
        gamma: float = 1.5,
        relaxation_factor: float = 1.5,
        epsilon: float = 1e-15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma
        self.relaxation_factor = relaxation_factor
        self.epsilon = epsilon

        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Shared layers
        self.shared_fc = nn.Linear(input_dim, n_d + n_a)
        self.shared_bn = nn.BatchNorm1d(n_d + n_a)

        # Step-specific layers
        self.step_attention = nn.ModuleList()
        self.step_transform = nn.ModuleList()

        for _ in range(n_steps):
            # Attention transformer
            self.step_attention.append(
                nn.Sequential(
                    nn.Linear(n_a, input_dim),
                    nn.BatchNorm1d(input_dim),
                )
            )
            # Feature transformer
            self.step_transform.append(GLUBlock(input_dim, n_d + n_a))

        # Final output layer
        self.final_fc = nn.Linear(n_d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Initial normalization
        x = self.initial_bn(x)

        # Prior scales (cumulative feature importance)
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)

        # Aggregate decision
        output_aggregate = torch.zeros(batch_size, self.n_d, device=x.device)

        # Initial processed features
        processed = self.shared_fc(x)
        processed = self.shared_bn(processed)
        processed = F.relu(processed)

        for step in range(self.n_steps):
            # Split into decision and attention
            d, a = processed[:, : self.n_d], processed[:, self.n_d :]

            # Aggregate decision contribution
            output_aggregate = output_aggregate + d

            # Compute attention mask
            mask = self.step_attention[step](a)
            mask = mask * prior_scales
            mask = F.softmax(mask, dim=-1)

            # Update prior scales (reduce importance of already attended features)
            prior_scales = prior_scales * (self.gamma - mask)

            # Apply mask to features
            masked_x = mask * x

            # Transform
            processed = self.step_transform[step](masked_x)
            processed = F.relu(processed)

        # Final prediction
        output = self.final_fc(output_aggregate)
        return output.squeeze(-1)


def create_model(model_type: str, **kwargs) -> nn.Module:
    """Factory function to create a model by type.

    Args:
        model_type: One of 'mlp', 'tabnet'
        **kwargs: Model-specific arguments

    Returns:
        Instantiated model
    """
    models = {
        "mlp": MLP,
        "tabnet": TabNet,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")

    return models[model_type](**kwargs)


def load_model(
    model_path: str,
    model_type: str | None = None,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """Load a trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        model_type: Model type (if not in checkpoint)
        device: Device to load model on
        **kwargs: Additional model arguments

    Returns:
        Loaded model in eval mode
    """
    model_path = Path(model_path)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Get model config
    if isinstance(checkpoint, dict) and "model_config" in checkpoint:
        config = checkpoint["model_config"]
        model_type = config.pop("model_type", model_type)
        kwargs.update(config)
        state_dict = checkpoint["model_state_dict"]
    else:
        # Legacy format: just state dict
        state_dict = checkpoint

    # Try to load config from separate file
    config_path = model_path.parent / "model_config.json"
    if config_path.exists() and model_type is None:
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.pop("model_type", model_type)
            kwargs.update(config)

    if model_type is None:
        raise ValueError("model_type must be provided or present in checkpoint")

    # Create and load model
    model = create_model(model_type, **kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
