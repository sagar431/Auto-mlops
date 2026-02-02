"""Text classification models."""

import torch
import torch.nn as nn

try:
    from transformers import DistilBertModel

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    DistilBertModel = None


class TextCNN(nn.Module):
    """CNN for text classification.

    Architecture:
    - Embedding layer
    - Multiple parallel convolutions with different kernel sizes
    - Max pooling over time
    - Fully connected layer with dropout

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of word embeddings.
        num_classes: Number of output classes.
        num_filters: Number of filters for each kernel size.
        kernel_sizes: List of kernel sizes for parallel convolutions.
        dropout: Dropout probability.
        padding_idx: Index of padding token.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        num_classes: int = 2,
        num_filters: int = 100,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes

        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)

        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # Transpose for conv1d: (batch_size, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)

        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))
            # Max pool over time
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)

        # Concatenate all pooled features
        cat = torch.cat(conv_outputs, dim=1)

        # Dropout and final FC
        out = self.dropout(cat)
        out = self.fc(out)

        return out


class DistilBERTClassifier(nn.Module):
    """DistilBERT-based classifier for text classification.

    Architecture:
    - Pretrained DistilBERT encoder
    - Classification head with dropout

    Args:
        num_classes: Number of output classes.
        pretrained_model_name: Name of the pretrained DistilBERT model.
        dropout: Dropout probability for the classifier head.
        freeze_encoder: Whether to freeze the DistilBERT encoder weights.

    Raises:
        ImportError: If transformers library is not available.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_model_name: str = "distilbert-base-uncased",
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for DistilBERTClassifier. "
                "Install it with: pip install transformers"
            )

        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder

        # Load pretrained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained(pretrained_model_name)

        # Get hidden size from config
        self.hidden_size = self.distilbert.config.hidden_size

        # Optionally freeze encoder weights
        if freeze_encoder:
            for param in self.distilbert.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Input token IDs, shape (batch_size, seq_len).
            attention_mask: Attention mask, shape (batch_size, seq_len).
                If None, no masking is applied.

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled)

        return logits

    def unfreeze_encoder(self, num_layers: int | None = None):
        """Unfreeze encoder layers for fine-tuning.

        Args:
            num_layers: Number of transformer layers to unfreeze from the top.
                If None, unfreezes all layers.
        """
        if num_layers is None:
            for param in self.distilbert.parameters():
                param.requires_grad = True
        else:
            # Unfreeze embeddings
            for param in self.distilbert.embeddings.parameters():
                param.requires_grad = False

            # Get total number of transformer layers
            total_layers = len(self.distilbert.transformer.layer)

            # Freeze all layers first
            for layer in self.distilbert.transformer.layer:
                for param in layer.parameters():
                    param.requires_grad = False

            # Unfreeze top num_layers
            for layer in self.distilbert.transformer.layer[total_layers - num_layers :]:
                for param in layer.parameters():
                    param.requires_grad = True


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for text classification.

    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Attention mechanism
    - Fully connected layer with dropout

    Args:
        vocab_size: Size of the vocabulary.
        embedding_dim: Dimension of word embeddings.
        hidden_dim: Hidden dimension of LSTM.
        num_classes: Number of output classes.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability.
        bidirectional: Whether to use bidirectional LSTM.
        padding_idx: Index of padding token.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Attention
        self.attention = nn.Linear(lstm_output_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)

        # Embedding: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # LSTM: (batch_size, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(embedded)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        # Dropout and FC
        out = self.dropout(context)
        out = self.fc(out)

        return out


def create_model(
    model_type: str = "textcnn",
    vocab_size: int = 30000,
    embedding_dim: int = 128,
    num_classes: int = 2,
    dropout: float = 0.5,
    **kwargs,
) -> nn.Module:
    """Create a text classification model.

    Args:
        model_type: Type of model ('textcnn', 'lstm', or 'distilbert').
        vocab_size: Size of the vocabulary (not used for distilbert).
        embedding_dim: Dimension of word embeddings (not used for distilbert).
        num_classes: Number of output classes.
        dropout: Dropout probability.
        **kwargs: Additional model-specific arguments.
            For distilbert:
                - pretrained_model_name: Name of pretrained model (default: 'distilbert-base-uncased')
                - freeze_encoder: Whether to freeze encoder weights (default: False)

    Returns:
        Model instance.
    """
    if model_type == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_filters=kwargs.get("num_filters", 100),
            kernel_sizes=kwargs.get("kernel_sizes", [3, 4, 5]),
        )
    elif model_type == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=kwargs.get("hidden_dim", 256),
            num_layers=kwargs.get("num_layers", 2),
            bidirectional=kwargs.get("bidirectional", True),
        )
    elif model_type == "distilbert":
        return DistilBERTClassifier(
            num_classes=num_classes,
            dropout=dropout,
            pretrained_model_name=kwargs.get("pretrained_model_name", "distilbert-base-uncased"),
            freeze_encoder=kwargs.get("freeze_encoder", False),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(
    path: str,
    model_type: str = "textcnn",
    vocab_size: int = 30000,
    num_classes: int = 2,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """Load a trained model from a checkpoint file.

    Args:
        path: Path to the model checkpoint file.
        model_type: Type of model ('textcnn', 'lstm', or 'distilbert').
        vocab_size: Size of the vocabulary (not used for distilbert).
        num_classes: Number of output classes.
        device: Device to load the model on.
        **kwargs: Additional model-specific arguments.

    Returns:
        Loaded model in eval mode.
    """
    model = create_model(
        model_type=model_type,
        vocab_size=vocab_size,
        num_classes=num_classes,
        **kwargs,
    )

    # For DistilBERT, weights_only=True may cause issues with HuggingFace model
    # architecture, so we use weights_only=False but only load state dict
    if model_type == "distilbert":
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    else:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    model.to(device)
    model.eval()
    return model
