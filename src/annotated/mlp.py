import torch.nn as nn

from src.annotated.utils import to_2tuple


class Mlp(nn.Module):
    """Multi-Layer Perceptron (MLP) module with two linear layers.

    This MLP implementation is commonly used in Vision Transformer, MLP-Mixer and related networks.
    It consists of two linear layers with an activation function and dropout in between.
    Each layer can have its own bias and dropout settings.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features (defaults to in_features)
        out_features: Number of output features (defaults to in_features)
        act_layer: Activation layer class (defaults to GELU)
        bias: Whether to use bias in linear layers. Can be:
            - A single bool: Same setting for both layers
            - A tuple of bools: Different settings for each layer
        drop: Dropout probability. Can be:
            - A single float: Same dropout rate for both layers
            - A tuple of floats: Different dropout rates for each layer
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()

        # Default hidden and output dimensions to input dimension if not specified
        self.in_features = in_features
        self.hidden_features = hidden_features or in_features
        self.out_features = out_features or in_features

        # Convert bias and dropout params to tuples for first/second layer
        # This allows specifying different values for each layer
        bias_tuple = to_2tuple(bias)  # e.g., True -> (True, True) or (True, False) -> (True, False)
        drop_tuple = to_2tuple(drop)  # e.g., 0.1 -> (0.1, 0.1) or (0.1, 0.2) -> (0.1, 0.2)

        # First linear layer + activation + dropout
        self.fc1 = nn.Linear(self.in_features, self.hidden_features, bias=bias_tuple[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_tuple[0])

        # Second linear layer + dropout
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=bias_tuple[1])
        self.drop2 = nn.Dropout(drop_tuple[1])

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through both MLP layers
        """
        # First layer
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        # Second layer
        x = self.fc2(x)
        x = self.drop2(x)

        return x
