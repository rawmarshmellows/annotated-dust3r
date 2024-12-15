import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample with detailed annotations.

    This class implements stochastic depth regularization by randomly dropping paths
    during training. It helps prevent overfitting in deep networks by introducing
    controlled randomness.

    Args:
        drop_prob: Probability of dropping a path (0.0 to 1.0)
        scale_by_keep: Whether to scale outputs by the keep probability

    Example:
        >>> layer = DropPathAnnotated(drop_prob=0.2)
        >>> x = torch.randn(32, 64, 32, 32)  # [batch, channels, height, width]
        >>> out = layer(x)  # During training, ~20% paths will be dropped
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        if not 0 <= drop_prob <= 1:
            raise ValueError(f"drop_prob must be between 0 and 1, got {drop_prob}")

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with stochastic depth.

        Args:
            x: Input tensor of any shape

        Returns:
            Tensor with same shape as input, with paths randomly dropped during training
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Create shape for broadcasting: (batch_size, 1, 1, ..., 1)
        # The number of 1s equals the number of non-batch dimensions
        batch_size = x.shape[0]
        num_other_dims = x.ndim - 1  # Subtract 1 for batch dimension
        shape = (batch_size,) + (1,) * num_other_dims

        # Generate random mask
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor

    def extra_repr(self) -> str:
        """String representation of the module parameters."""
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
