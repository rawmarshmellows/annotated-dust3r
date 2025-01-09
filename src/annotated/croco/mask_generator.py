import torch
import torch.nn as nn


class RandomMask(nn.Module):
    """
    Generates random binary masks for input tensors.

    This module creates masks by randomly selecting a subset of positions to be masked
    based on the specified mask ratio.

    Args:
        num_patches (int): Total number of patches in the input
        mask_ratio (float): Ratio of patches to be masked, between 0 and 1

    Returns:
        torch.Tensor: Binary mask tensor where True indicates masked positions
    """

    def __init__(self, num_patches: int, mask_ratio: float) -> None:
        super().__init__()
        if not 0 <= mask_ratio <= 1:
            raise ValueError("Mask ratio must be between 0 and 1")
        if num_patches <= 0:
            raise ValueError("Number of patches must be positive")

        self.num_patches = num_patches
        self.num_masks = int(mask_ratio * self.num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate random binary masks for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to generate masks for

        Returns:
            torch.Tensor: Binary mask tensor where True indicates masked positions
        """
        # Get batch size from input tensor
        batch_size = x.size(0)

        # Generate random values between 0-1 for each patch position
        # Shape: [batch_size, num_patches]
        random_noise = torch.rand(batch_size, self.num_patches, device=x.device)

        # Sort the random values to get indices
        # Lower values will be masked (first self.num_masks positions)
        # Shape: [batch_size, num_patches]
        sorted_positions = torch.argsort(random_noise, dim=1)

        # Create binary mask by comparing position indices with num_masks
        # True for indices < num_masks (masked positions)
        # False for indices >= num_masks (unmasked positions)
        # Shape: [batch_size, num_patches]
        mask = sorted_positions < self.num_masks
        return mask
