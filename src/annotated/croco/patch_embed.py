from functools import partial
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn

from .utils import to_2tuple


class PositionGetter:
    """
    Generates and caches patch position encodings for image patches.

    This class creates position encodings for image patches in a grid layout,
    caching them for reuse to improve efficiency when the same grid dimensions
    are requested multiple times.
    """

    def __init__(self):
        """Initialize an empty cache for position encodings."""
        self.cache_positions: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, num_patches_h: int, num_patches_w: int, device: torch.device) -> torch.Tensor:
        """
        Generate or retrieve cached position encodings for the specified patch grid.

        Parameters:
            batch_size (int): Number of samples in the batch
            num_patches_h (int): Number of patches in height dimension
            num_patches_w (int): Number of patches in width dimension
            device (torch.device): Device to place the tensors on

        Returns:
            torch.Tensor: Position encodings of shape (batch_size, num_patches_h * num_patches_w, 2)
        """
        if (num_patches_h, num_patches_w) not in self.cache_positions:
            self.cache_positions[num_patches_h, num_patches_w] = self._generate_patch_positions_with_dimension(
                num_patches_h, num_patches_w, device
            )
        return self._expand_patch_positions_with_batch_size(
            self.cache_positions[num_patches_h, num_patches_w], batch_size
        )

    def _generate_patch_positions_with_dimension(
        self, num_patches_h: int, num_patches_w: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate position encodings for a single sample.

        Parameters:
            num_patches_h (int): Number of patches in height dimension
            num_patches_w (int): Number of patches in width dimension
            device (torch.device): Device to place the tensors on

        Returns:
            torch.Tensor: Position encodings of shape (num_patches_h * num_patches_w, 2)
        """
        y = torch.arange(num_patches_h, device=device)
        x = torch.arange(num_patches_w, device=device)
        positions_for_patch = torch.cartesian_prod(y, x)
        return positions_for_patch

    def _expand_patch_positions_with_batch_size(self, patch_positions: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand position encodings to match batch size.

        Parameters:
            patch_positions (torch.Tensor): Position encodings for a single sample
            batch_size (int): Number of samples in the batch

        Returns:
            torch.Tensor: Position encodings of shape (batch_size, num_patches_h * num_patches_w, 2)
        """
        num_positions = patch_positions.size(0)
        unsqueezed_patch_positions = patch_positions.unsqueeze(0)
        expanded_patch_positions_with_batch_size = unsqueezed_patch_positions.expand(batch_size, num_positions, 2)
        return expanded_patch_positions_with_batch_size


class PatchEmbed(nn.Module):
    """
    Embeds image patches into a specified embedding dimension.

    This module splits an image into non-overlapping patches, projects each patch
    into a high-dimensional space using a convolutional layer, applies normalization,
    and provides positional encodings for each patch.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        flatten: bool = True,
    ):
        """
        Initialize the PatchEmbed module.

        Parameters:
            img_size (int): Size of the input image (assumed square)
            patch_size (int): Size of each patch (assumed square)
            in_channels (int): Number of input image channels
            embed_dim (int): Dimension of the patch embeddings
            norm_layer (nn.Module, optional): Normalization layer constructor
            flatten (bool): If True, flatten spatial dimensions after projection
        """
        super().__init__()

        # Check compatibility of norm_layer and flatten settings
        if not flatten and norm_layer is not None and norm_layer != nn.Identity:
            raise ValueError(
                "LayerNorm cannot be used with flatten=False. "
                "When flatten=False, the output shape is (B, embed_dim, H, W) "
                "which is incompatible with LayerNorm's expected shape (B, H*W, embed_dim)."
            )

        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)

        self.num_patches_h = self.img_size[0] // self.patch_size[0]
        self.num_patches_w = self.img_size[1] // self.patch_size[1]
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.flatten = flatten

        # Project patches to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.position_getter = PositionGetter()

        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input images through the patch embedding module.

        Parameters:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Patch embeddings of shape:
                    - If flatten=True: (B, num_patches, embed_dim)
                    - If flatten=False: (B, embed_dim, num_patches_h, num_patches_w)
                - Position encodings of shape (B, num_patches, 2)
        """
        B, C, H, W = x.shape

        assert (
            H == self.img_size[0]
        ), f"Input image height ({H}) doesn't match model's expected height ({self.img_size[0]})."
        assert (
            W == self.img_size[1]
        ), f"Input image width ({W}) doesn't match model's expected width ({self.img_size[1]})."

        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)

        pos_encodings = self.position_getter(
            batch_size=B, num_patches_h=self.num_patches_h, num_patches_w=self.num_patches_w, device=x.device
        )  # (B, num_patches, 2)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)

        return x, pos_encodings
