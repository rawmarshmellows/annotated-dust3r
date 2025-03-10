from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import nn


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, coordinate_grid: np.ndarray) -> np.ndarray:
    """
    Generate sinusoidal positional embeddings for 1D positions.

    This function creates embeddings where each dimension corresponds to a sinusoid
    of a different frequency. The first half uses sine, the second half uses cosine.

    Args:
        embed_dim: Dimension of the output embeddings (must be even)
        positions: Array of positions to encode, will be flattened

    Returns:
        np.ndarray: Position embeddings with shape [len(flattened_positions), embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    # Generate frequency bands for the sinusoidal embeddings
    # Each position will be encoded with sinusoids of these frequencies
    frequency_bands = np.arange(embed_dim // 2, dtype=float)
    frequency_bands /= embed_dim / 2.0
    frequency_bands = 1.0 / 10000**frequency_bands  # Shape: (D/2,)

    # Flatten input positions
    flattened_positions = coordinate_grid.reshape(-1)  # Shape: (M,)

    # Compute position-frequency products
    # This creates a matrix where each row corresponds to a position and
    # each column corresponds to that position multiplied by a frequency
    phase_matrix = np.einsum("m,d->md", flattened_positions, frequency_bands)  # Shape: (M, D/2)

    # Generate sine and cosine embeddings
    sin_embeddings = np.sin(phase_matrix)  # Shape: (M, D/2)
    cos_embeddings = np.cos(phase_matrix)  # Shape: (M, D/2)

    # Combine sine and cosine embeddings
    # Shape: (M, D) where D = embed_dim
    combined_embeddings = np.concatenate([sin_embeddings, cos_embeddings], axis=1)

    return combined_embeddings


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, coordinate_grid: np.ndarray) -> np.ndarray:
    """
    Convert a 2D coordinate grid into sinusoidal positional embeddings.

    Args:
        embed_dim: Total embedding dimension (must be even)
        coordinate_grid: Grid of coordinates with shape [2, 1, H, W]

    Returns:
        np.ndarray: Positional embeddings with shape [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even"

    # Split dimension evenly between height and width coordinates
    dim_per_coordinate = embed_dim // 2

    # Generate embeddings separately for height and width coordinates
    height_embeddings = get_1d_sincos_pos_embed_from_grid(
        dim_per_coordinate, coordinate_grid=coordinate_grid[0]
    )  # Shape: (H*W, D/2)
    width_embeddings = get_1d_sincos_pos_embed_from_grid(
        dim_per_coordinate, coordinate_grid=coordinate_grid[1]
    )  # Shape: (H*W, D/2)

    # Combine height and width embeddings
    # Shape: (H*W, D) where D = dim_per_coordinate * 2 = embed_dim
    combined_embeddings = np.concatenate([height_embeddings, width_embeddings], axis=1)

    return combined_embeddings


def construct_coordinate_grid_2d(grid_size_x, grid_size_y):
    coordinate_grid_x = np.arange(grid_size_x)
    coordinate_grid_y = np.arange(grid_size_y)
    grid_h, grid_w = np.meshgrid(coordinate_grid_y, coordinate_grid_x)
    coordinate_grid_2d = np.stack([grid_h, grid_w], axis=0)
    coordinate_grid_2d = coordinate_grid_2d.reshape([2, 1, grid_size_y, grid_size_x])
    return coordinate_grid_2d


class RoPE2D(torch.nn.Module):
    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 2 (y and x position of each token)
        output:
            * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
        D = tokens.size(3) // 2
        assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        # split features into two along the feature dimension, and apply rope1d on each half
        y, x = tokens.chunk(2, dim=-1)
        y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
        x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
        tokens = torch.cat((y, x), dim=-1)
        return tokens


class PositionalEmbedderFactory:
    @classmethod
    def create_sincos2d_positional_embedder(
        cls, register_named_buffer_fn: Callable, get_buffer_fn: Callable, embed_dim: int, num_patches: int
    ):
        return SinCos2dPositionalEmbedder(register_named_buffer_fn, get_buffer_fn, embed_dim, num_patches)

    @classmethod
    def create_rope_positional_embedder(cls, freq: float, F0: float):
        return RoPE100PositionalEmbedder(freq, F0)


class PositionalEmbedder(ABC):
    @abstractmethod
    def embed(self, x, patch_positions):
        pass


class SinCos2dPositionalEmbedder(PositionalEmbedder):
    def __init__(self, register_named_buffer_fn: Callable, get_buffer_fn: Callable, embed_dim: int, num_patches: int):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        coordinate_grid_2d = construct_coordinate_grid_2d(
            grid_size_x=int(self.num_patches**0.5), grid_size_y=int(self.num_patches**0.5)
        )
        enc_pos_embed = get_2d_sincos_pos_embed_from_grid(self.embed_dim, coordinate_grid_2d)
        register_named_buffer_fn(tensor=torch.from_numpy(enc_pos_embed.astype(np.float32)).float())
        self.enc_pos_embed = get_buffer_fn()

    def embed(self, x, patch_positions):
        return x + self.enc_pos_embed


class RoPE100PositionalEmbedder(PositionalEmbedder):
    def __init__(self, freq: float, F0: float):
        super().__init__()
        self.pos_embed = RoPE2D(freq, F0)

    def embed(self, x, patch_positions):
        return self.pos_embed(x, patch_positions)
