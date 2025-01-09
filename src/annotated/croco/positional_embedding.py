import numpy as np


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
