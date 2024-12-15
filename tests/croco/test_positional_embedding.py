import numpy as np
import pytest
import torch

from src.annotated.positional_embedding import (
    construct_coordinate_grid_2d,
    get_2d_sincos_pos_embed_from_grid,
)
from src.annotated.positional_embedding import (
    get_1d_sincos_pos_embed_from_grid as annotated_get_1d_sincos_pos_embed_from_grid,
)
from src.croco.models.pos_embed import (
    get_1d_sincos_pos_embed_from_grid as croco_get_1d_sincos_pos_embed_from_grid,
)
from src.croco.models.pos_embed import (
    get_2d_sincos_pos_embed,
)


def test_1d_sincos_pos_embed():
    """Test 1D sinusoidal position embeddings match between implementations"""
    embed_dim = 128  # Must be even
    positions = np.arange(16, dtype=np.float32)  # Test with 16 positions

    # Create grid for annotated version
    grid = positions.reshape(-1)

    # Get embeddings from both implementations
    annotated_embed = annotated_get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    croco_embed = croco_get_1d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Compare outputs
    max_diff = np.max(np.abs(annotated_embed - croco_embed))
    print(f"1D embeddings max difference: {max_diff:.2e}")
    assert np.allclose(annotated_embed, croco_embed, atol=1e-6)


def test_2d_sincos_pos_embed():
    """Test 2D sinusoidal position embeddings match between implementations"""
    embed_dim = 768  # Must be even
    grid_size = 14  # 14x14 grid (typical for 224x224 image with 16x16 patches)

    # Get embeddings using CroCo's implementation
    croco_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0)

    # Get embeddings using our annotated implementation
    coordinate_grid = construct_coordinate_grid_2d(grid_size, grid_size)
    annotated_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, coordinate_grid)

    # Print shapes for debugging
    print(f"CroCo embed shape: {croco_embed.shape}")
    print(f"Annotated embed shape: {annotated_embed.shape}")

    # Compare a few sample values
    print("\nSample values comparison:")
    for i in range(0, min(croco_embed.shape[0], 3)):  # Compare first 3 positions
        print(f"\nPosition {i}:")
        print(f"CroCo: {croco_embed[i, :5]}")  # First 5 values
        print(f"Annotated: {annotated_embed[i, :5]}")  # First 5 values

    # Compare full outputs
    max_diff = np.max(np.abs(annotated_embed - croco_embed))
    mean_diff = np.mean(np.abs(annotated_embed - croco_embed))
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")

    # Test with different grid sizes
    for size in [7, 14, 28]:  # Test with different grid sizes
        croco_embed = get_2d_sincos_pos_embed(embed_dim, size, n_cls_token=0)
        coordinate_grid = construct_coordinate_grid_2d(size, size)
        annotated_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, coordinate_grid)

        assert croco_embed.shape == annotated_embed.shape, f"Shape mismatch for grid size {size}"
        assert np.allclose(croco_embed, annotated_embed, atol=1e-6), f"Values mismatch for grid size {size}"

    # Test with different embedding dimensions
    for dim in [128, 256, 512, 768]:  # Test with different embedding dimensions
        croco_embed = get_2d_sincos_pos_embed(dim, grid_size, n_cls_token=0)
        coordinate_grid = construct_coordinate_grid_2d(grid_size, grid_size)
        annotated_embed = get_2d_sincos_pos_embed_from_grid(dim, coordinate_grid)

        assert croco_embed.shape == annotated_embed.shape, f"Shape mismatch for dimension {dim}"
        assert np.allclose(croco_embed, annotated_embed, atol=1e-6), f"Values mismatch for dimension {dim}"


def test_coordinate_grid_construction():
    """Test that our coordinate grid construction matches CroCo's grid construction"""
    grid_size = 14

    # CroCo's grid construction
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    croco_grid = np.meshgrid(grid_w, grid_h)  # Note: CroCo uses grid_w first
    croco_grid = np.stack(croco_grid, axis=0)
    croco_grid = croco_grid.reshape([2, 1, grid_size, grid_size])

    # Our grid construction
    annotated_grid = construct_coordinate_grid_2d(grid_size, grid_size)

    # Compare grids
    print("\nGrid shapes:")
    print(f"CroCo grid shape: {croco_grid.shape}")
    print(f"Annotated grid shape: {annotated_grid.shape}")

    print("\nSample grid values:")
    print("CroCo grid:")
    print("Height grid (first 3x3):")
    print(croco_grid[0, 0, :3, :3])
    print("Width grid (first 3x3):")
    print(croco_grid[1, 0, :3, :3])
    print("\nAnnotated grid:")
    print("Height grid (first 3x3):")
    print(annotated_grid[0, 0, :3, :3])
    print("Width grid (first 3x3):")
    print(annotated_grid[1, 0, :3, :3])

    # Print full grid for small size
    if grid_size <= 7:
        print("\nFull grids for comparison:")
        print("CroCo height grid:")
        print(croco_grid[0, 0])
        print("\nCroCo width grid:")
        print(croco_grid[1, 0])
        print("\nAnnotated height grid:")
        print(annotated_grid[0, 0])
        print("\nAnnotated width grid:")
        print(annotated_grid[1, 0])

    # Check if grids match
    max_diff = np.max(np.abs(annotated_grid - croco_grid))
    print(f"\nMax difference between grids: {max_diff:.2e}")

    # If grids don't match exactly, analyze the differences
    if max_diff > 1e-6:
        print("\nAnalyzing grid differences:")
        # Find positions where grids differ
        diff_mask = np.abs(annotated_grid - croco_grid) > 1e-6
        diff_positions = np.where(diff_mask)
        print(f"Number of differing positions: {len(diff_positions[0])}")
        for i in range(min(5, len(diff_positions[0]))):  # Show first 5 differences
            pos = tuple(p[i] for p in diff_positions)
            print(f"Position {pos}:")
            print(f"  CroCo value: {croco_grid[pos]}")
            print(f"  Annotated value: {annotated_grid[pos]}")

    assert np.allclose(annotated_grid, croco_grid, atol=1e-6)


def test_full_position_pipeline():
    """Test the full pipeline from grid construction to final embeddings"""
    grid_size = 7  # Small size for easier debugging
    embed_dim = 128  # Small dimension for easier debugging

    # 1. Grid Construction
    print("\n1. Grid Construction")
    # CroCo's approach
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    croco_grid = np.meshgrid(grid_w, grid_h)  # Note: width first
    croco_grid = np.stack(croco_grid, axis=0)
    croco_grid = croco_grid.reshape([2, 1, grid_size, grid_size])

    # Our approach
    annotated_grid = construct_coordinate_grid_2d(grid_size, grid_size)

    print("\nGrid values at position (0,0):")
    print(f"CroCo: {croco_grid[:, 0, 0, 0]}")
    print(f"Annotated: {annotated_grid[:, 0, 0, 0]}")

    # 2. Grid Processing
    print("\n2. Grid Processing")
    # Show how the grid is reshaped/processed before embedding
    croco_grid_processed = croco_grid.reshape([2, -1])
    annotated_grid_processed = annotated_grid.reshape([2, -1])

    print("\nProcessed grid first few positions:")
    print(f"CroCo: {croco_grid_processed[:, :5]}")
    print(f"Annotated: {annotated_grid_processed[:, :5]}")

    # 3. Final Embeddings
    print("\n3. Final Embeddings")
    croco_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0)
    annotated_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, annotated_grid)

    print("\nEmbedding shapes:")
    print(f"CroCo: {croco_embed.shape}")
    print(f"Annotated: {annotated_embed.shape}")

    print("\nFirst position embeddings (first 5 values):")
    print(f"CroCo: {croco_embed[0, :5]}")
    print(f"Annotated: {annotated_embed[0, :5]}")

    # 4. Detailed Embedding Analysis
    print("\n4. Detailed Embedding Analysis")
    # Compare embeddings for specific positions
    positions_to_check = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for h, w in positions_to_check:
        pos_idx = h * grid_size + w
        print(f"\nPosition ({h}, {w}) - index {pos_idx}:")
        print(f"CroCo: {croco_embed[pos_idx, :5]}")
        print(f"Annotated: {annotated_embed[pos_idx, :5]}")

    # 5. Check for Transposition
    print("\n5. Checking for Transposition")
    # Reshape embeddings to grid format for visualization
    croco_grid_embed = croco_embed.reshape(grid_size, grid_size, -1)
    annotated_grid_embed = annotated_embed.reshape(grid_size, grid_size, -1)

    print("\nFirst value at each grid position:")
    print("CroCo grid:")
    print(croco_grid_embed[:, :, 0])
    print("\nAnnotated grid:")
    print(annotated_grid_embed[:, :, 0])

    # Also try transposed version
    print("\nTransposed annotated grid:")
    print(annotated_grid_embed.transpose(1, 0, 2)[:, :, 0])


if __name__ == "__main__":
    test_1d_sincos_pos_embed()
    test_2d_sincos_pos_embed()
    test_coordinate_grid_construction()
    test_full_position_pipeline()
