import pytest
import torch

from src.annotated.croco.patchify import patchify, unpatchify


def test_patchify_unpatchify_round_trip():
    """Test that unpatchify(patchify(imgs)) == imgs for random tensors."""
    B, C, H, W = 2, 3, 128, 128  # Square images
    patch_size = 32
    imgs = torch.randn(B, C, H, W)

    patches, num_patches_h, num_patches_w = patchify(imgs, patch_size)
    reconstructed = unpatchify(patches, patch_size, num_patches_h, num_patches_w, channels=C)

    assert torch.allclose(
        imgs, reconstructed, atol=1e-6
    ), "Round-trip patchify -> unpatchify failed for square images."


def test_patchify_unpatchify_specific_case():
    """Test patchify and unpatchify with a specific tensor where the outcome is known."""
    # Create a simple tensor where each pixel value is unique
    B, C, H, W = 1, 1, 4, 6  # Non-square image
    patch_size = 2
    imgs = torch.arange(B * C * H * W).reshape(B, C, H, W).float()

    # Expected patches:
    # For H=4, W=6, patch_size=2 -> num_patches_h=2, num_patches_w=3
    # Patches are ordered row-wise
    # Patch 1: [[0, 1], [6, 7]]
    # Patch 2: [[2, 3], [8, 9]]
    # Patch 3: [[4, 5], [10,11]]
    # Patch 4: [[12,13], [18,19]]
    # Patch 5: [[14,15], [20,21]]
    # Patch 6: [[16,17], [22,23]]
    expected_patches = torch.tensor(
        [
            [
                [0.0, 1.0, 6.0, 7.0],
                [2.0, 3.0, 8.0, 9.0],
                [4.0, 5.0, 10.0, 11.0],
                [12.0, 13.0, 18.0, 19.0],
                [14.0, 15.0, 20.0, 21.0],
                [16.0, 17.0, 22.0, 23.0],
            ]
        ]
    )  # Shape: (1, 6, 4)

    patches, num_patches_h, num_patches_w = patchify(imgs, patch_size)
    assert torch.allclose(patches, expected_patches, atol=1e-6), "Patchify specific case failed."

    # Now test unpatchify
    reconstructed = unpatchify(patches, patch_size, num_patches_h, num_patches_w, channels=C)
    assert torch.allclose(imgs, reconstructed, atol=1e-6), "Unpatchify specific case failed."


def test_patchify_unpatchify_non_square():
    """Test patchify and unpatchify with non-square images."""
    B, C, H, W = 1, 3, 64, 32  # Non-square image
    patch_size = 16
    imgs = torch.randn(B, C, H, W)

    patches, num_patches_h, num_patches_w = patchify(imgs, patch_size)
    reconstructed = unpatchify(patches, patch_size, num_patches_h, num_patches_w, channels=C)

    assert torch.allclose(imgs, reconstructed, atol=1e-6), "Patchify -> Unpatchify failed for non-square images."


def test_invalid_inputs():
    """Test that the functions handle invalid inputs gracefully."""
    # Image dimensions not divisible by patch_size
    B, C, H, W = 1, 3, 65, 33  # Not divisible by 16
    patch_size = 16
    imgs = torch.randn(B, C, H, W)

    with pytest.raises(AssertionError, match="Image dimensions must be divisible by the patch size"):
        patchify(imgs, patch_size)
