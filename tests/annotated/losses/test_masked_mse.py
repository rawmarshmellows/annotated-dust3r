import pytest
import torch

from src.annotated.losses.masked_mse import AnnotatedMaskedMSE, MaskedMSE


@pytest.mark.parametrize("norm_pix_loss", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_masked_mse(norm_pix_loss, masked):
    """
    Test that AnnotatedMaskedMSE produces the same output as MaskedMSE
    for various configurations and inputs.

    Debug:
    - Tests both classes with different norm_pix_loss and masked settings
    - Verifies outputs are identical for the same inputs
    - Tests with different shaped tensors and mask patterns
    """
    # Debug: Initialize both loss functions with the same configuration
    original_loss_fn = MaskedMSE(norm_pix_loss=norm_pix_loss, masked=masked)
    annotated_loss_fn = AnnotatedMaskedMSE(norm_pix_loss=norm_pix_loss, masked=masked)

    # Test with different input shapes
    shapes = [
        (2, 4, 16),  # [batch_size, num_patches, patch_dim]
        (4, 8, 32),
    ]

    for shape in shapes:
        # Debug: Create test tensors
        batch_size, num_patches, patch_dim = shape
        pred = torch.randn(batch_size, num_patches, patch_dim)
        target = torch.randn(batch_size, num_patches, patch_dim)

        # Test with different mask patterns
        masks = [
            torch.ones(batch_size, num_patches),  # All patches masked
            torch.randint(0, 2, (batch_size, num_patches)).float(),  # Random mask
        ]

        # Add zero mask test case only when masked=False to avoid division by zero
        if not masked:
            masks.append(torch.zeros(batch_size, num_patches))

        for mask in masks:
            # Skip cases where mask.sum() == 0 and masked=True (would cause division by zero)
            if masked and mask.sum() == 0:
                continue

            # Debug: Compute losses from both implementations
            original_loss = original_loss_fn(pred, mask, target)
            annotated_loss = annotated_loss_fn(pred, mask, target)

            # Debug: Verify outputs are identical
            assert torch.allclose(original_loss, annotated_loss, rtol=1e-5, atol=1e-5), (
                f"Loss mismatch with norm_pix_loss={norm_pix_loss}, masked={masked}, "
                f"shape={shape}, mask.sum()={mask.sum()}"
            )
