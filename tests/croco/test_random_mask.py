import torch

from src.annotated.mask_generator import RandomMask as AnnotatedRandomMask
from src.croco.models.masking import RandomMask as CrocoRandomMask


def test_random_mask_equivalence():
    # Test parameters
    batch_size = 4
    num_patches = 196
    mask_ratio = 0.75

    # Create dummy input
    x = torch.randn(batch_size, num_patches, 32)  # 32 is arbitrary feature dim

    # Initialize both implementations
    croco_mask = CrocoRandomMask(num_patches, mask_ratio)
    annotated_mask = AnnotatedRandomMask(num_patches, mask_ratio)

    # Generate masks using both implementations
    torch.manual_seed(42)  # Set seed for reproducibility
    croco_output = croco_mask(x)

    torch.manual_seed(42)  # Reset seed
    annotated_output = annotated_mask(x)

    # Print results
    print(f"Shapes match: {croco_output.shape == annotated_output.shape}")
    print(f"Expected shape: {(batch_size, num_patches)}")
    print(f"Actual shape: {croco_output.shape}")
    print(f"\nMasks are identical: {torch.all(croco_output == annotated_output)}")

    expected_masks = int(mask_ratio * num_patches)
    print(f"\nExpected masks per sample: {expected_masks}")
    print(f"Actual masks per sample (Croco): {croco_output.sum(dim=1)}")
    print(f"Actual masks per sample (Annotated): {annotated_output.sum(dim=1)}")
