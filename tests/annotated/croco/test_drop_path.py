import torch

from src.annotated.croco.drop_path import DropPath as DropPathAnnotated
from src.croco.models.blocks import DropPath


def test_drop_path_equivalence():
    """Test that DropPathAnnotated produces equivalent results to original DropPath."""

    # Test parameters
    batch_size = 4
    channels = 8
    height = width = 16
    drop_prob = 0.2

    # Create test input
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(batch_size, channels, height, width)

    # Initialize both implementations
    original_layer = DropPath(drop_prob=drop_prob)
    annotated_layer = DropPathAnnotated(drop_prob=drop_prob)

    # Set to eval mode first
    original_layer.eval()
    annotated_layer.eval()

    # Test 1: Both should pass through unchanged in eval mode
    out_original = original_layer(x)
    out_annotated = annotated_layer(x)

    assert torch.allclose(x, out_original), "Original DropPath modified input in eval mode"
    assert torch.allclose(x, out_annotated), "DropPathAnnotated modified input in eval mode"
    print("✓ Both implementations pass through unchanged in eval mode")

    # Test 2: Training mode with same random seed
    original_layer.train()
    annotated_layer.train()

    torch.manual_seed(42)
    out_original = original_layer(x)

    torch.manual_seed(42)
    out_annotated = annotated_layer(x)

    assert torch.allclose(out_original, out_annotated), "Outputs differ in training mode"
    print("✓ Both implementations produce identical outputs in training mode")

    # Test 3: Check drop probability effect
    n_samples = 1000
    torch.manual_seed(42)

    # Count how many elements are dropped (zeroed out)
    dropped_count = 0
    for _ in range(n_samples):
        out = annotated_layer(torch.ones(1, 1, 1, 1))
        if out.item() == 0:
            dropped_count += 1

    actual_drop_prob = dropped_count / n_samples
    assert (
        abs(actual_drop_prob - drop_prob) < 0.05
    ), f"Drop probability {actual_drop_prob:.3f} differs significantly from expected {drop_prob}"
    print(f"✓ Drop probability {actual_drop_prob:.3f} matches expected {drop_prob}")
