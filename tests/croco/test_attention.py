import pytest
import torch
import torch.nn as nn

from src.annotated.attention import Attention as AnnotatedAttention
from src.annotated.attention import CrossAttention as AnnotatedCrossAttention
from src.croco.models.blocks import Attention as CrocoAttention
from src.croco.models.blocks import CrossAttention as CrocoCrossAttention


def test_cross_attention_equivalence():
    """Test equivalence between Annotated and CroCo CrossAttention implementations"""
    # Test parameters
    batch_size = 2
    seq_len = 196  # typical for decoder input length
    context_len = 77  # typical for encoder output length
    embed_dim = 768
    num_heads = 12

    # Create random inputs
    query_tokens = torch.randn(batch_size, seq_len, embed_dim)  # Decoder queries
    key_tokens = torch.randn(batch_size, context_len, embed_dim)  # Encoder keys
    value_tokens = torch.randn(batch_size, context_len, embed_dim)  # Encoder values
    query_positions = torch.randint(0, 14, (batch_size, seq_len, 2))  # Query positions
    key_positions = torch.randint(0, 14, (batch_size, context_len, 2))  # Key positions

    # Initialize both implementations with same parameters
    annotated_cross = AnnotatedCrossAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    )

    croco_cross = CrocoCrossAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # Set both to eval mode
    annotated_cross.eval()
    croco_cross.eval()

    # Initialize weights identically
    with torch.no_grad():
        # Get state dicts
        croco_state = croco_cross.state_dict()
        annotated_state = annotated_cross.state_dict()

        print("\nCroCo Cross Attention state dict:")
        for k, v in croco_state.items():
            print(f"{k}: {v.shape}")

        print("\nAnnotated Cross Attention state dict:")
        for k, v in annotated_state.items():
            print(f"{k}: {v.shape}")

        # Map weights between models
        mapping = {
            "projq.weight": "query_projection.weight",
            "projq.bias": "query_projection.bias",
            "projk.weight": "key_projection.weight",
            "projk.bias": "key_projection.bias",
            "projv.weight": "value_projection.weight",
            "projv.bias": "value_projection.bias",
            "proj.weight": "output_projection.weight",
            "proj.bias": "output_projection.bias",
        }

        # Copy weights using mapping
        new_state_dict = {}
        for croco_key, annotated_key in mapping.items():
            if croco_key in croco_state:
                new_state_dict[annotated_key] = croco_state[croco_key]
            else:
                print(f"Warning: {croco_key} not found in CroCo state dict")

        # Check for any keys in annotated model that weren't mapped
        for k in annotated_state.keys():
            if k not in new_state_dict:
                print(f"Warning: {k} from Annotated model not mapped")

        annotated_cross.load_state_dict(new_state_dict)

    # Forward pass through both implementations
    with torch.no_grad():
        # Pass separate query, key, value tokens and positions to cross attention
        annotated_output = annotated_cross(
            query_tokens=query_tokens,
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
        )

        croco_output = croco_cross(
            query_tokens,  # x in CroCo implementation
            key_tokens,  # context in CroCo implementation
            value_tokens,
            query_positions,
            key_positions,
        )

    # Test outputs are equal within numerical precision
    assert torch.allclose(annotated_output, croco_output, rtol=1e-5, atol=1e-5)


def test_attention_equivalence():
    """Test equivalence between Annotated and CroCo Attention implementations"""

    # Test parameters
    batch_size = 2
    seq_len = 196  # typical for 224x224 image with 16x16 patches
    embed_dim = 768
    num_heads = 12

    # Create random input
    x = torch.randn(batch_size, seq_len, embed_dim)
    pos = torch.randint(0, 14, (batch_size, seq_len, 2))  # Random positions

    # Initialize both implementations with same parameters
    annotated_attn = AnnotatedAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    )

    croco_attn = CrocoAttention(
        dim=embed_dim,
        num_heads=num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # Set both to eval mode
    annotated_attn.eval()
    croco_attn.eval()

    # Initialize weights identically
    with torch.no_grad():
        # Handle different naming conventions
        state_dict = croco_attn.state_dict()
        new_state_dict = {}

        # Map CroCo's qkv to our query_key_value
        if "qkv.weight" in state_dict:
            new_state_dict["query_key_value.weight"] = state_dict["qkv.weight"]
        if "qkv.bias" in state_dict:
            new_state_dict["query_key_value.bias"] = state_dict["qkv.bias"]

        # Copy projection weights (same naming)
        if "proj.weight" in state_dict:
            new_state_dict["proj.weight"] = state_dict["proj.weight"]
        if "proj.bias" in state_dict:
            new_state_dict["proj.bias"] = state_dict["proj.bias"]

        # Load the mapped state dict
        annotated_attn.load_state_dict(new_state_dict)

    # Forward pass
    with torch.no_grad():
        annotated_output = annotated_attn(x, pos)
        croco_output = croco_attn(x, pos)

    # Compare outputs
    max_diff = torch.max(torch.abs(annotated_output - croco_output)).item()
    mean_diff = torch.mean(torch.abs(annotated_output - croco_output)).item()

    print("\nAttention Implementation Comparison:")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Shapes match: {annotated_output.shape == croco_output.shape}")
    print(f"Output shape: {annotated_output.shape}")

    # Test with different input patterns
    test_cases = {
        "Uniform": torch.ones_like(x),
        "Random": torch.randn_like(x),
        "Zero": torch.zeros_like(x),
        "Large Values": torch.ones_like(x) * 100,
    }

    for name, test_input in test_cases.items():
        with torch.no_grad():
            annotated_out = annotated_attn(test_input, pos)
            croco_out = croco_attn(test_input, pos)

            max_diff = torch.max(torch.abs(annotated_out - croco_out)).item()
            print(f"\n{name} Input Test:")
            print(f"Max difference: {max_diff:.2e}")
            print(f"Outputs match: {torch.allclose(annotated_out, croco_out, atol=1e-6)}")
