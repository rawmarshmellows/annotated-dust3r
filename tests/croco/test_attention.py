import pytest
import torch
import torch.nn as nn
from test_utils import load_and_validate_state_dict_with_mapping

from src.annotated.croco.attention import Attention as AnnotatedAttention
from src.annotated.croco.attention import CrossAttention as AnnotatedCrossAttention
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
        # Map weights between models
        key_mapping = {
            "projq.weight": "query_projection.weight",
            "projq.bias": "query_projection.bias",
            "projk.weight": "key_projection.weight",
            "projk.bias": "key_projection.bias",
            "projv.weight": "value_projection.weight",
            "projv.bias": "value_projection.bias",
            "proj.weight": "output_projection.weight",
            "proj.bias": "output_projection.bias",
        }

        load_and_validate_state_dict_with_mapping(annotated_cross, croco_cross, key_mapping)

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
        # Map weights between models
        key_mapping = {
            "qkv.weight": "query_key_value.weight",
            "qkv.bias": "query_key_value.bias",
            "proj.weight": "proj.weight",
            "proj.bias": "proj.bias",
        }

        load_and_validate_state_dict_with_mapping(annotated_attn, croco_attn, key_mapping)

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
            assert torch.allclose(annotated_out, croco_out, atol=1e-6)
