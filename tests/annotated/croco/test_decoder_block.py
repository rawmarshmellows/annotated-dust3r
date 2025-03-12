import pytest
import torch
import torch.nn as nn
from test_utils import load_and_validate_state_dict_with_mapping

from src.annotated.croco.decoder_block import TransformerDecoderBlock as AnnotatedDecoderBlock
from src.annotated.croco.decoder_block import TransformerDecoderBlockV2 as AnnotatedDecoderBlockV2
from src.croco.models.blocks import DecoderBlock as CroCoDecoderBlock


def test_decoder_block_v2_equivalence():
    """Test equivalence between Annotated and CroCo DecoderBlock implementations"""

    # Test parameters
    batch_size = 2
    seq_len = 196
    context_len = 50  # Added context length for encoder output
    embed_dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True
    drop_rate = 0.0
    attn_drop_rate = 0.0

    # Create random inputs with fixed seed for reproducibility
    torch.manual_seed(42)
    query_input = torch.randn(batch_size, seq_len, embed_dim)
    memory_input = torch.randn(batch_size, context_len, embed_dim)
    query_pos = torch.randn(batch_size, seq_len, embed_dim)
    memory_pos = torch.randn(batch_size, context_len, embed_dim)

    # Initialize both implementations with same parameters
    annotated_block = AnnotatedDecoderBlockV2(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )

    croco_block = CroCoDecoderBlock(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
    )

    # Initialize weights identically
    key_mapping = {
        # Self attention layers
        "attn.qkv.weight": "self_attend_query_key_value_projection.weight",
        "attn.qkv.bias": "self_attend_query_key_value_projection.bias",
        "attn.proj.weight": "self_attn.output_projection.weight",
        "attn.proj.bias": "self_attn.output_projection.bias",
        # Cross attention layers
        "cross_attn.projq.weight": "cross_attn_query_projection.weight",
        "cross_attn.projq.bias": "cross_attn_query_projection.bias",
        "cross_attn.projk.weight": "cross_attn_key_projection.weight",
        "cross_attn.projk.bias": "cross_attn_key_projection.bias",
        "cross_attn.projv.weight": "cross_attn_value_projection.weight",
        "cross_attn.projv.bias": "cross_attn_value_projection.bias",
        "cross_attn.proj.weight": "cross_attn.output_projection.weight",
        "cross_attn.proj.bias": "cross_attn.output_projection.bias",
        # Norm layers
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "norm3.weight",
        "norm3.bias": "norm3.bias",
        # MLP layers
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
        # Norm layers
        "norm_y.weight": "norm_y.weight",
        "norm_y.bias": "norm_y.bias",
    }

    load_and_validate_state_dict_with_mapping(annotated_block, croco_block, key_mapping)

    # Test with different input patterns
    test_cases = {
        "Uniform": (
            torch.ones_like(query_input),
            torch.ones_like(memory_input),
            torch.ones_like(query_pos),
            torch.ones_like(memory_pos),
        ),
        "Random": (
            torch.randn_like(query_input),
            torch.randn_like(memory_input),
            torch.randn_like(query_pos),
            torch.randn_like(memory_pos),
        ),
        "Zero": (
            torch.zeros_like(query_input),
            torch.zeros_like(memory_input),
            torch.zeros_like(query_pos),
            torch.zeros_like(memory_pos),
        ),
        "Large Values": (
            torch.ones_like(query_input) * 100,
            torch.ones_like(memory_input) * 100,
            torch.ones_like(query_pos) * 100,
            torch.ones_like(memory_pos) * 100,
        ),
    }

    for name, (test_query, test_memory, test_query_pos, test_memory_pos) in test_cases.items():
        with torch.no_grad():
            torch.manual_seed(42)
            croco_query_out, croco_memory_out = croco_block(test_query, test_memory, test_query_pos, test_memory_pos)
            torch.manual_seed(42)
            annotated_query_out, annotated_memory_out = annotated_block(
                test_query, test_memory, test_query_pos, test_memory_pos
            )

            max_diff = torch.max(torch.abs(annotated_query_out - croco_query_out)).item()
            print(f"\n{name} Input Test:")
            print(f"Max difference: {max_diff:.2e}")
            assert max_diff < 1e-10, f"Maximum difference exceeds tolerance for {name} input"

            print(f"Outputs match: {torch.allclose(croco_query_out, annotated_query_out, atol=1e-6)}")
            assert torch.allclose(
                croco_query_out, annotated_query_out, atol=1e-6
            ), f"Outputs do not match for {name} input"

            max_diff_memory = torch.max(torch.abs(annotated_memory_out - croco_memory_out)).item()
            print(f"Max memory difference: {max_diff_memory:.2e}")
            assert max_diff_memory < 1e-10, f"Maximum memory difference exceeds tolerance for {name} input"

            print(f"Memory outputs match: {torch.allclose(croco_memory_out, annotated_memory_out, atol=1e-6)}")
            assert torch.allclose(
                croco_memory_out, annotated_memory_out, atol=1e-10
            ), f"Memory outputs do not match for {name} input"


def test_decoder_block_equivalence():
    """Test equivalence between Annotated and CroCo DecoderBlock implementations"""

    # Test parameters
    batch_size = 2
    seq_len = 196
    context_len = 50  # Added context length for encoder output
    embed_dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True
    drop_rate = 0.0
    attn_drop_rate = 0.0

    # Create random inputs with fixed seed for reproducibility
    torch.manual_seed(42)
    query_input = torch.randn(batch_size, seq_len, embed_dim)
    memory_input = torch.randn(batch_size, context_len, embed_dim)
    query_pos = torch.randn(batch_size, seq_len, embed_dim)
    memory_pos = torch.randn(batch_size, context_len, embed_dim)

    # Initialize both implementations with same parameters
    annotated_block = AnnotatedDecoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        proj_drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
    )

    croco_block = CroCoDecoderBlock(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop=drop_rate,
        attn_drop=attn_drop_rate,
    )

    # Initialize weights identically
    key_mapping = {
        # Self attention layers
        "attn.qkv.weight": "attn.query_key_value.weight",
        "attn.qkv.bias": "attn.query_key_value.bias",
        "attn.proj.weight": "attn.proj.weight",
        "attn.proj.bias": "attn.proj.bias",
        # Cross attention layers
        "cross_attn.projq.weight": "cross_attn.query_projection.weight",
        "cross_attn.projq.bias": "cross_attn.query_projection.bias",
        "cross_attn.projk.weight": "cross_attn.key_projection.weight",
        "cross_attn.projk.bias": "cross_attn.key_projection.bias",
        "cross_attn.projv.weight": "cross_attn.value_projection.weight",
        "cross_attn.projv.bias": "cross_attn.value_projection.bias",
        "cross_attn.proj.weight": "cross_attn.output_projection.weight",
        "cross_attn.proj.bias": "cross_attn.output_projection.bias",
        # Norm layers
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "norm3.weight": "norm3.weight",
        "norm3.bias": "norm3.bias",
        # MLP layers
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
        # Norm layers
        "norm_y.weight": "norm_y.weight",
        "norm_y.bias": "norm_y.bias",
    }

    load_and_validate_state_dict_with_mapping(annotated_block, croco_block, key_mapping)

    # Test with different input patterns
    test_cases = {
        "Uniform": (
            torch.ones_like(query_input),
            torch.ones_like(memory_input),
            torch.ones_like(query_pos),
            torch.ones_like(memory_pos),
        ),
        "Random": (
            torch.randn_like(query_input),
            torch.randn_like(memory_input),
            torch.randn_like(query_pos),
            torch.randn_like(memory_pos),
        ),
        "Zero": (
            torch.zeros_like(query_input),
            torch.zeros_like(memory_input),
            torch.zeros_like(query_pos),
            torch.zeros_like(memory_pos),
        ),
        "Large Values": (
            torch.ones_like(query_input) * 100,
            torch.ones_like(memory_input) * 100,
            torch.ones_like(query_pos) * 100,
            torch.ones_like(memory_pos) * 100,
        ),
    }

    for name, (test_query, test_memory, test_query_pos, test_memory_pos) in test_cases.items():
        with torch.no_grad():
            torch.manual_seed(42)
            croco_query_out, croco_memory_out = croco_block(test_query, test_memory, test_query_pos, test_memory_pos)
            torch.manual_seed(42)
            annotated_query_out, annotated_memory_out = annotated_block(
                test_query, test_memory, test_query_pos, test_memory_pos
            )

            max_diff = torch.max(torch.abs(annotated_query_out - croco_query_out)).item()
            print(f"\n{name} Input Test:")
            print(f"Max difference: {max_diff:.2e}")
            assert max_diff < 1e-10, f"Maximum difference exceeds tolerance for {name} input"

            print(f"Outputs match: {torch.allclose(croco_query_out, annotated_query_out, atol=1e-6)}")
            assert torch.allclose(
                croco_query_out, annotated_query_out, atol=1e-6
            ), f"Outputs do not match for {name} input"

            max_diff_memory = torch.max(torch.abs(annotated_memory_out - croco_memory_out)).item()
            print(f"Max memory difference: {max_diff_memory:.2e}")
            assert max_diff_memory < 1e-10, f"Maximum memory difference exceeds tolerance for {name} input"

            print(f"Memory outputs match: {torch.allclose(croco_memory_out, annotated_memory_out, atol=1e-6)}")
            assert torch.allclose(
                croco_memory_out, annotated_memory_out, atol=1e-10
            ), f"Memory outputs do not match for {name} input"
