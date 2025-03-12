import pytest
import torch
import torch.nn as nn
from test_utils import load_and_validate_state_dict_with_mapping

from src.annotated.croco.encoder_block import TransformerEncoderBlock as AnnotatedEncoderBlock
from src.annotated.croco.encoder_block import TransformerEncoderBlockV2 as AnnotatedEncoderBlockV2
from src.croco.models.blocks import Block as CrocoBlock


def test_encoder_block_initialization():
    """Test initialization with different parameters"""
    dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True

    # Test basic initialization
    block = AnnotatedEncoderBlock(embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    # Verify layer normalization
    assert isinstance(block.norm1, nn.LayerNorm)
    assert isinstance(block.norm2, nn.LayerNorm)
    assert block.norm1.normalized_shape[0] == dim
    assert block.norm2.normalized_shape[0] == dim


def test_encoder_block_forward():
    """Test forward pass and output shapes"""
    batch_size = 4
    seq_len = 196
    dim = 768
    num_heads = 12

    block = AnnotatedEncoderBlock(embed_dim=dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.randn(batch_size, seq_len, 2)

    # Test forward pass
    output = block(x, pos)

    # Check output shape
    assert output.shape == (batch_size, seq_len, dim)

    # Check output is different from input (transformation occurred)
    assert not torch.allclose(output, x)


def test_encoder_block_equivalence():
    """Test equivalence between Annotated and CroCo EncoderBlock implementations"""
    batch_size = 4
    seq_len = 196
    dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True
    drop_path = 0.0

    # Initialize both implementations
    annotated_block = AnnotatedEncoderBlock(embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    croco_block = CrocoBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    # Create input tensors
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.randn(batch_size, seq_len, 2)

    # Set both blocks to eval mode to disable dropout
    annotated_block.eval()
    croco_block.eval()

    # Initialize weights identically
    key_mapping = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "attn.qkv.weight": "attn.query_key_value.weight",
        "attn.qkv.bias": "attn.query_key_value.bias",
        "attn.proj.weight": "attn.proj.weight",
        "attn.proj.bias": "attn.proj.bias",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
    }

    load_and_validate_state_dict_with_mapping(annotated_block, croco_block, key_mapping)

    # Test forward pass
    with torch.no_grad():
        annotated_out = annotated_block(x, pos)
        croco_out = croco_block(x, pos)

        # Test outputs are close
        assert torch.allclose(annotated_out, croco_out, atol=1e-5)


def test_encoder_block_v2_equivalence():
    """Test equivalence between Annotated and CroCo EncoderBlock implementations"""
    batch_size = 4
    seq_len = 196
    dim = 768
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True

    # Initialize both implementations
    annotated_block = AnnotatedEncoderBlockV2(
        embed_dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias
    )

    croco_block = CrocoBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias)

    # Create input tensors
    x = torch.randn(batch_size, seq_len, dim)
    pos = torch.randn(batch_size, seq_len, 2)

    # Initialize weights identically
    key_mapping = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "attn.qkv.weight": "query_key_value_projection.weight",
        "attn.qkv.bias": "query_key_value_projection.bias",
        "attn.proj.weight": "attn.output_projection.weight",
        "attn.proj.bias": "attn.output_projection.bias",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
    }
    load_and_validate_state_dict_with_mapping(annotated_block, croco_block, key_mapping)

    # Test forward pass
    with torch.no_grad():
        annotated_out = annotated_block(x, pos, pos)
        croco_out = croco_block(x, pos)

        # Test outputs are close
        assert torch.allclose(annotated_out, croco_out)
