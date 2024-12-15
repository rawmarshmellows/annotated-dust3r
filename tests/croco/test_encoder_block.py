import pytest
import torch
import torch.nn as nn

from src.annotated.encoder_block import TransformerEncoderBlock as AnnotatedEncoderBlock
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
    with torch.no_grad():
        # Copy norm1 weights
        croco_block.norm1.weight.data = annotated_block.norm1.weight.data.clone()
        croco_block.norm1.bias.data = annotated_block.norm1.bias.data.clone()

        # Copy norm2 weights
        croco_block.norm2.weight.data = annotated_block.norm2.weight.data.clone()
        croco_block.norm2.bias.data = annotated_block.norm2.bias.data.clone()

        # Copy attention weights
        croco_block.attn.qkv.weight.data = annotated_block.attn.query_key_value.weight.data.clone()
        croco_block.attn.qkv.bias.data = annotated_block.attn.query_key_value.bias.data.clone()
        croco_block.attn.proj.weight.data = annotated_block.attn.proj.weight.data.clone()
        croco_block.attn.proj.bias.data = annotated_block.attn.proj.bias.data.clone()

        # Copy MLP weights
        croco_block.mlp.fc1.weight.data = annotated_block.mlp.fc1.weight.data.clone()
        croco_block.mlp.fc1.bias.data = annotated_block.mlp.fc1.bias.data.clone()
        croco_block.mlp.fc2.weight.data = annotated_block.mlp.fc2.weight.data.clone()
        croco_block.mlp.fc2.bias.data = annotated_block.mlp.fc2.bias.data.clone()

    # Test forward pass
    with torch.no_grad():
        annotated_out = annotated_block(x, pos)
        croco_out = croco_block(x, pos)

        # Test outputs are close
        assert torch.allclose(annotated_out, croco_out, atol=1e-5)
