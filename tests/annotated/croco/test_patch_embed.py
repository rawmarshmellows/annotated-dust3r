import numpy as np
import pytest
import torch
import torch.nn as nn

from src.annotated.croco.patch_embed import PatchEmbed as AnnotatedPatchEmbed
from src.annotated.croco.patch_embed import PositionGetter
from src.vendored.croco.models.blocks import PatchEmbed as CrocoPatchEmbed


def test_position_getter():
    """Test the PositionGetter class."""
    print("Testing PositionGetter...")

    # Test initialization
    pos_getter = PositionGetter()
    assert isinstance(pos_getter.cache_positions, dict), "Cache should be a dictionary"
    assert len(pos_getter.cache_positions) == 0, "Cache should be empty at initialization"
    print("✓ Initialization test passed")

    # Test position generation and caching
    batch_size = 2
    num_patches_h = 3
    num_patches_w = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate positions
    positions = pos_getter(batch_size, num_patches_h, num_patches_w, device)

    # Check shape
    assert positions.shape == (
        batch_size,
        num_patches_h * num_patches_w,
        2,
    ), f"Expected shape ({batch_size}, {num_patches_h * num_patches_w}, 2), got {positions.shape}"

    # Check cache
    cache_key = (num_patches_h, num_patches_w)
    assert cache_key in pos_getter.cache_positions, "Positions should be cached"
    cached_positions = pos_getter.cache_positions[cache_key]
    assert cached_positions.shape == (
        num_patches_h * num_patches_w,
        2,
    ), "Cached positions should have shape (num_patches_h * num_patches_w, 2)"

    # Check coordinate values
    assert torch.all(positions[0, 0] == torch.tensor([0, 0], device=device)), "First position should be (0, 0)"
    assert torch.all(
        positions[0, -1] == torch.tensor([num_patches_h - 1, num_patches_w - 1], device=device)
    ), f"Last position should be ({num_patches_h-1}, {num_patches_w-1})"
    print("✓ Position generation test passed")

    # Test cache reuse
    positions2 = pos_getter(batch_size, num_patches_h, num_patches_w, device)
    assert torch.equal(positions, positions2), "Cached positions should be identical"
    assert len(pos_getter.cache_positions) == 1, "Cache should have only one entry"
    print("✓ Cache reuse test passed")

    print("All PositionGetter tests passed!\n")


def test_patch_embed_initialization():
    """Test valid initialization with different parameters"""
    # Test with flatten=True and LayerNorm
    patch_embed = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.LayerNorm, flatten=True
    )
    assert isinstance(patch_embed.norm, nn.LayerNorm)

    # Test with flatten=False and no norm
    patch_embed_no_norm = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None, flatten=False
    )
    assert isinstance(patch_embed_no_norm.norm, nn.Identity)

    # Test with flatten=False and Identity norm
    patch_embed_identity = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.Identity, flatten=False
    )
    assert isinstance(patch_embed_identity.norm, nn.Identity)

    # Test LayerNorm incompatibility with flatten=False
    with pytest.raises(ValueError, match="LayerNorm cannot be used with flatten=False"):
        AnnotatedPatchEmbed(
            img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.LayerNorm, flatten=False
        )


def test_patch_embed_forward():
    """Test forward pass with different configurations"""
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Test with flatten=True
    patch_embed = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.LayerNorm, flatten=True
    )
    embeddings, pos_encodings = patch_embed(x)
    assert embeddings.shape == (batch_size, 196, 768)
    assert pos_encodings.shape == (batch_size, 196, 2)

    # Test with flatten=False
    patch_embed_no_norm = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None, flatten=False
    )
    embeddings, pos_encodings = patch_embed_no_norm(x)
    assert embeddings.shape == (batch_size, 768, 14, 14)
    assert pos_encodings.shape == (batch_size, 196, 2)


def test_patch_embed_invalid_input():
    """Test handling of invalid input sizes"""
    patch_embed = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.LayerNorm, flatten=True
    )

    # Test invalid input size
    with pytest.raises(AssertionError):
        x_invalid = torch.randn(4, 3, 256, 224)
        patch_embed(x_invalid)


def test_patch_embed_weight_initialization():
    """Test weight initialization"""
    patch_embed = AnnotatedPatchEmbed(
        img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=nn.LayerNorm, flatten=True
    )

    def check_xavier_uniform(tensor):
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
        bound = np.sqrt(6.0 / (fan_in + fan_out))
        return torch.all(tensor >= -bound) and torch.all(tensor <= bound)

    w = patch_embed.proj.weight
    assert check_xavier_uniform(w.view(w.size(0), -1))

    if patch_embed.proj.bias is not None:
        assert torch.all(patch_embed.proj.bias == 0)


def test_patch_embed_equivalence():
    """Test equivalence between Annotated and CroCo PatchEmbed implementations"""
    # Test parameters
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    batch_size = 2
    norm_layer = torch.nn.LayerNorm

    # Create random input
    x = torch.randn(batch_size, in_channels, img_size, img_size)

    # Initialize both implementations
    annotated_embed = AnnotatedPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        flatten=True,
    )

    croco_embed = CrocoPatchEmbed(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_channels,  # Note: different parameter name
        embed_dim=embed_dim,
        norm_layer=norm_layer,
        flatten=True,
    )

    # Initialize weights identically
    with torch.no_grad():
        croco_embed.proj.weight.data = annotated_embed.proj.weight.data.clone()
        croco_embed.proj.bias.data = annotated_embed.proj.bias.data.clone()
        if hasattr(croco_embed.norm, "weight"):
            croco_embed.norm.weight.data = annotated_embed.norm.weight.data.clone()
            croco_embed.norm.bias.data = annotated_embed.norm.bias.data.clone()

    # Test forward pass
    with torch.no_grad():
        annotated_out, annotated_pos = annotated_embed(x)
        croco_out, croco_pos = croco_embed(x)

        assert torch.allclose(annotated_out, croco_out, atol=1e-6)
        assert torch.allclose(annotated_pos, croco_pos, atol=1e-6)

    # Test properties
    assert annotated_embed.num_patches == croco_embed.num_patches
    assert annotated_embed.num_patches_h == croco_embed.grid_size[0]
    assert annotated_embed.num_patches_w == croco_embed.grid_size[1]
