import numpy as np
import pytest
import torch
import torch.nn as nn
from test_utils import load_and_validate_state_dict_with_mapping

from src.annotated.croco.patch_embed import PatchEmbed
from src.annotated.croco.vision_transformer import VisionTransformerDecoder as AnnotatedDecoder
from src.annotated.croco.vision_transformer import VisionTransformerDecoderV2 as AnnotatedDecoderV2
from src.annotated.croco.vision_transformer import VisionTransformerEncoder as AnnotatedEncoder
from src.annotated.croco.vision_transformer import VisionTransformerEncoderV2 as AnnotatedEncoderV2
from src.vendored.croco.models.croco import CroCoNet


def encoder_key_mapping():
    # Initialize weights identically
    key_mapping = {
        # Positional embedding and patch embedding
        "enc_pos_embed": "enc_pos_embed",
        # Note here that CroCo's norm_layer for the PatchEmbed is nn.Identity, so we don't need to map it
        # I believe this is a bug with their code
        # "patch_embed.norm.weight": "patch_embed.norm.weight",
        # "patch_embed.norm.bias": "patch_embed.norm.bias",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "enc_norm.weight": "norm.weight",
        "enc_norm.bias": "norm.bias",
    }

    # Add mappings for each encoder block
    for i in range(12):
        # Attention layers
        key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.attn.query_key_value.weight"
        key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.attn.query_key_value.bias"
        key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.proj.weight"
        key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.proj.bias"

        # Norm layers
        key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
        key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"

        # MLP layers
        key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"

    return key_mapping


def encoder_key_mapping_v2():
    # Initialize weights identically
    key_mapping = {
        # Positional embedding and patch embedding
        "enc_pos_embed": "enc_pos_embed",
        # Note here that CroCo's norm_layer for the PatchEmbed is nn.Identity, so we don't need to map it
        # I believe this is a bug with their code
        # "patch_embed.norm.weight": "patch_embed.norm.weight",
        # "patch_embed.norm.bias": "patch_embed.norm.bias",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "enc_norm.weight": "norm.weight",
        "enc_norm.bias": "norm.bias",
    }

    # Add mappings for each encoder block
    for i in range(12):
        # Attention layers
        key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.query_key_value_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.query_key_value_projection.bias"
        key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.output_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.output_projection.bias"

        # Norm layers
        key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
        key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"

        # MLP layers
        key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"

    return key_mapping


@pytest.mark.parametrize(
    "encoder_pointer_and_mapping",
    [(AnnotatedEncoderV2, encoder_key_mapping_v2())],
)
def test_vision_transformer_encoder_equivalence(encoder_pointer_and_mapping):
    """Test equivalence between Annotated and CroCo Encoder implementations"""

    encoder_pointer, key_mapping = encoder_pointer_and_mapping

    # Test parameters
    batch_size = 2
    img_size = 224
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0

    # Create random input with fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, 3, img_size, img_size)

    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        norm_layer=None,
        flatten=True,
    )

    # Initialize both implementations with same parameters
    annotated_encoder = encoder_pointer(
        patch_embed=patch_embed,
        embed_dim=embed_dim,
        num_layers=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        pos_embed_type="sincos2d",
    )

    croco = CroCoNet(
        img_size=img_size,
        patch_size=16,
        enc_embed_dim=embed_dim,
        enc_depth=depth,
        enc_num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        pos_embed="cosine",  # Ensure we're using the same positional embedding type
    )

    load_and_validate_state_dict_with_mapping(annotated_encoder, croco, key_mapping, strict_mapping=False)

    # Test with different input patterns
    test_cases = {
        "Uniform": torch.ones_like(x),
        "Random": torch.randn_like(x),
        "Zero": torch.zeros_like(x),
        "Large Values": torch.ones_like(x) * 100,
    }

    for name, test_input in test_cases.items():
        with torch.no_grad():
            torch.manual_seed(42)  # Reset seed for each test case
            croco_feat, croco_pos, croco_mask = croco._encode_image(test_input, do_mask=True)

            torch.manual_seed(42)  # Reset seed for each test case
            annotated_feat, annotated_pos, annotated_mask = annotated_encoder(test_input, do_mask=True)

            assert torch.allclose(croco_feat, annotated_feat, atol=1e-6), f"Outputs do not match for {name} input"


def decoder_key_mapping():
    # Create mapping between CroCo and Annotated decoder keys
    key_mapping = {
        "dec_pos_embed": "dec_pos_embed",
        "decoder_embed.weight": "decoder_embed.weight",
        "decoder_embed.bias": "decoder_embed.bias",
        "dec_norm.weight": "norm.weight",
        "dec_norm.bias": "norm.bias",
        "mask_token": "mask_token",
    }

    # Add mappings for each decoder block
    for i in range(8):
        # Attention layers
        key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.attn.query_key_value.weight"
        key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.attn.query_key_value.bias"
        key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.proj.weight"
        key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.proj.bias"

        # Cross attention layers
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = f"blocks.{i}.cross_attn.query_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = f"blocks.{i}.cross_attn.query_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = f"blocks.{i}.cross_attn.key_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = f"blocks.{i}.cross_attn.key_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = f"blocks.{i}.cross_attn.value_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = f"blocks.{i}.cross_attn.value_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = f"blocks.{i}.cross_attn.output_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = f"blocks.{i}.cross_attn.output_projection.bias"

        # Norm layers
        key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
        key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"
        key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"blocks.{i}.norm3.weight"
        key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"blocks.{i}.norm3.bias"
        key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"blocks.{i}.norm_y.weight"
        key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"blocks.{i}.norm_y.bias"

        # MLP layers
        key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"
    return key_mapping


def decoder_key_mapping_v2():
    # Create mapping between CroCo and Annotated decoder keys
    key_mapping = {
        "dec_pos_embed": "dec_pos_embed",
        "decoder_embed.weight": "decoder_embed.weight",
        "decoder_embed.bias": "decoder_embed.bias",
        "dec_norm.weight": "norm.weight",
        "dec_norm.bias": "norm.bias",
        "mask_token": "mask_token",
    }

    # Add mappings for each decoder block
    for i in range(8):
        # Attention layers
        key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.self_attend_query_key_value_projection.weight"
        key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.self_attend_query_key_value_projection.bias"
        key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.self_attn.output_projection.weight"
        key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.self_attn.output_projection.bias"

        # Cross attention layers
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = f"blocks.{i}.cross_attn_query_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = f"blocks.{i}.cross_attn_query_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = f"blocks.{i}.cross_attn_key_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = f"blocks.{i}.cross_attn_key_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = f"blocks.{i}.cross_attn_value_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = f"blocks.{i}.cross_attn_value_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = f"blocks.{i}.cross_attn.output_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = f"blocks.{i}.cross_attn.output_projection.bias"

        # Norm layers
        key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
        key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
        key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
        key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"
        key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"blocks.{i}.norm3.weight"
        key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"blocks.{i}.norm3.bias"
        key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"blocks.{i}.norm_y.weight"
        key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"blocks.{i}.norm_y.bias"

        # MLP layers
        key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"
    return key_mapping


@pytest.mark.parametrize(
    "decoder_pointer_and_mapping",
    [(AnnotatedDecoderV2, decoder_key_mapping_v2())],
)
def test_vision_transformer_decoder_equivalence(decoder_pointer_and_mapping):
    """Test equivalence between Annotated and CroCo Decoder implementations"""

    decoder_pointer, key_mapping = decoder_pointer_and_mapping

    # Test parameters
    batch_size = 2
    img_size = 224
    enc_embed_dim = 768
    dec_embed_dim = 512
    enc_depth = 12
    dec_depth = 8
    enc_num_heads = 12
    dec_num_heads = 8
    mlp_ratio = 4.0

    # Create random input with fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, 3, img_size, img_size)

    patch_embed = PatchEmbed(
        img_size=img_size,
        patch_size=16,
        embed_dim=enc_embed_dim,
        norm_layer=None,
        flatten=True,
    )

    # Initialize both implementations with same parameters
    annotated_decoder = decoder_pointer(
        num_patches=patch_embed.num_patches,
        enc_embed_dim=enc_embed_dim,
        embed_dim=dec_embed_dim,
        num_layers=dec_depth,
        num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        pos_embed_type="sincos2d",
    )

    croco = CroCoNet(
        img_size=img_size,
        patch_size=16,
        enc_embed_dim=enc_embed_dim,
        dec_embed_dim=dec_embed_dim,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        enc_num_heads=enc_num_heads,
        dec_num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        pos_embed="cosine",
    )

    # Initialize weights identically
    load_and_validate_state_dict_with_mapping(annotated_decoder, croco, key_mapping, strict_mapping=False)

    # Forward pass
    with torch.no_grad():
        torch.manual_seed(42)
        masked_image_tokens, masked_image_pos, masked_image_mask = croco._encode_image(x, do_mask=True)
        reference_image_tokens, reference_image_pos, _ = croco._encode_image(x, do_mask=False)

        torch.manual_seed(42)
        croco_block_outputs = croco._decoder(
            masked_image_tokens,
            masked_image_pos,
            masked_image_mask,
            reference_image_tokens,
            reference_image_pos,
            return_all_blocks=True,
        )

        torch.manual_seed(42)
        masked_image_tokens_for_annotated, masked_image_pos_for_annotated, masked_image_mask_for_annotated = (
            croco._encode_image(x, do_mask=True)
        )
        reference_image_tokens_for_annotated, reference_image_pos_for_annotated, _ = croco._encode_image(
            x, do_mask=False
        )

        # Verify inputs are equivalent before decoder forward pass
        assert torch.allclose(masked_image_tokens, masked_image_tokens_for_annotated), "Masked image tokens differ"
        assert torch.allclose(masked_image_pos, masked_image_pos_for_annotated), "Masked image positions differ"
        assert torch.allclose(masked_image_mask, masked_image_mask_for_annotated), "Masked image masks differ"
        assert torch.allclose(
            reference_image_tokens, reference_image_tokens_for_annotated
        ), "Reference image tokens differ"
        assert torch.allclose(
            reference_image_pos, reference_image_pos_for_annotated
        ), "Reference image positions differ"

        torch.manual_seed(42)
        annotated_block_outputs = annotated_decoder(
            masked_image_tokens_for_annotated,
            masked_image_pos_for_annotated,
            masked_image_mask_for_annotated,
            reference_image_tokens_for_annotated,
            reference_image_pos_for_annotated,
            return_all_blocks=True,
        )

        # Compare decoder embeddings
        croco_decoder_embed = croco.decoder_embed(masked_image_tokens)
        annotated_decoder_embed = annotated_decoder.decoder_embed(masked_image_tokens_for_annotated)
        assert torch.allclose(croco_decoder_embed, annotated_decoder_embed, atol=1e-6), "Decoder embeddings differ"

        # Compare block outputs
        for i, (croco_block, annotated_block) in enumerate(zip(croco_block_outputs, annotated_block_outputs)):
            block_max_diff = torch.max(torch.abs(croco_block - annotated_block)).item()
            assert block_max_diff < 1e-6, f"Block {i} difference exceeds tolerance"

        # Compare positional embeddings
        assert torch.allclose(
            croco.dec_pos_embed, annotated_decoder.dec_pos_embed, atol=1e-6
        ), "Positional embeddings differ"

        # Compare mask tokens
        assert torch.allclose(croco.mask_token, annotated_decoder.mask_token, atol=1e-6), "Mask tokens differ"

    # Compare outputs
    croco_dec_feat = croco_block_outputs[-1]
    annotated_dec_feat = annotated_block_outputs[-1]
    max_diff = torch.max(torch.abs(annotated_dec_feat - croco_dec_feat)).item()
    mean_diff = torch.mean(torch.abs(annotated_dec_feat - croco_dec_feat)).item()
    assert max_diff < 1e-6, "Maximum difference exceeds tolerance"
    assert mean_diff < 1e-6, "Mean difference exceeds tolerance"
    assert annotated_dec_feat.shape == croco_dec_feat.shape, "Output shapes do not match"

    # Compare outputs at each position
    if annotated_dec_feat.shape == croco_dec_feat.shape:
        B, N, D = annotated_dec_feat.shape
        for i in range(N):
            assert torch.allclose(
                annotated_dec_feat[:, i, :], croco_dec_feat[:, i, :], atol=1e-6
            ), f"Position {i} differs"
