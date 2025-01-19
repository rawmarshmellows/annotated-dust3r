import pytest
import torch

from src.annotated.croco.croco import AnnotatedCroCo
from src.croco.models.croco import CroCoNet
from tests.test_utils import load_and_validate_state_dict_with_mapping


def test_croco_equivalence():
    """Test that the annotated CroCo model is equivalent to the original CroCo model."""
    annotated_model = AnnotatedCroCo(img_size=224, patch_size=16)
    original_model = CroCoNet(img_size=224, patch_size=16)

    # Initialize weights identically
    encoder_key_mapping = {
        # Positional embedding and patch embedding
        "enc_pos_embed": "encoder.enc_pos_embed",
        # Note here that CroCo's norm_layer for the PatchEmbed is nn.Identity, so we don't need to map it
        # I believe this is a bug with their code
        # "patch_embed.norm.weight": "patch_embed.norm.weight",
        # "patch_embed.norm.bias": "patch_embed.norm.bias",
        "patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
        "patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
        "enc_norm.weight": "encoder.norm.weight",
        "enc_norm.bias": "encoder.norm.bias",
    }

    # Add mappings for each encoder block
    for i in range(12):
        # Attention layers
        encoder_key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = (
            f"encoder.blocks.{i}.query_key_value_projection.weight"
        )
        encoder_key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"encoder.blocks.{i}.query_key_value_projection.bias"
        encoder_key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"encoder.blocks.{i}.attn.output_projection.weight"
        encoder_key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"encoder.blocks.{i}.attn.output_projection.bias"

        # Norm layers
        encoder_key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"encoder.blocks.{i}.norm1.weight"
        encoder_key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"encoder.blocks.{i}.norm1.bias"
        encoder_key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"encoder.blocks.{i}.norm2.weight"
        encoder_key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"encoder.blocks.{i}.norm2.bias"

        # MLP layers
        encoder_key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"encoder.blocks.{i}.mlp.fc1.weight"
        encoder_key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"encoder.blocks.{i}.mlp.fc1.bias"
        encoder_key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"encoder.blocks.{i}.mlp.fc2.weight"
        encoder_key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"encoder.blocks.{i}.mlp.fc2.bias"

    decoder_key_mapping = {
        "dec_pos_embed": "decoder.dec_pos_embed",
        "decoder_embed.weight": "decoder.decoder_embed.weight",
        "decoder_embed.bias": "decoder.decoder_embed.bias",
        "dec_norm.weight": "decoder.norm.weight",
        "dec_norm.bias": "decoder.norm.bias",
        "mask_token": "decoder.mask_token",
    }

    # Add mappings for each decoder block
    for i in range(8):
        # Attention layers
        decoder_key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.bias"
        )
        decoder_key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = (
            f"decoder.blocks.{i}.self_attn.output_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"decoder.blocks.{i}.self_attn.output_projection.bias"

        # Cross attention layers
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = (
            f"decoder.blocks.{i}.cross_attn_query_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = (
            f"decoder.blocks.{i}.cross_attn_query_projection.bias"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = (
            f"decoder.blocks.{i}.cross_attn_key_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = (
            f"decoder.blocks.{i}.cross_attn_key_projection.bias"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = (
            f"decoder.blocks.{i}.cross_attn_value_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = (
            f"decoder.blocks.{i}.cross_attn_value_projection.bias"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = (
            f"decoder.blocks.{i}.cross_attn.output_projection.weight"
        )
        decoder_key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = (
            f"decoder.blocks.{i}.cross_attn.output_projection.bias"
        )

        # Norm layers
        decoder_key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"decoder.blocks.{i}.norm1.weight"
        decoder_key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"decoder.blocks.{i}.norm1.bias"
        decoder_key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"decoder.blocks.{i}.norm2.weight"
        decoder_key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"decoder.blocks.{i}.norm2.bias"
        decoder_key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"decoder.blocks.{i}.norm3.weight"
        decoder_key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"decoder.blocks.{i}.norm3.bias"
        decoder_key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"decoder.blocks.{i}.norm_y.weight"
        decoder_key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"decoder.blocks.{i}.norm_y.bias"

        # MLP layers
        decoder_key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"decoder.blocks.{i}.mlp.fc1.weight"
        decoder_key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"decoder.blocks.{i}.mlp.fc1.bias"
        decoder_key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"decoder.blocks.{i}.mlp.fc2.weight"
        decoder_key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"decoder.blocks.{i}.mlp.fc2.bias"

    key_mapping = {
        "prediction_head.weight": "prediction_head.weight",
        "prediction_head.bias": "prediction_head.bias",
        **encoder_key_mapping,
        **decoder_key_mapping,
    }

    load_and_validate_state_dict_with_mapping(annotated_model, original_model, key_mapping)

    # Verify outputs are equivalent
    with torch.no_grad():
        masked_image, reference_image = torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)
        torch.manual_seed(42)
        annotated_output, _, _ = annotated_model(masked_image, reference_image)
        torch.manual_seed(42)
        original_output, _, _ = original_model(masked_image, reference_image)
        assert annotated_output.shape == original_output.shape
        assert torch.allclose(annotated_output, original_output)
