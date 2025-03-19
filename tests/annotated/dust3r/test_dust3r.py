import pytest
import torch
from test_utils import load_and_validate_state_dict_with_mapping

from dust3r.model import AsymmetricCroCo3DStereo
from src.annotated.dust3r.dust3r import AnnotatedAsymmetricCroCo3DStereo


def test_asymmetric_croco_3d_equivalence():
    """Test that the annotated AsymmetricCroCo3DStereo is equivalent to the original."""

    # Initialize models
    annotated_model = AnnotatedAsymmetricCroCo3DStereo(
        img_size=224, patch_size=16, output_mode="pts3d", head_type="linear", pos_embed="RoPE100"
    )
    original_model = AsymmetricCroCo3DStereo(
        img_size=(224, 224), patch_size=16, output_mode="pts3d", head_type="linear", pos_embed="RoPE100"
    )

    # Define key mapping between original and annotated models
    key_mapping = {
        # Encoder mappings
        "patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
        "patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
        "enc_norm.weight": "encoder.norm.weight",
        "enc_norm.bias": "encoder.norm.bias",
        # Decoder mappings
        "decoder_embed.weight": "decoder.decoder_embed.weight",
        "decoder_embed.bias": "decoder.decoder_embed.bias",
        "dec_norm.weight": "decoder.norm.weight",
        "dec_norm.bias": "decoder.norm.bias",
        "mask_token": "decoder.mask_token",
        # TODO: Ignored as only CroCo has prediction head, dust3r doesn't have one, need to abstract this
        # Prediction head mappings
        # "prediction_head.weight": "prediction_head.weight",
        # "prediction_head.bias": "prediction_head.bias",
        # Downstream head mappings
        "downstream_head1.proj.weight": "downstream_head1.proj.weight",
        "downstream_head1.proj.bias": "downstream_head1.proj.bias",
        "downstream_head2.proj.weight": "downstream_head2.proj.weight",
        "downstream_head2.proj.bias": "downstream_head2.proj.bias",
    }

    for i in range(12):
        # Attention layers
        key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = f"encoder.blocks.{i}.query_key_value_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"encoder.blocks.{i}.query_key_value_projection.bias"
        key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"encoder.blocks.{i}.attn.output_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"encoder.blocks.{i}.attn.output_projection.bias"

        # Norm layers
        key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"encoder.blocks.{i}.norm1.weight"
        key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"encoder.blocks.{i}.norm1.bias"
        key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"encoder.blocks.{i}.norm2.weight"
        key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"encoder.blocks.{i}.norm2.bias"

        # MLP layers
        key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"encoder.blocks.{i}.mlp.fc1.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"encoder.blocks.{i}.mlp.fc1.bias"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"encoder.blocks.{i}.mlp.fc2.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"encoder.blocks.{i}.mlp.fc2.bias"

    # # Add mappings for each decoder block
    for i in range(8):
        # Source image decoder - Attention layers
        key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.bias"
        )
        key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = f"decoder.blocks.{i}.self_attn.output_projection.weight"
        key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"decoder.blocks.{i}.self_attn.output_projection.bias"

        # Source image decoder - Cross attention layers
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = (
            f"decoder.blocks.{i}.cross_attn_query_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = f"decoder.blocks.{i}.cross_attn_query_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = f"decoder.blocks.{i}.cross_attn_key_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = f"decoder.blocks.{i}.cross_attn_key_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = (
            f"decoder.blocks.{i}.cross_attn_value_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = f"decoder.blocks.{i}.cross_attn_value_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = (
            f"decoder.blocks.{i}.cross_attn.output_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = f"decoder.blocks.{i}.cross_attn.output_projection.bias"

        # Source image decoder - Norm layers
        key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"decoder.blocks.{i}.norm1.weight"
        key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"decoder.blocks.{i}.norm1.bias"
        key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"decoder.blocks.{i}.norm2.weight"
        key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"decoder.blocks.{i}.norm2.bias"
        key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"decoder.blocks.{i}.norm3.weight"
        key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"decoder.blocks.{i}.norm3.bias"
        key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"decoder.blocks.{i}.norm_y.weight"
        key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"decoder.blocks.{i}.norm_y.bias"

        # Source image decoder - MLP layers
        key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"decoder.blocks.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"decoder.blocks.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"decoder.blocks.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"decoder.blocks.{i}.mlp.fc2.bias"

        # Reference image decoder - Attention layers
        key_mapping[f"dec_blocks2.{i}.attn.qkv.weight"] = (
            f"dec_blocks2.{i}.self_attend_query_key_value_projection.weight"
        )
        key_mapping[f"dec_blocks2.{i}.attn.qkv.bias"] = f"dec_blocks2.{i}.self_attend_query_key_value_projection.bias"
        key_mapping[f"dec_blocks2.{i}.attn.proj.weight"] = f"dec_blocks2.{i}.self_attn.output_projection.weight"
        key_mapping[f"dec_blocks2.{i}.attn.proj.bias"] = f"dec_blocks2.{i}.self_attn.output_projection.bias"

        # Reference image decoder - Cross attention layers
        key_mapping[f"dec_blocks2.{i}.cross_attn.projq.weight"] = f"dec_blocks2.{i}.cross_attn_query_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projq.bias"] = f"dec_blocks2.{i}.cross_attn_query_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projk.weight"] = f"dec_blocks2.{i}.cross_attn_key_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projk.bias"] = f"dec_blocks2.{i}.cross_attn_key_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projv.weight"] = f"dec_blocks2.{i}.cross_attn_value_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projv.bias"] = f"dec_blocks2.{i}.cross_attn_value_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.proj.weight"] = f"dec_blocks2.{i}.cross_attn.output_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.proj.bias"] = f"dec_blocks2.{i}.cross_attn.output_projection.bias"

        # Reference image decoder - Norm layers
        key_mapping[f"dec_blocks2.{i}.norm1.weight"] = f"dec_blocks2.{i}.norm1.weight"
        key_mapping[f"dec_blocks2.{i}.norm1.bias"] = f"dec_blocks2.{i}.norm1.bias"
        key_mapping[f"dec_blocks2.{i}.norm2.weight"] = f"dec_blocks2.{i}.norm2.weight"
        key_mapping[f"dec_blocks2.{i}.norm2.bias"] = f"dec_blocks2.{i}.norm2.bias"
        key_mapping[f"dec_blocks2.{i}.norm3.weight"] = f"dec_blocks2.{i}.norm3.weight"
        key_mapping[f"dec_blocks2.{i}.norm3.bias"] = f"dec_blocks2.{i}.norm3.bias"
        key_mapping[f"dec_blocks2.{i}.norm_y.weight"] = f"dec_blocks2.{i}.norm_y.weight"
        key_mapping[f"dec_blocks2.{i}.norm_y.bias"] = f"dec_blocks2.{i}.norm_y.bias"

        # Reference image decoder - MLP layers
        key_mapping[f"dec_blocks2.{i}.mlp.fc1.weight"] = f"dec_blocks2.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks2.{i}.mlp.fc1.bias"] = f"dec_blocks2.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks2.{i}.mlp.fc2.weight"] = f"dec_blocks2.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks2.{i}.mlp.fc2.bias"] = f"dec_blocks2.{i}.mlp.fc2.bias"

    # Load and validate the state dict with the mapping
    load_and_validate_state_dict_with_mapping(annotated_model, original_model, key_mapping, strict_mapping=True)

    # Test forward pass equivalence
    with torch.no_grad():
        # Create sample inputs
        batch_size = 2
        img_size = 224
        view1 = {
            "img": torch.randn(batch_size, 3, img_size, img_size),
            "true_shape": torch.tensor([[img_size, img_size]] * batch_size),
            "idx": 0,
            "instance": "0",
        }
        view2 = {
            "img": torch.randn(batch_size, 3, img_size, img_size),
            "true_shape": torch.tensor([[img_size, img_size]] * batch_size),
            "idx": 1,
            "instance": "1",
        }

        # Set same random seed for both forward passes
        torch.manual_seed(42)
        annotated_output = annotated_model(view1, view2)

        torch.manual_seed(42)
        original_output = original_model(view1, view2)

        # Compare outputs
        for annotated_view, original_view in zip(annotated_output, original_output):
            for key in annotated_view:
                assert torch.allclose(
                    annotated_view[key], original_view[key], rtol=1e-4, atol=1e-4
                ), f"Mismatch in {key}"


# @pytest.mark.skip
def test_asymmetric_croco_3d_equivalence_from_pretrained_naver_DUSt3R_ViTLarge_BaseDecoder_512_dpt():
    """Test that the annotated AsymmetricCroCo3DStereo is equivalent to the original."""

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

    # Initialize models
    original_model = AsymmetricCroCo3DStereo.from_pretrained(model_name)

    # annotated_model = AnnotatedAsymmetricCroCo3DStereo(
    #     output_mode=original_model.output_mode,
    #     head_type=original_model.head_type,
    #     depth_mode=original_model.depth_mode,
    #     conf_mode=original_model.conf_mode,
    #     freeze=original_model.freeze,
    #     landscape_only=False,
    #     patch_embed_cls=original_model.patch_embed_cls,
    #     **original_model.croco_args,
    # )
    annotated_model = AnnotatedAsymmetricCroCo3DStereo.from_pretrained_naver_DUSt3R_ViTLarge_BaseDecoder_512_dpt(
        original_model
    )

    # Define key mapping between original and annotated models
    key_mapping = {
        # Encoder mappings
        "patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
        "patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
        "enc_norm.weight": "encoder.norm.weight",
        "enc_norm.bias": "encoder.norm.bias",
        # Decoder mappings
        "decoder_embed.weight": "decoder.decoder_embed.weight",
        "decoder_embed.bias": "decoder.decoder_embed.bias",
        "dec_norm.weight": "decoder.norm.weight",
        "dec_norm.bias": "decoder.norm.bias",
        "mask_token": "decoder.mask_token",
    }

    for i in range(1, 3):
        # Downstream head mappings
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.0.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.0.0.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.0.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.0.0.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.1.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.0.1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.1.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.0.1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.0.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.1.0.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.0.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.1.0.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.1.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.1.1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.1.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.1.1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.2.0.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.2.0.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.2.0.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.2.0.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.0.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.3.0.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.0.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.3.0.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.1.weight"] = (
            f"downstream_head{i}.dpt.act_postprocess.3.1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.1.bias"] = (
            f"downstream_head{i}.dpt.act_postprocess.3.1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.head.0.weight"] = f"downstream_head{i}.dpt.head.0.weight"
        key_mapping[f"downstream_head{i}.dpt.head.0.bias"] = f"downstream_head{i}.dpt.head.0.bias"
        key_mapping[f"downstream_head{i}.dpt.head.2.weight"] = f"downstream_head{i}.dpt.head.2.weight"
        key_mapping[f"downstream_head{i}.dpt.head.2.bias"] = f"downstream_head{i}.dpt.head.2.bias"
        key_mapping[f"downstream_head{i}.dpt.head.4.weight"] = f"downstream_head{i}.dpt.head.4.weight"
        key_mapping[f"downstream_head{i}.dpt.head.4.bias"] = f"downstream_head{i}.dpt.head.4.bias"
        key_mapping[f"downstream_head{i}.dpt.scratch.layer1_rn.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer1_rn.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer2_rn.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer2_rn.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer3_rn.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer3_rn.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer4_rn.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer4_rn.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.0.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer_rn.0.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.1.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer_rn.1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.2.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer_rn.2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.3.weight"] = (
            f"downstream_head{i}.dpt.scratch.layer_rn.3.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.bias"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.weight"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.weight"
        )
        key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.bias"] = (
            f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.bias"
        )

    for i in range(24):
        # Attention layers
        key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = f"encoder.blocks.{i}.query_key_value_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"encoder.blocks.{i}.query_key_value_projection.bias"
        key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"encoder.blocks.{i}.attn.output_projection.weight"
        key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"encoder.blocks.{i}.attn.output_projection.bias"

        # Norm layers
        key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"encoder.blocks.{i}.norm1.weight"
        key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"encoder.blocks.{i}.norm1.bias"
        key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"encoder.blocks.{i}.norm2.weight"
        key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"encoder.blocks.{i}.norm2.bias"

        # MLP layers
        key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"encoder.blocks.{i}.mlp.fc1.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"encoder.blocks.{i}.mlp.fc1.bias"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"encoder.blocks.{i}.mlp.fc2.weight"
        key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"encoder.blocks.{i}.mlp.fc2.bias"

    # # Add mappings for each decoder block
    for i in range(12):
        # Source image decoder - Attention layers
        key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = (
            f"decoder.blocks.{i}.self_attend_query_key_value_projection.bias"
        )
        key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = f"decoder.blocks.{i}.self_attn.output_projection.weight"
        key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"decoder.blocks.{i}.self_attn.output_projection.bias"

        # Source image decoder - Cross attention layers
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = (
            f"decoder.blocks.{i}.cross_attn_query_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = f"decoder.blocks.{i}.cross_attn_query_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = f"decoder.blocks.{i}.cross_attn_key_projection.weight"
        key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = f"decoder.blocks.{i}.cross_attn_key_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = (
            f"decoder.blocks.{i}.cross_attn_value_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = f"decoder.blocks.{i}.cross_attn_value_projection.bias"
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = (
            f"decoder.blocks.{i}.cross_attn.output_projection.weight"
        )
        key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = f"decoder.blocks.{i}.cross_attn.output_projection.bias"

        # Source image decoder - Norm layers
        key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"decoder.blocks.{i}.norm1.weight"
        key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"decoder.blocks.{i}.norm1.bias"
        key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"decoder.blocks.{i}.norm2.weight"
        key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"decoder.blocks.{i}.norm2.bias"
        key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"decoder.blocks.{i}.norm3.weight"
        key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"decoder.blocks.{i}.norm3.bias"
        key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"decoder.blocks.{i}.norm_y.weight"
        key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"decoder.blocks.{i}.norm_y.bias"

        # Source image decoder - MLP layers
        key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"decoder.blocks.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"decoder.blocks.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"decoder.blocks.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"decoder.blocks.{i}.mlp.fc2.bias"

        # Reference image decoder - Attention layers
        key_mapping[f"dec_blocks2.{i}.attn.qkv.weight"] = (
            f"dec_blocks2.{i}.self_attend_query_key_value_projection.weight"
        )
        key_mapping[f"dec_blocks2.{i}.attn.qkv.bias"] = f"dec_blocks2.{i}.self_attend_query_key_value_projection.bias"
        key_mapping[f"dec_blocks2.{i}.attn.proj.weight"] = f"dec_blocks2.{i}.self_attn.output_projection.weight"
        key_mapping[f"dec_blocks2.{i}.attn.proj.bias"] = f"dec_blocks2.{i}.self_attn.output_projection.bias"

        # Reference image decoder - Cross attention layers
        key_mapping[f"dec_blocks2.{i}.cross_attn.projq.weight"] = f"dec_blocks2.{i}.cross_attn_query_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projq.bias"] = f"dec_blocks2.{i}.cross_attn_query_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projk.weight"] = f"dec_blocks2.{i}.cross_attn_key_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projk.bias"] = f"dec_blocks2.{i}.cross_attn_key_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projv.weight"] = f"dec_blocks2.{i}.cross_attn_value_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.projv.bias"] = f"dec_blocks2.{i}.cross_attn_value_projection.bias"
        key_mapping[f"dec_blocks2.{i}.cross_attn.proj.weight"] = f"dec_blocks2.{i}.cross_attn.output_projection.weight"
        key_mapping[f"dec_blocks2.{i}.cross_attn.proj.bias"] = f"dec_blocks2.{i}.cross_attn.output_projection.bias"

        # Reference image decoder - Norm layers
        key_mapping[f"dec_blocks2.{i}.norm1.weight"] = f"dec_blocks2.{i}.norm1.weight"
        key_mapping[f"dec_blocks2.{i}.norm1.bias"] = f"dec_blocks2.{i}.norm1.bias"
        key_mapping[f"dec_blocks2.{i}.norm2.weight"] = f"dec_blocks2.{i}.norm2.weight"
        key_mapping[f"dec_blocks2.{i}.norm2.bias"] = f"dec_blocks2.{i}.norm2.bias"
        key_mapping[f"dec_blocks2.{i}.norm3.weight"] = f"dec_blocks2.{i}.norm3.weight"
        key_mapping[f"dec_blocks2.{i}.norm3.bias"] = f"dec_blocks2.{i}.norm3.bias"
        key_mapping[f"dec_blocks2.{i}.norm_y.weight"] = f"dec_blocks2.{i}.norm_y.weight"
        key_mapping[f"dec_blocks2.{i}.norm_y.bias"] = f"dec_blocks2.{i}.norm_y.bias"

        # Reference image decoder - MLP layers
        key_mapping[f"dec_blocks2.{i}.mlp.fc1.weight"] = f"dec_blocks2.{i}.mlp.fc1.weight"
        key_mapping[f"dec_blocks2.{i}.mlp.fc1.bias"] = f"dec_blocks2.{i}.mlp.fc1.bias"
        key_mapping[f"dec_blocks2.{i}.mlp.fc2.weight"] = f"dec_blocks2.{i}.mlp.fc2.weight"
        key_mapping[f"dec_blocks2.{i}.mlp.fc2.bias"] = f"dec_blocks2.{i}.mlp.fc2.bias"

    # Load and validate the state dict with the mapping
    load_and_validate_state_dict_with_mapping(annotated_model, original_model, key_mapping, strict_mapping=True)

    # Test forward pass equivalence
    with torch.no_grad():
        # Create sample inputs
        batch_size = 2
        img_size = 512
        view1 = {
            "img": torch.randn(batch_size, 3, img_size, img_size),
            "true_shape": torch.tensor([[img_size, img_size]] * batch_size),
            "idx": 0,
            "instance": "0",
        }
        view2 = {
            "img": torch.randn(batch_size, 3, img_size, img_size),
            "true_shape": torch.tensor([[img_size, img_size]] * batch_size),
            "idx": 1,
            "instance": "1",
        }

        # Set same random seed for both forward passes
        torch.manual_seed(42)
        annotated_output = annotated_model(view1, view2)

        torch.manual_seed(42)
        original_output = original_model(view1, view2)

        # Compare outputs
        for annotated_view, original_view in zip(annotated_output, original_output):
            for key in annotated_view:
                assert torch.allclose(
                    annotated_view[key], original_view[key], rtol=1e-4, atol=1e-4
                ), f"Mismatch in {key}"
