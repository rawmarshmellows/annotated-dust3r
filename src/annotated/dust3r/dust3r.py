import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import huggingface_hub
import icecream as ic
import torch
import torch.nn as nn
from torch import Tensor

from ...utils import load_and_validate_state_dict_with_mapping
from ..croco.croco import AnnotatedCroCo
from .heads import head_factory
from .utils import fill_default_args, freeze_all_params, interleave, is_symmetrized, transpose_to_landscape


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location="cpu")

    args = ckpt["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args:
        args = args[:-1] + ", landscape_only=False)"
    else:
        args = args.replace(" ", "").replace("landscape_only=True", "landscape_only=False")
    assert "landscape_only=False" in args
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    return net.to(device)


class AnnotatedAsymmetricCroCo3DStereo(
    AnnotatedCroCo,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """Annotated version of AsymmetricCroCo3DStereo."""

    @classmethod
    def from_pretrained_naver_DUSt3R_ViTLarge_BaseDecoder_512_dpt(cls, model_to_be_initialized_from):
        # Define key mapping between original and annotated models
        encoder_key_mapping = {
            "patch_embed.proj.weight": "encoder.patch_embed.proj.weight",
            "patch_embed.proj.bias": "encoder.patch_embed.proj.bias",
            "enc_norm.weight": "encoder.norm.weight",
            "enc_norm.bias": "encoder.norm.bias",
        }

        for i in range(24):
            # Attention layers
            encoder_key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = (
                f"encoder.blocks.{i}.query_key_value_projection.weight"
            )
            encoder_key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = (
                f"encoder.blocks.{i}.query_key_value_projection.bias"
            )
            encoder_key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = (
                f"encoder.blocks.{i}.attn.output_projection.weight"
            )
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
            "decoder_embed.weight": "decoder.decoder_embed.weight",
            "decoder_embed.bias": "decoder.decoder_embed.bias",
            "dec_norm.weight": "decoder.norm.weight",
            "dec_norm.bias": "decoder.norm.bias",
            "mask_token": "decoder.mask_token",
        }

        # # Add mappings for each decoder block
        for i in range(12):
            # Source image decoder - Attention layers
            decoder_key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = (
                f"decoder.blocks.{i}.self_attend_query_key_value_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = (
                f"decoder.blocks.{i}.self_attend_query_key_value_projection.bias"
            )
            decoder_key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = (
                f"decoder.blocks.{i}.self_attn.output_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = (
                f"decoder.blocks.{i}.self_attn.output_projection.bias"
            )

            # Source image decoder - Cross attention layers
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

            # Source image decoder - Norm layers
            decoder_key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"decoder.blocks.{i}.norm1.weight"
            decoder_key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"decoder.blocks.{i}.norm1.bias"
            decoder_key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"decoder.blocks.{i}.norm2.weight"
            decoder_key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"decoder.blocks.{i}.norm2.bias"
            decoder_key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"decoder.blocks.{i}.norm3.weight"
            decoder_key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"decoder.blocks.{i}.norm3.bias"
            decoder_key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"decoder.blocks.{i}.norm_y.weight"
            decoder_key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"decoder.blocks.{i}.norm_y.bias"

            # Source image decoder - MLP layers
            decoder_key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"decoder.blocks.{i}.mlp.fc1.weight"
            decoder_key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"decoder.blocks.{i}.mlp.fc1.bias"
            decoder_key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"decoder.blocks.{i}.mlp.fc2.weight"
            decoder_key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"decoder.blocks.{i}.mlp.fc2.bias"

            # Reference image decoder - Attention layers
            decoder_key_mapping[f"dec_blocks2.{i}.attn.qkv.weight"] = (
                f"dec_blocks2.{i}.self_attend_query_key_value_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.attn.qkv.bias"] = (
                f"dec_blocks2.{i}.self_attend_query_key_value_projection.bias"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.attn.proj.weight"] = (
                f"dec_blocks2.{i}.self_attn.output_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.attn.proj.bias"] = (
                f"dec_blocks2.{i}.self_attn.output_projection.bias"
            )

            # Reference image decoder - Cross attention layers
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projq.weight"] = (
                f"dec_blocks2.{i}.cross_attn_query_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projq.bias"] = (
                f"dec_blocks2.{i}.cross_attn_query_projection.bias"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projk.weight"] = (
                f"dec_blocks2.{i}.cross_attn_key_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projk.bias"] = (
                f"dec_blocks2.{i}.cross_attn_key_projection.bias"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projv.weight"] = (
                f"dec_blocks2.{i}.cross_attn_value_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.projv.bias"] = (
                f"dec_blocks2.{i}.cross_attn_value_projection.bias"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.proj.weight"] = (
                f"dec_blocks2.{i}.cross_attn.output_projection.weight"
            )
            decoder_key_mapping[f"dec_blocks2.{i}.cross_attn.proj.bias"] = (
                f"dec_blocks2.{i}.cross_attn.output_projection.bias"
            )

            # Reference image decoder - Norm layers
            decoder_key_mapping[f"dec_blocks2.{i}.norm1.weight"] = f"dec_blocks2.{i}.norm1.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.norm1.bias"] = f"dec_blocks2.{i}.norm1.bias"
            decoder_key_mapping[f"dec_blocks2.{i}.norm2.weight"] = f"dec_blocks2.{i}.norm2.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.norm2.bias"] = f"dec_blocks2.{i}.norm2.bias"
            decoder_key_mapping[f"dec_blocks2.{i}.norm3.weight"] = f"dec_blocks2.{i}.norm3.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.norm3.bias"] = f"dec_blocks2.{i}.norm3.bias"
            decoder_key_mapping[f"dec_blocks2.{i}.norm_y.weight"] = f"dec_blocks2.{i}.norm_y.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.norm_y.bias"] = f"dec_blocks2.{i}.norm_y.bias"

            # Reference image decoder - MLP layers
            decoder_key_mapping[f"dec_blocks2.{i}.mlp.fc1.weight"] = f"dec_blocks2.{i}.mlp.fc1.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.mlp.fc1.bias"] = f"dec_blocks2.{i}.mlp.fc1.bias"
            decoder_key_mapping[f"dec_blocks2.{i}.mlp.fc2.weight"] = f"dec_blocks2.{i}.mlp.fc2.weight"
            decoder_key_mapping[f"dec_blocks2.{i}.mlp.fc2.bias"] = f"dec_blocks2.{i}.mlp.fc2.bias"

        head_key_mapping = {}

        for i in range(1, 3):
            # Downstream head mappings
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.0.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.0.0.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.0.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.0.0.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.1.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.0.1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.0.1.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.0.1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.0.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.1.0.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.0.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.1.0.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.1.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.1.1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.1.1.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.1.1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.2.0.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.2.0.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.2.0.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.2.0.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.0.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.3.0.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.0.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.3.0.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.1.weight"] = (
                f"downstream_head{i}.dpt.act_postprocess.3.1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.act_postprocess.3.1.bias"] = (
                f"downstream_head{i}.dpt.act_postprocess.3.1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.head.0.weight"] = f"downstream_head{i}.dpt.head.0.weight"
            head_key_mapping[f"downstream_head{i}.dpt.head.0.bias"] = f"downstream_head{i}.dpt.head.0.bias"
            head_key_mapping[f"downstream_head{i}.dpt.head.2.weight"] = f"downstream_head{i}.dpt.head.2.weight"
            head_key_mapping[f"downstream_head{i}.dpt.head.2.bias"] = f"downstream_head{i}.dpt.head.2.bias"
            head_key_mapping[f"downstream_head{i}.dpt.head.4.weight"] = f"downstream_head{i}.dpt.head.4.weight"
            head_key_mapping[f"downstream_head{i}.dpt.head.4.bias"] = f"downstream_head{i}.dpt.head.4.bias"
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer1_rn.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer1_rn.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer2_rn.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer2_rn.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer3_rn.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer3_rn.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer4_rn.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer4_rn.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.0.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer_rn.0.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.1.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer_rn.1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.2.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer_rn.2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.layer_rn.3.weight"] = (
                f"downstream_head{i}.dpt.scratch.layer_rn.3.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.out_conv.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit1.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet1.resConfUnit2.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.out_conv.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit1.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet2.resConfUnit2.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.out_conv.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit1.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet3.resConfUnit2.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.out_conv.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit1.conv2.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv1.bias"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.weight"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.weight"
            )
            head_key_mapping[f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.bias"] = (
                f"downstream_head{i}.dpt.scratch.refinenet4.resConfUnit2.conv2.bias"
            )

        model = cls(
            output_mode=model_to_be_initialized_from.output_mode,
            head_type=model_to_be_initialized_from.head_type,
            depth_mode=model_to_be_initialized_from.depth_mode,
            conf_mode=model_to_be_initialized_from.conf_mode,
            freeze=model_to_be_initialized_from.freeze,
            landscape_only=False,
            patch_embed_cls=model_to_be_initialized_from.patch_embed_cls,
            **model_to_be_initialized_from.croco_args,
        )

        load_and_validate_state_dict_with_mapping(
            model, model_to_be_initialized_from, {**encoder_key_mapping, **decoder_key_mapping, **head_key_mapping}
        )

        return model

    def __init__(
        self,
        output_mode: str = "pts3d",
        head_type: str = "linear",
        depth_mode: tuple = ("exp", float("-inf"), float("inf")),
        conf_mode: tuple = ("exp", 1, float("inf")),
        freeze: str = "none",
        landscape_only: bool = True,
        patch_embed_cls: str = "PatchEmbedDust3R",
        **croco_kwargs,
    ):
        # Store the initialization arguments
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        assert "img_size" in self.croco_args, "img_size must be provided"
        self.img_size = self.croco_args["img_size"]

        if isinstance(self.img_size, int):
            self.img_size = (self.img_size, self.img_size)

        if isinstance(self.img_size, list):
            assert len(self.img_size) == 2, "img_size must be a tuple of two integers"
            self.img_size = tuple(self.img_size)

        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.landscape_only = landscape_only
        self.freeze = freeze

        # Initialize the base CroCo model
        super().__init__(**croco_kwargs)

        # Create second decoder by deep copying the first one
        self.dec_blocks2 = deepcopy(self.decoder.blocks)

        # Set up the downstream heads and freeze parameters if needed
        self.set_freeze(freeze)

    def set_freeze(self, freeze: str):
        """Freeze specified parts of the model."""
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.decoder.mask_token],
            "encoder": [self.decoder.mask_token, self.encoder],
        }
        freeze_all_params(to_be_frozen[freeze])

    def set_downstream_head(
        self,
    ):
        """Set up the downstream heads for 3D prediction."""
        assert (
            self.img_size[0] % self.patch_size[0] == 0 and self.img_size[1] % self.patch_size[1] == 0
        ), f"{self.img_size} must be multiple of {self.patch_size=}"

        # Create the downstream heads
        self.downstream_head1 = head_factory(self.head_type, self.output_mode, self, has_conf=bool(self.conf_mode))
        self.downstream_head2 = head_factory(self.head_type, self.output_mode, self, has_conf=bool(self.conf_mode))

        # Wrap heads with landscape transformation if needed
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=self.landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=self.landscape_only)

    def _encode_symmetrized(
        self, view1: Dict[str, Any], view2: Dict[str, Any]
    ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
        print(self.__class__.__name__)
        """Encode image pairs with symmetrization support."""
        img1, img2 = view1["img"], view2["img"]
        B = img1.shape[0]

        # Get true shapes or use image shapes
        shape1 = view1.get("true_shape", torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get("true_shape", torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        if is_symmetrized(view1, view2):
            # Compute half of forward pass for symmetrized inputs
            feat1, pos1, _ = self._encode_image(img1[::2])
            feat2, pos2, _ = self._encode_image(img2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            # Regular forward pass
            feat1, pos1, _ = self._encode_image(img1)
            feat2, pos2, _ = self._encode_image(img2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1: Tensor, pos1: Tensor, f2: Tensor, pos2: Tensor) -> Tuple[list[Tensor], list[Tensor]]:
        """Decode features using both decoders."""
        final_output = [(f1, f2)]

        # Project to decoder dimension
        f1 = self.decoder.decoder_embed(f1)
        f2 = self.decoder.decoder_embed(f2)

        final_output.append((f1, f2))

        # Apply decoder blocks
        for blk1, blk2 in zip(self.decoder.blocks, self.dec_blocks2):
            # Process through first decoder
            f1_new, _ = blk1(final_output[-1][0], final_output[-1][1], pos1, pos2)
            # Process through second decoder
            f2_new, _ = blk2(final_output[-1][1], final_output[-1][0], pos1, pos2)

            final_output.append((f1_new, f2_new))

        # Remove duplicate output and apply normalization
        del final_output[1]
        final_output[-1] = tuple(map(self.decoder.norm, final_output[-1]))
        return zip(*final_output)

    def forward(self, view1: Dict[str, Any], view2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Forward pass computing 3D predictions for both views.

        Args:
            view1: Dictionary containing first view data with keys:
                - img: Tensor of shape (batch_size, 3, height, width) containing the image
                - true_shape: Tensor of shape (batch_size, 2) with actual (height, width) of each image
                - idx: Index identifier for the view (typically 0)
                - instance: String identifier for the instance
            view2: Dictionary containing second view data with the same structure as view1
                  but representing a different viewpoint of the same scene

        Returns:
            Tuple containing two dictionaries with 3D predictions for each view:
            - First dictionary: 3D predictions for view1
            - Second dictionary: 3D predictions for view2, with "pts3d" renamed to
              "pts3d_in_other_view" to indicate they're in view1's reference frame
        """
        # Encode images
        (shape_view1, shape_view2), (features_view1, features_view2), (positions_view1, positions_view2) = (
            self._encode_symmetrized(view1, view2)
        )

        # Decode features
        decoded_view1, decoded_view2 = self._decoder(features_view1, positions_view1, features_view2, positions_view2)

        # Apply downstream heads
        with torch.cuda.amp.autocast(enabled=False):
            results_view1 = self._downstream_head(1, [token.float() for token in decoded_view1], shape_view1)
            results_view2 = self._downstream_head(2, [token.float() for token in decoded_view2], shape_view2)

        # Rename view2's 3D points to indicate they're in view1's frame
        results_view2["pts3d_in_other_view"] = results_view2.pop("pts3d")

        return results_view1, results_view2

    def _downstream_head(self, head_num: int, decout: list[Tensor], img_shape: Tensor) -> Dict[str, Any]:
        """Apply the specified downstream head to decoder output."""
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)
