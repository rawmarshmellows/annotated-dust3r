import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import huggingface_hub
import torch
import torch.nn as nn
from torch import Tensor

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

        import pdb

        pdb.set_trace()

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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        import pdb

        pdb.set_trace()
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(AnnotatedAsymmetricCroCo3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                print(e)
                raise Exception(f"tried to load {pretrained_model_name_or_path} from huggingface, but failed")
            return model

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
            self.img_size[0] % self.patch_size == 0 and self.img_size[1] % self.patch_size == 0
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
        """Forward pass computing 3D predictions for both views."""
        # Encode images
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # Decode features
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        # Apply downstream heads
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        # Rename view2's 3D points to indicate they're in view1's frame
        res2["pts3d_in_other_view"] = res2.pop("pts3d")

        return res1, res2

    def _downstream_head(self, head_num: int, decout: list[Tensor], img_shape: Tensor) -> Dict[str, Any]:
        """Apply the specified downstream head to decoder output."""
        head = getattr(self, f"head{head_num}")
        return head(decout, img_shape)
