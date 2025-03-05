from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .patchify import patchify, unpatchify
from .vision_transformer import VisionTransformerDecoderV2, VisionTransformerEncoderV2


class AnnotatedCroCo(nn.Module):
    def __init__(
        self,
        img_size: int = 224,  # input image size
        patch_size: int = 16,  # patch_size
        mask_ratio: float = 0.9,  # ratios of masked tokens
        enc_embed_dim: int = 768,  # encoder feature dimension
        enc_depth: int = 12,  # encoder depth
        enc_num_heads: int = 12,  # encoder number of heads in the transformer block
        dec_embed_dim: int = 512,  # decoder feature dimension
        dec_depth: int = 8,  # decoder depth
        dec_num_heads: int = 16,  # decoder number of heads in the transformer block
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        norm_im2_in_dec: bool = True,  # whether to apply normalization of the 'memory' = (second image) in the decoder
        pos_embed: str = "cosine",  # positional embedding (either cosine or RoPE100)
    ):
        super().__init__()

        self.patch_size = patch_size

        # Create encoder
        self.encoder = VisionTransformerEncoderV2(
            img_size=img_size,
            patch_size=patch_size,
            mask_ratio=mask_ratio,
            in_channels=3,
            embed_dim=enc_embed_dim,
            num_layers=enc_depth,
            num_heads=enc_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            path_drop_rate=0.0,
            norm_layer=norm_layer,
            embed_norm_layer=None,  # Match CroCo's lack of normalization in patch embed
            pos_embed_type="sincos2d" if pos_embed == "cosine" else pos_embed,
        )

        # Create decoder
        self.decoder = VisionTransformerDecoderV2(
            patch_size=patch_size,
            enc_embed_dim=enc_embed_dim,
            embed_dim=dec_embed_dim,
            num_heads=dec_num_heads,
            num_layers=dec_depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            path_drop_rate=0.0,
            norm_layer=norm_layer,
            norm_mem=norm_im2_in_dec,
            pos_embed_type="sincos2d" if pos_embed == "cosine" else pos_embed,
            img_size=img_size,
        )

        # prediction head
        self.set_downstream_head()

    def set_downstream_head(self):
        """Set up the downstream head for the model."""
        n_colors = 3
        n_pixels_in_patch = self.patch_size**2
        self.prediction_head = nn.Linear(self.decoder.embed_dim, n_pixels_in_patch * n_colors, bias=True)

        # Initialize prediction head
        if isinstance(self.prediction_head, nn.Linear):
            torch.nn.init.xavier_uniform_(self.prediction_head.weight)
            if self.prediction_head.bias is not None:
                nn.init.constant_(self.prediction_head.bias, 0)

    def _encode_image(
        self, image: Tensor, do_mask: bool = False, return_all_blocks: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Encode an image through the encoder.

        Args:
            image: Input image of shape (B, 3, H, W)
            do_mask: Whether to perform masking or not
            return_all_blocks: If True, return features from all blocks instead of just the last block

        Returns:
            tuple:
                - encoded_features: Encoded features of shape (B, N, C) or list of such features if return_all_blocks
                - positions: Position encodings of shape (B, N, 2)
                - masks: Boolean mask of shape (B, N) indicating which patches were masked'
            where N is the number of patches in the image.
        """
        return self.encoder(image, do_mask=do_mask, return_all_blocks=return_all_blocks)

    def _decoder(
        self,
        feat1: Tensor,
        pos1: Tensor,
        masks1: Optional[Tensor],
        feat2: Tensor,
        pos2: Tensor,
        return_all_blocks: bool = False,
    ) -> Tensor | list[Tensor]:
        """
        Decode features through the decoder.

        Args:
            feat1: Features from masked image of shape (B, N1, C)
            pos1: Position encodings for masked image of shape (B, N1, 2)
            masks1: Boolean mask for masked image of shape (B, N), can be None for downstream tasks
            feat2: Features from reference image of shape (B, N2, C)
            pos2: Position encodings for reference image of shape (B, N2, 2)
            return_all_blocks: If True, return features from all blocks instead of just the last block

        Returns:
            Decoded features of shape (B, N, C) or list of such features if return_all_blocks
        """
        return self.decoder(feat1, pos1, masks1, feat2, pos2, return_all_blocks=return_all_blocks)

    def patchify(self, imgs: Tensor) -> Tensor:
        """
        Convert images to patches.

        Args:
            imgs: Input images of shape (B, 3, H, W)

        Returns:
            Patches of shape (B, L, patch_size**2 * 3)
        """
        return patchify(imgs, self.patch_size)[0]  # Return only the patches

    def unpatchify(self, x: Tensor, channels: int = 3) -> Tensor:
        """
        Convert patches back to images.

        Args:
            x: Input patches of shape (B, L, patch_size**2 * channels)
            channels: Number of channels in the output image

        Returns:
            Images of shape (B, channels, H, W)
        """
        return unpatchify(x, self.patch_size, channels=channels)[0]  # Return only the images

    def forward(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of the model.

        Args:
            img1: First input image of shape (B, 3, H, W)
            img2: Second input image of shape (B, 3, H, W)

        Returns:
            tuple:
                - output: Reconstructed patches of shape (B, N, patch_size**2 * 3)
                - mask: Boolean mask indicating which patches were masked
                - target: Ground truth patches from img1
        """
        # encoder of the masked first image
        feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
        # return feat1, pos1, mask1

        # encoder of the second image
        feat2, pos2, _ = self._encode_image(img2, do_mask=False)
        # decoder
        decfeat = self._decoder(feat1, pos1, mask1, feat2, pos2)
        # prediction head
        out = self.prediction_head(decfeat)
        # get target
        target = self.patchify(img1)
        return out, mask1, target
