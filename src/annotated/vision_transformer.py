from functools import partial
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.annotated.encoder_block import TransformerEncoderBlock
from src.annotated.mask_generator import RandomMask
from src.annotated.patch_embed import PatchEmbed
from src.annotated.positional_embedding import construct_coordinate_grid_2d, get_2d_sincos_pos_embed_from_grid


class VisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        mask_ratio: float = 0.9,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        embed_norm_layer: Callable[..., nn.Module] = None,
        pos_embed_type: str = "sincos2d",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.norm_layer = norm_layer
        self.pos_embed_type = pos_embed_type

        if self.pos_embed_type == "sincos2d":
            # (n_patches, embed_dim)
            coordinate_grid_2d = construct_coordinate_grid_2d(
                grid_size_x=int(self.num_patches**0.5), grid_size_y=int(self.num_patches**0.5)
            )
            enc_pos_embed = get_2d_sincos_pos_embed_from_grid(self.embed_dim, coordinate_grid_2d)
            self.register_buffer("enc_pos_embed", torch.from_numpy(enc_pos_embed.astype(np.float32)).float())
        else:
            raise NotImplementedError(f"Positional embedding {self.pos_embed_type} not implemented")

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            norm_layer=embed_norm_layer,
            flatten=True,
        )

        self.mask_generator = RandomMask(self.num_patches, self.mask_ratio)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop_rate=proj_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=path_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = norm_layer(embed_dim)

    def forward(
        self, x: Tensor, do_mask: bool = False, return_all_blocks: bool = False
    ) -> tuple[Tensor, Tensor] | Tensor:
        """Forward pass of the Vision Transformer Encoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            do_mask: Whether to apply masking
            return_all_blocks: Whether to return intermediate features

        Returns:
            If return_all_blocks:
                Tuple of (final_output, intermediate_features)
            Else:
                final_output
        """
        # 1. Create patches from input image
        x, pos_encodings = self.patch_embed(x)
        batch_size, num_patches, embed_dim = x.shape
        assert num_patches == self.num_patches, f"Expected {self.num_patches} patches, got {num_patches}"

        # 2. Add positional embeddings
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]

        # 3. Apply masking if requested
        if do_mask:
            masks = self.mask_generator(x)  # True indicates masked tokens
            # Keep only unmasked tokens (~masks inverts the mask)
            x = x[~masks].view(batch_size, -1, embed_dim)
            pos_encodings = pos_encodings[~masks].view(batch_size, -1, 2)
        else:
            masks = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=x.device)
            pos_encodings = pos_encodings

        # 4. Apply transformer encoder blocks
        if return_all_blocks:
            features = []
            for blk in self.blocks:
                x = blk(x, pos_encodings)
                features.append(x)
            features[-1] = self.norm(features[-1])
            return features, pos_encodings, masks
        else:
            for blk in self.blocks:
                x = blk(x, pos_encodings)
            x = self.norm(x)
            return x, pos_encodings, masks
