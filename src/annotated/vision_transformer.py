from functools import partial
from typing import Callable, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.annotated.decoder_block import TransformerDecoderBlock
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
        qkv_bias: bool = False,
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


class VisionTransformerDecoder(nn.Module):
    """Vision Transformer Decoder.

    Args:
        num_patches: Number of patches in the input sequence
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
        qkv_bias: If True, add a learnable bias to query, key, value
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        norm_layer: Normalization layer
        act_layer: Activation layer
    """

    def __init__(
        self,
        patch_size: int,
        enc_embed_dim: int,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        norm_mem: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        pos_embed_type: str = "sincos2d",
        rope: nn.Module = None,
        img_size: int = None,
        num_patches: int = None,
    ):
        super().__init__()

        # make sure either img_size or num_patches is defined, but not both
        if not ((img_size is None) ^ (num_patches is None)):
            raise ValueError(
                f"Exactly one of img_size or num_patches must be defined, got img_size={img_size}, num_patches={num_patches}"
            )

        if num_patches is None:
            self.num_patches: int = (img_size // patch_size) ** 2
        else:
            self.num_patches: int = num_patches

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)
        self.pos_embed_type = pos_embed_type
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if self.pos_embed_type == "sincos2d":
            # (n_patches, embed_dim)
            coordinate_grid_2d = construct_coordinate_grid_2d(
                grid_size_x=int(self.num_patches**0.5), grid_size_y=int(self.num_patches**0.5)
            )
            enc_pos_embed = get_2d_sincos_pos_embed_from_grid(self.embed_dim, coordinate_grid_2d)
            self.register_buffer("dec_pos_embed", torch.from_numpy(enc_pos_embed.astype(np.float32)).float())
        else:
            raise NotImplementedError(f"Positional embedding {self.pos_embed_type} not implemented")

        # Create decoder blocks
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop_rate=proj_drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    path_drop_rate=path_drop_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.num_patches = num_patches

    def forward(
        self,
        masked_image_tokens: Tensor,
        masked_image_pos: Tensor,
        masked_image_mask: Tensor | None,
        reference_image_tokens: Tensor,
        reference_image_pos: Tensor,
        return_all_blocks: bool = False,
    ) -> Tensor | list[Tensor]:
        """Forward pass of the Vision Transformer Decoder.

        Args:
            masked_image: Image to be reconstructed of shape (batch_size, num_patches, embed_dim)
            masked_image_mask: Boolean mask indicating which patches to reconstruct in masked_image (True = masked)
            masked_image_pos: Positional encodings for masked image
            reference_image: Reference image from other perspective of shape (batch_size, num_patches, embed_dim)
            reference_image_pos: Positional encodings for reference image
            return_all_blocks: If True, return features from all blocks instead of just the last block

        Returns:
            If return_all_blocks is False:
                Output tensor of shape (batch_size, num_patches, embed_dim)
            If return_all_blocks is True:
                List of output tensors from each block
        """
        print(f"masked_image_tokens shape: {masked_image_tokens.shape}")
        print(f"masked_image_pos shape: {masked_image_pos.shape}")
        print(f"masked_image_mask shape: {masked_image_mask.shape}")
        print(f"reference_image_tokens shape: {reference_image_tokens.shape}")
        print(f"reference_image_pos shape: {reference_image_pos.shape}")
        print(f"return_all_blocks: {return_all_blocks}")
        masked_image_tokens_reembedded = self.decoder_embed(masked_image_tokens)
        reference_image_tokens_reembedded = self.decoder_embed(reference_image_tokens)

        print(f"masked_image_tokens_reembedded shape: {masked_image_tokens_reembedded.shape}")
        print(f"reference_image_tokens_reembedded shape: {reference_image_tokens_reembedded.shape}")

        batch_size, num_patches, embed_dim = masked_image_tokens_reembedded.size()

        if masked_image_mask is None:
            only_unmasked_image_tokens_reembedded = masked_image_tokens_reembedded
        else:
            print(f"masked_image_mask shape: {masked_image_mask.shape}")
            only_unmasked_image_tokens_reembedded = self.mask_token.repeat(
                batch_size, masked_image_mask.size(1), 1
            ).to(dtype=masked_image_tokens_reembedded.dtype)
            only_unmasked_image_tokens_reembedded[~masked_image_mask] = masked_image_tokens_reembedded.view(
                batch_size * num_patches, embed_dim
            )

        print(f"only_unmasked_image_tokens_reembedded shape: {only_unmasked_image_tokens_reembedded.shape}")

        # Add positional embeddings if provided
        if self.dec_pos_embed is not None:
            print(f"dec_pos_embed shape: {self.dec_pos_embed.shape}")
            only_unmasked_image_tokens_reembedded = only_unmasked_image_tokens_reembedded + self.dec_pos_embed
            reference_image_tokens_reembedded = reference_image_tokens_reembedded + self.dec_pos_embed

        # Apply decoder blocks
        only_unmasked_image_tokens_for_block = only_unmasked_image_tokens_reembedded
        reference_image_tokens_for_block = reference_image_tokens_reembedded

        print(f"reference_image_tokens_for_block shape: {reference_image_tokens_for_block.shape}")
        print(f"only_unmasked_image_tokens_for_block shape: {only_unmasked_image_tokens_for_block.shape}")

        if return_all_blocks:
            outputs = []
            for i, blk in enumerate(self.blocks):
                only_unmasked_image_tokens_for_block, reference_image_tokens_for_block = blk(
                    only_unmasked_image_tokens_for_block,
                    reference_image_tokens_for_block,
                    masked_image_pos,
                    reference_image_pos,
                )
                outputs.append(only_unmasked_image_tokens_for_block)
            outputs[-1] = self.norm(outputs[-1])
            print(f"Final normalized output shape: {outputs[-1].shape}")
            return outputs

        for i, blk in enumerate(self.blocks):
            only_unmasked_image_tokens_for_block, reference_image_tokens_for_block = blk(
                only_unmasked_image_tokens_for_block,
                reference_image_tokens_for_block,
                masked_image_pos,
                reference_image_pos,
            )
            print(f"Block {i} output shapes:")
            print(f"  only_unmasked_image_tokens_for_block: {only_unmasked_image_tokens_for_block.shape}")
            print(f"  reference_image_tokens_for_block: {reference_image_tokens_for_block.shape}")
        only_unmasked_image_tokens_for_block = self.norm(only_unmasked_image_tokens_for_block)
        print(f"Final normalized output shape: {only_unmasked_image_tokens_for_block.shape}")
        return only_unmasked_image_tokens_for_block
