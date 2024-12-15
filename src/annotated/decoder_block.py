from functools import partial
from typing import Callable

import torch.nn as nn

from src.annotated.attention import Attention, CrossAttention
from src.annotated.drop_path import DropPath
from src.annotated.mlp import Mlp


class TransformerDecoderBlock(nn.Module):
    """
    A transformer decoder block that performs self-attention, cross-attention, and feed-forward operations.

    Args:
        embed_dim: Dimension of the input embeddings
        num_heads: Number of attention heads
        mlp_ratio: Ratio to determine hidden dimension size of MLP
        qkv_bias: Whether to include bias in query/key/value projections
        proj_drop_rate: Dropout rate for projection layers
        attn_drop_rate: Dropout rate for attention
        path_drop_rate: Dropout rate for DropPath
        act_layer: Activation layer class
        norm_layer: Normalization layer class
        norm_mem: Whether to normalize memory input
        rope: Rotary position embedding module
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        path_drop_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        norm_mem: bool = True,
        rope: nn.Module = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(self.embed_dim)
        self.attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            rope=rope,
        )
        self.cross_attn = CrossAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            rope=rope,
        )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(self.embed_dim)
        self.norm3 = norm_layer(self.embed_dim)
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop_rate,
        )
        self.norm_y = norm_layer(self.embed_dim) if norm_mem else nn.Identity()

    def forward(self, query_input, memory_input, query_pos, memory_pos):
        """
        Forward pass of the decoder block.

        Args:
            query_input: Input tensor for self-attention query
            memory_input: Input tensor from encoder (memory)
            query_pos: Positional encoding for query
            memory_pos: Positional encoding for memory

        Returns:
            tuple: (processed query tensor, unchanged memory tensor)
        """
        # Self attention
        query_input = query_input + self.drop_path(self.attn(self.norm1(query_input), query_pos))

        # Cross attention with normalized memory
        normalized_memory = self.norm_y(memory_input)

        query_input = query_input + self.drop_path(
            self.cross_attn(self.norm2(query_input), normalized_memory, normalized_memory, query_pos, memory_pos)
        )

        # MLP block
        query_input = query_input + self.drop_path(self.mlp(self.norm3(query_input)))

        return query_input, memory_input
