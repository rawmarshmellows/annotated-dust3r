from typing import Callable

import torch.nn as nn

from .attention import Attention, MultiHeadAttentionV2
from .drop_path import DropPath
from .mlp import Mlp


class TransformerEncoderBlock(nn.Module):
    """A Transformer encoder block with self-attention and feed-forward layers.

    This block implements the standard Transformer encoder architecture consisting of:
    1. Layer normalization
    2. Multi-head self-attention with optional RoPE
    3. Residual connection
    4. Layer normalization
    5. MLP feed-forward network
    6. Residual connection

    Args:
        embed_dim (int): Input dimension/number of features
        num_heads (int): Number of attention heads
        mlp_ratio (float, optional): Ratio of mlp hidden dim to input dim. Default: 4.0
        qkv_bias (bool, optional): If True, add bias to qkv projection. Default: False
        proj_drop_rate (float, optional): Dropout rate after mlp and projection. Default: 0.0
        attn_drop_rate (float, optional): Dropout rate after attention. Default: 0.0
        path_drop_rate (float, optional): Drop path rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        rope (optional): Rotary position embedding instance. Default: None
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
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop_rate,
        )
        self.norm2 = norm_layer(self.embed_dim)

    def forward(self, input_tensor, pos_encoding):
        """Forward pass of the transformer encoder block.

        Architecture:
                    TransformerEncoderBlock
                               ↓
                    ┌──────────────────────┐
                    │     Input Tensor     │  <- input_tensor
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Layer Norm 1      │  <- self.norm1(input_tensor)
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Multi-Head Self    │
                    │      Attention       │  <- self.attn(normalized, pos_encoding)
                    └──────────┬───────────┘      │
                               │                  │
                               ↓            Skip Connection
                    ┌──────────┴──────────┐       │
                    │          +          ◄───────┘  <- attn_residual + self.drop_path(attended)
                    └──────────┬──────────┘
                               │
                               ↓
                    ┌──────────────────────┐
                    │    Layer Norm 2      │  <- self.norm2(post_attention)
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Feed Forward      │  <- self.mlp(normalized)
                    └──────────┬───────────┘      │
                               │                  │
                               ↓            Skip Connection
                    ┌──────────┴──────────┐       │
                    │          +          ◄───────┘  <- mlp_residual + self.drop_path(transformed)
                    └──────────┬──────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Output Tensor     │  <- output
                    └──────────────────────┘

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (B, N, D) where B is batch size,
                N is sequence length, and D is dimension
            pos_encoding (torch.Tensor): Position encoding tensor of same shape as input_tensor

        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        # First block: Input -> Norm1 -> Attention -> Skip Connection
        attn_residual = input_tensor  # Store input for skip connection
        normalized = self.norm1(input_tensor)  # First normalization layer
        attended = self.attn(normalized, pos_encoding)  # Multi-head self-attention
        post_attention = attn_residual + self.drop_path(attended)  # Skip connection with drop path

        # Second block: Norm2 -> MLP -> Skip Connection
        mlp_residual = post_attention  # Store input for skip connection
        normalized = self.norm2(post_attention)  # Second normalization layer
        transformed = self.mlp(normalized)  # MLP layer
        output = mlp_residual + self.drop_path(transformed)  # Skip connection with drop path

        return output


class TransformerEncoderBlockV2(nn.Module):
    """A Transformer encoder block with self-attention and feed-forward layers.

    This block implements the standard Transformer encoder architecture consisting of:
    1. Layer normalization
    2. Multi-head self-attention with optional RoPE
    3. Residual connection
    4. Layer normalization
    5. MLP feed-forward network
    6. Residual connection

    Args:
        embed_dim (int): Input dimension/number of features
        num_heads (int): Number of attention heads
        mlp_ratio (float, optional): Ratio of mlp hidden dim to input dim. Default: 4.0
        qkv_bias (bool, optional): If True, add bias to qkv projection. Default: False
        proj_drop_rate (float, optional): Dropout rate after mlp and projection. Default: 0.0
        attn_drop_rate (float, optional): Dropout rate after attention. Default: 0.0
        path_drop_rate (float, optional): Drop path rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        rope (optional): Rotary position embedding instance. Default: None
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(self.embed_dim)
        self.query_key_value_projection = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)
        self.attn = MultiHeadAttentionV2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_score_dropout_rate=attn_drop_rate,
            output_dropout_rate=proj_drop_rate,
        )
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=self.embed_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop_rate,
        )
        self.norm2 = norm_layer(self.embed_dim)

    def forward(self, input_tensor):
        # First block: Input -> Norm1 -> Attention -> Skip Connection
        attn_residual = input_tensor  # Store input for skip connection
        normalized = self.norm1(input_tensor)  # First normalization layer

        # Create query, key, value tensors
        # query_key_value shape: (batch_size, seq_len, 3*embed_dim)
        query_key_value = self.query_key_value_projection(normalized)

        # Split into query, key, value
        query, key, value = query_key_value.split(self.embed_dim, dim=-1)

        attended = self.attn(query, key, value)  # Multi-head self-attention
        post_attention = attn_residual + self.drop_path(attended)  # Skip connection with drop path

        # Second block: Norm2 -> MLP -> Skip Connection
        mlp_residual = post_attention  # Store input for skip connection
        normalized = self.norm2(post_attention)  # Second normalization layer
        transformed = self.mlp(normalized)  # MLP layer
        output = mlp_residual + self.drop_path(transformed)  # Skip connection with drop path

        return output
