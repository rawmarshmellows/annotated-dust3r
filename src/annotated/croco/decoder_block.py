from typing import Callable

import torch.nn as nn

from .attention import Attention, CrossAttention, MultiHeadAttentionV2
from .drop_path import DropPath
from .mlp import Mlp


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


class TransformerDecoderBlockV2(nn.Module):
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = norm_layer(self.embed_dim)
        self.self_attend_query_key_value_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=qkv_bias)
        self.self_attn = MultiHeadAttentionV2(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            attention_score_dropout_rate=attn_drop_rate,
            output_dropout_rate=proj_drop_rate,
        )
        self.cross_attn_query_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.cross_attn_key_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.cross_attn_value_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.cross_attn = MultiHeadAttentionV2(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            attention_score_dropout_rate=attn_drop_rate,
            output_dropout_rate=proj_drop_rate,
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

    def forward(self, input_tokens, memory_tokens, query_pos, memory_pos):
        """
        Forward pass of the decoder block. This block processes input tokens by allowing them to:
        1. Self-attend to each other (self-attention)
        2. Attend to the memory tokens (cross-attention)
        3. Pass through a final MLP layer

        The memory tokens remain constant throughout - they serve as a reference that the input tokens
        can attend to and draw information from, but are never modified themselves. This creates a
        one-way flow of information from memory to input tokens.

        Args:
            input_tokens: Tokens to be processed and updated, typically from a masked input image
            memory_tokens: Reference tokens that remain unchanged, typically from a reference image
            query_pos: Positional encoding for input tokens (currently unused)
            memory_pos: Positional encoding for memory tokens (currently unused)

        Returns:
            tuple: (processed_input_tokens, unchanged_memory_tokens)
                  - processed_input_tokens have been updated through self-attention,
                    cross-attention with memory, and MLP processing
                  - memory_tokens are returned unchanged as they only serve as a reference
        """
        # Self-attention: Input tokens attend to each other
        # This allows each input token to gather information from other input tokens
        self_attn_residual = input_tokens
        normalized_input_tokens = self.norm1(input_tokens)
        query_key_value_input_tokens = self.self_attend_query_key_value_projection(normalized_input_tokens)
        query_for_input_tokens, key_for_input_tokens, value_for_input_tokens = query_key_value_input_tokens.split(
            self.embed_dim, dim=-1
        )
        self_attented_input_tokens = self.self_attn(
            query_for_input_tokens, key_for_input_tokens, value_for_input_tokens
        )
        self_attented_input_tokens_with_residual = self_attn_residual + self.drop_path(self_attented_input_tokens)

        # Cross-attention: Input tokens attend to memory tokens
        # Memory tokens are only used as key/value pairs - they provide information but aren't modified
        normalized_self_attented_input_tokens_with_residual = self.norm2(self_attented_input_tokens_with_residual)
        normalized_memory_tokens = self.norm_y(memory_tokens)

        cross_attn_query = self.cross_attn_query_projection(normalized_self_attented_input_tokens_with_residual)
        cross_attn_key = self.cross_attn_key_projection(normalized_memory_tokens)
        cross_attn_value = self.cross_attn_value_projection(normalized_memory_tokens)
        cross_attented_input_tokens = self.cross_attn(
            cross_attn_query,  # Query: input tokens being updated
            cross_attn_key,  # Key: reference memory tokens
            cross_attn_value,  # Value: reference memory tokens
        )
        cross_attented_input_tokens_with_residual = self_attented_input_tokens_with_residual + self.drop_path(
            cross_attented_input_tokens
        )

        # Final MLP processing of input tokens
        # Memory tokens remain untouched while input tokens get final processing
        normalized_cross_attented_input_tokens_with_residual = self.norm3(cross_attented_input_tokens_with_residual)
        output = cross_attented_input_tokens_with_residual + self.drop_path(
            self.mlp(normalized_cross_attented_input_tokens_with_residual)
        )

        return output, memory_tokens
