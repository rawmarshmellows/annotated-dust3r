from typing import Callable

import torch.nn as nn
from icecream import ic

from .attention import Attention, CrossAttention, MultiHeadAttentionV2, MultiHeadAttentionWithRoPEV2
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

    def forward(
        self,
        query_image_tokens,
        reference_image_tokens,
        query_image_token_positions,
        reference_image_token_positions,
    ):
        """
        Forward pass of the decoder block. This block processes input tokens by allowing them to:
        1. Self-attend to each other (self-attention)
        2. Attend to the memory tokens (cross-attention)
        3. Pass through a final MLP layer

        The memory tokens remain constant throughout - they serve as a reference that the input tokens
        can attend to and draw information from, but are never modified themselves. This creates a
        one-way flow of information from memory to input tokens.

        Args:
            query_image_tokens: Tokens to be processed and updated, typically from a masked input image
            reference_image_tokens: Reference tokens that remain unchanged, typically from a reference image

        Returns:
            tuple: (processed_query_image_tokens, unchanged_reference_image_tokens)
                  - processed_query_image_tokens have been updated through self-attention,
                    cross-attention with memory, and MLP processing
                  - reference_image_tokens are returned unchanged as they only serve as a reference
        """
        # Self-attention: Input tokens attend to each other
        # This allows each input token to gather information from other input tokens
        ic(self.__class__.__name__)
        ic(query_image_tokens.shape, reference_image_tokens.shape)

        self_attn_residual = query_image_tokens
        normalized_query_image_tokens = self.norm1(query_image_tokens)

        # Create the projection for the query, key, and value of the query image - in the context of CroCo this is a masked.
        query_key_value_normalised_query_image_tokens = self.self_attend_query_key_value_projection(
            normalized_query_image_tokens
        )
        (
            query_for_normalized_query_image_tokens,
            key_for_normalized_query_image_tokens,
            value_for_normalized_query_image_tokens,
        ) = query_key_value_normalised_query_image_tokens.split(self.embed_dim, dim=-1)

        ic(query_for_normalized_query_image_tokens.shape)
        ic(key_for_normalized_query_image_tokens.shape)
        ic(value_for_normalized_query_image_tokens.shape)
        ic(query_image_token_positions.shape)

        # Perform the self-attention on the query image tokens
        query_image_tokens = self.self_attn(
            query_for_normalized_query_image_tokens,
            key_for_normalized_query_image_tokens,
            value_for_normalized_query_image_tokens,
            query_image_token_positions,
            query_image_token_positions,
        )
        query_image_tokens_with_residual = self_attn_residual + self.drop_path(query_image_tokens)

        # Cross-attention: query image tokens attend to reference image tokens
        normalized_query_image_tokens_with_residual = self.norm2(query_image_tokens_with_residual)
        normalized_reference_image_tokens = self.norm_y(reference_image_tokens)

        # Create the projection for the query - in the context of CroCo from a masked input image -
        # and the key, and value of the reference image, typically from a reference image
        cross_attn_query = self.cross_attn_query_projection(normalized_query_image_tokens_with_residual)

        cross_attn_key = self.cross_attn_key_projection(normalized_reference_image_tokens)
        cross_attn_value = self.cross_attn_value_projection(normalized_reference_image_tokens)

        # Perform the cross-attention, this is where the "filling in" of the masked query image tokens occurs
        cross_attended_input_tokens = self.cross_attn(
            cross_attn_query,  # Query: query image tokens being updated
            cross_attn_key,  # Key: reference image tokens
            cross_attn_value,  # Value: reference image tokens
            query_image_token_positions,
            reference_image_token_positions,
        )

        # Add the residual of the output of the self-attended query image tokens to the output of the cross-attended query image tokens
        cross_attended_input_tokens_with_residual = query_image_tokens_with_residual + self.drop_path(
            cross_attended_input_tokens
        )

        # Final MLP processing of input tokens
        normalized_cross_attended_input_tokens_with_residual = self.norm3(cross_attended_input_tokens_with_residual)
        output = cross_attended_input_tokens_with_residual + self.drop_path(
            self.mlp(normalized_cross_attended_input_tokens_with_residual)
        )

        return output, reference_image_tokens


class TransformerDecoderBlockWithRoPEV2(TransformerDecoderBlockV2):
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
        rope_freq: float = 100.0,
        rope_F0: float = 1.0,
    ):
        super().__init__(
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
        )

        self.self_attn = MultiHeadAttentionWithRoPEV2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_score_dropout_rate=attn_drop_rate,
            output_dropout_rate=proj_drop_rate,
            rope_freq=rope_freq,
            rope_F0=rope_F0,
        )
        self.cross_attn = MultiHeadAttentionWithRoPEV2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_score_dropout_rate=attn_drop_rate,
            output_dropout_rate=proj_drop_rate,
            rope_freq=rope_freq,
            rope_F0=rope_F0,
        )
