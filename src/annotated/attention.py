from typing import Optional

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention module that allows attention between different sets of tokens.

    Cross-attention is a mechanism where one set of tokens (queries) attends to another set
    of tokens (keys/values). Unlike self-attention where tokens attend to themselves,
    cross-attention enables interaction between different sequences, making it crucial for
    tasks like encoder-decoder architectures.

    Args:
        query_tokens: Input tokens that will attend to the key/value tokens
                     Shape: (batch_size, num_queries, embedding_dim)
        key_tokens: Tokens that will be used to compute attention scores with queries
                   Shape: (batch_size, num_keys, embedding_dim)
        value_tokens: Tokens that will be aggregated based on attention scores
                     Shape: (batch_size, num_values, embedding_dim)
        query_positions: Position encodings for query tokens, used with RoPE if provided
                       Shape: matches query_tokens
        key_positions: Position encodings for key tokens, used with RoPE if provided
                      Shape: matches key_tokens
    """

    def __init__(
        self,
        embed_dim,
        rope=None,
        num_heads=8,
        qkv_bias=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
    ):
        """Initialize the cross-attention module.

        Args:
            embedding_dim: Dimension of the token embeddings
            rope: Optional rotary position embedding module
            num_attention_heads: Number of parallel attention heads
            use_bias: Whether to include bias terms in linear projections
            attention_dropout: Dropout rate for attention weights
            projection_dropout: Dropout rate for final output projection
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.attention_scale = head_dim**-0.5

        # Separate projection layers for queries, keys and values
        self.query_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.key_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.value_projection = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        # Dropout layers
        self.attention_dropout = nn.Dropout(attn_drop_rate)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(proj_drop_rate)

        # Rotary position embedding for incorporating positional information
        self.rope = rope

    def forward(self, query_tokens, key_tokens, value_tokens, query_positions, key_positions):
        """
        Forward pass of cross-attention.

        The cross-attention mechanism works in several steps:
        1. Project inputs to queries, keys, and values
        2. Apply position encodings (if RoPE is used), if not it would have already been applied
        3. Compute attention scores between queries and keys
        4. Use these scores to create a weighted sum of values

        Args:
            query_tokens: Tokens that will attend to key/value tokens (e.g., decoder states)
                         Shape: (batch_size, num_query_patches, embedding_dim)
            key_tokens: Tokens used to assess relevance (e.g., encoder states)
                       Shape: (batch_size, num_key_patches, embedding_dim)
            value_tokens: Tokens that will be aggregated based on attention weights (typically encoder outputs)
                         Shape: (batch_size, num_value_patches, embedding_dim)
            query_positions: Position information for query tokens
                           Shape: (batch_size, num_query_patches)
            key_positions: Position information for key tokens
                          Shape: (batch_size, num_key_patches)
        """
        # Extract shapes for clarity
        batch_size, num_query_patches, channels = query_tokens.shape
        num_key_patches = key_tokens.shape[1]
        num_value_patches = value_tokens.shape[1]

        # Project and reshape to multi-head format
        # For each projection, shape becomes: (batch_size, num_heads, sequence_length, head_dimension)
        queries = (
            self.query_projection(query_tokens)
            .reshape(batch_size, num_query_patches, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        keys = (
            self.key_projection(key_tokens)
            .reshape(batch_size, num_key_patches, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        values = (
            self.value_projection(value_tokens)
            .reshape(batch_size, num_value_patches, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        # Apply rotary position embeddings if provided
        # RoPE helps model consider relative positions of tokens
        if self.rope is not None:
            queries = self.rope(queries, query_positions)
            keys = self.rope(keys, key_positions)

        # === CROSS ATTENTION COMPUTATION ===
        # 1. Compute attention scores: how much each query should attend to each key
        # Shape: (batch_size, num_heads, num_queries, num_keys)
        attention_scores = (queries @ keys.transpose(-2, -1)) * self.attention_scale

        # 2. Convert scores to probabilities with softmax
        # This determines how much each query will "focus" on different keys
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # 3. Use attention weights to aggregate values
        # This creates a weighted sum of values for each query
        # Shape: (batch_size, num_queries, embedding_dim)
        attended_values = (attention_weights @ values).transpose(1, 2).reshape(batch_size, num_query_patches, channels)

        # Final projection and dropout
        output = self.output_projection(attended_values)
        output = self.output_dropout(output)

        return output


class Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.query_key_value = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=self.qkv_bias)

        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.rope = rope
        self.mha = MultiHeadAttention(num_heads=num_heads, attn_drop=attn_drop_rate)

    def forward(self, x, xpos):
        # x shape: (batch_size=batch_size, seq_len=seq_len, embed_dim=embed_dim)
        batch_size, seq_len, embed_dim = x.shape

        # Project input to query, key, value
        # query_key_value shape: (batch_size, seq_len, 3*embed_dim)
        query_key_value = self.query_key_value(x)
        # Apply RoPE if provided
        if self.rope is not None:
            # Reshape for RoPE: (batch_size, seq_len, 3, num_heads, embed_dim/num_heads) -> (batch_size, num_heads, seq_len, 3, embed_dim/num_heads)
            query_key_value = query_key_value.reshape(
                batch_size, seq_len, 3, self.num_heads, embed_dim // self.num_heads
            ).transpose(1, 3)

            q, k, v = [
                query_key_value[:, :, i] for i in range(3)
            ]  # Each shape (batch_size, num_heads, seq_len, embed_dim/num_heads)

            q = self.rope(q, xpos)
            k = self.rope(k, xpos)
            # Reshape back to (batch_size, seq_len, 3*embed_dim)
            query_key_value = torch.stack([q, k, v], dim=2).transpose(1, 3).reshape(batch_size, seq_len, 3 * embed_dim)
        # Multi-head attention
        out = self.mha(query_key_value)  # Shape: (batch_size, num_heads, seq_len, embed_dim/num_heads)

        # Reshape and project output back to (batch_size, seq_len, embed_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query_key_value):
        # query_key_value shape: (batch_size, seq_length, 3 * embedding_dim)
        batch_size, seq_length, projected_dim = query_key_value.shape
        embedding_dim = projected_dim // 3

        # Validate dimensions
        assert (
            embedding_dim % self.num_heads == 0
        ), f"Embedding dimension {embedding_dim} must be divisible by number of heads {self.num_heads}"
        assert (
            embedding_dim * 3 == projected_dim
        ), f"Projected dimension {projected_dim} must be 3x embedding dimension {embedding_dim}"

        head_dim = embedding_dim // self.num_heads
        scale = head_dim**-0.5

        # Reshape and transpose to match reference implementation
        query_key_value = query_key_value.reshape(batch_size, seq_length, 3, self.num_heads, head_dim).transpose(1, 3)

        # Split into q,k,v - each shape (batch_size, num_heads, seq_length, head_dim)
        # Split QKV into separate tensors
        # Each has shape (batch_size, num_heads, seq_length, head_dim)
        q, k, v = [query_key_value[:, :, i] for i in range(3)]

        # Compute attention scores by multiplying Q with K^T
        # (batch_size, num_heads, seq_length, head_dim) @ (batch_size, num_heads, head_dim, seq_length)
        # = (batch_size, num_heads, seq_length, seq_length)
        attention_scores = q @ k.transpose(-2, -1)

        # Scale attention scores by sqrt(head_dim)
        attention_scores = attention_scores * scale

        # Apply softmax to get attention probabilities
        attention_probs = attention_scores.softmax(dim=-1)

        # Apply dropout to attention probabilities
        attention_probs = self.attn_drop(attention_probs)

        # Multiply attention probabilities with values
        # (batch_size, num_heads, seq_length, seq_length) @ (batch_size, num_heads, seq_length, head_dim)
        # = (batch_size, num_heads, seq_length, head_dim)
        out = attention_probs @ v

        return out