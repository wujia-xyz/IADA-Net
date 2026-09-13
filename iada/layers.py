"""Depth aggregation operators shared with the original DABI-Net implementation."""
import math
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

class SimpleDepthEncoding(nn.Module):
    """
    Simple depth encoding without physics assumptions.
    Uses learnable embeddings + sinusoidal positional encoding.
    """

    def __init__(self, embed_dim: int = 64, num_depths: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_depths = num_depths

        # Learnable depth embeddings
        self.depth_embed = nn.Embedding(num_depths, embed_dim // 2)

        # Sinusoidal encoding frequencies (fixed)
        self.register_buffer(
            'freq_bands',
            torch.linspace(1.0, num_depths / 2, embed_dim // 4)
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim // 2 + embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

    def forward(self, depth_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_indices: [N] integer depth indices (0 to num_depths-1)
        Returns:
            depth_encoding: [N, embed_dim]
        """
        # Normalized depth (0 to 1)
        depth_normalized = depth_indices.float() / (self.num_depths - 1)

        # Learnable embedding
        learned = self.depth_embed(depth_indices)  # [N, embed_dim//2]

        # Sinusoidal encoding
        depth_expanded = depth_normalized.unsqueeze(-1)  # [N, 1]
        sin_features = torch.sin(depth_expanded * self.freq_bands * math.pi)
        cos_features = torch.cos(depth_expanded * self.freq_bands * math.pi)
        sinusoidal = torch.cat([sin_features, cos_features], dim=-1)

        # Combine
        combined = torch.cat([learned, sinusoidal], dim=-1)
        encoding = self.out_proj(combined)

        return encoding

class AttentionRowPooling(nn.Module):
    """
    Attention-based pooling for aggregating patch features within a row.
    """

    def __init__(self, feature_dim: int = 768, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim

        # Query for attention (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Key and Value projections
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = feature_dim ** -0.5

    def forward(self, row_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            row_features: [B, num_rows, patches_per_row, feature_dim]
        Returns:
            pooled: [B, num_rows, feature_dim]
        """
        B, num_rows, patches_per_row, D = row_features.shape

        # Reshape for attention: [B * num_rows, patches_per_row, D]
        x = row_features.reshape(B * num_rows, patches_per_row, D)

        # Expand query for batch
        query = self.query.expand(B * num_rows, -1, -1)

        # Compute key and value
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Attention scores
        attn = torch.bmm(query, key.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum
        out = torch.bmm(attn, value)
        out = self.out_proj(out.squeeze(1))

        return out.reshape(B, num_rows, D)

class DepthAwareAttention(nn.Module):
    """
    Depth-Aware Attention mechanism.

    Attention weights = feature similarity + depth bias
    Depth bias is learned by MLP, not physics formula.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_depths: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # Depth bias network: learns relationship between depth pairs
        # Input: [depth_i, depth_j, |depth_i - depth_j|] normalized
        self.depth_bias_net = nn.Sequential(
            nn.Linear(3, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, num_heads)
        )

        # Pre-compute depth pair features for efficiency
        self.num_depths = num_depths
        self._precompute_depth_pairs()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def _precompute_depth_pairs(self):
        """Pre-compute depth pair features [i, j, |i-j|] for all pairs."""
        num_depths = self.num_depths
        # Create all pairs
        i_idx = torch.arange(num_depths).unsqueeze(1).expand(-1, num_depths)
        j_idx = torch.arange(num_depths).unsqueeze(0).expand(num_depths, -1)

        # Normalize to [0, 1]
        i_norm = i_idx.float() / (num_depths - 1)
        j_norm = j_idx.float() / (num_depths - 1)
        diff_norm = (i_idx - j_idx).abs().float() / (num_depths - 1)

        # Stack: [num_depths, num_depths, 3]
        depth_pairs = torch.stack([i_norm, j_norm, diff_norm], dim=-1)
        self.register_buffer('depth_pairs', depth_pairs)

    def forward(self, x: torch.Tensor, depth_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] where N = num_depths
            depth_indices: not used (for API compatibility)
        Returns:
            output: [B, N, D]
        """
        B, N, D = x.shape

        # Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores from features
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add depth bias
        depth_bias = self.depth_bias_net(self.depth_pairs)  # [N, N, H]
        depth_bias = depth_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, N, N]
        attn = attn + depth_bias

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = torch.matmul(attn, v)  # [B, H, N, D_head]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(x + out)

        return out

class BidirectionalInteractionLayer(nn.Module):
    """
    Bidirectional Interaction Layer.

    Top-Down and Bottom-Up paths with cross-attention interaction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_depths: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        # Depth-aware attention for each direction
        self.td_attn = DepthAwareAttention(dim, num_heads, num_depths, dropout)
        self.bu_attn = DepthAwareAttention(dim, num_heads, num_depths, dropout)

        # Cross-attention for interaction
        self.cross_attn_td = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_bu = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Layer norms for cross-attention
        self.norm_td = nn.LayerNorm(dim)
        self.norm_bu = nn.LayerNorm(dim)

        # FFN for each direction
        self.ffn_td = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_bu = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn_td = nn.LayerNorm(dim)
        self.norm_ffn_bu = nn.LayerNorm(dim)

    def forward(self, h_td: torch.Tensor, h_bu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_td: Top-Down features [B, N, D]
            h_bu: Bottom-Up features [B, N, D]
        Returns:
            h_td_out: Updated Top-Down features [B, N, D]
            h_bu_out: Updated Bottom-Up features [B, N, D]
        """
        # Depth-aware self-attention
        h_td = self.td_attn(h_td)
        h_bu = self.bu_attn(h_bu)

        # Cross-attention interaction
        # TD attends to BU
        h_td_cross, _ = self.cross_attn_td(h_td, h_bu, h_bu)
        h_td = self.norm_td(h_td + h_td_cross)

        # BU attends to TD
        h_bu_cross, _ = self.cross_attn_bu(h_bu, h_td, h_td)
        h_bu = self.norm_bu(h_bu + h_bu_cross)

        # FFN
        h_td = self.norm_ffn_td(h_td + self.ffn_td(h_td))
        h_bu = self.norm_ffn_bu(h_bu + self.ffn_bu(h_bu))

        return h_td, h_bu


class ConditionalRowPooling(AttentionRowPooling):
    """Original row query plus an image-conditioned, zero-initialized residual."""

    def __init__(self, feature_dim=768, num_heads=4):
        super().__init__(feature_dim, num_heads)
        self.context_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)

    def forward(self, row_features):
        batch, rows, columns, channels = row_features.shape
        context = row_features.mean(dim=(1, 2))
        query = self.query.expand(batch, -1, -1) + self.context_proj(context).unsqueeze(1)
        query = query[:, None].expand(-1, rows, -1, -1).reshape(batch * rows, 1, channels)
        features = row_features.reshape(batch * rows, columns, channels)
        weights = F.softmax(torch.bmm(query, self.key_proj(features).transpose(1, 2)) * self.scale, dim=-1)
        pooled = torch.bmm(weights, self.value_proj(features)).squeeze(1)
        return self.out_proj(pooled).reshape(batch, rows, channels)
