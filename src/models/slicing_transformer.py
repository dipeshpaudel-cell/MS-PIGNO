"""
Physics-Aware Slicing Transformer (Transolver Engine).
Computes linear-complexity non-local attention on unstructured mesh representations.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .film_conditioning import FiLMBlock


class SlicingAttentionLayer(nn.Module):
    """
    Single Physics-Aware Slicing Transformer Layer.
    Maps N_c coarse tokens -> M physical slices -> Multi-Head Self-Attention -> N_c tokens.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_slices: int = 64,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.05,
        slice_temp: float = 1.0,
        slice_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slices = num_slices
        self.num_heads = num_heads
        self.slice_temp = slice_temp
        self.slice_norm_eps = slice_norm_eps

        # Slicing Projection Matrix
        self.slice_proj = nn.Linear(hidden_dim, num_slices)

        # Multi-Head Attention on Latent Slices (M x D)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout)
        )

        self.film_1 = FiLMBlock(hidden_dim)
        self.film_2 = FiLMBlock(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h: Node tokens of shape [N_c, hidden_dim] or [B, N_c, hidden_dim]
            gamma, beta: FiLM modulation parameters [B, hidden_dim]
        """
        has_batch = h.dim() == 3
        if not has_batch:
            h = h.unsqueeze(0)  # [1, N_c, D]

        B, N_c, D = h.shape

        # 1. Physics-Aware Slicing Weights: [B, N_c, M]
        slice_logits = self.slice_proj(h) / self.slice_temp
        S = F.softmax(slice_logits, dim=-1)  # [B, N_c, M]

        # 2. Normalized Slice Aggregation: Z_m = sum(S_im * H_i) / sum(S_im)
        Z_unnorm = torch.bmm(S.transpose(1, 2), h)  # [B, M, D]
        S_sum = S.sum(dim=1, keepdim=False).unsqueeze(-1)  # [B, M, 1]
        Z = Z_unnorm / (S_sum + self.slice_norm_eps)  # [B, M, D]

        # 3. Multi-Head Self-Attention on Slice Space
        Z_norm = self.film_1(Z, gamma, beta)
        Z_attn, _ = self.mha(Z_norm, Z_norm, Z_norm)
        Z = Z + Z_attn

        # 4. Feed-Forward Network
        Z_ffn = self.film_2(Z, gamma, beta)
        Z = Z + self.ffn(Z_ffn)

        # 5. De-slicing Back to Graph Tokens: H_out = S @ Z -> [B, N_c, D]
        h_out = torch.bmm(S, Z)
        out = h + h_out

        return out.squeeze(0) if not has_batch else out


class PhysicsAwareSlicingTransformer(nn.Module):
    """
    Stacked Physics-Aware Slicing Transformer Backbone.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        num_slices: int = 64,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        dropout: float = 0.05,
        slice_temp: float = 1.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SlicingAttentionLayer(
                hidden_dim=hidden_dim,
                num_slices=num_slices,
                num_heads=num_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                slice_temp=slice_temp
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        h: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = h
        for layer in self.layers:
            out = layer(out, gamma=gamma, beta=beta)
        return out
