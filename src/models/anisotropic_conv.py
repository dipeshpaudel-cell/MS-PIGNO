"""
Anisotropic Mesh Graph Convolution Block.
Handles aspect-ratio-aware directional message passing for boundary layer meshes.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .film_conditioning import FiLMBlock


class AnisotropicMeshConv(MessagePassing):
    """
    Message passing layer with anisotropic edge attributes and directional attention.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int = 7,
        hidden_dim: int = 128
    ):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Message MLP: [h_i, h_j, e_ij] -> message
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )

        # Anisotropic Attention Gate: [h_i, h_j, e_ij] -> attention scalar
        self.attn_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Node Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )

        # Skip linear projection if channel dimensions differ
        self.skip_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            edge_attr: Anisotropic edge features [E, edge_channels]
        """
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        skip = self.skip_proj(x)
        return self.update_mlp(torch.cat([skip, out], dim=-1))

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        feat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(feat)
        gate = self.attn_mlp(feat)
        return msg * gate


class AnisotropicMeshConvBlock(nn.Module):
    """
    Complete Graph Convolution Block: AnisotropicConv + LayerNorm + FiLM + Residual.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int = 7,
        hidden_dim: int = 128,
        dropout: float = 0.05
    ):
        super().__init__()
        self.conv = AnisotropicMeshConv(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_channels=edge_channels,
            hidden_dim=hidden_dim
        )
        self.film = FiLMBlock(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        res = self.res_proj(x)
        h = self.conv(x, edge_index, edge_attr)
        h = self.film(h, gamma, beta)
        h = self.dropout(h)
        return res + h
