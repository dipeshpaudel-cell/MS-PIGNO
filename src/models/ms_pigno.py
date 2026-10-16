"""
Multi-Scale Physics-Informed Graph Neural Operator (MS-PIGNO) with Slicing Attention.
Integrates Multi-Scale Graph Hierarchy, Transolver Slicing Transformer,
Boundary Layer Refinement Head, and Exact Physical Boundary Constraints.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anisotropic_conv import AnisotropicMeshConvBlock
from .film_conditioning import FiLMGenerator, FiLMBlock
from .slicing_transformer import PhysicsAwareSlicingTransformer
from .bl_refinement_head import BoundaryLayerRefinementHead


class MSPIGNO(nn.Module):
    """
    State-of-the-Art Neural Operator Surrogate for 2D RANS Aerodynamics.
    """
    def __init__(
        self,
        node_in_channels: int = 5,       # [U_inf_x, U_inf_y, phi, n_x, n_y]
        pos_channels: int = 2,           # [x, y]
        edge_in_channels: int = 8,       # Anisotropic edge attributes (8 channels incl. norm_dot)
        cond_channels: int = 3,          # [log10(Re), AoA, U_inf]
        out_channels: int = 4,           # [u_x, u_y, p, nu_t]
        hidden_dims: Tuple[int, int, int] = (64, 128, 256),
        transformer_layers: int = 6,
        num_slices: int = 64,
        num_heads: int = 8,
        use_bl_refinement: bool = True,
        bl_delta: float = 0.05,
        wall_decay_delta: float = 1e-5,
        dropout: float = 0.05
    ):
        super().__init__()
        self.wall_decay_delta = wall_decay_delta
        self.use_bl_refinement = use_bl_refinement
        d0, d1, d2 = hidden_dims

        # 1. Global Condition FiLM Generators
        self.film_gen_enc = FiLMGenerator(cond_channels=cond_channels, target_dim=d2, hidden_dim=128)
        self.film_gen_trans = FiLMGenerator(cond_channels=cond_channels, target_dim=d2, hidden_dim=128)
        self.film_gen_dec = FiLMGenerator(cond_channels=cond_channels, target_dim=d0, hidden_dim=128)
        self.film_gen_bl = FiLMGenerator(cond_channels=cond_channels, target_dim=128, hidden_dim=64)

        # 2. Input Lifting Projections
        self.node_lift = nn.Sequential(
            nn.Linear(node_in_channels + pos_channels, d0),
            nn.SiLU(),
            nn.Linear(d0, d0)
        )

        # 3. Multi-Scale Graph Encoder
        # Level 0 (Fine Mesh)
        self.enc_l0 = AnisotropicMeshConvBlock(d0, d0, edge_channels=edge_in_channels, hidden_dim=d0, dropout=dropout)
        self.l0_to_l1_proj = nn.Linear(d0, d1)

        # Level 1 (Mid Mesh)
        self.enc_l1 = AnisotropicMeshConvBlock(d1, d1, edge_channels=edge_in_channels, hidden_dim=d1, dropout=dropout)
        self.l1_to_l2_proj = nn.Linear(d1, d2)

        # Level 2 (Coarse Mesh)
        self.enc_l2 = AnisotropicMeshConvBlock(d2, d2, edge_channels=edge_in_channels, hidden_dim=d2, dropout=dropout)

        # 4. Physics-Aware Slicing Transformer (Transolver Engine on Level 2)
        self.transformer = PhysicsAwareSlicingTransformer(
            hidden_dim=d2,
            num_slices=num_slices,
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # 5. Multi-Scale Graph Decoder
        # Level 2 -> Level 1
        self.dec_l2 = AnisotropicMeshConvBlock(d2, d2, edge_channels=edge_in_channels, hidden_dim=d2, dropout=dropout)
        self.l2_to_l1_up = nn.Linear(d2, d1)

        # Level 1 Decoder
        self.dec_l1 = AnisotropicMeshConvBlock(d1 + d1, d1, edge_channels=edge_in_channels, hidden_dim=d1, dropout=dropout)
        self.l1_to_l0_up = nn.Linear(d1, d0)

        # Level 0 Decoder
        self.dec_l0 = AnisotropicMeshConvBlock(d0 + d0, d0, edge_channels=edge_in_channels, hidden_dim=d0, dropout=dropout)

        # 6. Primary Output Projection Head
        self.out_head = nn.Sequential(
            nn.Linear(d0 + node_in_channels, d0),
            nn.SiLU(),
            nn.Linear(d0, out_channels)
        )

        # 7. Dedicated Boundary Layer Refinement Head (with pre-encoder skip connection)
        if use_bl_refinement:
            self.bl_head = BoundaryLayerRefinementHead(
                in_channels=d0,
                skip_channels=d0,
                out_channels=out_channels,
                hidden_dim=128,
                delta_bl=bl_delta,
                dropout=dropout
            )

    def _pool(self, x: torch.Tensor, cluster: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """
        Deterministic scatter-mean pooling across graph levels.
        """
        device = x.device
        out = torch.zeros(num_clusters, x.shape[1], device=device, dtype=x.dtype)
        counts = torch.zeros(num_clusters, 1, device=device, dtype=x.dtype)
        cluster_expanded = cluster.unsqueeze(-1).expand_as(x)
        out.scatter_add_(0, cluster_expanded, x)
        counts.scatter_add_(0, cluster.unsqueeze(-1), torch.ones_like(cluster.unsqueeze(-1), dtype=x.dtype))
        return out / torch.clamp(counts, min=1.0)

    def _unpool(self, x_coarse: torch.Tensor, cluster: torch.Tensor) -> torch.Tensor:
        """
        Unpooling by index broadcasting from coarse to fine level.
        """
        return x_coarse[cluster]

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        cond: torch.Tensor,
        cluster_l0_l1: torch.Tensor,
        cluster_l1_l2: torch.Tensor,
        edge_index_l1: Optional[torch.Tensor] = None,
        edge_index_l2: Optional[torch.Tensor] = None,
        apply_physical_constraints: bool = True
    ) -> torch.Tensor:
        """
        Args:
            x: Input node features [N, 5] -> [U_inf_x, U_inf_y, phi, n_x, n_y]
            pos: Coordinates [N, 2]
            edge_index: Connectivity at Level 0 [2, E0]
            edge_attr: Anisotropic edge features [E0, 7]
            cond: Operating parameters [3] or [B, 3] -> [log10(Re), AoA, U_inf]
            cluster_l0_l1: Coarsening indices L0 -> L1 [N]
            cluster_l1_l2: Coarsening indices L1 -> L2 [N1]
        Returns:
            y_pred: Predicted flow fields [N, 4] -> [u_x, u_y, p, nu_t]
        """
        num_l1 = int(cluster_l0_l1.max().item() + 1)
        num_l2 = int(cluster_l1_l2.max().item() + 1)

        # Generate FiLM conditioning parameters
        gamma_enc, beta_enc = self.film_gen_enc(cond)
        gamma_trans, beta_trans = self.film_gen_trans(cond)
        gamma_dec, beta_dec = self.film_gen_dec(cond)
        gamma_bl, beta_bl = self.film_gen_bl(cond)

        edge_dim = edge_attr.shape[1]

        # Edge fallback for coarsened graphs if not explicitly passed
        if edge_index_l1 is None:
            edge_index_l1 = torch.stack([torch.arange(num_l1, device=x.device), torch.arange(num_l1, device=x.device)], dim=0)
            edge_attr_l1 = torch.zeros(num_l1, edge_dim, device=x.device, dtype=edge_attr.dtype)
        else:
            edge_attr_l1 = torch.zeros(edge_index_l1.shape[1], edge_dim, device=x.device, dtype=edge_attr.dtype)

        if edge_index_l2 is None:
            edge_index_l2 = torch.stack([torch.arange(num_l2, device=x.device), torch.arange(num_l2, device=x.device)], dim=0)
            edge_attr_l2 = torch.zeros(num_l2, edge_dim, device=x.device, dtype=edge_attr.dtype)
        else:
            edge_attr_l2 = torch.zeros(edge_index_l2.shape[1], edge_dim, device=x.device, dtype=edge_attr.dtype)

        # 1. Lift Node Features: [N, 5 + 2] -> [N, d0]
        h0 = self.node_lift(torch.cat([x, pos], dim=-1))

        # 2. Multi-Scale Graph Encoder
        # Level 0 (Fine)
        h0_enc = self.enc_l0(h0, edge_index, edge_attr, gamma=None, beta=None)

        # Downsample L0 -> L1
        h1 = self._pool(h0_enc, cluster_l0_l1, num_l1)
        h1 = self.l0_to_l1_proj(h1)
        h1_enc = self.enc_l1(h1, edge_index_l1, edge_attr_l1, gamma=None, beta=None)

        # Downsample L1 -> L2
        h2 = self._pool(h1_enc, cluster_l1_l2, num_l2)
        h2 = self.l1_to_l2_proj(h2)
        h2_enc = self.enc_l2(h2, edge_index_l2, edge_attr_l2, gamma=gamma_enc, beta=beta_enc)

        # 3. Physics-Aware Slicing Transformer (Transolver on Level 2)
        h2_trans = self.transformer(h2_enc, gamma=gamma_trans, beta=beta_trans)

        # 4. Multi-Scale Graph Decoder
        h2_dec = self.dec_l2(h2_trans, edge_index_l2, edge_attr_l2, gamma=gamma_enc, beta=beta_enc)

        # Upsample L2 -> L1 with Skip Connection
        h1_up = self._unpool(self.l2_to_l1_up(h2_dec), cluster_l1_l2)
        h1_dec = self.dec_l1(torch.cat([h1_up, h1_enc], dim=-1), edge_index_l1, edge_attr_l1)

        # Upsample L1 -> L0 with Skip Connection
        h0_up = self._unpool(self.l1_to_l0_up(h1_dec), cluster_l0_l1)
        h0_dec = self.dec_l0(torch.cat([h0_up, h0_enc], dim=-1), edge_index, edge_attr, gamma=gamma_dec, beta=beta_dec)

        # 5. Primary Prediction Head
        y_raw = self.out_head(torch.cat([h0_dec, x], dim=-1)) # [N, 4]

        # 6. Boundary Layer Refinement Head (with pre-encoder skip connection)
        if self.use_bl_refinement:
            sdf = x[:, 2:3]
            normals = x[:, 3:5]
            delta_y = self.bl_head(h0_dec, sdf, normals, gamma=gamma_bl, beta=beta_bl, h_skip=h0)
            y_primary = y_raw + delta_y
        else:
            y_primary = y_raw

        # 7. Exact Physical Boundary Condition & Realizability Masking
        if apply_physical_constraints:
            sdf = x[:, 2:3]
            # Exact No-Slip velocity condition at solid wall: u -> 0 as phi -> 0
            wall_attenuation = 1.0 - torch.exp(-torch.clamp(sdf, min=0.0) / self.wall_decay_delta)
            vel_masked = y_primary[:, 0:2] * wall_attenuation

            # Kinematic pressure: unconstrained (Neumann wall BC)
            pressure = y_primary[:, 2:3]

            # Turbulent Eddy Viscosity: Strict positivity constraint (nu_t >= 0)
            nu_t_positive = F.softplus(y_primary[:, 3:4]) + 1.0e-8

            y_final = torch.cat([vel_masked, pressure, nu_t_positive], dim=-1)
            return y_final

        return y_primary
