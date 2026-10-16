"""
Boundary Layer Refinement Head for Near-Wall Shear Stress & Viscous Sublayer Accuracy.
Uses log(y+) Fourier Positional Features and anisotropic wall-distance scaling.
"""
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .film_conditioning import FiLMBlock


class LogYPlusFourierEmbedding(nn.Module):
    """
    Fourier feature mapping in log(y+) space:
      gamma(phi) = [sin(2^k * pi * log(1 + phi/delta) / ln2), cos(...)]
    """
    def __init__(self, num_modes: int = 16, delta_bl: float = 0.05):
        super().__init__()
        self.num_modes = num_modes
        self.delta_bl = delta_bl
        frequencies = 2.0 ** torch.arange(num_modes, dtype=torch.float32)
        self.register_buffer("freqs", frequencies)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: Distance to wall [N_bl, 1]
        Returns:
            Fourier features [N_bl, 2 * num_modes]
        """
        # Normalized logarithmic coordinate
        phi_clamped = torch.clamp(phi, min=0.0, max=self.delta_bl)
        log_coord = torch.log(1.0 + phi_clamped / (self.delta_bl + 1e-8)) / math.log(2.0) # [0, 1]

        angles = log_coord * self.freqs.unsqueeze(0) * math.pi # [N_bl, num_modes]
        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        return torch.cat([sin_feat, cos_feat], dim=-1)


class BoundaryLayerRefinementHead(nn.Module):
    """
    High-resolution dedicated network that predicts residual corrections for near-wall flow:
        Y_final = Y_primary + alpha(phi) * Delta_Y
    """
    def __init__(
        self,
        in_channels: int = 64,       # Latent features from Level 0 decoder
        skip_channels: int = 0,      # Pre-encoder lifted features for skip connection
        out_channels: int = 4,       # Residual output fields [du_x, du_y, dp, dnu_t]
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_fourier_modes: int = 16,
        delta_bl: float = 0.05,
        dropout: float = 0.05
    ):
        super().__init__()
        self.delta_bl = delta_bl
        self.fourier_embed = LogYPlusFourierEmbedding(num_modes=num_fourier_modes, delta_bl=delta_bl)

        # [h_dec, h_skip, fourier, y_tilde, normal]
        input_dim = in_channels + skip_channels + 2 * num_fourier_modes + 1 + 2

        layers = []
        curr_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            curr_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.film = FiLMBlock(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_channels)

        # Zero-initialize head so refinement starts as neutral identity
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        h_0: torch.Tensor,
        phi: torch.Tensor,
        normals: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None,
        h_skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_0: Fine graph decoded node features [N, in_channels]
            phi: Physical signed distance to wall [N, 1]
            normals: Wall orientation / surface normal [N, 2]
            h_skip: Pre-encoder lifted features [N, skip_channels] for skip connection
        Returns:
            delta_y: Full mesh residual correction [N, out_channels]
        """
        N = h_0.shape[0]
        bl_mask = (phi.squeeze(-1) < self.delta_bl)

        delta_y_full = torch.zeros(N, 4, device=h_0.device, dtype=h_0.dtype)

        if not bl_mask.any():
            return delta_y_full

        h_bl = h_0[bl_mask]
        phi_bl = phi[bl_mask]
        norm_bl = normals[bl_mask]

        fourier_feat = self.fourier_embed(phi_bl)
        y_tilde = phi_bl / (self.delta_bl + 1e-8)

        if h_skip is not None:
            h_skip_bl = h_skip[bl_mask]
            inputs = torch.cat([h_bl, h_skip_bl, fourier_feat, y_tilde, norm_bl], dim=-1)
        else:
            inputs = torch.cat([h_bl, fourier_feat, y_tilde, norm_bl], dim=-1)

        feat = self.backbone(inputs)
        feat = self.film(feat, gamma, beta)
        delta_y_active = self.head(feat)

        decay_weight = torch.clamp(1.0 - (phi_bl / self.delta_bl), min=0.0, max=1.0) ** 2
        delta_y_damped = delta_y_active * decay_weight

        delta_y_full[bl_mask] = delta_y_damped
        return delta_y_full
