"""
Feature-wise Linear Modulation (FiLM) Conditioning Generator for Flow Operating Parameters.
Maps (Re, AoA, U_inf) to scale (gamma) and shift (beta) latent modulation tensors.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """
    MLP that generates scale (gamma) and shift (beta) parameters from global conditions.
    """
    def __init__(
        self,
        cond_channels: int = 3,
        target_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        activation: str = "silu"
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.target_dim = target_dim

        act_layer = nn.SiLU() if activation.lower() == "silu" else nn.GELU()

        layers = [nn.Linear(cond_channels, hidden_dim), act_layer]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_layer])

        self.backbone = nn.Sequential(*layers)
        self.gamma_head = nn.Linear(hidden_dim, target_dim)
        self.beta_head = nn.Linear(hidden_dim, target_dim)

        # Initialize heads close to identity: gamma -> 1, beta -> 0
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: Tensor of shape [B, cond_channels] or [cond_channels]
        Returns:
            gamma: [B, target_dim] (centered around 1.0)
            beta:  [B, target_dim] (centered around 0.0)
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)

        feat = self.backbone(cond)
        gamma = self.gamma_head(feat) + 1.0
        beta = self.beta_head(feat)
        return gamma, beta


class FiLMBlock(nn.Module):
    """
    Applies LayerNorm and FiLM modulation to an input latent tensor.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        x: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        beta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [N, channels] or [B, N, channels]
            gamma: Scale tensor of shape [1, channels] or [B, 1, channels]
            beta: Shift tensor of shape [1, channels] or [B, 1, channels]
        """
        out = self.norm(x)
        if gamma is not None and beta is not None:
            if gamma.dim() < out.dim():
                gamma = gamma.unsqueeze(1) if out.dim() == 3 else gamma
                beta = beta.unsqueeze(1) if out.dim() == 3 else beta
            out = gamma * out + beta
        return out
