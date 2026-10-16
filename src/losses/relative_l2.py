"""
SDF-Weighted Near-Wall Relative L2 Loss.
Prevents low near-wall velocities from drowning out viscous sublayer error signals.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn


class SDFWeightedRelativeL2Loss(nn.Module):
    """
    Evaluates field error weighted by distance to solid wall:
      L_data = mean_i ( omega(phi_i) * ( w_u * ||u - u_gt||^2 / (||u_gt||^2 + eps) + w_p * (p - p_gt)^2 + w_nu * (nu - nu_gt)^2 ) )
    """
    def __init__(
        self,
        w_u: float = 1.0,
        w_p: float = 1.0,
        w_nu: float = 1.0,
        vel_eps: float = 0.1,
        bl_delta: float = 0.05
    ):
        super().__init__()
        self.w_u = w_u
        self.w_p = w_p
        self.w_nu = w_nu
        self.vel_eps = vel_eps
        self.bl_delta = bl_delta

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sdf: torch.Tensor,
        bl_boost_weight: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred: Predicted normalized fields [N, 4] -> [u_x, u_y, p, nu_t]
            y_true: Ground truth normalized fields [N, 4]
            sdf: Physical distance to wall [N, 1]
            bl_boost_weight: Dynamic curriculum boost multiplier for near-wall nodes
        Returns:
            Dictionary with 'loss_u', 'loss_p', 'loss_nu', 'loss_total'
        """
        # Proximity weight: omega(phi) = 1.0 + (boost - 1.0) * exp(-phi / delta)
        phi = torch.clamp(sdf.squeeze(-1), min=0.0)
        omega = 1.0 + max(0.0, bl_boost_weight - 1.0) * torch.exp(-phi / self.bl_delta)

        # 1. Relative Velocity Error
        u_pred = y_pred[:, 0:2]
        u_true = y_true[:, 0:2]
        u_diff_sq = torch.sum((u_pred - u_true) ** 2, dim=-1) # [N]
        u_norm_sq = torch.sum(u_true ** 2, dim=-1) + (self.vel_eps ** 2)
        rel_u_err = u_diff_sq / u_norm_sq

        # 2. Pressure Error
        p_pred = y_pred[:, 2]
        p_true = y_true[:, 2]
        p_err = (p_pred - p_true) ** 2

        # 3. Turbulent Viscosity Error (in normalized log-space)
        nu_pred = y_pred[:, 3]
        nu_true = y_true[:, 3]
        nu_err = (nu_pred - nu_true) ** 2

        # Weighted Averages
        loss_u = torch.mean(omega * rel_u_err)
        loss_p = torch.mean(omega * p_err)
        loss_nu = torch.mean(omega * nu_err)

        loss_total = self.w_u * loss_u + self.w_p * loss_p + self.w_nu * loss_nu

        return {
            "loss_u": loss_u,
            "loss_p": loss_p,
            "loss_nu": loss_nu,
            "loss_total": loss_total
        }
