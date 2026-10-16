"""
Incompressible RANS Momentum & Continuity PDE Residual Evaluators.
Computes governing fluid equations using precomputed LS-FV gradient operators.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn

from .lsfv_gradients import LSFVSparseGradientOperator


class RANSResidualEvaluator(nn.Module):
    """
    Evaluates Continuity and Navier-Stokes / RANS Momentum PDE residuals:
      div(u) = 0
      (u . grad)u + grad(p) - div(nu_eff * (grad u + grad u^T)) = 0
    """
    def __init__(self, nu_laminar: float = 1.5e-5, rho: float = 1.0):
        super().__init__()
        self.nu_laminar = nu_laminar
        self.rho = rho
        self.grad_op = LSFVSparseGradientOperator()

    def evaluate_residuals(
        self,
        y_pred: torch.Tensor,
        ls_indices: torch.Tensor,
        ls_wx: torch.Tensor,
        ls_wy: torch.Tensor,
        sdf: torch.Tensor,
        fluid_sdf_threshold: float = 0.001
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred: Predicted physical fields [N, 4] -> [u_x, u_y, p, nu_t]
            ls_indices, ls_wx, ls_wy: Precomputed LS-FV sparse gradient weights
            sdf: Signed distance to wall [N, 1]
            fluid_sdf_threshold: Distance cutoff for evaluating fluid interior residuals
        Returns:
            Dictionary with 'res_cont', 'res_mom_x', 'res_mom_y', 'pde_loss'
        """
        num_nodes = y_pred.shape[0]

        u = y_pred[:, 0:1] # [N, 1]
        v = y_pred[:, 1:2] # [N, 1]
        p = y_pred[:, 2:3] # [N, 1]
        nu_t = y_pred[:, 3:4] # [N, 1] (strictly positive)

        nu_eff = self.nu_laminar + nu_t # [N, 1]

        # 1. First Derivatives: Velocity & Pressure
        du_dx, du_dy = self.grad_op(u, ls_indices, ls_wx, ls_wy, num_nodes)
        dv_dx, dv_dy = self.grad_op(v, ls_indices, ls_wx, ls_wy, num_nodes)
        dp_dx, dp_dy = self.grad_op(p, ls_indices, ls_wx, ls_wy, num_nodes)

        # 2. Continuity Residual: div(u) = du/dx + dv/dy
        res_cont = du_dx + dv_dy # [N, 1]

        # 3. Viscous Stress Tensor Components:
        # tau_xx = 2 * nu_eff * du/dx
        # tau_yy = 2 * nu_eff * dv/dy
        # tau_xy = nu_eff * (du/dy + dv/dx)
        tau_xx = 2.0 * nu_eff * du_dx
        tau_yy = 2.0 * nu_eff * dv_dy
        tau_xy = nu_eff * (du_dy + dv_dx)

        # 4. Divergence of Viscous Stresses:
        dtau_xx_dx, _ = self.grad_op(tau_xx, ls_indices, ls_wx, ls_wy, num_nodes)
        _, dtau_xy_dy = self.grad_op(tau_xy, ls_indices, ls_wx, ls_wy, num_nodes)
        dtau_xy_dx, _ = self.grad_op(tau_xy, ls_indices, ls_wx, ls_wy, num_nodes)
        _, dtau_yy_dy = self.grad_op(tau_yy, ls_indices, ls_wx, ls_wy, num_nodes)

        visc_div_x = dtau_xx_dx + dtau_xy_dy
        visc_div_y = dtau_xy_dx + dtau_yy_dy

        # 5. Momentum Residuals (Incompressible RANS):
        # res_x = u*du/dx + v*du/dy + (1/rho)*dp/dx - visc_div_x
        # res_y = u*dv/dx + v*dv/dy + (1/rho)*dp/dy - visc_div_y
        conv_x = u * du_dx + v * du_dy
        conv_y = u * dv_dx + v * dv_dy

        res_mom_x = conv_x + (dp_dx / self.rho) - visc_div_x
        res_mom_y = conv_y + (dp_dy / self.rho) - visc_div_y

        # Fluid interior mask: enforce PDE strictly outside the viscous sublayer wall mask
        fluid_mask = (sdf.squeeze(-1) > fluid_sdf_threshold).float()
        num_fluid = torch.clamp(fluid_mask.sum(), min=1.0)

        loss_cont = (fluid_mask * (res_cont.squeeze(-1) ** 2)).sum() / num_fluid
        loss_mom_x = (fluid_mask * (res_mom_x.squeeze(-1) ** 2)).sum() / num_fluid
        loss_mom_y = (fluid_mask * (res_mom_y.squeeze(-1) ** 2)).sum() / num_fluid

        total_pde_loss = loss_cont + 0.5 * (loss_mom_x + loss_mom_y)

        return {
            "res_cont": res_cont,
            "res_mom_x": res_mom_x,
            "res_mom_y": res_mom_y,
            "loss_cont": loss_cont,
            "loss_mom_x": loss_mom_x,
            "loss_mom_y": loss_mom_y,
            "pde_loss": total_pde_loss
        }
