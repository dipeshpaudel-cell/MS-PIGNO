"""
Sobolev (H1 Semi-Norm) Gradient Supervision Loss.
Directly penalizes derivative prediction errors for velocity and pressure.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn

from ..physics.lsfv_gradients import LSFVSparseGradientOperator


class SobolevGradientLoss(nn.Module):
    """
    Computes Sobolev H1 loss using GPU-accelerated LS-FV sparse gradients:
      L_grad = mean( ||grad(u_pred) - grad(u_true)||^2 + ||grad(p_pred) - grad(p_true)||^2 )
    """
    def __init__(self, w_grad_u: float = 1.0, w_grad_p: float = 0.5):
        super().__init__()
        self.w_grad_u = w_grad_u
        self.w_grad_p = w_grad_p
        self.grad_op = LSFVSparseGradientOperator()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        ls_indices: torch.Tensor,
        ls_wx: torch.Tensor,
        ls_wy: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred: Predicted physical fields [N, 4]
            y_true: Ground truth physical fields [N, 4]
            ls_indices, ls_wx, ls_wy: Precomputed LS-FV gradient operators
        Returns:
            Dictionary with 'loss_grad_u', 'loss_grad_p', 'loss_grad_total'
        """
        num_nodes = y_pred.shape[0]

        u_pred = y_pred[:, 0:2] # [N, 2]
        u_true = y_true[:, 0:2] # [N, 2]

        p_pred = y_pred[:, 2:3] # [N, 1]
        p_true = y_true[:, 2:3] # [N, 1]

        # Velocity Gradients
        du_dx_pred, du_dy_pred = self.grad_op(u_pred, ls_indices, ls_wx, ls_wy, num_nodes)
        du_dx_true, du_dy_true = self.grad_op(u_true, ls_indices, ls_wx, ls_wy, num_nodes)

        grad_u_err = (du_dx_pred - du_dx_true) ** 2 + (du_dy_pred - du_dy_true) ** 2
        loss_grad_u = torch.mean(grad_u_err)

        # Pressure Gradients
        dp_dx_pred, dp_dy_pred = self.grad_op(p_pred, ls_indices, ls_wx, ls_wy, num_nodes)
        dp_dx_true, dp_dy_true = self.grad_op(p_true, ls_indices, ls_wx, ls_wy, num_nodes)

        grad_p_err = (dp_dx_pred - dp_dx_true) ** 2 + (dp_dy_pred - dp_dy_true) ** 2
        loss_grad_p = torch.mean(grad_p_err)

        loss_grad_total = self.w_grad_u * loss_grad_u + self.w_grad_p * loss_grad_p

        return {
            "loss_grad_u": loss_grad_u,
            "loss_grad_p": loss_grad_p,
            "loss_grad_total": loss_grad_total
        }
