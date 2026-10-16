"""
Differentiable Surface Force Integrator for Airfoil Geometries.
Computes Lift (C_L), Pressure Drag (C_D,p), Skin Friction Drag (C_D,f), and Total Drag (C_D).
"""
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn

from .lsfv_gradients import LSFVSparseGradientOperator


class DifferentiableSurfaceIntegrator(nn.Module):
    """
    Integrates wall pressure and wall shear stress over airfoil surface boundary nodes.
    """
    def __init__(self, nu_laminar: float = 1.5e-5, rho: float = 1.0, chord: float = 1.0):
        super().__init__()
        self.nu_laminar = nu_laminar
        self.mu_laminar = rho * nu_laminar
        self.rho = rho
        self.chord = chord
        self.grad_op = LSFVSparseGradientOperator()

    def integrate_forces(
        self,
        y_pred: torch.Tensor,
        wall_indices: torch.Tensor,
        wall_delta_s: torch.Tensor,
        wall_normals: torch.Tensor,
        wall_tangents: torch.Tensor,
        u_inf: float,
        ls_indices: torch.Tensor,
        ls_wx: torch.Tensor,
        ls_wy: torch.Tensor,
        aoa_rad: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred: Predicted physical fields [N, 4] -> [u_x, u_y, p, nu_t]
            wall_indices: Surface node indices [N_wall]
            wall_delta_s: Differential arc length segments [N_wall, 1]
            wall_normals: Unit outward normals (n_x, n_y) [N_wall, 2]
            wall_tangents: Unit tangents (t_x, t_y) [N_wall, 2]
            u_inf: Freestream velocity magnitude (m/s)
            ls_indices, ls_wx, ls_wy: LS-FV gradient weights for computing du/dn
            aoa_rad: Angle of attack in radians for lift/drag rotation
        Returns:
            Dictionary with C_L, C_D_press, C_D_fric, C_D_total, C_p, tau_w
        """
        num_nodes = y_pred.shape[0]
        q_inf = 0.5 * self.rho * (max(u_inf, 1.0) ** 2) * self.chord

        u = y_pred[:, 0:1]
        v = y_pred[:, 1:2]
        p = y_pred[:, 2:3]

        # 1. Surface Pressure Distribution
        p_wall = p[wall_indices]
        c_p = p_wall / (0.5 * (max(u_inf, 1.0) ** 2))

        # 2. Wall Shear Stress via Spatial Gradients (signed, no abs)
        du_dx, du_dy = self.grad_op(u, ls_indices, ls_wx, ls_wy, num_nodes)
        dv_dx, dv_dy = self.grad_op(v, ls_indices, ls_wx, ls_wy, num_nodes)

        du_dx_wall = du_dx[wall_indices]
        du_dy_wall = du_dy[wall_indices]
        dv_dx_wall = dv_dx[wall_indices]
        dv_dy_wall = dv_dy[wall_indices]

        tx = wall_tangents[:, 0:1]
        ty = wall_tangents[:, 1:2]
        nx = wall_normals[:, 0:1]
        ny = wall_normals[:, 1:2]

        du_t_dx = du_dx_wall * tx + dv_dx_wall * ty
        du_t_dy = du_dy_wall * tx + dv_dy_wall * ty

        du_t_dn = du_t_dx * nx + du_t_dy * ny
        tau_w = self.mu_laminar * du_t_dn

        # 3. Cartesian Force Integration
        # Pressure forces (acting along inward normal, i.e. -n direction)
        f_p_x = -p_wall * nx * wall_delta_s
        f_p_y = -p_wall * ny * wall_delta_s

        # Viscous forces (acting along tangent direction, signed tau_w)
        f_v_x = tau_w * tx * wall_delta_s
        f_v_y = tau_w * ty * wall_delta_s

        # Total Cartesian forces on the body
        F_x = (f_p_x + f_v_x).sum()
        F_y = (f_p_y + f_v_y).sum()

        F_p_x = f_p_x.sum()
        F_p_y = f_p_y.sum()
        F_v_x = f_v_x.sum()
        F_v_y = f_v_y.sum()

        # 4. Rotate Cartesian forces into aerodynamic (lift/drag) frame
        cos_a = math.cos(aoa_rad)
        sin_a = math.sin(aoa_rad)

        # Drag = F . e_freestream, Lift = F . e_perp
        c_d_press = (F_p_x * cos_a + F_p_y * sin_a) / q_inf
        c_d_fric = (F_v_x * cos_a + F_v_y * sin_a) / q_inf
        c_d_total = c_d_press + c_d_fric

        c_l = (-F_x * sin_a + F_y * cos_a) / q_inf

        return {
            "c_l": c_l,
            "c_d_press": c_d_press,
            "c_d_fric": c_d_fric,
            "c_d_total": c_d_total,
            "c_p": c_p,
            "tau_w": tau_w
        }
