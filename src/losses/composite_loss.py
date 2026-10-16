"""
Composite Curriculum Loss Coordinator.
Combines Data, Boundary, Soft RANS PDE, Sobolev Gradient, and Surface Force Losses
across the 3-phase curriculum schedule.
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from .relative_l2 import SDFWeightedRelativeL2Loss
from .sobolev_loss import SobolevGradientLoss
from .loss_balancer import ReLoBRaLoLossBalancer
from ..physics.rans_residuals import RANSResidualEvaluator
from ..physics.surface_integrals import DifferentiableSurfaceIntegrator
from ..data.transforms import FlowFeatureNormalizer


class CompositeCurriculumLoss(nn.Module):
    """
    Unified loss engine managing multi-objective loss computation and phase transitions.
    """
    def __init__(
        self,
        normalizer: FlowFeatureNormalizer,
        nu_laminar: float = 1.5e-5,
        rho: float = 1.0,
        chord: float = 1.0,
        use_relobralo: bool = True
    ):
        super().__init__()
        self.normalizer = normalizer
        self.nu_laminar = nu_laminar
        self.rho = rho
        self.chord = chord
        self.use_relobralo = use_relobralo

        # Sub-loss modules
        self.data_loss_fn = SDFWeightedRelativeL2Loss()
        self.sobolev_loss_fn = SobolevGradientLoss()
        self.pde_evaluator = RANSResidualEvaluator(nu_laminar=nu_laminar, rho=rho)
        self.surface_integrator = DifferentiableSurfaceIntegrator(nu_laminar=nu_laminar, rho=rho, chord=chord)

        loss_keys = ["loss_data", "loss_bc", "loss_pde", "loss_grad", "loss_coeff"]
        self.balancer = ReLoBRaLoLossBalancer(loss_names=loss_keys)

    def compute_loss(
        self,
        y_pred_norm: torch.Tensor,
        batch_data,
        phase: int = 1,
        phase_config: Optional[Dict] = None,
        current_epoch: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            y_pred_norm: Predicted standardized fields [N, 4]
            batch_data: PyG Data object
            phase: Current curriculum phase (1, 2, or 3)
            phase_config: Dictionary of phase loss weights
        Returns:
            Dictionary containing individual losses and total scalar loss
        """
        device = y_pred_norm.device
        num_nodes = y_pred_norm.shape[0]

        # Extract features and targets
        y_true_norm = batch_data.y_norm.to(device)
        y_true_phys = batch_data.y.to(device)
        sdf = batch_data.x[:, 2:3].to(device)

        # Invert predictions to physical space for PDE, Sobolev, and Force integrals
        y_pred_phys = self.normalizer.inverse_transform_y(y_pred_norm)

        weights = phase_config or {}
        w_data = weights.get("lambda_data", 1.0)
        w_bc = weights.get("lambda_bc", 1.0)
        w_pde = weights.get("lambda_pde", 0.0)
        w_grad = weights.get("lambda_grad", 0.0)
        w_coeff = weights.get("lambda_coeff", 0.0)
        bl_boost = weights.get("bl_weight_boost", 1.0)
        w_cd_fric = weights.get("lambda_cd_fric", 5.0)

        # 1. SDF-Weighted Relative Data Loss
        data_res = self.data_loss_fn(
            y_pred=y_pred_norm,
            y_true=y_true_norm,
            sdf=sdf,
            bl_boost_weight=bl_boost
        )
        loss_data = data_res["loss_total"]

        # 2. Boundary Condition Loss (No-slip at wall and freestream velocity matching)
        wall_mask = (sdf.squeeze(-1) < 1e-4)
        if wall_mask.any():
            u_wall = y_pred_norm[wall_mask, 0:2]
            # In normalized space, target is zero physical velocity
            target_wall_norm = (-self.normalizer.y_mean[0:2] / self.normalizer.y_std[0:2]).to(device)
            loss_bc = torch.mean((u_wall - target_wall_norm.unsqueeze(0)) ** 2)
        else:
            loss_bc = torch.tensor(0.0, device=device)

        # 3. Soft RANS PDE Residual Loss (Phase 2 & 3)
        loss_pde = torch.tensor(0.0, device=device)
        if w_pde > 0.0 and hasattr(batch_data, "ls_indices"):
            pde_res = self.pde_evaluator.evaluate_residuals(
                y_pred=y_pred_phys,
                ls_indices=batch_data.ls_indices.to(device),
                ls_wx=batch_data.ls_wx.to(device),
                ls_wy=batch_data.ls_wy.to(device),
                sdf=sdf
            )
            loss_pde = pde_res["pde_loss"]

        # 4. Sobolev Gradient Loss (Phase 2 & 3)
        loss_grad = torch.tensor(0.0, device=device)
        if w_grad > 0.0 and hasattr(batch_data, "ls_indices"):
            grad_res = self.sobolev_loss_fn(
                y_pred=y_pred_phys,
                y_true=y_true_phys,
                ls_indices=batch_data.ls_indices.to(device),
                ls_wx=batch_data.ls_wx.to(device),
                ls_wy=batch_data.ls_wy.to(device)
            )
            loss_grad = grad_res["loss_grad_total"]

        # 5. Differentiable Surface Force Coefficient Loss (Phase 3)
        loss_coeff = torch.tensor(0.0, device=device)
        c_l_pred, c_d_pred = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        if w_coeff > 0.0 and hasattr(batch_data, "wall_indices") and hasattr(batch_data, "ls_indices"):
            u_inf = float(batch_data.cond[2].item()) if hasattr(batch_data, "cond") else 30.0
            aoa_rad = float(batch_data.cond[1].item()) if hasattr(batch_data, "cond") else 0.0

            # Compute predicted forces
            forces_pred = self.surface_integrator.integrate_forces(
                y_pred=y_pred_phys,
                wall_indices=batch_data.wall_indices.to(device),
                wall_delta_s=batch_data.wall_delta_s.to(device),
                wall_normals=batch_data.wall_normals.to(device),
                wall_tangents=batch_data.wall_tangents.to(device),
                u_inf=u_inf,
                ls_indices=batch_data.ls_indices.to(device),
                ls_wx=batch_data.ls_wx.to(device),
                ls_wy=batch_data.ls_wy.to(device),
                aoa_rad=aoa_rad
            )
            # Compute ground truth forces
            with torch.no_grad():
                forces_true = self.surface_integrator.integrate_forces(
                    y_pred=y_true_phys,
                    wall_indices=batch_data.wall_indices.to(device),
                    wall_delta_s=batch_data.wall_delta_s.to(device),
                    wall_normals=batch_data.wall_normals.to(device),
                    wall_tangents=batch_data.wall_tangents.to(device),
                    u_inf=u_inf,
                    ls_indices=batch_data.ls_indices.to(device),
                    ls_wx=batch_data.ls_wx.to(device),
                    ls_wy=batch_data.ls_wy.to(device),
                    aoa_rad=aoa_rad
                )

            err_cl = (forces_pred["c_l"] - forces_true["c_l"]) ** 2
            err_cd_press = (forces_pred["c_d_press"] - forces_true["c_d_press"]) ** 2
            err_cd_fric = (forces_pred["c_d_fric"] - forces_true["c_d_fric"]) ** 2

            loss_coeff = err_cl + err_cd_press + w_cd_fric * err_cd_fric
            c_l_pred = forces_pred["c_l"]
            c_d_pred = forces_pred["c_d_total"]

        # Dynamic Loss Balancing
        dyn_weights = {"loss_data": 1.0, "loss_bc": 1.0, "loss_pde": 1.0, "loss_grad": 1.0, "loss_coeff": 1.0}
        if self.use_relobralo:
            raw_loss_dict = {
                "loss_data": loss_data,
                "loss_bc": loss_bc,
                "loss_pde": loss_pde,
                "loss_grad": loss_grad,
                "loss_coeff": loss_coeff
            }
            dyn_weights = self.balancer.update_weights(raw_loss_dict, current_epoch)

        total_loss = (
            w_data * dyn_weights["loss_data"] * loss_data +
            w_bc * dyn_weights["loss_bc"] * loss_bc +
            w_pde * dyn_weights["loss_pde"] * loss_pde +
            w_grad * dyn_weights["loss_grad"] * loss_grad +
            w_coeff * dyn_weights["loss_coeff"] * loss_coeff
        )

        return {
            "loss": total_loss,
            "loss_data": loss_data,
            "loss_bc": loss_bc,
            "loss_pde": loss_pde,
            "loss_grad": loss_grad,
            "loss_coeff": loss_coeff,
            "c_l_pred": c_l_pred,
            "c_d_pred": c_d_pred
        }
