"""
Official AirfRANS Benchmark Metric Evaluator.
Computes field-wise variance-normalized relative L2 error, surface Cp RMSE,
Cf error, and integrated force coefficient discrepancies (C_L, C_D,press, C_D,fric).
"""
from typing import Dict, List, Optional
import numpy as np
import torch

from ..physics.surface_integrals import DifferentiableSurfaceIntegrator


def compute_relative_l2_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Official AirfRANS coefficient-normalized relative mean squared error:
      Error = ||y_hat - y||^2 / Var(y)
    """
    diff_sq = torch.sum((y_pred - y_true) ** 2).item()
    mean_val = torch.mean(y_true).item()
    var_norm = torch.sum((y_true - mean_val) ** 2).item() + 1e-12
    return diff_sq / var_norm


def evaluate_airfrans_predictions(
    predictions: List[torch.Tensor],
    targets: List[torch.Tensor],
    u_infs: Optional[List[float]] = None,
    aoa_rads: Optional[List[float]] = None,
    wall_data: Optional[List[Dict[str, torch.Tensor]]] = None,
    nu_laminar: float = 1.5e-5,
    rho: float = 1.0
) -> Dict[str, float]:
    """
    Comprehensive benchmark evaluation across all test simulations.
    """
    # Concatenate all test predictions and targets
    y_pred_all = torch.cat(predictions, dim=0) # [Total_N, 4]
    y_true_all = torch.cat(targets, dim=0)     # [Total_N, 4]

    err_ux = compute_relative_l2_error(y_pred_all[:, 0], y_true_all[:, 0])
    err_uy = compute_relative_l2_error(y_pred_all[:, 1], y_true_all[:, 1])
    err_p = compute_relative_l2_error(y_pred_all[:, 2], y_true_all[:, 2])
    err_nut = compute_relative_l2_error(y_pred_all[:, 3], y_true_all[:, 3])

    rel_l2_total = (err_ux + err_uy + err_p + err_nut) / 4.0

    metrics = {
        "val_rel_l2_ux": err_ux,
        "val_rel_l2_uy": err_uy,
        "val_rel_l2_p": err_p,
        "val_rel_l2_nut": err_nut,
        "val_rel_l2_total": rel_l2_total
    }

    # If surface wall metadata is provided, evaluate force coefficients
    if wall_data is not None and len(wall_data) == len(predictions):
        surface_integrator = DifferentiableSurfaceIntegrator(nu_laminar=nu_laminar, rho=rho)
        cl_errs = []
        cd_errs = []
        cd_press_errs = []
        cd_fric_errs = []
        cp_rmses = []

        for i in range(len(predictions)):
            y_pred_i = predictions[i]
            y_true_i = targets[i]
            w_i = wall_data[i]
            u_inf = u_infs[i] if u_infs else 30.0
            aoa_rad = aoa_rads[i] if aoa_rads else 0.0

            res_pred = surface_integrator.integrate_forces(
                y_pred=y_pred_i,
                wall_indices=w_i["wall_indices"],
                wall_delta_s=w_i["wall_delta_s"],
                wall_normals=w_i["wall_normals"],
                wall_tangents=w_i["wall_tangents"],
                u_inf=u_inf,
                ls_indices=w_i["ls_indices"],
                ls_wx=w_i["ls_wx"],
                ls_wy=w_i["ls_wy"],
                aoa_rad=aoa_rad
            )
            res_true = surface_integrator.integrate_forces(
                y_pred=y_true_i,
                wall_indices=w_i["wall_indices"],
                wall_delta_s=w_i["wall_delta_s"],
                wall_normals=w_i["wall_normals"],
                wall_tangents=w_i["wall_tangents"],
                u_inf=u_inf,
                ls_indices=w_i["ls_indices"],
                ls_wx=w_i["ls_wx"],
                ls_wy=w_i["ls_wy"],
                aoa_rad=aoa_rad
            )

            cl_p = float(res_pred["c_l"].item())
            cl_t = float(res_true["c_l"].item())
            cd_p = float(res_pred["c_d_total"].item())
            cd_t = float(res_true["c_d_total"].item())
            cd_pr_p = float(res_pred["c_d_press"].item())
            cd_pr_t = float(res_true["c_d_press"].item())
            cd_fr_p = float(res_pred["c_d_fric"].item())
            cd_fr_t = float(res_true["c_d_fric"].item())

            cl_errs.append(abs(cl_p - cl_t) / (abs(cl_t) + 1e-4))
            cd_errs.append(abs(cd_p - cd_t) / (abs(cd_t) + 1e-4))
            cd_press_errs.append(abs(cd_pr_p - cd_pr_t) / (abs(cd_pr_t) + 1e-4))
            cd_fric_errs.append(abs(cd_fr_p - cd_fr_t) / (abs(cd_fr_t) + 1e-4))

            cp_p = res_pred["c_p"].squeeze().numpy()
            cp_t = res_true["c_p"].squeeze().numpy()
            cp_rmses.append(float(np.sqrt(np.mean((cp_p - cp_t) ** 2))))

        metrics["val_cl_rel_err"] = float(np.mean(cl_errs))
        metrics["val_cd_rel_err"] = float(np.mean(cd_errs))
        metrics["val_cd_press_rel_err"] = float(np.mean(cd_press_errs))
        metrics["val_cd_fric_rel_err"] = float(np.mean(cd_fric_errs))
        metrics["val_cp_rmse"] = float(np.mean(cp_rmses))

    return metrics
