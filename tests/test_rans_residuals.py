"""
Unit Tests for Incompressible RANS Momentum & Continuity PDE Residual Evaluator.
Validates physical residual computation and loss tensor shapes on synthetic mock flow fields.
"""
import os
import sys
import numpy as np
import pytest
import scipy.spatial
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import AirfRANSGraphPreprocessor
from src.physics.rans_residuals import RANSResidualEvaluator


def test_rans_residuals_evaluation():
    # Construct synthetic 2D point cloud and Delaunay mesh
    nx, ny = 25, 25
    gx, gy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    pos = np.stack([gx.flatten(), gy.flatten()], axis=1).astype(np.float32)
    num_nodes = len(pos)

    tri = scipy.spatial.Delaunay(pos)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((simplex[i], simplex[j]))
    edge_index = np.array(list(edges)).T

    # Compute LS-FV weights
    preprocessor = AirfRANSGraphPreprocessor()
    ls_indices, ls_wx, ls_wy = preprocessor.compute_lsfv_weights(pos, edge_index)

    evaluator = RANSResidualEvaluator(nu_laminar=1.5e-5, rho=1.0)

    # Synthetic prediction: [u_x, u_y, p, nu_t]
    y_pred = torch.randn(num_nodes, 4, dtype=torch.float32, requires_grad=True)
    # Synthetic signed distance field (SDF)
    sdf = torch.tensor(pos[:, 1:2], dtype=torch.float32)

    res_dict = evaluator.evaluate_residuals(
        y_pred=y_pred,
        ls_indices=ls_indices,
        ls_wx=ls_wx,
        ls_wy=ls_wy,
        sdf=sdf,
        fluid_sdf_threshold=0.01
    )

    assert "res_cont" in res_dict
    assert "res_mom_x" in res_dict
    assert "res_mom_y" in res_dict
    assert "loss_cont" in res_dict
    assert "loss_mom_x" in res_dict
    assert "loss_mom_y" in res_dict
    assert "pde_loss" in res_dict

    assert res_dict["res_cont"].shape == (num_nodes, 1)
    assert res_dict["res_mom_x"].shape == (num_nodes, 1)
    assert res_dict["res_mom_y"].shape == (num_nodes, 1)
    assert res_dict["pde_loss"].dim() == 0  # Scalar loss

    # Test backward autograd through PDE loss
    pde_loss = res_dict["pde_loss"]
    pde_loss.backward()
    assert y_pred.grad is not None
    assert torch.sum(torch.abs(y_pred.grad)) > 0.0
