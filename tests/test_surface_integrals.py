"""
Unit Tests for Differentiable Surface Force Integrator.
Validates force coefficient equations and backward autograd gradient flow.
"""
import os
import sys
import numpy as np
import pytest
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.physics.surface_integrals import DifferentiableSurfaceIntegrator
from src.data.preprocessor import AirfRANSGraphPreprocessor


def test_surface_integrator_differentiability():
    num_nodes = 200
    pos = np.random.randn(num_nodes, 2).astype(np.float32)
    edges = np.random.randint(0, num_nodes, size=(2, 800))

    preprocessor = AirfRANSGraphPreprocessor()
    ls_indices, ls_wx, ls_wy = preprocessor.compute_lsfv_weights(pos, edges)

    # Dummy wall surface (50 nodes)
    wall_idx = torch.arange(50, dtype=torch.long)
    wall_ds = torch.full((50, 1), 0.02, dtype=torch.float32)
    wall_n = torch.zeros(50, 2, dtype=torch.float32)
    wall_n[:, 1] = 1.0 # Normals pointing in +y
    wall_t = torch.zeros(50, 2, dtype=torch.float32)
    wall_t[:, 0] = 1.0 # Tangents pointing in +x

    integrator = DifferentiableSurfaceIntegrator()

    # Differentiable prediction tensor
    y_pred = torch.randn(num_nodes, 4, requires_grad=True)

    forces = integrator.integrate_forces(
        y_pred=y_pred,
        wall_indices=wall_idx,
        wall_delta_s=wall_ds,
        wall_normals=wall_n,
        wall_tangents=wall_t,
        u_inf=30.0,
        ls_indices=ls_indices,
        ls_wx=ls_wx,
        ls_wy=ls_wy,
        aoa_rad=0.05
    )

    assert "c_l" in forces
    assert "c_d_press" in forces
    assert "c_d_fric" in forces
    assert "c_d_total" in forces

    # Test backward pass gradient flow
    total_drag = forces["c_d_total"]
    total_drag.backward()

    assert y_pred.grad is not None, "Gradient must propagate to y_pred"
    assert torch.sum(torch.abs(y_pred.grad)) > 0.0, "Gradient through force integrator must be non-zero"
