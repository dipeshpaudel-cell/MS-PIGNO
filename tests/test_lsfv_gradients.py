"""
Unit Tests for Least-Squares Finite-Volume (LS-FV) Spatial Gradient Operator.
Validates numerical derivatives against exact analytical trigonometric vortex fields.
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
from src.physics.lsfv_gradients import LSFVSparseGradientOperator


def test_lsfv_gradient_analytical_accuracy():
    # 1. Create a 2D perturbed unstructured point cloud
    nx, ny = 35, 35
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    pos = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    # Add random perturbation to make it unstructured
    np.random.seed(42)
    pos += np.random.uniform(-0.005, 0.005, size=pos.shape)
    num_nodes = len(pos)

    # Build Delaunay triangulation graph
    tri = scipy.spatial.Delaunay(pos)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    edges.add((simplex[i], simplex[j]))
    edge_index = np.array(list(edges)).T

    # 2. Compute LS-FV weights
    preprocessor = AirfRANSGraphPreprocessor()
    ls_indices, ls_wx, ls_wy = preprocessor.compute_lsfv_weights(pos, edge_index)

    # 3. Define analytical test function: f(x, y) = sin(2*pi*x) * cos(2*pi*y)
    x = pos[:, 0]
    y = pos[:, 1]
    f_val = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    df_dx_exact = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)
    df_dy_exact = -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

    # 4. Evaluate via GPU/CPU SpMM operator
    grad_op = LSFVSparseGradientOperator()
    f_tensor = torch.tensor(f_val, dtype=torch.float32).unsqueeze(-1)

    df_dx_num, df_dy_num = grad_op(f_tensor, ls_indices, ls_wx, ls_wy, num_nodes)

    # Check interior points (ignore boundary boundary-stencil effects)
    interior_mask = (x > 0.1) & (x < 0.9) & (y > 0.1) & (y < 0.9)
    df_dx_num_int = df_dx_num.squeeze().numpy()[interior_mask]
    df_dy_num_int = df_dy_num.squeeze().numpy()[interior_mask]

    err_x = np.mean(np.abs(df_dx_num_int - df_dx_exact[interior_mask]))
    err_y = np.mean(np.abs(df_dy_num_int - df_dy_exact[interior_mask]))

    assert err_x < 0.25, f"LS-FV X-gradient error too high: {err_x:.4f}"
    assert err_y < 0.25, f"LS-FV Y-gradient error too high: {err_y:.4f}"
    print(f"LS-FV analytical gradient test passed! (Mean interior error: x={err_x:.4e}, y={err_y:.4e})")
