"""
Unit Tests for Boundary Layer Refinement Head and log(y+) Fourier Positional Embeddings.
"""
import os
import sys
import pytest
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.bl_refinement_head import LogYPlusFourierEmbedding, BoundaryLayerRefinementHead


def test_log_y_plus_fourier_embedding():
    num_modes = 16
    delta_bl = 0.05
    embedder = LogYPlusFourierEmbedding(num_modes=num_modes, delta_bl=delta_bl)

    # 100 points in near-wall region
    phi = torch.linspace(0.0, 0.05, 100).unsqueeze(-1)
    feats = embedder(phi)

    assert feats.shape == (100, 2 * num_modes)
    assert not torch.isnan(feats).any()
    assert not torch.isinf(feats).any()


def test_boundary_layer_refinement_head_forward():
    in_channels = 64
    skip_channels = 64
    out_channels = 4
    num_nodes = 200
    delta_bl = 0.05

    bl_head = BoundaryLayerRefinementHead(
        in_channels=in_channels,
        skip_channels=skip_channels,
        out_channels=out_channels,
        hidden_dim=128,
        num_layers=3,
        num_fourier_modes=16,
        delta_bl=delta_bl
    )

    h_0 = torch.randn(num_nodes, in_channels)
    h_skip = torch.randn(num_nodes, skip_channels)
    # Ensure mixture of nodes inside and outside the BL
    phi = torch.cat([torch.linspace(0.0, 0.04, 100), torch.linspace(0.06, 1.0, 100)]).unsqueeze(-1)
    normals = torch.randn(num_nodes, 2)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    gamma = torch.ones(1, 128)
    beta = torch.zeros(1, 128)

    delta_y = bl_head(h_0, phi, normals, gamma=gamma, beta=beta, h_skip=h_skip)

    assert delta_y.shape == (num_nodes, out_channels)
    # Far-field nodes (phi >= delta_bl) must receive zero residual correction
    far_field_correction = delta_y[100:]
    assert torch.allclose(far_field_correction, torch.zeros_like(far_field_correction)), "Far-field corrections must be zero"


def test_boundary_layer_refinement_head_no_skip():
    """Backward compatibility: BL head with skip_channels=0 (no skip connection)."""
    in_channels = 64
    out_channels = 4
    num_nodes = 100
    delta_bl = 0.05

    bl_head = BoundaryLayerRefinementHead(
        in_channels=in_channels,
        skip_channels=0,
        out_channels=out_channels,
        hidden_dim=128,
        num_layers=3,
        num_fourier_modes=16,
        delta_bl=delta_bl
    )

    h_0 = torch.randn(num_nodes, in_channels)
    phi = torch.linspace(0.0, 0.1, num_nodes).unsqueeze(-1)
    normals = torch.randn(num_nodes, 2)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    delta_y = bl_head(h_0, phi, normals)
    assert delta_y.shape == (num_nodes, out_channels)
