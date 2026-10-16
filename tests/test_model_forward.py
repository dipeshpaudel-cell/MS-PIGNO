"""
Unit Tests for MS-PIGNO Architecture, Slicing Transformer, and Hard Boundary Masks.
"""
import os
import sys
import pytest
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.ms_pigno import MSPIGNO
from src.models.slicing_transformer import PhysicsAwareSlicingTransformer
from src.models.film_conditioning import FiLMGenerator


def test_film_generator():
    film = FiLMGenerator(cond_channels=3, target_dim=128)
    cond = torch.tensor([[6.5, 0.1, 45.0]], dtype=torch.float32)
    gamma, beta = film(cond)

    assert gamma.shape == (1, 128)
    assert beta.shape == (1, 128)
    # Check identity initialization: gamma ~ 1, beta ~ 0
    assert torch.allclose(gamma, torch.ones_like(gamma), atol=1e-5)
    assert torch.allclose(beta, torch.zeros_like(beta), atol=1e-5)


def test_slicing_transformer():
    num_coarse_nodes = 128
    hidden_dim = 64
    transformer = PhysicsAwareSlicingTransformer(
        hidden_dim=hidden_dim,
        num_slices=16,
        num_layers=2,
        num_heads=4
    )
    tokens = torch.randn(num_coarse_nodes, hidden_dim)
    out = transformer(tokens)
    assert out.shape == (num_coarse_nodes, hidden_dim)


def test_ms_pigno_forward_and_backward():
    num_nodes = 300
    num_edges = 1500
    num_l1 = 40
    num_l2 = 10

    # Synthetic mesh inputs
    x = torch.randn(num_nodes, 5)
    # Ensure some nodes are exactly on the wall (phi = 0)
    x[:20, 2] = 0.0
    x[20:, 2] = torch.rand(num_nodes - 20) * 0.2

    pos = torch.randn(num_nodes, 2)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 8)
    cond = torch.tensor([6.5, 0.05, 30.0], dtype=torch.float32)

    cluster_l0_l1 = torch.randint(0, num_l1, (num_nodes,))
    cluster_l1_l2 = torch.randint(0, num_l2, (num_l1,))

    model = MSPIGNO(
        node_in_channels=5,
        pos_channels=2,
        edge_in_channels=8,
        cond_channels=3,
        out_channels=4,
        hidden_dims=(32, 64, 128),
        transformer_layers=2,
        num_slices=8,
        num_heads=4,
        use_bl_refinement=True
    )

    y_pred = model(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cond=cond,
        cluster_l0_l1=cluster_l0_l1,
        cluster_l1_l2=cluster_l1_l2,
        apply_physical_constraints=True
    )

    # 1. Shape Verification
    assert y_pred.shape == (num_nodes, 4), f"Expected shape (300, 4), got {y_pred.shape}"

    # 2. Hard No-Slip Wall Mask Verification
    wall_vel = y_pred[:20, 0:2]
    assert torch.allclose(wall_vel, torch.zeros_like(wall_vel), atol=1e-6), "No-slip wall velocity must be zero at phi=0"

    # 3. Turbulent Viscosity Strict Positivity
    nu_t = y_pred[:, 3]
    assert torch.all(nu_t >= 0.0), "Eddy viscosity must be strictly positive"

    # 4. Backward Pass Verification
    loss = torch.sum(y_pred ** 2)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for parameter {name}"
