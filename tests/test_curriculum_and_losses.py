"""
Unit Tests for Curriculum Schedule Manager, ReLoBRaLo Dynamic Loss Balancer, and Composite Loss Engine.
"""
import os
import sys
import numpy as np
import pytest
import scipy.spatial
import torch
from torch_geometric.data import Data

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.curriculum_manager import CurriculumManager
from src.losses.loss_balancer import ReLoBRaLoLossBalancer
from src.losses.sobolev_loss import SobolevGradientLoss
from src.losses.composite_loss import CompositeCurriculumLoss
from src.data.transforms import FlowFeatureNormalizer
from src.data.preprocessor import AirfRANSGraphPreprocessor


def test_curriculum_manager_phases():
    cfg = {
        "epochs": {
            "total": 250,
            "phase_1_end": 50,
            "phase_2_end": 150,
            "transition_width": 10
        },
        "loss_weights": {
            "phase_1": {"lambda_data": 1.0, "lambda_bc": 1.0, "lambda_pde": 0.0, "lambda_coeff": 0.0},
            "phase_2": {"lambda_data": 1.0, "lambda_bc": 1.0, "lambda_pde": 0.05, "pde_warmup_epochs": 20, "lambda_coeff": 0.0},
            "phase_3": {"lambda_data": 1.0, "lambda_bc": 1.0, "lambda_pde": 0.05, "lambda_coeff": 1.0}
        },
        "optimizer": {
            "lr_phase_1": 1e-3,
            "lr_phase_2": 5e-4,
            "lr_phase_3": 1e-4
        }
    }

    mgr = CurriculumManager(cfg)
    assert mgr.get_phase(1) == 1
    assert mgr.get_phase(50) == 1
    assert mgr.get_phase(51) == 2
    assert mgr.get_phase(150) == 2
    assert mgr.get_phase(151) == 3
    assert mgr.get_phase(250) == 3

    # Early Phase 1: PDE weight must be near zero
    p1_cfg = mgr.get_phase_config(10)
    assert p1_cfg["lambda_pde"] < 1e-4, "PDE weight must be ~0 in early Phase 1"

    # Deep Phase 3: coefficient weight must approach 1.0
    p3_cfg = mgr.get_phase_config(220)
    assert p3_cfg["lambda_coeff"] > 0.9, "Coeff weight must be near 1.0 deep in Phase 3"

    # Smooth monotonic ramp for lambda_pde across Phase 2 warmup
    p2_cfg_start = mgr.get_phase_config(55)
    p2_cfg_mid = mgr.get_phase_config(65)
    p2_cfg_end = mgr.get_phase_config(75)
    assert p2_cfg_start["lambda_pde"] <= p2_cfg_mid["lambda_pde"] <= p2_cfg_end["lambda_pde"]


def test_relobralo_loss_balancer():
    balancer = ReLoBRaLoLossBalancer(
        loss_names=["loss_data", "loss_bc", "loss_pde"],
        burn_in_epochs=3
    )

    dummy_losses = {
        "loss_data": torch.tensor(1.0),
        "loss_bc": torch.tensor(0.5),
        "loss_pde": torch.tensor(2.0)
    }

    # Burn-in epochs 1, 2
    w1 = balancer.update_weights(dummy_losses, current_epoch=1)
    assert w1["loss_data"] == 1.0

    w2 = balancer.update_weights(dummy_losses, current_epoch=2)
    assert w2["loss_data"] == 1.0

    # Post burn-in (epoch 4)
    mutated_losses = {
        "loss_data": torch.tensor(0.2),  # Decreased rapidly
        "loss_bc": torch.tensor(0.5),
        "loss_pde": torch.tensor(3.0)   # Stiff, increased
    }
    w4 = balancer.update_weights(mutated_losses, current_epoch=4)
    assert "loss_data" in w4 and "loss_pde" in w4
    assert all(balancer.min_multiplier <= v <= balancer.max_multiplier for v in w4.values())


def test_composite_loss_computation():
    normalizer = FlowFeatureNormalizer()
    # Dummy fit
    x = np.random.randn(100, 5).astype(np.float32)
    y = np.abs(np.random.randn(100, 4).astype(np.float32))
    normalizer.fit([x], [y])

    comp_loss = CompositeCurriculumLoss(normalizer=normalizer, use_relobralo=False)

    num_nodes = 80
    pos = np.random.randn(num_nodes, 2).astype(np.float32)
    edges = np.random.randint(0, num_nodes, size=(2, 300))

    preprocessor = AirfRANSGraphPreprocessor()
    ls_indices, ls_wx, ls_wy = preprocessor.compute_lsfv_weights(pos, edges)

    # Synthetic batch data
    batch_data = Data(
        x=torch.randn(num_nodes, 5),
        y=torch.randn(num_nodes, 4),
        y_norm=torch.randn(num_nodes, 4),
        pos=torch.from_numpy(pos),
        ls_indices=ls_indices,
        ls_wx=ls_wx,
        ls_wy=ls_wy,
        cond=torch.tensor([6.0, 0.0, 30.0], dtype=torch.float32)
    )

    y_pred_norm = torch.randn(num_nodes, 4, requires_grad=True)

    phase_config = {
        "lambda_data": 1.0,
        "lambda_bc": 1.0,
        "lambda_pde": 0.05,
        "lambda_grad": 0.1,
        "lambda_coeff": 0.0
    }

    loss_dict = comp_loss.compute_loss(
        y_pred_norm=y_pred_norm,
        batch_data=batch_data,
        phase=2,
        phase_config=phase_config,
        current_epoch=10
    )

    assert "loss" in loss_dict
    assert "loss_data" in loss_dict
    assert "loss_pde" in loss_dict
    assert "loss_grad" in loss_dict

    loss_dict["loss"].backward()
    assert y_pred_norm.grad is not None
