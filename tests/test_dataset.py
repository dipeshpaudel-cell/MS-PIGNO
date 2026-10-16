"""
Unit Tests for Data Pipeline, Feature Normalizers, and Task Splitters.
"""
import os
import sys
import numpy as np
import pytest
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.transforms import FlowFeatureNormalizer, EddyViscosityLogTransform
from src.data.splits import AirfRANSSplitter


def test_eddy_viscosity_log_transform():
    nu_lam = 1.5e-5
    transform = EddyViscosityLogTransform(nu_laminar=nu_lam)

    # Test exact zero -> log10(0 + 1) = 0.0
    zero_val = torch.tensor([0.0])
    trans_zero = transform.forward(zero_val)
    assert torch.isclose(trans_zero, torch.tensor([0.0])), "Zero eddy viscosity must map to 0.0"

    # Test physical eddy viscosity recovery
    nu_t_phys = torch.tensor([1.5e-5, 1.5e-4, 1.5e-2])
    transformed = transform.forward(nu_t_phys)
    recovered = transform.inverse(transformed)
    assert torch.allclose(nu_t_phys, recovered, rtol=1e-5), "Inverse transform must recover exact input"


def test_flow_feature_normalizer():
    normalizer = FlowFeatureNormalizer(nu_laminar=1.5e-5)

    # Create dummy dataset of 2 simulations
    x1 = np.random.randn(500, 5).astype(np.float32)
    x2 = np.random.randn(600, 5).astype(np.float32)
    y1 = np.abs(np.random.randn(500, 4).astype(np.float32))
    y2 = np.abs(np.random.randn(600, 4).astype(np.float32))

    normalizer.fit(x_list=[x1, x2], y_list=[y1, y2])
    assert normalizer.fitted is True

    # Test transformation and inverse
    y_tensor = torch.from_numpy(y1)
    y_norm = normalizer.transform_y(y_tensor)
    y_rec = normalizer.inverse_transform_y(y_norm)

    assert torch.allclose(y_tensor[:, :3], y_rec[:, :3], atol=1e-4), "Velocity and pressure recovery failed"
    assert torch.allclose(y_tensor[:, 3:], y_rec[:, 3:], atol=1e-4), "Eddy viscosity recovery failed"


def test_splitter_integrity():
    # Test task dispatcher with dummy directory
    splitter = AirfRANSSplitter(raw_data_dir="./Dataset", seed=42)
    for task_name in ["full", "scarce", "reynolds_extrapolation", "aoa_extrapolation"]:
        train, val, test = splitter.get_split_by_task_name(task_name)
        assert isinstance(train, list)
        assert isinstance(val, list)
        assert isinstance(test, list)
        # Ensure zero leakage between train and test
        overlap = set(train).intersection(set(test))
        assert len(overlap) == 0, f"Data leakage detected in task {task_name}!"
