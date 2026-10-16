"""
Unit Tests for AirfRANS Graph Preprocessor (LS-FV, Anisotropic Attributes, Hierarchical Coarsening).
"""
import os
import sys
import numpy as np
import pytest
import torch

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessor import AirfRANSGraphPreprocessor


def test_anisotropic_edge_attr_computation():
    preprocessor = AirfRANSGraphPreprocessor()
    num_nodes = 50
    pos = torch.randn(num_nodes, 2)
    edge_index = torch.randint(0, num_nodes, (2, 150))
    sdf = torch.rand(num_nodes, 1)
    normals = torch.randn(num_nodes, 2)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

    edge_attr = preprocessor.compute_anisotropic_edge_attr(pos, edge_index, sdf, normals)
    assert edge_attr.shape == (150, 8)
    assert not torch.isnan(edge_attr).any()


def test_multilevel_clustering_generation():
    preprocessor = AirfRANSGraphPreprocessor(l1_target_nodes=100, l2_target_nodes=20)
    num_nodes = 500
    pos = np.random.randn(num_nodes, 2).astype(np.float32)
    sdf = np.random.uniform(0.0, 0.5, (num_nodes, 1)).astype(np.float32)

    c_l0_l1, c_l1_l2, e_l1, e_l2 = preprocessor.compute_multilevel_clustering(pos, sdf)

    assert len(c_l0_l1) == num_nodes
    assert len(c_l1_l2) == 100
    assert e_l1.shape[0] == 2  # [2, E_l1]
    assert e_l2.shape[0] == 2  # [2, E_l2]


def test_wall_metadata_extraction():
    preprocessor = AirfRANSGraphPreprocessor(wall_phi_threshold=0.01)
    num_nodes = 300
    pos = np.random.randn(num_nodes, 2).astype(np.float32)
    sdf = np.random.uniform(0.0, 0.2, (num_nodes, 1)).astype(np.float32)
    # Put 30 nodes on the wall
    sdf[:30] = 0.005
    normals = np.random.randn(num_nodes, 2).astype(np.float32)
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    wall_meta = preprocessor.extract_wall_metadata(pos, sdf, normals)

    assert "wall_indices" in wall_meta
    assert "wall_delta_s" in wall_meta
    assert "wall_normals" in wall_meta
    assert "wall_tangents" in wall_meta

    n_wall = len(wall_meta["wall_indices"])
    assert wall_meta["wall_delta_s"].shape == (n_wall, 1)
    assert wall_meta["wall_normals"].shape == (n_wall, 2)
    assert wall_meta["wall_tangents"].shape == (n_wall, 2)
