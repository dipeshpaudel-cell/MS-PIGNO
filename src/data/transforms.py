"""
Feature Normalization and Target Transforms for AirfRANS Datasets.
Implements Z-score standardization and dimensionless log-scaling for eddy viscosity.
"""
from typing import Dict, Optional, Tuple, Union
import json
import numpy as np
import torch
import torch.nn as nn


class EddyViscosityLogTransform:
    """
    Dimensionless logarithmic transform for turbulent eddy viscosity nu_t:
        nu_tilde = log10(nu_t / nu_laminar + 1.0)
    
    Compresses dynamic range from 5 orders of magnitude down to [0.0, 3.5].
    """
    def __init__(self, nu_laminar: float = 1.5e-5):
        self.nu_laminar = float(nu_laminar)

    def forward(self, nu_t: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(nu_t, torch.Tensor):
            nu_clipped = torch.clamp(nu_t, min=0.0)
            return torch.log10(nu_clipped / self.nu_laminar + 1.0)
        else:
            nu_clipped = np.clip(nu_t, a_min=0.0, a_max=None)
            return np.log10(nu_clipped / self.nu_laminar + 1.0)

    def inverse(self, nu_tilde: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(nu_tilde, torch.Tensor):
            return self.nu_laminar * (torch.pow(10.0, torch.clamp(nu_tilde, min=0.0)) - 1.0)
        else:
            return self.nu_laminar * (np.power(10.0, np.clip(nu_tilde, a_min=0.0, a_max=None)) - 1.0)


class FlowFeatureNormalizer(nn.Module):
    """
    Comprehensive normalizer for inputs, targets, conditions, and geometric features.
    Guarantees strict zero-leakage training by computing statistics solely on train splits.
    """
    def __init__(self, nu_laminar: float = 1.5e-5, eps: float = 1e-8):
        super().__init__()
        self.nu_laminar = nu_laminar
        self.eps = eps
        self.eddy_transform = EddyViscosityLogTransform(nu_laminar=nu_laminar)

        # Register persistent buffers for PyTorch module compatibility
        self.register_buffer("x_mean", torch.zeros(5, dtype=torch.float32))
        self.register_buffer("x_std", torch.ones(5, dtype=torch.float32))
        self.register_buffer("y_mean", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("y_std", torch.ones(4, dtype=torch.float32))
        self.register_buffer("pos_mean", torch.zeros(2, dtype=torch.float32))
        self.register_buffer("pos_std", torch.ones(2, dtype=torch.float32))
        self.register_buffer("cond_mean", torch.zeros(3, dtype=torch.float32))
        self.register_buffer("cond_std", torch.ones(3, dtype=torch.float32))

        self.fitted = False

    def fit(
        self,
        x_list: list,
        y_list: list,
        pos_list: Optional[list] = None,
        cond_list: Optional[list] = None
    ) -> "FlowFeatureNormalizer":
        """
        Fit normalization statistics on an aggregated list of training arrays/tensors.
        """
        # Node features X: [U_inf_x, U_inf_y, phi, n_x, n_y]
        x_all = np.concatenate([np.asarray(x, dtype=np.float32) for x in x_list], axis=0)
        self.x_mean.copy_(torch.from_numpy(np.mean(x_all, axis=0)))
        self.x_std.copy_(torch.from_numpy(np.std(x_all, axis=0) + self.eps))

        # Target fields Y: [u_x, u_y, p, nu_t]
        y_all = np.concatenate([np.asarray(y, dtype=np.float32) for y in y_list], axis=0)
        # Apply log-transform to nu_t (channel 3) before calculating Z-score
        y_transformed = y_all.copy()
        y_transformed[:, 3] = self.eddy_transform.forward(y_all[:, 3])

        self.y_mean.copy_(torch.from_numpy(np.mean(y_transformed, axis=0)))
        self.y_std.copy_(torch.from_numpy(np.std(y_transformed, axis=0) + self.eps))

        # Spatial Coordinates (x, y)
        if pos_list is not None and len(pos_list) > 0:
            pos_all = np.concatenate([np.asarray(p, dtype=np.float32) for p in pos_list], axis=0)
            self.pos_mean.copy_(torch.from_numpy(np.mean(pos_all, axis=0)))
            self.pos_std.copy_(torch.from_numpy(np.std(pos_all, axis=0) + self.eps))

        # Global Conditions [log10(Re), AoA_rad, U_inf_mag]
        if cond_list is not None and len(cond_list) > 0:
            cond_all = np.stack([np.asarray(c, dtype=np.float32) for c in cond_list], axis=0)
            self.cond_mean.copy_(torch.from_numpy(np.mean(cond_all, axis=0)))
            self.cond_std.copy_(torch.from_numpy(np.std(cond_all, axis=0) + self.eps))

        self.fitted = True
        return self

    def transform_x(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize input node features."""
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def inverse_transform_x(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse-transform standardized node features to physical space."""
        return (x_norm * self.x_std.to(x_norm.device)) + self.x_mean.to(x_norm.device)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """
        Transform raw physical targets [u_x, u_y, p, nu_t] into standardized normalized space.
        """
        device = y.device
        y_trans = y.clone()
        # Apply log-scaling to turbulent viscosity
        y_trans[:, 3] = self.eddy_transform.forward(y[:, 3])
        return (y_trans - self.y_mean.to(device)) / self.y_std.to(device)

    def inverse_transform_y(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        Inverse-transform normalized network predictions to physical fields [u_x, u_y, p, nu_t].
        """
        device = y_norm.device
        y_phys = (y_norm * self.y_std.to(device)) + self.y_mean.to(device)
        # Apply inverse log-transform to recover physical nu_t
        y_phys_out = y_phys.clone()
        y_phys_out[:, 3] = self.eddy_transform.inverse(y_phys[:, 3])
        return y_phys_out

    def transform_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Standardize global operating condition vector."""
        return (cond - self.cond_mean.to(cond.device)) / self.cond_std.to(cond.device)

    def save_to_json(self, json_path: str):
        """Export normalization statistics to JSON."""
        stats = {
            "x_mean": self.x_mean.cpu().tolist(),
            "x_std": self.x_std.cpu().tolist(),
            "y_mean": self.y_mean.cpu().tolist(),
            "y_std": self.y_std.cpu().tolist(),
            "pos_mean": self.pos_mean.cpu().tolist(),
            "pos_std": self.pos_std.cpu().tolist(),
            "cond_mean": self.cond_mean.cpu().tolist(),
            "cond_std": self.cond_std.cpu().tolist(),
            "nu_laminar": self.nu_laminar,
            "fitted": self.fitted
        }
        with open(json_path, "w") as f:
            json.dump(stats, f, indent=2)

    def load_from_json(self, json_path: str) -> "FlowFeatureNormalizer":
        """Load normalization statistics from JSON."""
        with open(json_path, "r") as f:
            stats = json.load(f)
        self.x_mean.copy_(torch.tensor(stats["x_mean"], dtype=torch.float32))
        self.x_std.copy_(torch.tensor(stats["x_std"], dtype=torch.float32))
        self.y_mean.copy_(torch.tensor(stats["y_mean"], dtype=torch.float32))
        self.y_std.copy_(torch.tensor(stats["y_std"], dtype=torch.float32))
        self.pos_mean.copy_(torch.tensor(stats["pos_mean"], dtype=torch.float32))
        self.pos_std.copy_(torch.tensor(stats["pos_std"], dtype=torch.float32))
        self.cond_mean.copy_(torch.tensor(stats["cond_mean"], dtype=torch.float32))
        self.cond_std.copy_(torch.tensor(stats["cond_std"], dtype=torch.float32))
        self.nu_laminar = stats.get("nu_laminar", 1.5e-5)
        self.eddy_transform = EddyViscosityLogTransform(self.nu_laminar)
        self.fitted = stats.get("fitted", True)
        return self
