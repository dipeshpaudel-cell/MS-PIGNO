"""
Dynamic Multi-Objective Loss Balancing (ReLoBRaLo).
Prevents gradient pathologies and balances stiff PDE and Sobolev losses against data losses.
"""
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class ReLoBRaLoLossBalancer(nn.Module):
    """
    Relative Loss Balancing with Residual Annealing (Bischof & Kraus).
    Tracks exponential moving average of relative loss progression ratios.
    """
    def __init__(
        self,
        loss_names: List[str],
        alpha: float = 0.999,
        temperature: float = 0.1,
        burn_in_epochs: int = 5,
        max_multiplier: float = 10.0,
        min_multiplier: float = 0.05,
        eps: float = 1e-8
    ):
        super().__init__()
        self.loss_names = loss_names
        self.alpha = alpha
        self.temperature = temperature
        self.burn_in_epochs = burn_in_epochs
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.eps = eps

        # Persistent loss history
        for name in loss_names:
            self.register_buffer(f"l0_{name}", torch.tensor(0.0))
            self.register_buffer(f"l_prev_{name}", torch.tensor(0.0))
            self.register_buffer(f"weight_{name}", torch.tensor(1.0))

        self.initialized = False

    def update_weights(self, loss_dict: Dict[str, torch.Tensor], current_epoch: int) -> Dict[str, float]:
        """
        Update loss balancing weights based on current epoch loss values.
        """
        if current_epoch < self.burn_in_epochs:
            # During burn-in, record baseline L0 losses
            for name in self.loss_names:
                if name in loss_dict:
                    val = float(loss_dict[name].detach().item())
                    getattr(self, f"l0_{name}").copy_(torch.tensor(max(val, self.eps)))
                    getattr(self, f"l_prev_{name}").copy_(torch.tensor(max(val, self.eps)))
            self.initialized = True
            return {name: 1.0 for name in self.loss_names}

        # Calculate relative loss changes: rho_k = L_k(t) / (L_k(0) + eps)
        ratios = {}
        for name in self.loss_names:
            if name in loss_dict:
                val = float(loss_dict[name].detach().item())
                l0 = float(getattr(self, f"l0_{name}").item())
                ratios[name] = max(val, self.eps) / (l0 + self.eps)

        total_ratio = sum(ratios.values()) + self.eps
        num_terms = len(ratios)

        # Softmax / Normalized relative weights
        updated_weights = {}
        for name in self.loss_names:
            if name in ratios:
                target_w = (ratios[name] / total_ratio) * num_terms
                target_w = max(self.min_multiplier, min(self.max_multiplier, target_w))

                prev_w = float(getattr(self, f"weight_{name}").item())
                # Exponential smoothing
                new_w = self.alpha * prev_w + (1.0 - self.alpha) * target_w
                getattr(self, f"weight_{name}").copy_(torch.tensor(new_w))
                updated_weights[name] = new_w
            else:
                updated_weights[name] = 1.0

        return updated_weights

    def get_weights(self) -> Dict[str, float]:
        return {name: float(getattr(self, f"weight_{name}").item()) for name in self.loss_names}
