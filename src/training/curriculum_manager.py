"""
Curriculum Strategy Manager.
Handles phase transitions, physics loss warmup ramps, and phase-specific learning rates.
Smooth sigmoid transitions prevent gradient spikes at phase boundaries.
"""
from typing import Any, Dict
import math


class CurriculumManager:
    """
    Coordinates the 3-Phase Curriculum Optimization Schedule:
      Phase 1 (Epochs 1..50): Data-Driven Pretraining
      Phase 2 (Epochs 51..150): Physics & Sobolev Warmup
      Phase 3 (Epochs 151..250): Boundary Layer Refinement & Force Coefficients

    Phase transitions use a smooth sigmoid ramp over a configurable window
    to prevent gradient stiffness spikes at hard epoch boundaries.
    """
    def __init__(self, training_cfg: Dict[str, Any]):
        self.cfg = training_cfg
        self.epochs_cfg = training_cfg.get("epochs", {})
        self.p1_end = self.epochs_cfg.get("phase_1_end", 50)
        self.p2_end = self.epochs_cfg.get("phase_2_end", 150)
        self.total_epochs = self.epochs_cfg.get("total", 250)
        self.transition_width = self.epochs_cfg.get("transition_width", 10)

        self.loss_weights_cfg = training_cfg.get("loss_weights", {})
        self.opt_cfg = training_cfg.get("optimizer", {})

    @staticmethod
    def _sigmoid_ramp(epoch: float, center: float, width: float) -> float:
        """Smooth sigmoid transition: 0 well before center, 1 well after."""
        sigma = max(width / 6.0, 0.5)
        z = (epoch - center) / sigma
        z = max(-20.0, min(20.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def get_phase(self, epoch: int) -> int:
        """Return the current curriculum phase (1, 2, or 3)."""
        if epoch <= self.p1_end:
            return 1
        elif epoch <= self.p2_end:
            return 2
        else:
            return 3

    def get_phase_config(self, epoch: int) -> Dict[str, Any]:
        """
        Get active loss weights for the given epoch with smooth sigmoid
        transitions between phases. Each loss weight ramps from its previous
        phase value to its current phase target over the transition window.
        """
        w = self.transition_width

        p1_cfg = dict(self.loss_weights_cfg.get("phase_1", {}))
        p2_cfg = dict(self.loss_weights_cfg.get("phase_2", {}))
        p3_cfg = dict(self.loss_weights_cfg.get("phase_3", {}))

        # Sigmoid blend factor for Phase 1 -> Phase 2 transition
        alpha_12 = self._sigmoid_ramp(float(epoch), float(self.p1_end), w)
        # Sigmoid blend factor for Phase 2 -> Phase 3 transition
        alpha_23 = self._sigmoid_ramp(float(epoch), float(self.p2_end), w)

        all_keys = set(list(p1_cfg.keys()) + list(p2_cfg.keys()) + list(p3_cfg.keys()))
        blended = {}
        for key in all_keys:
            if key == "pde_warmup_epochs":
                blended[key] = p2_cfg.get(key, 20)
                continue

            v1 = p1_cfg.get(key, 0.0)
            v2 = p2_cfg.get(key, 0.0)
            v3 = p3_cfg.get(key, 0.0)

            # Two-stage blend: P1 -> P2 -> P3
            v_12 = v1 + alpha_12 * (v2 - v1)
            v_final = v_12 + alpha_23 * (v3 - v_12)
            blended[key] = v_final

        # PDE warmup ramp within Phase 2 region (on top of sigmoid transition)
        if alpha_12 > 0.01:
            warmup_len = p2_cfg.get("pde_warmup_epochs", 20)
            p2_start = self.p1_end + 1
            warmup_progress = min(1.0, max(0.0, (epoch - p2_start + 1) / float(warmup_len)))
            target_pde = blended.get("lambda_pde", 0.05)
            blended["lambda_pde"] = target_pde * warmup_progress

        return blended

    def get_target_lr(self, epoch: int) -> float:
        """Get base learning rate for current phase."""
        phase = self.get_phase(epoch)
        return float(self.opt_cfg.get(f"lr_phase_{phase}", 1e-3))
