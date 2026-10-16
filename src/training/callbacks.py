"""
Training Callbacks: Model Checkpointing, Early Stopping, and Telemetry Logging.
"""
from typing import Dict, Optional
import os
import torch
import torch.nn as nn


class ModelCheckpoint:
    """
    Saves top-K best model weights based on validation performance, plus latest checkpoint.
    """
    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3
    ):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.saved_checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def is_better(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        normalizer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> bool:
        score = metrics.get(self.monitor, None)
        if score is None:
            return False

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pt")
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer_state_dict": normalizer.state_dict() if normalizer else None,
            "metrics": metrics
        }
        torch.save(state, latest_path)

        # Check if new best
        if self.is_better(score):
            self.best_score = score
            best_path = os.path.join(self.checkpoint_dir, f"best_model_epoch_{epoch}_{self.monitor}_{score:.4e}.pt")
            torch.save(state, best_path)
            # Maintain symbolic best link
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pt"))
            return True
        return False


class EarlyStopping:
    """
    Early stopping monitor to prevent overfitting or divergence.
    """
    def __init__(self, patience: int = 40, min_delta: float = 1e-5, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def step(self, score: float) -> bool:
        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class AerodynamicMetricsLogger:
    """
    Prints clean, structured aerodynamic summaries to console.
    """
    def log_epoch(self, epoch: int, phase: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        print(f"\n[Epoch {epoch:03d} | Phase {phase}]")
        print(f"  Train -> Loss: {train_metrics.get('loss', 0.0):.4e} | Data: {train_metrics.get('loss_data', 0.0):.4e} | PDE: {train_metrics.get('loss_pde', 0.0):.4e}")
        print(f"  Val   -> Rel L2 (Total): {val_metrics.get('val_rel_l2_total', 0.0):.4e} | Rel u_x: {val_metrics.get('val_rel_l2_ux', 0.0):.4e} | Rel p: {val_metrics.get('val_rel_l2_p', 0.0):.4e}")
        if "val_cd_rel_err" in val_metrics:
            print(f"  Aero  -> Cd Err: {val_metrics.get('val_cd_rel_err', 0.0)*100:.2f}% | Cl Err: {val_metrics.get('val_cl_rel_err', 0.0)*100:.2f}% | Cp RMSE: {val_metrics.get('val_cp_rmse', 0.0):.4e}")
