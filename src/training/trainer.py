"""
High-Performance AMP Training Engine for AirfRANS Neural Operator Surrogates.
Features micro-batching, gradient accumulation, 3-phase curriculum management, and AMP acceleration.
"""
from typing import Any, Dict, Optional
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .curriculum_manager import CurriculumManager
from .callbacks import ModelCheckpoint, EarlyStopping, AerodynamicMetricsLogger
from ..losses.composite_loss import CompositeCurriculumLoss
from ..data.transforms import FlowFeatureNormalizer
from ..utils.metrics import evaluate_airfrans_predictions


class SurrogateTrainer:
    """
    Production-grade training coordinator for MS-PIGNO on AirfRANS benchmarks.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        normalizer: FlowFeatureNormalizer,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = normalizer
        self.cfg = config
        self.device = device

        # Training & Hardware Hyperparameters
        train_cfg = config.get("training", {})
        self.total_epochs = train_cfg.get("max_epochs", train_cfg.get("epochs", {}).get("total", 250))
        self.limit_train_batches = train_cfg.get("limit_train_batches", None)
        self.limit_val_batches = train_cfg.get("limit_val_batches", None)
        self.accumulate_batches = train_cfg.get("batching", {}).get("accumulate_grad_batches", 4)
        self.max_norm = train_cfg.get("gradient_clipping", {}).get("max_norm", 1.0)

        # Precision & AMP
        self.amp_enabled = train_cfg.get("precision", {}).get("amp_enabled", True) and (device.type == "cuda")
        amp_dtype_str = train_cfg.get("precision", {}).get("amp_dtype", "bfloat16")
        self.amp_dtype = torch.bfloat16 if amp_dtype_str == "bfloat16" else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.amp_enabled and self.amp_dtype == torch.float16))

        # Curriculum & Loss Modules
        self.curriculum = CurriculumManager(train_cfg)
        self.loss_fn = CompositeCurriculumLoss(
            normalizer=normalizer,
            nu_laminar=config.get("fluid_constants", {}).get("nu_laminar", 1.5e-5),
            rho=config.get("fluid_constants", {}).get("rho", 1.0),
            chord=config.get("fluid_constants", {}).get("chord", 1.0),
            use_relobralo=train_cfg.get("balancing", {}).get("enabled", True)
        ).to(device)

        # Optimizer & Scheduler Setup
        base_lr = self.curriculum.get_target_lr(1)
        weight_decay = float(train_cfg.get("optimizer", {}).get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=train_cfg.get("scheduler", {}).get("t_0", 50),
            T_mult=train_cfg.get("scheduler", {}).get("t_mult", 2),
            eta_min=float(train_cfg.get("scheduler", {}).get("eta_min", 1e-6))
        )

        # Callbacks
        chk_dir = config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
        self.checkpoint = ModelCheckpoint(checkpoint_dir=chk_dir, monitor="val_rel_l2_total", mode="min")
        self.early_stopping = EarlyStopping(patience=40, mode="min")
        self.logger = AerodynamicMetricsLogger()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        phase = self.curriculum.get_phase(epoch)
        phase_config = self.curriculum.get_phase_config(epoch)

        epoch_losses = {"loss": 0.0, "loss_data": 0.0, "loss_pde": 0.0, "loss_grad": 0.0, "loss_coeff": 0.0}
        num_batches = len(self.train_loader)
        if self.limit_train_batches is not None:
            num_batches = min(num_batches, self.limit_train_batches)
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, total=num_batches, desc=f"Epoch {epoch:03d} (P{phase}) [Train]", leave=False)
        for i, batch in enumerate(pbar):
            if self.limit_train_batches is not None and i >= self.limit_train_batches:
                break
            batch = batch.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                # Forward Pass
                y_pred_norm = self.model(
                    x=batch.x_norm,
                    pos=batch.pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    cond=batch.cond_norm if hasattr(batch, "cond_norm") else batch.cond,
                    cluster_l0_l1=batch.cluster_l0_l1,
                    cluster_l1_l2=batch.cluster_l1_l2,
                    edge_index_l1=getattr(batch, "edge_index_l1", None),
                    edge_index_l2=getattr(batch, "edge_index_l2", None),
                    apply_physical_constraints=True
                )

                # Compute Composite Loss
                loss_dict = self.loss_fn.compute_loss(
                    y_pred_norm=y_pred_norm,
                    batch_data=batch,
                    phase=phase,
                    phase_config=phase_config,
                    current_epoch=epoch
                )
                loss = loss_dict["loss"] / self.accumulate_batches

            # Backward with AMP scaling
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Accumulation Step
            if (i + 1) % self.accumulate_batches == 0 or (i + 1) == num_batches:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            for k in epoch_losses.keys():
                epoch_losses[k] += float(loss_dict[k].detach().item())

            pbar.set_postfix({"L_tot": f"{float(loss_dict['loss'].item()):.4f}", "L_data": f"{float(loss_dict['loss_data'].item()):.4f}"})

        denom = max(1, num_batches)
        return {k: v / denom for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        val_predictions = []
        val_targets = []
        val_u_infs = []
        val_aoa_rads = []
        val_wall_data = []

        total_val_batches = len(self.val_loader)
        if self.limit_val_batches is not None:
            total_val_batches = min(total_val_batches, self.limit_val_batches)

        for i, batch in enumerate(tqdm(self.val_loader, total=total_val_batches, desc="Validation", leave=False)):
            if self.limit_val_batches is not None and i >= self.limit_val_batches:
                break
            batch = batch.to(self.device)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=self.amp_dtype):
                y_pred_norm = self.model(
                    x=batch.x_norm,
                    pos=batch.pos,
                    edge_index=batch.edge_index,
                    edge_attr=batch.edge_attr,
                    cond=batch.cond_norm if hasattr(batch, "cond_norm") else batch.cond,
                    cluster_l0_l1=batch.cluster_l0_l1,
                    cluster_l1_l2=batch.cluster_l1_l2,
                    edge_index_l1=getattr(batch, "edge_index_l1", None),
                    edge_index_l2=getattr(batch, "edge_index_l2", None),
                    apply_physical_constraints=True
                )
                y_pred_phys = self.normalizer.inverse_transform_y(y_pred_norm)

            val_predictions.append(y_pred_phys.cpu())
            val_targets.append(batch.y.cpu())
            u_inf = float(batch.cond[2].item()) if hasattr(batch, "cond") else 30.0
            aoa_rad = float(batch.cond[1].item()) if hasattr(batch, "cond") else 0.0
            val_u_infs.append(u_inf)
            val_aoa_rads.append(aoa_rad)

            if hasattr(batch, "wall_indices"):
                val_wall_data.append({
                    "wall_indices": batch.wall_indices.cpu(),
                    "wall_delta_s": batch.wall_delta_s.cpu(),
                    "wall_normals": batch.wall_normals.cpu(),
                    "wall_tangents": batch.wall_tangents.cpu(),
                    "ls_indices": batch.ls_indices.cpu(),
                    "ls_wx": batch.ls_wx.cpu(),
                    "ls_wy": batch.ls_wy.cpu(),
                    "pos": batch.pos.cpu()
                })

        metrics = evaluate_airfrans_predictions(
            predictions=val_predictions,
            targets=val_targets,
            u_infs=val_u_infs,
            aoa_rads=val_aoa_rads,
            wall_data=val_wall_data,
            nu_laminar=self.cfg.get("fluid_constants", {}).get("nu_laminar", 1.5e-5)
        )
        return metrics

    def fit(self) -> Dict[str, Any]:
        """
        Execute full multi-epoch curriculum training loop.
        """
        print(f"=== Starting AirfRANS MS-PIGNO Surrogate Training ===")
        print(f"Device: {self.device} | AMP: {self.amp_enabled} ({self.amp_dtype}) | Total Epochs: {self.total_epochs}")

        history = {"train_loss": [], "val_metrics": []}
        start_time = time.time()

        for epoch in range(1, self.total_epochs + 1):
            phase = self.curriculum.get_phase(epoch)
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            history["train_loss"].append(train_metrics)
            history["val_metrics"].append(val_metrics)

            self.logger.log_epoch(epoch, phase, train_metrics, val_metrics)
            self.checkpoint.step(self.model, self.optimizer, self.normalizer, epoch, val_metrics)

            if self.early_stopping.step(val_metrics["val_rel_l2_total"]):
                print(f"[INFO] Early stopping triggered at Epoch {epoch}!")
                break

        elapsed = time.time() - start_time
        print(f"\n[DONE] Training complete in {elapsed/3600.0:.2f} hours. Best Score: {self.checkpoint.best_score:.4e}")
        return history
