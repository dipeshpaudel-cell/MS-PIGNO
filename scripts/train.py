#!/usr/bin/env python3
"""
Main Training CLI for AirfRANS Surrogate Models.
Orchestrated via Hydra configuration management.
"""
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch_geometric.loader import DataLoader

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splits import AirfRANSSplitter
from src.data.transforms import FlowFeatureNormalizer
from src.data.dataset import AirfRANSPyGDataset
from src.models.ms_pigno import MSPIGNO
from src.training.trainer import SurrogateTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("=" * 70)
    print("  AirfRANS MS-PIGNO Surrogate Model Training Engine")
    print(f"  Task: {cfg.task.name.upper()} | Model: {cfg.model.name.upper()}")
    print("=" * 70)

    # Set seeds & system configurations
    torch.manual_seed(cfg.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.system.seed)
        torch.set_float32_matmul_precision(cfg.system.matmul_precision)

    device = torch.device(cfg.system.device if torch.cuda.is_available() and cfg.system.device == "cuda" else "cpu")

    # 1. Dataset Partitioning (Strict Zero-Leakage)
    splitter = AirfRANSSplitter(raw_data_dir=cfg.paths.raw_data_dir, seed=cfg.system.seed)
    train_files, val_files, test_files = splitter.get_split_by_task_name(cfg.task.name)
    print(f"Data Splits -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    # 2. Fit Normalizer exclusively on Training Split
    normalizer_path = os.path.join(cfg.paths.cache_dir, f"normalizer_{cfg.task.name}.json")
    normalizer = FlowFeatureNormalizer(nu_laminar=cfg.fluid_constants.nu_laminar)

    if os.path.exists(normalizer_path):
        print(f"Loading existing normalization statistics from {normalizer_path}")
        normalizer.load_from_json(normalizer_path)
    else:
        print("Computing zero-leakage normalization statistics on training files...")
        os.makedirs(cfg.paths.cache_dir, exist_ok=True)
        # Sample training features to fit normalizer
        x_samples, y_samples, cond_samples = [], [], []
        sample_subset = train_files[:min(100, len(train_files))]
        for f in sample_subset:
            try:
                raw_d = torch.load(f, map_location="cpu", weights_only=False)
                x_samples.append(raw_d.x.numpy() if hasattr(raw_d, "x") else raw_d["x"].numpy())
                y_samples.append(raw_d.y.numpy() if hasattr(raw_d, "y") else raw_d["y"].numpy())
                if hasattr(raw_d, "cond"):
                    cond_samples.append(raw_d.cond.numpy())
            except Exception:
                continue

        normalizer.fit(x_list=x_samples, y_list=y_samples, cond_list=cond_samples if cond_samples else None)
        normalizer.save_to_json(normalizer_path)
        print(f"Saved normalization statistics to {normalizer_path}")

    # 3. Create Datasets and DataLoaders
    train_dataset = AirfRANSPyGDataset(file_paths=train_files, cache_dir=cfg.paths.cache_dir, normalizer=normalizer)
    val_dataset = AirfRANSPyGDataset(file_paths=val_files, cache_dir=cfg.paths.cache_dir, normalizer=normalizer)

    # Strictly micro-batched with batch_size=1
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batching.batch_size,
        shuffle=True,
        num_workers=cfg.training.batching.num_workers,
        pin_memory=cfg.system.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=cfg.system.pin_memory
    )

    # 4. Initialize Neural Operator Architecture
    model = MSPIGNO(
        node_in_channels=cfg.model.inputs.node_in_channels,
        pos_channels=cfg.model.inputs.pos_channels,
        edge_in_channels=cfg.model.inputs.edge_in_channels,
        cond_channels=cfg.model.inputs.cond_channels,
        out_channels=cfg.model.inputs.out_channels,
        hidden_dims=(cfg.model.hierarchy.encoder_channels[0], cfg.model.hierarchy.encoder_channels[1], cfg.model.hierarchy.encoder_channels[2]),
        transformer_layers=cfg.model.transformer.num_layers,
        num_slices=cfg.model.transformer.num_slices,
        num_heads=cfg.model.transformer.num_heads,
        use_bl_refinement=cfg.model.bl_refinement.enabled,
        bl_delta=cfg.model.bl_refinement.sdf_threshold,
        wall_decay_delta=cfg.model.constraints.wall_no_slip.decay_delta,
        dropout=cfg.model.transformer.dropout
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: MS-PIGNO | Trainable Parameters: {total_params:,}")

    # 5. Launch Training Pipeline
    trainer = SurrogateTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        config=OmegaConf.to_container(cfg, resolve=True),
        device=device
    )

    history = trainer.fit()
    print("Training successfully finished.")


if __name__ == "__main__":
    main()
