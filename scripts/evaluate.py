#!/usr/bin/env python3
"""
Official AirfRANS Benchmark Evaluation & Baseline Comparison Generator.
Evaluates trained MS-PIGNO checkpoints on held-out test splits and outputs publication tables.
"""
import argparse
import os
import sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Ensure codebase root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splits import AirfRANSSplitter
from src.data.transforms import FlowFeatureNormalizer
from src.data.dataset import AirfRANSPyGDataset
from src.models.ms_pigno import MSPIGNO
from src.utils.metrics import evaluate_airfrans_predictions
from src.utils.visualization import plot_flow_contours, plot_surface_pressure_friction
from src.utils.logger import ExperimentLogger


def main():
    parser = argparse.ArgumentParser(description="Evaluate AirfRANS Surrogate Model on Benchmark Splits.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--task", type=str, default="full", choices=["full", "scarce", "reynolds_extrapolation", "aoa_extrapolation"])
    parser.add_argument("--raw_data_dir", type=str, default="./Dataset", help="Directory with raw AirfRANS data.")
    parser.add_argument("--cache_dir", type=str, default="./cache/preprocessed_airfrans", help="Preprocessed cache directory.")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_results", help="Directory to save evaluation reports and plots.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_plots", type=int, default=5, help="Number of test simulation cases to plot.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print("=" * 70)
    print("  AirfRANS Official SOTA Benchmark Evaluator")
    print(f"  Task: {args.task.upper()} | Checkpoint: {args.checkpoint}")
    print("=" * 70)

    # 1. Load Split & Normalizer
    splitter = AirfRANSSplitter(raw_data_dir=args.raw_data_dir)
    _, _, test_files = splitter.get_split_by_task_name(args.task)
    print(f"Loaded {len(test_files)} held-out test simulations for task {args.task}.")

    normalizer = FlowFeatureNormalizer()
    norm_path = os.path.join(args.cache_dir, f"normalizer_{args.task}.json")
    if os.path.exists(norm_path):
        normalizer.load_from_json(norm_path)
    else:
        print("[WARN] Specific normalizer JSON not found. Initializing from checkpoint state dict if available.")

    # 2. Instantiate Model and Load Checkpoint
    model = MSPIGNO(
        node_in_channels=5,
        pos_channels=2,
        edge_in_channels=8,
        cond_channels=3,
        out_channels=4,
        hidden_dims=(64, 128, 256),
        transformer_layers=6,
        num_slices=64,
        num_heads=8,
        use_bl_refinement=True
    ).to(device)

    chk = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in chk:
        model.load_state_dict(chk["model_state_dict"])
        if "normalizer_state_dict" in chk and chk["normalizer_state_dict"] is not None:
            normalizer.load_state_dict(chk["normalizer_state_dict"])
    else:
        model.load_state_dict(chk)

    model.eval()

    # 3. Create Test DataLoader
    test_dataset = AirfRANSPyGDataset(file_paths=test_files, cache_dir=args.cache_dir, normalizer=normalizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 4. Evaluation Loop
    predictions = []
    targets = []
    u_infs = []
    aoa_rads = []
    wall_data_list = []
    plot_cases = []

    print("Running inference across test simulations...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch = batch.to(device)
            y_pred_norm = model(
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
            y_pred_phys = normalizer.inverse_transform_y(y_pred_norm).cpu()
            y_true_phys = batch.y.cpu()

            predictions.append(y_pred_phys)
            targets.append(y_true_phys)
            u_inf = float(batch.cond[2].item()) if hasattr(batch, "cond") else 30.0
            aoa_rad = float(batch.cond[1].item()) if hasattr(batch, "cond") else 0.0
            u_infs.append(u_inf)
            aoa_rads.append(aoa_rad)

            if hasattr(batch, "wall_indices"):
                wall_dict = {
                    "wall_indices": batch.wall_indices.cpu(),
                    "wall_delta_s": batch.wall_delta_s.cpu(),
                    "wall_normals": batch.wall_normals.cpu(),
                    "wall_tangents": batch.wall_tangents.cpu(),
                    "ls_indices": batch.ls_indices.cpu(),
                    "ls_wx": batch.ls_wx.cpu(),
                    "ls_wy": batch.ls_wy.cpu(),
                    "pos": batch.pos.cpu()
                }
                wall_data_list.append(wall_dict)

            if i < args.num_plots:
                plot_cases.append((batch.pos.cpu().numpy(), y_true_phys.numpy(), y_pred_phys.numpy(), i))

    # 5. Compute Comprehensive Metrics
    metrics = evaluate_airfrans_predictions(
        predictions=predictions,
        targets=targets,
        u_infs=u_infs,
        aoa_rads=aoa_rads,
        wall_data=wall_data_list if len(wall_data_list) == len(predictions) else None
    )

    # 6. Save Markdown Summary & SOTA Benchmark Comparison
    logger = ExperimentLogger(log_dir=args.output_dir)
    report_path = os.path.join(args.output_dir, f"benchmark_report_{args.task}.md")
    logger.save_summary_table(task_name=args.task, final_metrics=metrics, output_path=report_path)

    # 7. Generate Visualization Figures
    print("Generating CFD field contour comparison plots...")
    for pos_np, y_t_np, y_p_np, case_idx in plot_cases:
        fig_path = os.path.join(args.output_dir, f"contour_comparison_case_{case_idx}.png")
        plot_flow_contours(
            pos=pos_np,
            y_true=y_t_np,
            y_pred=y_p_np,
            output_path=fig_path,
            title=f"AirfRANS Task {args.task.upper()} - Case {case_idx}"
        )

    print("\n" + "=" * 70)
    print(f"  BENCHMARK EVALUATION RESULTS ({args.task.upper()})")
    print("=" * 70)
    for k, v in metrics.items():
        if "rel_err" in k:
            print(f"  {k:25s}: {v*100:6.3f} %")
        else:
            print(f"  {k:25s}: {v:.6e}")
    print(f"\nSaved benchmark report to: {report_path}")
    print(f"Saved visualization plots to: {args.output_dir}/")


if __name__ == "__main__":
    main()
