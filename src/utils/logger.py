"""
Telemetry Logger for SciML Experiments.
"""
from typing import Any, Dict
import json
import os
import time


class ExperimentLogger:
    """
    Saves JSON logs, summaries, and Markdown benchmark tables.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = []

    def log(self, metrics: Dict[str, Any]):
        metrics_with_time = dict(metrics)
        metrics_with_time["timestamp"] = time.time()
        self.metrics_history.append(metrics_with_time)

        # Write to JSON Lines
        jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(metrics_with_time) + "\n")

    def save_summary_table(self, task_name: str, final_metrics: Dict[str, float], output_path: str):
        """Generates clean Markdown table for publication and reporting."""
        lines = [
            f"# AirfRANS Benchmark Results: Task {task_name.upper()}",
            f"**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| Metric | Value | Description |",
            "| :--- | :--- | :--- |",
            f"| **Relative $L_2$ Total** | `{final_metrics.get('val_rel_l2_total', 0.0):.4e}` | Average across 4 fields |",
            f"| **Relative $L_2$ $u_x$** | `{final_metrics.get('val_rel_l2_ux', 0.0):.4e}` | Horizontal velocity |",
            f"| **Relative $L_2$ $u_y$** | `{final_metrics.get('val_rel_l2_uy', 0.0):.4e}` | Vertical velocity |",
            f"| **Relative $L_2$ $p$** | `{final_metrics.get('val_rel_l2_p', 0.0):.4e}` | Kinematic pressure |",
            f"| **Relative $L_2$ $\\nu_t$** | `{final_metrics.get('val_rel_l2_nut', 0.0):.4e}` | Turbulent viscosity |",
            f"| **Surface $C_p$ RMSE** | `{final_metrics.get('val_cp_rmse', 0.0):.4e}` | Pressure distribution |",
            f"| **$C_L$ Relative Error** | `{final_metrics.get('val_cl_rel_err', 0.0)*100:.2f}%` | Lift coefficient |",
            f"| **$C_D$ Relative Error** | `{final_metrics.get('val_cd_rel_err', 0.0)*100:.2f}%` | Total drag |",
            f"| **$C_{{D,\\text{{press}}}}$ Relative Error** | `{final_metrics.get('val_cd_press_rel_err', 0.0)*100:.2f}%` | Form/Pressure drag |",
            f"| **$C_{{D,\\text{{fric}}}}$ Relative Error** | `{final_metrics.get('val_cd_fric_rel_err', 0.0)*100:.2f}%` | Skin-friction drag |"
        ]
        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")
