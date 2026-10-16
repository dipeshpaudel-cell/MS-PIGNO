"""
CFD Visualization Tools: Flow Field Contours and Surface Aerodynamic Distributions.
"""
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_flow_contours(
    pos: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "AirfRANS Flow Field Comparison"
):
    """
    Renders 4-row comparison: [u_x, u_y, p, nu_t] (Ground Truth, Prediction, Absolute Error).
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), dpi=150)
    field_names = [r"Velocity $u_x$ (m/s)", r"Velocity $u_y$ (m/s)", r"Kinematic Pressure $p$ ($m^2/s^2$)", r"Eddy Viscosity $\nu_t$ ($m^2/s$)"]

    x = pos[:, 0]
    y = pos[:, 1]

    for i in range(4):
        true_f = y_true[:, i]
        pred_f = y_pred[:, i]
        diff_f = np.abs(pred_f - true_f)

        vmin, vmax = np.min(true_f), np.max(true_f)

        # Ground Truth
        sc0 = axes[i, 0].scatter(x, y, c=true_f, cmap="viridis", s=1, vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Ground Truth: {field_names[i]}")
        axes[i, 0].set_aspect("equal")
        plt.colorbar(sc0, ax=axes[i, 0], fraction=0.03, pad=0.02)

        # Prediction
        sc1 = axes[i, 1].scatter(x, y, c=pred_f, cmap="viridis", s=1, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"MS-PIGNO: {field_names[i]}")
        axes[i, 1].set_aspect("equal")
        plt.colorbar(sc1, ax=axes[i, 1], fraction=0.03, pad=0.02)

        # Absolute Error
        sc2 = axes[i, 2].scatter(x, y, c=diff_f, cmap="magma", s=1)
        axes[i, 2].set_title(f"Absolute Error: {field_names[i]}")
        axes[i, 2].set_aspect("equal")
        plt.colorbar(sc2, ax=axes[i, 2], fraction=0.03, pad=0.02)

    plt.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_surface_pressure_friction(
    x_chord: np.ndarray,
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    cf_true: np.ndarray,
    cf_pred: np.ndarray,
    output_path: str,
    title: str = "Surface Aerodynamic Distributions"
):
    """
    Renders chordwise surface pressure coefficient Cp and skin-friction Cf curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # 1. Surface Pressure Coefficient Cp (Inverted Y-axis standard CFD convention)
    ax1.plot(x_chord, cp_true, "k-", label="CFD Ground Truth", linewidth=2.0)
    ax1.plot(x_chord, cp_pred, "r--", label="MS-PIGNO Prediction", linewidth=1.8)
    ax1.set_xlabel("Chord Station (x/c)")
    ax1.set_ylabel(r"$C_p$ (Pressure Coefficient)")
    ax1.set_title(r"Surface Pressure Coefficient $C_p$")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # 2. Skin Friction Coefficient Cf
    ax2.plot(x_chord, cf_true, "k-", label="CFD Ground Truth", linewidth=2.0)
    ax2.plot(x_chord, cf_pred, "b--", label="MS-PIGNO Prediction", linewidth=1.8)
    ax2.set_xlabel("Chord Station (x/c)")
    ax2.set_ylabel(r"$C_f$ (Skin Friction Coefficient)")
    ax2.set_title(r"Skin Friction Coefficient $C_f$")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
