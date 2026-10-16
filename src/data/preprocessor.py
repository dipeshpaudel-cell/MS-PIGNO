"""
Offline Preprocessing Engine for AirfRANS Graphs.
Precomputes:
  1. Metric-conditioned Anisotropic LS-FV gradient reconstruction weights (W_ij).
  2. Multi-level graph coarsening (Level 0 -> Level 1 -> Level 2) assignments and edge indices.
  3. Anisotropic edge attributes.
  4. Wall surface segment integration metadata.
"""
from typing import Dict, Optional, Tuple
import math
import os
import numpy as np
import scipy.spatial
import torch
from torch_geometric.data import Data


class AirfRANSGraphPreprocessor:
    """
    Transforms raw simulation graph files into enriched, multi-level accelerated graphs.
    """
    def __init__(
        self,
        l1_target_nodes: int = 16384,
        l2_target_nodes: int = 1024,
        lambda_reg: float = 1e-6,
        nu_laminar: float = 1.5e-5,
        wall_phi_threshold: float = 1e-4
    ):
        self.l1_target_nodes = l1_target_nodes
        self.l2_target_nodes = l2_target_nodes
        self.lambda_reg = lambda_reg
        self.nu_laminar = nu_laminar
        self.wall_phi_threshold = wall_phi_threshold

    def compute_anisotropic_edge_attr(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        sdf: torch.Tensor,
        normals: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Construct 7-channel anisotropy-aware edge attributes:
          [dx, dy, ||dx||, dx/||dx||, dy/||dx||, avg_phi, d_phi/||dx||, n_i . n_j]
        """
        src, dst = edge_index[0], edge_index[1]
        delta_pos = pos[dst] - pos[src]                         # [E, 2]
        dist = torch.norm(delta_pos, p=2, dim=1, keepdim=True)  # [E, 1]
        dir_unit = delta_pos / (dist + eps)                     # [E, 2]

        avg_phi = 0.5 * (sdf[src] + sdf[dst])                   # [E, 1]
        d_phi_dr = torch.abs(sdf[dst] - sdf[src]) / (dist + eps)# [E, 1]

        norm_dot = torch.sum(normals[src] * normals[dst], dim=1, keepdim=True) # [E, 1]

        edge_attr = torch.cat([delta_pos, dist, dir_unit, avg_phi, d_phi_dr, norm_dot], dim=1)  # [E, 8]
        return edge_attr.to(torch.float32)

    def compute_lsfv_weights(
        self,
        pos_np: np.ndarray,
        edge_index_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes metric-regularized Least-Squares Finite-Volume weights W_x, W_y.
        All intermediate computation is performed in FP64 for numerical stability
        on high-aspect-ratio boundary layer cells. Adaptive trace-proportional
        regularization prevents singular normal equation matrices.
        Returns:
          row_indices, col_indices, weights_x, weights_y (Sparse COO layout)
        """
        num_nodes = pos_np.shape[0]
        src, dst = edge_index_np[0], edge_index_np[1]

        pos64 = pos_np.astype(np.float64)
        dpos = pos64[dst] - pos64[src]
        dist = np.linalg.norm(dpos, axis=1, keepdims=True) + 1e-15
        inv_dist = 1.0 / dist

        w_dpos = inv_dist * dpos
        dxx = w_dpos[:, 0] * dpos[:, 0]
        dxy = w_dpos[:, 0] * dpos[:, 1]
        dyy = w_dpos[:, 1] * dpos[:, 1]

        A_xx = np.bincount(src, weights=dxx, minlength=num_nodes).astype(np.float64)
        A_xy = np.bincount(src, weights=dxy, minlength=num_nodes).astype(np.float64)
        A_yy = np.bincount(src, weights=dyy, minlength=num_nodes).astype(np.float64)

        # Adaptive trace-proportional regularization: lambda_i = eps_rel * trace(A_i)
        eps_rel = 1e-4
        trace_A = A_xx + A_yy
        lambda_adaptive = eps_rel * np.maximum(trace_A, 1e-12)
        A_xx += lambda_adaptive
        A_yy += lambda_adaptive

        det = (A_xx * A_yy - A_xy * A_xy)
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)

        inv_A_xx = A_yy / det
        inv_A_yy = A_xx / det
        inv_A_xy = -A_xy / det

        inv_xx_e = inv_A_xx[src]
        inv_xy_e = inv_A_xy[src]
        inv_yy_e = inv_A_yy[src]

        w_x_ij = inv_xx_e * w_dpos[:, 0] + inv_xy_e * w_dpos[:, 1]
        w_y_ij = inv_xy_e * w_dpos[:, 0] + inv_yy_e * w_dpos[:, 1]

        diag_w_x = -np.bincount(src, weights=w_x_ij, minlength=num_nodes).astype(np.float64)
        diag_w_y = -np.bincount(src, weights=w_y_ij, minlength=num_nodes).astype(np.float64)

        all_rows = np.concatenate([src, np.arange(num_nodes)])
        all_cols = np.concatenate([dst, np.arange(num_nodes)])
        all_wx = np.concatenate([w_x_ij, diag_w_x])
        all_wy = np.concatenate([w_y_ij, diag_w_y])

        indices = torch.tensor(np.stack([all_rows, all_cols], axis=0), dtype=torch.long)
        weight_x_tensor = torch.tensor(all_wx.astype(np.float32), dtype=torch.float32)
        weight_y_tensor = torch.tensor(all_wy.astype(np.float32), dtype=torch.float32)

        return indices, weight_x_tensor, weight_y_tensor

    def compute_multilevel_clustering(
        self,
        pos_np: np.ndarray,
        sdf_np: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        SDF-Stratified Hierarchical Graph Coarsening:
          Level 0 (~180k) -> Level 1 (~16k) -> Level 2 (1024).
        Prioritizes boundary layer resolution (SDF < 0.05m).
        """
        num_nodes = pos_np.shape[0]

        # Strata: Boundary layer (phi < 0.02), Near-wall (0.02-0.1), Far-field (>0.1)
        bl_mask = sdf_np.squeeze() < 0.02
        nw_mask = (sdf_np.squeeze() >= 0.02) & (sdf_np.squeeze() < 0.10)
        ff_mask = sdf_np.squeeze() >= 0.10

        # Subsample budgets for Level 1
        n_bl = int(self.l1_target_nodes * 0.45)
        n_nw = int(self.l1_target_nodes * 0.35)
        n_ff = self.l1_target_nodes - (n_bl + n_nw)

        idx_bl = np.where(bl_mask)[0]
        idx_nw = np.where(nw_mask)[0]
        idx_ff = np.where(ff_mask)[0]

        selected_l1_parts = []
        if len(idx_bl) > 0:
            step_bl = max(1, len(idx_bl) // n_bl)
            selected_l1_parts.append(idx_bl[::step_bl][:n_bl])
        if len(idx_nw) > 0:
            step_nw = max(1, len(idx_nw) // n_nw)
            selected_l1_parts.append(idx_nw[::step_nw][:n_nw])
        if len(idx_ff) > 0:
            step_ff = max(1, len(idx_ff) // n_ff)
            selected_l1_parts.append(idx_ff[::step_ff][:n_ff])

        l1_node_indices = np.unique(np.concatenate(selected_l1_parts))
        if len(l1_node_indices) < self.l1_target_nodes:
            rem = self.l1_target_nodes - len(l1_node_indices)
            remaining_pool = np.setdiff1d(np.arange(num_nodes), l1_node_indices)
            l1_node_indices = np.concatenate([l1_node_indices, remaining_pool[:rem]])

        l1_node_indices = l1_node_indices[:self.l1_target_nodes]
        pos_l1 = pos_np[l1_node_indices]

        # Build KDTree for Level 0 -> Level 1 clustering
        kdtree_l1 = scipy.spatial.cKDTree(pos_l1)
        _, cluster_l0_to_l1 = kdtree_l1.query(pos_np, k=1)

        # Level 1 -> Level 2 Coarsening (1024 nodes via spatial uniform decimation)
        step_l2 = max(1, len(pos_l1) // self.l2_target_nodes)
        l2_in_l1 = np.arange(0, len(pos_l1), step_l2)[:self.l2_target_nodes]
        pos_l2 = pos_l1[l2_in_l1]

        kdtree_l2 = scipy.spatial.cKDTree(pos_l2)
        _, cluster_l1_to_l2 = kdtree_l2.query(pos_l1, k=1)

        # Construct KNN graphs for Level 1 and Level 2
        edges_l1 = kdtree_l1.query_pairs(r=0.08, output_type="ndarray")
        if len(edges_l1) == 0:
            _, knn_l1 = kdtree_l1.query(pos_l1, k=8)
            edges_l1 = np.column_stack([np.repeat(np.arange(len(pos_l1)), 8), knn_l1.flatten()])
        else:
            edges_l1 = np.vstack([edges_l1, np.fliplr(edges_l1)])

        edges_l2 = kdtree_l2.query_pairs(r=0.25, output_type="ndarray")
        if len(edges_l2) == 0:
            _, knn_l2 = kdtree_l2.query(pos_l2, k=10)
            edges_l2 = np.column_stack([np.repeat(np.arange(len(pos_l2)), 10), knn_l2.flatten()])
        else:
            edges_l2 = np.vstack([edges_l2, np.fliplr(edges_l2)])

        return (
            cluster_l0_to_l1.astype(np.int64),
            cluster_l1_to_l2.astype(np.int64),
            edges_l1.T.astype(np.int64),
            edges_l2.T.astype(np.int64)
        )

    def extract_wall_metadata(
        self,
        pos_np: np.ndarray,
        sdf_np: np.ndarray,
        normals_np: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """
        Extract wall boundary nodes and compute differential arc length delta_s.
        """
        wall_mask = (sdf_np.squeeze() <= self.wall_phi_threshold)
        wall_indices = np.where(wall_mask)[0]

        if len(wall_indices) < 20:
            # Fallback to nearest 500 nodes to wall if threshold too tight
            wall_indices = np.argsort(sdf_np.squeeze())[:500]

        wall_pos = pos_np[wall_indices]
        # Sort along chord or angle around center
        center_x = (wall_pos[:, 0].max() + wall_pos[:, 0].min()) * 0.5
        center_y = (wall_pos[:, 1].max() + wall_pos[:, 1].min()) * 0.5
        angles = np.arctan2(wall_pos[:, 1] - center_y, wall_pos[:, 0] - center_x)
        sort_order = np.argsort(angles)
        sorted_wall_idx = wall_indices[sort_order]

        sorted_pos = pos_np[sorted_wall_idx]
        diffs = np.roll(sorted_pos, -1, axis=0) - np.roll(sorted_pos, 1, axis=0)
        delta_s = 0.5 * np.linalg.norm(diffs, axis=1, keepdims=True)

        # Tangent vectors: t_x = -n_y, t_y = n_x
        norm_wall = normals_np[sorted_wall_idx]
        tangents = np.stack([-norm_wall[:, 1], norm_wall[:, 0]], axis=1)

        return {
            "wall_indices": torch.tensor(sorted_wall_idx, dtype=torch.long),
            "wall_delta_s": torch.tensor(delta_s, dtype=torch.float32),
            "wall_normals": torch.tensor(norm_wall, dtype=torch.float32),
            "wall_tangents": torch.tensor(tangents, dtype=torch.float32)
        }

    def process_simulation_file(self, file_path: str, output_path: str) -> str:
        """
        Process a single raw PyG graph file into an enriched cache file.
        """
        raw_data = torch.load(file_path, map_location="cpu", weights_only=False)

        pos = raw_data.pos if hasattr(raw_data, "pos") else raw_data["pos"]
        x = raw_data.x if hasattr(raw_data, "x") else raw_data["x"]
        y = raw_data.y if hasattr(raw_data, "y") else raw_data["y"]
        edge_index = raw_data.edge_index if hasattr(raw_data, "edge_index") else raw_data["edge_index"]

        pos_np = pos.numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos)
        x_np = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        edge_np = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)

        # Extract features
        sdf_np = x_np[:, 2:3]
        normals_np = x_np[:, 3:5]

        # 1. Anisotropic edge attributes
        edge_attr = self.compute_anisotropic_edge_attr(
            pos=torch.from_numpy(pos_np),
            edge_index=torch.from_numpy(edge_np).long(),
            sdf=torch.from_numpy(sdf_np),
            normals=torch.from_numpy(normals_np)
        )

        # 2. LS-FV gradient weights
        ls_indices, ls_wx, ls_wy = self.compute_lsfv_weights(pos_np, edge_np)

        # 3. Multi-level coarsening
        c_l0_l1, c_l1_l2, e_l1, e_l2 = self.compute_multilevel_clustering(pos_np, sdf_np)

        # 4. Wall metadata
        wall_meta = self.extract_wall_metadata(pos_np, sdf_np, normals_np)

        # 5. Global condition vector [log10(Re), AoA_rad, U_inf]
        ux_in, uy_in = x_np[0, 0], x_np[0, 1]
        u_inf = math.sqrt(ux_in**2 + uy_in**2)
        aoa_rad = math.atan2(uy_in, ux_in)
        reynolds = (u_inf * 1.0) / self.nu_laminar
        cond_tensor = torch.tensor([math.log10(max(reynolds, 1e4)), aoa_rad, u_inf], dtype=torch.float32)

        # Construct enriched PyG Data object
        enriched_data = Data(
            x=torch.as_tensor(x, dtype=torch.float32),
            y=torch.as_tensor(y, dtype=torch.float32),
            pos=torch.as_tensor(pos, dtype=torch.float32),
            edge_index=torch.as_tensor(edge_index, dtype=torch.long),
            edge_attr=edge_attr,
            cond=cond_tensor,
            # LS-FV Sparse Gradient Operators
            ls_indices=ls_indices,
            ls_wx=ls_wx,
            ls_wy=ls_wy,
            # Multi-level Coarsening
            cluster_l0_l1=torch.from_numpy(c_l0_l1),
            cluster_l1_l2=torch.from_numpy(c_l1_l2),
            edge_index_l1=torch.from_numpy(e_l1),
            edge_index_l2=torch.from_numpy(e_l2),
            # Surface Wall Metadata
            wall_indices=wall_meta["wall_indices"],
            wall_delta_s=wall_meta["wall_delta_s"],
            wall_normals=wall_meta["wall_normals"],
            wall_tangents=wall_meta["wall_tangents"]
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(enriched_data, output_path)
        return output_path
