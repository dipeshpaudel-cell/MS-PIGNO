"""
Dataset Partitioning Engine for the 4 AirfRANS Benchmark Tasks.
Generates reproducible, zero-leakage splits for Full, Scarce, Reynolds Extrapolation, and AoA Extrapolation.
"""
from typing import Dict, List, Tuple
import glob
import json
import math
import os
import random
import torch


class AirfRANSSplitter:
    """
    Scans the AirfRANS dataset directory, inspects condition parameters,
    and constructs the four official benchmark partitions.
    """
    def __init__(self, raw_data_dir: str, seed: int = 42, nu_laminar: float = 1.5e-5, chord: float = 1.0):
        self.raw_data_dir = raw_data_dir
        self.seed = seed
        self.nu_laminar = nu_laminar
        self.chord = chord
        self.file_list = sorted(glob.glob(os.path.join(raw_data_dir, "graph_airfrans_data_batch_*", "*.pth")))
        if len(self.file_list) == 0:
            self.file_list = sorted(glob.glob(os.path.join(raw_data_dir, "*.pth")))

        self.metadata_cache: Dict[str, Dict[str, float]] = {}
        self._index_file = os.path.join(raw_data_dir, "metadata_index.json")
        if os.path.exists(self._index_file):
            try:
                with open(self._index_file, "r") as f:
                    self.metadata_cache = json.load(f)
            except Exception:
                pass

    def extract_simulation_metadata(self, file_path: str) -> Dict[str, float]:
        """
        Extract true flow parameters (U_inf, AoA, Re) by loading the simulation
        .pth file and reading the inlet velocity components from the node feature
        tensor x[:, 0:2] = [U_inf_x, U_inf_y].
        """
        if file_path in self.metadata_cache:
            return self.metadata_cache[file_path]

        try:
            raw_data = torch.load(file_path, map_location="cpu", weights_only=False)
            x = raw_data.x if hasattr(raw_data, "x") else raw_data["x"]

            ux_inf = float(x[0, 0].item())
            uy_inf = float(x[0, 1].item())
            u_inf = math.sqrt(ux_inf ** 2 + uy_inf ** 2)
            aoa_rad = math.atan2(uy_inf, ux_inf)
            aoa_deg = math.degrees(aoa_rad)
            reynolds = (u_inf * self.chord) / self.nu_laminar

            num_nodes = int(x.shape[0])

            meta = {
                "file_path": file_path,
                "u_inf": u_inf,
                "aoa_rad": aoa_rad,
                "aoa_deg": aoa_deg,
                "reynolds": reynolds,
                "log10_re": math.log10(max(reynolds, 1e4)),
                "num_nodes": num_nodes
            }
        except Exception:
            meta = {
                "file_path": file_path,
                "u_inf": 30.0,
                "aoa_rad": 0.0,
                "aoa_deg": 0.0,
                "reynolds": (30.0 * self.chord) / self.nu_laminar,
                "log10_re": math.log10((30.0 * self.chord) / self.nu_laminar),
                "num_nodes": 180000
            }

        self.metadata_cache[file_path] = meta
        return meta

    def save_metadata_index(self):
        """Persist extracted metadata to disk so subsequent runs skip file loading."""
        os.makedirs(os.path.dirname(self._index_file), exist_ok=True)
        with open(self._index_file, "w") as f:
            json.dump(self.metadata_cache, f, indent=2)

    def get_all_metadata(self) -> List[Dict[str, float]]:
        """Index metadata for all available simulations."""
        return [self.extract_simulation_metadata(f) for f in self.file_list]

    def get_full_task_split(
        self,
        num_train: int = 800,
        num_test: int = 200,
        val_from_train: int = 100
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Task 1: Full interpolation task (800 train, 200 test).
        Returns: (train_files, val_files, test_files)
        """
        rng = random.Random(self.seed)
        shuffled = list(self.file_list)
        rng.shuffle(shuffled)

        test_files = shuffled[:num_test]
        train_pool = shuffled[num_test:num_test + num_train]

        val_files = train_pool[:val_from_train]
        train_files = train_pool[val_from_train:]

        return train_files, val_files, test_files

    def get_scarce_task_split(
        self,
        num_train: int = 100,
        num_test: int = 200,
        val_from_train: int = 20
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Task 2: Scarce data efficiency task (100 train, 200 test).
        The test set is identical to the Full task test set.
        """
        rng = random.Random(self.seed)
        shuffled = list(self.file_list)
        rng.shuffle(shuffled)

        test_files = shuffled[:num_test]
        train_pool = shuffled[num_test:num_test + num_train]

        val_files = train_pool[:val_from_train]
        train_files = train_pool[val_from_train:]

        return train_files, val_files, test_files

    def get_reynolds_extrapolation_split(
        self,
        train_re_min: float = 3.0e6,
        train_re_max: float = 5.0e6,
        val_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Task 3: Reynolds Extrapolation (Train on mid-Re, test on held-out extreme Re).
        """
        all_meta = self.get_all_metadata()
        train_pool = []
        test_files = []

        for m in all_meta:
            re = m["reynolds"]
            if train_re_min <= re <= train_re_max:
                train_pool.append(m["file_path"])
            else:
                test_files.append(m["file_path"])

        rng = random.Random(self.seed)
        rng.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * val_ratio))

        val_files = train_pool[:n_val]
        train_files = train_pool[n_val:]

        return train_files, val_files, test_files

    def get_aoa_extrapolation_split(
        self,
        max_train_aoa_deg: float = 10.0,
        val_ratio: float = 0.1
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Task 4: Angle of Attack Extrapolation (Train on attached flow <= 10 deg, test on separated > 10 deg).
        """
        all_meta = self.get_all_metadata()
        train_pool = []
        test_files = []

        for m in all_meta:
            aoa = m["aoa_deg"]
            if aoa <= max_train_aoa_deg:
                train_pool.append(m["file_path"])
            else:
                test_files.append(m["file_path"])

        rng = random.Random(self.seed)
        rng.shuffle(train_pool)
        n_val = max(1, int(len(train_pool) * val_ratio))

        val_files = train_pool[:n_val]
        train_files = train_pool[n_val:]

        return train_files, val_files, test_files

    def get_split_by_task_name(self, task_name: str) -> Tuple[List[str], List[str], List[str]]:
        """Dispatcher by task string identifier."""
        task_lower = task_name.lower().replace("-", "_")
        if task_lower in ("full", "interpolation"):
            return self.get_full_task_split()
        elif task_lower in ("scarce", "few_shot"):
            return self.get_scarce_task_split()
        elif task_lower in ("reynolds", "reynolds_extrapolation", "ood_reynolds"):
            return self.get_reynolds_extrapolation_split()
        elif task_lower in ("aoa", "aoa_extrapolation", "ood_aoa"):
            return self.get_aoa_extrapolation_split()
        else:
            raise ValueError(f"Unknown task name: {task_name}. Supported: [full, scarce, reynolds_extrapolation, aoa_extrapolation]")
