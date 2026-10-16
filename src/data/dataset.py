"""
PyTorch Geometric Dataset Implementation for AirfRANS Surrogate Training.
Supports memory-mapped cached enriched graphs and zero-leakage on-the-fly normalization.
"""
from typing import List, Optional
import os
import torch
from torch_geometric.data import Data, Dataset

from .transforms import FlowFeatureNormalizer
from .preprocessor import AirfRANSGraphPreprocessor


class AirfRANSPyGDataset(Dataset):
    """
    AirfRANS Graph Dataset with Multi-Scale Topology, Precomputed LS-FV Weights,
    and Exact Physical Boundary Conditions.
    """
    def __init__(
        self,
        file_paths: List[str],
        cache_dir: Optional[str] = None,
        normalizer: Optional[FlowFeatureNormalizer] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        super().__init__(root=cache_dir, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)
        self.file_paths = file_paths
        self.cache_dir = cache_dir
        self.normalizer = normalizer
        self.preprocessor = AirfRANSGraphPreprocessor()

    def len(self) -> int:
        return len(self.file_paths)

    def _get_cache_path(self, raw_path: str) -> Optional[str]:
        if self.cache_dir is None:
            return None
        # Derive unique relative filename
        base_name = os.path.basename(raw_path)
        parent_dir = os.path.basename(os.path.dirname(raw_path))
        return os.path.join(self.cache_dir, parent_dir, base_name)

    def get(self, idx: int) -> Data:
        raw_path = self.file_paths[idx]
        cache_path = self._get_cache_path(raw_path)

        data = None
        if cache_path is not None and os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
            except Exception:
                data = None

        if data is None:
            # Load raw data and enrich on-the-fly if cache missing
            if cache_path is not None:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                self.preprocessor.process_simulation_file(raw_path, cache_path)
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
            else:
                # Direct memory object without disk cache
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
                    tmp_name = tmp.name
                self.preprocessor.process_simulation_file(raw_path, tmp_name)
                data = torch.load(tmp_name, map_location="cpu", weights_only=False)
                os.remove(tmp_name)

        # Apply Feature Normalization if provided
        if self.normalizer is not None and self.normalizer.fitted:
            data.x_norm = self.normalizer.transform_x(data.x)
            data.y_norm = self.normalizer.transform_y(data.y)
            if hasattr(data, "cond"):
                data.cond_norm = self.normalizer.transform_cond(data.cond)
        else:
            data.x_norm = data.x.clone()
            data.y_norm = data.y.clone()
            if hasattr(data, "cond"):
                data.cond_norm = data.cond.clone()

        data.file_path = raw_path
        return data
