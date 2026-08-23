# satellite.py
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm


def _process_file_chunk(file_paths: List[Path]) -> Tuple[np.ndarray | None, np.ndarray | None, int]:
    """
    Fast reading directly via netCDF4.Dataset (C-level) without Xarray metadata overhead.
    Must be defined at the top level for Windows multiprocessing to pickle it properly.
    """
    sum_c = None
    sum_sq_c = None
    total_pixels = 0

    for file_path in file_paths:
        try:
            with Dataset(file_path, "r") as nc:
                dssf = np.squeeze(nc.variables["DSSF_TOT"][:])
                diff = np.squeeze(nc.variables["FRACTION_DIFFUSE"][:])
                img = np.stack([dssf, diff], axis=0)

            valid_mask = ~np.isnan(img[0]) & (img[0] > 0) & (img[1] >= 0)
            valid_pixels_count = np.count_nonzero(valid_mask)

            if valid_pixels_count == 0:
                continue

            if sum_c is None:
                c = img.shape[0]
                sum_c = np.zeros(c, dtype=np.float64)
                sum_sq_c = np.zeros(c, dtype=np.float64)

            sum_c += np.nansum(img, axis=(1, 2), dtype=np.float64)
            sum_sq_c += np.nansum(img**2, axis=(1, 2), dtype=np.float64)
            total_pixels += valid_pixels_count

        except Exception as e:
            print(f"\nSkipping file {file_path} due to error: {e}")
            continue

    return sum_c, sum_sq_c, total_pixels


class SatelliteScaler:
    def __init__(self, epsilon: float = 1e-8, num_workers: int = None):
        self.epsilon = epsilon
        self.num_workers = num_workers or os.cpu_count() or 4
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, file_paths: Iterable[Path | str]) -> "SatelliteScaler":
        """
        Fits the scaler by calculating the mean and standard deviation 
        across all valid pixels in the provided satellite files in parallel.
        """
        file_paths_list = [Path(p) for p in file_paths]
        total_files = len(file_paths_list)

        if total_files == 0:
            raise ValueError("No file paths provided for fitting.")

        chunk_size = max(1, total_files // (self.num_workers * 4))

        chunks = [
            file_paths_list[i : i + chunk_size]
            for i in range(0, total_files, chunk_size)
        ]

        sum_c = None
        sum_sq_c = None
        total_pixels = 0

        print(f"Fitting SatelliteScaler on {total_files:,} files using {self.num_workers} workers...")

        # Process in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(_process_file_chunk, chunks),
                    total=len(chunks),
                    desc="Fitting Satellite Scaler",
                )
            )

        for b_sum, b_sum_sq, b_pixels in results:
            if b_pixels == 0 or b_sum is None:
                continue

            if sum_c is None:
                sum_c = np.zeros_like(b_sum)
                sum_sq_c = np.zeros_like(b_sum_sq)

            sum_c += b_sum
            sum_sq_c += b_sum_sq
            total_pixels += b_pixels

        if total_pixels == 0 or sum_c is None:
            raise ValueError("No valid pixels were found to fit the scaler.")

        mean = sum_c / total_pixels
        var = (sum_sq_c / total_pixels) - (mean**2)
        var = np.maximum(var, 0.0)

        self.mean = mean.astype(np.float32)
        self.std = (np.sqrt(var) + self.epsilon).astype(np.float32)

        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        """Applies Z-score normalization to a NumPy array."""
        if self.mean is None or self.std is None:
            raise RuntimeError("The scaler must be fitted before calling transform!")

        if img.ndim == 3:  # Shape: [C, H, W]
            return (img - self.mean[:, None, None]) / self.std[:, None, None]
        elif img.ndim == 4:  # Shape with batch: [B, C, H, W]
            return (img - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        else:
            raise ValueError(f"Unexpected array dimension: {img.ndim}")

    def save_scaler(self, save_path: str | Path = "checkpoints/scalers/sat_scaler.json") -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "epsilon": self.epsilon,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        print(f"SatelliteScaler saved to: {save_path}")

    def load_scaler(self, load_path: str | Path = "checkpoints/scalers/sat_scaler.json") -> None:
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Satellite scaler was not found at: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.mean = np.array(data["mean"], dtype=np.float32)
        self.std = np.array(data["std"], dtype=np.float32)
        self.epsilon = data["epsilon"]

        print(f"SatelliteScaler was loaded from: {load_path}")