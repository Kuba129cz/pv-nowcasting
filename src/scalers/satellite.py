# satellite.py
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr


class SatelliteScaler:
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, file_paths: Iterable[Path | str]) -> "SatelliteScaler":
        sum_c = None
        sum_sq_c = None
        total_pixels = 0

        for file_path in file_paths:
            file_path = Path(file_path)
            with xr.open_dataset(file_path) as ds:
                dssf = ds["DSSF_TOT"].squeeze().values
                diff = ds["FRACTION_DIFFUSE"].squeeze().values
                img = np.stack([dssf, diff], axis=0)

            if sum_c is None:
                c = img.shape[0]
                sum_c = np.zeros(c, dtype=np.float64)
                sum_sq_c = np.zeros(c, dtype=np.float64)

            valid_mask = ~np.isnan(img[0])
            valid_pixels_count = np.count_nonzero(valid_mask)

            if valid_pixels_count == 0:
                continue

            sum_c += np.nansum(img, axis=(1, 2), dtype=np.float64)
            sum_sq_c += np.nansum(img**2, axis=(1, 2), dtype=np.float64)
            total_pixels += valid_pixels_count

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