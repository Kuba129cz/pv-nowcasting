# dataset.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from src.scalers.satellite import SatelliteScaler


class PVSatelliteDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        sat_file_map: dict[pd.Timestamp, Path], 
        power_df: pd.DataFrame, 
        sat_scaler: SatelliteScaler,
        seq_len_in: int, 
        seq_len_out: int, 
        target_col: str = "energy", 
        latency_min: int = 30
    ):
        self.sat_file_map = sat_file_map
        self.power_df = power_df
        self.sat_scaler = sat_scaler
        
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.target_col = target_col
        self.latency_min = latency_min

        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict[str, Any]]:
        samples = []

        for t_anchor in sorted(self.sat_file_map.keys()):
            sat_times = [t_anchor - pd.Timedelta(minutes=self.latency_min + 15 * i) for i in reversed(range(self.seq_len_in))]

            first_target = t_anchor.ceil("h")
            if first_target == t_anchor:
                first_target += pd.Timedelta(hours=1)

            power_times = [first_target + pd.Timedelta(hours=i) for i in range(self.seq_len_out)]

            if all(t in self.sat_file_map for t in sat_times) and all(t in self.power_df.index for t in power_times):
                samples.append({
                    "sat_files": [self.sat_file_map[t] for t in sat_times],
                    "target_power": self.power_df.loc[power_times, self.target_col].values
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        sat_files = sample["sat_files"]       
        target_power = sample["target_power"] 

        sat_seq = []
        for file_path in sat_files:
            with xr.open_dataset(file_path) as ds:
                dssf = ds["DSSF_TOT"].squeeze().values       
                diff = ds["FRACTION_DIFFUSE"].squeeze().values        
                
                img = np.stack([dssf, diff], axis=0) 
                img_scaled = self.sat_scaler.transform(img)
                img_scaled = np.nan_to_num(img_scaled, nan=0.0)
                
                sat_seq.append(img_scaled)

        X_sat = torch.tensor(np.array(sat_seq), dtype=torch.float32)
        y_pwr = torch.tensor(target_power, dtype=torch.float32)

        return {
            "sat_seq": X_sat,   # Shape = [seq_len_in, C, H, W]
            "target": y_pwr     # Shape = [seq_len_out]
        }