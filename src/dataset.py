# dataset.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from netCDF4 import Dataset
from tqdm import tqdm

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
        history_cols: list[str] = None,
        history_cols_1h: list[str] = None,
        future_cols: list[str] = None,
        latency_min: int = 30
    ):
        self.sat_file_map = sat_file_map
        self.power_df = power_df
        self.sat_scaler = sat_scaler
        
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out
        self.target_col = target_col
        self.history_cols = history_cols or []
        self.history_cols_1h = [col for col in (history_cols_1h or []) if col != self.target_col]
        self.future_cols = [col for col in (future_cols or []) if col != self.target_col]
        self.latency_min = latency_min

        self.samples = self._build_samples()

    def _build_samples(self) -> list[dict[str, Any]]:
        samples = []

        for t_anchor in tqdm(sorted(self.sat_file_map.keys()), desc="Building dataset samples"):
            sat_times = [
                t_anchor - pd.Timedelta(minutes=self.latency_min + 15 * i) 
                for i in reversed(range(self.seq_len_in))
            ]

            last_hourly_anchor = t_anchor.floor("h")
            history_power_times = [
                last_hourly_anchor - pd.Timedelta(hours=i) 
                for i in reversed(range(self.seq_len_out))
            ]

            latency_steps = self.latency_min // 15
            seq_len_out_15m = latency_steps + (self.seq_len_out * 4)
            last_sat_time = t_anchor - pd.Timedelta(minutes=self.latency_min)

            future_15m_meteo_times = [
                last_sat_time + pd.Timedelta(minutes=15 * (i + 1)) 
                for i in range(seq_len_out_15m)
            ]

            first_target = t_anchor.ceil("h")
            if first_target == t_anchor:
                first_target += pd.Timedelta(hours=1)

            target_times = [first_target + pd.Timedelta(hours=i) for i in range(self.seq_len_out)]

            has_sat = all(t in self.sat_file_map for t in sat_times)
            has_meteo_history = all(t in self.power_df.index for t in sat_times)
            has_history_power = all(t in self.power_df.index for t in history_power_times)
            has_meteo_future = all(t in self.power_df.index for t in future_15m_meteo_times)
            has_target = all(t in self.power_df.index for t in target_times)

            if has_sat and has_meteo_history and has_history_power and has_meteo_future and has_target:
                meteo_hist = self.power_df.loc[sat_times, self.history_cols].values
                hist_power = self.power_df.loc[history_power_times, self.target_col].values
                meteo_fut = self.power_df.loc[future_15m_meteo_times, self.future_cols].values if self.future_cols else np.array([])
                target_p = self.power_df.loc[target_times, self.target_col].values

                has_nan = (
                    np.isnan(meteo_hist).any() or 
                    np.isnan(hist_power).any() or
                    np.isnan(target_p).any() or 
                    (len(meteo_fut) > 0 and np.isnan(meteo_fut).any())
                )

                if not has_nan:
                    sample = {
                        "sat_files": [self.sat_file_map[t] for t in sat_times],
                        "meteo_history": meteo_hist,   # [seq_len_in, num_history_cols] (15m)
                        "history_power": hist_power,   # [seq_len_out] (1h)
                        "meteo_future": meteo_fut,     # [seq_len_out_15m, num_future_cols] (15m)
                        "target_power": target_p       # [seq_len_out] (1h)
                    }
                        
                    samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        sat_seq = []
        for file_path in sample["sat_files"]:
            with Dataset(file_path, "r") as nc:
                dssf = np.squeeze(nc.variables["DSSF_TOT"][:]) 
                diff = np.squeeze(nc.variables["FRACTION_DIFFUSE"][:]) 

                img = np.stack([dssf, diff], axis=0)
                img_scaled = self.sat_scaler.transform(img)
                img_scaled = np.nan_to_num(img_scaled, nan=0.0)

                sat_seq.append(img_scaled)

        X_sat = torch.tensor(np.array(sat_seq), dtype=torch.float32)
        X_meteo_hist = torch.tensor(sample["meteo_history"], dtype=torch.float32)
        X_hist_pwr = torch.tensor(sample["history_power"], dtype=torch.float32)
        X_meteo_fut = torch.tensor(sample["meteo_future"], dtype=torch.float32)
        y_target_pwr = torch.tensor(sample["target_power"], dtype=torch.float32)

        return {
            "sat_seq": X_sat,           # [seq_len_in, C, H, W]
            "meteo_history": X_meteo_hist, # [seq_len_in, num_history_cols]
            "history_power": X_hist_pwr,   # [seq_len_out]
            "meteo_future": X_meteo_fut,   # [seq_len_out_15m, num_future_cols]
            "target": y_target_pwr         # [seq_len_out]
        }