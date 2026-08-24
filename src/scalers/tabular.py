# tabular.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os
from pathlib import Path

class PowerScaler:
    def __init__(self, target_col: str, input_cols: list[str] | None = None):
        self.target_col = target_col
        self.input_cols = input_cols or []

        self.input_cols = [col for col in self.input_cols if col != self.target_col]

        self.feature_scaler = StandardScaler() if self.input_cols else None
        self.target_scaler = StandardScaler()

    def fit(self, train_dataset_df: pd.DataFrame) -> "PowerScaler":
        required_cols = {self.target_col} | set(self.input_cols)
        missing_cols = [col for col in required_cols if col not in train_dataset_df.columns]

        if missing_cols:
            raise KeyError(f"Missing required columns for scaling: {missing_cols}.\n Available columns: {list(train_dataset_df.columns)}")

        self.target_scaler.fit(train_dataset_df[[self.target_col]])

        if self.feature_scaler and self.input_cols:
            self.feature_scaler.fit(train_dataset_df[self.input_cols])

        return self
    
    def transform(self, dataset_df: pd.DataFrame) -> pd.DataFrame:
        df_scaled = dataset_df.copy()
        df_scaled[self.target_col] = self.target_scaler.transform(dataset_df[[self.target_col]])

        if self.feature_scaler and self.input_cols:
            df_scaled[self.input_cols] = self.feature_scaler.transform(dataset_df[self.input_cols])

        return df_scaled

    def inverse_transform_target(self, y_scaled: np.array):
        orig_shape = y_scaled.shape
        
        y_flat = y_scaled.reshape(-1, 1)

        y_flat_df = pd.DataFrame(y_flat, columns=[self.target_col])

        y_inv_flat = self.target_scaler.inverse_transform(y_flat_df)
        y_inverse = y_inv_flat.reshape(orig_shape)

        return y_inverse

    def save_scalers(self, save_dir: str = "checkpoints/scalers"):
        os.makedirs(save_dir, exist_ok=True)

        if self.feature_scaler:
            joblib.dump(self.feature_scaler, f"{save_dir}/features_scaler.pkl")
            
        joblib.dump(self.target_scaler, f"{save_dir}/target_scaler.pkl")
        print("Scalers were successfully saved!")
    
    def load_scalers(self, save_dir: str | Path = "checkpoints/scalers") -> None:
        save_path = Path(save_dir)
        target_scaler_path = save_path / "target_scaler.pkl"
        feature_scaler_path = save_path / "features_scaler.pkl"

        if not target_scaler_path.exists():
            raise FileNotFoundError(f"No target scaler file found in: {save_path}")

        self.target_scaler = joblib.load(target_scaler_path)

        if feature_scaler_path.exists():
            self.feature_scaler = joblib.load(feature_scaler_path)
        elif self.input_cols:
            raise FileNotFoundError(f"Expected feature scaler at {feature_scaler_path}, but it was not found!")

        print(f"Scalers were successfully loaded from: {save_path}")