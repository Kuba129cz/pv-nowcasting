# processing.py
from pathlib import Path
import pandas as pd

def load_dataset(dataset_path: str|Path, target_col: str, input_cols: list[str] | None = None, timestamp_col: str = "timestamp",) -> pd.DataFrame:
    """
    Loads and preprocesses a CSV dataset containing time-series data.

    Reads the dataset, parses the timestamp column as the index, selects
    the required columns, drops rows with missing values, and sorts chronologically.

    Args:
        dataset_path (str | Path): Path to the CSV file.
        target_col (str): The name of the target column (e.g., power output).
        input_cols (list[str] | None, optional): List of additional input columns to load. Defaults to None.
        timestamp_col (str, optional): The name of the timestamp column. Defaults to "timestamp".

    Raises:
        FileNotFoundError: If the specified dataset file does not exist.
        KeyError: If the target column or any of the input columns are missing.

    Returns:
        pd.DataFrame: Processed and chronologically sorted DataFrame with no missing values.
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"File does not exist at path: {dataset_path.resolve()}")

    dataset = pd.read_csv(dataset_path, index_col="timestamp", parse_dates=[timestamp_col])

    if dataset.index.tz is not None:
        dataset.index = dataset.index.tz_localize(None)
        
    if target_col not in dataset.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in {dataset_path.name}.\n"
            f"Available columns: {list(dataset.columns)}"
        )

    cols_to_check = [target_col] + (input_cols or [])
    missing_cols = [c for c in cols_to_check if c not in dataset.columns]
    if missing_cols:
        raise KeyError(f"Missing columns {missing_cols} in {dataset_path.name}.\nAvailable: {list(dataset.columns)}")
    
    return dataset[cols_to_check].dropna().sort_index()

def load_satellite_map(sat_dir: str|Path) -> dict[pd.Timestamp, Path]:
    """
    Scans a directory for NetCDF satellite files and maps them to their timestamps.

    Recursively searches for '*.nc' files in the given directory. It assumes the 
    filename stem matches the format 'YYYYMMDDHHMM' to extract the correct timestamp.

    Args:
        sat_dir (str | Path): Root directory containing the NetCDF files.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        ValueError: If no '.nc' files are found in the directory.

    Returns:
        dict[pd.Timestamp, Path]: A dictionary mapping parsed timestamps to their respective file paths.
    """
    sat_dir = Path(sat_dir)

    if not sat_dir.is_dir():
        raise FileNotFoundError(f"Satellite directory not found at: {sat_dir.resolve()}")

    sat_map = {
        pd.to_datetime(f.stem, format="%Y%m%d%H%M"): f for f in sat_dir.rglob("*.nc")
    }

    if not sat_map:
        raise ValueError(f"No NetCDF files found in directory: {sat_dir}")

    return sat_map

def create_splits(dataset_df: pd.DataFrame, sat_map: dict[pd.Timestamp, Path], train_ratio: float, val_ratio: float) -> tuple[dict[str, pd.Series], dict[str, dict[pd.Timestamp, Path]]]:
    """
    Splits the tabular dataset and satellite map into training, validation, and test sets.

    The split is performed chronologically without shuffling to prevent data leakage 
    in time-series forecasting.

    Args:
        dataset_df (pd.DataFrame): The preprocessed dataset with a chronological datetime index.
        sat_map (dict[pd.Timestamp, Path]): Dictionary mapping timestamps to satellite NetCDF file paths.
        train_ratio (float): Proportion of the dataset to use for training (e.g., 0.7).
        val_ratio (float): Proportion of the dataset to use for validation (e.g., 0.2).

    Raises:
        ValueError: If the sum of train_ratio and val_ratio exceeds 1.0.

    Returns:
        tuple[dict[str, pd.Series | pd.DataFrame], dict[str, dict[pd.Timestamp, Path]]]: 
            - power_splits: Dictionary containing 'train', 'val', and 'test' DataFrames.
            - sat_splits: Dictionary containing 'train', 'val', and 'test' satellite mappings.
    """
    test_ratio = 1.0 - val_ratio - train_ratio 
    if test_ratio < 0:
        raise ValueError(f"Sum of train_ratio ({train_ratio}) and val_ratio ({val_ratio}) exceeds 1.0!")

    sorted_times = dataset_df.index
    n_samples = len(sorted_times)

    idx_train_end = int(n_samples * train_ratio)
    idx_val_end = int(n_samples * (train_ratio + val_ratio))

    t_train_end = sorted_times[idx_train_end - 1]
    t_val_start = sorted_times[idx_train_end]
    t_val_end = sorted_times[idx_val_end - 1]
    t_test_start = sorted_times[idx_val_end]

    power_splits = {
        "train": dataset_df.loc[:t_train_end],
        "val": dataset_df.loc[t_val_start:t_val_end],
        "test": dataset_df.loc[t_test_start:],
    }

    sat_splits = {
        "train": {t: f for t, f in sat_map.items() if t <= t_train_end},
        "val": {t: f for t, f in sat_map.items() if t_val_start <= t <= t_val_end},
        "test": {t: f for t, f in sat_map.items() if t >= t_test_start},
    }

    return power_splits, sat_splits

