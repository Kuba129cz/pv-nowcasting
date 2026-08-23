# train.py
import argparse
from pathlib import Path
import pandas as pd
import src.processing as processing
import torch
import os

from src.scalers.tabular import PowerScaler
from src.scalers.satellite import SatelliteScaler
from src.dataset import PVSatelliteDataset
from src.models.dummy_model import Model

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PV forecast with satellite data")

    parser.add_argument("--train_ratio", type=float, default=0.75, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.20, help="Validation split ratio")

    parser.add_argument("--sat_dir", type=Path, default=Path("LSA_MDSSTFD_CROPPED_128x128"), help="Directory path to satellite NetCDF files.")
    parser.add_argument("--dataset_path", type=Path, default=Path("dataset/aba_train.csv"), help="File path to power generation CSV dataset.")
    parser.add_argument("--target_col", type=str, default="energy", help="Target column name in CSV.")
    parser.add_argument("--input_cols", type=str, nargs="+", default=[], help="List of input columns from CSV (separated by space).")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoaders.")
    parser.add_argument("--seq_len_in", type=int, default=4, help="Input satellite sequence length.")
    parser.add_argument("--seq_len_out", type=int, default=24, help="Output target sequence length.")

    parser.add_argument("--save_dir_power_scalers", type=str, default="checkpoints/scalers", help="Directory path for power scalers to be saved.")
    parser.add_argument("--save_path_sat_scalers", type=str, default="checkpoints/scalers/sat_scaler.json", help="Directory path for satellite scalers to be saved.")

    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="Number of subprocesses to use for data loading.")

    return parser

def prepare_and_save_scalers(power_splits: dict, sat_splits: dict, args: argparse.Namespace) -> tuple:
    """Fits, transforms, and saves both tabular and satellite scalers."""
    print("Data split details:")
    for split_name, df in power_splits.items():
        start_time = df.index.min()
        end_time = df.index.max()
        num_records = len(df)
        num_sat_files = len(sat_splits.get(split_name, []))
        
        print(f"  - {split_name.capitalize():<5}: {num_records:,} rows ({num_sat_files:,} sat files) | {start_time} -> {end_time}")

    print("Fitting and transforming power data (PowerScaler)...")
    power_scaler = PowerScaler(target_col=args.target_col, input_cols=args.input_cols)
    power_scaler.fit(train_dataset_df=power_splits["train"])
    
    for split in ["train", "val", "test"]:
        power_splits[split] = power_scaler.transform(power_splits[split])
    power_scaler.save_scalers(save_dir=args.save_dir_power_scalers)

    sat_scaler_path = Path(args.save_path_sat_scalers)
    sat_scaler = SatelliteScaler(num_workers=args.num_workers)
    
    if sat_scaler_path.is_file():
        print(f"Loading existing satellite scaler from {sat_scaler_path}...")
        sat_scaler.load_scaler(load_path=sat_scaler_path)
    else:
        print(f"Fitting satellite data from {len(sat_splits['train'])} files (this might take a while)...")
        sat_scaler.fit(file_paths=sat_splits["train"].values())
        sat_scaler.save_scaler(save_path=sat_scaler_path)
    
    print("Done! All scalers are fitted and saved successfully.")
    return power_splits, power_scaler, sat_scaler

def main(args: argparse.Namespace):
    print("Loading and splitting data...")
    dataset = processing.load_dataset(dataset_path=args.dataset_path, target_col=args.target_col)
    satellite_data = processing.load_satellite_map(sat_dir=args.sat_dir)
    power_splits, sat_splits = processing.create_splits(dataset_df=dataset, sat_map=satellite_data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    power_splits, power_scaler, sat_scaler = prepare_and_save_scalers(power_splits, sat_splits, args)

    print("Creating datasets and dataloaders...")
    splits = ["train", "val", "test"]

    datasets = {
        split: PVSatelliteDataset(
            sat_file_map=sat_splits[split],
            power_df=power_splits[split],
            sat_scaler=sat_scaler,
            seq_len_in=args.seq_len_in,
            seq_len_out=args.seq_len_out,
            target_col=args.target_col
            ) 
            for split in splits
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True
        ) 
        for split in splits
    }
    for split in splits:
        print(f"{split.capitalize()} dataset samples: {len(datasets[split])}")
    print("Everything is ready for training loop!")

    model = Model(in_channels=2, seq_len_in=args.seq_len_in, seq_len_out=args.seq_len_out)

if __name__ == "__main__":
    parser = build_parser()
    cli_args = [] if "__file__" not in globals() else None
    args = parser.parse_args(cli_args)
    main(args=args)