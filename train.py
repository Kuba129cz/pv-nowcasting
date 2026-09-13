# train.py
from pathlib import Path
from datetime import datetime

import argparse
import pandas as pd
import src.processing as processing
import torch
import os

from src.scalers.tabular import PowerScaler
from src.scalers.satellite import SatelliteScaler
from src.dataset import PVSatelliteDataset
from src.models.model import Model
from src.metrics import ErrorTracker
from src.logger import TensorBoardLogger
from src.trainer import EarlyStopping, run_training, run_testing

LOSS_FUNCTIONS = {
    "mae": torch.nn.L1Loss(),
    "mse": torch.nn.MSELoss(),
    "huber": torch.nn.HuberLoss()
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PV forecast with satellite data")

    # Data & Paths
    parser.add_argument("--train_ratio", type=float, default=0.75, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.20, help="Validation split ratio")
    parser.add_argument("--sat_dir", type=Path, default=Path("dataset/LSA_MDSSTFD_CROPPED_128x128"), help="Directory path to satellite NetCDF files.")
    parser.add_argument("--dataset_path", type=Path, default=Path("dataset/aba_train_15min.csv"), help="File path to power generation CSV dataset.")
    parser.add_argument("--target_col", type=str, default="energy", help="Target column name in CSV.")
    parser.add_argument("--history_cols", type=str, nargs="+", default=[], help="List of tabular columns available for past sequence.")
    parser.add_argument("--future_cols", type=str, nargs="+", default=[], help="List of NWP forecast columns available for future horizons.")

    # Sequence lengths
    parser.add_argument("--seq_len_in", type=int, default=8, help="Input satellite/meteo history sequence length.")
    parser.add_argument("--seq_len_history_power", type=int, default=24, help="Power history sequence length.")
    parser.add_argument("--seq_len_out", type=int, default=4, help="Output target sequence length in hours.")
    parser.add_argument("--seq_len_out_15m", type=int, default=16, help="Output target sequence length in 15min steps (e.g. 4h * 4 = 16).")
    parser.add_argument("--latency_min", type=int, default=15, help="Latency of satellite image.")

    # History Meteo Encoder
    parser.add_argument("--past_hidden_size", type=int, default=32, help="LSTM hidden size for history meteo encoder.")
    parser.add_argument("--past_cnn_filters", type=int, default=32, help="CNN filter count for history meteo encoder.")
    parser.add_argument("--past_kernel", type=int, default=3, help="CNN kernel size for history meteo encoder.")
    parser.add_argument("--past_dropout", type=float, default=0.1, help="Dropout for history meteo encoder.")

    # Future Meteo Encoder
    parser.add_argument("--future_hidden_size", type=int, default=32, help="LSTM hidden size for future meteo encoder.")
    parser.add_argument("--future_cnn_filters", type=int, default=32, help="CNN filter count for future meteo encoder.")
    parser.add_argument("--future_dropout", type=float, default=0.1, help="Dropout for future meteo encoder.")
    parser.add_argument("--future_kernel_L0", type=int, default=3, help="Kernel size for layer 0 in future meteo CNN.")
    parser.add_argument("--future_kernel_L1", type=int, default=3, help="Kernel size for layer 1 in future meteo CNN.")

    # Satellite Encoder & Spatial dimensions
    parser.add_argument("--sat_hidden_dim", type=int, default=64, help="Hidden channels for ConvLSTM in SatelliteEncoder.")
    parser.add_argument("--sat_h_out", type=int, default=32, help="Spatial height of satellite features after CNN stem.")
    parser.add_argument("--sat_w_out", type=int, default=32, help="Spatial width of satellite features after CNN stem.")

    # History Power Encoder
    parser.add_argument("--power_hidden_size", type=int, default=32, help="LSTM hidden size for power history encoder.")
    parser.add_argument("--power_cnn_filters", type=int, default=32, help="CNN filter count for power history encoder.")
    parser.add_argument("--power_dropout", type=float, default=0.1, help="Dropout for power history encoder.")

    # Decoder & Attention
    parser.add_argument("--attention_dim", type=int, default=64, help="Embedding dimension for CrossAttention.")
    parser.add_argument("--feed_forward_net_dim", type=int, default=128, help="Feed forward hidden dimension in Decoder.")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in Decoder.")

    # Training & Logging
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoaders.")
    parser.add_argument("--num_epochs", default=5, type=int)
    parser.add_argument("--save_dir_power_scalers", type=str, default="checkpoints/scalers", help="Directory path for power scalers.")
    parser.add_argument("--save_path_sat_scalers", type=str, default="checkpoints/scalers/sat_scaler.json", help="Path for sat scalers.")
    parser.add_argument("--log_dir", type=str, default="checkpoints/runs/aba/", help="Directory path for logger.")
    parser.add_argument("--nominal_capacity_fve", type=int, default=1293, help="Nominal output of PV.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of subprocesses for data loading.")
    parser.add_argument("--loss_func", type=str, default="mae", choices=["mae", "mse", "huber"], help="Choose loss function.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience.")

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
    power_scaler = PowerScaler(target_col=args.target_col, input_cols=pd.unique(args.history_cols + args.future_cols))
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
            seq_len_history_power= args.seq_len_history_power,
            target_col=args.target_col,
            latency_min=args.latency_min
            ) 
            for split in splits
    }
    loaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=8,
            persistent_workers=True
        ) 
        for split in splits
    }
    for split in splits:
        print(f"{split.capitalize()} dataset samples: {len(datasets[split])}")
    print("Everything is ready for training loop!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(in_channels=2, seq_len_in=args.seq_len_in, seq_len_out=args.seq_len_out).to(device=device)

    criterion = LOSS_FUNCTIONS[args.loss_func]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    tracker = ErrorTracker(seq_len_out=args.seq_len_out, power_scaler=power_scaler, nominal_capacity_fve=args.nominal_capacity_fve)

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = Path(args.log_dir) / run_name
    logger = TensorBoardLogger(log_dir=str(run_log_dir), args=args)

    print(f"Starting training on {device}...")

    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=Path("checkpoints/best_model.pt"))
    run_training(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        tracker=tracker,
        logger=logger,
        early_stopping=early_stopping,
        epochs=args.num_epochs,
        device=device
    )

    run_testing(
        model=model,
        dataloader=loaders["test"],
        criterion=criterion,
        tracker=tracker,
        device=device,
        best_model_path=early_stopping.checkpoint_path,
        logger=logger, 
        args=args     
    )

    logger.close()

if __name__ == "__main__":
    parser = build_parser()
    cli_args = [] if "__file__" not in globals() else None
    args = parser.parse_args(cli_args)
    main(args=args)