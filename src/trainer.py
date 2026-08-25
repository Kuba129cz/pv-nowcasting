# src/engine.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics import ErrorTracker
from src.logger import TensorBoardLogger


class EarlyStopping:
    def __init__(self, patience: int = 6, min_delta: float = 1e-4, checkpoint_path: Path | str = "checkpoints/best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f" -> Checkpoint saved (val_loss: {val_loss:.5f})")
        else:
            self.counter += 1
            print(f" -> No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    tracker: ErrorTracker,
    device: torch.device,
    epoch: int = 1,
    epochs: int = 1
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch:03d}/{epochs:03d} [Train]", 
        leave=False, 
        unit="batch"
    )

    for batch in pbar:
        sat_seq = batch["sat_seq"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        predictions = model(sat_seq)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        running_loss += loss_value * sat_seq.size(0)
        tracker.update(predictions, target)

        pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    tracker: ErrorTracker,
    device: torch.device,
    stage: str = "Val"
) -> tuple[float, dict]:
    model.eval()
    running_loss = 0.0

    pbar = tqdm(
        dataloader, 
        desc=f"[{stage}]", 
        leave=False, 
        unit="batch"
    )

    with torch.no_grad():
        for batch in pbar:
            sat_seq = batch["sat_seq"].to(device)
            target = batch["target"].to(device)

            predictions = model(sat_seq)
            loss = criterion(predictions, target)

            loss_value = loss.item()
            running_loss += loss_value * sat_seq.size(0)
            tracker.update(predictions, target)

            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return epoch_loss, metrics


def run_training(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    tracker: ErrorTracker,
    logger: TensorBoardLogger,
    early_stopping: EarlyStopping,
    epochs: int,
    device: torch.device
):
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, tracker, device, epoch=epoch, epochs=epochs
        )
        train_metrics = tracker.compute()

        val_loss, val_metrics = evaluate(
            model, loaders["val"], criterion, tracker, device, stage="Val"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)

        # Extract overall metrics for cleaner code
        tr_o, val_o = train_metrics["overall"], val_metrics["overall"]
        tr_a, val_a = train_metrics["overall_active"], val_metrics["overall_active"]

        print(f"\n{'='*90}")
        print(f" EPOCH {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        print(f"{'-'*90}")

        # Overall Metrics (All Hours)
        print(" [ OVERALL METRICS - ALL HOURS ]")
        print(f"   Train | MAE: {tr_o['mae']:6.2f} kW | RMSE: {tr_o['rmse']:6.2f} kW | nMAE: {tr_o['nmae']:5.2f}% | MBE: {tr_o['mbe']:6.2f} kW | R²: {tr_o['r2']:6.4f}")
        print(f"   Val   | MAE: {val_o['mae']:6.2f} kW | RMSE: {val_o['rmse']:6.2f} kW | nMAE: {val_o['nmae']:5.2f}% | MBE: {val_o['mbe']:6.2f} kW | R²: {val_o['r2']:6.4f}")
        print(f"{'-'*90}")

        # Overall Metrics (Daylight Only)
        print(" [ OVERALL METRICS - DAYLIGHT ONLY (P > 0) ]")
        print(f"   Train | MAE: {tr_a['mae']:6.2f} kW | RMSE: {tr_a['rmse']:6.2f} kW | nMAE: {tr_a['nmae']:5.2f}% | MBE: {tr_a['mbe']:6.2f} kW | R²: {tr_a['r2']:6.4f}")
        print(f"   Val   | MAE: {val_a['mae']:6.2f} kW | RMSE: {val_a['rmse']:6.2f} kW | nMAE: {val_a['nmae']:5.2f}% | MBE: {val_a['mbe']:6.2f} kW | R²: {val_a['r2']:6.4f}")
        print(f"{'-'*90}")

        # Per-Step Metrics Table (Daylight Only)
        print(" [ HOURLY BREAKDOWN - MAE & nMAE (Daylight Only) ]")
        print("   Step  |  Train MAE  |   Val MAE   | Train nMAE |  Val nMAE ")
        print("  -------------------------------------------------------------")

        seq_len_out = len(train_metrics["per_step"]["mae"])
        for step in range(seq_len_out):
            tr_mae = train_metrics["per_step_active"]["mae"][step]
            val_mae = val_metrics["per_step_active"]["mae"][step]
            tr_nmae = train_metrics["per_step_active"]["nmae"][step]
            val_nmae = val_metrics["per_step_active"]["nmae"][step]
            
            print(f"   t+{step+1:<2}  | {tr_mae:8.2f} kW | {val_mae:8.2f} kW | {tr_nmae:7.2f} % | {val_nmae:7.2f} %")
            
        print(f"{'='*90}\n")

        # 5. Control Early Stopping
        if early_stopping(val_loss, model):
            print("Early stopping triggered! Training finished.")
            break