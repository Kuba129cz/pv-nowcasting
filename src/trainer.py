from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

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


def _print_metrics_summary(metrics: dict, stage_name: str, second_metrics: dict | None = None, second_stage_name: str | None = None):
    o = metrics["overall"]
    a = metrics["overall_active"]
    
    print(f"\n{'='*90}")
    print(f" [ OVERALL METRICS - ALL HOURS ({stage_name}) ]")
    print(f"   {stage_name:<5} | MAE: {o['mae']:6.2f} kW | RMSE: {o['rmse']:6.2f} kW | nMAE: {o['nmae']:5.2f}% | MBE: {o['mbe']:6.2f} kW | R²: {o['r2']:6.4f}")
    
    if second_metrics and second_stage_name:
        so = second_metrics["overall"]
        print(f"   {second_stage_name:<5} | MAE: {so['mae']:6.2f} kW | RMSE: {so['rmse']:6.2f} kW | nMAE: {so['nmae']:5.2f}% | MBE: {so['mbe']:6.2f} kW | R²: {so['r2']:6.4f}")
    print(f"{'-'*90}")

    print(f" [ OVERALL METRICS - DAYLIGHT ONLY (P > 0) ({stage_name}) ]")
    print(f"   {stage_name:<5} | MAE: {a['mae']:6.2f} kW | RMSE: {a['rmse']:6.2f} kW | nMAE: {a['nmae']:5.2f}% | MBE: {a['mbe']:6.2f} kW | R²: {a['r2']:6.4f}")
    
    if second_metrics and second_stage_name:
        sa = second_metrics["overall_active"]
        print(f"   {second_stage_name:<5} | MAE: {sa['mae']:6.2f} kW | RMSE: {sa['rmse']:6.2f} kW | nMAE: {sa['nmae']:5.2f}% | MBE: {sa['mbe']:6.2f} kW | R²: {sa['r2']:6.4f}")
    print(f"{'-'*90}")

    print(f" [ HOURLY BREAKDOWN - MAE & nMAE (Daylight Only) ]")
    if second_metrics and second_stage_name:
        print(f"   Step  | {stage_name} MAE | {second_stage_name} MAE | {stage_name} nMAE | {second_stage_name} nMAE")
        print(f"  -------------------------------------------------------------")
        seq_len_out = len(metrics["per_step"]["mae"])
        for step in range(seq_len_out):
            m1 = metrics["per_step_active"]["mae"][step]
            m2 = second_metrics["per_step_active"]["mae"][step]
            n1 = metrics["per_step_active"]["nmae"][step]
            n2 = second_metrics["per_step_active"]["nmae"][step]
            print(f"   t+{step+1:<2}  | {m1:8.2f} kW | {m2:8.2f} kW | {n1:7.2f} % | {n2:7.2f} %")
    else:
        print(f"   Step  | {stage_name} MAE | {stage_name} nMAE")
        print(f"  -----------------------------------")
        seq_len_out = len(metrics["per_step"]["mae"])
        for step in range(seq_len_out):
            m = metrics["per_step_active"]["mae"][step]
            n = metrics["per_step_active"]["nmae"][step]
            print(f"   t+{step+1:<2}  | {m:8.2f} kW | {n:7.2f} %")
            
    print(f"{'='*90}\n")


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

        print(f"\n EPOCH {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")
        
        _print_metrics_summary(train_metrics, "Train", val_metrics, "Val")

        if early_stopping(val_loss, model):
            print("Early stopping triggered! Training finished.")
            break


def run_testing(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    tracker: ErrorTracker,
    device: torch.device,
    best_model_path: Path | str,
    logger: TensorBoardLogger, 
    args: argparse.Namespace   
) -> dict:
    print("\n" + "="*90)
    print(" LOADING BEST MODEL FOR FINAL TEST EVALUATION")
    print("="*90)
    
    model.load_state_dict(torch.load(best_model_path))

    test_loss, test_metrics = evaluate(
        model=model, 
        dataloader=dataloader, 
        criterion=criterion, 
        tracker=tracker, 
        device=device, 
        stage="Test"
    )

    logger.log_test(epoch=0, test_loss=test_loss, metrics=test_metrics, args=args)

    _print_metrics_summary(test_metrics, "Test")
    
    return test_metrics