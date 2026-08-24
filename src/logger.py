# src/logger.py
from torch.utils.tensorboard import SummaryWriter
import argparse

class TensorBoardLogger:
    def __init__(self, log_dir: str, args: argparse.Namespace | None = None):
        self.writer = SummaryWriter(log_dir=log_dir)
        if args is not None:
            self.log_hparams_config(args)

    def log_hparams_config(self, args: argparse.Namespace) -> None:
        config_dict = vars(args)
        md_table = "| Parameter | Value |\n| --- | --- |\n"
        for key, value in config_dict.items():
            md_table += f"| `{key}` | `{value}` |\n"
        
        self.writer.add_text("Hyperparameters/Configuration", md_table, global_step=0)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, t_metrics: dict, v_metrics: dict, lr: float):
        self.writer.add_scalar('Global/Learning_Rate', lr, epoch)
        self.writer.add_scalars('Global/Scaled_Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        
        # Overall metrics
        self.writer.add_scalars('Overall/MAE', {'Train': t_metrics["overall"]["mae"], 'Val': v_metrics["overall"]["mae"]}, epoch)
        self.writer.add_scalars('Overall/RMSE', {'Train': t_metrics["overall"]["rmse"], 'Val': v_metrics["overall"]["rmse"]}, epoch)
        
        # Active metrics
        self.writer.add_scalars('Active/MAE', {'Train': t_metrics["overall_active"]["mae"], 'Val': v_metrics["overall_active"]["mae"]}, epoch)
        self.writer.add_scalars('Active/nMAE_pct', {'Train': t_metrics["overall_active"]["nmae"], 'Val': v_metrics["overall_active"]["nmae"]}, epoch)

    def log_test(self, epoch: int, test_loss: float, metrics: dict, args: argparse.Namespace):
        self.writer.add_scalar('Test/Scaled_Loss', test_loss, epoch)
        self.writer.add_scalar('Test/MAE', metrics['overall']['mae'], epoch)
        self.writer.add_scalar('Test/nMAE_Active', metrics['overall_active']['nmae'], epoch)
        
        for step, val in enumerate(metrics['per_step']['mae']):
            self.writer.add_scalar('Test_PerStep/MAE', val, step)

        hparams = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
        self.writer.add_hparams(hparams, {
            "hparams/Test_Overall_MAE": metrics['overall']['mae'],
            "hparams/Test_Active_nMAE": metrics['overall_active']['nmae']
        })

    def close(self):
        self.writer.close()