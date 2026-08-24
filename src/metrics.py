import numpy as np
import torch

from src.scalers.tabular import PowerScaler

class ErrorTracker:
    def __init__(self, seq_len_out: int, power_scaler: PowerScaler, nominal_capacity_fve:float):
        if not isinstance(power_scaler, PowerScaler):
            raise ValueError(f"power_scaler is not instance of PowerScaler class.")

        if nominal_capacity_fve <= 0:
            raise ValueError(f"nominal capacity of fve cannot be equal or less than zero!")

        self._power_scaler = power_scaler
        self.seq_len_out = seq_len_out
        self.nominal_capacity_fve = nominal_capacity_fve

        self._preds = []
        self._targets = []

    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._preds.append(y_pred.detach().cpu().numpy())
        self._targets.append(y_true.detach().cpu().numpy())

    def _reset(self):
        self._preds.clear()
        self._targets.clear()

    def compute(self) -> dict:
        if not self._preds:
            return {}
        preds_scaled = np.concatenate(self._preds, axis=0)
        targets_scaled = np.concatenate(self._targets, axis=0)
 
        preds_real = self._power_scaler.inverse_transform_target(preds_scaled) # zde to ale bude spoustet ten warning ne?
        targets_real = self._power_scaler.inverse_transform_target(targets_scaled)

        results = self._calculate_all_metrics(preds_real, targets_real)

        self._reset()

        return results

    def _calc_core(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        if len(targets) == 0:
            return {"mae": 0.0, "nmae": 0.0, "mbe": 0.0, "mape": 0.0, "rmse": 0.0, "r2": 0.0}

        diff = preds - targets
        mae = np.mean(np.abs(diff))
        nmae = (mae / self.nominal_capacity_fve) * 100.0
        mbe = np.mean(diff)
        rmse = np.sqrt(np.mean(diff**2))

        safe_targets = np.where(np.abs(targets) < 1e-3, 1e-3, targets)
        mape = np.mean(np.abs(diff / safe_targets)) * 100.0

        ss_res = np.sum(diff**2)
        ss_tot = np.sum((targets - np.mean(targets))**2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        return {"mae": mae, "nmae": nmae, "mbe": mbe, "mape": mape, "rmse": rmse, "r2": r2}
    
    def _calculate_all_metrics(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        # OVERALL
        overall = self._calc_core(preds.flatten(), targets.flatten())
        
        # OVERALL ACTIVE  P > 0
        active_mask = targets > 0
        overall_active = self._calc_core(preds[active_mask], targets[active_mask])

        metrics_keys = overall.keys()
        per_step = {k: np.zeros(self.seq_len_out) for k in metrics_keys}
        per_step_active = {k: np.zeros(self.seq_len_out) for k in metrics_keys}

        # PER-STEP a PER-STEP ACTIVE 
        for step in range(self.seq_len_out):
            p_step = preds[:, step]
            t_step = targets[:, step]
            
            res_step = self._calc_core(p_step, t_step)
            for k in metrics_keys:
                per_step[k][step] = res_step[k]
                
            mask_step = t_step > 0
            res_step_act = self._calc_core(p_step[mask_step], t_step[mask_step])
            for k in metrics_keys:
                per_step_active[k][step] = res_step_act[k]

        return {
            "overall": overall,
            "overall_active": overall_active,
            "per_step": per_step,
            "per_step_active": per_step_active
        }


