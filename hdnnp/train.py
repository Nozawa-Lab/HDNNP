# train.py (修正版)
'''
hdnnp.train

Training loop for HDNNP with selectable optimizer and early stopping.
Records per-atom Energy and Force RMSE per iteration.
'''
import time
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader

from . import config as hdnnp_config
from .symmetry_calculator import SymmetryCalculator
# EarlyStopper クラスは、このファイル内またはインポートされるモジュールで定義されていると仮定します。
# もし未定義の場合は、以前の回答から EarlyStopper クラスの定義をここに含めてください。

class EarlyStopper:
    """
    学習の早期終了を管理するクラス。
    指定された回数 (patience) だけ監視対象の指標が改善しなかった場合に停止する。
    """
    def __init__(self, patience: int, min_delta: float, mode: str = 'min', target_value: Optional[float] = None):
        """
        Args:
            patience (int): 改善が見られなくなってから待つエポック数/イテレーション数。
            min_delta (float): 「改善」とみなす最小の変化量。
            mode (str): 'min' または 'max'。監視対象が小さいほど良い場合は 'min'。
            target_value (Optional[float]): 監視対象がこの値に達したら停止する目標値。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.target_value = target_value
        self.counter = 0
        self.best_value: Optional[float] = None
        self.early_stop = False

        if self.mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")
        self.delta_sign = -1 if self.mode == 'min' else 1

    def __call__(self, current_value: float) -> bool:
        """
        現在の監視対象の値を受け取り、早期終了すべきかどうかを判断する。
        Args:
            current_value (float): 現在の監視対象の値。
        Returns:
            bool: 早期終了すべきなら True。
        """
        if self.early_stop:
            return True

        if self.target_value is not None:
            if self.mode == 'min' and current_value <= self.target_value:
                print(f"EarlyStopping: Monitored value {current_value:.6e} reached target {self.target_value:.6e}.")
                self.early_stop = True
                return True
            elif self.mode == 'max' and current_value >= self.target_value:
                print(f"EarlyStopping: Monitored value {current_value:.6e} reached target {self.target_value:.6e}.")
                self.early_stop = True
                return True

        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = (self.best_value - current_value) > self.min_delta
        else:
            improved = (current_value - self.best_value) > self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"EarlyStopping: Monitored value did not improve for {self.patience} iterations from best {self.best_value:.6e} (current: {current_value:.6e}).")
                self.early_stop = True
        return self.early_stop


def train(
    dataloader: DataLoader,
    model: torch.nn.Module,
    sym_calc: SymmetryCalculator,
    device: torch.device,
    log_file_path: Path
) -> Tuple[torch.nn.Module, List[float], List[float], List[float]]:
    """
    HDNNPモデルの学習ループ。
    Args:
        dataloader (DataLoader): 学習データを提供するデータローダ。
        model (torch.nn.Module): 学習対象のHDNNPモデル。
        sym_calc (SymmetryCalculator): 対称性関数計算機。
        device (torch.device): 計算に使用するデバイス。
        log_file_path (Path): イテレーションごとのログを保存するファイルパス。
    Returns:
        Tuple[torch.nn.Module, List[float], List[float], List[float]]:
            - 学習済みモデル, 損失の履歴, エネルギーRMSEの履歴, 力RMSEの履歴
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    try:
        batch_data: Dict[str, Tensor] = next(iter(dataloader))
    except StopIteration:
        error_msg = "[ERROR] DataLoader is empty. Check your dataset and splits.json."
        print(error_msg, file=sys.stderr)
        return model, [], [], []

    R_padded = batch_data['R'].to(device)
    Z_padded = batch_data['Z'].to(device)
    cell_batch = batch_data['cell'].to(device)
    E_true_batch = batch_data['E'].squeeze(-1).to(device)
    F_true_padded = batch_data['F'].to(device)
    N_atoms_batch = batch_data['N_atoms'].to(device).float()
    atom_mask_batch = batch_data['atom_mask'].to(device)

    B, N_max, _ = R_padded.shape

    optimizer_name = hdnnp_config.OPTIMIZER_NAME.lower()
    opt_params_all = {k.lower(): v for k, v in hdnnp_config.OPTIMIZER_PARAMS.items()}
    opt_params = opt_params_all.get(optimizer_name, {})

    if optimizer_name == "lbfgs":
        optimizer = optim.LBFGS(model.parameters(), **opt_params)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), **opt_params)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), **opt_params)
    else:
        raise ValueError(f"Unsupported optimizer: {hdnnp_config.OPTIMIZER_NAME}")

    early_stopper = None
    if hdnnp_config.EARLY_STOPPING_ENABLED:
        early_stopper = EarlyStopper(
            patience=hdnnp_config.EARLY_STOPPING_PATIENCE,
            min_delta=hdnnp_config.EARLY_STOPPING_MIN_DELTA,
            mode=hdnnp_config.EARLY_STOPPING_MODE,
            target_value=hdnnp_config.EARLY_STOPPING_TARGET_VALUE
        )

    loss_hist: List[float] = []
    rmse_E_hist: List[float] = []
    rmse_F_hist: List[float] = []
    total_training_time_tracker = 0.0

    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            header_str = (f"{'Iter':>5s}/{hdnnp_config.TRAIN_MAX_ITERATIONS:<5d} | "
                          f"{'Loss':<12s} | {'RMSE E/atom':<15s} | {'RMSE F/atom':<15s} | "
                          f"{'LR':<10s} | {'Time/iter (s)':<15s}")
            print(header_str)
            print("-" * len(header_str))
            log_f.write(header_str + "\n")
            log_f.write("-" * len(header_str) + "\n")
            log_f.flush()

            eval_results = {}

            for iteration in range(hdnnp_config.TRAIN_MAX_ITERATIONS):
                iter_start_time = time.time()
                model.train()

                current_loss_val_item: float = 0.0
                E_pred_batch_eval = torch.zeros_like(E_true_batch)
                F_pred_padded_eval = torch.zeros_like(F_true_padded)

                # --- 勾配計算が必要かどうかを判断 ---
                requires_force_grad = hdnnp_config.USE_FORCE_TRAINING and optimizer_name in ["adam", "sgd"]
                R_local_for_grad = R_padded.detach().clone().requires_grad_(hdnnp_config.USE_FORCE_TRAINING)

                if optimizer_name == "lbfgs":
                    def closure():
                        optimizer.zero_grad()
                        # LBFGSでは常に勾配が必要
                        R_closure_grad = R_padded.detach().clone().requires_grad_(True)
                        
                        E_pred_samples_list_closure = []
                        for i in range(B):
                            n_atoms_i = N_atoms_batch[i].int().item()
                            if n_atoms_i == 0:
                                E_pred_samples_list_closure.append(torch.tensor(0.0, device=device, dtype=R_closure_grad.dtype))
                                continue
                            Ri, Zi, cell_mati = R_closure_grad[i, :n_atoms_i], Z_padded[i, :n_atoms_i], cell_batch[i]
                            Gi = sym_calc.compute(Ri, Zi, cell_mati)
                            Ei_atoms = model(Gi, Zi)
                            E_pred_samples_list_closure.append(Ei_atoms.sum())

                        E_pred_batch_for_loss = torch.stack(E_pred_samples_list_closure)
                        energy_loss_val = torch.mean((E_pred_batch_for_loss - E_true_batch)**2)
                        loss = hdnnp_config.LOSS_ALPHA_E * energy_loss_val
                        
                        F_pred_padded_closure = torch.zeros_like(R_closure_grad)
                        
                        if hdnnp_config.USE_FORCE_TRAINING:
                            total_energy_for_forces = E_pred_batch_for_loss.sum()
                            forces_grad_tuple = torch.autograd.grad(
                                outputs=total_energy_for_forces, inputs=R_closure_grad,
                                grad_outputs=torch.ones_like(total_energy_for_forces),
                                create_graph=True, retain_graph=True, allow_unused=True
                            )
                            forces_grad = forces_grad_tuple[0]
                            if forces_grad is not None:
                                F_pred_padded_closure = -forces_grad

                            force_diff_sq = (F_pred_padded_closure - F_true_padded)**2
                            masked_force_loss_components = force_diff_sq * atom_mask_batch.unsqueeze(-1)
                            num_actual_force_components = atom_mask_batch.sum().float() * 3.0
                            force_loss_val = torch.tensor(0.0, device=device)
                            if num_actual_force_components > hdnnp_config.NUMERICAL_EPSILON:
                                force_loss_val = masked_force_loss_components.sum() / num_actual_force_components
                            loss += hdnnp_config.LOSS_BETA_F * force_loss_val

                        if loss.requires_grad:
                            loss.backward()
                        
                        eval_results['E_pred'] = E_pred_batch_for_loss.detach()
                        eval_results['F_pred'] = F_pred_padded_closure.detach()
                        
                        return loss
                    
                    loss_after_step = optimizer.step(closure)
                    current_loss_val_item = loss_after_step.item()
                    E_pred_batch_eval = eval_results['E_pred']
                    F_pred_padded_eval = eval_results['F_pred']

                else: # Adam, SGD
                    optimizer.zero_grad()
                    
                    E_pred_samples_list = []
                    for i in range(B):
                        n_atoms_i = N_atoms_batch[i].int().item()
                        if n_atoms_i == 0:
                            E_pred_samples_list.append(torch.tensor(0.0, device=device, dtype=R_local_for_grad.dtype))
                            continue
                        Ri, Zi, cell_mati = R_local_for_grad[i, :n_atoms_i], Z_padded[i, :n_atoms_i], cell_batch[i]
                        Gi = sym_calc.compute(Ri, Zi, cell_mati)
                        Ei_atoms = model(Gi, Zi)
                        E_pred_samples_list.append(Ei_atoms.sum())
                    
                    E_pred_batch_for_loss = torch.stack(E_pred_samples_list)
                    energy_loss_val = torch.mean((E_pred_batch_for_loss - E_true_batch)**2)
                    loss = hdnnp_config.LOSS_ALPHA_E * energy_loss_val
                    
                    F_pred_padded_for_loss = torch.zeros_like(R_local_for_grad)
                    
                    if hdnnp_config.USE_FORCE_TRAINING:
                        total_energy_for_forces = E_pred_batch_for_loss.sum()
                        forces_grad_tuple = torch.autograd.grad(
                            outputs=total_energy_for_forces, inputs=R_local_for_grad,
                            grad_outputs=torch.ones_like(total_energy_for_forces),
                            create_graph=False, retain_graph=True, allow_unused=True
                        )
                        forces_grad = forces_grad_tuple[0]

                        if forces_grad is not None:
                            F_pred_padded_for_loss = -forces_grad

                        force_diff_sq = (F_pred_padded_for_loss - F_true_padded)**2
                        masked_force_loss_components = force_diff_sq * atom_mask_batch.unsqueeze(-1)
                        num_actual_force_components = atom_mask_batch.sum().float() * 3.0
                        force_loss_val = torch.tensor(0.0, device=device)
                        if num_actual_force_components > hdnnp_config.NUMERICAL_EPSILON:
                             force_loss_val = masked_force_loss_components.sum() / num_actual_force_components
                        loss += hdnnp_config.LOSS_BETA_F * force_loss_val

                    if loss.requires_grad:
                        loss.backward()
                    optimizer.step()
                    current_loss_val_item = loss.item()
                    
                    E_pred_batch_eval = E_pred_batch_for_loss.detach()
                    F_pred_padded_eval = F_pred_padded_for_loss.detach()

                loss_hist.append(current_loss_val_item)
                model.eval()

                with torch.no_grad():
                    current_rmse_E_pa_val = float('nan')
                    current_rmse_F_pa_val = float('nan')

                    valid_samples_mask_e = N_atoms_batch > 0
                    if valid_samples_mask_e.any():
                        N_atoms_valid = N_atoms_batch[valid_samples_mask_e]
                        E_pred_per_atom = E_pred_batch_eval[valid_samples_mask_e] / N_atoms_valid
                        E_true_per_atom = E_true_batch[valid_samples_mask_e] / N_atoms_valid
                        if E_pred_per_atom.numel() > 0:
                            rmse_tensor = torch.sqrt(torch.mean((E_pred_per_atom - E_true_per_atom)**2))
                            current_rmse_E_pa_val = rmse_tensor.item()
                    
                    if hdnnp_config.USE_FORCE_TRAINING:
                        force_error_sq = (F_pred_padded_eval - F_true_padded)**2
                        masked_force_error_sq = force_error_sq * atom_mask_batch.unsqueeze(-1)
                        total_sum_sq_force_error = masked_force_error_sq.sum()
                        num_total_actual_atoms = atom_mask_batch.sum().float()
                        
                        if num_total_actual_atoms > hdnnp_config.NUMERICAL_EPSILON:
                            rmse_tensor = torch.sqrt(total_sum_sq_force_error / (num_total_actual_atoms * 3.0))
                            current_rmse_F_pa_val = rmse_tensor.item()

                rmse_E_hist.append(current_rmse_E_pa_val)
                rmse_F_hist.append(current_rmse_F_pa_val)
                
                iter_duration = time.time() - iter_start_time
                total_training_time_tracker += iter_duration

                current_lr_str = "N/A"
                if optimizer_name in ["adam", "sgd"]:
                    if optimizer.param_groups:
                        current_lr = optimizer.param_groups[0].get('lr', float('nan'))
                        current_lr_str = f"{current_lr:.1e}" if not np.isnan(current_lr) else "N/A"
                
                log_line = (f"{iteration+1:>5d}/{hdnnp_config.TRAIN_MAX_ITERATIONS:<5d} | "
                            f"{current_loss_val_item:<12.6e} | "
                            f"{current_rmse_E_pa_val:<15.4e} | "
                            f"{current_rmse_F_pa_val:<15.4e} | "
                            f"{current_lr_str:<10s} | "
                            f"{iter_duration:<15.2f}")
                
                print(log_line)
                log_f.write(log_line + "\n")
                log_f.flush()
                sys.stdout.flush()

                if early_stopper:
                    monitored_value: Optional[float] = None
                    if hdnnp_config.EARLY_STOPPING_MONITOR == 'loss':
                        monitored_value = current_loss_val_item
                    elif hdnnp_config.EARLY_STOPPING_MONITOR == 'rmse_e_pa':
                        monitored_value = current_rmse_E_pa_val
                    elif hdnnp_config.EARLY_STOPPING_MONITOR == 'rmse_f_pa' and hdnnp_config.USE_FORCE_TRAINING:
                        monitored_value = current_rmse_F_pa_val
                    
                    if monitored_value is not None and not np.isnan(monitored_value):
                        if early_stopper(monitored_value):
                            early_stop_msg = f"Early stopping triggered at iteration {iteration+1}."
                            print(early_stop_msg)
                            log_f.write(early_stop_msg + "\n")
                            break
                    elif monitored_value is not None and np.isnan(monitored_value):
                        nan_warn_msg = f"[WARN] Early stopping monitor value is NaN at iteration {iteration+1}. Skipping check."
                        print(nan_warn_msg)
                        log_f.write(nan_warn_msg + "\n")

            log_f.write("-" * len(header_str) + "\n")
            avg_iter_time = total_training_time_tracker / len(loss_hist) if len(loss_hist) > 0 else float('nan')
            avg_iter_msg = f"Average time per iteration: {avg_iter_time:.2f}s"
            print(avg_iter_msg)
            log_f.write(avg_iter_msg + "\n")
            log_f.flush()

    except IOError as e_io:
        print(f"[ERROR] Could not write to log file {log_file_path}: {e_io}", file=sys.stderr)
    except Exception as e_train:
        error_summary = f"[ERROR] An unexpected error occurred during training: {e_train}"
        print(error_summary, file=sys.stderr)
        if 'log_f' in locals() and hasattr(log_f, 'closed') and not log_f.closed:
            try:
                log_f.write(error_summary + "\nTraining loop terminated due to an error.\n")
                log_f.flush()
            except Exception as e_log_final:
                print(f"[ERROR] Additionally, failed to write final error to log file: {e_log_final}", file=sys.stderr)

    model.eval()
    return model, loss_hist, rmse_E_hist, rmse_F_hist