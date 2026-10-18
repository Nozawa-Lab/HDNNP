# train.py (修正版)
'''
hdnnp.train

Training loop for HDNNP with selectable optimizer and early stopping.
Records per-atom Energy and Force RMSE per iteration.
'''
import math
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


def _compute_energy_loss(
    e_pred_batch: Tensor,
    e_true_batch: Tensor,
    n_atoms_batch: Tensor,
    normalize_per_atom: bool = False,
) -> Tensor:
    """
    エネルギー損失を計算する。
    normalize_per_atom=True の場合は各系の原子数で割って 1 原子あたりの誤差として扱う。
    原子数が 0 または教師エネルギーが非有限値の系は平均から除外する。
    """
    n_atoms = n_atoms_batch.to(device=e_pred_batch.device, dtype=e_pred_batch.dtype)
    e_true = e_true_batch.to(device=e_pred_batch.device, dtype=e_pred_batch.dtype)
    valid = (n_atoms > 0) & torch.isfinite(e_true)
    if not valid.any():
        # Keep the result connected to the prediction graph while contributing no loss.
        return e_pred_batch.sum() * 0.0

    error = e_pred_batch[valid] - e_true[valid]
    if normalize_per_atom:
        error = error / n_atoms[valid]
    return error.square().mean()


def _process_batch_structure_by_structure(
    batch_data: Dict[str, Any],
    model: torch.nn.Module,
    sym_calc: SymmetryCalculator,
    device: torch.device,
    backward: bool,
) -> Dict[str, float]:
    """Compute a batch while releasing each structure's graph immediately."""
    r_batch = batch_data['R']
    z_batch = batch_data['Z']
    cell_batch = batch_data['cell']
    e_true_batch = batch_data['E'].squeeze(-1)
    f_true_batch = batch_data['F']
    n_atoms_batch = batch_data['N_atoms'].long()
    atom_mask_batch = batch_data['atom_mask'].bool()

    batch_size = int(n_atoms_batch.numel())
    if batch_size == 0:
        raise ValueError("Received an empty batch.")

    valid_energy_mask = (n_atoms_batch > 0) & torch.isfinite(e_true_batch)
    num_valid_energies = int(valid_energy_mask.sum().item())

    expanded_atom_mask = atom_mask_batch.unsqueeze(-1).expand_as(f_true_batch)
    valid_force_mask = expanded_atom_mask & torch.isfinite(f_true_batch)
    num_valid_force_components = (
        int(valid_force_mask.sum().item())
        if hdnnp_config.USE_FORCE_TRAINING
        else 0
    )

    if num_valid_energies == 0 and num_valid_force_components == 0:
        raise ValueError("Batch contains no valid energy or force targets.")

    total_loss_value = 0.0
    energy_squared_error_sum = 0.0
    energy_sample_count = 0
    force_squared_error_sum = 0.0
    force_component_count = 0

    for i in range(batch_size):
        n_atoms_i = int(n_atoms_batch[i].item())
        if n_atoms_i <= 0:
            continue

        # Only the current, unpadded structure is resident on the accelerator.
        ri = r_batch[i, :n_atoms_i].to(device).detach()
        if hdnnp_config.USE_FORCE_TRAINING:
            ri = ri.clone().requires_grad_(True)
        zi = z_batch[i, :n_atoms_i].to(device)
        cell_i = cell_batch[i].to(device)

        gi = sym_calc.compute(ri, zi, cell_i)
        e_pred_i = model(gi, zi).sum()
        structure_loss = e_pred_i * 0.0

        has_valid_energy = bool(valid_energy_mask[i].item())
        if has_valid_energy:
            e_true_i = e_true_batch[i].to(device=device, dtype=e_pred_i.dtype)
            energy_error = e_pred_i - e_true_i
            energy_error_per_atom = energy_error / float(n_atoms_i)
            energy_error_for_loss = (
                energy_error_per_atom
                if hdnnp_config.ENERGY_LOSS_NORMALIZE_PER_ATOM
                else energy_error
            )
            structure_loss = structure_loss + (
                hdnnp_config.LOSS_ALPHA_E
                * energy_error_for_loss.square()
                / num_valid_energies
            )
            energy_squared_error_sum += float(
                energy_error_per_atom.detach().square().cpu()
            )
            energy_sample_count += 1

        if hdnnp_config.USE_FORCE_TRAINING:
            force_grad = torch.autograd.grad(
                outputs=e_pred_i,
                inputs=ri,
                create_graph=backward,
                retain_graph=backward,
                allow_unused=False,
            )[0]
            f_pred_i = -force_grad
            f_true_i = f_true_batch[i, :n_atoms_i].to(
                device=device, dtype=f_pred_i.dtype
            )
            valid_force_i = valid_force_mask[i, :n_atoms_i].to(device)

            has_valid_force = bool(valid_force_i.any().item())
            if has_valid_force:
                force_error = f_pred_i[valid_force_i] - f_true_i[valid_force_i]
                force_error_sq_sum = force_error.square().sum()
                structure_loss = structure_loss + (
                    hdnnp_config.LOSS_BETA_F
                    * force_error_sq_sum
                    / num_valid_force_components
                )
                force_squared_error_sum += float(force_error_sq_sum.detach().cpu())
                force_component_count += int(valid_force_i.sum().item())

        if backward and structure_loss.requires_grad:
            structure_loss.backward()

        total_loss_value += float(structure_loss.detach().cpu())

        # Do not keep the previous structure alive while building the next graph.
        del structure_loss, e_pred_i, gi, ri, zi, cell_i
        if has_valid_energy:
            del e_true_i, energy_error, energy_error_per_atom, energy_error_for_loss
        if hdnnp_config.USE_FORCE_TRAINING:
            del force_grad, f_pred_i, f_true_i, valid_force_i
            if has_valid_force:
                del force_error, force_error_sq_sum

    rmse_energy = (
        math.sqrt(energy_squared_error_sum / energy_sample_count)
        if energy_sample_count > 0
        else float('nan')
    )
    rmse_force = (
        math.sqrt(force_squared_error_sum / force_component_count)
        if force_component_count > 0
        else float('nan')
    )
    return {
        'loss': total_loss_value,
        'rmse_energy_per_atom': rmse_energy,
        'rmse_force': rmse_force,
    }


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
            target_value=hdnnp_config.EARLY_STOPPING_TARGET_VALUE,
        )

    loss_hist: List[float] = []
    rmse_E_hist: List[float] = []
    rmse_F_hist: List[float] = []
    total_training_time_tracker = 0.0

    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            header_str = (
                f"{'Iter':>5s}/{hdnnp_config.TRAIN_MAX_ITERATIONS:<5d} | "
                f"{'Loss':<12s} | {'RMSE E/atom':<15s} | {'RMSE F/atom':<15s} | "
                f"{'LR':<10s} | {'Time/iter (s)':<15s}"
            )
            print(header_str)
            print("-" * len(header_str))
            log_f.write(header_str + "\n")
            log_f.write("-" * len(header_str) + "\n")
            log_f.flush()

            data_iterator = iter(dataloader)
            for iteration in range(hdnnp_config.TRAIN_MAX_ITERATIONS):
                iter_start_time = time.time()
                model.train()

                try:
                    batch_data = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(dataloader)
                    try:
                        batch_data = next(data_iterator)
                    except StopIteration:
                        print(
                            "[ERROR] DataLoader is empty. Check your dataset and splits.json.",
                            file=sys.stderr,
                        )
                        break

                if optimizer_name == "lbfgs":
                    closure_metrics: Dict[str, float] = {}

                    def closure() -> Tensor:
                        optimizer.zero_grad(set_to_none=True)
                        metrics = _process_batch_structure_by_structure(
                            batch_data=batch_data,
                            model=model,
                            sym_calc=sym_calc,
                            device=device,
                            backward=True,
                        )
                        closure_metrics.clear()
                        closure_metrics.update(metrics)
                        # The gradients are already accumulated structure by structure.
                        return torch.tensor(metrics['loss'], device=device)

                    optimizer.step(closure)
                    if not closure_metrics:
                        raise RuntimeError("LBFGS closure was not evaluated.")
                    current_metrics = closure_metrics.copy()
                else:
                    optimizer.zero_grad(set_to_none=True)
                    current_metrics = _process_batch_structure_by_structure(
                        batch_data=batch_data,
                        model=model,
                        sym_calc=sym_calc,
                        device=device,
                        backward=True,
                    )
                    optimizer.step()

                current_loss_val_item = current_metrics['loss']
                current_rmse_E_pa_val = current_metrics['rmse_energy_per_atom']
                current_rmse_F_pa_val = current_metrics['rmse_force']
                loss_hist.append(current_loss_val_item)
                rmse_E_hist.append(current_rmse_E_pa_val)
                rmse_F_hist.append(current_rmse_F_pa_val)

                iter_duration = time.time() - iter_start_time
                total_training_time_tracker += iter_duration

                current_lr_str = "N/A"
                if optimizer_name in ["adam", "sgd"] and optimizer.param_groups:
                    current_lr = optimizer.param_groups[0].get('lr', float('nan'))
                    current_lr_str = f"{current_lr:.1e}" if not np.isnan(current_lr) else "N/A"

                log_line = (
                    f"{iteration + 1:>5d}/{hdnnp_config.TRAIN_MAX_ITERATIONS:<5d} | "
                    f"{current_loss_val_item:<12.6e} | "
                    f"{current_rmse_E_pa_val:<15.4e} | "
                    f"{current_rmse_F_pa_val:<15.4e} | "
                    f"{current_lr_str:<10s} | "
                    f"{iter_duration:<15.2f}"
                )
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
                    elif (
                        hdnnp_config.EARLY_STOPPING_MONITOR == 'rmse_f_pa'
                        and hdnnp_config.USE_FORCE_TRAINING
                    ):
                        monitored_value = current_rmse_F_pa_val

                    if monitored_value is not None and not np.isnan(monitored_value):
                        if early_stopper(monitored_value):
                            early_stop_msg = f"Early stopping triggered at iteration {iteration + 1}."
                            print(early_stop_msg)
                            log_f.write(early_stop_msg + "\n")
                            break
                    elif monitored_value is not None:
                        nan_warn_msg = (
                            "[WARN] Early stopping monitor value is NaN at iteration "
                            f"{iteration + 1}. Skipping check."
                        )
                        print(nan_warn_msg)
                        log_f.write(nan_warn_msg + "\n")

            log_f.write("-" * len(header_str) + "\n")
            avg_iter_time = (
                total_training_time_tracker / len(loss_hist)
                if loss_hist
                else float('nan')
            )
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
                print(
                    f"[ERROR] Additionally, failed to write final error to log file: {e_log_final}",
                    file=sys.stderr,
                )

    model.eval()
    return model, loss_hist, rmse_E_hist, rmse_F_hist
