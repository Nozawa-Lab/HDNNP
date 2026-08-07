# train_lbfgs_batch_epoch.py
"""
L-BFGS をバッチ処理で回し、
- 1エポックごとの要約ログ
- 各バッチに割り当てられた構造名
- 各更新ごとの損失・RMSE
を別ファイルに出力する学習ループ。
"""

import json
import time
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader

from . import config as hdnnp_config
from .symmetry_calculator import SymmetryCalculator


class EarlyStopper:
    """学習の早期終了を管理するクラス。"""

    def __init__(self, patience: int, min_delta: float, mode: str = 'min', target_value: Optional[float] = None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.target_value = target_value
        self.counter = 0
        self.best_value: Optional[float] = None
        self.early_stop = False

        if self.mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'")

    def __call__(self, current_value: float) -> bool:
        if self.early_stop:
            return True

        if self.target_value is not None:
            if self.mode == 'min' and current_value <= self.target_value:
                self.early_stop = True
                return True
            elif self.mode == 'max' and current_value >= self.target_value:
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
                self.early_stop = True
        return self.early_stop


def _write_jsonl(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def _build_dataloader_for_epoch(dataloader: DataLoader) -> DataLoader:
    """各エポックでシャッフルされた新しい DataLoader を生成する。"""
    return DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        collate_fn=dataloader.collate_fn,
        drop_last=getattr(dataloader, "drop_last", False), #あまりのデータを捨てるかどうか
    )


def _compute_energy_loss(
    e_pred_batch: Tensor,
    e_true_batch: Tensor,
    n_atoms_batch: Tensor,
    normalize_per_atom: bool = False,
) -> Tensor:
    """エネルギー損失を計算する。"""
    if normalize_per_atom:
        n_atoms_safe = n_atoms_batch.to(dtype=e_pred_batch.dtype).clamp_min(1.0)
        e_pred_for_loss = e_pred_batch / n_atoms_safe
        e_true_for_loss = e_true_batch / n_atoms_safe
    else:
        e_pred_for_loss = e_pred_batch
        e_true_for_loss = e_true_batch

    return torch.mean((e_pred_for_loss - e_true_for_loss) ** 2)


def _compute_batch_loss_and_predictions(
    batch_data: Dict[str, Tensor],
    model: torch.nn.Module,
    sym_calc: SymmetryCalculator,
    device: torch.device,
    use_force_training: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """1 バッチ分の損失と予測値を計算する。"""
    R_padded = batch_data['R'].to(device)
    Z_padded = batch_data['Z'].to(device)
    cell_batch = batch_data['cell'].to(device)
    E_true_batch = batch_data['E'].squeeze(-1).to(device)
    F_true_padded = batch_data['F'].to(device)
    N_atoms_batch = batch_data['N_atoms'].to(device).float()
    atom_mask_batch = batch_data['atom_mask'].to(device)

    R_local_for_grad = R_padded.detach().clone().requires_grad_(use_force_training)

    E_pred_samples_list = []
    for i in range(R_padded.shape[0]): #バッチに含まれる各構造に対して、原子数分のエネルギーを計算する
        n_atoms_i = N_atoms_batch[i].int().item() 
        if n_atoms_i == 0:
            E_pred_samples_list.append(torch.tensor(0.0, device=device, dtype=R_local_for_grad.dtype))
            continue

        Ri, Zi, cell_mati = R_local_for_grad[i, :n_atoms_i], Z_padded[i, :n_atoms_i], cell_batch[i]
        Gi = sym_calc.compute(Ri, Zi, cell_mati)
        Ei_atoms = model(Gi, Zi)
        E_pred_samples_list.append(Ei_atoms.sum()) #各構造の全エネルギーをまとめたリスト

    E_pred_batch_for_loss = torch.stack(E_pred_samples_list)
    energy_loss_val = _compute_energy_loss(
        e_pred_batch=E_pred_batch_for_loss,
        e_true_batch=E_true_batch,
        n_atoms_batch=N_atoms_batch,
        normalize_per_atom=hdnnp_config.ENERGY_LOSS_NORMALIZE_PER_ATOM,
    )
    loss = hdnnp_config.LOSS_ALPHA_E * energy_loss_val

    F_pred_padded_for_loss = torch.zeros_like(R_local_for_grad)
    if use_force_training:
        total_energy_for_forces = E_pred_batch_for_loss.sum()
        forces_grad_tuple = torch.autograd.grad(
            outputs=total_energy_for_forces,
            inputs=R_local_for_grad,
            grad_outputs=torch.ones_like(total_energy_for_forces),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        forces_grad = forces_grad_tuple[0]
        if forces_grad is not None:
            F_pred_padded_for_loss = -forces_grad

        force_diff_sq = (F_pred_padded_for_loss - F_true_padded) ** 2
        masked_force_loss_components = force_diff_sq * atom_mask_batch.unsqueeze(-1)
        num_actual_force_components = atom_mask_batch.sum().float() * 3.0
        force_loss_val = torch.tensor(0.0, device=device)
        if num_actual_force_components > hdnnp_config.NUMERICAL_EPSILON:
            force_loss_val = masked_force_loss_components.sum() / num_actual_force_components
        loss += hdnnp_config.LOSS_BETA_F * force_loss_val

    return (
        loss, #損失関数値
        E_pred_batch_for_loss, #各系の全エネルギーの予測値
        F_pred_padded_for_loss, #各系の全原子の力の予測値
        E_true_batch, #各系の全エネルギーの真の値
        F_true_padded, #各系の全原子の力の真の値
        N_atoms_batch, #各系の原子数
        atom_mask_batch, #各系の原子数に応じたマスクテンソル (原子が存在する位置は1, 存在しない位置は0)
    )


def train_lbfgs_batch_epoch(
    dataloader: DataLoader, #run_train.pyで作成されたtrain_loaderを想定
    model: torch.nn.Module,
    sym_calc: SymmetryCalculator,
    device: torch.device,
    log_file_path: Path,
    batch_size: Optional[int] = None,
) -> Tuple[torch.nn.Module, List[float], List[float], List[float]]:
    """
    L-BFGS をバッチ処理で回し、エポックごとのログと各バッチ/更新結果をファイルに保存する。
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    try:
        data_iterator = iter(dataloader) #train_loaderのイテレータを作成
        batch_data: Dict[str, Tensor] = next(data_iterator)
    except StopIteration:
        print("[ERROR] DataLoader is empty. Check your dataset and splits.json.", file=sys.stderr)
        return model, [], [], []

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("lr", 0.1),
        max_iter=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("max_iter", 200),
        line_search_fn=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("line_search_fn", "strong_wolfe"),
        history_size=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("history_size", 100),
        tolerance_grad=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("tolerance_grad", 1e-7),
        tolerance_change=hdnnp_config.OPTIMIZER_PARAMS["LBFGS"].get("tolerance_change", 1e-9),
    )

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

    log_dir = log_file_path.parent
    batch_structures_path = log_dir / f"{log_file_path.stem}_batch_structures.jsonl"
    batch_updates_path = log_dir / f"{log_file_path.stem}_batch_updates.jsonl"
    epoch_summary_path = log_dir / f"{log_file_path.stem}_epoch_summary.jsonl"

    for path in [batch_structures_path, batch_updates_path, epoch_summary_path]:
        if path.exists():
            path.unlink() #既存のファイルを削除して新しいログを作成する

    try:
        with open(log_file_path, 'w', encoding='utf-8') as log_f:
            header_str = (f"{'Epoch':>5s} | {'Updates':>7s} | {'Avg Loss':<12s} | {'Avg RMSE E/atom':<15s} | {'Avg RMSE F/atom':<15s} | {'Elapsed[s]':<10s}")
            print(header_str)
            print("-" * len(header_str))
            log_f.write(header_str + "\n")
            log_f.write("-" * len(header_str) + "\n")
            log_f.flush()

            eval_results = {}
            epoch_num = 0
            epoch_update_count = 0
            epoch_loss_vals: List[float] = []
            epoch_start_time: Optional[float] = None
            epoch_rmse_e_vals: List[float] = []
            epoch_rmse_f_vals: List[float] = []

            for iteration in range(hdnnp_config.TRAIN_MAX_ITERATIONS): #MAXITERATIONは、batch数で割り切れる数に指定する。
                if epoch_start_time is None:
                    epoch_start_time = time.time()
                iter_start_time = time.time()
                model.train()

                current_loss_val_item = 0.0
                E_pred_batch_eval = torch.zeros_like(batch_data['E'].squeeze(-1).to(device))
                F_pred_padded_eval = torch.zeros_like(batch_data['F'].to(device))

                def closure():
                    optimizer.zero_grad()
                    loss, E_pred_batch_for_loss, F_pred_padded_closure, _, _, _, _ = _compute_batch_loss_and_predictions(
                        batch_data=batch_data, #1バッチ分のデータを渡す
                        model=model,
                        sym_calc=sym_calc,
                        device=device,
                        use_force_training=hdnnp_config.USE_FORCE_TRAINING,
                    ) #損失関数値loss、バッチ内の各系の全エネルギーの予測値E_pred_batch_for_loss、バッチ内の各系の全原子の力の予測値F_pred_padded_closureを返す
                    if loss.requires_grad:
                        loss.backward()
                    eval_results['E_pred'] = E_pred_batch_for_loss.detach()
                    eval_results['F_pred'] = F_pred_padded_closure.detach()
                    return loss

                loss_after_step = optimizer.step(closure)
                current_loss_val_item = loss_after_step.item() #更新後の損失関数値
                E_pred_batch_eval = eval_results['E_pred'] #最終的なバッチ内の各系の全エネルギーの予測値
                F_pred_padded_eval = eval_results['F_pred'] #最終的なバッチ内の各系の全原子の力の予測値

                loss_hist.append(current_loss_val_item)
                model.eval()

                with torch.no_grad():
                    current_rmse_E_pa_val = float('nan')
                    current_rmse_F_pa_val = float('nan')

                    E_true_batch = batch_data['E'].squeeze(-1).to(device)
                    F_true_padded = batch_data['F'].to(device)
                    N_atoms_batch = batch_data['N_atoms'].to(device).float()
                    atom_mask_batch = batch_data['atom_mask'].to(device)

                    valid_samples_mask_e = N_atoms_batch > 0
                    if valid_samples_mask_e.any():
                        N_atoms_valid = N_atoms_batch[valid_samples_mask_e]
                        E_pred_per_atom = E_pred_batch_eval[valid_samples_mask_e] / N_atoms_valid #各系の全エネルギーの予測値を原子数で割って、各原子あたりのエネルギーに変換
                        E_true_per_atom = E_true_batch[valid_samples_mask_e] / N_atoms_valid #各系の全エネルギーの真値を原子数で割って、各原子あたりのエネルギーに変換
                        if E_pred_per_atom.numel() > 0:
                            rmse_tensor = torch.sqrt(torch.mean((E_pred_per_atom - E_true_per_atom) ** 2))
                            current_rmse_E_pa_val = rmse_tensor.item()

                    if hdnnp_config.USE_FORCE_TRAINING:
                        force_error_sq = (F_pred_padded_eval - F_true_padded) ** 2
                        masked_force_error_sq = force_error_sq * atom_mask_batch.unsqueeze(-1)
                        total_sum_sq_force_error = masked_force_error_sq.sum()
                        num_total_actual_atoms = atom_mask_batch.sum().float()
                        if num_total_actual_atoms > hdnnp_config.NUMERICAL_EPSILON:
                            rmse_tensor = torch.sqrt(total_sum_sq_force_error / (num_total_actual_atoms * 3.0)) #各原子の成分あたりの力のRMSEを計算する。
                            current_rmse_F_pa_val = rmse_tensor.item()

                rmse_E_hist.append(current_rmse_E_pa_val)
                rmse_F_hist.append(current_rmse_F_pa_val)

                epoch_loss_vals.append(current_loss_val_item)
                epoch_rmse_e_vals.append(current_rmse_E_pa_val)
                epoch_rmse_f_vals.append(current_rmse_F_pa_val)
                epoch_update_count += 1

                structure_ids = batch_data.get('sample_id', []) or []
                if isinstance(structure_ids, torch.Tensor):
                    structure_ids = structure_ids.tolist()
                elif not isinstance(structure_ids, list):
                    structure_ids = [structure_ids]
                batch_record = {
                    'epoch': epoch_num + 1,
                    'update_idx': epoch_update_count,
                    'batch_size': len(structure_ids),
                    'structure_ids': structure_ids,
                }
                _write_jsonl(batch_structures_path, batch_record)

                update_record = {
                    'epoch': epoch_num + 1,
                    'update_idx': epoch_update_count,
                    'loss': float(current_loss_val_item),
                    'rmse_e_pa': float(current_rmse_E_pa_val),
                    'rmse_f_pa': float(current_rmse_F_pa_val),
                    'batch_size': len(structure_ids),
                    'structure_ids': structure_ids,
                }
                _write_jsonl(batch_updates_path, update_record)

                try:
                    batch_data = next(data_iterator) #バッチが尽きるまで、順に取得
                except StopIteration: #data_iteratorが尽きたら、次のエポックに進む
                    dataloader = _build_dataloader_for_epoch(dataloader) #一度train_loaderを作り直して、シャッフルされた新しいイテレータを作成する
                    data_iterator = iter(dataloader)
                    batch_data = next(data_iterator)
                    epoch_num += 1

                    if epoch_loss_vals:
                        avg_loss = float(np.mean(epoch_loss_vals))
                        avg_rmse_e = float(np.mean([v for v in epoch_rmse_e_vals if not np.isnan(v)])) if epoch_rmse_e_vals else float('nan')
                        avg_rmse_f = float(np.mean([v for v in epoch_rmse_f_vals if not np.isnan(v)])) if epoch_rmse_f_vals else float('nan')
                    else:
                        avg_loss = float('nan')
                        avg_rmse_e = float('nan')
                        avg_rmse_f = float('nan')

                    epoch_elapsed_sec = float(time.time() - epoch_start_time) if epoch_start_time is not None else float('nan')
                    epoch_summary = {
                        'epoch': epoch_num,
                        'updates': len(epoch_loss_vals),
                        'avg_loss': avg_loss,
                        'avg_rmse_e_pa': avg_rmse_e,
                        'avg_rmse_f_pa': avg_rmse_f,
                        'elapsed_sec': epoch_elapsed_sec,
                    }
                    _write_jsonl(epoch_summary_path, epoch_summary)

                    log_line = (f"{epoch_num:>5d} | {len(epoch_loss_vals):>7d} | "
                                f"{avg_loss:<12.6e} | {avg_rmse_e:<15.6e} | {avg_rmse_f:<15.6e} | {epoch_elapsed_sec:<10.3f}")
                    print(log_line)
                    log_f.write(log_line + "\n")
                    log_f.flush()
                    sys.stdout.flush()

                    if early_stopper:
                        monitored_value = None
                        if hdnnp_config.EARLY_STOPPING_MONITOR == 'loss':
                            monitored_value = avg_loss
                        elif hdnnp_config.EARLY_STOPPING_MONITOR == 'rmse_e_pa':
                            monitored_value = avg_rmse_e
                        elif hdnnp_config.EARLY_STOPPING_MONITOR == 'rmse_f_pa' and hdnnp_config.USE_FORCE_TRAINING:
                            monitored_value = avg_rmse_f

                        if monitored_value is not None and not np.isnan(monitored_value):
                            if early_stopper(monitored_value):
                                break

                    epoch_loss_vals = []
                    epoch_rmse_e_vals = []
                    epoch_rmse_f_vals = []
                    epoch_update_count = 0
                    epoch_start_time = None

            log_f.write("-" * len(header_str) + "\n")

    except IOError as e_io:
        print(f"[ERROR] Could not write to log file {log_file_path}: {e_io}", file=sys.stderr)
    except Exception as e_train:
        print(f"[ERROR] An unexpected error occurred during training: {e_train}", file=sys.stderr)

    model.eval()
    return model, loss_hist, rmse_E_hist, rmse_F_hist
