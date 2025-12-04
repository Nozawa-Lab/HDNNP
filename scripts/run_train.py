#!/usr/bin/env python3
"""
scripts/run_train.py  — HDNNP training script with configurable optimizer and early stopping.
"""
import sys
import json
import time
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import math
import random
from typing import Optional
import argparse # argparseをインポート

# Project root resolution should be placed before importing hdnnp
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hdnnp import config as hdnnp_config

# ANSIカラーコード
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"

def log(message, level="INFO"):
    """指定されたレベルでメッセージを色付きでコンソールに出力します。"""
    colors = {"INFO": GREEN, "DEBUG": CYAN, "WARN": YELLOW, "ERROR": RED}
    print(f"{colors.get(level, RESET)}[{level}]{RESET} {message}")

try:
    # configからCPU使用率を読み込む
    ratio = hdnnp_config.CPU_USAGE_RATIO
    if not (0.0 < ratio <= 1.0):
        log(f"CPU_USAGE_RATIO in config must be between 0.0 and 1.0, but got {ratio}. Defaulting to 0.8.", level="WARN")
        ratio = 0.8
        
    total_cores = os.cpu_count()
    # 指定された割合でCPUスレッド数を計算
    num_threads = max(1, math.floor(total_cores * ratio))
    torch.set_num_threads(num_threads)
    # ログメッセージを修正
    log(f"PyTorch thread count set to {num_threads} ({total_cores} total cores available, {ratio*100:.0f}% usage).")
except Exception as e:
    log(f"Could not set PyTorch thread count automatically. Error: {e}", level="WARN")

from hdnnp.data_loader import get_dataloader
from hdnnp.train import train
from hdnnp.symmetry_calculator import SymmetryCalculator
from hdnnp.model import HDNNPModel

PROCESSED_DIR = hdnnp_config.PROCESSED_DATA_DIR
SPLITS_JSON = PROCESSED_DIR / hdnnp_config.SPLITS_FILENAME

def setup_device():
    """計算に使用するデバイスを設定します。"""
    if hdnnp_config.DEVICE == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(hdnnp_config.DEVICE)
    log(f"Using device: {device}")
    return device

def set_random_seed(seed: int):
    """乱数シードを設定して再現性を確保します。"""
    torch.manual_seed(seed)
    # if use multi-GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # CuDNNの設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"Random seed set to {seed} for reproducibility.")

def calculate_and_save_sf_range(
    dataloader: torch.utils.data.DataLoader,
    sym_calc: SymmetryCalculator,
    device: torch.device,
    output_path: Path
):
    """
    データセット全体の対称性関数を計算し、各成分の最小値と最大値を保存します。
    LBFGS用に全データを1バッチで読み込むことを想定しています。
    """
    log("Calculating symmetry function range over the entire training set for extrapolation check...")
    
    try:
        data_batch = next(iter(dataloader))
    except StopIteration:
        log("Dataloader is empty. Cannot calculate SF range.", level="WARN")
        return

    R_padded = data_batch['R'].to(device)
    Z_padded = data_batch['Z'].to(device)
    cell_batch = data_batch['cell'].to(device)
    N_atoms_batch = data_batch['N_atoms'].to(device).int()
    
    all_g_vectors = []
    with torch.no_grad():
        for i in range(len(R_padded)):
            n_atoms = N_atoms_batch[i].item()
            if n_atoms == 0:
                continue
            
            R = R_padded[i, :n_atoms]
            Z = Z_padded[i, :n_atoms]
            cell = cell_batch[i]
            
            # 対称性関数を計算
            g_vectors = sym_calc.compute(R, Z, cell)
            all_g_vectors.append(g_vectors)

    if not all_g_vectors:
        log("No valid structures found to calculate SF range.", level="WARN")
        return

    full_g_tensor = torch.cat(all_g_vectors, dim=0)
    
    # 最小値と最大値を計算
    g_min = torch.min(full_g_tensor, dim=0).values
    g_max = torch.max(full_g_tensor, dim=0).values

    # 辞書にまとめて保存
    sf_range_data = {'G_min': g_min.cpu(), 'G_max': g_max.cpu()}
    torch.save(sf_range_data, output_path)
    log(f"Symmetry function range (min/max) saved to: {output_path}")
    log(f"  - Total atoms processed: {full_g_tensor.shape[0]}")
    log(f"  - SF vector dimension: {full_g_tensor.shape[1]}")

def evaluate(
    model_eval: HDNNPModel,
    dataloader_eval: torch.utils.data.DataLoader,
    sym_calc_eval: SymmetryCalculator,
    device_eval: torch.device,
    force_training_enabled: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    指定されたデータローダを使用してモデルを評価し、
    エネルギーと力のRMSE、および真値と予測値のペアを返します。
    """
    log(f"Evaluating model on {dataloader_eval.dataset.split if hasattr(dataloader_eval.dataset, 'split') else 'provided'} data...")
    eval_start_time = time.time()
    model_eval.eval()

    try:
        data_batch = next(iter(dataloader_eval))
    except StopIteration:
        log(f"[ERROR] DataLoader for evaluation is empty.", level="ERROR")
        nan_tensor = torch.tensor(float('nan'), device=device_eval)
        empty_tensor = torch.empty(0, device=device_eval)
        return nan_tensor, nan_tensor, empty_tensor, empty_tensor

    R_batch = data_batch['R'].to(device_eval)
    Z_batch = data_batch['Z'].to(device_eval)
    cell_batch = data_batch['cell'].to(device_eval)
    E_true_batch = data_batch['E'].squeeze(-1).to(device_eval)
    F_true_batch = data_batch['F'].to(device_eval)
    N_atoms_batch = data_batch['N_atoms'].to(device_eval).float()
    atom_mask_batch = data_batch['atom_mask'].to(device_eval)

    B, N_max, _ = R_batch.shape

    E_pred_list = []
    with torch.no_grad():
        for i in range(B):
            n_atoms_i = N_atoms_batch[i].int().item()
            if n_atoms_i == 0:
                E_pred_list.append(torch.tensor(0.0, device=device_eval, dtype=R_batch.dtype))
                continue
            current_R_frame = R_batch[i, :n_atoms_i]
            current_Z_frame = Z_batch[i, :n_atoms_i]
            current_cell = cell_batch[i]
            G_atoms_frame = sym_calc_eval.compute(current_R_frame, current_Z_frame, current_cell)
            E_atoms_pred_frame = model_eval(G_atoms_frame, current_Z_frame)
            E_pred_list.append(E_atoms_pred_frame.sum())
    E_pred_batch = torch.stack(E_pred_list)

    F_pred_batch = torch.full_like(F_true_batch, float('nan'))
    if force_training_enabled:
        with torch.enable_grad():
            r_input_for_grad = R_batch.detach().clone().requires_grad_(True)
            E_pred_samples_list = []
            for i in range(B):
                n_atoms_i = N_atoms_batch[i].int().item()
                if n_atoms_i == 0:
                    E_pred_samples_list.append(torch.tensor(0.0, device=device_eval, dtype=r_input_for_grad.dtype))
                    continue
                Gi = sym_calc_eval.compute(r_input_for_grad[i, :n_atoms_i], Z_batch[i, :n_atoms_i], cell_batch[i])
                Ei_atoms = model_eval(Gi, Z_batch[i, :n_atoms_i])
                E_pred_samples_list.append(Ei_atoms.sum())
            
            E_total_for_grad = torch.stack(E_pred_samples_list).sum()
            
            forces_grad_tuple = torch.autograd.grad(
                outputs=E_total_for_grad, inputs=r_input_for_grad,
                grad_outputs=torch.ones_like(E_total_for_grad),
                create_graph=False, retain_graph=False, allow_unused=True
            )
            
            if forces_grad_tuple[0] is not None:
                F_pred_batch = -forces_grad_tuple[0]
            else:
                log("Warning: Force gradient was None during evaluation.", level="WARN")

    with torch.no_grad():
        valid_samples_mask_e = N_atoms_batch > 0
        rmse_E_pa = torch.tensor(float('nan'), device=device_eval)
        E_true_pa_for_plot = torch.empty(0, device=device_eval)
        E_pred_pa_for_plot = torch.empty(0, device=device_eval)

        if valid_samples_mask_e.any():
            N_atoms_valid = N_atoms_batch[valid_samples_mask_e]
            E_pred_per_atom_valid = E_pred_batch[valid_samples_mask_e] / N_atoms_valid
            E_true_per_atom_valid = E_true_batch[valid_samples_mask_e] / N_atoms_valid
            if E_pred_per_atom_valid.numel() > 0:
                rmse_E_pa = torch.sqrt(torch.mean((E_pred_per_atom_valid - E_true_per_atom_valid)**2))
                E_true_pa_for_plot = E_true_per_atom_valid
                E_pred_pa_for_plot = E_pred_per_atom_valid

        rmse_F_pa = torch.tensor(float('nan'), device=device_eval)
        if force_training_enabled:
            force_error_sq = (F_pred_batch - F_true_batch)**2
            masked_force_error_sq_components = force_error_sq * atom_mask_batch.unsqueeze(-1)
            total_sum_sq_force_error = masked_force_error_sq_components.sum()
            num_total_actual_atoms = atom_mask_batch.sum().float()
            if num_total_actual_atoms > 0:
                rmse_F_pa = torch.sqrt(total_sum_sq_force_error / num_total_actual_atoms)

    log(f"Evaluation completed in {time.time() - eval_start_time:.2f} seconds")
    return rmse_E_pa, rmse_F_pa, E_true_pa_for_plot.cpu(), E_pred_pa_for_plot.cpu()

def save_config_to_json(config_module, output_path: Path):
    """実行時のconfigの内容をJSONファイルに保存する"""
    config_dict = {
        key: getattr(config_module, key)
        for key in dir(config_module)
        if not key.startswith('__') and not callable(getattr(config_module, key))
    }
    try:
        # PathオブジェクトなどをJSONシリアライズ可能な型に変換
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, (tuple)): # tupleはjsonではlistになる
                config_dict[key] = list(value)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4)
        log(f"Saved current configuration to {output_path}")
    except Exception as e:
        log(f"Failed to save configuration: {e}", level="ERROR")


def main():
    # --- コマンドライン引数の設定 ---
    parser = argparse.ArgumentParser(description="HDNNP Training Script")
    parser.add_argument('--R_CUT', type=float, help=f"Override R_CUT (default: {hdnnp_config.R_CUT})")
    parser.add_argument('--HIDDEN_LAYERS', type=int, nargs='+', help=f"Override HIDDEN_LAYERS (default: {hdnnp_config.HIDDEN_LAYERS})")
    parser.add_argument('--LOSS_ALPHA_E', type=float, help=f"Override LOSS_ALPHA_E (default: {hdnnp_config.LOSS_ALPHA_E})")
    parser.add_argument('--LOSS_BETA_F', type=float, help=f"Override LOSS_BETA_F (default: {hdnnp_config.LOSS_BETA_F})")
    parser.add_argument('--CHECKPOINT_DIR', type=Path, help="Set a custom checkpoint directory for outputs")
    
    args = parser.parse_args()

    # --- configの値を引数で上書き ---
    if args.R_CUT is not None:
        hdnnp_config.R_CUT = args.R_CUT
        log(f"[CONFIG OVERRIDE] Set R_CUT to {hdnnp_config.R_CUT}", "WARN")
    if args.HIDDEN_LAYERS is not None:
        hdnnp_config.HIDDEN_LAYERS = args.HIDDEN_LAYERS
        log(f"[CONFIG OVERRIDE] Set HIDDEN_LAYERS to {hdnnp_config.HIDDEN_LAYERS}", "WARN")
    if args.LOSS_ALPHA_E is not None:
        hdnnp_config.LOSS_ALPHA_E = args.LOSS_ALPHA_E
        log(f"[CONFIG OVERRIDE] Set LOSS_ALPHA_E to {hdnnp_config.LOSS_ALPHA_E}", "WARN")
    if args.LOSS_BETA_F is not None:
        hdnnp_config.LOSS_BETA_F = args.LOSS_BETA_F
        log(f"[CONFIG OVERRIDE] Set LOSS_BETA_F to {hdnnp_config.LOSS_BETA_F}", "WARN")

    # --- 出力ディレクトリの決定 ---
    if args.CHECKPOINT_DIR:
        CHECKPOINT_DIR = args.CHECKPOINT_DIR
    else:
        # 引数指定がない場合、configのデフォルトのchkptディレクトリを使用
        CHECKPOINT_DIR = hdnnp_config.CHECKPOINT_DIR
    log(f"Results will be saved in: {CHECKPOINT_DIR.resolve()}")
    
    # --- 設定をJSONに保存 ---
    save_config_to_json(hdnnp_config, CHECKPOINT_DIR / "config_used.json")
    
    # ログディレクトリもCHECKPOINT_DIRを基準にする
    LOG_DIR = CHECKPOINT_DIR / "log"

    # --- 以下、元のmain関数の処理 ---
    if hdnnp_config.SET_RANDOM_SEED:
        set_random_seed(hdnnp_config.RANDOM_SEED_VALUE)

    DEVICE = setup_device()

    log("Loading symmetry calculator...")
    sym_calc = SymmetryCalculator()
    INPUT_DIM = sym_calc.total_sf_dim
    log(f"Symmetry calculator loaded. Input dim: {INPUT_DIM}")
    
    use_force = hdnnp_config.USE_FORCE_TRAINING
    log(f"Force training is {'ENABLED' if use_force else 'DISABLED'}.")

    log("Loading training/validation splits...")
    if not SPLITS_JSON.exists():
        log(f"Error: splits.json not found at {SPLITS_JSON}", level="ERROR")
        sys.exit(1)
    with open(SPLITS_JSON, 'r', encoding='utf-8') as f:
        splits = json.load(f)

    if 'train' not in splits or not splits['train']:
        log(f"Error: No training data found in {SPLITS_JSON}", level="ERROR")
        sys.exit(1)
    n_train = len(splits['train'])
    log(f"Train samples: {n_train}")

    log("Loading training data...")
    is_lbfgs = hdnnp_config.OPTIMIZER_NAME.lower() == "lbfgs"
    train_batch_size = n_train if is_lbfgs else hdnnp_config.DATALOADER_TRAIN_BATCH_SIZE
    log(f"Using training batch size: {train_batch_size}")

    train_loader_full_no_shuffle = get_dataloader(
        splits_json=SPLITS_JSON,
        processed_dir=PROCESSED_DIR,
        split='train',
        batch_size=n_train,
        shuffle=False,
        num_workers=hdnnp_config.DATALOADER_NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda' and hdnnp_config.DATALOADER_PIN_MEMORY)
    )

    train_loader = get_dataloader(
        splits_json=SPLITS_JSON,
        processed_dir=PROCESSED_DIR,
        split='train',
        batch_size=train_batch_size,
        shuffle=hdnnp_config.DATALOADER_SHUFFLE_TRAIN and not is_lbfgs,
        num_workers=hdnnp_config.DATALOADER_NUM_WORKERS,
        pin_memory=(DEVICE.type == 'cuda' and hdnnp_config.DATALOADER_PIN_MEMORY)
    )
    log("Training data loaded.")

    if hdnnp_config.EXTRAPOLATION_CHECK_ENABLED:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        sf_range_path = CHECKPOINT_DIR / hdnnp_config.SF_RANGE_FILENAME
        calculate_and_save_sf_range(train_loader_full_no_shuffle, sym_calc, DEVICE, sf_range_path)

    log("Initializing model...")
    model = HDNNPModel(INPUT_DIM).to(DEVICE)
    log("Model initialized.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"train_log_{hdnnp_config.OPTIMIZER_NAME.lower()}_{timestamp}.log"
    log_file_path = LOG_DIR / log_file_name
    log(f"Training logs will be saved to: {log_file_path}")

    log(f"Starting training using {hdnnp_config.OPTIMIZER_NAME} optimizer for max {hdnnp_config.TRAIN_MAX_ITERATIONS} iterations...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    training_start_time = time.time()
    model_trained, loss_hist, rmse_E_hist, rmse_F_hist = train(
        dataloader=train_loader,
        model=model,
        sym_calc=sym_calc,
        device=DEVICE,
        log_file_path=log_file_path
    )
    training_duration = time.time() - training_start_time
    log(f"Training completed in {training_duration:.2f} seconds. Ran for {len(loss_hist)} iterations.")

    if model_trained is not None:
        model_save_path = CHECKPOINT_DIR / hdnnp_config.TRAINED_MODEL_FILENAME
        try:
            torch.save(model_trained.state_dict(), model_save_path)
            log(f"Trained model state_dict saved to: {model_save_path}")
        except Exception as e:
            log(f"Error saving model: {e}", level="ERROR")
    else:
        log("Training did not return a model. Skipping model save.", level="WARN")

    if model_trained is not None:
        log("Evaluating final trained model on training data...")
        rmse_E_final, rmse_F_final, E_true_pa_plot, E_pred_pa_plot = evaluate(
            model_trained, train_loader_full_no_shuffle, sym_calc, DEVICE, use_force
        )
        log(f"Final per-atom Energy RMSE (on training data): {rmse_E_final.item() if not torch.isnan(rmse_E_final) else 'N/A':.6f}")
        if use_force:
            log(f"Final per-atom Force RMSE (on training data): {rmse_F_final.item() if not torch.isnan(rmse_F_final) else 'N/A':.6f}")

        if E_true_pa_plot.numel() > 0 and E_pred_pa_plot.numel() > 0:
            log("Generating scatter plot for training data...")
            E_true_pa_np = E_true_pa_plot.numpy()
            E_pred_pa_np = E_pred_pa_plot.numpy()

            plt.figure(figsize=(8,6))
            plt.scatter(E_true_pa_np, E_pred_pa_np, alpha=0.5, label="Data points")
            min_val_scatter, max_val_scatter = 0, 0
            if E_true_pa_np.size > 0 and E_pred_pa_np.size > 0:
                min_val_scatter = min(E_true_pa_np.min(), E_pred_pa_np.min())
                max_val_scatter = max(E_true_pa_np.max(), E_pred_pa_np.max())
                plt.plot([min_val_scatter, max_val_scatter], [min_val_scatter, max_val_scatter], 'r--', label="y=x")

            plt.xlabel('True Energy per Atom (eV/atom)')
            plt.ylabel('Predicted Energy per Atom (eV/atom)')
            plt.title('Training Data: Per-atom Energy True vs Predicted')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            scatter_plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_ENERGY_SCATTER_TRAIN_FILENAME
            plt.savefig(scatter_plot_path, dpi=hdnnp_config.PLOT_DPI)
            log(f'Saved scatter plot: {scatter_plot_path}')
            plt.close()
        else:
            log("No data points to generate scatter plot for training data.", level="WARN")

        log("Generating RMSE plots for training...")
        iterations_ran = range(1, len(loss_hist) + 1)

        if loss_hist:
            fig_loss, ax_loss = plt.subplots(figsize=(8,6))
            ax_loss.plot(iterations_ran, loss_hist, marker='.', linestyle='-', label="Total Loss")
            ax_loss.set_xlabel('Iteration'); ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Training: Total Loss vs. Iteration')
            loss_np_array = np.array([val for val in loss_hist if val is not None and not np.isnan(val)])
            if loss_np_array.size > 0 and np.all(loss_np_array > 0):
                 ax_loss.set_yscale('log')
            else:
                log("Warning: Loss contains non-positive, NaN or no values. Using linear scale for y-axis.", level="WARN")
            ax_loss.grid(True, which="both", ls="-"); ax_loss.legend(); fig_loss.tight_layout()
            loss_plot_path = CHECKPOINT_DIR / "loss_vs_iter_train.png"
            fig_loss.savefig(loss_plot_path, dpi=hdnnp_config.PLOT_DPI)
            log(f'Saved Loss plot: {loss_plot_path}')
            plt.close(fig_loss)
        else:
            log("No Loss history to plot.", level="WARN")

        if rmse_E_hist:
            fig_e, ax_e = plt.subplots(figsize=(8,6))
            ax_e.plot(iterations_ran, rmse_E_hist, marker='o', linestyle='-', label="Energy RMSE/atom")
            ax_e.set_xlabel('Iteration'); ax_e.set_ylabel('Per-atom Energy RMSE (eV/atom)')
            ax_e.set_title('Training: Per-atom Energy RMSE vs. Iteration')
            rmse_E_np_array = np.array([val for val in rmse_E_hist if val is not None and not np.isnan(val)])
            if rmse_E_np_array.size > 0 and np.all(rmse_E_np_array > 0):
                ax_e.set_yscale('log')
            else:
                log("Warning: Energy RMSE contains non-positive, NaN or no values. Using linear scale for y-axis.", level="WARN")
            ax_e.grid(True, which="both", ls="-"); ax_e.legend(); fig_e.tight_layout()
            rmse_e_plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_RMSE_ENERGY_TRAIN_FILENAME
            fig_e.savefig(rmse_e_plot_path, dpi=hdnnp_config.PLOT_DPI)
            log(f'Saved Energy RMSE plot: {rmse_e_plot_path}')
            plt.close(fig_e)
        else:
            log("No Energy RMSE history to plot.", level="WARN")

        if use_force and rmse_F_hist:
            fig_f, ax_f = plt.subplots(figsize=(8,6))
            ax_f.plot(iterations_ran, rmse_F_hist, marker='s', linestyle='-', label="Force RMSE/atom", color='orange')
            ax_f.set_xlabel('Iteration'); ax_f.set_ylabel('Per-atom Force RMSE (eV/Å)')
            ax_f.set_title('Training: Per-atom Force RMSE vs. Iteration')
            rmse_F_np_array = np.array([val for val in rmse_F_hist if val is not None and not np.isnan(val)])
            if rmse_F_np_array.size > 0 and np.all(rmse_F_np_array > 0):
                ax_f.set_yscale('log')
            else:
                log("Warning: Force RMSE contains non-positive, NaN or no values. Using linear scale for y-axis.", level="WARN")
            ax_f.grid(True, which="both", ls="-"); ax_f.legend(); fig_f.tight_layout()
            rmse_f_plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_RMSE_FORCE_TRAIN_FILENAME
            fig_f.savefig(rmse_f_plot_path, dpi=hdnnp_config.PLOT_DPI)
            log(f'Saved Force RMSE plot: {rmse_f_plot_path}')
            plt.close(fig_f)
        elif use_force:
            log("No Force RMSE history to plot.", level="WARN")
    else:
        log("Model training failed. Skipping final evaluation and plotting.", level="WARN")

if __name__ == '__main__':
    main()