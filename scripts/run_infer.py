#!/usr/bin/env python3
"""
scripts/run_infer.py  — 学習済みHDNNPモデルによる推論・評価 (最適化・結果出力・グラフ描画機能付き)
"""
import sys
import json
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Tuple, Dict, Optional
import os
import math
import random
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
    colors = {"INFO": GREEN, "DEBUG": CYAN, "WARN": YELLOW, "ERROR": RED}
    print(f"{colors.get(level, RESET)}[{level}]{RESET} {message}")

try:
    ratio = hdnnp_config.CPU_USAGE_RATIO
    if not (0.0 < ratio <= 1.0):
        ratio = 0.8
    total_cores = os.cpu_count()
    num_threads = max(1, math.floor(total_cores * ratio))
    torch.set_num_threads(num_threads)
except Exception as e:
    log(f"Could not set PyTorch thread count automatically. Error: {e}", level="WARN")
    
from hdnnp.data_loader import get_dataloader, HDNPDataset
from hdnnp.model import HDNNPModel
from hdnnp.symmetry_calculator import SymmetryCalculator
from hdnnp.loss import combined_loss

PROCESSED_DIR = hdnnp_config.PROCESSED_DATA_DIR
TRAIN_VALID_DIR = PROCESSED_DIR / "train_valid"
TEST_DIR = PROCESSED_DIR / "test"
SPLITS_JSON = PROCESSED_DIR / hdnnp_config.SPLITS_FILENAME

def update_config_from_json(config_path: Path):
    """JSONファイルから設定を読み込み、configモジュールを上書きする"""
    if not config_path.exists():
        log(f"Configuration file not found at {config_path}. Using default config.", level="WARN")
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        log(f"Loading and applying settings from {config_path}")
        for key, value in saved_config.items():
            if hasattr(hdnnp_config, key):
                if isinstance(getattr(hdnnp_config, key), Path):
                    setattr(hdnnp_config, key, Path(value))
                else:
                    setattr(hdnnp_config, key, value)
        log("Settings applied successfully.")
    except Exception as e:
        log(f"Error loading config from {config_path}: {e}. Using default config.", level="ERROR")

def set_random_seed(seed: int):
    """乱数シードを設定して再現性を確保します。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"Random seed set to {seed} for reproducibility.")

def infer_and_evaluate_optimized(
    model_eval: HDNNPModel, 
    dataloader_eval: torch.utils.data.DataLoader, 
    sym_calc_eval: SymmetryCalculator, 
    device_eval: torch.device,
    g_min: Optional[torch.Tensor],
    g_max: Optional[torch.Tensor],
    data_split_name: str = "Test",
    force_training_enabled: bool = True
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    最適化された推論・評価関数。
    """
    model_eval.eval()
    log(f"Starting optimized inference and evaluation for {data_split_name} set...")
    start_time = time.time()

    extrapolation_detected_summary = False
    if g_min is not None and g_max is not None:
        g_min = g_min.to(device_eval)
        g_max = g_max.to(device_eval)
        log("Extrapolation check enabled.")
    else:
        log("Extrapolation check disabled (sf_range.pt not found or not provided).", level="WARN")

    max_n_atoms_global = 0
    if isinstance(dataloader_eval.dataset, HDNPDataset):
        try:
            for i in range(len(dataloader_eval.dataset)):
                n_atoms_item = dataloader_eval.dataset[i]['N_atoms'].item()
                if n_atoms_item > max_n_atoms_global:
                    max_n_atoms_global = n_atoms_item
            if max_n_atoms_global > 0:
                log(f"Global maximum number of atoms for '{data_split_name}' set is {max_n_atoms_global}", level="DEBUG")
        except Exception as e:
            log(f"Could not determine global max N_atoms beforehand: {e}.", level="WARN")

    all_E_true_total, all_E_pred_total, all_F_true_padded, all_F_pred_padded = [], [], [], []
    all_N_atoms, all_atom_masks = [], []
    
    for batch_idx, data_batch in enumerate(dataloader_eval):
        R_batch = data_batch['R'].to(device_eval)
        Z_batch = data_batch['Z'].to(device_eval)
        cell_batch = data_batch['cell'].to(device_eval)
        E_true_total = data_batch['E'].squeeze(-1).to(device_eval)
        F_true_batch = data_batch['F'].to(device_eval)
        N_atoms_batch = data_batch['N_atoms'].to(device_eval).float()
        atom_mask_batch = data_batch['atom_mask'].to(device_eval)
        B, N_max_batch, _ = R_batch.shape
        if B == 0: continue
        
        R_batch.requires_grad_(force_training_enabled)
        
        E_pred_total_list = []
        with torch.no_grad() if not force_training_enabled else torch.enable_grad():
            for i in range(B):
                n_atoms_i = N_atoms_batch[i].int().item()
                if n_atoms_i == 0:
                    E_pred_total_list.append(torch.tensor(0.0, device=device_eval, dtype=R_batch.dtype))
                    continue
                Gi = sym_calc_eval.compute(R_batch[i, :n_atoms_i], Z_batch[i, :n_atoms_i], cell_batch[i])
                
                # ▼▼▼ 修正箇所: 外挿検知ログをさらに詳細化 ▼▼▼
                if g_min is not None and g_max is not None:
                    is_outside = (Gi < g_min) | (Gi > g_max)
                    if torch.any(is_outside):
                        extrapolation_detected_summary = True
                        file_stem = dataloader_eval.dataset.file_list[i]
                        filename = f"{file_stem}.npz"
                        
                        outlier_atom_indices = torch.where(is_outside.any(dim=1))[0]
                        for atom_idx in outlier_atom_indices:
                            z_val = Z_batch[i, atom_idx].item()
                            species = sym_calc_eval.z2element_map.get(int(z_val), f"Z={z_val}")
                            
                            # 外挿を引き起こしたSF成分のインデックスを取得
                            outlier_sf_indices = torch.where(is_outside[atom_idx])[0]
                            for sf_idx_tensor in outlier_sf_indices:
                                sf_idx = sf_idx_tensor.item()
                                # SFマップからパラメータ情報を取得
                                sf_params = sym_calc_eval.sf_index_map[sf_idx]
                                # 詳細なログを出力
                                log(f"Extrapolation in '{filename}', atom_idx={atom_idx.item()} ({species}), SF_idx={sf_idx} [{sf_params}]", level="WARN")
                # ▲▲▲ 修正ここまで ▲▲▲

                Ei_atoms_pred = model_eval(Gi, Z_batch[i, :n_atoms_i])
                E_pred_total_list.append(Ei_atoms_pred.sum())
            E_pred_total_for_grad = torch.stack(E_pred_total_list)

        F_pred_batch = torch.full_like(R_batch, float('nan'))
        if force_training_enabled:
            total_energy_for_grad = E_pred_total_for_grad.sum()
            forces_grad_tuple = torch.autograd.grad(
                outputs=total_energy_for_grad, inputs=R_batch,
                grad_outputs=torch.ones_like(total_energy_for_grad),
                create_graph=False, retain_graph=False, allow_unused=True
            )
            if forces_grad_tuple[0] is not None:
                F_pred_batch = -forces_grad_tuple[0]
        
        E_pred_total = E_pred_total_for_grad.detach()

        all_F_true_padded.append(F_true_batch)
        all_F_pred_padded.append(F_pred_batch.detach())
        all_atom_masks.append(atom_mask_batch)
        all_E_true_total.append(E_true_total)
        all_E_pred_total.append(E_pred_total)
        all_N_atoms.append(N_atoms_batch)

    if not all_E_true_total:
        return {}, {}
        
    if extrapolation_detected_summary:
        log("Summary: Extrapolation was detected in at least one structure.", level="WARN")

    E_true_all, E_pred_all = torch.cat(all_E_true_total), torch.cat(all_E_pred_total)
    F_true_all, F_pred_all = torch.cat(all_F_true_padded), torch.cat(all_F_pred_padded)
    N_atoms_all, atom_mask_all = torch.cat(all_N_atoms), torch.cat(all_atom_masks)

    with torch.no_grad():
        energy_loss = torch.mean((E_pred_all - E_true_all) ** 2)
        force_loss, rmse_F_components = torch.tensor(float('nan')), torch.tensor(float('nan'))
        
        if force_training_enabled:
            force_loss = torch.mean((F_pred_all[atom_mask_all] - F_true_all[atom_mask_all]) ** 2)
            num_force_components = atom_mask_all.sum() * 3
            if num_force_components > 0:
                rmse_F_components = torch.sqrt(((F_pred_all - F_true_all)**2 * atom_mask_all.unsqueeze(-1)).sum() / num_force_components)

        total_loss = hdnnp_config.LOSS_ALPHA_E * energy_loss
        if force_training_enabled:
            total_loss += hdnnp_config.LOSS_BETA_F * force_loss

        valid_samples_mask = N_atoms_all > 0
        E_true_pa, E_pred_pa, rmse_E_pa = torch.zeros(0), torch.zeros(0), torch.tensor(float('nan'))
        if valid_samples_mask.any():
            E_true_pa = E_true_all[valid_samples_mask] / N_atoms_all[valid_samples_mask]
            E_pred_pa = E_pred_all[valid_samples_mask] / N_atoms_all[valid_samples_mask]
            rmse_E_pa = torch.sqrt(torch.mean((E_pred_pa - E_true_pa) ** 2))

    log(f"Inference for {data_split_name} completed in {time.time() - start_time:.2f}s")
    
    metrics = {
        'Total Loss': total_loss.item(), 'Energy Loss': energy_loss.item(), 'Force Loss': force_loss.item(),
        'Energy RMSE (eV/atom)': rmse_E_pa.item(), 'Force RMSE (eV/A)': rmse_F_components.item(),
    }
    plot_data = {
        'E_true_pa': E_true_pa.cpu().numpy(), 'E_pred_pa': E_pred_pa.cpu().numpy(),
        'F_true_flat': F_true_all[atom_mask_all].cpu().numpy().flatten() if force_training_enabled else np.array([]),
        'F_pred_flat': F_pred_all[atom_mask_all].cpu().numpy().flatten() if force_training_enabled else np.array([]),
    }
    return metrics, plot_data

def save_summary_table(results: Dict[str, Dict[str, float]], output_path: Path):
    header = f"| {'Data Set':<12} | {'Total Loss':<15} | {'Energy RMSE/atom':<20} | {'Force RMSE/comp':<20} |"
    separator = "-" * len(header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Inference Results Summary\n=========================\n\n")
        f.write(separator + "\n" + header + "\n" + separator + "\n")
        for name, metrics in results.items():
            line = (f"| {name:<12} | {metrics.get('Total Loss', float('nan')):<15.6e} | "
                    f"{metrics.get('Energy RMSE (eV/atom)', float('nan')):<20.6e} | "
                    f"{metrics.get('Force RMSE (eV/A)', float('nan')):<20.6e} |")
            f.write(line + "\n")
        f.write(separator + "\n")
    log(f"Inference summary table saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="HDNNP Inference Script")
    parser.add_argument('--CHECKPOINT_DIR', type=Path, required=True, 
                        help="Path to the checkpoint directory containing the trained model and config.")
    args = parser.parse_args()
    
    CHECKPOINT_DIR = args.CHECKPOINT_DIR
    if not CHECKPOINT_DIR.is_dir():
        log(f"Checkpoint directory not found at: {CHECKPOINT_DIR}", level="ERROR")
        sys.exit(1)

    config_path = CHECKPOINT_DIR / "config_used.json"
    update_config_from_json(config_path)

    LOG_DIR = CHECKPOINT_DIR / "log"
    TRAINED_MODEL_PATH = CHECKPOINT_DIR / hdnnp_config.TRAINED_MODEL_FILENAME
    SF_RANGE_PATH = CHECKPOINT_DIR / hdnnp_config.SF_RANGE_FILENAME

    if hdnnp_config.SET_RANDOM_SEED:
        set_random_seed(hdnnp_config.RANDOM_SEED_VALUE)

    if hdnnp_config.DEVICE == "auto":
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(hdnnp_config.DEVICE)
    log(f"Using device: {DEVICE}")

    log("Initializing SymmetryCalculator...")
    sym_calc = SymmetryCalculator()
    INPUT_DIM = sym_calc.total_sf_dim
    log(f"SymmetryCalculator initialized. Input dimension: {INPUT_DIM}")
    
    log(f"Loading splits data from: {SPLITS_JSON}")
    if not SPLITS_JSON.exists(): sys.exit(log(f"[ERROR] splits.json not found at {SPLITS_JSON}", "ERROR"))
    with open(SPLITS_JSON, 'r') as f: splits = json.load(f)
    
    if not TRAINED_MODEL_PATH.exists(): sys.exit(log(f"[ERROR] Trained model not found at {TRAINED_MODEL_PATH}", "ERROR"))

    use_force = hdnnp_config.USE_FORCE_TRAINING
    log(f"Force evaluation is {'ENABLED' if use_force else 'DISABLED'} based on training config.")
    
    g_min, g_max = None, None
    if hdnnp_config.EXTRAPOLATION_CHECK_ENABLED:
        if SF_RANGE_PATH.exists():
            log(f"Loading symmetry function range from: {SF_RANGE_PATH}")
            try:
                sf_range_data = torch.load(SF_RANGE_PATH)
                g_min, g_max = sf_range_data['G_min'], sf_range_data['G_max']
                log("Symmetry function range loaded successfully.")
            except Exception as e:
                log(f"Could not load SF range file: {e}", level="ERROR")
    
    log(f"Loading trained model from: {TRAINED_MODEL_PATH}")
    model = HDNNPModel(INPUT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
    log("Trained model loaded successfully.")

    results_summary = {}

    if 'train' in splits and splits['train']:
        n_train = len(splits['train'])
        train_loader_infer = get_dataloader(
            splits_json=SPLITS_JSON, processed_dir=TRAIN_VALID_DIR, split='train',
            batch_size=n_train, shuffle=False,
            num_workers=hdnnp_config.DATALOADER_NUM_WORKERS, 
            pin_memory=(DEVICE.type == 'cuda' and hdnnp_config.DATALOADER_PIN_MEMORY))
        train_metrics, train_plot_data = infer_and_evaluate_optimized(
            model, train_loader_infer, sym_calc, DEVICE, g_min, g_max, "Train", use_force
        )
        results_summary['Train'] = train_metrics
        
        if train_plot_data and train_plot_data['E_true_pa'].size > 0:
            log("Generating plot for per-atom energy vs. structure number (Train Set)...")
            E_true_pa_train = train_plot_data['E_true_pa']
            E_pred_pa_train = train_plot_data['E_pred_pa']

            for i in range(len(splits['train'])):
                frame_name = splits['train'][i]
                train_per_ene_true = E_true_pa_train[i]
                train_per_ene_pred = E_pred_pa_train[i]
                print(f'{frame_name}のエネルギーの真値(1原子あたり): {train_per_ene_true}')
                print(f'{frame_name}のエネルギーの予測値(1原子あたり): {train_per_ene_pre}')

            plt.figure(figsize=(10, 6))
            plt.plot(E_true_pa_train, 'o', color='royalblue', markersize=5, label='True Energy per Atom (Train)')
            plt.plot(E_pred_pa_train, 's', color='darkorange', markersize=5, alpha=0.7, label='Predicted Energy per Atom (Train)')
            plt.xlabel('Structure Index'); plt.ylabel('Energy per Atom (eV/atom)'); plt.title('Train Set: Per-atom Energy vs. Structure Index')
            plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
            plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_ENERGY_VS_STRUCT_TRAIN_INFER_FILENAME
            plt.savefig(plot_path, dpi=hdnnp_config.PLOT_DPI); plt.close()
            log(f'Plot saved to: {plot_path}')

    if 'test' in splits and splits['test']:
        n_test = len(splits['test'])
        test_loader = get_dataloader(
            splits_json=SPLITS_JSON, processed_dir=TEST_DIR, split='test',
            batch_size=n_test, shuffle=False,
            num_workers=hdnnp_config.DATALOADER_NUM_WORKERS, 
            pin_memory=(DEVICE.type == 'cuda' and hdnnp_config.DATALOADER_PIN_MEMORY))
        test_metrics, test_plot_data = infer_and_evaluate_optimized(
            model, test_loader, sym_calc, DEVICE, g_min, g_max, "Test", use_force
        )
        results_summary['Test'] = test_metrics

        if test_plot_data:
            E_true_pa, E_pred_pa = test_plot_data['E_true_pa'], test_plot_data['E_pred_pa']
            F_true_flat, F_pred_flat = test_plot_data['F_true_flat'], test_plot_data['F_pred_flat']

            if E_true_pa.size > 0:
                log("Generating plot for per-atom energy vs. structure number (Test Set)...")
                plt.figure(figsize=(10, 6))
                plt.plot(E_true_pa, 'o', color='mediumseagreen', markersize=5, label='True Energy per Atom (Test)')
                plt.plot(E_pred_pa, 's', color='tomato', markersize=5, alpha=0.7, label='Predicted Energy per Atom (Test)')
                plt.xlabel('Structure Index'); plt.ylabel('Energy per Atom (eV/atom)'); plt.title('Test Set: Per-atom Energy vs. Structure Index')
                plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
                plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_ENERGY_VS_STRUCT_TEST_INFER_FILENAME
                plt.savefig(plot_path, dpi=hdnnp_config.PLOT_DPI); plt.close()
                log(f'Plot saved to: {plot_path}')

                log("Generating scatter plot for per-atom energy (Test Set)...")
                plt.figure(figsize=(8, 8))
                plt.scatter(E_true_pa, E_pred_pa, alpha=0.5, s=20)
                min_val, max_val = min(E_true_pa.min(), E_pred_pa.min()), max(E_true_pa.max(), E_pred_pa.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="Ideal (y=x)")
                plt.xlabel('True Energy per Atom (eV/atom)'); plt.ylabel('Predicted Energy per Atom (eV/atom)')
                plt.title('Test Set: Per-atom Energy True vs. Predicted'); plt.grid(True, alpha=0.5); plt.legend(); plt.tight_layout()
                plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_ENERGY_SCATTER_TEST_INFER_FILENAME
                plt.savefig(plot_path, dpi=hdnnp_config.PLOT_DPI); plt.close()
                log(f'Plot saved to: {plot_path}')

                for i in range(len(splits['test'])):
                    frame_name = splits['test'][i]
                    test_per_ene_true = E_true_pa[i]
                    test_per_ene_pred = E_pred_pa[i]
                    print(f'{frame_name}のエネルギーの真値(1原子あたり): {test_per_ene_true}')
                    print(f'{frame_name}のエネルギーの予測値(1原子あたり): {test_per_ene_pred}')

            if use_force and F_true_flat.size > 0:
                log("Generating scatter plot for force components (Test Set)...")
                plt.figure(figsize=(8, 8))
                plt.scatter(F_true_flat, F_pred_flat, alpha=0.1, s=10, color='forestgreen')
                min_val, max_val = min(F_true_flat.min(), F_pred_flat.min()), max(F_true_flat.max(), F_pred_flat.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)
                plt.xlabel('True Force Components (eV/Å)'); plt.ylabel('Predicted Force Components (eV/Å)')
                plt.title('Test Set: All Force Components True vs. Predicted'); plt.grid(True, alpha=0.5); plt.tight_layout()
                plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_FORCE_COMPONENTS_SCATTER_TEST_INFER_FILENAME
                plt.savefig(plot_path, dpi=hdnnp_config.PLOT_DPI); plt.close()
                log(f'Plot saved to: {plot_path}')
            
                log("Generating histogram for force errors (Test Set)...")
                force_errors = F_pred_flat - F_true_flat
                plt.figure(figsize=(8, 6))
                plt.hist(force_errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
                plt.xlabel('Force Error (Predicted - True) (eV/Å)'); plt.ylabel('Frequency')
                plt.title('Test Set: Histogram of Force Component Errors'); plt.grid(True, alpha=0.5)
                mean_error = np.mean(force_errors)
                plt.axvline(mean_error, color='k', linestyle='dashed', linewidth=1, label=f'Mean Error: {mean_error:.3f}')
                plt.legend(); plt.tight_layout()
                plot_path = CHECKPOINT_DIR / hdnnp_config.PLOT_FORCE_ERROR_HISTOGRAM_TEST_INFER_FILENAME
                plt.savefig(plot_path, dpi=hdnnp_config.PLOT_DPI); plt.close()
                log(f'Plot saved to: {plot_path}')

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOG_DIR / "inference_summary.txt"
    save_summary_table(results_summary, summary_path)

if __name__ == '__main__':
    main()