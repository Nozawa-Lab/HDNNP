#!/usr/bin/env python3
"""
scripts/analyze_descriptors.py

訓練データとテストデータの記述子（対称性関数）を計算し、
主成分分析（PCA）を用いて2次元に削減・可視化することで、
記述子空間における分布の差異を分析するスクリプト。
"""
import sys
import json
from pathlib import Path
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- プロジェクトルートを解決し、sys.pathに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hdnnp.data_loader import get_dataloader
from hdnnp.symmetry_calculator import SymmetryCalculator
from hdnnp import config as hdnnp_config

# --- ANSIカラーコード ---
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"

def log(message, level="INFO"):
    colors = {"INFO": GREEN, "DEBUG": CYAN, "WARN": YELLOW, "ERROR": RED}
    print(f"{colors.get(level, RESET)}[{level}]{RESET} {message}")

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
                # Pathオブジェクトなどを適切に変換
                if isinstance(getattr(hdnnp_config, key), Path):
                    setattr(hdnnp_config, key, Path(value))
                else:
                    setattr(hdnnp_config, key, value)
        log("Settings applied successfully for analysis.")
    except Exception as e:
        log(f"Error loading config from {config_path}: {e}. Using default config.", level="ERROR")


def get_all_descriptors(dataloader, sym_calc, device, z2element):
    """データローダーから全構造の記述子と原子種ラベルを取得する"""
    all_descriptors = []
    all_species_labels = []
    log(f"Calculating descriptors for {len(dataloader.dataset)} structures...")
    with torch.no_grad():
        for data in dataloader:
            R_batch, Z_batch, cell_batch, N_atoms_batch = \
                data['R'].to(device), data['Z'].to(device), data['cell'].to(device), data['N_atoms'].to(device)
            
            for i in range(len(R_batch)):
                n_atoms = N_atoms_batch[i].item()
                if n_atoms == 0: continue
                
                R, Z, cell = R_batch[i, :n_atoms], Z_batch[i, :n_atoms], cell_batch[i]
                descriptors = sym_calc.compute(R, Z, cell)
                all_descriptors.append(descriptors.cpu().numpy())
                
                species = [z2element.get(int(z.item()), 'Unknown') for z in Z] # int()でキャストを追加
                all_species_labels.extend(species)

    if not all_descriptors:
        return np.array([]), []
        
    return np.concatenate(all_descriptors, axis=0), all_species_labels


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize descriptor space using PCA.")
    parser.add_argument('--checkpoint-dir', '-c', type=Path, required=True,
                        help="Path to the experiment's checkpoint directory (containing config_used.json).")
    parser.add_argument('--output-file', '-o', type=Path,
                        help="Path to save the output PCA plot (default: <checkpoint-dir>/descriptor_pca.png).")
    args = parser.parse_args()

    # 出力ファイルパスが指定されていない場合、checkpoint-dir内にデフォルト名で保存
    if args.output_file is None:
        args.output_file = args.checkpoint_dir / "descriptor_pca.png"

    # --- 環境設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")
    
    # --- config読み込み処理を追加 ---
    config_path = args.checkpoint_dir / "config_used.json"
    update_config_from_json(config_path)

    # --- SymmetryCalculatorの初期化 ---
    # (configが更新された後に初期化することで、実験固有のパラメータが使われる)
    sym_calc = SymmetryCalculator()
    log("SymmetryCalculator initialized.")
    
    # --- データローダーの準備 ---
    log("Loading train and test datasets...")
    splits_json_path = hdnnp_config.PROCESSED_DATA_DIR / hdnnp_config.SPLITS_FILENAME
    processed_dir_path = hdnnp_config.PROCESSED_DATA_DIR
    
    train_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='train',
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='test',
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # --- 記述子の計算 ---
    train_descriptors, train_species = get_all_descriptors(train_loader, sym_calc, device, sym_calc.z2element_map)
    test_descriptors, test_species = get_all_descriptors(test_loader, sym_calc, device, sym_calc.z2element_map)
    
    if train_descriptors.size == 0 or test_descriptors.size == 0:
        log("No descriptors were generated. Exiting.", level="ERROR")
        return

    log(f"Train descriptors shape: {train_descriptors.shape}")
    log(f"Test descriptors shape: {test_descriptors.shape}")

    # --- PCAの実行 ---
    log("Performing PCA...")
    # データを結合し、標準化
    all_descriptors = np.vstack([train_descriptors, test_descriptors])
    scaler = StandardScaler()
    all_descriptors_scaled = scaler.fit_transform(all_descriptors)
    
    # PCAで2次元に削減
    pca = PCA(n_components=2)
    descriptors_pca = pca.fit_transform(all_descriptors_scaled)
    
    # 訓練データとテストデータに再分割
    train_pca = descriptors_pca[:len(train_descriptors)]
    test_pca = descriptors_pca[len(train_descriptors):]
    
    log(f"Explained variance ratio by PC1 and PC2: {pca.explained_variance_ratio_}")

    # --- 可視化 ---
    log("Generating PCA plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))

    species_colors = {'Al': 'royalblue', 'Fe': 'darkorange', 'Pt': 'mediumseagreen'}
    
    # 訓練データをプロット (半透明)
    for sp in hdnnp_config.SPECIES:
        indices = [i for i, s in enumerate(train_species) if s == sp]
        if indices:
            ax.scatter(train_pca[indices, 0], train_pca[indices, 1], 
                       label=f'Train-{sp}', alpha=0.3, s=20, 
                       color=species_colors.get(sp, 'gray'))

    # テストデータをプロット (不透明、大きめのマーカー)
    for sp in hdnnp_config.SPECIES:
        indices = [i for i, s in enumerate(test_species) if s == sp]
        if indices:
            ax.scatter(test_pca[indices, 0], test_pca[indices, 1], 
                       label=f'Test-{sp}', alpha=0.9, s=50, marker='x',
                       color=species_colors.get(sp, 'black'))

    ax.set_xlabel('Principal Component 1', fontsize=14)
    ax.set_ylabel('Principal Component 2', fontsize=14)
    ax.set_title(f'PCA of Descriptor Space ({args.checkpoint_dir.name})', fontsize=16, fontweight='bold')
    ax.legend(title='Dataset-Species', markerscale=1.5, fontsize=12)
    fig.tight_layout()
    
    plt.savefig(args.output_file, dpi=300)
    log(f"PCA plot saved to: {args.output_file}")

if __name__ == '__main__':
    main()