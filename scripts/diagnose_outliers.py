#!/usr/bin/env python3
"""
scripts/diagnose_outliers.py

PCA空間で訓練データ分布から外れたテストデータ原子を特定し、
その原子が含まれる構造ファイルと原子インデックスを出力する。
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
import pandas as pd

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
    # ... (この関数は変更なし) ...
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
        log("Settings applied successfully for analysis.")
    except Exception as e:
        log(f"Error loading config from {config_path}: {e}. Using default config.", level="ERROR")

def get_all_descriptors_with_ids(dataloader, sym_calc, device, z2element):
    """記述子と共に、それがどのファイルの何番目の原子かを記録する"""
    all_descriptors, all_species, all_ids = [], [], []
    
    file_list = dataloader.dataset.file_list
    log(f"Calculating descriptors for {len(file_list)} structures...")
    
    with torch.no_grad():
        for i, file_stem in enumerate(file_list):
            try:
                data = dataloader.dataset[i]
            except Exception as e:
                log(f"Could not load data for {file_stem}.npz. Skipping. Error: {e}", level="WARN")
                continue

            R, Z, cell, n_atoms = \
                data['R'].to(device), data['Z'].to(device), data['cell'].to(device), data['N_atoms'].item()
            if n_atoms == 0: continue

            descriptors = sym_calc.compute(R, Z, cell)
            all_descriptors.append(descriptors.cpu().numpy())
            
            for atom_idx_in_structure, z_val in enumerate(Z):
                species = z2element.get(int(z_val.item()), 'Unknown')
                all_species.append(species)
                all_ids.append({'file': f"{file_stem}.npz", 'atom_index': atom_idx_in_structure})

    if not all_descriptors:
        return np.array([]), [], []
        
    return np.concatenate(all_descriptors, axis=0), all_species, all_ids

def main():
    parser = argparse.ArgumentParser(description="Diagnose outlier atoms in descriptor space.")
    parser.add_argument('--checkpoint-dir', '-c', type=Path, required=True,
                        help="Path to the experiment's checkpoint directory.")
    # ▼▼▼ 外れ値の定義に使えそうな閾値を追加 ▼▼▼
    parser.add_argument('--pc1-threshold', type=float, default=2.5,
                        help="PC1 value threshold to identify the isolated Pt cluster.")
    args = parser.parse_args()

    # ... (環境設定、config読み込みは同じ) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_path = args.checkpoint_dir / "config_used.json"
    update_config_from_json(config_path)

    sym_calc = SymmetryCalculator()
    
    # ... (データローダー準備は同じ) ...
    splits_json_path = hdnnp_config.PROCESSED_DATA_DIR / hdnnp_config.SPLITS_FILENAME
    processed_dir_path = hdnnp_config.PROCESSED_DATA_DIR
    train_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='train',
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='test',
        batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # --- 記述子とIDを計算 ---
    train_descriptors, train_species, train_ids = get_all_descriptors_with_ids(train_loader, sym_calc, device, sym_calc.z2element_map)
    test_descriptors, test_species, test_ids = get_all_descriptors_with_ids(test_loader, sym_calc, device, sym_calc.z2element_map)
    
    if test_descriptors.size == 0:
        log("No test descriptors were generated. Exiting.", level="ERROR")
        return

    # --- PCAとデータフレーム化 ---
    log("Performing PCA and creating DataFrame...")
    all_descriptors = np.vstack([train_descriptors, test_descriptors])
    scaler = StandardScaler()
    all_descriptors_scaled = scaler.fit_transform(all_descriptors)
    pca = PCA(n_components=2)
    descriptors_pca = pca.fit_transform(all_descriptors_scaled)
    
    # テストデータのみに絞る
    test_pca = descriptors_pca[len(train_descriptors):]
    
    # 結果をPandas DataFrameにまとめる
    df_test = pd.DataFrame(test_ids)
    df_test['species'] = test_species
    df_test['PC1'] = test_pca[:, 0]
    df_test['PC2'] = test_pca[:, 1]

    # --- 外れ値の特定 ---
    log("Identifying outlier atoms from the test set...")
    
    # PCAプロットから、孤立しているPt原子はPC1が約2.5より大きい領域にあると判断
    outlier_condition = (df_test['species'] == 'Pt') & (df_test['PC1'] > args.pc1_threshold)
    df_outliers = df_test[outlier_condition]

    if df_outliers.empty:
        log("No outlier Pt atoms found based on the current threshold.")
    else:
        log(f"Found {len(df_outliers)} outlier Pt atoms. Saving details to outliers.csv")
        # 結果をCSVファイルに保存
        output_csv_path = args.checkpoint_dir / "outliers.csv"
        df_outliers.to_csv(output_csv_path, index=False)
        
        print("\n--- Outlier Atoms Summary ---")
        # ファイルごとに集計して表示
        outlier_summary = df_outliers.groupby('file')['atom_index'].apply(list).reset_index()
        print(outlier_summary.to_string())
        print("---------------------------\n")
        log(f"A detailed list of outlier atoms has been saved to: {output_csv_path}")

if __name__ == '__main__':
    main()