#!/usr/bin/env python3
"""
scripts/analyze_environment.py

outliers.csv を読み込み、外れ値原子の局所環境（配位数、原子間距離）を計算し、
訓練データの代表的な原子の環境と比較・出力することで、
不足している訓練データの具体的な特徴を明らかにする。
レポートはファイルにも保存される。
"""
import sys
import json
from pathlib import Path
import torch
import numpy as np
import argparse
import pandas as pd
import random
from itertools import product

# --- プロジェクトルートを解決し、sys.pathに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

def get_local_environment(R, Z, cell, target_index, r_cut, z2element):
    """
    指定された原子の局所環境（配位数、距離）を計算する。
    周期境界条件を考慮し、隣接セル内の原子も探索対象に含める。
    """
    target_pos = R[target_index]
    num_atoms = R.shape[0]

    neighbors = {}
    neighbor_distances = []

    # 3x3x3のスーパーセルを作成するため、-1, 0, 1のシフトを生成
    for i, j, k in product([-1, 0, 1], repeat=3):
        # (0,0,0)は自分自身のセルなので、自分自身との距離は除外する
        if i == 0 and j == 0 and k == 0:
            # セル内の他の原子との距離を計算
            diff = R - target_pos.reshape(1, 3)
            distances = np.linalg.norm(diff, axis=1)
            for atom_idx in range(num_atoms):
                if atom_idx == target_index:
                    continue
                if distances[atom_idx] < r_cut:
                    neighbor_z = int(Z[atom_idx])
                    neighbor_symbol = z2element.get(neighbor_z, f"Z={neighbor_z}")
                    neighbors[neighbor_symbol] = neighbors.get(neighbor_symbol, 0) + 1
                    neighbor_distances.append((neighbor_symbol, distances[atom_idx]))
        else:
            # 隣接セルの原子位置を計算
            shift_vector = i * cell[0] + j * cell[1] + k * cell[2]
            shifted_R = R + shift_vector
            
            # 対象原子と隣接セル内の全原子との距離を計算
            diff = shifted_R - target_pos.reshape(1, 3)
            distances = np.linalg.norm(diff, axis=1)

            for atom_idx in range(num_atoms):
                if distances[atom_idx] < r_cut:
                    neighbor_z = int(Z[atom_idx])
                    neighbor_symbol = z2element.get(neighbor_z, f"Z={neighbor_z}")
                    neighbors[neighbor_symbol] = neighbors.get(neighbor_symbol, 0) + 1
                    neighbor_distances.append((neighbor_symbol, distances[atom_idx]))

    # 距離でソート
    neighbor_distances.sort(key=lambda x: x[1])
    
    return neighbors, neighbor_distances


def print_environment_report(title, env_data, z2element, file_handle=None):
    """計算された局所環境レポートをコンソールとファイルに出力する"""
    # 色付けを除いたプレーンテキストをファイルに書き出すためのヘルパー関数
    def write_to_file(message):
        if file_handle:
            file_handle.write(message + '\n')

    # コンソール出力用の色付きタイトルとファイル出力用のプレーンタイトル
    console_title = f"\n--- {BOLD}{title}{RESET} ---"
    file_title = f"\n--- {title} ---"
    print(console_title)
    write_to_file(file_title)

    if not env_data:
        message = "  (No data available)"
        print(message)
        write_to_file(message)
        return
        
    file = env_data.get('file', 'N/A')
    atom_index = env_data.get('atom_index', 'N/A')
    species = env_data.get('species', 'N/A')
    
    # 各行のメッセージを作成
    messages = [
        f"  File         : {file}",
        f"  Atom         : Index {atom_index} ({species})"
    ]

    neighbors, neighbor_distances = env_data['analysis']
    
    coord_str = ", ".join([f"{symbol}: {count}" for symbol, count in sorted(neighbors.items())])
    total_coord = sum(neighbors.values())
    messages.append(f"  Coordination : {total_coord} ({coord_str})")
    
    messages.append("  Distances (Top 5 nearest):")
    if not neighbor_distances:
        messages.append("    - No neighbors found within cutoff.")
    for symbol, dist in neighbor_distances[:5]:
        messages.append(f"    - {species}-{symbol:<2s} : {dist:.3f} Å")
    
    messages.append("-" * (len(title) + 8))

    # 作成したメッセージをコンソールとファイルに出力
    for msg in messages:
        print(msg)
        write_to_file(msg)


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare local environments of all outlier atoms.")
    parser.add_argument('--checkpoint-dir', '-c', type=Path, required=True,
                        help="Path to the experiment's checkpoint directory containing outliers.csv.")
    args = parser.parse_args()

    outliers_csv_path = args.checkpoint_dir / "outliers.csv"
    if not outliers_csv_path.exists():
        log(f"Error: outliers.csv not found in '{args.checkpoint_dir}'.", level="ERROR")
        log("Please run diagnose_outliers.py first.", level="INFO")
        sys.exit(1)
        
    # ▼▼▼ レポートファイルのパスを定義 ▼▼▼
    report_file_path = args.checkpoint_dir / "environment_analysis_report.txt"

    # --- configから設定を読み込む ---
    config_path = args.checkpoint_dir / "config_used.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        R_CUT = config_data.get('R_CUT', 7.0)
        PROCESSED_DATA_DIR = Path(config_data.get('PROCESSED_DATA_DIR', './data/processed'))
        Z2ELEMENT = {int(k): v for k, v in config_data.get('Z2ELEMENT', {}).items()}
    else:
        log("Warning: config_used.json not found. Using default settings from hdnnp.config.", level="WARN")
        R_CUT = hdnnp_config.R_CUT
        PROCESSED_DATA_DIR = hdnnp_config.PROCESSED_DATA_DIR
        Z2ELEMENT = hdnnp_config.Z2ELEMENT

    log(f"Analyzing local environments with R_CUT = {R_CUT:.1f} Å")

    df_outliers = pd.read_csv(outliers_csv_path)
    
    splits_json_path = PROCESSED_DATA_DIR / "splits.json"
    if not splits_json_path.exists():
        log(f"Error: splits.json not found at '{splits_json_path}'.", level="ERROR")
        sys.exit(1)
    with open(splits_json_path, 'r') as f:
        splits_data = json.load(f)
    train_files = splits_data.get('train', [])

    log(f"Found {len(df_outliers)} outlier atoms in outliers.csv. Analyzing all of them.")
    
    # ▼▼▼ レポートファイルを開き、分析結果を書き込む ▼▼▼
    with open(report_file_path, 'w', encoding='utf-8') as report_f:
        report_f.write(f"Local Environment Analysis Report for Experiment: {args.checkpoint_dir.name}\n")
        report_f.write(f"Analyzed {len(df_outliers)} outlier atoms found in outliers.csv\n")
        
        for i, outlier in df_outliers.iterrows():
            outlier_file = PROCESSED_DATA_DIR / outlier['file']
            outlier_atom_index = outlier['atom_index']
            outlier_species = outlier['species']

            outlier_env_data = None
            if outlier_file.exists():
                data = np.load(outlier_file)
                R, Z, cell = data['R'], data['Z'], data['cell']
                analysis_result = get_local_environment(R, Z, cell, outlier_atom_index, R_CUT, Z2ELEMENT)
                outlier_env_data = {
                    'file': outlier['file'], 'atom_index': outlier_atom_index, 'species': outlier_species,
                    'analysis': analysis_result
                }
            else:
                log(f"Warning: Could not find npz file for outlier: {outlier_file}", level="WARN")

            typical_env_data = None
            random.shuffle(train_files)
            for train_file_stem in train_files:
                train_file_path = PROCESSED_DATA_DIR / f"{train_file_stem}.npz"
                if not train_file_path.exists(): continue
                
                data = np.load(train_file_path)
                R, Z, cell = data['R'], data['Z'], data['cell']
                
                indices_of_same_species = [idx for idx, z in enumerate(Z) if Z2ELEMENT.get(int(z)) == outlier_species]
                
                if indices_of_same_species:
                    typical_atom_index = random.choice(indices_of_same_species)
                    analysis_result = get_local_environment(R, Z, cell, typical_atom_index, R_CUT, Z2ELEMENT)
                    typical_env_data = {
                        'file': f"{train_file_stem}.npz", 'atom_index': typical_atom_index, 'species': outlier_species,
                        'analysis': analysis_result
                    }
                    break
            
            # レポート表示とファイル書き込み
            header_str = f"{'='*20} DIAGNOSIS REPORT {i+1}/{len(df_outliers)} {'='*20}"
            print(f"\n{header_str}")
            report_f.write(f"\n{header_str}\n")
            
            print_environment_report(f"Outlier Atom Environment (from Test Set)", outlier_env_data, Z2ELEMENT, file_handle=report_f)
            print_environment_report(f"Typical Atom Environment (from Train Set)", typical_env_data, Z2ELEMENT, file_handle=report_f)
            
            footer_str = "=" * len(header_str)
            print(footer_str)
            report_f.write(f"{footer_str}\n")
    
    log(f"Environment analysis report saved to: {report_file_path}")

if __name__ == '__main__':
    main()