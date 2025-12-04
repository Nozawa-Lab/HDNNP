#!/usr/bin/env python3
"""
scripts/make_splits.py

Generate splits.json by scanning processed .npz files automatically.

Usage:
    python scripts/make_splits.py [-p <processed_dir>] [-s <seed>] [--train-fraction <frac>] [--valid-fraction <frac>]
"""
import argparse
import json
import random
from pathlib import Path
import sys

# hdnnp パッケージの config をインポート

# プロジェクトルートを解決し、sys.path に追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

    
try:
    from hdnnp import config as hdnnp_config
except ImportError:
    # スクリプトが hdnnp パッケージの外から直接実行される場合のためのフォールバック
    # この場合、hdnnp_config の代わりにデフォルト値を直接使うか、
    # もしくはスクリプトの親ディレクトリを sys.path に追加する必要がある。
    # ここでは、configが読み込めない場合はエラー終了とする。
    print("Error: Could not import hdnnp.config. Ensure the script is run from the project root or hdnnp is in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Generate train/valid/test splits from processed .npz files'
    )
    parser.add_argument(
        '--processed-dir', '-p',
        type=Path,
        default=hdnnp_config.PROCESSED_DATA_DIR, # config からデフォルト値を取得
        help=f'Directory containing processed .npz files (default from config: {hdnnp_config.PROCESSED_DATA_DIR})'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=hdnnp_config.SPLIT_SEED, # config からデフォルト値を取得
        help=f'Random seed for reproducible shuffling (default from config: {hdnnp_config.SPLIT_SEED})'
    )
    parser.add_argument(
        '--train-fraction',
        type=float,
        default=hdnnp_config.SPLIT_TRAIN_FRACTION, # config からデフォルト値を取得
        help=f'Fraction of samples for training set (default from config: {hdnnp_config.SPLIT_TRAIN_FRACTION})'
    )
    parser.add_argument(
        '--valid-fraction',
        type=float,
        default=hdnnp_config.SPLIT_VALID_FRACTION, # config からデフォルト値を取得
        help=f'Fraction of samples for validation set (default from config: {hdnnp_config.SPLIT_VALID_FRACTION})'
    )
    parser.add_argument(
        '--splits-filename',
        type=str,
        default=hdnnp_config.SPLITS_FILENAME, # config からデフォルト値を取得
        help=f'Output filename for the splits JSON (default from config: {hdnnp_config.SPLITS_FILENAME})'
    )
    args = parser.parse_args()

    if not args.processed_dir.is_dir():
        print(f"Error: Processed directory not found: {args.processed_dir}", file=sys.stderr)
        sys.exit(1)
        
    # Scan processed directory for .npz files
    files = sorted([p.stem for p in args.processed_dir.glob('*.npz')])
    if not files:
        # raise RuntimeError(f'No .npz files found in {args.processed_dir}')
        print(f"Warning: No .npz files found in {args.processed_dir}. Generating empty splits.json.", file=sys.stderr)
        # 空の splits を作成して終了する
        splits = {'train': [], 'valid': [], 'test': []}
    else:
        # Shuffle with fixed seed
        random.seed(args.seed)
        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_fraction)
        n_valid = int(n * args.valid_fraction)

        if n_train + n_valid > n:
            print(f"Error: train_fraction ({args.train_fraction}) + valid_fraction ({args.valid_fraction}) > 1.0. Cannot create splits.", file=sys.stderr)
            # n_valid を調整するなどの処理も可能だが、ここではエラーとする
            n_valid = n - n_train
            if n_valid < 0: n_valid = 0
            print(f"Adjusting valid samples to {n_valid} to fit total samples {n}.", file=sys.stderr)

        # ▼▼▼ 変更箇所 ▼▼▼
        # 各リストを分割した後に、sorted() を使ってアルファベット順にソートする
        splits = {
            'train': sorted(files[:n_train]),
            'valid': sorted(files[n_train : n_train + n_valid]),
            'test':  sorted(files[n_train + n_valid:]),
        }
        # ▲▲▲ 変更ここまで ▲▲▲

    # Save splits.json
    out_path = args.processed_dir / args.splits_filename # config から取得したファイル名を使用
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(splits, f, indent=2, ensure_ascii=False)
        print(f"splits.json generated at: {out_path}")
        print(f" Counts -> train: {len(splits['train'])}, valid: {len(splits['valid'])}, test: {len(splits['test'])}")
    except IOError as e:
        print(f"Error: Could not write splits file to {out_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()