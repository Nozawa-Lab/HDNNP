'''
hdnnp.data_loader

DatasetおよびDataLoader作成モジュール

- HDNPDataset: processed ディレクトリの npz/pkl ファイルから R, Z, cell, E, F を読み込み
- get_dataloader: splits.json と組み合わせて DataLoader を返す
'''
from pathlib import Path
import json
import sys
from typing import List, Dict, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# config は get_dataloader の呼び出し元で参照されるため、ここでは直接インポート不要


class HDNPDataset(Dataset):
    """
    HDNNP用データセット

    各サンプルは npz ファイルに保存されている R, Z, cell, E, F を読み込む。
    """
    def __init__(
        self,
        file_list: List[str],
        processed_dir: Path,
        # use_precomputed: bool = False # config.USE_PRECOMPUTED_FEATURES を参照するなら引数で渡す
    ) -> None:
        """
        Args:
            file_list (List[str]): サンプルファイル名のリスト（拡張子なし）
            processed_dir (Path): npz ファイルが置かれたディレクトリ
        """
        self.file_list = file_list
        self.processed_dir = processed_dir
        # self.use_precomputed = use_precomputed # config から渡す場合

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Returns:
            Dict[str, Tensor]: 辞書形式でデータを返す
                - 'R': 原子座標 (N,3)
                - 'Z': 原子番号 (N,)
                - 'cell': 格子ベクトル (3,3)
                - 'E': 全エネルギー (1,)
                - 'F': 原子ごとの力 (N,3)
                - 'N_atoms': 実際の原子数 (スカラーTensor)
        """
        npz_path = self.processed_dir / (self.file_list[idx] + ".npz")
        try:
            data = np.load(npz_path)
        except FileNotFoundError:
            print(f"[ERROR] File not found: {npz_path}", file=sys.stderr)
            # エラー処理: 空のデータや例外送出など、アプリケーションの要件に応じて変更
            # ここでは、後続で問題が起きにくいように、キーを持つが空の可能性のあるデータを返す試み
            # ただし、これは根本的な解決策ではない
            return {
                'R': torch.empty(0, 3, dtype=torch.float32), 
                'Z': torch.empty(0, dtype=torch.long),
                'cell': torch.eye(3, dtype=torch.float32) * 1.0, # ダミーのセル
                'E': torch.empty(0, dtype=torch.float32), # (0,) or (0,1)
                'F': torch.empty(0, 3, dtype=torch.float32),
                'N_atoms': torch.tensor(0, dtype=torch.long)
            }


        R = torch.from_numpy(data['R']).float()
        Z = torch.from_numpy(data['Z']).long()
        cell = torch.from_numpy(data['cell']).float()
        E = torch.from_numpy(data['E']).float().unsqueeze(0) if data['E'].ndim == 0 else torch.from_numpy(data['E']).float()
        F = torch.from_numpy(data['F']).float()
        
        if torch.isnan(F).any():
            print(f"[WARN] NaN found in forces for sample: {self.file_list[idx]}.npz")

        N_atoms = torch.tensor(R.shape[0], dtype=torch.long)

        return {'R': R, 'Z': Z, 'cell': cell, 'E': E, 'F': F, 'N_atoms': N_atoms}


def pad_collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    可変サイズの原子数を持つサンプルをパディングしてバッチ化するカスタムcollate関数。
    """
    if not batch:
        print("[WARN] pad_collate_fn received an empty batch.")
        return {
            'R': torch.empty(0, 0, 3, dtype=torch.float32), 
            'Z': torch.empty(0, 0, dtype=torch.long),
            'cell': torch.empty(0, 3, 3, dtype=torch.float32),
            'E': torch.empty(0, 1, dtype=torch.float32), # (B,1) を想定
            'F': torch.empty(0, 0, 3, dtype=torch.float32),
            'N_atoms': torch.empty(0, dtype=torch.long),
            'atom_mask': torch.empty(0, 0, dtype=torch.bool)
        }
    
    # Filter out samples that might be empty due to file not found errors in __getitem__
    # This is a workaround; ideally, __getitem__ should not return malformed data.
    valid_batch = [s for s in batch if s['N_atoms'].numel() > 0 and s['R'].numel() > 0]
    if not valid_batch: # If all samples were invalid
        print("[WARN] pad_collate_fn: all samples in batch were invalid (e.g., due to missing files).")
        # Return structure expected by DataLoader, but empty
        return {
            'R': torch.empty(0, 0, 3, dtype=torch.float32), 
            'Z': torch.empty(0, 0, dtype=torch.long),
            'cell': torch.empty(0, 3, 3, dtype=torch.float32),
            'E': torch.empty(0, 1, dtype=torch.float32),
            'F': torch.empty(0, 0, 3, dtype=torch.float32),
            'N_atoms': torch.empty(0, dtype=torch.long),
            'atom_mask': torch.empty(0, 0, dtype=torch.bool)
        }
    batch = valid_batch


    max_N_atoms = 0
    for sample in batch:
        if sample['N_atoms'].item() > max_N_atoms:
            max_N_atoms = sample['N_atoms'].item()
    
    if max_N_atoms == 0: # If all valid samples have 0 atoms
        max_N_atoms = 1 # Avoid creating 0-dimension tensors where not intended for padding

    batched_R = []
    batched_Z = []
    batched_F = []
    batched_E = []
    batched_cell = []
    batched_N_atoms = [] 
    atom_mask_list = []

    default_dtype_float = batch[0]['R'].dtype if batch else torch.float32
    default_dtype_long = batch[0]['Z'].dtype if batch else torch.long
    default_dtype_bool = torch.bool


    for sample in batch:
        N_atoms_current = sample['N_atoms'].item()
        pad_size_R_atoms = max_N_atoms - N_atoms_current
        
        padded_R = torch.cat([sample['R'], torch.zeros(pad_size_R_atoms, 3, dtype=sample.get('R', torch.empty(0,3,dtype=default_dtype_float)).dtype)], dim=0)
        batched_R.append(padded_R)

        padded_Z = torch.cat([sample['Z'], torch.zeros(pad_size_R_atoms, dtype=sample.get('Z', torch.empty(0,dtype=default_dtype_long)).dtype)], dim=0)
        batched_Z.append(padded_Z)

        padded_F = torch.cat([sample['F'], torch.zeros(pad_size_R_atoms, 3, dtype=sample.get('F', torch.empty(0,3,dtype=default_dtype_float)).dtype)], dim=0)
        batched_F.append(padded_F)

        batched_E.append(sample['E'])
        batched_cell.append(sample['cell'])
        batched_N_atoms.append(sample['N_atoms'])

        mask = torch.zeros(max_N_atoms, dtype=default_dtype_bool)
        if N_atoms_current > 0:
            mask[:N_atoms_current] = True
        atom_mask_list.append(mask)

    collated_batch = {
        'R': torch.stack(batched_R, dim=0),
        'Z': torch.stack(batched_Z, dim=0),
        'cell': torch.stack(batched_cell, dim=0),
        'E': torch.stack(batched_E, dim=0),
        'F': torch.stack(batched_F, dim=0),
        'N_atoms': torch.stack(batched_N_atoms, dim=0),
        'atom_mask': torch.stack(atom_mask_list, dim=0)
    }
    return collated_batch


def get_dataloader(
    splits_json: Path,
    processed_dir: Path,
    split: str,
    batch_size: int, # デフォルト値削除
    shuffle: bool,   # デフォルト値削除
    num_workers: int,# デフォルト値削除
    pin_memory: bool # デフォルト値削除
    # use_precomputed_features: bool # config から渡す場合
) -> DataLoader:
    """
    splits.json からファイルリストを読み込み、対応する DataLoader を返す

    Args:
        splits_json (Path): splits.json のパス
        processed_dir (Path): processed ディレクトリ
        split (str): 'train' | 'valid' | 'test'
        batch_size (int): バッチサイズ
        shuffle (bool): 訓練時にシャッフルするか
        num_workers (int): DataLoader のワーカープロセス数
        pin_memory (bool): ピンメモリを使うか (GPU 利用時に有効)
    Returns:
        DataLoader: バッチイテレータ
    """
    if not splits_json.exists():
        # raise FileNotFoundError(f"splits.json not found at {splits_json}") # より明確なエラー
        print(f"[ERROR] splits.json not found at {splits_json}", file=sys.stderr)
        # 空のDataLoaderを返すか、呼び出し側で処理するか。
        # ここでは空のデータセットでDataLoaderを作成する（エラーの原因特定が難しくなる可能性あり）
        file_list: List[str] = []
    else:
        with open(splits_json, 'r') as f:
            splits_data = json.load(f)
        if split not in splits_data:
            raise ValueError(f"Invalid split name: {split}. Available splits: {list(splits_data.keys())}")
        file_list: List[str] = splits_data[split]

    if not file_list:
        print(f"[WARN] No files found for split '{split}' in {splits_json}.")
    
    dataset = HDNPDataset(
        file_list=file_list,
        processed_dir=processed_dir,
        # use_precomputed=use_precomputed_features # config から渡す場合
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_fn
    )
    return dataloader
