"""
hdnnp.config

Configuration parameters for the HDNNP model and symmetry-function calculations.
"""

import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Union, Optional

import torch

# ------------------------------------------------------------------
# Precision Configuration
# ------------------------------------------------------------------
#: 全体のPyTorch演算精度。環境変数 HDNNP_FLOAT_PRECISION で切り替え可能
DEFAULT_FLOAT_PRECISION: str = os.getenv("HDNNP_FLOAT_PRECISION", "float64").lower()
if DEFAULT_FLOAT_PRECISION in {"64", "float64", "double"}:
    DEFAULT_TORCH_DTYPE: torch.dtype = torch.float64
elif DEFAULT_FLOAT_PRECISION in {"32", "float32", "single"}:
    DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32
else:
    raise ValueError("HDNNP_FLOAT_PRECISION must be 'float32' or 'float64'.")

# ------------------------------------------------------------------
# Project Structure Configuration
# ------------------------------------------------------------------
#: プロジェクトのルートディレクトリ
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
#: データディレクトリのベースパス
DATA_DIR: Path = PROJECT_ROOT / "data"
#: 生データが格納されるディレクトリ (run_extract.py の入力)
RAW_DATA_DIR: Path = DATA_DIR / "raw"
#: 前処理済みデータ (.npz ファイル) が格納されるディレクトリ
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
#: チェックポイント (学習済みモデルなど) が保存されるディレクトリ
CHECKPOINT_DIR: Path = PROJECT_ROOT / "chkpt"
#: データ分割情報が記述されたJSONファイル名 (PROCESSED_DATA_DIR 内に配置)
SPLITS_FILENAME: str = "splits.json"
#: 学習済みモデルのファイル名 (CHECKPOINT_DIR 内に配置)
TRAINED_MODEL_FILENAME: str = "model_fullbatch.pt"
# ▼▼▼ 以下を追加 ▼▼▼
#: 対称性関数の最小値/最大値を保存するファイル名 (CHECKPOINT_DIR 内に配置)
SF_RANGE_FILENAME: str = "sf_range.pt"
# ▲▲▲ ここまで ▲▲▲

# ------------------------------------------------------------------
# Execution Environment
# ------------------------------------------------------------------
#: 計算に使用するデバイス ("cuda", "cpu", または "auto" で自動検出)
DEVICE: str = "cuda:0"
#: PyTorchが使用するCPUコア数の割合 (0.0 < ratio <= 1.0)
CPU_USAGE_RATIO: float = 0.4
#: 数値計算上の微小量 (対称性関数計算などで使用)
NUMERICAL_EPSILON: float = 1e-8

# ------------------------------------------------------------------
# Data Extraction Configuration (for run_extract.py)
# ------------------------------------------------------------------
#: VASP出力から抽出したnpzファイルの命名規則フォーマット
NPZ_FILENAME_FORMAT: str = "{prefix}_frame{frame_num:04d}.npz"

# ------------------------------------------------------------------
# Data Splitting Configuration (for make_splits.py)
# ------------------------------------------------------------------
SPLIT_SEED: int = 1234
SPLIT_TRAIN_FRACTION: float = 0.8
SPLIT_VALID_FRACTION: float = 0.1

# ------------------------------------------------------------------
# DataLoader Configuration
# ------------------------------------------------------------------
DATALOADER_TRAIN_BATCH_SIZE: int = 64 # L-BFGSでは無視される可能性あり (フルバッチの場合)
DATALOADER_INFER_BATCH_SIZE: int = 128
DATALOADER_SHUFFLE_TRAIN: bool = False
DATALOADER_SHUFFLE_INFER: bool = False
DATALOADER_NUM_WORKERS: int = 0
DATALOADER_PIN_MEMORY: bool = True

# ------------------------------------------------------------------
# Symmetry Function Configuration
# ------------------------------------------------------------------
R_CUT: float = 6.0
# G2_RS: [0.0]から、原子間距離をカバーする複数の値に変更
G2_RS: List[float] = [0.0] 

# G2_ETA: 短距離の相互作用を捉えるため、より大きな値も追加
G2_ETA: List[float] = [0.01, 0.05, 0.1] 

# G3_ETA: G2と同様に設定を少し広げる
G3_ETA: List[float] = [0.01, 0.1] 

G3_LAM: List[int] = [1, -1]
G3_ZET: List[float] = [1.0, 4.0]
SPECIES: Tuple[str, ...] = ("Al", "Fe", "Pt")
Z2ELEMENT: Dict[int, str] = {13: "Al", 26: "Fe", 78: "Pt"}

# ------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------
HIDDEN_LAYERS: List[int] = [20, 20]

#: Dropout を使用するかどうかのスイッチ
USE_DROPOUT: bool = False
#: Dropout率 (USE_DROPOUTがTrueの場合にのみ有効)
DROPOUT_RATE: float = 0.1 # 一般的には 0.1 ~ 0.5 の値が使われます


# ------------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------------
#: 学習に力の情報を使用するかどうか
USE_FORCE_TRAINING: bool = False

# ▼▼▼ 以下を追加 ▼▼▼
# --- Extrapolation Check Configuration ---
#: 推論時に外挿検出を有効にするか
EXTRAPOLATION_CHECK_ENABLED: bool = True
# ▲▲▲ ここまで ▲▲▲

# --- Random Seed Configuration ---
#: 乱数シードを固定するかどうかのスイッチ (Trueで固定)
SET_RANDOM_SEED: bool = True
#: 固定する場合のシード値
RANDOM_SEED_VALUE: int = 42

#: 使用するオプティマイザの名前 ("LBFGS", "Adam", "SGD" など)
OPTIMIZER_NAME: str = "LBFGS"

#: オプティマイザのパラメータ (オプティマイザごとにキーと値を設定)
OPTIMIZER_PARAMS: Dict[str, Any] = {
    "LBFGS": {
        "lr": 0.1,
        "max_iter": 200,  # 1回のoptimizer.step()内のイテレーション (LBFGSの内部イテレーション)
        "line_search_fn": "strong_wolfe", # "strong_wolfe" または None
        "history_size": 100,
        "tolerance_grad": 1e-7,
        "tolerance_change": 1e-9,
    },
    "Adam": {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0,
    },
    "SGD": {
        "lr": 1e-2,
        "momentum": 0.9,
        "weight_decay": 0,
    }
    # 他のオプティマイザのパラメータもここに追加可能
}

#: 学習の最大イテレーション数 (optimizer.step() が呼ばれる回数)
#: LBFGSの場合、TRAIN_MAX_ITERATIONS * OPTIMIZER_PARAMS["LBFGS"]["max_iter"] が総計算回数に近いイメージ
TRAIN_MAX_ITERATIONS: int = 200 # run_train.py でのメインループのイテレーション数

#: エネルギー損失 (MSE) に対する重み α
LOSS_ALPHA_E: float = 1.0
#: 力損失 (MSE) に対する重み β (USE_FORCE_TRAININGがTrueの場合のみ有効)
LOSS_BETA_F: float = 8.0

# --- Early Stopping Configuration ---
#: 早期終了を有効にするか
EARLY_STOPPING_ENABLED: bool = True
#: 早期終了の監視対象 ('loss', 'rmse_e', 'rmse_f')
EARLY_STOPPING_MONITOR: str = "loss" # または 'rmse_e_pa', 'rmse_f_pa'
#: 監視対象の改善が見られなくなってから待つイテレーション数
EARLY_STOPPING_PATIENCE: int = 10
#: 「改善」とみなす最小の変化量 (絶対値)
EARLY_STOPPING_MIN_DELTA: float = 1e-6
#: 早期終了の目標値 (この値を下回ったら/上回ったら終了、監視対象による)
EARLY_STOPPING_TARGET_VALUE: Optional[float] = None # 例: 0.001 (lossの場合)
#: 早期終了のモード ('min' または 'max')。lossなら'min', 精度なら'max'
EARLY_STOPPING_MODE: str = "min" # loss や RMSE の場合は 'min'

# ------------------------------------------------------------------
# Plotting Configuration
# ------------------------------------------------------------------
PLOT_DPI: int = 300
PLOT_ENERGY_SCATTER_TRAIN_FILENAME: str = "energy_scatter_train.png"
PLOT_RMSE_ENERGY_TRAIN_FILENAME: str = "rmse_energy_vs_iter_train.png"
PLOT_RMSE_FORCE_TRAIN_FILENAME: str = "rmse_force_vs_iter_train.png"
PLOT_ENERGY_VS_STRUCT_TRAIN_INFER_FILENAME: str = "energy_vs_structure_train_infer.png"
PLOT_ENERGY_SCATTER_TEST_INFER_FILENAME: str = "energy_scatter_infer_test.png"
PLOT_ENERGY_VS_STRUCT_TEST_INFER_FILENAME: str = "energy_vs_structure_infer_test.png"
PLOT_FORCE_COMPONENTS_SCATTER_TEST_INFER_FILENAME: str = "force_components_scatter_infer_test.png"
PLOT_FORCE_ERROR_HISTOGRAM_TEST_INFER_FILENAME: str = "force_error_histogram_infer_test.png"