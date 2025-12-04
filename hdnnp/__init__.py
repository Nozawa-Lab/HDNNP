# hdnnp/__init__.py の修正案

__version__ = "0.1.0"

# パッケージ公開 API
from .config import (
    PROJECT_ROOT, R_CUT, G2_ETA, G2_RS, G3_ETA, G3_LAM, G3_ZET,
    HIDDEN_LAYERS as HIDDEN,       # HIDDEN_LAYERS を HIDDEN としてインポート
    LOSS_ALPHA_E as ALPHA_E,       # LOSS_ALPHA_E を ALPHA_E としてインポート
    LOSS_BETA_F as BETA_F,         # LOSS_BETA_F を BETA_F としてインポート
    SPECIES, Z2ELEMENT
)
from .symmetry_calculator import SymmetryCalculator
from .model import ElementNN, HDNNPModel
from .loss import combined_loss
from .data_loader import HDNPDataset, get_dataloader

__all__ = [
    "PROJECT_ROOT", "R_CUT", "G2_ETA", "G2_RS", "G3_ETA", "G3_LAM", "G3_ZET",
    "HIDDEN", "ALPHA_E", "BETA_F", "SPECIES", "Z2ELEMENT", # __all__ はそのままでOK
    "cutoff", "g2_block", "g3_block",
    "SymmetryCalculator", "ElementNN", "HDNNPModel",
    "combined_loss", "HDNPDataset", "get_dataloader",
]