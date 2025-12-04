'''
hdnnp.loss

Definition of the combined energy-force loss function for HDNNP.
'''

import torch
from torch import Tensor
# config をインポート
from . import config as hdnnp_config


def combined_loss(
    E_pred: Tensor,
    E_true: Tensor,
    F_pred: Tensor,
    F_true: Tensor,
    alpha: float = hdnnp_config.LOSS_ALPHA_E, # config からデフォルト値を取得
    beta: float = hdnnp_config.LOSS_BETA_F   # config からデフォルト値を取得
) -> Tensor:
    """
    Compute the combined loss:
        L = alpha * MSE(E_pred, E_true)
          + beta  * MSE(F_pred, F_true)

    Args:
        E_pred (Tensor): predicted total energy (scalar or batch)
        E_true (Tensor): true total energy (same shape as E_pred)
        F_pred (Tensor): predicted forces, shape (..., N_atoms, 3)
        F_true (Tensor): true forces, same shape as F_pred
        alpha (float): weight for energy loss
        beta  (float): weight for force loss
    Returns:
        Tensor: combined loss value
    """
    # Energy loss (MSE)
    energy_loss = torch.mean((E_pred - E_true)**2)
    # Force loss (MSE)
    force_loss = torch.mean((F_pred - F_true)**2)
    # Combined loss
    loss = alpha * energy_loss + beta * force_loss
    return loss