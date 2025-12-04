'''
hdnnp.model

PyTorch implementation of HDNNP neural network modules.

Classes:
  - ElementNN: element-specific feedforward network.
  - HDNNPModel: full model computing per-atom energies from symmetry descriptors.
'''

import torch
from torch import nn, Tensor
# config をインポート
from . import config as hdnnp_config


class ElementNN(nn.Module):
    """Element-specific neural network mapping local descriptor to atomic energy."""
    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim (int): number of input features per atom
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        # config から隠れ層の定義を読み込む
        for hidden_dim in hdnnp_config.HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            # ▼▼▼ 修正箇所 ▼▼▼
            # configに基づいてDropout層を追加
            if hdnnp_config.USE_DROPOUT and hdnnp_config.DROPOUT_RATE > 0:
                layers.append(nn.Dropout(hdnnp_config.DROPOUT_RATE))
            # ▲▲▲ 修正箇所 ▲▲▲
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): descriptor tensor, shape (n_atoms, n_features)
        Returns:
            Tensor: per-atom energy, shape (n_atoms,)
        """
        return self.net(x).squeeze(-1)


class HDNNPModel(nn.Module):
    """High-Dimensional Neural Network Potential model."""
    def __init__(self, input_dim: int) -> None:
        """
        Args:
            input_dim (int): number of symmetry-function features per atom (|G2|+|G3|)
        """
        super().__init__()
        # element-specific networks
        # config から SPECIES を読み込む
        self.element_nns = nn.ModuleDict({
            element: ElementNN(input_dim) for element in hdnnp_config.SPECIES
        })
        # atomic number -> element symbol mapping
        # config から Z2ELEMENT を読み込む
        self.z2element = hdnnp_config.Z2ELEMENT

    def forward(self, G: Tensor, Z: Tensor) -> Tensor:
        """
        Compute per-atom energies.

        Args:
            G (Tensor): symmetry descriptors, shape (N, F)
            Z (Tensor): atomic numbers, shape (N,)
        Returns:
            Tensor: per-atom energies, shape (N,)
        """
        device = G.device
        N = G.size(0)
        E_atoms = torch.zeros(N, device=device, dtype=G.dtype)
        for z_val, element_symbol in self.z2element.items(): # z2element.items() を使用
            # Ensure z_val is an int for comparison if Z contains float representations
            mask = (Z.long() == int(z_val)) # Zをlongにキャストして比較
            if torch.any(mask):
                idx = torch.nonzero(mask, as_tuple=True)[0]
                G_sel = G[idx]
                # ElementNNのキーもelement_symbolであることを確認
                if element_symbol in self.element_nns:
                    E_sel = self.element_nns[element_symbol](G_sel)
                    E_atoms[idx] = E_sel
                else:
                    # このケースは通常発生しないはず (SPECIESとZ2ELEMENTが対応していれば)
                    print(f"Warning: Element symbol '{element_symbol}' for Z={z_val} not found in ElementNNs.")
        return E_atoms