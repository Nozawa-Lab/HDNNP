'''
hdnnp.symmetry_calculator

PyTorch ベースの Behler--Parrinello 型対称性関数計算クラス

Class:
  - SymmetryCalculator: R, Z, cell から対称性関数テンソル G を計算
    (unordered species pair support, ネイバーリスト導入による最適化)
'''

import math
import torch
from torch import Tensor
from typing import Tuple, List, Dict

# config をインポート
from . import config as hdnnp_config

# --- Helper Functions ---
def _cutoff(r: Tensor, r_cut_val: float) -> Tensor:
    """
    カットオフ関数。r > r_cut_val で 0 となる。
    Args:
        r (Tensor): 原子間距離のテンソル
        r_cut_val (float): カットオフ半径
    Returns:
        Tensor: カットオフ関数の値
    """
    fc = 0.5 * (torch.cos(math.pi * r / r_cut_val) + 1.0)
    return torch.where(r <= r_cut_val, fc, torch.zeros_like(r))

def _get_species_indices(Z: Tensor, species_map: Dict[str, int], z2element_map: Dict[int, str]) -> Tensor:
    """
    原子番号 Z のテンソルを、species_list に基づく種インデックスのテンソルに変換する。
    Args:
        Z (Tensor): 原子番号のテンソル (N_atoms,)
        species_map (Dict[str, int]): 要素記号を種インデックスにマッピングする辞書
        z2element_map (Dict[int, str]): 原子番号を要素記号にマッピングする辞書
    Returns:
        Tensor: 種インデックスのテンソル (N_atoms,)
    """
    idx = torch.empty(Z.size(0), dtype=torch.long, device=Z.device)
    for atom_idx, z_val_tensor in enumerate(Z):
        z_val = int(z_val_tensor.item())
        sym = z2element_map.get(z_val)
        if sym is None:
            raise ValueError(f"Atomic number {z_val} not found in z2element_map.")
        if sym not in species_map:
            raise ValueError(f"Element symbol '{sym}' (from Z={z_val}) not in SPECIES list: {list(species_map.keys())}")
        idx[atom_idx] = species_map[sym]
    return idx

# --- Main SymmetryCalculator Class ---
class SymmetryCalculator:
    def __init__(self) -> None:
        """
        SymmetryCalculator の初期化。
        config から対称性関数のパラメータや原子種情報を読み込み、
        計算に必要な内部状態（ルックアップテーブルなど）を準備する。
        """
        self.r_cut = hdnnp_config.R_CUT
        self.g2_eta_t = torch.tensor(hdnnp_config.G2_ETA, dtype=torch.float32)
        self.g2_rs_t = torch.tensor(hdnnp_config.G2_RS, dtype=torch.float32)
        self.g3_eta_t = torch.tensor(hdnnp_config.G3_ETA, dtype=torch.float32)
        self.g3_lam_t = torch.tensor(hdnnp_config.G3_LAM, dtype=torch.float32)
        self.g3_zet_t = torch.tensor(hdnnp_config.G3_ZET, dtype=torch.float32)
        self.species_list = hdnnp_config.SPECIES
        
        self.z2element_map = {int(k): v for k, v in hdnnp_config.Z2ELEMENT.items()}
        self.eps = hdnnp_config.NUMERICAL_EPSILON

        self.species_map = {symbol: i for i, symbol in enumerate(self.species_list)}
        self.num_species = len(self.species_list)

        self.species_pair_indices_list: List[Tuple[int, int]] = []
        self.species_pair_to_channel_idx: Dict[Tuple[int, int], int] = {}
        pair_channel_counter = 0
        for i in range(self.num_species):
            for j in range(i, self.num_species):
                self.species_pair_indices_list.append((i, j))
                self.species_pair_to_channel_idx[(i,j)] = pair_channel_counter
                pair_channel_counter += 1
        self.num_species_pairs = len(self.species_pair_indices_list)

        self.num_g2_base_params = len(self.g2_eta_t) * len(self.g2_rs_t)
        self.num_g3_base_params = len(self.g3_eta_t) * len(self.g3_lam_t) * len(self.g3_zet_t)
        
        self.g2_dim_total = self.num_g2_base_params * self.num_species_pairs
        self.g3_dim_total = self.num_g3_base_params * self.num_species_pairs
        self.total_sf_dim = self.g2_dim_total + self.g3_dim_total
        
        self.species_pair_lookup_matrix = torch.full(
            (self.num_species, self.num_species), -1, dtype=torch.long
        )
        for (s_idx1, s_idx2), ch_idx in self.species_pair_to_channel_idx.items():
            self.species_pair_lookup_matrix[s_idx1, s_idx2] = ch_idx
            self.species_pair_lookup_matrix[s_idx2, s_idx1] = ch_idx

        # ▼▼▼ 追加箇所: SFインデックスとパラメータのマッピングを作成 ▼▼▼
        self.sf_index_map: List[str] = []
        
        # G2パラメータのマッピング
        for sp_idx1, sp_idx2 in self.species_pair_indices_list:
            s1_name = self.species_list[sp_idx1]
            s2_name = self.species_list[sp_idx2]
            for eta in self.g2_eta_t:
                for rs in self.g2_rs_t:
                    self.sf_index_map.append(f"G2({s1_name}-{s2_name}): eta={eta.item():.2f}, rs={rs.item():.2f}")

        # G3パラメータのマッピング
        for sp_idx1, sp_idx2 in self.species_pair_indices_list:
            s1_name = self.species_list[sp_idx1]
            s2_name = self.species_list[sp_idx2]
            for eta in self.g3_eta_t:
                for lam in self.g3_lam_t:
                    for zet in self.g3_zet_t:
                        self.sf_index_map.append(f"G3({s1_name}-{s2_name}): eta={eta.item():.2f}, lam={lam.item():.1f}, zet={zet.item():.1f}")
        # ▲▲▲ 追加ここまで ▲▲▲

    def _get_neighbors_mic(self, R: Tensor, cell: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        最小イメージ規則 (MIC) を用いて、各原子の近傍原子リストと関連情報を計算する。
        """
        N = R.size(0)
        device = R.device
        if N == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_float = torch.empty(0, dtype=R.dtype, device=device)
            empty_float_3d = torch.empty(0, 3, dtype=R.dtype, device=device)
            return empty_long, empty_long, empty_float, empty_float_3d, empty_float
        
        _cell = cell.detach() if not cell.requires_grad and R.requires_grad else cell
        _inv_cell = torch.inverse(_cell)

        frac_coords = R @ _inv_cell
        diff_frac_all = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
        diff_frac_all = diff_frac_all - diff_frac_all.round()
        diff_cart_all = diff_frac_all @ _cell

        dists_sq_all = torch.sum(diff_cart_all.pow(2), dim=2)
        
        diag_indices = torch.arange(N, device=device)
        dists_sq_all[diag_indices, diag_indices] = (self.r_cut + 1.0)**2
        
        adj = dists_sq_all < (self.r_cut**2)
        idx_i, idx_j = torch.where(adj)

        if idx_i.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_float = torch.empty(0, dtype=R.dtype, device=device)
            empty_float_3d = torch.empty(0, 3, dtype=R.dtype, device=device)
            return empty_long, empty_long, empty_float, empty_float_3d, empty_float
        
        R_ij_sq = dists_sq_all[idx_i, idx_j]
        R_ij = torch.sqrt(R_ij_sq + self.eps)
        diff_ij = diff_cart_all[idx_i, idx_j]
        fc_ij = _cutoff(R_ij, self.r_cut)

        return idx_i, idx_j, R_ij, diff_ij, fc_ij

    def compute(self, R_global: Tensor, Z: Tensor, cell: Tensor) -> Tensor:
        """
        原子座標 R, 原子番号 Z, 格子ベクトル cell から対称性関数テンソル G を計算する。
        """
        N = R_global.size(0)
        device = R_global.device
        dtype = R_global.dtype

        if N == 0:
            return torch.zeros((0, self.total_sf_dim), device=device, dtype=dtype)

        species_atomic_indices = _get_species_indices(Z, self.species_map, self.z2element_map)
        
        idx_i, idx_j, R_ij, diff_ij, fc_ij = self._get_neighbors_mic(R_global, cell)
        
        num_valid_pairs = idx_i.size(0)

        # --- G2 対称性関数の計算 (変更なし) ---
        G2_contrib = torch.zeros((N, self.g2_dim_total), device=device, dtype=dtype)
        if num_valid_pairs > 0:
            g2_eta_dev = self.g2_eta_t.to(device)
            g2_rs_dev = self.g2_rs_t.to(device)
            
            term_g2_radial_parts = torch.exp(
                -g2_eta_dev.view(1, -1, 1) * \
                (R_ij.view(-1, 1, 1) - g2_rs_dev.view(1, 1, -1)).pow(2)
            )
            term_g2_radial_flat = term_g2_radial_parts.reshape(num_valid_pairs, -1)
            g2_vals_per_pair_base = term_g2_radial_flat * fc_ij.unsqueeze(-1)

            species_idx_i_of_pairs = species_atomic_indices[idx_i]
            species_idx_j_of_pairs = species_atomic_indices[idx_j]
            
            pair_channels = self.species_pair_lookup_matrix.to(device)[species_idx_i_of_pairs, species_idx_j_of_pairs]
            
            expanded_pair_channels = pair_channels.unsqueeze(1)
            base_param_indices = torch.arange(self.num_g2_base_params, device=device).unsqueeze(0)
            target_g2_columns_for_pairs = expanded_pair_channels * self.num_g2_base_params + base_param_indices
            
            G2_contrib.index_put_(
                (idx_i.long().unsqueeze(1).expand_as(g2_vals_per_pair_base), target_g2_columns_for_pairs.long()),
                g2_vals_per_pair_base,
                accumulate=True
            )

        # --- G3 対称性関数の計算 (完全ベクトル化) ---
        G3_contrib = torch.zeros((N, self.g3_dim_total), device=device, dtype=dtype)
        if num_valid_pairs >= 2:
            # Step 1: 全てのトリプレット候補 (i, j, k) を生成する
            same_i_mask = (idx_i.unsqueeze(1) == idx_i.unsqueeze(0))
            not_same_j_mask = (idx_j.unsqueeze(1) != idx_j.unsqueeze(0))
            triu_mask = torch.triu(torch.ones_like(same_i_mask), diagonal=1).bool()
            valid_triplet_pair_indices_mask = same_i_mask & not_same_j_mask & triu_mask
            pair1_indices, pair2_indices = torch.where(valid_triplet_pair_indices_mask)

            if pair1_indices.numel() > 0:
                # Step 2: トリプレット情報を一括で取得
                triplet_i_idx = idx_i[pair1_indices]
                triplet_j_idx = idx_j[pair1_indices]
                triplet_k_idx = idx_j[pair2_indices]

                R_ij_triplet    = R_ij[pair1_indices]
                diff_ij_triplet = diff_ij[pair1_indices]
                fc_ij_triplet   = fc_ij[pair1_indices]
                
                R_ik_triplet    = R_ij[pair2_indices]
                diff_ik_triplet = diff_ij[pair2_indices]
                fc_ik_triplet   = fc_ij[pair2_indices]
                
                # Step 3: R_jk と cos(theta_ijk) を一括計算
                diff_jk_triplet = diff_ik_triplet - diff_ij_triplet
                R_jk_sq_triplet = torch.sum(diff_jk_triplet.pow(2), dim=1)
                
                valid_jk_mask = R_jk_sq_triplet < (self.r_cut**2)
                
                if valid_jk_mask.any():
                    # 有効なトリプレットのみに情報をフィルタリング
                    R_ij_triplet = R_ij_triplet[valid_jk_mask]
                    diff_ij_triplet = diff_ij_triplet[valid_jk_mask]
                    fc_ij_triplet = fc_ij_triplet[valid_jk_mask]
                    
                    R_ik_triplet = R_ik_triplet[valid_jk_mask]
                    diff_ik_triplet = diff_ik_triplet[valid_jk_mask]
                    fc_ik_triplet = fc_ik_triplet[valid_jk_mask]
                    
                    R_jk_sq_triplet = R_jk_sq_triplet[valid_jk_mask]
                    R_jk_triplet = torch.sqrt(R_jk_sq_triplet + self.eps)
                    fc_jk_triplet = _cutoff(R_jk_triplet, self.r_cut)
                    
                    triplet_i_idx = triplet_i_idx[valid_jk_mask]
                    triplet_j_idx = triplet_j_idx[valid_jk_mask]
                    triplet_k_idx = triplet_k_idx[valid_jk_mask]

                    num_valid_triplets = R_ij_triplet.size(0)

                    dot_product_ijk = torch.sum(diff_ij_triplet * diff_ik_triplet, dim=1)
                    cos_theta_ijk = dot_product_ijk / (R_ij_triplet * R_ik_triplet + self.eps)
                    cos_theta_ijk = torch.clamp(cos_theta_ijk, -1.0 + self.eps, 1.0 - self.eps)
                    
                    # Step 4: G3値を一括計算
                    sum_sq_dists_triplet = R_ij_triplet.pow(2) + R_ik_triplet.pow(2) + R_jk_sq_triplet
                    
                    g3_eta_dev = self.g3_eta_t.to(device)
                    g3_lam_dev = self.g3_lam_t.to(device)
                    g3_zet_dev = self.g3_zet_t.to(device)

                    radial_terms = torch.exp(-g3_eta_dev.view(1, -1) * sum_sq_dists_triplet.view(-1, 1))
                    base_angular = 1.0 + g3_lam_dev.view(1, -1) * cos_theta_ijk.view(-1, 1)
                    angular_terms = base_angular.unsqueeze(-1).pow(g3_zet_dev.view(1, 1, -1))
                    power_of_2_factor = torch.pow(2.0, 1.0 - g3_zet_dev)
                    
                    g3_base_values_no_fc = radial_terms.unsqueeze(-1).unsqueeze(-1) * \
                                           angular_terms.unsqueeze(1) * \
                                           power_of_2_factor.view(1, 1, 1, -1)
                    
                    fc_combined_triplet = fc_ij_triplet * fc_ik_triplet * fc_jk_triplet
                    
                    g3_vals_per_triplet_base = (g3_base_values_no_fc * fc_combined_triplet.view(-1, 1, 1, 1)).reshape(num_valid_triplets, -1)

                    # Step 5: 結果を集約
                    species_idx_j_of_triplets = species_atomic_indices[triplet_j_idx]
                    species_idx_k_of_triplets = species_atomic_indices[triplet_k_idx]
                    
                    species_pair_lookup_matrix_dev = self.species_pair_lookup_matrix.to(device)
                    triplet_pair_channels = species_pair_lookup_matrix_dev[species_idx_j_of_triplets, species_idx_k_of_triplets]
                    
                    base_g3_param_indices = torch.arange(self.num_g3_base_params, device=device).unsqueeze(0)
                    target_g3_columns_for_triplets = triplet_pair_channels.unsqueeze(1) * self.num_g3_base_params + base_g3_param_indices
                    
                    G3_contrib.index_put_(
                        (triplet_i_idx.long().unsqueeze(1).expand_as(g3_vals_per_triplet_base), target_g3_columns_for_triplets.long()),
                        g3_vals_per_triplet_base,
                        accumulate=True
                    )

        # --- 全対称性関数を結合 ---
        output = torch.cat([G2_contrib, G3_contrib], dim=1)
        return output