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
    _G3_TRIPLET_CHUNK_SIZE = 32_768

    def __init__(self) -> None:
        """
        SymmetryCalculator の初期化。
        config から対称性関数のパラメータや原子種情報を読み込み、
        計算に必要な内部状態（ルックアップテーブルなど）を準備する。
        """
        self.r_cut = hdnnp_config.R_CUT
        self.g2_eta_t = torch.tensor(hdnnp_config.G2_ETA, dtype=hdnnp_config.DEFAULT_TORCH_DTYPE)
        self.g2_rs_t = torch.tensor(hdnnp_config.G2_RS, dtype=hdnnp_config.DEFAULT_TORCH_DTYPE)
        self.g3_eta_t = torch.tensor(hdnnp_config.G3_ETA, dtype=hdnnp_config.DEFAULT_TORCH_DTYPE)
        self.g3_lam_t = torch.tensor(hdnnp_config.G3_LAM, dtype=hdnnp_config.DEFAULT_TORCH_DTYPE)
        self.g3_zet_t = torch.tensor(hdnnp_config.G3_ZET, dtype=hdnnp_config.DEFAULT_TORCH_DTYPE)
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
        
        # 近傍の採否は微分できないため、N x N の探索テンソルを計算グラフに
        # 保持しない。選択されたペアだけを後段で再計算して微分対象にする。
        with torch.no_grad():
            cell_search = cell.detach()
            frac_coords = R.detach() @ torch.inverse(cell_search)
            diff_frac_all = frac_coords.unsqueeze(1) - frac_coords.unsqueeze(0)
            image_shifts_all = diff_frac_all.round()
            diff_frac_all.sub_(image_shifts_all)
            dists_sq_all = torch.sum((diff_frac_all @ cell_search).square(), dim=2)
            dists_sq_all.fill_diagonal_((self.r_cut + 1.0) ** 2)
            idx_i, idx_j = torch.where(dists_sq_all < self.r_cut ** 2)
            image_shifts_ij = image_shifts_all[idx_i, idx_j]
        del frac_coords, diff_frac_all, image_shifts_all, dists_sq_all # メモリ解放

        if idx_i.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=device)
            empty_float = torch.empty(0, dtype=R.dtype, device=device)
            empty_float_3d = torch.empty(0, 3, dtype=R.dtype, device=device)
            return empty_long, empty_long, empty_float, empty_float_3d, empty_float
        
        diff_ij = R[idx_i] - R[idx_j] - image_shifts_ij @ cell
        R_ij_sq = torch.sum(diff_ij.square(), dim=1)
        R_ij = torch.sqrt(R_ij_sq + self.eps)
        fc_ij = _cutoff(R_ij, self.r_cut)

        return idx_i, idx_j, R_ij, diff_ij, fc_ij

    @staticmethod
    def _build_triplet_pair_indices(idx_i: Tensor) -> Tuple[Tensor, Tensor]:
        """同じ中心原子を持つ近傍ペアの組合せを O(T) メモリで作る。"""
        device = idx_i.device
        if idx_i.numel() < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        # torch.where 由来の idx_i は中心原子の昇順に並んでいる。
        neighbor_counts = torch.unique_consecutive(idx_i, return_counts=True)[1].tolist()
        num_triplets = sum(count * (count - 1) // 2 for count in neighbor_counts)
        pair1_indices = torch.empty(num_triplets, dtype=torch.long, device=device)
        pair2_indices = torch.empty_like(pair1_indices)

        edge_offset = 0
        triplet_offset = 0
        for count in neighbor_counts:
            count = int(count)
            count_triplets = count * (count - 1) // 2
            if count_triplets: # count_triplets が 0 でない場合のみ処理
                local_pairs = torch.triu_indices(
                    count, count, offset=1, device=device
                )
                target = slice(triplet_offset, triplet_offset + count_triplets)
                pair1_indices[target] = local_pairs[0] + edge_offset
                pair2_indices[target] = local_pairs[1] + edge_offset
                triplet_offset += count_triplets
            edge_offset += count

        return pair1_indices, pair2_indices

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
            g2_eta_dev = self.g2_eta_t.to(device, dtype=dtype)
            g2_rs_dev = self.g2_rs_t.to(device, dtype=dtype)
            
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
            
            g2_vals_per_pair_base = g2_vals_per_pair_base.to(dtype=dtype)
            G2_contrib.index_put_(
                (idx_i.long().unsqueeze(1).expand_as(g2_vals_per_pair_base), target_g2_columns_for_pairs.long()),
                g2_vals_per_pair_base,
                accumulate=True
            )

        # --- G3 対称性関数の計算 (中心原子ごとの候補生成 + チャンク処理) ---
        G3_contrib = torch.zeros((N, self.g3_dim_total), device=device, dtype=dtype)
        if num_valid_pairs >= 2:
            # P x P マスクを作らず、各中心原子の近傍同士だけを組み合わせる。
            pair1_indices, pair2_indices = self._build_triplet_pair_indices(idx_i)

            if pair1_indices.numel() > 0:
                g3_eta_dev = self.g3_eta_t.to(device, dtype=dtype)
                g3_lam_dev = self.g3_lam_t.to(device, dtype=dtype)
                g3_zet_dev = self.g3_zet_t.to(device, dtype=dtype)
                power_of_2_factor = torch.pow(2.0, 1.0 - g3_zet_dev)
                species_pair_lookup_matrix_dev = self.species_pair_lookup_matrix.to(device)
                base_g3_param_indices = torch.arange(
                    self.num_g3_base_params, device=device
                ).unsqueeze(0)

                # パラメータ展開テンソルを全トリプレット分保持しない。
                for chunk_start in range(
                    0, pair1_indices.numel(), self._G3_TRIPLET_CHUNK_SIZE
                ):
                    chunk_end = min(
                        chunk_start + self._G3_TRIPLET_CHUNK_SIZE,
                        pair1_indices.numel(),
                    )
                    pair1_chunk = pair1_indices[chunk_start:chunk_end]
                    pair2_chunk = pair2_indices[chunk_start:chunk_end]

                    diff_ij_triplet = diff_ij[pair1_chunk]
                    diff_ik_triplet = diff_ij[pair2_chunk]
                    diff_jk_triplet = diff_ik_triplet - diff_ij_triplet
                    R_jk_sq_triplet = torch.sum(diff_jk_triplet.square(), dim=1)
                    valid_jk_mask = R_jk_sq_triplet < self.r_cut ** 2
                    valid_indices = torch.where(valid_jk_mask)[0]
                    if valid_indices.numel() == 0:
                        continue

                    pair1_valid = pair1_chunk[valid_indices]
                    pair2_valid = pair2_chunk[valid_indices]
                    R_ij_triplet = R_ij[pair1_valid]
                    R_ik_triplet = R_ij[pair2_valid]
                    diff_ij_triplet = diff_ij_triplet[valid_indices]
                    diff_ik_triplet = diff_ik_triplet[valid_indices]
                    R_jk_sq_triplet = R_jk_sq_triplet[valid_indices]
                    R_jk_triplet = torch.sqrt(R_jk_sq_triplet + self.eps)

                    dot_product_ijk = torch.sum(
                        diff_ij_triplet * diff_ik_triplet, dim=1
                    )
                    cos_theta_ijk = dot_product_ijk / (
                        R_ij_triplet * R_ik_triplet + self.eps
                    )
                    cos_theta_ijk = torch.clamp(
                        cos_theta_ijk, -1.0 + self.eps, 1.0 - self.eps
                    )
                    sum_sq_dists_triplet = (
                        R_ij_triplet.square()
                        + R_ik_triplet.square()
                        + R_jk_sq_triplet
                    )

                    radial_terms = torch.exp(
                        -g3_eta_dev.view(1, -1)
                        * sum_sq_dists_triplet.view(-1, 1)
                    )
                    base_angular = (
                        1.0
                        + g3_lam_dev.view(1, -1)
                        * cos_theta_ijk.view(-1, 1)
                    )
                    angular_terms = base_angular.unsqueeze(-1).pow(
                        g3_zet_dev.view(1, 1, -1)
                    )
                    g3_base_values_no_fc = (
                        radial_terms.unsqueeze(-1).unsqueeze(-1)
                        * angular_terms.unsqueeze(1)
                        * power_of_2_factor.view(1, 1, 1, -1)
                    )
                    fc_combined_triplet = (
                        fc_ij[pair1_valid]
                        * fc_ij[pair2_valid]
                        * _cutoff(R_jk_triplet, self.r_cut)
                    )
                    g3_vals_per_triplet_base = (
                        g3_base_values_no_fc
                        * fc_combined_triplet.view(-1, 1, 1, 1)
                    ).reshape(valid_indices.numel(), -1)

                    triplet_i_idx = idx_i[pair1_valid]
                    species_idx_j = species_atomic_indices[idx_j[pair1_valid]]
                    species_idx_k = species_atomic_indices[idx_j[pair2_valid]]
                    triplet_pair_channels = species_pair_lookup_matrix_dev[
                        species_idx_j, species_idx_k
                    ]
                    target_g3_columns = (
                        triplet_pair_channels.unsqueeze(1)
                        * self.num_g3_base_params
                        + base_g3_param_indices
                    )
                    G3_contrib.index_put_(
                        (
                            triplet_i_idx.unsqueeze(1).expand_as(
                                g3_vals_per_triplet_base
                            ),
                            target_g3_columns,
                        ),
                        g3_vals_per_triplet_base,
                        accumulate=True,
                    )

        # --- 全対称性関数を結合 ---
        output = torch.cat([G2_contrib, G3_contrib], dim=1)
        return output
