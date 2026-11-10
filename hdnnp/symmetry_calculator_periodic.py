'''
hdnnp.symmetry_calculator_periodic

スーパーセル表現に依存しない Behler--Parrinello 型対称性関数。

既存の ``hdnnp.symmetry_calculator`` は変更せず、このモジュールを使用する
場合だけ、カットオフ球内の全周期像を列挙する。公開インターフェースと
最終出力形状は既存の ``SymmetryCalculator`` と同じである。
'''

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor

from . import config as hdnnp_config


def _cutoff(r: Tensor, r_cut_val: float) -> Tensor:
    """Cosine cutoff function."""
    fc = 0.5 * (torch.cos(math.pi * r / r_cut_val) + 1.0)
    return torch.where(r <= r_cut_val, fc, torch.zeros_like(r))


def _get_species_indices(
    Z: Tensor,
    species_map: Dict[str, int],
    z2element_map: Dict[int, str],
) -> Tensor:
    """原子番号を設定ファイル(config.py)中の元素インデックスへ変換する。"""
    idx = torch.empty(Z.size(0), dtype=torch.long, device=Z.device)
    for atom_idx, z_val_tensor in enumerate(Z):
        z_val = int(z_val_tensor.item())
        symbol = z2element_map.get(z_val)
        if symbol is None:
            raise ValueError(f"Atomic number {z_val} not found in z2element_map.")
        if symbol not in species_map:
            raise ValueError(
                f"Element symbol '{symbol}' (from Z={z_val}) not in SPECIES "
                f"list: {list(species_map.keys())}"
            )
        idx[atom_idx] = species_map[symbol]
    return idx


class SymmetryCalculator:
    """
    全周期像を用いるスーパーセル不変な対称性関数計算器。

    ``compute(R, Z, cell)`` は既存実装と同じく ``(N, F)`` の Tensor を返す。
    近傍は ``(原子 j, 格子シフト n_a, n_b, n_c)`` で識別するため、同じ
    原子の異なる周期像も別々の近傍として正しく数える。
    """

    _G3_TRIPLET_CHUNK_SIZE = 32_768

    def __init__(self) -> None:
        self.r_cut = hdnnp_config.R_CUT # カットオフ距離
        dtype = hdnnp_config.DEFAULT_TORCH_DTYPE # デフォルトの torch dtype を取得
        self.g2_eta_t = torch.tensor(hdnnp_config.G2_ETA, dtype=dtype) # G2 の eta パラメータ
        self.g2_rs_t = torch.tensor(hdnnp_config.G2_RS, dtype=dtype) # G2 の rs パラメータ
        self.g3_eta_t = torch.tensor(hdnnp_config.G3_ETA, dtype=dtype) # G3 の eta パラメータ
        self.g3_lam_t = torch.tensor(hdnnp_config.G3_LAM, dtype=dtype) # G3 の lambda パラメータ
        self.g3_zet_t = torch.tensor(hdnnp_config.G3_ZET, dtype=dtype) # G3 の zeta パラメータ
        self.species_list = hdnnp_config.SPECIES # 元素リスト
        self.z2element_map = {
            int(key): value for key, value in hdnnp_config.Z2ELEMENT.items()
        } # 原子番号から元素記号へのマッピング(例: 1 -> 'H')
        self.eps = hdnnp_config.NUMERICAL_EPSILON # 数値計算用の微小値

        self.species_map = {
            symbol: index for index, symbol in enumerate(self.species_list)
        } # 元素記号からインデックスへのマッピング(例: species_listが['H', 'O']なら{'H': 0, 'O': 1})
        self.num_species = len(self.species_list) # 元素の種類数

        self.species_pair_indices_list: List[Tuple[int, int]] = [] # 元素ペアのインデックスリスト
        self.species_pair_to_channel_idx: Dict[Tuple[int, int], int] = {} # 元素ペアからチャネルインデックスへのマッピング。例: (0, 1) -> 0
        for first in range(self.num_species):
            for second in range(first, self.num_species):
                channel = len(self.species_pair_indices_list)
                pair = (first, second)
                self.species_pair_indices_list.append(pair)
                self.species_pair_to_channel_idx[pair] = channel
        self.num_species_pairs = len(self.species_pair_indices_list) # 元素ペアの総数

        self.num_g2_base_params = len(self.g2_eta_t) * len(self.g2_rs_t) # G2 の基本パラメータ数
        self.num_g3_base_params = (
            len(self.g3_eta_t) * len(self.g3_lam_t) * len(self.g3_zet_t)
        ) # G3 の基本パラメータ数
        self.g2_dim_total = self.num_g2_base_params * self.num_species_pairs # G2 の総次元数
        self.g3_dim_total = self.num_g3_base_params * self.num_species_pairs # G3 の総次元数
        self.total_sf_dim = self.g2_dim_total + self.g3_dim_total # 総対称関数次元数

        self.species_pair_lookup_matrix = torch.full(
            (self.num_species, self.num_species), -1, dtype=torch.long
        ) # 元素ペアのチャネルインデックスを格納するルックアップ行列
        for pair, channel in self.species_pair_to_channel_idx.items():
            first, second = pair
            self.species_pair_lookup_matrix[first, second] = channel
            self.species_pair_lookup_matrix[second, first] = channel

        # 既存の推論・外挿診断コードと同じマッピングを公開する。
        self.sf_index_map: List[str] = [] # 対称関数インデックスのマッピング(例: "G2(H-O): eta=0.10, rs=0.50")
        for first, second in self.species_pair_indices_list:
            first_name = self.species_list[first]
            second_name = self.species_list[second]
            for eta in self.g2_eta_t:
                for rs in self.g2_rs_t:
                    self.sf_index_map.append(
                        f"G2({first_name}-{second_name}): "
                        f"eta={eta.item():.2f}, rs={rs.item():.2f}"
                    )
        for first, second in self.species_pair_indices_list:
            first_name = self.species_list[first]
            second_name = self.species_list[second]
            for eta in self.g3_eta_t:
                for lam in self.g3_lam_t:
                    for zet in self.g3_zet_t:
                        self.sf_index_map.append(
                            f"G3({first_name}-{second_name}): "
                            f"eta={eta.item():.2f}, lam={lam.item():.1f}, "
                            f"zet={zet.item():.1f}"
                        )

    def _translation_ranges(self, cell: Tensor) -> Tuple[int, int, int]:
        """
        カットオフ球と交差し得る格子並進範囲を逆格子から求める。

        ``||inv(cell)[:, i]||`` の逆数は i 方向のセル面間距離である。(逆格子ベクトルg_iの長さの逆数が、対応するセル面間距離に相当する)
        原子を中心セルへ wrap した後は、以下の範囲ですべての周期像を
        必ず含められる。非直交セルでも格子ベクトル長だけに依存しない。
        """
        with torch.no_grad():
            search_cell = cell.detach()
            determinant = torch.linalg.det(search_cell)
            if not torch.isfinite(determinant) or determinant.abs() <= self.eps:
                raise ValueError("cell must be finite and non-singular")
            inverse_cell = torch.linalg.inv(search_cell) # 逆格子ベクトルを列ベクトルとして持つ行列
            reciprocal_norms = torch.linalg.vector_norm(inverse_cell, dim=0) # 各逆格子ベクトルの長さ
            limits = torch.ceil(self.r_cut * reciprocal_norms).to(torch.long) # 各方向の格子シフトの上限値(整数) (カットオフ球と交差する範囲)
            limits = torch.clamp(limits, min=1).cpu().tolist() # 各方向の格子シフトの下限値を1以上に制限してリスト化。シフト範囲を少なくとも1以上にすることで、カットオフ球と交差する範囲を確保する。
        return int(limits[0]), int(limits[1]), int(limits[2]) # 各方向の格子シフトの上限値を返す

    def _lattice_translations(self, cell: Tensor) -> Tensor:
        """探索対象の整数格子シフトを ``(M, 3)`` で返す。"""
        limit_a, limit_b, limit_c = self._translation_ranges(cell)
        device = cell.device
        range_a = torch.arange(-limit_a, limit_a + 1, device=device)
        range_b = torch.arange(-limit_b, limit_b + 1, device=device)
        range_c = torch.arange(-limit_c, limit_c + 1, device=device)
        return torch.cartesian_prod(range_a, range_b, range_c).to(torch.long) # 各方向の格子シフト(各方向にどれだけ移動するか)の組み合わせを返す

    def _get_neighbors_periodic(
        self, R: Tensor, cell: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        カットオフ内の全周期像を列挙する。

        自己相互作用として除外するのは ``i == j`` かつ格子シフトが
        ``(0, 0, 0)`` の場合だけである。自己原子の非ゼロ周期像は、
        カットオフ内なら通常の近傍として含める。
        """
        num_atoms = R.size(0)
        device = R.device
        dtype = R.dtype
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_float = torch.empty(0, dtype=dtype, device=device)
        empty_vectors = torch.empty((0, 3), dtype=dtype, device=device)
        if num_atoms == 0:
            return (
                empty_long,
                empty_long,
                empty_float,
                empty_vectors,
                empty_float,
            )

        # 原子がセル外表現でも同じ結果となるよう、探索前に中心セルへ wrap
        # する。floor は離散的なので detached 座標で決め、選ばれた近傍の
        # 距離だけを後段で再計算して自動微分可能にする。
        with torch.no_grad():
            search_cell = cell.detach()
            fractional = R.detach() @ torch.linalg.inv(search_cell)
            atom_wrap_shifts = torch.floor(fractional).to(torch.long) # 各原子のセル内ラップ(0~1に収める)に必要な整数シフト量 (各方向に何セル移動するか)
            translations = self._lattice_translations(search_cell)
            wrapped_search = R.detach() - atom_wrap_shifts.to(dtype) @ search_cell # 中心セル内に原子をラップした座標

            idx_i_parts: List[Tensor] = [] # 各格子シフトに対する原子ペアのiインデックスの部分リスト
            idx_j_parts: List[Tensor] = [] # 各格子シフトに対する原子ペアのjインデックスの部分リスト
            image_shift_parts: List[Tensor] = [] # 各格子シフトに対する原子ペアの格子シフトベクトルの部分リスト
            diagonal = torch.eye(num_atoms, dtype=torch.bool, device=device)

            for translation in translations: # 各格子シフトベクトルに対して近傍原子ペアを探索
                translation_cart = translation.to(dtype) @ search_cell
                diff_all = (
                    wrapped_search.unsqueeze(1)
                    - wrapped_search.unsqueeze(0)
                    - translation_cart
                ) # 各原子ペアの差ベクトル
                distance_sq = torch.sum(diff_all.square(), dim=2) # 各原子ペアの距離の二乗。shape=(num_atoms, num_atoms)
                mask = distance_sq < self.r_cut ** 2
                if bool(torch.all(translation == 0)): # 中心セル内に対する処理。自己原子ペアを除外する。
                    mask &= ~diagonal # 対角成分が0, 非対角成分が1のマスクと、mask(距離がカットオフ内ならTrue=0)を組み合わせて、自己原子ペアを除外する。
                pair_i, pair_j = torch.where(mask) # カットオフ内の原子ペアのインデックスを取得。
                if pair_i.numel() == 0:
                    continue
                idx_i_parts.append(pair_i)
                idx_j_parts.append(pair_j)
                image_shift_parts.append(
                    translation.expand(pair_i.numel(), 3).clone()
                )

            if not idx_i_parts:
                return (
                    empty_long,
                    empty_long,
                    empty_float,
                    empty_vectors,
                    empty_float,
                )

            idx_i = torch.cat(idx_i_parts) # 各格子シフトに対する原子ペアのiインデックスを結合
            idx_j = torch.cat(idx_j_parts) # 各格子シフトに対する原子ペアのjインデックスを結合
            image_shifts = torch.cat(image_shift_parts) # 各格子シフトに対する原子ペアの格子シフトベクトルを結合

            # G3候補生成が中心原子ごとの連続領域を仮定するため安定ソート。
            order = torch.argsort(idx_i, stable=True) # idx_iを昇順にソートし、インデックスの順序を取得
            idx_i = idx_i[order] # ソート後のidx_i
            idx_j = idx_j[order] # ソート後のidx_j
            image_shifts = image_shifts[order] # ソート後の格子シフトベクトル

        wrapped_R = R - atom_wrap_shifts.to(dtype) @ cell
        diff_ij = (
            wrapped_R[idx_i]
            - wrapped_R[idx_j]
            - image_shifts.to(dtype) @ cell
        )
        distance_sq = torch.sum(diff_ij.square(), dim=1)
        distances = torch.sqrt(distance_sq + self.eps)
        cutoff_values = _cutoff(distances, self.r_cut)
        return idx_i, idx_j, distances, diff_ij, cutoff_values

    @staticmethod
    def _build_triplet_pair_indices(idx_i: Tensor) -> Tuple[Tensor, Tensor]:
        """同じ中心原子を持つ異なる近傍像の組合せを作る。"""
        device = idx_i.device
        if idx_i.numel() < 2:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        neighbor_counts = torch.unique_consecutive(
            idx_i, return_counts=True
        )[1].tolist()
        num_triplets = sum(count * (count - 1) // 2 for count in neighbor_counts)
        pair1_indices = torch.empty(num_triplets, dtype=torch.long, device=device)
        pair2_indices = torch.empty_like(pair1_indices)

        edge_offset = 0
        triplet_offset = 0
        for count_value in neighbor_counts:
            count = int(count_value)
            count_triplets = count * (count - 1) // 2
            if count_triplets:
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
        """``R, Z, cell`` から shape ``(N, total_sf_dim)`` の G を返す。"""
        num_atoms = R_global.size(0)
        device = R_global.device
        dtype = R_global.dtype
        if num_atoms == 0:
            return torch.zeros(
                (0, self.total_sf_dim), device=device, dtype=dtype
            )
        if R_global.ndim != 2 or R_global.shape[1] != 3:
            raise ValueError("R_global must have shape (N, 3)")
        if Z.ndim != 1 or Z.shape[0] != num_atoms:
            raise ValueError("Z must have shape (N,)")
        if cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")

        species_indices = _get_species_indices(
            Z, self.species_map, self.z2element_map
        )
        idx_i, idx_j, R_ij, diff_ij, fc_ij = self._get_neighbors_periodic(
            R_global, cell
        )
        num_pairs = idx_i.size(0)

        G2 = torch.zeros(
            (num_atoms, self.g2_dim_total), device=device, dtype=dtype
        )
        if num_pairs:
            g2_eta = self.g2_eta_t.to(device=device, dtype=dtype)
            g2_rs = self.g2_rs_t.to(device=device, dtype=dtype)
            radial = torch.exp(
                -g2_eta.view(1, -1, 1)
                * (R_ij.view(-1, 1, 1) - g2_rs.view(1, 1, -1)).square()
            ).reshape(num_pairs, -1)
            g2_values = radial * fc_ij.unsqueeze(-1)

            lookup = self.species_pair_lookup_matrix.to(device)
            pair_channels = lookup[species_indices[idx_i], species_indices[idx_j]]
            base_indices = torch.arange(
                self.num_g2_base_params, device=device
            ).unsqueeze(0)
            target_columns = (
                pair_channels.unsqueeze(1) * self.num_g2_base_params
                + base_indices
            )
            G2.index_put_(
                (
                    idx_i.unsqueeze(1).expand_as(g2_values),
                    target_columns,
                ),
                g2_values,
                accumulate=True,
            )

        G3 = torch.zeros(
            (num_atoms, self.g3_dim_total), device=device, dtype=dtype
        )
        if num_pairs >= 2:
            pair1_indices, pair2_indices = self._build_triplet_pair_indices(idx_i)
            if pair1_indices.numel():
                g3_eta = self.g3_eta_t.to(device=device, dtype=dtype)
                g3_lam = self.g3_lam_t.to(device=device, dtype=dtype)
                g3_zet = self.g3_zet_t.to(device=device, dtype=dtype)
                power_of_2 = torch.pow(2.0, 1.0 - g3_zet)
                lookup = self.species_pair_lookup_matrix.to(device)
                base_indices = torch.arange(
                    self.num_g3_base_params, device=device
                ).unsqueeze(0)

                for chunk_start in range(
                    0, pair1_indices.numel(), self._G3_TRIPLET_CHUNK_SIZE
                ):
                    chunk_end = min(
                        chunk_start + self._G3_TRIPLET_CHUNK_SIZE,
                        pair1_indices.numel(),
                    )
                    pair1 = pair1_indices[chunk_start:chunk_end]
                    pair2 = pair2_indices[chunk_start:chunk_end]
                    diff_first = diff_ij[pair1]
                    diff_second = diff_ij[pair2]
                    diff_jk = diff_second - diff_first
                    R_jk_sq = torch.sum(diff_jk.square(), dim=1)
                    valid = torch.where(R_jk_sq < self.r_cut ** 2)[0]
                    if valid.numel() == 0:
                        continue

                    pair1 = pair1[valid]
                    pair2 = pair2[valid]
                    diff_first = diff_first[valid]
                    diff_second = diff_second[valid]
                    R_jk_sq = R_jk_sq[valid]
                    R_first = R_ij[pair1]
                    R_second = R_ij[pair2]
                    R_jk = torch.sqrt(R_jk_sq + self.eps)

                    cosine = torch.sum(diff_first * diff_second, dim=1) / (
                        R_first * R_second + self.eps
                    )
                    cosine = torch.clamp(
                        cosine, -1.0 + self.eps, 1.0 - self.eps
                    )
                    sum_squared_distances = (
                        R_first.square() + R_second.square() + R_jk_sq
                    )
                    radial = torch.exp(
                        -g3_eta.view(1, -1)
                        * sum_squared_distances.view(-1, 1)
                    )
                    angular_base = (
                        1.0 + g3_lam.view(1, -1) * cosine.view(-1, 1)
                    )
                    angular = angular_base.unsqueeze(-1).pow(
                        g3_zet.view(1, 1, -1)
                    )
                    values_without_cutoff = (
                        radial.unsqueeze(-1).unsqueeze(-1)
                        * angular.unsqueeze(1)
                        * power_of_2.view(1, 1, 1, -1)
                    )
                    combined_cutoff = (
                        fc_ij[pair1]
                        * fc_ij[pair2]
                        * _cutoff(R_jk, self.r_cut)
                    )
                    g3_values = (
                        values_without_cutoff
                        * combined_cutoff.view(-1, 1, 1, 1)
                    ).reshape(valid.numel(), -1)

                    central_indices = idx_i[pair1]
                    neighbor1_species = species_indices[idx_j[pair1]]
                    neighbor2_species = species_indices[idx_j[pair2]]
                    channels = lookup[neighbor1_species, neighbor2_species]
                    target_columns = (
                        channels.unsqueeze(1) * self.num_g3_base_params
                        + base_indices
                    )
                    G3.index_put_(
                        (
                            central_indices.unsqueeze(1).expand_as(g3_values),
                            target_columns,
                        ),
                        g3_values,
                        accumulate=True,
                    )

        return torch.cat([G2, G3], dim=1)


# 説明的な名前でも import できるようにする。
PeriodicSymmetryCalculator = SymmetryCalculator
