#!/usr/bin/env python3
# scripts/run_extract.py
"""
Extract VASP outputs (XDATCAR / OSZICAR / OUTCAR) into per‑frame .npz files
保持する配列:  R (N,3)  Z (N,)  cell (3,3)  E ()  F (N,3)
"""

import argparse
import re
import sys
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional # List, Tuple, Optional を追加

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
# from ase.io.vasp import read_vasp_out # コメントアウトのまま

# プロジェクトルートを解決し、sys.path に追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# hdnnp パッケージの config をインポート
try:
    from hdnnp import config as hdnnp_config
except ImportError:
    print("Error: Could not import hdnnp.config. Ensure the script is run from the project root or hdnnp is in PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def parse_args():
    # project_root は config から取得するので、ここでの定義は不要
    p = argparse.ArgumentParser(
        description="Extract VASP outputs into .npz (one file / frame)"
    )
    p.add_argument("-r", "--raw-dir", type=Path,
                   default=hdnnp_config.RAW_DATA_DIR, # config からデフォルト値を取得
                   help=f"directory containing VASP case folders (default from config: {hdnnp_config.RAW_DATA_DIR})")
    p.add_argument("-o", "--out-dir", type=Path,
                   default=hdnnp_config.PROCESSED_DATA_DIR, # config からデフォルト値を取得
                   help=f"directory to write .npz (default from config: {hdnnp_config.PROCESSED_DATA_DIR})")
    p.add_argument("--npz-format", type=str,
                   default=hdnnp_config.NPZ_FILENAME_FORMAT, # config からデフォルト値を取得
                   help=f"Format string for npz filenames (default from config: '{hdnnp_config.NPZ_FILENAME_FORMAT}')")
    return p.parse_args()


# --------------------------------------------------------------------
# OSZICAR → list[energy]
# --------------------------------------------------------------------
# ENERGY_PAT はこのファイル固有なので、ここに残す
ENERGY_PAT = re.compile(
    r"""
    (?:
        TOTEN\s+=\s*|  # VASP 5.x and later
        E0=\s*|        # VASP 4.x
        energy\s+without\s+entropy=\s* # Another common pattern
    )
    ([\-+0-9\.EeDd]+)   # 数値キャプチャ
    """, re.VERBOSE)


def parse_energies(osz_path: Path) -> list[float]:
    """ionic step ごとのエネルギー (eV) をリストで返す"""
    energies = []
    if not osz_path.exists():
        print(f"[WARN] OSZICAR file not found: {osz_path}", file=sys.stderr)
        return energies

    for line in osz_path.read_text().splitlines():
        m = ENERGY_PAT.search(line)
        if m:
            try:
                energies.append(float(m.group(1).replace('D', 'E')))
            except ValueError:
                print(f"[WARN] Could not parse energy value: {m.group(1)} in {osz_path}", file=sys.stderr)
    if not energies:
        print(f"[INFO] No energies found with pattern in {osz_path}", file=sys.stderr)
    return energies


# --------------------------------------------------------------------
# XDATCAR → cell, Z, list[fractional‑coords]
# --------------------------------------------------------------------
def parse_xdatcar(xdat_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[np.ndarray]]:
    """XDATCAR を読み込み (cell, Z, frac_frames) を返す。失敗時は (None, None, [])"""
    if not xdat_path.exists():
        print(f"[ERROR][XDATCAR] File not found: {xdat_path}", file=sys.stderr)
        return None, None, []

    lines = [l.rstrip() for l in xdat_path.read_text().splitlines()]
    if len(lines) < 7:
        print(f"[ERROR][XDATCAR] File is too short (less than 7 lines): {xdat_path}", file=sys.stderr)
        return None, None, []

    try:
        scale_factor_str = lines[1].split()
        if not scale_factor_str: raise ValueError("Scale factor line is empty.")
        scale_factor = float(scale_factor_str[0])
        
        cell_lines = [lines[i].split() for i in (2, 3, 4)]
        if not all(len(cl) == 3 for cl in cell_lines): raise ValueError("Cell matrix lines are malformed.")
        cell = np.array([[float(c) for c in cl] for cl in cell_lines], dtype=np.float32) * scale_factor
        
        species_symbols = lines[5].split()
        atom_counts_str = lines[6].split()

        if not species_symbols or not atom_counts_str:
            raise ValueError("Species symbols or counts line is empty.")
        atom_counts = list(map(int, atom_counts_str))


        if len(species_symbols) != len(atom_counts):
            raise ValueError(f"Mismatch between species symbols ({len(species_symbols)}) and counts ({len(atom_counts)}).")

        Z_list = []
        for sym, count in zip(species_symbols, atom_counts):
            if sym not in atomic_numbers: # ase.data.atomic_numbers
                raise ValueError(f"Unknown atomic symbol '{sym}'.")
            Z_list.extend([atomic_numbers[sym]] * count)
        Z_np = np.array(Z_list, dtype=np.int16) # Z を Z_np に変更

    except (ValueError, IndexError) as e:
        print(f"[ERROR][XDATCAR] Error parsing header of {xdat_path}: {e}", file=sys.stderr)
        return None, None, []

    N_atoms_from_header = sum(atom_counts)
    if N_atoms_from_header == 0:
        print(f"[ERROR][XDATCAR] Zero atoms specified in header of {xdat_path}", file=sys.stderr)
        return cell, Z_np, [] # Z_np を返す

    frac_frames: List[np.ndarray] = []
    current_line_idx = 7
    frame_count = 0
    while current_line_idx < len(lines):
        line_content = lines[current_line_idx].strip()
        # "Direct configuration=" または "Cartesian configuration=" を探す
        if line_content.lower().startswith("direct configuration=") or \
           line_content.lower().startswith("cartesian configuration="):
            frame_count += 1
            coord_block_start = current_line_idx + 1
            coord_block_end = coord_block_start + N_atoms_from_header
            if coord_block_end > len(lines):
                print(f"[WARN] [XDATCAR] Incomplete coordinate block for frame {frame_count} in {xdat_path}. Expected {N_atoms_from_header} atoms, found less. Stopping frame parsing.", file=sys.stderr)
                break
            
            try:
                frame_coords_str_list = [lines[j].split() for j in range(coord_block_start, coord_block_end)]
                # 各行が少なくとも3つの要素を持っているか確認
                if not all(len(fcs) >= 3 for fcs in frame_coords_str_list):
                    raise ValueError("Coordinate line has less than 3 elements.")
                
                frame_coords_str = [fcs[:3] for fcs in frame_coords_str_list] # 最初の3要素のみ取得
                frame_coords = np.array(frame_coords_str, dtype=np.float32)


                if frame_coords.shape != (N_atoms_from_header, 3):
                    print(f"[WARN] [XDATCAR] Coordinate block for frame {frame_count} in {xdat_path} has incorrect shape: {frame_coords.shape}. Expected ({N_atoms_from_header}, 3). Skipping frame.", file=sys.stderr)
                    current_line_idx = coord_block_end 
                    continue
                frac_frames.append(frame_coords)
                current_line_idx = coord_block_end
            except ValueError as e:
                print(f"[WARN] [XDATCAR] Error parsing coordinates for frame {frame_count} in {xdat_path}: {e}. Skipping frame.", file=sys.stderr)
                current_line_idx = coord_block_end # エラーがあっても次のフレームヘッダを探すために進む
                continue # このフレームはスキップ
        else:
            current_line_idx += 1

    if not frac_frames:
        print(f"[INFO] [XDATCAR] No valid coordinate frames found in {xdat_path}")

    return cell, Z_np, frac_frames # Z_np を返す


# --------------------------------------------------------------------
# OUTCAR 手動フォースパース (変更なし)
# --------------------------------------------------------------------
def parse_forces_manually_from_outcar(outcar_content: str, num_atoms: int, name_prefix: str = "") -> Union[np.ndarray, None]:
    """OUTCARの文字列から "TOTAL-FORCE" ブロックを直接パースする"""
    forces_list = []
    try:
        force_header_pattern = re.compile(r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)", re.IGNORECASE)
        
        last_match_start = -1
        for match in force_header_pattern.finditer(outcar_content):
            last_match_start = match.end()

        if last_match_start == -1:
            # print(f"[DEBUG] {name_prefix}: Manual force parse: 'TOTAL-FORCE' header not found.") # デバッグ用
            return None

        content_after_header = outcar_content[last_match_start:]
        lines = content_after_header.splitlines()
        
        data_started = False
        for line_idx, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped: 
                continue

            if "--------" in line_stripped: 
                if not data_started:
                    data_started = True 
                else: 
                    break 
                continue

            if data_started:
                parts = line_stripped.split()
                if len(parts) == 6: 
                    try:
                        fx, fy, fz = map(float, parts[3:6])
                        forces_list.append([fx, fy, fz])
                    except ValueError:
                        print(f"[WARN] {name_prefix}: Manual force parse: Could not parse force components from line: '{line_stripped}'", file=sys.stderr)
                        return None 
                else: 
                    # print(f"[DEBUG] {name_prefix}: Manual force parse: Unexpected line format in force block: '{line_stripped}'") # デバッグ用
                    if len(forces_list) > 0: 
                        break

            if len(forces_list) == num_atoms: 
                break
        
        if len(forces_list) == num_atoms:
            # print(f"[INFO] {name_prefix}: Manual force parse: Successfully parsed {len(forces_list)} forces.") # デバッグ用
            return np.array(forces_list, dtype=np.float32)
        else:
            print(f"[WARN] {name_prefix}: Manual force parse: Expected {num_atoms} forces, but found {len(forces_list)}.", file=sys.stderr)
            return None

    except Exception as e:
        print(f"[WARN] {name_prefix}: Manual force parse: Exception during manual force parsing: {e}", file=sys.stderr)
        return None

# --------------------------------------------------------------------
# メイン抽出
# --------------------------------------------------------------------
def extract_case(case_dir: Path, out_dir: Path, raw_dir_base: Path, npz_format: str): # npz_format を引数に追加
    try:
        relative_path_parts = case_dir.relative_to(raw_dir_base).parts
        name_prefix = "_".join(filter(None, relative_path_parts))
        if not name_prefix: 
            name_prefix = case_dir.name
    except ValueError: 
        name_prefix = "_".join(case_dir.parts[-2:]) 
        if not name_prefix:
            name_prefix = case_dir.name

    print(f"── Extracting {name_prefix} (from {case_dir.resolve()})")

    xdatcar_path = case_dir / "XDATCAR"
    oszicar_path = case_dir / "OSZICAR"
    outcar_path = case_dir / "OUTCAR"

    if not xdatcar_path.exists():
        print(f"[FAIL] {name_prefix}: XDATCAR not found. Skipping.", file=sys.stderr)
        return
    if not oszicar_path.exists():
        print(f"[FAIL] {name_prefix}: OSZICAR not found. Skipping.", file=sys.stderr)
        return
    if not outcar_path.exists():
        print(f"[FAIL] {name_prefix}: OUTCAR not found. Skipping.", file=sys.stderr)
        return

    cell, Z_atoms, frac_frames = parse_xdatcar(xdatcar_path) # Z_atoms に変更
    if cell is None or Z_atoms is None: # parse_xdatcar が失敗した場合
        print(f"[FAIL] {name_prefix}: Critical error parsing XDATCAR. Skipping.", file=sys.stderr)
        return
    
    N_atoms = len(Z_atoms) # Z_atoms を使用
    if N_atoms == 0:
        print(f"[FAIL] {name_prefix}: No atoms found (Z is empty) after parsing XDATCAR. Skipping.", file=sys.stderr)
        return
    
    T_xdat = len(frac_frames)
    if T_xdat == 0:
        print(f"[INFO] {name_prefix}: No frames found in XDATCAR. Skipping.")
        return

    E_list = parse_energies(oszicar_path)
    T_osz = len(E_list)
    if T_osz == 0 and T_xdat > 0 : 
        print(f"[WARN] {name_prefix}: No energies found in OSZICAR, but XDATCAR has {T_xdat} frames. Will attempt to use OUTCAR energies if available, or skip frames.")
    
    traj_outcar: List[Any] = [] # ASE Atoms object のリスト
    outcar_content_cache = "" 
    try:
        with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f_outcar: # errors='ignore' を追加
            outcar_content_cache = f_outcar.read()
        
        force_block_pattern = re.compile(r"POSITION\s+TOTAL-FORCE\s+\(eV/Angst\)", re.IGNORECASE) 
        force_block_count = len(force_block_pattern.findall(outcar_content_cache))
        
        # print(f"  Reading OUTCAR: {outcar_path} (Found {force_block_count} force blocks using regex)") # デバッグ用

        if force_block_count == 0:
            print(f"[WARN] {name_prefix}: No 'TOTAL-FORCE' section found in OUTCAR using regex. Forces will be missing for this file.")
        elif force_block_count == 1: 
            # print(f"[INFO] {name_prefix}: Single 'TOTAL-FORCE' section found. Attempting to read with ASE (index=-1).") # デバッグ用
            try:
                # ASEのreadはファイルパスを文字列で受け取る
                single_atoms_object = read(str(outcar_path), index=-1, format="vasp-out")
                if single_atoms_object:
                    try:
                        _ = single_atoms_object.get_forces(apply_constraint=False) # 互換性チェック
                        traj_outcar = [single_atoms_object]
                        # print(f"[INFO] {name_prefix}: Successfully read OUTCAR with ASE (index=-1) and get_forces() seems OK.") # デバッグ用
                    except Exception as e_get_forces_check:
                        print(f"[WARN] {name_prefix}: ASE read(index=-1) OK, but get_forces() failed: {e_get_forces_check}. Attempting manual parse.", file=sys.stderr)
                        pass # manual parse にフォールバック
                else: 
                    print(f"[WARN] {name_prefix}: ASE read(index=-1) returned None. Attempting manual parse.")
            except Exception as e_ase_single:
                print(f"[WARN] {name_prefix}: Error reading OUTCAR with ASE (index=-1): {e_ase_single}. Attempting manual parse.", file=sys.stderr)
        else: 
            read_index_str = f":{T_xdat}" # T_xdat フレームまで読むように修正
            # print(f"[INFO] {name_prefix}: Multiple ({force_block_count}) 'TOTAL-FORCE' sections. Reading up to {T_xdat} frames from OUTCAR (index='{read_index_str}').") # デバッグ用
            try:
                traj_outcar = list(read(str(outcar_path), index=read_index_str, format="vasp-out"))
            except Exception as e_ase_multi:
                 print(f"[WARN] {name_prefix}: Error reading OUTCAR with ASE (index='{read_index_str}'): {e_ase_multi}. Forces might be missing or incomplete.", file=sys.stderr)

    except Exception as e:
        print(f"[WARN] {name_prefix}: Error processing OUTCAR file {outcar_path}: {e}. Forces might be missing.", file=sys.stderr)
    
    T_outcar_actual = len(traj_outcar)

    if T_osz == 0 and T_outcar_actual > 0:
        print(f"[INFO] {name_prefix}: OSZICAR empty, attempting to use energies from OUTCAR frames.")
        try:
            E_list_from_outcar = [atoms.get_potential_energy() for atoms in traj_outcar]
            if len(E_list_from_outcar) == T_outcar_actual: # T_outcar_actual と比較
                E_list = E_list_from_outcar
                T_osz = T_outcar_actual 
                print(f"[INFO] {name_prefix}: Successfully extracted {T_osz} energies from OUTCAR.")
            else:
                print(f"[WARN] {name_prefix}: Could not extract consistent energies from all OUTCAR frames ({len(E_list_from_outcar)} vs {T_outcar_actual}). Energy data might be incomplete.")
        except Exception as e_get_energy:
            print(f"[WARN] {name_prefix}: Error getting potential energy from OUTCAR trajectory objects: {e_get_energy}")

    T = min(T_xdat, T_osz) 
    if T_outcar_actual > 0 : 
        T = min(T, T_outcar_actual)
    # else: # OUTCARからフレームが全く読めなかった場合
    #     if force_block_count > 0: # OUTCARに力のブロック自体はあった場合
    #         print(f"[WARN] {name_prefix}: Force blocks exist in OUTCAR but ASE could not read frames. Forces will be NaN.")
    #     # force_block_count == 0 の場合は既に警告済み

    if T == 0:
        print(f"[INFO] {name_prefix}: No common frames after all parsing attempts (XDATCAR:{T_xdat}, OSZICAR:{T_osz}, OUTCAR:{T_outcar_actual}). Skipping.")
        return
    
    frac_frames = frac_frames[:T]
    E_list = E_list[:T]
    if traj_outcar: 
        traj_outcar = traj_outcar[:T]

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_frames_count = 0
    
    manual_forces_for_frames: Optional[np.ndarray] = None
    if force_block_count == 1 and not traj_outcar and outcar_content_cache: 
        print(f"[INFO] {name_prefix}: Attempting final manual force parse for single block OUTCAR as ASE failed.")
        manual_forces_for_frames = parse_forces_manually_from_outcar(outcar_content_cache, N_atoms, name_prefix)
        if manual_forces_for_frames is None:
            print(f"[WARN] {name_prefix}: Final manual force parse failed. Forces will be NaN.")

    for i in range(T):
        R_cartesian = (frac_frames[i] @ cell).astype(np.float32) # cell は XDATCAR から取得したものを使い続ける
        E_total = np.array(E_list[i], dtype=np.float32)
        current_Z_frame = Z_atoms # XDATCAR から取得した Z_atoms を使用
        F_cartesian = np.full((N_atoms, 3), np.nan, dtype=np.float32) 

        if traj_outcar and i < len(traj_outcar): 
            try:
                atoms_obj = traj_outcar[i]
                # OUTCARの原子数とXDATCARの原子数が一致するか確認
                if len(atoms_obj) != N_atoms:
                    print(f"[WARN] {name_prefix} frame {i}: Atom count mismatch between OUTCAR ({len(atoms_obj)}) and XDATCAR ({N_atoms}). Storing NaNs for forces.", file=sys.stderr)
                else:
                    forces_raw = atoms_obj.get_forces(apply_constraint=False)
                    if forces_raw is not None and isinstance(forces_raw, np.ndarray) and forces_raw.shape == (N_atoms, 3):
                        F_cartesian = forces_raw.astype(np.float32)
                    elif forces_raw is not None: 
                        print(f"[WARN] {name_prefix} frame {i}: Forces from OUTCAR (ASE) have incorrect shape. Expected ({N_atoms}, 3), got {forces_raw.shape}. Storing NaNs.", file=sys.stderr)
            except Exception as e_get_forces:
                print(f"[WARN] {name_prefix} frame {i}: Error calling get_forces() via ASE: {e_get_forces}. Storing NaNs.", file=sys.stderr)
        elif manual_forces_for_frames is not None and i == 0 : # manual_forces は最初のフレームにのみ適用される想定 (単一ブロックOUTCARの場合)
            if manual_forces_for_frames.shape == (N_atoms, 3):
                F_cartesian = manual_forces_for_frames.astype(np.float32)
                # print(f"[INFO] {name_prefix} frame {i}: Used manually parsed forces.") # デバッグ用
            else:
                print(f"[WARN] {name_prefix} frame {i}: Manually parsed forces have incorrect shape. Expected ({N_atoms}, 3), got {manual_forces_for_frames.shape}. Storing NaNs.", file=sys.stderr)
        elif not traj_outcar and force_block_count == 0: 
             if i == 0: print(f"[INFO] {name_prefix}: No OUTCAR trajectory/force data available. Forces will be NaN.")
        elif not traj_outcar and force_block_count > 0 and manual_forces_for_frames is None: 
             if i == 0: print(f"[WARN] {name_prefix}: ASE and manual parsing of forces failed despite force blocks in OUTCAR. Forces will be NaN.")


        # npzファイル名を config から取得したフォーマットで生成
        npz_filename = npz_format.format(prefix=name_prefix, frame_num=i)
        npz_path = out_dir / npz_filename
        try:
            np.savez_compressed(npz_path, R=R_cartesian, Z=current_Z_frame, cell=cell, E=E_total, F=F_cartesian)
            if (i < 3 and saved_frames_count < 3) or (i > 0 and i % 50 == 0 and i < T -1) or i == T-1 : # ログ出力を調整
                 print(f"  saved => {npz_path.name}")
            saved_frames_count += 1
        except Exception as e:
            print(f"[FAIL] {name_prefix} frame {i}: Could not save npz file {npz_path}: {e}", file=sys.stderr)

    if saved_frames_count > 0:
        print(f"  Successfully processed '{name_prefix}': saved {saved_frames_count} / {T} frames to {out_dir.resolve()}")
    elif T_xdat > 0 : # T_xdat > 0 で T=0 になった場合も考慮
        print(f"  Processed '{name_prefix}': No frames were saved despite {T_xdat} frames in XDATCAR (T={T}, check warnings above).")


def main():
    args = parse_args()
    if not args.raw_dir.is_dir():
        sys.exit(f"[ERR] raw-dir not found: {args.raw_dir}")
    
    print(f"Starting extraction from: {args.raw_dir.resolve()}")
    print(f"Output directory: {args.out_dir.resolve()}")
    print(f"NPZ filename format: '{args.npz_format}'")


    for root_str, _, _ in os.walk(args.raw_dir):
        case_dir = Path(root_str)
        # XDATCAR, OSZICAR, OUTCAR が全て存在するディレクトリを処理対象とする
        if all((case_dir / fname).exists() for fname in ["XDATCAR", "OSZICAR", "OUTCAR"]):
            extract_case(case_dir, args.out_dir, args.raw_dir, args.npz_format) # npz_format を渡す

if __name__ == "__main__":
    main()