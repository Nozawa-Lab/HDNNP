#!/usr/bin/env python3
"""Extract POSCAR structures into inference-ready .npz files."""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from ase.io import read

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from hdnnp import config as hdnnp_config
except ImportError:
    print(
        "Error: Could not import hdnnp.config. Ensure the script is run from "
        "the project root or hdnnp is in PYTHONPATH.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract POSCAR coordinates into inference-ready .npz files"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="POSCAR file or directory containing POSCAR files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write .npz files",
    )
    parser.add_argument(
        "--npz-format",
        type=str,
        default=hdnnp_config.NPZ_FILENAME_FORMAT,
        help=(
            "format string for output names; supports {prefix} and {frame_num} "
            f"(default: '{hdnnp_config.NPZ_FILENAME_FORMAT}')"
        ),
    )
    return parser.parse_args()


def find_poscars(input_path: Path) -> List[Path]:
    """Return one POSCAR file or all files named POSCAR under a directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(path for path in input_path.rglob("POSCAR") if path.is_file())
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def get_prefix(poscar_path: Path, input_path: Path) -> str:
    """Build a flat, filesystem-safe name prefix from the structure directory."""
    if input_path.is_file():
        prefix_path = Path(poscar_path.parent.name)
    else:
        relative_parent = poscar_path.parent.relative_to(input_path)
        prefix_path = relative_parent if relative_parent.parts else Path(input_path.name)

    prefix = "_".join(prefix_path.parts)
    prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("._-")
    return prefix or "structure"


def extract_poscar(poscar_path: Path, output_path: Path) -> None:
    """Read a POSCAR and save its geometry using the standard dataset keys."""
    atoms = read(str(poscar_path), format="vasp")
    dtype = np.float32 if hdnnp_config.DEFAULT_TORCH_DTYPE == torch.float32 else np.float64

    positions = np.asarray(atoms.get_positions(), dtype=dtype)
    atomic_numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int16)
    cell = np.asarray(atoms.cell.array, dtype=dtype)
    energy = np.array(np.nan, dtype=dtype)
    forces = np.full((len(atoms), 3), np.nan, dtype=dtype)

    np.savez_compressed(
        output_path,
        R=positions,
        Z=atomic_numbers,
        cell=cell,
        E=energy,
        F=forces,
    )


def main() -> int:
    args = parse_args()
    try:
        poscars = find_poscars(args.input)
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1

    if not poscars:
        print(f"[ERROR] No POSCAR files found under: {args.input}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    saved = 0

    for poscar_path in poscars:
        try:
            filename = args.npz_format.format(
                prefix=get_prefix(poscar_path, args.input),
                frame_num=0,
            )
        except (KeyError, ValueError, IndexError) as error:
            print(f"[ERROR] Invalid --npz-format '{args.npz_format}': {error}", file=sys.stderr)
            return 1

        output_path = args.output_dir / filename
        if output_path.exists():
            print(f"[ERROR] Output already exists; refusing to overwrite: {output_path}", file=sys.stderr)
            failures += 1
            continue

        try:
            extract_poscar(poscar_path, output_path)
        except Exception as error:
            print(f"[ERROR] Failed to extract {poscar_path}: {error}", file=sys.stderr)
            failures += 1
            continue

        print(f"  saved => {output_path}")
        saved += 1

    print(f"Finished: saved {saved} / {len(poscars)} structures to {args.output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
