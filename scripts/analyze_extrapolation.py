#!/usr/bin/env python3
"""
scripts/analyze_extrapolation.py

Generates plots to visually inspect the learned potential energy surface
for each symmetry function component, helping to detect extrapolation.
"""
import sys
import json
from pathlib import Path
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Project Root and Imports ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from hdnnp.data_loader import get_dataloader
from hdnnp.symmetry_calculator import SymmetryCalculator
from hdnnp.model import HDNNPModel
from hdnnp import config as hdnnp_config

# --- ANSI Color Codes ---
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"

def log(message, level="INFO"):
    """Logs a message to the console with color-coding."""
    colors = {"INFO": GREEN, "DEBUG": CYAN, "WARN": YELLOW, "ERROR": RED}
    print(f"{colors.get(level, RESET)}[{level}]{RESET} {message}")

def update_config_from_json(config_path: Path):
    """Loads settings from a JSON file and updates the config module."""
    if not config_path.exists():
        log(f"Configuration file not found at {config_path}. Using default config.", level="WARN")
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        log(f"Loading and applying settings from {config_path}")
        for key, value in saved_config.items():
            if hasattr(hdnnp_config, key):
                if isinstance(getattr(hdnnp_config, key), Path):
                    setattr(hdnnp_config, key, Path(value))
                else:
                    setattr(hdnnp_config, key, value)
        log("Settings applied successfully for analysis.")
    except Exception as e:
        log(f"Error loading config from {config_path}: {e}. Using default config.", level="ERROR")


def get_all_atomic_data(dataloader, model, sym_calc, device):
    """
    Calculates descriptors (G) and predicted atomic energies for all atoms in a dataset.
    Returns a dictionary mapping element symbols to their respective G and E_atom tensors.
    """
    model.eval()
    all_data = {species: {'G': [], 'E_atom': []} for species in hdnnp_config.SPECIES}
    z2element = {int(k): v for k, v in hdnnp_config.Z2ELEMENT.items()}
    
    log(f"Processing {len(dataloader.dataset)} structures...")
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Calculating descriptors and energies"):
            R, Z, cell, n_atoms_batch = \
                data['R'].to(device), data['Z'].to(device), data['cell'].to(device), data['N_atoms']

            for i in range(len(R)):
                n_atoms = n_atoms_batch[i].item()
                if n_atoms == 0: continue
                
                R_i, Z_i, cell_i = R[i, :n_atoms], Z[i, :n_atoms], cell[i]
                
                G_i = sym_calc.compute(R_i, Z_i, cell_i)
                E_atoms_i = model(G_i, Z_i)
                
                # Assign data to the correct element
                for atom_idx in range(n_atoms):
                    z_val = int(Z_i[atom_idx].item())
                    element_symbol = z2element.get(z_val)
                    if element_symbol in all_data:
                        all_data[element_symbol]['G'].append(G_i[atom_idx].unsqueeze(0))
                        all_data[element_symbol]['E_atom'].append(E_atoms_i[atom_idx].unsqueeze(0))

    # Concatenate lists of tensors into single tensors
    for species in all_data:
        if all_data[species]['G']:
            all_data[species]['G'] = torch.cat(all_data[species]['G'], dim=0)
            all_data[species]['E_atom'] = torch.cat(all_data[species]['E_atom'], dim=0)
        else: # ensure they are tensors even if empty
            all_data[species]['G'] = torch.empty((0, sym_calc.total_sf_dim), device=device)
            all_data[species]['E_atom'] = torch.empty((0,), device=device)

    return all_data


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize descriptor space for extrapolation detection.")
    parser.add_argument('--checkpoint-dir', '-c', type=Path, required=True,
                        help="Path to the experiment's checkpoint directory.")
    args = parser.parse_args()

    # --- Environment Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"Using device: {device}")

    config_path = args.checkpoint_dir / "config_used.json"
    update_config_from_json(config_path)

    output_dir = args.checkpoint_dir / "extrapolation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Plots will be saved to: {output_dir}")

    # --- SymmetryCalculator and Model Initialization ---
    sym_calc = SymmetryCalculator()
    model = HDNNPModel(sym_calc.total_sf_dim).to(device)
    model_path = args.checkpoint_dir / hdnnp_config.TRAINED_MODEL_FILENAME
    if not model_path.exists():
        log(f"Trained model not found at {model_path}", level="ERROR")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    log("SymmetryCalculator and trained model initialized.")

    # --- Data Loading ---
    log("Loading train and test datasets...")
    splits_json_path = hdnnp_config.PROCESSED_DATA_DIR / hdnnp_config.SPLITS_FILENAME
    processed_dir_path = hdnnp_config.PROCESSED_DATA_DIR
    
    # Use a larger batch size for faster processing during analysis
    analysis_batch_size = 128
    
    train_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='train',
        batch_size=analysis_batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = get_dataloader(
        splits_json=splits_json_path, processed_dir=processed_dir_path, split='test',
        batch_size=analysis_batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # --- Calculate Descriptors and Energies for all data ---
    train_data_atomic = get_all_atomic_data(train_loader, model, sym_calc, device)
    test_data_atomic = get_all_atomic_data(test_loader, model, sym_calc, device)
    
    # --- Main Plotting Loop ---
    total_sf_dim = sym_calc.total_sf_dim
    for species in hdnnp_config.SPECIES:
        log(f"--- Generating plots for element: {species} ---")
        
        G_train = train_data_atomic[species]['G']
        E_atom_train = train_data_atomic[species]['E_atom']
        G_test = test_data_atomic[species]['G']
        E_atom_test = test_data_atomic[species]['E_atom']

        if G_train.shape[0] == 0:
            log(f"No training data found for {species}. Skipping.", level="WARN")
            continue

        species_plot_dir = output_dir / species
        species_plot_dir.mkdir(exist_ok=True)
        
        mean_g_vector = torch.mean(G_train, dim=0)

        for sf_idx in tqdm(range(total_sf_dim), desc=f"Plotting for {species}"):
            g_train_comp = G_train[:, sf_idx].cpu().numpy()
            e_train_comp = E_atom_train.cpu().numpy()
            g_test_comp = G_test[:, sf_idx].cpu().numpy()
            e_test_comp = E_atom_test.cpu().numpy()

            min_val, max_val = g_train_comp.min(), g_train_comp.max()
            if min_val == max_val:
                min_val -= 0.1
                max_val += 0.1

            x_curve = torch.linspace(min_val, max_val, 100, device=device)
            
            g_curve_batch = mean_g_vector.repeat(100, 1)
            g_curve_batch[:, sf_idx] = x_curve
            
            with torch.no_grad():
                element_nn = model.element_nns[species]
                y_curve = element_nn(g_curve_batch).cpu().numpy()

            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(10, 7))

            ax.scatter(g_train_comp, e_train_comp, alpha=0.3, s=20, label='Train Data', color='royalblue')
            if g_test_comp.size > 0:
                ax.scatter(g_test_comp, e_test_comp, alpha=0.6, s=30, label='Test Data', color='darkorange', marker='x')

            ax.plot(x_curve.cpu().numpy(), y_curve, color='crimson', linewidth=2.5, label='HDNNP Prediction')

            ax.set_xlabel(f'Symmetry Function Component {sf_idx}', fontsize=12)
            ax.set_ylabel('Predicted Atomic Energy (eV)', fontsize=12)
            ax.set_title(f'Energy vs. SF Component {sf_idx} for {species}', fontsize=14, fontweight='bold')
            ax.legend()
            
            ax.axvline(min_val, color='gray', linestyle='--', linewidth=1)
            ax.axvline(max_val, color='gray', linestyle='--', linewidth=1)

            fig.tight_layout()
            plot_filename = species_plot_dir / f"sf_comp_{sf_idx:03d}.png"
            plt.savefig(plot_filename, dpi=150)
            plt.close(fig)

    log("--- All plots generated successfully. ---")


if __name__ == '__main__':
    main()