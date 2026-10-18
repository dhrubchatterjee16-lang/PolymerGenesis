"""
Master Molecular Dynamics In-Silico Physics Verification Script (Milestone 3).
1. Polymerizes a generated monomer repeat unit into a 3D macromolecular chain.
2. Constructs a 3D periodic amorphous bulk simulation cell.
3. Executes Thermal Annealing Langevin dynamics to extract in-silico Glass Transition (Tg).
4. Executes Uniaxial Tensile Deformation to extract Young's Modulus and Yield Stress.
5. Generates publication-ready physics diagnostic plots and saves 3D XYZ coordinates.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write

from src.physics.chain_builder import PolymerChainBuilder
from src.physics.amorphous_cell import AmorphousCellBuilder
from src.physics.md_engine import PolymerMDEngine
from src.physics.stress_strain import PolymerTensileEngine

def main():
    parser = argparse.ArgumentParser(description="PolymerGenesis Milestone 3 MD Physics Verification")
    parser.add_argument(
        "--smiles",
        type=str,
        default="[*]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCOC(=O)CCCCCC(=O)N[*]",
        help="Polymer repeat unit SMILES with [*] handles"
    )
    parser.add_argument("--units", type=int, default=6, help="Monomers per polymer chain")
    parser.add_argument("--chains", type=int, default=4, help="Number of polymer chains in periodic box")
    parser.add_argument("--temp_start", type=float, default=650.0, help="Annealing starting temperature (K)")
    parser.add_argument("--temp_end", type=float, default=250.0, help="Annealing ending temperature (K)")
    parser.add_argument("--temp_step", type=float, default=50.0, help="Temperature step (K)")
    parser.add_argument("--md_steps", type=int, default=120, help="MD steps per temperature")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n==================================================================")
    print("[MD] Launching PolymerGenesis Automated Molecular Dynamics Engine")
    print(f"[*] Polymer SMILES: {args.smiles}")
    print(f"[*] Box Architecture: {args.chains} Chains x {args.units} Monomers per chain")
    print("==================================================================\n")

    # Step 1: Build 3D Polymer Chain
    print("[Step 1] Constructing 3D Polymer Chain & Energy Minimization...")
    builder = PolymerChainBuilder(n_units=args.units)
    oligomer_mol = builder.build_chain(args.smiles)
    mol_3d = builder.generate_3d_conformer(oligomer_mol)
    print(f"  Single Chain Atoms: {mol_3d.GetNumAtoms()} | Heavy Atoms: {oligomer_mol.GetNumHeavyAtoms()}")

    # Step 2: Pack 3D Amorphous Bulk Cell
    print("\n[Step 2] Packing 3D Periodic Bulk Amorphous Simulation Cell...")
    cell_builder = AmorphousCellBuilder(target_density=1.0)
    box_mol, L, all_positions = cell_builder.pack_cell(mol_3d, num_chains=args.chains)
    total_atoms = box_mol.GetNumAtoms()
    print(f"  Box Dimensions: {L:.2f} A x {L:.2f} A x {L:.2f} A")
    print(f"  Total Atoms in Cell: {total_atoms} | UFF Force Field Minimization: Complete")

    # Save 3D coordinates for visualizers
    xyz_path = os.path.join(results_dir, "amorphous_cell.xyz")
    cell_builder.save_xyz(box_mol, xyz_path)
    print(f"  Saved 3D Periodic Structure to: {xyz_path}")

    # Step 3: Run Thermal Annealing MD Simulation (In-Silico Tg)
    print("\n[Step 3] Running Thermal Annealing Simulation (Langevin Thermostat)...")
    md_engine = PolymerMDEngine(box_mol, box_length=L)
    trajectory, in_silico_tg = md_engine.run_thermal_annealing(
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        temp_step=args.temp_step,
        num_sweeps=args.md_steps
    )
    print(f"\n[+] In-Silico Glass Transition Discontinuity (MD Tg): {in_silico_tg:.1f} K")

    # Step 4: Run Uniaxial Tensile Mechanical Deformation
    print("\n[Step 4] Running Uniaxial Tensile Mechanical Deformation...")
    tensile_engine = PolymerTensileEngine(box_mol, box_length=L)
    strains, stresses_mpa, youngs_modulus_gpa, yield_stress_mpa = tensile_engine.run_tensile_test(
        max_strain=0.20,
        num_steps=8,
        relaxation_steps=35
    )
    print(f"\n[+] Simulated Young's Modulus (E): {youngs_modulus_gpa:.2f} GPa")
    print(f"[+] Simulated Yield Strength: {yield_stress_mpa:.2f} MPa")

    # Save JSON Physics Dossier
    dossier = {
        "polymer_smiles": args.smiles,
        "box_architecture": {
            "chains": args.chains,
            "monomers_per_chain": args.units,
            "total_atoms": total_atoms,
            "box_length_A": float(L)
        },
        "thermal_properties": {
            "in_silico_md_tg_K": float(in_silico_tg),
            "density_at_300K_g_cm3": float(trajectory[-2]["density_g_cm3"]),
            "trajectory": trajectory
        },
        "mechanical_properties": {
            "youngs_modulus_GPa": float(youngs_modulus_gpa),
            "yield_stress_MPa": float(yield_stress_mpa),
            "strains": strains.tolist(),
            "stresses_MPa": stresses_mpa.tolist()
        }
    }
    dossier_path = os.path.join(results_dir, "physics_dossier.json")
    with open(dossier_path, "w") as f:
        json.dump(dossier, f, indent=2)

    # Plot 1: Thermal Annealing V-T curve & Tg Knee
    print("\n--- Generating Physics Diagnostic Plots ---")
    fig, ax = plt.subplots(figsize=(7, 5))
    temps = np.array([d["temperature"] for d in trajectory])
    v_spec = np.array([d["specific_volume"] for d in trajectory])
    
    ax.plot(temps, v_spec, 'o-', color='navy', lw=2, markersize=7, label="MD Specific Volume $v(T)$")
    ax.axvline(in_silico_tg, color='red', linestyle='--', lw=2, label=f"In-Silico $T_g$ = {in_silico_tg:.1f} K")
    ax.set_title(f"Thermal Annealing & In-Silico $T_g$ Detection\nDiscovered Polymer (Target: 560 K, MD: {in_silico_tg:.1f} K)", fontsize=11)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Specific Volume ($cm^3/g$)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    tg_plot_path = os.path.join(results_dir, "md_tg_annealing.png")
    plt.tight_layout()
    plt.savefig(tg_plot_path, dpi=300)
    plt.close()

    # Plot 2: Simulated Stress-Strain Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(strains * 100, stresses_mpa, 's-', color='darkgreen', lw=2, markersize=6, label="MD Tensile Curve")
    ax.set_title(f"Simulated Uniaxial Stress-Strain Curve\n$E$ = {youngs_modulus_gpa:.2f} GPa, Yield Stress = {yield_stress_mpa:.1f} MPa", fontsize=11)
    ax.set_xlabel(r"Tensile Strain $\epsilon$ (%)")
    ax.set_ylabel(r"True Stress $\sigma$ (MPa)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    stress_plot_path = os.path.join(results_dir, "md_stress_strain.png")
    plt.tight_layout()
    plt.savefig(stress_plot_path, dpi=300)
    plt.close()

    print(f"[Done] Thermal Annealing Plot saved to: {tg_plot_path}")
    print(f"[Done] Stress-Strain Curve saved to: {stress_plot_path}")
    print(f"[Done] Complete Physics Dossier saved to: {dossier_path}")

if __name__ == "__main__":
    main()
