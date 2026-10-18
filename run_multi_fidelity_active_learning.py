"""
Master Multi-Fidelity Closed-Loop Active Learning Experiment.
Fuses Low-Fidelity GNN Screening (s=0.0) with High-Fidelity Molecular Dynamics (s=1.0)
using BoTorch Multi-Fidelity Gaussian Processes.
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.features.tokenizer import PolymerTokenizer
from src.features.featurizer import smiles_to_graph
from src.models.polymer_vae import PolymerVAE
from src.models.gnn_surrogate import PolymerGNN
from src.physics.chain_builder import PolymerChainBuilder
from src.physics.amorphous_cell import AmorphousCellBuilder
from src.physics.md_engine import PolymerMDEngine
from src.models.multi_fidelity_opt import MultiFidelityPolymerOptimizer
from src.utils.chemistry import is_valid_polymer, compute_synthetic_accessibility

def main():
    parser = argparse.ArgumentParser(description="PolymerGenesis Multi-Fidelity Active Learning")
    parser.add_argument("--target_tg", type=float, default=560.0, help="Target Tg in Kelvin")
    parser.add_argument("--low_fid_samples", type=int, default=12, help="Initial low-fidelity GNN queries")
    parser.add_argument("--high_fid_rounds", type=int, default=3, help="Active high-fidelity MD rounds")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cpu")

    print("\n==================================================================")
    print("[*] Launching Multi-Fidelity Closed-Loop Active Learning")
    print(f"[*] Target Glass Transition (Tg): {args.target_tg:.1f} K")
    print(f"[*] Fusing Low-Fidelity GNN (s=0.0) + High-Fidelity 3D MD (s=1.0)")
    print("==================================================================\n")

    # 1. Load VAE and GNN
    tokenizer = PolymerTokenizer()
    vae = PolymerVAE(vocab_size=tokenizer.vocab_size, embed_dim=64, hidden_dim=128, latent_dim=64)
    vae_path = os.path.join(results_dir, "polymer_vae_best.pt")
    vae.load_state_dict(torch.load(vae_path, map_location=device, weights_only=True))
    vae.eval()

    sample_g = smiles_to_graph("[*]CC([*])c1ccccc1")
    gnn = PolymerGNN(in_channels=sample_g.x.shape[1], hidden_dim=128, num_layers=4, dropout=0.2)
    gnn_path = os.path.join(results_dir, "gnn_tg_best.pt")
    gnn.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=True))
    gnn.eval()

    y_mean = 545.23
    y_std = 49.41

    optimizer = MultiFidelityPolymerOptimizer(latent_dim=64, bounds_range=3.0)

    # Helper evaluation functions
    def eval_low_fidelity(smiles):
        """Low-fidelity evaluation via PolymerGNN (1 ms)."""
        g = smiles_to_graph(smiles)
        if g is None or g.x.shape[0] == 0:
            return None
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long)
        with torch.no_grad():
            mu_norm, _, _, _ = gnn.predict_monte_carlo(g.x, g.edge_index, batch_idx, num_samples=10)
            return float(mu_norm[0] * y_std + y_mean)

    def eval_high_fidelity(smiles):
        """High-fidelity evaluation via Automated 3D Molecular Dynamics."""
        try:
            builder = PolymerChainBuilder(n_units=4)
            chain_mol = builder.build_chain(smiles)
            mol_3d = builder.generate_3d_conformer(chain_mol)
            cell_builder = AmorphousCellBuilder(target_density=1.0)
            box_mol, L, _ = cell_builder.pack_cell(mol_3d, num_chains=3)
            engine = PolymerMDEngine(box_mol, box_length=L)
            _, in_silico_tg = engine.run_thermal_annealing(
                temp_start=620.0,
                temp_end=300.0,
                temp_step=60.0,
                num_sweeps=25
            )
            return float(in_silico_tg)
        except Exception:
            return None

    # Step 1: Initial Low-Fidelity Exploration (s = 0.0)
    print(f"[Phase 1] Seeding {args.low_fid_samples} Low-Fidelity GNN Evaluations (s = 0.0)...")
    train_X_list = []
    train_Y_list = []
    experiment_log = []

    seed_z = torch.randn(args.low_fid_samples, 64)
    for i, z in enumerate(seed_z):
        s_list = vae.decode_latent(z.unsqueeze(0), tokenizer, temperature=0.7)
        smiles = s_list[0]
        if is_valid_polymer(smiles):
            pred_tg = eval_low_fidelity(smiles)
            if pred_tg is not None:
                sa = compute_synthetic_accessibility(smiles)
                dist = abs(pred_tg - args.target_tg)
                reward = - (dist / (y_std * 0.5))**2 - 0.2 * (sa - 1.0)
                
                # Append [z, fidelity=0.0]
                x_entry = torch.cat([z, torch.tensor([0.0])])
                train_X_list.append(x_entry)
                train_Y_list.append(reward)

                experiment_log.append({
                    "round": 0,
                    "fidelity": 0.0,
                    "fidelity_label": "GNN (Low)",
                    "smiles": smiles,
                    "measured_tg": pred_tg,
                    "target_error": dist,
                    "reward": float(reward)
                })
                print(f"  [GNN s=0.0] {smiles[:35]}... -> Pred Tg: {pred_tg:.1f} K (Error: {dist:.1f} K)")

    # Step 2: Multi-Fidelity Closed-Loop Active Learning (s = 1.0)
    print(f"\n[Phase 2] Multi-Fidelity Bayesian Active Learning ({args.high_fid_rounds} High-Fidelity MD Rounds)...")
    train_X = torch.stack(train_X_list)
    train_Y = torch.tensor(train_Y_list, dtype=torch.float).unsqueeze(-1)

    for round_idx in range(1, args.high_fid_rounds + 1):
        print(f"\n--- Round {round_idx}/{args.high_fid_rounds}: Optimizing Acquisition for s = 1.0 (MD) ---")
        # Propose candidates conditioned on target high-fidelity s = 1.0
        candidate_batch = optimizer.suggest_candidates(
            train_X, train_Y, q=3, beta=2.5, target_fidelity=1.0
        )

        for cand in candidate_batch:
            z_cand = cand[:64]
            s_cand = vae.decode_latent(z_cand.unsqueeze(0), tokenizer, temperature=0.65)[0]
            
            if is_valid_polymer(s_cand):
                print(f"  [Acquisition Proposed] {s_cand[:35]}...")
                print(f"  -> Triggering Automated 3D Molecular Dynamics Physics Verification...")
                md_tg = eval_high_fidelity(s_cand)
                
                if md_tg is not None:
                    sa = compute_synthetic_accessibility(s_cand)
                    dist = abs(md_tg - args.target_tg)
                    reward_md = - (dist / (y_std * 0.5))**2 - 0.2 * (sa - 1.0)

                    # Append [z_cand, fidelity=1.0] to multi-fidelity dataset
                    train_X = torch.cat([train_X, cand.unsqueeze(0)], dim=0)
                    train_Y = torch.cat([train_Y, torch.tensor([[reward_md]], dtype=torch.float)], dim=0)

                    experiment_log.append({
                        "round": round_idx,
                        "fidelity": 1.0,
                        "fidelity_label": "MD (High)",
                        "smiles": s_cand,
                        "measured_tg": md_tg,
                        "target_error": dist,
                        "reward": float(reward_md)
                    })
                    print(f"  [+] In-Silico MD Simulated Tg: {md_tg:.1f} K (Target Error: {dist:.1f} K) | Reward: {reward_md:.2f}")
                    break # Successfully completed high-fidelity run for this round

    # Save Multi-Fidelity History
    log_path = os.path.join(results_dir, "multi_fidelity_history.json")
    with open(log_path, "w") as f:
        json.dump(experiment_log, f, indent=2)

    # Plot Multi-Fidelity Pareto Frontier
    fig, ax = plt.subplots(figsize=(8, 5))
    low_errs = [e["target_error"] for e in experiment_log if e["fidelity"] == 0.0]
    high_errs = [e["target_error"] for e in experiment_log if e["fidelity"] == 1.0]

    ax.scatter(range(len(low_errs)), low_errs, color='steelblue', s=60, alpha=0.7, label='Low-Fidelity GNN (s=0.0)')
    ax.scatter(
        range(len(low_errs), len(low_errs) + len(high_errs)),
        high_errs,
        color='crimson',
        s=120,
        marker='*',
        label='High-Fidelity MD Simulation (s=1.0)'
    )
    ax.axhline(0.0, color='black', linestyle='--', alpha=0.5, label='Target Parity')
    ax.set_title(f"Multi-Fidelity Active Learning Convergence\nTarget $T_g$ = {args.target_tg:.1f} K (GNN Surrogate vs In-Silico MD)", fontsize=11)
    ax.set_xlabel("Cumulative Acquisition Step")
    ax.set_ylabel("Absolute Error to Target (K)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(results_dir, "multi_fidelity_active_learning.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[Done] Multi-Fidelity Active Learning plot saved to: {plot_path}")
    print(f"[Done] Experiment history saved to: {log_path}")

if __name__ == "__main__":
    main()
