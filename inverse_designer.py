"""
Inverse Generative Polymer Designer for PolymerGenesis (Milestone 2).
Performs:
1. Target-Driven Latent Space Optimization (BoTorch Bayesian Optimization).
2. Decoding candidate latent vectors to valid macromolecular repeat units (*).
3. In-silico property screening & uncertainty quantification using trained PolymerGNN.
4. Chemical validity, synthesizability (SAScore), and novelty filtering.
5. 2D chemical structure grid rendering of top-ranked polymer discoveries.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from rdkit.Chem import Draw

from src.features.tokenizer import PolymerTokenizer
from src.features.featurizer import smiles_to_graph, smiles_to_mol
from src.models.polymer_vae import PolymerVAE
from src.models.gnn_surrogate import PolymerGNN
from src.models.bayesian_opt import LatentBayesianOptimizer
from src.chemistry.reaction_grammar import enumerate_combinatorial_library
from src.utils.chemistry import is_valid_polymer, compute_synthetic_accessibility, compute_novelty, compute_tanimoto_similarity

class PolymerInverseDesigner:
    def __init__(self, property_name: str = "bandgap", device: str = None):
        self.prop_name = property_name.lower()
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        self.base_dir = os.path.dirname(__file__)
        self.results_dir = os.path.join(self.base_dir, "results")
        self.data_dir = os.path.join(self.base_dir, "data")
        
        # 1. Load Tokenizer & VAE
        self.tokenizer = PolymerTokenizer()
        self.vae = PolymerVAE(
            vocab_size=self.tokenizer.vocab_size,
            embed_dim=64,
            hidden_dim=128,
            latent_dim=64
        ).to(self.device)
        
        vae_weights = os.path.join(self.results_dir, "polymer_vae_best.pt")
        if not os.path.exists(vae_weights):
            raise FileNotFoundError(f"Missing VAE weights at {vae_weights}. Run train_vae.py first.")
        self.vae.load_state_dict(torch.load(vae_weights, map_location=self.device, weights_only=True))
        self.vae.eval()

        # 2. Load Property Surrogate (PolymerGNN)
        gnn_weights = os.path.join(self.results_dir, f"gnn_{self.prop_name}_best.pt")
        if not os.path.exists(gnn_weights):
            raise FileNotFoundError(f"Missing GNN weights at {gnn_weights}. Run train_and_benchmark.py first.")
        
        # Determine in_channels from a sample graph
        sample_g = smiles_to_graph("[*]C=C([*])c1ccccc1")
        in_channels = sample_g.x.shape[1]
        
        self.gnn = PolymerGNN(in_channels=in_channels, hidden_dim=128, num_layers=4, dropout=0.2).to(self.device)
        self.gnn.load_state_dict(torch.load(gnn_weights, map_location=self.device, weights_only=True))
        self.gnn.eval()

        # Target scaling parameters for unscaling predictions
        if self.prop_name == "bandgap":
            self.y_mean = 2.45
            self.y_std = 1.45
            self.unit = "eV"
        else: # tg
            self.y_mean = 545.23
            self.y_std = 49.41
            self.unit = "K"

        # 3. Load reference training library for novelty checking
        ref_csv = os.path.join(self.data_dir, "bandgap_chain.csv" if self.prop_name == "bandgap" else "tg_large.csv")
        ref_df = pd.read_csv(ref_csv)
        col = "smiles" if "smiles" in ref_df.columns else "smile"
        self.reference_set = set(ref_df[col].dropna().tolist())

        # 4. Latent Space Optimizer with TuRBO
        self.bo = LatentBayesianOptimizer(latent_dim=64, bounds_range=3.0, use_turbo=True)
        print(f"[Designer] Initialized Inverse Designer targeting '{self.prop_name.upper()}' ({self.unit})")

    def screen_polymer(self, smiles: str):
        """Screens a single candidate with PolymerGNN using Monte Carlo Dropout."""
        if not is_valid_polymer(smiles, require_wildcards=True):
            return None, None
        
        g = smiles_to_graph(smiles)
        if g is None or g.x.shape[0] == 0:
            return None, None

        g = g.to(self.device)
        batch_idx = torch.zeros(g.x.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            mu_norm, std_total_norm, _, _ = self.gnn.predict_monte_carlo(
                g.x, g.edge_index, batch_idx, num_samples=15
            )
            # Unscale
            pred_mu = float(mu_norm[0] * self.y_std + self.y_mean)
            pred_std = float(std_total_norm[0] * self.y_std)

        return pred_mu, pred_std

    def design(self, target_value: float, num_iterations: int = 6, batch_size: int = 15, top_k: int = 6, use_grammar: bool = False, acq_func: str = "qLogEI"):
        """
        Executes Active Latent Bayesian Optimization to discover polymers matching the target property.
        Supports both continuous VAE latent optimization and discrete reaction-templated grammar screening.
        """
        print(f"\n==================================================================")
        print(f"[*] Launching PolymerGenesis Inverse Design Loop")
        print(f"[*] Target {self.prop_name.upper()}: {target_value:.2f} {self.unit}")
        print(f"[*] Optimization: TuRBO BoTorch | Acquisition: {acq_func} | Reaction Grammar: {use_grammar}")
        print(f"[*] Iterations: {num_iterations} | Batch Size per Step: {batch_size}")
        print(f"==================================================================\n")

        # Initial random latent exploration
        z_history = torch.randn(25, 64, device=self.device)
        reward_history = []
        discovered_candidates = []

        # Optional reaction grammar candidate screening
        if use_grammar:
            print("[Grammar Space] Evaluating 100% synthetically accessible combinatorial library...")
            grammar_entries = enumerate_combinatorial_library()
            for entry in grammar_entries:
                smi = entry["repeat_unit_smiles"]
                p_mu, p_std = self.screen_polymer(smi)
                if p_mu is not None:
                    sa = compute_synthetic_accessibility(smi)
                    dist = abs(p_mu - target_value)
                    r = - (dist / (self.y_std * 0.5))**2 - 0.25 * (sa - 1.0)
                    discovered_candidates.append({
                        "smiles": smi,
                        "pred_val": p_mu,
                        "pred_std": p_std,
                        "sa_score": sa,
                        "abs_diff": dist,
                        "origin": "Reaction Grammar"
                    })

        print("[Phase 1] Evaluating Initial Exploration Seeds...")
        for z in z_history:
            smiles_batch = self.vae.decode_latent(z.unsqueeze(0), self.tokenizer, temperature=0.7)
            smiles = smiles_batch[0]
            pred_mu, pred_std = self.screen_polymer(smiles)

            if pred_mu is not None:
                sa = compute_synthetic_accessibility(smiles)
                # Proximity penalty + SAScore penalty
                dist = abs(pred_mu - target_value)
                reward = - (dist / (self.y_std * 0.5))**2 - 0.25 * (sa - 1.0)
                reward_history.append(reward)
                discovered_candidates.append({
                    "smiles": smiles,
                    "pred_val": pred_mu,
                    "pred_std": pred_std,
                    "sa_score": sa,
                    "abs_diff": dist,
                    "origin": "VAE Latent Space"
                })
            else:
                reward_history.append(-10.0) # Penalty for invalid chemistry

        reward_history = torch.tensor(reward_history, device=self.device, dtype=torch.float)

        # Active Bayesian Optimization iterations
        print(f"\n[Phase 2] Guided Latent Space Bayesian Optimization (TuRBO {acq_func})...")
        for it in range(1, num_iterations + 1):
            next_z = self.bo.optimize_step(z_history, reward_history, q=batch_size, beta=2.5, acq_type=acq_func)
            
            step_valid = 0
            step_rewards = []
            for z in next_z:
                smiles_batch = self.vae.decode_latent(z.unsqueeze(0), self.tokenizer, temperature=0.65)
                smiles = smiles_batch[0]
                pred_mu, pred_std = self.screen_polymer(smiles)

                if pred_mu is not None:
                    step_valid += 1
                    sa = compute_synthetic_accessibility(smiles)
                    dist = abs(pred_mu - target_value)
                    r = - (dist / (self.y_std * 0.5))**2 - 0.25 * (sa - 1.0)
                    step_rewards.append(r)
                    discovered_candidates.append({
                        "smiles": smiles,
                        "pred_val": pred_mu,
                        "pred_std": pred_std,
                        "sa_score": sa,
                        "abs_diff": dist
                    })
                else:
                    step_rewards.append(-10.0)

            step_rewards = torch.tensor(step_rewards, device=self.device, dtype=torch.float)
            z_history = torch.cat([z_history, next_z], dim=0)
            reward_history = torch.cat([reward_history, step_rewards], dim=0)

            print(f"  Iteration {it:02d}/{num_iterations:02d} | Valid Candidates Found: {step_valid}/{batch_size} | Total Pool: {len(discovered_candidates)}")

        # Deduplicate & rank
        unique_dict = {}
        for c in discovered_candidates:
            s = c["smiles"]
            if s not in unique_dict or c["abs_diff"] < unique_dict[s]["abs_diff"]:
                unique_dict[s] = c
        
        ranked = sorted(unique_dict.values(), key=lambda x: (x["abs_diff"], x["sa_score"]))
        top_candidates = ranked[:top_k]

        # Calculate Novelty & Diversity
        top_smiles = [c["smiles"] for c in top_candidates]
        novelty = compute_novelty(top_smiles, self.reference_set)

        print(f"\n==================================================================")
        print(f"[+] Top {len(top_candidates)} Discovered Polymers for Target {self.prop_name.upper()} = {target_value:.2f} {self.unit}")
        print(f"[+] Library Novelty: {novelty * 100:.1f}% novel structures")
        print(f"==================================================================")

        for i, c in enumerate(top_candidates, 1):
            is_new = " [NOVEL]" if c["smiles"] not in self.reference_set else " [KNOWN]"
            print(f"{i}. {c['smiles']}{is_new}")
            print(f"   Predicted {self.prop_name.upper()}: {c['pred_val']:.2f} +/- {c['pred_std']:.2f} {self.unit} (Target error: {c['abs_diff']:.2f})")
            print(f"   Synthesizability (SAScore): {c['sa_score']:.2f}/10\n")

        # Save JSON Report
        catalog_path = os.path.join(self.results_dir, f"discovery_{self.prop_name}_{int(target_value)}.json")
        with open(catalog_path, "w") as f:
            json.dump({
                "target_property": self.prop_name,
                "target_value": target_value,
                "unit": self.unit,
                "novelty_fraction": novelty,
                "top_candidates": top_candidates
            }, f, indent=2)

        # Render 2D Chemical Grid Image
        mols = [smiles_to_mol(c["smiles"]) for c in top_candidates]
        legends = [
            f"#{i}: {c['pred_val']:.2f}±{c['pred_std']:.2f}{self.unit}\nSA:{c['sa_score']:.1f}"
            for i, c in enumerate(top_candidates, 1)
        ]
        grid_img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(350, 240), legends=legends)
        grid_path = os.path.join(self.results_dir, f"discovery_{self.prop_name}_{int(target_value)}_grid.png")
        grid_img.save(grid_path)
        print(f"[Done] 2D Structure Catalog image saved to: {grid_path}")
        print(f"[Done] Discovery JSON saved to: {catalog_path}")

        return top_candidates, grid_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolymerGenesis Inverse Designer")
    parser.add_argument("--property", type=str, default="bandgap", choices=["bandgap", "tg"])
    parser.add_argument("--target", type=float, default=3.5, help="Target property value (eV for bandgap, K for tg)")
    parser.add_argument("--iterations", type=int, default=5, help="BoTorch optimization rounds")
    parser.add_argument("--batch_size", type=int, default=15, help="Batch size per BO step")
    parser.add_argument("--use_grammar", action="store_true", help="Enable reaction grammar library candidate injection")
    parser.add_argument("--acq_func", type=str, default="qLogEI", choices=["qLogEI", "qUCB"], help="BoTorch acquisition function")
    args = parser.parse_args()

    designer = PolymerInverseDesigner(property_name=args.property)
    designer.design(
        target_value=args.target,
        num_iterations=args.iterations,
        batch_size=args.batch_size,
        use_grammar=args.use_grammar,
        acq_func=args.acq_func
    )
