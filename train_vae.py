"""
Training Pipeline for PolymerVAE.
Trains on polymer repeat units, tracks reconstruction & KL divergence,
and verifies generative chemical validity, novelty, and synthesizability.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.features.tokenizer import PolymerTokenizer
from src.models.polymer_vae import PolymerVAE, vae_loss_function
from src.utils.chemistry import is_valid_polymer, compute_novelty, compute_synthetic_accessibility

class PolymerSmilesDataset(Dataset):
    def __init__(self, smiles_list: list, tokenizer: PolymerTokenizer, max_len: int = 100):
        self.data = []
        for s in smiles_list:
            if isinstance(s, str) and len(s.strip()) > 0:
                self.data.append(tokenizer.encode(s.strip(), max_len=max_len))
        self.data = torch.tensor(self.data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        # x_in: sequence without last token; x_target: sequence without first token
        return seq[:-1], seq[1:]

def train_vae(epochs: int = 25, batch_size: int = 64, latent_dim: int = 64, lr: float = 1e-3):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "bandgap_chain.csv")
    df = pd.read_csv(csv_path).dropna(subset=["smiles"])
    raw_smiles = df["smiles"].tolist()
    print(f"[PolymerVAE] Loaded {len(raw_smiles)} training polymer repeat units.")

    tokenizer = PolymerTokenizer()
    dataset = PolymerSmilesDataset(raw_smiles, tokenizer, max_len=100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolymerVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        latent_dim=latent_dim
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"[PolymerVAE] Training on {device} (Vocab size: {tokenizer.vocab_size}, Latent dim: {latent_dim})...")
    best_loss = float('inf')
    best_weights_path = os.path.join(results_dir, "polymer_vae_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss_accum = 0.0
        recon_accum = 0.0
        kl_accum = 0.0

        # Dynamic KL annealing
        beta = min(0.05, 0.001 + (epoch / epochs) * 0.049)

        for x_in, x_target in dataloader:
            x_in = x_in.to(device)
            x_target = x_target.to(device)

            optimizer.zero_grad()
            logits, mu, log_var, z = model(x_in, x_in)
            loss, recon, kl = vae_loss_function(logits, x_target, mu, log_var, beta=beta, pad_idx=tokenizer.pad_idx)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss_accum += loss.item() * len(x_in)
            recon_accum += recon.item() * len(x_in)
            kl_accum += kl.item() * len(x_in)

        n = len(dataset)
        avg_loss = total_loss_accum / n
        avg_recon = recon_accum / n
        avg_kl = kl_accum / n

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, Beta: {beta:.4f})")

    # Save tokenizer vocabulary
    with open(os.path.join(results_dir, "tokenizer_vocab.json"), "w") as f:
        json.dump(tokenizer.vocab, f)

    print(f"\n[PolymerVAE] Model saved to: {best_weights_path}")

    # ================= GENERATIVE SAMPLING VALIDATION =================
    print("\n--- Validating Generative Capacity (50 Unconditioned Samples) ---")
    model.load_state_dict(torch.load(best_weights_path, weights_only=True))
    sampled_smiles = model.sample(num_samples=50, tokenizer=tokenizer, device=device, temperature=0.7)

    valid_polymers = [s for s in sampled_smiles if is_valid_polymer(s, require_wildcards=True)]
    validity_rate = len(valid_polymers) / len(sampled_smiles)
    reference_set = set(raw_smiles)
    novelty_rate = compute_novelty(valid_polymers, reference_set)
    sa_scores = [compute_synthetic_accessibility(s) for s in valid_polymers]
    avg_sa = float(np.mean(sa_scores)) if len(sa_scores) > 0 else 0.0

    print(f"Generated Samples: {len(sampled_smiles)}")
    print(f"Valid Polymers: {len(valid_polymers)} ({validity_rate * 100:.1f}%)")
    print(f"Novelty Rate: {novelty_rate * 100:.1f}%")
    print(f"Average Synthesizability (SAScore): {avg_sa:.2f}/10 (Lower = easier to synthesize)")

    print("\nSample Generated Polymers:")
    for i, s in enumerate(valid_polymers[:5], 1):
        print(f"  {i}. {s} | SAScore: {compute_synthetic_accessibility(s):.2f}")

    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    train_vae(epochs=args.epochs, batch_size=args.batch_size)
