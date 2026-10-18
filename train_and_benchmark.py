"""
Master Benchmark Script for Milestone 1:
1. Featurizes polymers into tabular descriptors (Morgan FP + RDKit) and PyG molecular graphs.
2. Trains XGBoost Baseline.
3. Trains Uncertainty-Aware PolymerGNN with Monte Carlo Dropout and Heteroscedastic Aleatoric Loss.
4. Benchmarks both models on Test Set: R2, RMSE, MAE, MACE (Calibration), and Sparsification.
5. Saves model checkpoints and generates publication-quality visualization plots.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from collections import defaultdict
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

from src.features.featurizer import smiles_to_graph, get_morgan_fingerprint, get_rdkit_descriptors, smiles_to_mol
from src.models.baseline_xgb import PolymerXGBoostBaseline
from src.models.gnn_surrogate import PolymerGNN, gaussian_nll_loss
from src.utils.metrics import compute_regression_metrics, compute_calibration_curve, compute_sparsification_curve

def scaffold_split(smiles_list: list, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """
    Bemis-Murcko scaffold split to prevent train/test homologous series data leakage.
    Ensures that structural polymer families do not span both training and evaluation splits.
    """
    scaffold_to_indices = defaultdict(list)
    for idx, s in enumerate(smiles_list):
        mol = smiles_to_mol(s)
        scaffold = ""
        if mol is not None:
            try:
                scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaffold = ""
        scaffold_to_indices[scaffold].append(idx)

    # Sort scaffold sets by size
    scaffold_groups = list(scaffold_to_indices.values())
    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)

    train_cutoff = int(train_ratio * len(smiles_list))
    val_cutoff = int((train_ratio + val_ratio) * len(smiles_list))

    idx_train, idx_val, idx_test = [], [], []
    for group in scaffold_groups:
        if len(idx_train) + len(group) <= train_cutoff:
            idx_train.extend(group)
        elif len(idx_train) + len(idx_val) + len(group) <= val_cutoff:
            idx_val.extend(group)
        else:
            idx_test.extend(group)

    if len(idx_val) == 0:
        idx_val = idx_test[:max(1, len(idx_test) // 2)]
        idx_test = idx_test[max(1, len(idx_test) // 2):]

    return np.array(idx_train), np.array(idx_val), np.array(idx_test)

def load_dataset(prop_name: str, max_samples: int = 3500):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    if prop_name.lower() == "bandgap":
        csv_path = os.path.join(data_dir, "bandgap_chain.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        smiles_col = "smiles"
        target_col = "bandgap_chain"
        unit = "eV"
    elif prop_name.lower() == "tg":
        csv_path = os.path.join(data_dir, "tg_large.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        smiles_col = "smile"
        target_col = "Tg_1"
        unit = "K"
    else:
        raise ValueError(f"Unknown property: {prop_name}")

    # Drop NaNs
    df = df.dropna(subset=[smiles_col, target_col])
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    print(f"[Data] Loaded {len(df)} polymers for property: {prop_name} ({unit})")
    return df[smiles_col].tolist(), df[target_col].to_numpy(dtype=np.float32), unit

def main():
    parser = argparse.ArgumentParser(description="PolymerGenesis Milestone 1 Benchmark")
    parser.add_argument("--property", type=str, default="bandgap", choices=["bandgap", "tg"], help="Property to benchmark")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs for GNN training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for GNN")
    parser.add_argument("--samples", type=int, default=3000, help="Max samples to train on")
    parser.add_argument("--split_type", type=str, default="scaffold", choices=["scaffold", "random"], help="Train/val/test split protocol")
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data
    smiles_list, targets, unit = load_dataset(args.property, max_samples=args.samples)

    # 2. Featurize for GNN and XGBoost
    print("[Featurization] Generating PyG graphs and Tabular descriptors...")
    graph_data_list = []
    valid_smiles = []
    valid_targets = []
    xgb_features = []

    for s, y in zip(smiles_list, targets):
        g = smiles_to_graph(s, y_val=y)
        if g is not None and g.x.shape[0] > 0:
            graph_data_list.append(g)
            valid_smiles.append(s)
            valid_targets.append(y)
            fp = get_morgan_fingerprint(s)
            desc = get_rdkit_descriptors(s)
            xgb_features.append(np.concatenate([fp, desc]))

    valid_targets = np.array(valid_targets, dtype=np.float32)
    xgb_features = np.array(xgb_features, dtype=np.float32)
    print(f"[Featurization] Successfully processed {len(graph_data_list)} valid polymers.")

    # 3. Train/Val/Test Splits (80 / 10 / 10)
    if args.split_type == "scaffold":
        print("[Split] Performing Bemis-Murcko Scaffold Split to prevent homologous series leakage...")
        idx_train, idx_val, idx_test = scaffold_split(valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    else:
        indices = np.arange(len(graph_data_list))
        idx_train, idx_temp = train_test_split(indices, test_size=0.20, random_state=42)
        idx_val, idx_test = train_test_split(idx_temp, test_size=0.50, random_state=42)

    print(f"[Split ({args.split_type})] Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # ================= XGBOOST BASELINE =================
    print("\n--- Training XGBoost Baseline ---")
    xgb_baseline = PolymerXGBoostBaseline(n_estimators=250, max_depth=6)
    xgb_baseline.fit(xgb_features[idx_train], valid_targets[idx_train])

    xgb_test_pred, xgb_test_std = xgb_baseline.predict_with_uncertainty(xgb_features[idx_test])
    xgb_metrics = compute_regression_metrics(valid_targets[idx_test], xgb_test_pred)
    print(f"[XGBoost Test Metrics] R2: {xgb_metrics['R2']:.4f} | RMSE: {xgb_metrics['RMSE']:.4f} {unit} | MAE: {xgb_metrics['MAE']:.4f} {unit}")

    # Target normalization for GNN stability
    y_mean = float(valid_targets[idx_train].mean())
    y_std = float(valid_targets[idx_train].std())
    if y_std < 1e-6:
        y_std = 1.0

    print(f"[PolymerGNN Target Normalization] Mean: {y_mean:.2f}, Std: {y_std:.2f}")

    # ================= PYG GRAPH NEURAL NETWORK =================
    print("\n--- Training Uncertainty-Aware PolymerGNN (GINEConv with Bond Features) ---")
    train_graphs = [graph_data_list[i] for i in idx_train]
    val_graphs = [graph_data_list[i] for i in idx_val]
    test_graphs = [graph_data_list[i] for i in idx_test]

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = graph_data_list[0].x.shape[1]
    gnn_model = PolymerGNN(in_channels=in_channels, hidden_dim=128, num_layers=4, dropout=0.2, edge_dim=6).to(device)
    optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

    best_val_loss = float('inf')
    best_weights_path = os.path.join(results_dir, f"gnn_{args.property}_best.pt")

    for epoch in range(1, args.epochs + 1):
        gnn_model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            y_norm = (batch.y - y_mean) / y_std
            mu, log_var = gnn_model(batch.x, batch.edge_index, batch.batch, apply_dropout=True, edge_attr=batch.edge_attr)
            loss = gaussian_nll_loss(y_norm, mu, log_var)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_graphs)

        # Validation
        gnn_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                y_norm = (batch.y - y_mean) / y_std
                mu, log_var = gnn_model(batch.x, batch.edge_index, batch.batch, apply_dropout=False, edge_attr=batch.edge_attr)
                loss = gaussian_nll_loss(y_norm, mu, log_var)
                val_loss += loss.item() * batch.num_graphs
        val_loss /= len(val_graphs)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gnn_model.state_dict(), best_weights_path)

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss (NLL): {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Load best checkpoint for testing
    gnn_model.load_state_dict(torch.load(best_weights_path, weights_only=True))

    # ================= EVALUATION & UNCERTAINTY QUANTIFICATION =================
    print("\n--- Evaluating Test Set with Monte Carlo Dropout UQ ---")
    gnn_model.eval()
    test_mus = []
    test_stds = []
    test_y_true = []

    for batch in test_loader:
        batch = batch.to(device)
        mu_mean, std_total, std_epi, std_alea = gnn_model.predict_monte_carlo(
            batch.x, batch.edge_index, batch.batch, num_samples=30, edge_attr=batch.edge_attr
        )
        # Unscale back to original units
        mu_unscaled = mu_mean * y_std + y_mean
        std_unscaled = std_total * y_std

        test_mus.extend(mu_unscaled)
        test_stds.extend(std_unscaled)
        test_y_true.extend(batch.y.cpu().numpy())

    test_mus = np.array(test_mus)
    test_stds = np.array(test_stds)
    test_y_true = np.array(test_y_true)

    gnn_metrics = compute_regression_metrics(test_y_true, test_mus)
    print(f"[PolymerGNN Test Metrics] R2: {gnn_metrics['R2']:.4f} | RMSE: {gnn_metrics['RMSE']:.4f} {unit} | MAE: {gnn_metrics['MAE']:.4f} {unit}")

    # Calibration & Sparsification
    gnn_calib = compute_calibration_curve(test_y_true, test_mus, test_stds)
    gnn_sparse = compute_sparsification_curve(test_y_true, test_mus, test_stds)
    print(f"[UQ Calibration] Mean Absolute Calibration Error (MACE): {gnn_calib['MACE']:.4f}")

    # Save metrics JSON
    summary = {
        "property": args.property,
        "unit": unit,
        "test_size": len(test_y_true),
        "xgboost": xgb_metrics,
        "polymer_gnn": {
            **gnn_metrics,
            "MACE": gnn_calib['MACE']
        }
    }
    with open(os.path.join(results_dir, f"benchmark_{args.property}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ================= PUBLICATION PLOTS =================
    print("\n--- Generating Benchmark Plots ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Parity Plot with Error Bars
    ax = axes[0]
    ax.errorbar(test_y_true, test_mus, yerr=test_stds, fmt='o', alpha=0.5, ecolor='lightcoral', elinewidth=1, capsize=2, label='Predictions (±1σ)')
    min_v = min(test_y_true.min(), test_mus.min())
    max_v = max(test_y_true.max(), test_mus.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.5, label='Ideal Parity')
    ax.set_title(f"PolymerGNN Parity ({args.property.capitalize()})\n$R^2$ = {gnn_metrics['R2']:.3f}, RMSE = {gnn_metrics['RMSE']:.3f} {unit}", fontsize=11)
    ax.set_xlabel(f"Ground Truth ({unit})")
    ax.set_ylabel(f"Predicted Mean ({unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Calibration Curve
    ax = axes[1]
    ax.plot(gnn_calib['expected_confidence'], gnn_calib['observed_confidence'], 'o-', color='navy', label=f"PolymerGNN (MACE: {gnn_calib['MACE']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_title("UQ Calibration Curve\n(Expected vs Observed Confidence)", fontsize=11)
    ax.set_xlabel("Expected Confidence Interval Level")
    ax.set_ylabel("Empirical Coverage Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Sparsification Error Curve
    ax = axes[2]
    ax.plot(np.array(gnn_sparse['fractions_removed']) * 100, gnn_sparse['rmse_curve'], 'r-', lw=2, label="PolymerGNN")
    ax.set_title("UQ Sparsification Curve\n(Pruning High-Uncertainty Polymers)", fontsize=11)
    ax.set_xlabel("% Samples Discarded (Most Uncertain First)")
    ax.set_ylabel(f"RMSE on Remaining Polymers ({unit})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"benchmark_{args.property}_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Done] Plots successfully saved to: {plot_path}")
    print(f"[Done] Model checkpoint saved to: {best_weights_path}")

if __name__ == "__main__":
    main()
