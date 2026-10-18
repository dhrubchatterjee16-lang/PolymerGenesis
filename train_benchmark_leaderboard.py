"""
Definitive Benchmark Leaderboard Script for PolymerGenesis.
Rigorously evaluates 4 model paradigms on out-of-distribution Bemis-Murcko Scaffold Splits:
1. Tabular Baseline (Morgan Fingerprint + 2D Physicochemical Descriptors + XGBoost)
2. Standard Molecular GNN (GCNConv - ignores edge/bond attributes)
3. PolymerGenesis GINEConv (Bond-Aware Message Passing + Monte Carlo Dropout UQ)
4. PolymerGenesis Physics-Informed GNN (GINEConv + Fox-Flory Thermodynamic Conditioning)

Computes:
- Regression Accuracy: R², RMSE, MAE
- Uncertainty Calibration: MACE (Mean Absolute Calibration Error)
- Error Sparsification: AUSE (Area Under Sparsification Error Curve)
Saves publication-quality comparison figure and JSON leaderboard.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader

from src.features.featurizer import smiles_to_graph, get_morgan_fingerprint, get_rdkit_descriptors, smiles_to_mol
from src.models.baseline_xgb import PolymerXGBoostBaseline
from src.models.gnn_surrogate import PolymerGNN, PhysicsInformedPolymerGNN, gaussian_nll_loss
from src.utils.metrics import compute_regression_metrics, compute_calibration_curve, compute_sparsification_curve
from train_and_benchmark import load_dataset, scaffold_split

# Baseline Standard GCN (no bond attributes) for comparative ablation
class StandardGCN(nn.Module):
    def __init__(self, in_channels: int = 30, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2):
        super(StandardGCN, self).__init__()
        self.node_proj = nn.Linear(in_channels, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.head_mu = nn.Linear(64, 1)
        self.head_var = nn.Linear(64, 1)
        self.dropout_rate = dropout

    def forward(self, x, edge_index, batch, apply_dropout: bool = False):
        h = F.relu(self.node_proj(x))
        for conv, bn in zip(self.convs, self.batch_norms):
            h_c = F.relu(bn(conv(h, edge_index)))
            if apply_dropout:
                h_c = F.dropout(h_c, p=self.dropout_rate, training=True)
            h = h + h_c
        h_pool = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)
        out = F.relu(self.fc1(h_pool))
        if apply_dropout:
            out = F.dropout(out, p=self.dropout_rate, training=True)
        out = F.relu(self.fc2(out))
        return self.head_mu(out).squeeze(-1), self.head_var(out).squeeze(-1)

    def predict_mcd(self, x, edge_index, batch, num_samples: int = 25):
        self.train()
        mus = []
        with torch.no_grad():
            for _ in range(num_samples):
                mu_s, _ = self.forward(x, edge_index, batch, apply_dropout=True)
                mus.append(mu_s.cpu())
        mus = torch.stack(mus)
        return torch.mean(mus, dim=0).numpy(), torch.std(mus, dim=0).numpy()


def run_benchmark():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 75)
    print("POLYMERGENESIS DEFINITIVE BENCHMARK & MODEL LEADERBOARD")
    print("Protocol: Bemis-Murcko Scaffold Split (Zero Homologous Series Leakage)")
    print("=" * 75)

    # 1. Load Dataset (Bandgap Property Benchmark)
    smiles_list, targets, unit = load_dataset("bandgap", max_samples=2500)

    # 2. Featurize
    print("\n[Step 1] Featurizing Molecular Graphs and Physicochemical Vectors...")
    graphs = []
    valid_smiles = []
    valid_targets = []
    xgb_feats = []

    for s, y in zip(smiles_list, targets):
        g = smiles_to_graph(s, y_val=y)
        if g is not None and g.x.shape[0] > 0:
            graphs.append(g)
            valid_smiles.append(s)
            valid_targets.append(y)
            fp = get_morgan_fingerprint(s)
            desc = get_rdkit_descriptors(s)
            xgb_feats.append(np.concatenate([fp, desc]))

    valid_targets = np.array(valid_targets, dtype=np.float32)
    xgb_feats = np.array(xgb_feats, dtype=np.float32)

    # 3. Bemis-Murcko Scaffold Split (80 / 10 / 10)
    print("\n[Step 2] Applying Bemis-Murcko Scaffold Split...")
    idx_train, idx_val, idx_test = scaffold_split(valid_smiles, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

    train_graphs = [graphs[i] for i in idx_train]
    val_graphs = [graphs[i] for i in idx_val]
    test_graphs = [graphs[i] for i in idx_test]

    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

    y_mean = float(valid_targets[idx_train].mean())
    y_std = float(valid_targets[idx_train].std())
    y_test_true = valid_targets[idx_test]

    leaderboard = {}

    # ================= MODEL 1: XGBOOST BASELINE =================
    print("\n[Model 1/4] Training Tabular XGBoost Baseline...")
    xgb = PolymerXGBoostBaseline(n_estimators=200, max_depth=6)
    xgb.fit(xgb_feats[idx_train], valid_targets[idx_train])
    pred_xgb, std_xgb = xgb.predict_with_uncertainty(xgb_feats[idx_test])
    m_xgb = compute_regression_metrics(y_test_true, pred_xgb)
    cal_xgb = compute_calibration_curve(y_test_true, pred_xgb, std_xgb)
    spar_xgb = compute_sparsification_curve(y_test_true, pred_xgb, std_xgb)

    leaderboard["XGBoost (Morgan FP + 2D RDKit)"] = {
        "R2": float(m_xgb["R2"]),
        "RMSE": float(m_xgb["RMSE"]),
        "MAE": float(m_xgb["MAE"]),
        "MACE": float(cal_xgb["MACE"]),
        "AUSE": float(spar_xgb["AUSE"]),
        "preds": pred_xgb.tolist(),
        "stds": std_xgb.tolist()
    }
    print(f"  XGBoost -> R²: {m_xgb['R2']:.4f} | RMSE: {m_xgb['RMSE']:.3f} eV | MACE: {cal_xgb['MACE']:.4f}")

    # ================= MODEL 2: STANDARD GCN (NO BOND FEATURES) =================
    print("\n[Model 2/4] Training Standard GCN Baseline (Ignoring Bond Attributes)...")
    in_dim = graphs[0].x.shape[1]
    gcn = StandardGCN(in_channels=in_dim, hidden_dim=64, num_layers=3, dropout=0.2).to(device)
    opt_gcn = torch.optim.AdamW(gcn.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(12):
        gcn.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_gcn.zero_grad()
            y_norm = (batch.y - y_mean) / y_std
            mu, log_var = gcn(batch.x, batch.edge_index, batch.batch, apply_dropout=True)
            loss = gaussian_nll_loss(y_norm, mu, log_var)
            loss.backward()
            opt_gcn.step()

    gcn.eval()
    pred_gcn, std_gcn = [], []
    for batch in test_loader:
        batch = batch.to(device)
        mu_m, std_m = gcn.predict_mcd(batch.x, batch.edge_index, batch.batch, num_samples=15)
        pred_gcn.extend(mu_m * y_std + y_mean)
        std_gcn.extend(std_m * y_std)

    pred_gcn = np.array(pred_gcn)
    std_gcn = np.array(std_gcn)
    m_gcn = compute_regression_metrics(y_test_true, pred_gcn)
    cal_gcn = compute_calibration_curve(y_test_true, pred_gcn, std_gcn)
    spar_gcn = compute_sparsification_curve(y_test_true, pred_gcn, std_gcn)

    leaderboard["Standard GCN (Topology Only)"] = {
        "R2": float(m_gcn["R2"]),
        "RMSE": float(m_gcn["RMSE"]),
        "MAE": float(m_gcn["MAE"]),
        "MACE": float(cal_gcn["MACE"]),
        "AUSE": float(spar_gcn["AUSE"]),
        "preds": pred_gcn.tolist(),
        "stds": std_gcn.tolist()
    }
    print(f"  Standard GCN -> R²: {m_gcn['R2']:.4f} | RMSE: {m_gcn['RMSE']:.3f} eV | MACE: {cal_gcn['MACE']:.4f}")

    # ================= MODEL 3: POLYMERGENESIS GINECONV (BOND-AWARE) =================
    print("\n[Model 3/4] Training PolymerGenesis GINEConv (Bond-Aware Edge MPNN)...")
    gine = PolymerGNN(in_channels=in_dim, hidden_dim=64, num_layers=3, dropout=0.2, edge_dim=6).to(device)
    opt_gine = torch.optim.AdamW(gine.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(12):
        gine.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_gine.zero_grad()
            y_norm = (batch.y - y_mean) / y_std
            mu, log_var = gine(batch.x, batch.edge_index, batch.batch, apply_dropout=True, edge_attr=batch.edge_attr)
            loss = gaussian_nll_loss(y_norm, mu, log_var)
            loss.backward()
            opt_gine.step()

    gine.eval()
    pred_gine, std_gine = [], []
    for batch in test_loader:
        batch = batch.to(device)
        mu_m, std_tot, _, _ = gine.predict_monte_carlo(batch.x, batch.edge_index, batch.batch, num_samples=15, edge_attr=batch.edge_attr)
        pred_gine.extend(mu_m * y_std + y_mean)
        std_gine.extend(std_tot * y_std)

    pred_gine = np.array(pred_gine)
    std_gine = np.array(std_gine)
    m_gine = compute_regression_metrics(y_test_true, pred_gine)
    cal_gine = compute_calibration_curve(y_test_true, pred_gine, std_gine)
    spar_gine = compute_sparsification_curve(y_test_true, pred_gine, std_gine)

    leaderboard["PolymerGenesis GINEConv (Bond-Aware)"] = {
        "R2": float(m_gine["R2"]),
        "RMSE": float(m_gine["RMSE"]),
        "MAE": float(m_gine["MAE"]),
        "MACE": float(cal_gine["MACE"]),
        "AUSE": float(spar_gine["AUSE"]),
        "preds": pred_gine.tolist(),
        "stds": std_gine.tolist()
    }
    print(f"  PolymerGenesis GINEConv -> R²: {m_gine['R2']:.4f} | RMSE: {m_gine['RMSE']:.3f} eV | MACE: {cal_gine['MACE']:.4f}")

    # ================= MODEL 4: PHYSICS-INFORMED GNN (FOX-FLORY) =================
    print("\n[Model 4/4] Training PolymerGenesis Physics-Informed GNN (Fox-Flory)...")
    pi_gnn = PhysicsInformedPolymerGNN(in_channels=in_dim, hidden_dim=64, num_layers=3, dropout=0.2, edge_dim=6).to(device)
    opt_pi = torch.optim.AdamW(pi_gnn.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(12):
        pi_gnn.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt_pi.zero_grad()
            y_norm = (batch.y - y_mean) / y_std
            mu, log_var = pi_gnn(batch.x, batch.edge_index, batch.batch, apply_dropout=True, edge_attr=batch.edge_attr)
            loss = gaussian_nll_loss(y_norm, mu, log_var)
            loss.backward()
            opt_pi.step()

    pi_gnn.eval()
    pred_pi, std_pi = [], []
    for batch in test_loader:
        batch = batch.to(device)
        mu_m, std_tot, _, _ = pi_gnn.predict_monte_carlo(batch.x, batch.edge_index, batch.batch, num_samples=15, edge_attr=batch.edge_attr)
        pred_pi.extend(mu_m * y_std + y_mean)
        std_pi.extend(std_tot * y_std)

    pred_pi = np.array(pred_pi)
    std_pi = np.array(std_pi)
    m_pi = compute_regression_metrics(y_test_true, pred_pi)
    cal_pi = compute_calibration_curve(y_test_true, pred_pi, std_pi)
    spar_pi = compute_sparsification_curve(y_test_true, pred_pi, std_pi)

    leaderboard["PolymerGenesis PI-GNN (Thermodynamic)"] = {
        "R2": float(m_pi["R2"]),
        "RMSE": float(m_pi["RMSE"]),
        "MAE": float(m_pi["MAE"]),
        "MACE": float(cal_pi["MACE"]),
        "AUSE": float(spar_pi["AUSE"]),
        "preds": pred_pi.tolist(),
        "stds": std_pi.tolist()
    }
    print(f"  PolymerGenesis PI-GNN -> R²: {m_pi['R2']:.4f} | RMSE: {m_pi['RMSE']:.3f} eV | MACE: {cal_pi['MACE']:.4f}")

    # ================= GENERATE PUBLICATION FIGURE =================
    print("\n[Step 3] Rendering 4-Panel Benchmark Visual...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: R² & RMSE Leaderboard Comparison
    names = list(leaderboard.keys())
    short_names = ["XGBoost", "Std GCN", "GINEConv", "PI-GNN"]
    r2_scores = [leaderboard[k]["R2"] for k in names]
    rmse_scores = [leaderboard[k]["RMSE"] for k in names]

    x = np.arange(len(short_names))
    w = 0.35
    axes[0, 0].bar(x - w/2, r2_scores, w, label="R² (higher is better)", color="#3B82F6")
    axes[0, 0].bar(x + w/2, rmse_scores, w, label="RMSE (eV, lower is better)", color="#EF4444")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(short_names, fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Score", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("A. Benchmark Accuracy on Scaffold Split (Out-of-Distribution)", fontsize=12, fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].legend(frameon=True)

    # Panel B: Parity Plot with Error Bars for PolymerGenesis GINEConv
    sub_idx = np.random.choice(len(y_test_true), min(80, len(y_test_true)), replace=False)
    axes[0, 1].errorbar(y_test_true[sub_idx], pred_gine[sub_idx], yerr=std_gine[sub_idx], fmt='o', color="#10B981", ecolor="#93C5FD", elinewidth=1.2, capsize=2, alpha=0.85, label="PolymerGenesis Predictions")
    min_val = min(y_test_true.min(), pred_gine.min()) - 0.2
    max_val = max(y_test_true.max(), pred_gine.max()) + 0.2
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="Ideal Parity (y = x)")
    axes[0, 1].set_xlabel("DFT True Bandgap (eV)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Predicted Bandgap (eV)", fontsize=11, fontweight="bold")
    axes[0, 1].set_title(f"B. Parity Plot with Epistemic UQ (R² = {m_gine['R2']:.3f})", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(frameon=True)

    # Panel C: Uncertainty Calibration Curves
    axes[1, 0].plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    axes[1, 0].plot(cal_xgb["expected_confidence"], cal_xgb["observed_confidence"], label=f"XGBoost (MACE: {cal_xgb['MACE']:.3f})", color="#F59E0B")
    axes[1, 0].plot(cal_gcn["expected_confidence"], cal_gcn["observed_confidence"], label=f"Std GCN (MACE: {cal_gcn['MACE']:.3f})", color="#EF4444")
    axes[1, 0].plot(cal_gine["expected_confidence"], cal_gine["observed_confidence"], label=f"GINEConv (MACE: {cal_gine['MACE']:.3f})", color="#3B82F6")
    axes[1, 0].plot(cal_pi["expected_confidence"], cal_pi["observed_confidence"], label=f"PI-GNN (MACE: {cal_pi['MACE']:.3f})", color="#10B981", linewidth=2.5)
    axes[1, 0].set_xlabel("Expected Confidence Level (1 - α)", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Observed Empirical Coverage", fontsize=11, fontweight="bold")
    axes[1, 0].set_title("C. Uncertainty Calibration Reliability (MACE)", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(frameon=True)

    # Panel D: Sparsification Error Curves (AUSE)
    axes[1, 1].plot(spar_xgb["fractions_removed"], spar_xgb["rmse_curve"], label=f"XGBoost (AUSE: {spar_xgb['AUSE']:.3f})", color="#F59E0B")
    axes[1, 1].plot(spar_gcn["fractions_removed"], spar_gcn["rmse_curve"], label=f"Std GCN (AUSE: {spar_gcn['AUSE']:.3f})", color="#EF4444")
    axes[1, 1].plot(spar_gine["fractions_removed"], spar_gine["rmse_curve"], label=f"GINEConv (AUSE: {spar_gine['AUSE']:.3f})", color="#3B82F6")
    axes[1, 1].plot(spar_pi["fractions_removed"], spar_pi["rmse_curve"], label=f"PI-GNN (AUSE: {spar_pi['AUSE']:.3f})", color="#10B981", linewidth=2.5)
    axes[1, 1].set_xlabel("Removed Fraction (Pruned by Uncertainty)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("RMSE on Retained Polymers (eV)", fontsize=11, fontweight="bold")
    axes[1, 1].set_title("D. Sparsification Error Curve (Active Pruning)", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "benchmark_leaderboard.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Benchmark 4-panel visual saved to: {plot_path}")

    # 4. Save JSON Leaderboard Summary
    summary_leaderboard = {
        "benchmark_protocol": "Bemis-Murcko Scaffold Split (80/10/10)",
        "property": "bandgap",
        "unit": "eV",
        "num_test_samples": len(idx_test),
        "models": {
            k: {
                "R2": leaderboard[k]["R2"],
                "RMSE": leaderboard[k]["RMSE"],
                "MAE": leaderboard[k]["MAE"],
                "MACE": leaderboard[k]["MACE"],
                "AUSE": leaderboard[k]["AUSE"]
            } for k in leaderboard
        }
    }
    json_path = os.path.join(results_dir, "benchmark_leaderboard.json")
    with open(json_path, "w") as f:
        json.dump(summary_leaderboard, f, indent=2)
    print(f"  Saved Benchmark Leaderboard JSON to: {json_path}")
    print("=" * 75)
    return summary_leaderboard

if __name__ == "__main__":
    run_benchmark()
